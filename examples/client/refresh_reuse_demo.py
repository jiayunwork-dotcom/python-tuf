#!/usr/bin/env python3
# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""In-process demo: reusing one Updater across many refresh() calls.

Everything runs in one process: the repository is a
``RepositorySimulator`` (an in-memory ``FetcherInterface``) and the client
uses temporary metadata directories, so no network or server is needed.

Run from a checkout with the runtime + test dependencies installed::

    python examples/client/refresh_reuse_demo.py

The script exits non-zero if any demonstrated property fails.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

# Allow importing the test-only RepositorySimulator from a checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.repository_simulator import RepositorySimulator  # noqa: E402
from tuf.api import exceptions  # noqa: E402
from tuf.api.metadata import (  # noqa: E402
    DelegatedRole,
    Metadata,
    Targets,
    Timestamp,
)
from tuf.ngclient import Updater  # noqa: E402

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
    from typing import Any

METADATA_URL = "https://demo.example/metadata/"
TARGETS_URL = "https://demo.example/targets/"

_failures: list[str] = []


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check(description: str, condition: bool) -> None:
    if condition:
        print(f"  [OK]   {description}")
    else:
        print(f"  [FAIL] {description}")
        _failures.append(description)


def make_updater(sim: RepositorySimulator, metadata_dir: str) -> Updater:
    return Updater(
        metadata_dir,
        METADATA_URL,
        metadata_dir,
        TARGETS_URL,
        sim,
        bootstrap=sim.signed_roots[0],
    )


def fail_role_fetch(
    sim: RepositorySimulator, role: str
) -> AbstractContextManager[Any]:
    """Context manager making metadata of ``role`` fail with HTTP 500."""
    real_fetch = sim.fetch_metadata

    def failing_fetch(role_name: str, version: int | None = None) -> bytes:
        if role_name == role:
            raise exceptions.DownloadHTTPError(f"simulated {role} outage", 500)
        return real_fetch(role_name, version)

    return patch.object(sim, "fetch_metadata", side_effect=failing_fetch)


def persisted_timestamp_version(metadata_dir: str) -> int:
    md = Metadata.from_file(os.path.join(metadata_dir, "timestamp.json"))
    return md.signed.version


def demo_consecutive_refresh(metadata_dir: str) -> None:
    section("1. Multiple consecutive refresh() calls")
    sim = RepositorySimulator()
    updater = make_updater(sim, metadata_dir)

    sim.add_target("targets", b"first content", "target-v1.txt")
    sim.targets.version += 1
    sim.update_snapshot()
    updater.refresh()
    ts1 = persisted_timestamp_version(metadata_dir)
    check(
        "first refresh finds target-v1.txt",
        updater.get_targetinfo("target-v1.txt") is not None,
    )

    sim.add_target("targets", b"second content", "target-v2.txt")
    sim.targets.version += 1
    sim.update_snapshot()
    updater.refresh()  # second refresh on the same instance
    ts2 = persisted_timestamp_version(metadata_dir)
    check(f"second refresh advances timestamp v{ts1} -> v{ts2}", ts2 > ts1)
    check(
        "new target-v2.txt is visible",
        updater.get_targetinfo("target-v2.txt") is not None,
    )
    check(
        "old target-v1.txt is still visible",
        updater.get_targetinfo("target-v1.txt") is not None,
    )


def demo_failed_round_keeps_view(metadata_dir: str) -> None:
    section("2. Failed refresh keeps last good view, then retries")
    sim = RepositorySimulator()
    sim.add_target("targets", b"old", "old.txt")
    sim.targets.version += 1
    sim.update_snapshot()
    updater = make_updater(sim, metadata_dir)
    updater.refresh()
    check(
        "old.txt available before the failed round",
        updater.get_targetinfo("old.txt") is not None,
    )

    sim.add_target("targets", b"new", "new.txt")
    sim.targets.version += 1
    sim.update_snapshot()

    with fail_role_fetch(sim, "snapshot"):
        try:
            updater.refresh()
            check("refresh with broken snapshot download raises", False)
        except exceptions.DownloadHTTPError:
            check("refresh with broken snapshot download raises", True)

    # Behaviorally the client still serves exactly the last good view: the
    # old target is found and the unpublished new one is not.
    check(
        "last-good old.txt still served after failure",
        updater.get_targetinfo("old.txt") is not None,
    )
    check(
        "half-downloaded new.txt is not visible",
        updater.get_targetinfo("new.txt") is None,
    )

    # Same instance, repository healthy again: retry succeeds.
    updater.refresh()
    check(
        "new.txt visible after successful retry",
        updater.get_targetinfo("new.txt") is not None,
    )


def demo_rollback_memory_and_rebuild(metadata_dir: str) -> None:
    section("3. Accepted versions bind retries and rebuilt instances")
    sim = RepositorySimulator()
    updater = make_updater(sim, metadata_dir)
    updater.refresh()

    # Publish timestamp v2 / snapshot v2, then fail the snapshot download:
    # timestamp v2 was verified and persisted before the failure.
    sim.update_snapshot()
    with (
        fail_role_fetch(sim, "snapshot"),
        contextlib.suppress(exceptions.DownloadHTTPError),
    ):
        updater.refresh()

    check(
        "accepted timestamp v2 persisted despite later failure",
        persisted_timestamp_version(metadata_dir) == 2,
    )

    # Attacker now serves an old timestamp v1 from the same simulator.
    sim.timestamp.version = 1
    sim.timestamp.snapshot_meta.version = 1

    try:
        updater.refresh()
        check("same-instance rollback to timestamp v1 is rejected", False)
    except exceptions.BadVersionNumberError:
        check("same-instance rollback to timestamp v1 is rejected", True)

    rebuilt = make_updater(sim, metadata_dir)
    try:
        rebuilt.refresh()
        check("rebuilt instance also rejects timestamp v1 rollback", False)
    except exceptions.BadVersionNumberError:
        check("rebuilt instance also rejects timestamp v1 rollback", True)


def demo_forged_high_version() -> None:
    section("4. Forged high versions do not raise the lower bound")
    sim = RepositorySimulator()
    metadata_dir = tempfile.mkdtemp(prefix="tuf-demo-forged-")
    updater = make_updater(sim, metadata_dir)
    updater.refresh()

    sim.timestamp.version = 999
    forged = Metadata[Timestamp].from_bytes(sim.fetch_metadata("timestamp"))
    forged.signatures.clear()
    forged_bytes = forged.to_bytes()
    real_fetch = sim.fetch_metadata
    with patch.object(
        sim,
        "fetch_metadata",
        side_effect=lambda role, version=None: (
            forged_bytes if role == "timestamp" else real_fetch(role, version)
        ),
    ):
        try:
            updater.refresh()
            check("forged timestamp v999 is rejected", False)
        except exceptions.UnsignedMetadataError:
            check("forged timestamp v999 is rejected", True)

    check(
        "forged v999 did not write a rollback anchor",
        not os.path.isfile(
            os.path.join(metadata_dir, "rollback_anchors", "timestamp.json")
        ),
    )

    # An authenticated but expired timestamp is different: it was really
    # signed by the trusted key, so its version bound must be remembered.
    expired_dir = tempfile.mkdtemp(prefix="tuf-demo-expired-")
    expired_sim = RepositorySimulator()
    expired_updater = make_updater(expired_sim, expired_dir)
    expired_updater.refresh()
    expired_sim.timestamp.version = 2
    expired_sim.timestamp.expires = datetime.now(UTC) - timedelta(days=1)
    with contextlib.suppress(exceptions.ExpiredMetadataError):
        expired_updater.refresh()
    check(
        "expired-but-signed timestamp anchor is written",
        os.path.isfile(
            os.path.join(expired_dir, "rollback_anchors", "timestamp.json")
        ),
    )


def demo_removed_delegated_target(metadata_dir: str) -> None:
    section("5. Successful refresh invalidates delegated role cache")
    sim = RepositorySimulator()
    role_md = Targets(1, None, sim.safe_expiry, {}, None)
    delegation = DelegatedRole("role1", [], 1, False, ["*"], None)
    sim.add_delegation("targets", delegation, role_md)
    sim.add_target("role1", b"delegated content", "delegated.txt")
    sim.update_snapshot()

    updater = make_updater(sim, metadata_dir)
    check(
        "delegated.txt found via role1",
        updater.get_targetinfo("delegated.txt") is not None,
    )

    # Remove the target, bump role1 + snapshot, refresh successfully.
    sim.md_delegates["role1"].signed.targets.clear()
    sim.md_delegates["role1"].signed.version += 1
    sim.update_snapshot()
    updater.refresh()

    # The new round must reload and re-verify role1 rather than reuse the
    # object cached under the previous snapshot.
    sim.fetch_tracker.metadata.clear()
    targetinfo = updater.get_targetinfo("delegated.txt")
    reloaded_role1 = any(
        role == "role1" for role, _ in sim.fetch_tracker.metadata
    )
    check("role1 is reloaded/re-verified in the new round", reloaded_role1)
    check(
        "removed delegated.txt does not reappear from stale cache",
        targetinfo is None,
    )


def main() -> int:
    base_dir = tempfile.mkdtemp(prefix="tuf-refresh-reuse-demo-")
    print(f"Using temporary metadata directories under {base_dir}")

    # Each act gets its own metadata directory to keep its trust state clear.
    demo_consecutive_refresh(tempfile.mkdtemp(dir=base_dir))
    demo_failed_round_keeps_view(tempfile.mkdtemp(dir=base_dir))
    demo_rollback_memory_and_rebuild(tempfile.mkdtemp(dir=base_dir))
    demo_forged_high_version()
    demo_removed_delegated_target(tempfile.mkdtemp(dir=base_dir))

    if _failures:
        print(f"\nDemo FAILED: {len(_failures)} check(s) failed")
        return 1
    print("\nDemo complete: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
