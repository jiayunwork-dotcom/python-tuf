# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Regression tests for reusing one :class:`Updater` across refreshes.

These tests exercise, entirely in-process via
:class:`tests.repository_simulator.RepositorySimulator`:

* multiple consecutive ``refresh()`` calls on the same instance,
* keeping the last successful target view when a later refresh fails,
* retrying on the same instance after a failure,
* rollback lower bounds that survive both retry and instance rebuild,
* authenticated intermediate timestamp/snapshot anchors,
* forged high-version metadata that must not raise the lower bound, and
* invalidation/re-verification of delegated role caches after a successful
  refresh when a delegated target was removed.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from tests import utils
from tests.repository_simulator import RepositorySimulator
from tuf.api import exceptions
from tuf.api.metadata import (
    DelegatedRole,
    Metadata,
    Root,
    Snapshot,
    Targets,
    Timestamp,
)
from tuf.ngclient import Updater

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

METADATA_URL = "https://example.com/metadata/"
TARGETS_URL = "https://example.com/targets/"


class RefreshReuseTest(unittest.TestCase):
    """Tests for multiple refreshes on a single Updater instance."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_dir = os.path.join(self.temp_dir.name, "metadata")
        self.targets_dir = os.path.join(self.temp_dir.name, "targets")
        self.sim = RepositorySimulator()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def updater(self) -> Updater:
        """Create an Updater backed by the simulator and a temp dir."""
        return Updater(
            self.metadata_dir,
            METADATA_URL,
            self.targets_dir,
            TARGETS_URL,
            self.sim,
            bootstrap=self.sim.signed_roots[0],
        )

    def publish_new_targets(self, *paths: str) -> None:
        """Add targets to top-level targets and publish snapshot/timestamp."""
        for path in paths:
            self.sim.add_target("targets", f"content-{path}".encode(), path)
        self.sim.targets.version += 1
        self.sim.update_snapshot()

    def fail_metadata_role(
        self, role: str, status_code: int = 500
    ) -> AbstractContextManager[Any]:
        """Return a patcher that makes fetching ``role`` always fail."""
        real_fetch = self.sim.fetch_metadata

        def failing_fetch(
            fetched_role: str, version: int | None = None
        ) -> bytes:
            if fetched_role == role:
                raise exceptions.DownloadHTTPError(
                    f"simulated failure for {role}", status_code
                )
            return real_fetch(fetched_role, version)

        return patch.object(
            self.sim, "fetch_metadata", side_effect=failing_fetch
        )

    # ------------------------------------------------------------------
    # 1. Consecutive refreshes on the same instance
    # ------------------------------------------------------------------
    def test_consecutive_refreshes_see_new_metadata(self) -> None:
        """refresh() can be called repeatedly and observes new rounds."""
        updater = self.updater()
        updater.refresh()
        self.assertEqual(updater._trusted_set.timestamp.version, 1)
        self.assertIsNone(updater.get_targetinfo("a.txt"))

        self.publish_new_targets("a.txt")
        updater.refresh()
        self.assertEqual(updater._trusted_set.timestamp.version, 2)
        self.assertIsNotNone(updater.get_targetinfo("a.txt"))

        # A second, later round on the very same instance.
        self.publish_new_targets("b.txt")
        updater.refresh()
        self.assertEqual(updater._trusted_set.timestamp.version, 3)
        self.assertIsNotNone(updater.get_targetinfo("a.txt"))
        self.assertIsNotNone(updater.get_targetinfo("b.txt"))

    def test_unchanged_repository_can_refresh_repeatedly(self) -> None:
        """Re-refreshing an unchanged repo is not an error."""
        updater = self.updater()
        for _ in range(3):
            updater.refresh()
        self.assertEqual(updater._trusted_set.timestamp.version, 1)

    # ------------------------------------------------------------------
    # 2. Failed round keeps the last successful complete view
    # ------------------------------------------------------------------
    def test_failed_refresh_keeps_last_good_view(self) -> None:
        """get_targetinfo keeps using the last complete successful view."""
        self.publish_new_targets("old.txt")
        updater = self.updater()
        updater.refresh()
        old_view = updater._trusted_set
        old_timestamp_version = old_view.timestamp.version
        self.assertIsNotNone(updater.get_targetinfo("old.txt"))

        self.publish_new_targets("new.txt")
        new_timestamp_version = self.sim.timestamp.version
        self.assertGreater(new_timestamp_version, old_timestamp_version)

        with (
            self.fail_metadata_role(Snapshot.type),
            self.assertRaises(exceptions.DownloadHTTPError),
        ):
            updater.refresh()

        # Serving view is the exact same object: no half-built state leaked.
        self.assertIs(updater._trusted_set, old_view)
        self.assertEqual(
            updater._trusted_set.timestamp.version, old_timestamp_version
        )
        self.assertIsNotNone(updater.get_targetinfo("old.txt"))
        self.assertIsNone(updater.get_targetinfo("new.txt"))

    def test_failed_round_can_be_retried_on_same_instance(self) -> None:
        """After failures, a later refresh on the same instance succeeds."""
        self.publish_new_targets("old.txt")
        updater = self.updater()
        updater.refresh()

        self.publish_new_targets("new.txt")

        with self.fail_metadata_role(Snapshot.type):
            for _ in range(2):
                with self.assertRaises(exceptions.DownloadHTTPError):
                    updater.refresh()
            self.assertIsNotNone(updater.get_targetinfo("old.txt"))
            self.assertIsNone(updater.get_targetinfo("new.txt"))

        # Failure is transient: same instance now completes the new round.
        updater.refresh()
        self.assertIsNotNone(updater.get_targetinfo("new.txt"))
        self.assertIsNotNone(updater.get_targetinfo("old.txt"))

    def test_implicit_refresh_retries_after_failure(self) -> None:
        """A failed implicit refresh does not wedge get_targetinfo()."""
        updater = self.updater()

        with (
            self.fail_metadata_role(Timestamp.type),
            self.assertRaises(exceptions.DownloadHTTPError),
        ):
            updater.get_targetinfo("anything")

        # Restored repository: implicit refresh now succeeds.
        self.assertIsNone(updater.get_targetinfo("anything"))

    # ------------------------------------------------------------------
    # 3. Rollback lower bounds survive retry and instance rebuild
    # ------------------------------------------------------------------
    def test_persisted_timestamp_bound_constrains_rebuild(self) -> None:
        """A final timestamp accepted before a later failure is durable."""
        updater = self.updater()
        updater.refresh()

        # Repository publishes timestamp v2 / snapshot v2; make the snapshot
        # download fail *after* timestamp v2 was accepted and persisted.
        self.sim.update_snapshot()
        self.assertEqual(self.sim.timestamp.version, 2)

        with (
            self.fail_metadata_role(Snapshot.type),
            self.assertRaises(exceptions.DownloadHTTPError),
        ):
            updater.refresh()

        persisted_ts = Metadata.from_file(
            os.path.join(self.metadata_dir, "timestamp.json")
        )
        self.assertEqual(persisted_ts.signed.version, 2)

        # Rollback: repository now serves old timestamp v1.
        self.sim.timestamp.version = 1
        self.sim.timestamp.snapshot_meta.version = 1

        # Same instance must reject the rollback ...
        with self.assertRaises(exceptions.BadVersionNumberError):
            updater.refresh()

        # ... and so must a freshly rebuilt instance over the same dir.
        rebuilt = self.updater()
        with self.assertRaises(exceptions.BadVersionNumberError):
            rebuilt.refresh()
        self._assert_disk_version(Timestamp.type, 2)

    def test_intermediate_expired_timestamp_anchor_is_durable(self) -> None:
        """An authenticated but expired timestamp raises a durable bound."""
        updater = self.updater()
        updater.refresh()

        now = datetime.now(UTC)
        self.sim.timestamp.version = 2
        self.sim.timestamp.expires = now - timedelta(days=1)
        self.sim.timestamp.snapshot_meta.version = self.sim.snapshot.version

        with self.assertRaises(exceptions.ExpiredMetadataError):
            updater.refresh()

        anchor_path = os.path.join(
            self.metadata_dir, "rollback_anchors", "timestamp.json"
        )
        self.assertTrue(os.path.isfile(anchor_path))

        # Feed an old, non-expired timestamp v1: the v2 anchor must block it
        # both on the existing instance and after a rebuild.
        self.sim.timestamp.version = 1
        self.sim.timestamp.expires = self.sim.safe_expiry
        self.sim.timestamp.snapshot_meta.version = 1

        with self.assertRaises(exceptions.BadVersionNumberError):
            updater.refresh()
        with self.assertRaises(exceptions.BadVersionNumberError):
            self.updater().refresh()

        # A genuinely newer timestamp v3 allows the same instance to recover.
        self.sim.timestamp.version = 3
        self.sim.timestamp.expires = self.sim.safe_expiry
        self.sim.timestamp.snapshot_meta.version = self.sim.snapshot.version
        updater.refresh()
        self.assertEqual(updater._trusted_set.timestamp.version, 3)

    def test_intermediate_snapshot_anchor_protects_meta_versions(
        self,
    ) -> None:
        """Accepted intermediate snapshot meta versions cannot roll back."""
        updater = self.updater()
        updater.refresh()

        # Authenticated snapshot v2 with targets meta v7, while timestamp v2
        # claims snapshot version 3: snapshot v2 is added for rollback
        # protection but fails the final version match.
        self.sim.snapshot.meta["targets.json"].version = 7
        self.sim.snapshot.version = 2
        self.sim.timestamp.snapshot_meta.version = 3
        self.sim.timestamp.version = 2

        with self.assertRaises(exceptions.BadVersionNumberError):
            updater.refresh()

        anchor_path = os.path.join(
            self.metadata_dir, "rollback_anchors", "snapshot.json"
        )
        self.assertTrue(os.path.isfile(anchor_path))

        # Replay an old snapshot whose targets meta version is 1.
        self.sim.snapshot.version = 1
        self.sim.snapshot.meta["targets.json"].version = 1

        with self.assertRaises(exceptions.BadVersionNumberError):
            updater.refresh()
        with self.assertRaises(exceptions.BadVersionNumberError):
            self.updater().refresh()

    # ------------------------------------------------------------------
    # Forged high versions must not raise the lower bound
    # ------------------------------------------------------------------
    def test_forged_high_timestamp_does_not_raise_bound(self) -> None:
        """A high-version timestamp without valid signatures is not anchored."""
        updater = self.updater()
        updater.refresh()

        self.sim.timestamp.version = 999
        valid_bytes = self.sim.fetch_metadata(Timestamp.type)
        forged = Metadata[Timestamp].from_bytes(valid_bytes)
        forged.signatures.clear()
        forged_bytes = forged.to_bytes()

        real_fetch = self.sim.fetch_metadata

        def serve_forged_timestamp(
            role: str, version: int | None = None
        ) -> bytes:
            if role == Timestamp.type:
                return forged_bytes
            return real_fetch(role, version)

        with (
            patch.object(
                self.sim,
                "fetch_metadata",
                side_effect=serve_forged_timestamp,
            ),
            self.assertRaises(exceptions.UnsignedMetadataError),
        ):
            updater.refresh()

        anchor_path = os.path.join(
            self.metadata_dir, "rollback_anchors", "timestamp.json"
        )
        self.assertFalse(os.path.isfile(anchor_path))

        # The bound is still v1, so serving the coherent v1 round works.
        self.sim.timestamp.version = 1
        self.sim.timestamp.snapshot_meta.version = 1
        updater.refresh()
        self.assertEqual(updater._trusted_set.timestamp.version, 1)

    def test_forged_high_root_is_not_persisted(self) -> None:
        """A consecutive root failing verification is not written to cache."""
        self.sim.root.version += 1
        self.sim.publish_root()
        good_root_v2 = self.sim.signed_roots[1]

        forged_root = Metadata[Root].from_bytes(good_root_v2)
        forged_root.signatures.clear()
        forged_bytes = forged_root.to_bytes()

        real_fetch = self.sim.fetch_metadata

        def serve_forged_root(role: str, version: int | None = None) -> bytes:
            if role == Root.type and version == 2:
                return forged_bytes
            return real_fetch(role, version)

        updater = self.updater()
        with (
            patch.object(
                self.sim, "fetch_metadata", side_effect=serve_forged_root
            ),
            self.assertRaises(exceptions.UnsignedMetadataError),
        ):
            updater.refresh()

        self.assertFalse(
            os.path.isfile(
                os.path.join(self.metadata_dir, "root_history", "2.root.json")
            )
        )

        # Once the valid root v2 is served, the same instance advances.
        updater.refresh()
        self.assertEqual(updater._trusted_set.root.version, 2)

    # ------------------------------------------------------------------
    # 4. Successful refresh changes delegation caches
    # ------------------------------------------------------------------
    def test_removed_delegated_target_disappears_after_refresh(self) -> None:
        """Stale delegated role cache cannot keep a removed target visible."""
        role_metadata = Targets(1, None, self.sim.safe_expiry, {}, None)
        delegation = DelegatedRole("role1", [], 1, False, ["*"], None)
        self.sim.add_delegation("targets", delegation, role_metadata)
        self.sim.add_target("role1", b"delegated", "delegated.txt")
        self.sim.update_snapshot()

        updater = self.updater()
        self.assertIsNotNone(updater.get_targetinfo("delegated.txt"))
        self.assertTrue(updater._trusted_set.contains("role1", "targets"))
        self.assertTrue(
            os.path.isfile(os.path.join(self.metadata_dir, "role1.json"))
        )

        # Remove the target from role1 and publish a new role/snapshot round.
        self.sim.md_delegates["role1"].signed.targets.clear()
        self.sim.md_delegates["role1"].signed.version += 1
        self.sim.update_snapshot()

        updater.refresh()

        # The new serving view starts with an empty delegation cache ...
        self.assertFalse(updater._trusted_set.contains("role1", "targets"))
        self.sim.fetch_tracker.metadata.clear()

        # ... lookup reloads/re-verifies role1 under the new snapshot and the
        # removed target is gone.
        self.assertIsNone(updater.get_targetinfo("delegated.txt"))
        self.assertEqual(self.sim.fetch_tracker.metadata, [("role1", 2)])
        # The reloaded role is cached again within the new view.
        self.assertTrue(updater._trusted_set.contains("role1", "targets"))

    def test_removed_delegation_makes_target_disappear(self) -> None:
        """A delegation removed from the parent must not be traversed again."""
        role_metadata = Targets(1, None, self.sim.safe_expiry, {}, None)
        delegation = DelegatedRole("roleY", [], 1, False, ["*"], None)
        self.sim.add_delegation("targets", delegation, role_metadata)
        self.sim.add_target("roleY", b"delegated", "y.txt")
        self.sim.update_snapshot()

        updater = self.updater()
        self.assertIsNotNone(updater.get_targetinfo("y.txt"))

        # Drop the delegation from top-level targets and publish a new round.
        delegations = self.sim.md_targets.signed.delegations
        assert delegations is not None and delegations.roles is not None
        delegations.roles.clear()
        self.sim.targets.version += 1
        self.sim.update_snapshot()
        updater.refresh()

        self.assertIsNone(updater.get_targetinfo("y.txt"))

    def test_delegated_role_must_be_reverified_in_new_round(self) -> None:
        """A previously valid signature does not substitute new verification."""
        role_metadata = Targets(1, None, self.sim.safe_expiry, {}, None)
        delegation = DelegatedRole("role1", [], 1, False, ["*"], None)
        self.sim.add_delegation("targets", delegation, role_metadata)
        self.sim.add_target("role1", b"delegated", "delegated.txt")
        self.sim.update_snapshot()

        updater = self.updater()
        self.assertIsNotNone(updater.get_targetinfo("delegated.txt"))

        # Publish role1 v2 (still containing the target) ...
        self.sim.md_delegates["role1"].signed.version += 1
        self.sim.update_snapshot()
        updater.refresh()

        # ... but strip the role1 signing key for the new round. The stale
        # on-disk v1 fails the version check first, then the downloaded v2
        # fails signature verification: prior verification does not help.
        self.sim.signers["role1"].clear()
        with self.assertRaises(exceptions.UnsignedMetadataError):
            updater.get_targetinfo("delegated.txt")

    # ------------------------------------------------------------------
    # bootstrap trust anchor still behaves as before
    # ------------------------------------------------------------------
    def test_bootstrap_none_uses_cached_root_and_repeats(self) -> None:
        """bootstrap=None keeps trusting the cached root across refreshes."""
        first = self.updater()
        first.refresh()

        # A second instance opts into the cached root.json trust anchor.
        cached = Updater(
            self.metadata_dir,
            METADATA_URL,
            self.targets_dir,
            TARGETS_URL,
            self.sim,
            bootstrap=None,
        )
        cached.refresh()
        cached.refresh()
        self.assertEqual(cached._trusted_set.root.version, 1)

    def _assert_disk_version(self, role: str, expected: int) -> None:
        metadata = Metadata.from_file(
            os.path.join(self.metadata_dir, f"{role}.json")
        )
        self.assertEqual(metadata.signed.version, expected)


if __name__ == "__main__":
    utils.configure_test_logging(sys.argv)
    unittest.main()
