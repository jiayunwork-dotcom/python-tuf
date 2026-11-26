# Copyright 2026, the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Regression tests for failure boundaries of the target download path.

These tests run entirely in-process (no network, no external services) using
:class:`tests.repository_simulator.RepositorySimulator` for trusted targets and
small fault-injecting iterators for the download stream.

The covered failure modes are:

* caller closes ``fetch()`` before requesting the first chunk,
* caller closes ``fetch()`` midway through the stream,
* the stream raises while a chunk is being read,
* the server sends more data than the metadata allows ("oversized"),
* the downloaded bytes fail length/hash verification, and
* writing (or final publishing) the destination fails halfway.

For every case the tests assert that

* an existing destination keeps its old content,
* a missing destination is not created (not even partially),
* the temporary file used for publishing is removed,
* the download stream was explicitly closed (not left to garbage collection),
* and the same ``Updater``/fetcher can subsequently complete a download.

Run from the repository root with::

    python3 -m unittest tests.test_updater_download_failure -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from typing import IO, TYPE_CHECKING
from unittest.mock import patch
from urllib import parse

from tests import utils
from tests.repository_simulator import RepositorySimulator
from tuf.api import exceptions
from tuf.ngclient import Updater

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tuf.api.metadata import TargetFile

METADATA_URL = "https://example.com/metadata/"
TARGETS_URL = "https://example.com/targets/"

TARGET_PATH = "targetpath"
TARGET_CONTENT = b"the complete and correct target content"


def failing_copyfileobj(
    source: IO[bytes], destination: IO[bytes], *_args: object
) -> None:
    """Write a partial copy and then fail, imitating a full disk."""
    destination.write(source.read(4))
    raise OSError("simulated disk full")


class FaultyStream:
    """Byte iterator imitating a response body with injectable faults.

    Unlike a generator, ``close()`` is recorded even when iteration never
    started, which is exactly the property the production adapters must
    guarantee.
    """

    def __init__(
        self,
        data: bytes,
        chunk_size: int,
        *,
        fail_after: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._chunks = [
            data[offset : offset + chunk_size]
            for offset in range(0, len(data), chunk_size)
        ]
        self._index = 0
        self._fail_after = fail_after
        self._error = error or exceptions.DownloadError("simulated read error")
        self.closed_count = 0
        self.exhausted = False
        self.started = False
        self.delivered_bytes = 0

    def __iter__(self) -> FaultyStream:
        return self

    def __next__(self) -> bytes:
        self.started = True
        if self.closed_count:
            raise StopIteration
        if self._fail_after is not None and self._index >= self._fail_after:
            raise self._error
        if self._index >= len(self._chunks):
            self.exhausted = True
            raise StopIteration
        chunk = self._chunks[self._index]
        self._index += 1
        self.delivered_bytes += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed_count += 1

    @property
    def closed(self) -> bool:
        return self.closed_count > 0


class FaultySimulator(RepositorySimulator):
    """RepositorySimulator whose target responses can be instrumented.

    Metadata downloads are served unchanged, so the ``Updater`` can refresh
    and look up trusted target information normally.
    """

    def __init__(self) -> None:
        super().__init__()
        self.chunk_size = 8
        self.fail_after: int | None = None
        self.read_error: BaseException | None = None
        # Every target stream handed to a caller, in order.
        self.streams: list[FaultyStream] = []

    def _fetch(self, url: str) -> Iterator[bytes]:
        path = parse.urlparse(url).path
        if not path.startswith("/targets/"):
            return super()._fetch(url)

        # Reuse the parent implementation for path/hash-prefix validation and
        # target lookup, but consume it ourselves so we can re-chunk the bytes.
        data = b"".join(super()._fetch(url))
        stream = FaultyStream(
            data,
            self.chunk_size,
            fail_after=self.fail_after,
            error=self.read_error,
        )
        self.streams.append(stream)
        return stream

    def reset_faults(self) -> None:
        self.fail_after = None
        self.read_error = None


class TargetDownloadFailureTest(unittest.TestCase):
    """Failure boundaries around ``Updater.download_target()``."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metadata_dir = os.path.join(self.temp_dir.name, "metadata")
        self.targets_dir = os.path.join(self.temp_dir.name, "targets")
        os.mkdir(self.metadata_dir)
        os.mkdir(self.targets_dir)

        self.sim = FaultySimulator()
        self.sim.add_target("targets", TARGET_CONTENT, TARGET_PATH)
        self.sim.targets.version += 1
        self.sim.update_snapshot()

        self.updater = self._new_updater()
        info = self.updater.get_targetinfo(TARGET_PATH)
        assert info is not None
        self.info: TargetFile = info
        self.destination = os.path.join(self.targets_dir, TARGET_PATH)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _new_updater(self) -> Updater:
        return Updater(
            self.metadata_dir,
            METADATA_URL,
            self.targets_dir,
            TARGETS_URL,
            self.sim,
            bootstrap=self.sim.signed_roots[0],
        )

    # helpers ---------------------------------------------------------

    def _write_old_destination(self, content: bytes = b"old content") -> None:
        with open(self.destination, "wb") as f:
            f.write(content)

    def _assert_old_destination_intact(
        self, content: bytes = b"old content"
    ) -> None:
        with open(self.destination, "rb") as f:
            self.assertEqual(f.read(), content)

    def _assert_no_publish_leftovers(self) -> None:
        """No temporary publish file may remain in the destination dir."""
        leftovers = [
            name for name in os.listdir(self.targets_dir) if name != TARGET_PATH
        ]
        self.assertEqual(leftovers, [])

    def _last_stream(self) -> FaultyStream:
        self.assertTrue(self.sim.streams)
        return self.sim.streams[-1]

    def _target_url(self) -> str:
        """Build the URL the Updater uses (with hash prefix if applicable)."""
        target_filepath = self.info.path
        if (
            self.updater._trusted_set.root.consistent_snapshot
            and self.updater.config.prefix_targets_with_hash
        ):
            hashes = list(self.info.hashes.values())
            dirname, sep, basename = target_filepath.rpartition("/")
            target_filepath = f"{dirname}{sep}{hashes[0]}.{basename}"
        return f"{TARGETS_URL}{target_filepath}"

    def _assert_stream_released(self) -> None:
        """Exactly one explicit close, and never a silent drop."""
        stream = self._last_stream()
        self.assertEqual(stream.closed_count, 1)

    def _assert_recovers(self) -> None:
        """The same Updater can still download the target successfully."""
        self.sim.reset_faults()
        path = self.updater.download_target(self.info)
        self.assertEqual(path, self.destination)
        with open(self.destination, "rb") as f:
            self.assertEqual(f.read(), TARGET_CONTENT)
        # The successful run must have released its stream as well.
        self._assert_stream_released()
        self.assertIsNotNone(self.updater.find_cached_target(self.info))

    # 1. close() before the first chunk / midway, when using fetch() directly
    def test_fetch_close_before_first_chunk_releases_stream(self) -> None:
        stream = self.sim.fetch(self._target_url())
        self.assertFalse(self._last_stream().started)

        stream.close()

        self.assertTrue(self._last_stream().closed)
        self.assertEqual(self._last_stream().closed_count, 1)
        self.assertFalse(self._last_stream().exhausted)
        self._assert_recovers()

    def test_fetch_close_midway_releases_stream(self) -> None:
        stream = self.sim.fetch(self._target_url())
        next(stream)
        next(stream)
        stream.close()

        self.assertTrue(self._last_stream().closed)
        self.assertFalse(self._last_stream().exhausted)
        # No further chunks are served after an explicit close.
        with self.assertRaises(StopIteration):
            next(stream)
        self.assertEqual(self._last_stream().closed_count, 1)
        self._assert_recovers()

    # 2. failure modes expressed through download_target() ------------

    def test_stream_read_error_preserves_old_target(self) -> None:
        self._write_old_destination()
        self.sim.fail_after = 2
        self.sim.read_error = exceptions.DownloadError("connection reset")

        with self.assertRaises(exceptions.DownloadError):
            self.updater.download_target(self.info)

        self._assert_old_destination_intact()
        self._assert_no_publish_leftovers()
        self._assert_stream_released()
        self._assert_recovers()

    def test_stream_non_download_error_is_wrapped(self) -> None:
        self._write_old_destination()
        self.sim.fail_after = 2
        self.sim.read_error = RuntimeError("unexpected transport failure")

        with self.assertRaises(exceptions.DownloadError):
            self.updater.download_target(self.info)

        self._assert_old_destination_intact()
        self._assert_stream_released()
        self._assert_recovers()

    def test_oversized_download_rejected_without_draining(self) -> None:
        self._write_old_destination()
        # Serve more bytes than the metadata declares. The URL hash prefix
        # still matches, so only the length check can reject this.
        repo_target = self.sim.target_files[TARGET_PATH]
        original_data = repo_target.data
        self.sim.target_files[TARGET_PATH].data = TARGET_CONTENT + b"x" * 64

        with self.assertRaises(exceptions.DownloadLengthMismatchError):
            self.updater.download_target(self.info)

        self._assert_old_destination_intact()
        self._assert_no_publish_leftovers()
        stream = self._last_stream()
        self.assertEqual(stream.closed_count, 1)
        # The rejected stream must not have been drained to the end: only the
        # chunk that crossed the limit may have been read beyond it.
        self.assertFalse(stream.exhausted)
        self.assertLess(stream.delivered_bytes, len(TARGET_CONTENT) + 64)
        self.assertLessEqual(
            stream.delivered_bytes, len(TARGET_CONTENT) + self.sim.chunk_size
        )

        self.sim.target_files[TARGET_PATH].data = original_data
        self._assert_recovers()

    def test_hash_mismatch_preserves_old_target(self) -> None:
        self._write_old_destination()
        # Same length, different bytes: length passes, hash must fail.
        repo_target = self.sim.target_files[TARGET_PATH]
        original_data = repo_target.data
        corrupted = bytes(b ^ 0xFF for b in TARGET_CONTENT)
        self.sim.target_files[TARGET_PATH].data = corrupted

        with self.assertRaises(exceptions.LengthOrHashMismatchError):
            self.updater.download_target(self.info)

        self._assert_old_destination_intact()
        self._assert_no_publish_leftovers()
        self._assert_stream_released()
        # The stream was fully read (and released) before verification ran.
        self.assertTrue(self._last_stream().exhausted)

        self.sim.target_files[TARGET_PATH].data = original_data
        self._assert_recovers()

    def test_write_failure_midway_preserves_old_target(self) -> None:
        self._write_old_destination()

        with (
            patch(
                "tuf.ngclient.updater.shutil.copyfileobj",
                side_effect=failing_copyfileobj,
            ),
            self.assertRaises(OSError),
        ):
            self.updater.download_target(self.info)

        self._assert_old_destination_intact()
        self._assert_no_publish_leftovers()
        self._assert_stream_released()
        self._assert_recovers()

    def test_write_failure_without_existing_target_leaves_nothing(self) -> None:
        with (
            patch(
                "tuf.ngclient.updater.shutil.copyfileobj",
                side_effect=failing_copyfileobj,
            ),
            self.assertRaises(OSError),
        ):
            self.updater.download_target(self.info)

        self.assertFalse(os.path.exists(self.destination))
        self._assert_no_publish_leftovers()
        self._assert_recovers()

    def test_publish_replace_failure_preserves_old_target(self) -> None:
        self._write_old_destination()

        with (
            patch(
                "tuf.ngclient.updater.os.replace",
                side_effect=OSError("simulated rename failure"),
            ),
            self.assertRaises(OSError),
        ):
            self.updater.download_target(self.info)

        self._assert_old_destination_intact()
        self._assert_no_publish_leftovers()
        self._assert_stream_released()
        self._assert_recovers()

    def test_verification_failure_without_existing_target_leaves_nothing(
        self,
    ) -> None:
        repo_target = self.sim.target_files[TARGET_PATH]
        original_data = repo_target.data
        self.sim.target_files[TARGET_PATH].data = b"wrong"

        with self.assertRaises(exceptions.LengthOrHashMismatchError):
            self.updater.download_target(self.info)

        self.assertFalse(os.path.exists(self.destination))
        self._assert_no_publish_leftovers()
        self._assert_stream_released()

        self.sim.target_files[TARGET_PATH].data = original_data
        self._assert_recovers()


if __name__ == "__main__":
    utils.configure_test_logging(sys.argv)
    unittest.main()
