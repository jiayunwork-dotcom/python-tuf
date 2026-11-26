# Copyright 2021, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Unit test for Urllib3Fetcher."""

from __future__ import annotations

import io
import logging
import math
import os
import sys
import tempfile
import unittest
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import Mock, patch

import urllib3

from tests import utils
from tuf.api import exceptions
from tuf.ngclient import Urllib3Fetcher
from tuf.ngclient.fetcher import FetcherInterface

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class TestFetcher(unittest.TestCase):
    """Test Urllib3Fetcher class."""

    server_process_handler: ClassVar[utils.TestServerProcess]

    @classmethod
    def setUpClass(cls) -> None:
        """
        Create a temporary file and launch a simple server in the
        current working directory.
        """
        cls.server_process_handler = utils.TestServerProcess(log=logger)

        cls.file_contents = b"junk data"
        cls.file_length = len(cls.file_contents)
        with tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False
        ) as cls.target_file:
            cls.target_file.write(cls.file_contents)

        cls.url_prefix = (
            f"http://{utils.TEST_HOST_ADDRESS}:"
            f"{cls.server_process_handler.port!s}"
        )
        target_filename = os.path.basename(cls.target_file.name)
        cls.url = f"{cls.url_prefix}/{target_filename}"

    @classmethod
    def tearDownClass(cls) -> None:
        # Stop server process and perform clean up.
        cls.server_process_handler.clean()
        os.remove(cls.target_file.name)

    def setUp(self) -> None:
        # Instantiate a concrete instance of FetcherInterface
        self.fetcher = Urllib3Fetcher()

    # Simple fetch.
    def test_fetch(self) -> None:
        with tempfile.TemporaryFile() as temp_file:
            for chunk in self.fetcher.fetch(self.url):
                temp_file.write(chunk)

            temp_file.seek(0)
            self.assertEqual(self.file_contents, temp_file.read())

    # URL data downloaded in more than one chunk
    def test_fetch_in_chunks(self) -> None:
        # Set a smaller chunk size to ensure that the file will be downloaded
        # in more than one chunk
        self.fetcher.chunk_size = 4

        # expected_chunks_count: 3 (depends on length of self.file_length)
        expected_chunks_count = math.ceil(
            self.file_length / self.fetcher.chunk_size
        )
        self.assertEqual(expected_chunks_count, 3)

        chunks_count = 0
        with tempfile.TemporaryFile() as temp_file:
            for chunk in self.fetcher.fetch(self.url):
                temp_file.write(chunk)
                chunks_count += 1

            temp_file.seek(0)
            self.assertEqual(self.file_contents, temp_file.read())
            # Check that we calculate chunks as expected
            self.assertEqual(chunks_count, expected_chunks_count)

    # Incorrect URL parsing
    def test_url_parsing(self) -> None:
        with self.assertRaises(exceptions.DownloadError):
            self.fetcher.fetch("http://invalid/")

    # File not found error
    def test_http_error(self) -> None:
        with self.assertRaises(exceptions.DownloadHTTPError) as cm:
            self.url = f"{self.url_prefix}/non-existing-path"
            self.fetcher.fetch(self.url)
        self.assertEqual(cm.exception.status_code, 404)

    # Response read timeout error
    @patch.object(urllib3.PoolManager, "request")
    def test_response_read_timeout(self, mock_session_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status = 200
        attr = {
            "stream.side_effect": urllib3.exceptions.MaxRetryError(
                urllib3.connectionpool.ConnectionPool("localhost"),
                "",
                urllib3.exceptions.TimeoutError(),
            )
        }
        mock_response.configure_mock(**attr)
        mock_session_get.return_value = mock_response

        with self.assertRaises(exceptions.SlowRetrievalError):
            next(self.fetcher.fetch(self.url))
        mock_response.stream.assert_called_once()

    # urllib3 raises ReadTimeoutError directly when the gap timeout expires
    # mid-stream: it is a TimeoutError but not a MaxRetryError, so it is not
    # covered by the MaxRetryError case above.
    @patch.object(urllib3.PoolManager, "request")
    def test_response_read_timeout_error(self, mock_session_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status = 200
        attr = {
            "stream.side_effect": urllib3.exceptions.ReadTimeoutError(
                urllib3.connectionpool.ConnectionPool("localhost"),
                "",
                "Read timed out.",
            )
        }
        mock_response.configure_mock(**attr)
        mock_session_get.return_value = mock_response

        with self.assertRaises(exceptions.SlowRetrievalError):
            next(self.fetcher.fetch(self.url))
        mock_response.stream.assert_called_once()

    # The public download_* API documents DownloadError: a mid-stream timeout
    # must not escape as a raw urllib3 error.
    @patch.object(urllib3.PoolManager, "request")
    def test_download_bytes_read_timeout(self, mock_session_get: Mock) -> None:
        mock_response = Mock()
        mock_response.status = 200
        attr = {
            "stream.side_effect": urllib3.exceptions.ReadTimeoutError(
                urllib3.connectionpool.ConnectionPool("localhost"),
                "",
                "Read timed out.",
            )
        }
        mock_response.configure_mock(**attr)
        mock_session_get.return_value = mock_response

        with self.assertRaises(exceptions.SlowRetrievalError):
            self.fetcher.download_bytes(self.url, self.file_length)

    # Read/connect session timeout error
    @patch.object(
        urllib3.PoolManager,
        "request",
        side_effect=urllib3.exceptions.MaxRetryError(
            urllib3.connectionpool.ConnectionPool("localhost"),
            "",
            urllib3.exceptions.TimeoutError(),
        ),
    )
    def test_session_get_timeout(self, mock_session_get: Mock) -> None:
        with self.assertRaises(exceptions.SlowRetrievalError):
            self.fetcher.fetch(self.url)
        mock_session_get.assert_called_once()

    # A non-timeout connection failure (e.g. TLS error) must surface as the
    # original error, not be masked by an UnboundLocalError on "response".
    @patch.object(
        urllib3.PoolManager,
        "request",
        side_effect=urllib3.exceptions.MaxRetryError(
            urllib3.connectionpool.ConnectionPool("localhost"),
            "",
            urllib3.exceptions.SSLError("certificate verify failed"),
        ),
    )
    def test_session_get_ssl_error(self, mock_session_get: Mock) -> None:
        with self.assertRaises(exceptions.DownloadError) as cm:
            self.fetcher.fetch(self.url)
        mock_session_get.assert_called_once()
        self.assertIsInstance(
            cm.exception.__cause__, urllib3.exceptions.MaxRetryError
        )

    # Simple bytes download
    def test_download_bytes(self) -> None:
        data = self.fetcher.download_bytes(self.url, self.file_length)
        self.assertEqual(self.file_contents, data)

    # Download file smaller than required max_length
    def test_download_bytes_upper_length(self) -> None:
        data = self.fetcher.download_bytes(self.url, self.file_length + 4)
        self.assertEqual(self.file_contents, data)

    # Download a file bigger than expected
    def test_download_bytes_length_mismatch(self) -> None:
        with self.assertRaises(exceptions.DownloadLengthMismatchError):
            self.fetcher.download_bytes(self.url, self.file_length - 4)

    # Simple file download
    def test_download_file(self) -> None:
        with self.fetcher.download_file(
            self.url, self.file_length
        ) as temp_file:
            temp_file.seek(0, io.SEEK_END)
            self.assertEqual(self.file_length, temp_file.tell())

    # Download file smaller than required max_length
    def test_download_file_upper_length(self) -> None:
        with self.fetcher.download_file(
            self.url, self.file_length + 4
        ) as temp_file:
            temp_file.seek(0, io.SEEK_END)
            self.assertEqual(self.file_length, temp_file.tell())

    # Download a file bigger than expected
    def test_download_file_length_mismatch(self) -> None:
        with (
            self.assertRaises(exceptions.DownloadLengthMismatchError),
            self.fetcher.download_file(self.url, self.file_length - 4),
        ):
            pass  # we never get here as download_file() raises

    # ------------------------------------------------------------------
    # Response ownership: fetch() returns a stream the caller must close
    # ------------------------------------------------------------------

    @patch.object(urllib3.PoolManager, "request")
    def test_close_before_first_chunk_closes_response(
        self, mock_request: Mock
    ) -> None:
        # The response is acquired eagerly in _fetch(), but the body generator
        # has not been started. Closing must still close the response instead
        # of relying on garbage collection.
        mock_response = Mock()
        mock_response.status = 200
        mock_response.stream.return_value = iter([b"a", b"b"])
        mock_request.return_value = mock_response

        chunks = self.fetcher.fetch(self.url)
        chunks.close()

        mock_response.close.assert_called_once()
        mock_response.release_conn.assert_not_called()
        # An explicitly closed stream yields nothing more.
        self.assertEqual(list(chunks), [])
        mock_response.close.assert_called_once()

    @patch.object(urllib3.PoolManager, "request")
    def test_close_midway_closes_response(self, mock_request: Mock) -> None:
        mock_response = Mock()
        mock_response.status = 200
        mock_response.stream.return_value = iter([b"a", b"b", b"c"])
        mock_request.return_value = mock_response

        chunks = self.fetcher.fetch(self.url)
        self.assertEqual(next(chunks), b"a")
        chunks.close()

        mock_response.close.assert_called_once()
        mock_response.release_conn.assert_not_called()
        with self.assertRaises(StopIteration):
            next(chunks)

    @patch.object(urllib3.PoolManager, "request")
    def test_full_read_releases_connection(self, mock_request: Mock) -> None:
        mock_response = Mock()
        mock_response.status = 200
        mock_response.stream.return_value = iter([b"a", b"b"])
        mock_request.return_value = mock_response

        self.assertEqual(list(self.fetcher.fetch(self.url)), [b"a", b"b"])

        # A clean end returns the connection to the pool for reuse.
        mock_response.release_conn.assert_called_once()
        mock_response.close.assert_not_called()

    @patch.object(urllib3.PoolManager, "request")
    def test_midstream_error_closes_response(self, mock_request: Mock) -> None:
        def failing_stream(_amt: int) -> Iterator[bytes]:
            yield b"a"
            raise urllib3.exceptions.ProtocolError("connection broken")

        mock_response = Mock()
        mock_response.status = 200
        mock_response.stream.side_effect = failing_stream
        mock_request.return_value = mock_response

        chunks = self.fetcher.fetch(self.url)
        self.assertEqual(next(chunks), b"a")
        with self.assertRaises(exceptions.DownloadError):
            next(chunks)

        mock_response.close.assert_called_once()
        mock_response.release_conn.assert_not_called()

    @patch.object(urllib3.PoolManager, "request")
    def test_fetcher_reusable_after_early_close(
        self, mock_request: Mock
    ) -> None:
        first = Mock()
        first.status = 200
        first.stream.return_value = iter([b"a", b"b"])
        second = Mock()
        second.status = 200
        second.stream.return_value = iter([b"c"])
        mock_request.side_effect = [first, second]

        chunks = self.fetcher.fetch(self.url)
        chunks.close()

        # Same fetcher instance completes another download afterwards.
        self.assertEqual(list(self.fetcher.fetch(self.url)), [b"c"])
        second.release_conn.assert_called_once()

    @patch.object(urllib3.PoolManager, "request")
    def test_non_timeout_retry_error_propagates(
        self, mock_request: Mock
    ) -> None:
        # A non-timeout MaxRetryError must not be silently swallowed (which
        # would look like a complete, empty download).
        mock_response = Mock()
        mock_response.status = 200
        mock_response.stream.side_effect = urllib3.exceptions.MaxRetryError(
            urllib3.connectionpool.ConnectionPool("localhost"),
            "",
            urllib3.exceptions.SSLError("certificate verify failed"),
        )
        mock_request.return_value = mock_response

        chunks = self.fetcher.fetch(self.url)
        with self.assertRaises(exceptions.DownloadError):
            next(chunks)

        mock_response.close.assert_called_once()

    @patch.object(urllib3.PoolManager, "request")
    def test_stream_close_after_normal_exhaustion_is_noop(
        self, mock_request: Mock
    ) -> None:
        mock_response = Mock()
        mock_response.status = 200
        mock_response.stream.return_value = iter([b"a"])
        mock_request.return_value = mock_response

        chunks = self.fetcher.fetch(self.url)
        self.assertEqual(list(chunks), [b"a"])
        # Exhaustion already released the connection; closing afterwards must
        # not close or release it a second time.
        chunks.close()
        mock_response.release_conn.assert_called_once()
        mock_response.close.assert_not_called()


class _BareFetcher(FetcherInterface):
    """Fetcher returning iterators with and without a ``close`` method."""

    def __init__(self, iterator: Iterator[bytes]) -> None:
        self._iterator = iterator

    def _fetch(self, _url: str) -> Iterator[bytes]:
        return self._iterator


class TestDownloadStreamAdapter(unittest.TestCase):
    """Behaviour of the ``fetch()`` stream adapter itself."""

    def test_close_without_underlying_close_is_noop(self) -> None:
        # A plain list iterator has no close(): closing must simply stop it.
        fetcher = _BareFetcher(iter([b"a"]))
        chunks = fetcher.fetch("http://example.com/")
        chunks.close()
        self.assertEqual(list(chunks), [])

    def test_close_swallows_underlying_close_error(self) -> None:
        class BadIterator:
            def __iter__(self) -> BadIterator:
                return self

            def __next__(self) -> bytes:
                return b"a"

            def close(self) -> None:
                raise RuntimeError("cleanup failed")

        chunks = _BareFetcher(BadIterator()).fetch("http://example.com/")
        # A failing close must not raise to the caller.
        chunks.close()

    def test_wraps_streaming_error_as_download_error(self) -> None:
        class ExplodingIterator:
            def __iter__(self) -> ExplodingIterator:
                return self

            def __next__(self) -> bytes:
                raise RuntimeError("boom")

            def close(self) -> None:
                pass

        chunks = _BareFetcher(ExplodingIterator()).fetch("http://example.com/")
        with self.assertRaises(exceptions.DownloadError):
            next(chunks)


# Run unit test.
if __name__ == "__main__":
    utils.configure_test_logging(sys.argv)
    unittest.main()
