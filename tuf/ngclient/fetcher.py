# Copyright 2021, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Provides an interface for network IO abstraction."""

from __future__ import annotations

# Imports
import abc
import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO

from tuf.api import exceptions

logger = logging.getLogger(__name__)


class _DownloadStream(Iterator[bytes]):
    """Iterator adapter returned by :meth:`FetcherInterface.fetch`.

    It makes two guarantees that plain iterators do not provide:

    * ``close()`` is idempotent and works even before the first chunk has been
      requested. This matters for fetchers such as ``Urllib3Fetcher`` that
      acquire their network resource eagerly in ``_fetch()``: a generator that
      has never been started does not run its ``finally`` block on ``close()``.
    * Any exception raised while streaming is converted to a ``DownloadError``,
      matching what ``fetch()`` already promises for errors raised while
      calling ``_fetch()``.

    The wrapper never closes the underlying iterator on normal exhaustion:
    that would turn a clean end into an early close for fetchers that use the
    distinction to decide whether a connection may be reused.
    """

    def __init__(self, iterator: Iterator[bytes], url: str) -> None:
        self._iterator = iter(iterator)
        self._url = url
        self._closed = False

    def __iter__(self) -> _DownloadStream:
        return self

    def __next__(self) -> bytes:
        if self._closed:
            raise StopIteration
        try:
            return next(self._iterator)
        except StopIteration:
            raise
        except exceptions.DownloadError:
            self.close()
            raise
        except Exception as e:
            self.close()
            raise exceptions.DownloadError(
                f"Failed to download {self._url}"
            ) from e

    def close(self) -> None:
        """Release the underlying iterator, even if it was never started."""
        if self._closed:
            return
        self._closed = True
        close = getattr(self._iterator, "close", None)
        if close is not None:
            try:
                close()
            except Exception as e:  # noqa: BLE001
                # Never let cleanup mask the streaming error that caused it.
                logger.debug("Error while closing download stream: %s", e)


# Classes
class FetcherInterface(metaclass=abc.ABCMeta):
    """Defines an interface for abstract network download.

    By providing a concrete implementation of the abstract interface,
    users of the framework can plug-in their preferred/customized
    network stack.

    Implementations of FetcherInterface only need to implement ``_fetch()``.
    The public API of the class is already implemented.
    """

    @abc.abstractmethod
    def _fetch(self, url: str) -> Iterator[bytes]:
        """Fetch the contents of HTTP/HTTPS ``url`` from a remote server.

        Implementations must raise ``DownloadHTTPError`` if they receive
        an HTTP error code.

        Implementations may raise any errors but the ones that are not
        ``DownloadErrors`` will be wrapped in a ``DownloadError`` by
        ``fetch()``.

        Args:
            url: URL string that represents a file location.

        Raises:
            DownloadHTTPError: HTTP error code was received.

        Returns:
            Bytes iterator
        """
        raise NotImplementedError  # pragma: no cover

    def fetch(self, url: str) -> _DownloadStream:
        """Fetch the contents of HTTP/HTTPS ``url`` from a remote server.

        Args:
            url: URL string that represents a file location.

        Raises:
            DownloadError: An error occurred during download.
            DownloadHTTPError: An HTTP error code was received.

        Returns:
            Bytes iterator. The returned iterator must be treated as a
            resource: callers that stop reading before exhaustion (or before
            reading at all) must call ``close()`` on it so that the fetcher
            can release its response. Iterating to the end releases the
            resource as well.
        """
        # Ensure that fetch() only raises DownloadErrors, regardless of the
        # fetcher implementation
        try:
            iterator = self._fetch(url)
        except exceptions.DownloadError:
            raise
        except Exception as e:
            raise exceptions.DownloadError(f"Failed to download {url}") from e

        return _DownloadStream(iterator, url)

    @contextmanager
    def download_file(self, url: str, max_length: int) -> Iterator[IO]:
        """Download file from given ``url``.

        It is recommended to use ``download_file()`` within a ``with``
        block to guarantee that allocated file resources will always
        be released even if download fails.

        Args:
            url: URL string that represents the location of the file.
            max_length: Upper bound of file size in bytes.

        Raises:
            DownloadError: An error occurred during download.
            DownloadLengthMismatchError: Downloaded bytes exceed
                ``max_length``.
            DownloadHTTPError: An HTTP error code was received.

        Yields:
            ``TemporaryFile`` object that points to the contents of ``url``.
        """
        logger.debug("Downloading: %s", url)

        number_of_bytes_received = 0

        with tempfile.TemporaryFile() as temp_file:
            chunks = self.fetch(url)
            try:
                for chunk in chunks:
                    number_of_bytes_received += len(chunk)
                    if number_of_bytes_received > max_length:
                        raise exceptions.DownloadLengthMismatchError(
                            f"Downloaded {number_of_bytes_received} bytes"
                            f" exceeding the maximum allowed length of"
                            f" {max_length}"
                        )

                    temp_file.write(chunk)
            finally:
                # Release the response whether streaming failed, the length
                # limit was hit, or the caller stopped consuming. A stream
                # that has simply finished needs no explicit close.
                chunks.close()

            logger.debug(
                "Downloaded %d out of %d bytes",
                number_of_bytes_received,
                max_length,
            )

            temp_file.seek(0)
            yield temp_file

    def download_bytes(self, url: str, max_length: int) -> bytes:
        """Download bytes from given ``url``.

        Returns the downloaded bytes, otherwise like ``download_file()``.

        Args:
            url: URL string that represents the location of the file.
            max_length: Upper bound of data size in bytes.

        Raises:
            DownloadError: An error occurred during download.
            DownloadLengthMismatchError: Downloaded bytes exceed
                ``max_length``.
            DownloadHTTPError: An HTTP error code was received.

        Returns:
            Content of the file in bytes.
        """
        with self.download_file(url, max_length) as dl_file:
            return dl_file.read()
