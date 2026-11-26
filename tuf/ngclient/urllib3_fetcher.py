# Copyright 2021, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Provides an implementation of ``FetcherInterface`` using the urllib3 HTTP
library.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

# Imports
import urllib3

import tuf
from tuf.api import exceptions
from tuf.ngclient._internal.proxy import ProxyEnvironment
from tuf.ngclient.fetcher import FetcherInterface

if TYPE_CHECKING:
    from collections.abc import Iterator

# Globals
logger = logging.getLogger(__name__)


class _ResponseIterator:
    """Iterator that owns an urllib3 response.

    ``_fetch()`` acquires the response eagerly (so that connection and HTTP
    errors surface from ``fetch()`` instead of from the first ``next()``), but
    the body is only streamed lazily. A plain generator cannot express that:
    closing a generator that has never been started does not run its
    ``finally`` block, so the response would stay open until garbage
    collection. This class closes the response explicitly, whether or not
    streaming has started.
    """

    def __init__(
        self,
        response: urllib3.response.BaseHTTPResponse,
        chunk_size: int,
    ) -> None:
        self._response = response
        self._chunks = self._iter_chunks(response, chunk_size)
        self._completed = False
        self._closed = False

    def __iter__(self) -> _ResponseIterator:
        return self

    def __next__(self) -> bytes:
        try:
            return next(self._chunks)
        except StopIteration:
            self._completed = True
            # The stream was read to the end: the connection is clean and can
            # be returned to the pool for reuse.
            self._release()
            raise
        except Exception:
            # A read/translation error means the connection is no longer
            # usable for this download: close it before propagating.
            self.close()
            raise

    @staticmethod
    def _iter_chunks(
        response: urllib3.response.BaseHTTPResponse, chunk_size: int
    ) -> Iterator[bytes]:
        """Stream the response body, translating urllib3 errors."""
        try:
            yield from response.stream(chunk_size)
        except urllib3.exceptions.TimeoutError as e:
            # Raised directly when the gap timeout expires mid-stream:
            # ReadTimeoutError is a TimeoutError but not a MaxRetryError.
            raise exceptions.SlowRetrievalError from e
        except urllib3.exceptions.MaxRetryError as e:
            if isinstance(e.reason, urllib3.exceptions.TimeoutError):
                raise exceptions.SlowRetrievalError from e
            # Any other reason: propagate rather than ending the stream
            # silently, which would look like a complete download.
            raise

    def _release(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._completed:
            self._response.release_conn()
        else:
            # Early close (length limit exceeded, caller abort, or read
            # error): do not drain the remaining body just to reuse the
            # connection. Closing drops the connection instead, so a huge or
            # never-ending response cannot keep us busy.
            self._response.close()

    def close(self) -> None:
        """Stop streaming and release the response, started or not."""
        if self._closed:
            return
        # Closing the body generator first drops an in-flight connection
        # through urllib3's own error handling; then make sure the response
        # itself is closed as well.
        with contextlib.suppress(Exception):
            self._chunks.close()
        self._release()


# Classes
class Urllib3Fetcher(FetcherInterface):
    """An implementation of ``FetcherInterface`` based on the urllib3 library.

    Attributes:
        socket_timeout: Timeout in seconds, used for both initial connection
            delay and the maximum delay between bytes received.
        chunk_size: Chunk size in bytes used when downloading.
    """

    def __init__(
        self,
        socket_timeout: int = 30,
        chunk_size: int = 400000,
        app_user_agent: str | None = None,
    ) -> None:
        # Default settings
        self.socket_timeout: int = socket_timeout  # seconds
        self.chunk_size: int = chunk_size  # bytes

        # Create User-Agent.
        ua = f"python-tuf/{tuf.__version__}"
        if app_user_agent is not None:
            ua = f"{app_user_agent} {ua}"

        self._proxy_env = ProxyEnvironment(headers={"User-Agent": ua})

    def _fetch(self, url: str) -> Iterator[bytes]:
        """Fetch the contents of HTTP/HTTPS url from a remote server.

        Args:
            url: URL string that represents a file location.

        Raises:
            SlowRetrievalError: Timeout occurs while receiving data.
            DownloadHTTPError: HTTP error code is received.

        Returns:
            Bytes iterator
        """

        # Defer downloading the response body with preload_content=False.
        # Always set the timeout. This timeout value is interpreted by
        # urllib3 as:
        #  - connect timeout (max delay before first byte is received)
        #  - read (gap) timeout (max delay between bytes received)
        try:
            response = self._proxy_env.request(
                "GET",
                url,
                preload_content=False,
                timeout=urllib3.Timeout(self.socket_timeout),
            )
        except urllib3.exceptions.MaxRetryError as e:
            if isinstance(e.reason, urllib3.exceptions.TimeoutError):
                raise exceptions.SlowRetrievalError from e
            # Any other reason (e.g. TLS or connection error): let the original
            # error propagate instead of falling through to an unbound response.
            raise

        if response.status >= 400:
            response.close()
            raise exceptions.DownloadHTTPError(
                f"HTTP error occurred with status {response.status}",
                response.status,
            )

        return _ResponseIterator(response, self.chunk_size)
