# Failure boundaries for target downloads

* Status: accepted
* Date: 2026-11-26

## Context and Problem Statement

The ``Updater.download_target()`` path used to stream a target directly into
the destination path (``open(filepath, "wb")``). If anything failed while
writing, an existing, complete target file was truncated to whatever had been
written so far. Two related resource leaks existed as well:

* ``FetcherInterface.fetch()`` returned a generator. ``Urllib3Fetcher`` acquired
  its ``HTTPResponse`` eagerly (so connection/HTTP errors surface from
  ``fetch()``), but a generator that has never been started does not run its
  ``finally`` block when ``close()`` is called. Closing before the first
  ``next()`` therefore leaked the response until garbage collection.
* When a download exceeded ``max_length``, the response was left open. Draining
  the rest of the body in order to reuse the connection is not acceptable for
  an oversized or endless response.

We need a clear, layered contract: which layer owns the download stream, which
layer owns the destination file, where (and when) the handover happens, and how
every failure unwinds.

## Decision Drivers

* A failed download must never damage the previous destination file or leave a
  partial file at the destination path.
* Network resources (response/connection) must be released deterministically,
  not by garbage collection, including when the caller aborts before the first
  chunk.
* Memory use must not scale with the target size.
* ``Updater``/``FetcherInterface`` public APIs and successful return values
  stay unchanged.
* Ordinary files, serial calls: no power-loss recovery and no concurrency
  requirements.

## Considered Options

1. *Write directly to the destination* (the old behaviour). Rejected: it
   destroys the old file on failure.
2. *Read the whole target into memory, verify, then write.* Rejected: memory
   grows with file size.
3. *Stream into the destination's own directory under a temporary name,
   verify in place, then ``os.replace()``.* This keeps memory constant and makes
   publication atomic, but verification happens after the download has already
   touched the destination filesystem.
4. *Let the fetcher stream into its own temporary file and verify there; only a
   fully verified file is copied to a temporary file in the destination
   directory and renamed.* This is option 3 plus a clear ownership boundary.

## Decision Outcome

Chosen option 4, with the following ownership split.

### Fetcher layer (``FetcherInterface`` / ``Urllib3Fetcher`` / ``RequestsFetcher``)

* ``fetch()`` returns a ``_DownloadStream``: an idempotent, always-closeable
  iterator adapter. Calling ``close()`` releases the underlying response even if
  no chunk was ever requested, and any exception raised while streaming is
  surfaced as ``DownloadError`` (consistent with what ``fetch()`` already
  promised for eager errors).
* ``Urllib3Fetcher._fetch()`` acquires the response eagerly but wraps it in a
  ``_ResponseIterator`` that owns the response explicitly:
  * iteration to the end → ``response.release_conn()`` (clean connection,
    eligible for pooling/reuse);
  * explicit ``close()`` at any point → ``response.close()``. The remaining
    body is deliberately **not** drained, so an oversized or endless response
    cannot block the caller; the connection is simply discarded.
  * a mid-stream error → ``response.close()`` before the error propagates.
* ``download_file()`` streams chunks into an anonymous ``TemporaryFile`` with a
  running byte counter (rejecting as soon as ``max_length`` is exceeded) and
  always closes the download stream in a ``finally`` block. The fetcher's
  temporary file therefore never survives a failure, and it lives outside the
  destination directory, so a failed download cannot even transiently pollute
  the target directory.

### Updater layer (``Updater.download_target()``)

* The ``download_file()`` context owns the temporary download file and the
  response; when the context exits, both are gone.
* Verification (length and hashes) happens on that temporary file, i.e. before
  anything is written under the destination path.
* Only then does ``_publish_file()`` run: it streams the verified file into a
  ``NamedTemporaryFile`` **in the destination directory** (same filesystem, so
  ``os.replace()`` is a rename), preserves the permission bits of an existing
  destination (or the umask-derived mode for a new file), then atomically
  replaces the destination. On any failure the temporary file is removed and
  the destination is untouched.
* The handover point is therefore: *the fetcher stops owning bytes once
  ``download_file()`` has yielded a verified-able local file; the Updater takes
  ownership of publishing only after verification passed, and publishes in one
  atomic step.*

### Consequences

* **Disk cost:** the payload is written twice — once by the fetcher into its
  temporary file, once by the Updater into the destination-directory temporary
  file — plus one rename. This is the price of keeping memory constant while
  still verifying before touching the destination. The extra copy is
  sequential streaming, not an additional in-memory buffer.
* **Stream reuse:** a fully read response returns its connection to the pool;
  an aborted, oversized or failed one is closed instead of being drained. This
  trades a small amount of connection reuse on aborted downloads for a bounded,
  predictable abort.
* **Exception unwinding:** every layer closes what it owns in ``finally``
  blocks. An exception in any step leaves (a) no half-written target path,
  (b) no leftover publish temporary file, (c) no open response or download
  stream, and (d) the caller free to retry on the same ``Updater``/fetcher.

## Links

* Regression tests: ``tests/test_updater_download_failure.py`` and the response
  ownership cases in ``tests/test_fetcher_ng.py``.
