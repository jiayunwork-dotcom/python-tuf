# Copyright 2020, New York University and the TUF contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Client update workflow implementation.

The ``Updater`` class provides an implementation of the
`TUF client workflow
<https://theupdateframework.github.io/specification/latest/#detailed-client-workflow>`_.
``Updater`` provides an API to query available targets and to download them in a
secure manner: All downloaded files are verified by signed metadata.

High-level description of ``Updater`` functionality:
  * Initializing an ``Updater`` loads and validates the trusted local root
    metadata: This root metadata is used as the source of trust for all other
    metadata. Updater should always be initialized with the ``bootstrap``
    argument: pass ``bootstrap=None`` only to explicitly opt into using the
    cached root.json as the trust anchor.
  * ``refresh()`` downloads, verifies, and loads metadata for the top-level
    roles as described in the specification. It may be called any number of
    times on the same ``Updater`` instance, including retrying after a failed
    refresh. Each successful refresh atomically replaces the previously
    available metadata view: until a refresh succeeds, ``get_targetinfo()``
    keeps serving the most recent complete successful view. Version lower
    bounds of authenticated top-level metadata are persisted immediately and
    continue to constrain later refreshes, including after an ``Updater`` is
    rebuilt over the same metadata directory.
  * ``Updater`` can be used to download targets. For each target:

      * ``Updater.get_targetinfo()`` is first used to find information about a
        specific target. This will load new targets metadata as needed (from
        local cache or remote repository).
      * ``Updater.find_cached_target()`` can optionally be used to check if a
        target file is already locally cached.
      * ``Updater.download_target()`` downloads a target file and ensures it is
        verified correct by the metadata.

Note that applications using ``Updater`` should be 'single instance'
applications: running multiple instances that use the same cache directories at
the same time is not supported.

A simple example of using the Updater to implement a Python TUF client that
downloads target files is available in `examples/client
<https://github.com/theupdateframework/python-tuf/tree/develop/examples/client>`_.

Notes on how Updater uses HTTP by default:
  * urllib3 is the HTTP library
  * Typically all requests are retried by urllib3 three times (in cases where
    this seems useful)
  * Operating system certificate store is used for TLS, in other words
    ``certifi`` is not used as the certificate source
  * Proxy use can be configured with ``https_proxy`` and other similar
    environment variables

All of the HTTP decisions can be changed with ``fetcher`` argument:
Custom ``FetcherInterface`` implementations are possible. The alternative
``RequestsFetcher`` implementation is also provided (although deprecated).
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib import parse

from tuf.api import exceptions
from tuf.api.metadata import Root, Snapshot, TargetFile, Targets, Timestamp
from tuf.ngclient._internal.trusted_metadata_set import TrustedMetadataSet
from tuf.ngclient.config import EnvelopeType, UpdaterConfig
from tuf.ngclient.urllib3_fetcher import Urllib3Fetcher

if TYPE_CHECKING:
    from tuf.ngclient.fetcher import FetcherInterface

logger = logging.getLogger(__name__)


class Updater:
    """Creates a new ``Updater`` instance and loads trusted root metadata.

    Args:
        metadata_dir: Local metadata directory. Directory must be
            writable. If ``bootstrap`` is ``None``, this directory must contain
            a trusted root.json file.
        metadata_base_url: Base URL for all remote metadata downloads
        target_dir: Local targets directory. Directory must be writable. It
            will be used as the default target download directory by
            ``find_cached_target()`` and ``download_target()``
        target_base_url: ``Optional``; Default base URL for all remote target
            downloads. Can be individually set in ``download_target()``
        fetcher: ``Optional``; ``FetcherInterface`` implementation used to
            download both metadata and targets. Default is ``Urllib3Fetcher``
        config: ``Optional``; ``UpdaterConfig`` could be used to setup common
            configuration options. Configuration cannot be changed after
            initialization: create a new ``Updater`` to use different options.
        bootstrap: Initial root metadata bytes. This argument is required.
            Pass the embedded root metadata bytes for secure initialization.
            Pass ``None`` only if you explicitly want to use the cached
            root.json as the trust anchor (not recommended for most
            deployments).

    Raises:
        OSError: Local root.json cannot be read
        RepositoryError: Local root.json is invalid
    """

    def __init__(  # noqa: PLR0917
        self,
        metadata_dir: str,
        metadata_base_url: str,
        target_dir: str | None = None,
        target_base_url: str | None = None,
        fetcher: FetcherInterface | None = None,
        config: UpdaterConfig | None = None,
        *,
        bootstrap: bytes | None,
    ):
        self._dir = metadata_dir
        self._metadata_base_url = _ensure_trailing_slash(metadata_base_url)
        self.target_dir = target_dir
        if target_base_url is None:
            self._target_base_url = None
        else:
            self._target_base_url = _ensure_trailing_slash(target_base_url)

        self._config = config or UpdaterConfig()
        if fetcher is not None:
            self._fetcher = fetcher
        else:
            self._fetcher = Urllib3Fetcher(
                app_user_agent=self.config.app_user_agent
            )
        supported_envelopes = [EnvelopeType.METADATA, EnvelopeType.SIMPLE]
        if self.config.envelope_type not in supported_envelopes:
            raise ValueError(
                f"config: envelope_type must be one of {supported_envelopes}, "
                f"got '{self.config.envelope_type}'"
            )

        if bootstrap is None:
            # if no root was provided, use the cached non-versioned root.json
            bootstrap = self._load_local_metadata(Root.type)

        # The trust anchor is retained so that every refresh can build a new
        # candidate TrustedMetadataSet from exactly the same starting point.
        # This makes an in-process retry equivalent to rebuilding the Updater
        # over the same metadata directory.
        self._bootstrap = bootstrap

        # ``_trusted_set`` is the last *complete and successful* metadata view.
        # It is the only view used by target lookup and is replaced atomically
        # when a refresh finishes successfully: a failed refresh discards its
        # candidate and never mutates this view.
        self._trusted_set = TrustedMetadataSet(
            bootstrap, self.config.envelope_type
        )
        self._persist_root(self._trusted_set.root.version, bootstrap)
        self._update_root_symlink()

    @property
    def config(self) -> UpdaterConfig:
        """Read-only configuration used for this Updater's lifetime."""
        return self._config

    def refresh(self) -> None:
        """Refresh top-level metadata.

        Downloads, verifies, and loads metadata for the top-level roles in the
        specified order (root -> timestamp -> snapshot -> targets) implementing
        all the checks required in the TUF client workflow.

        ``refresh()`` may be called repeatedly on the same ``Updater``
        instance. Every call builds a fresh candidate
        ``TrustedMetadataSet`` from the bootstrap root and the persisted
        metadata (including rollback anchors written by earlier, possibly
        failed, rounds), so a failed round cannot leave partially updated
        metadata in the view used by ``get_targetinfo()`` and a later call can
        simply retry.

        The most recent complete successful view stays available for target
        lookups until a new refresh completes successfully. Authenticated
        root/timestamp/snapshot version advances made by a round are persisted
        immediately -- even when a later check in that same round fails -- so
        they keep protecting against rollback attacks on both in-process
        retries and freshly constructed instances using the same metadata
        directory. Metadata that fails signature or trust-chain verification is
        never persisted as an anchor and therefore cannot raise the rollback
        version lower bound.

        A successful refresh atomically replaces the serving view, including
        its delegated metadata cache. Delegated roles visited under an older
        view are consequently read from disk and verified again against the
        new snapshot/delegations before they are reused.

        If ``refresh()`` has not been explicitly called before the first
        ``get_targetinfo()`` call, it will be done implicitly at that time.

        The metadata for delegated roles is not updated by ``refresh()``:
        that happens on demand during ``get_targetinfo()``. However, if the
        repository uses `consistent_snapshot
        <https://theupdateframework.github.io/specification/latest/#consistent-snapshots>`_,
        then all metadata downloaded by the Updater will use the same consistent
        repository state.

        Raises:
            OSError: New metadata could not be written to disk
            RepositoryError: Metadata failed to verify in some way
            DownloadError: Download of a metadata file failed in some way
        """

        # Build the entire new top-level view in a private candidate. The
        # serving view (``self._trusted_set``) is untouched until every
        # top-level role has been verified, so any exception below leaves the
        # last complete view fully usable.
        candidate = TrustedMetadataSet(
            self._bootstrap, self.config.envelope_type
        )

        self._load_root(candidate)
        self._load_timestamp(candidate)
        self._load_snapshot(candidate)
        self._load_targets(Targets.type, Root.type, candidate)

        # Complete top-level update: publish the new view atomically. This also
        # drops the per-view delegated metadata cache, forcing re-verification
        # under the new snapshot/delegations on the next target lookup.
        self._trusted_set = candidate

    def _generate_target_file_path(self, targetinfo: TargetFile) -> str:
        if self.target_dir is None:
            raise ValueError("target_dir must be set if filepath is not given")

        # Use URL encoded target path as filename
        filename = parse.quote(targetinfo.path, "")
        return os.path.join(self.target_dir, filename)

    def get_targetinfo(self, target_path: str) -> TargetFile | None:
        """Return ``TargetFile`` instance with information for ``target_path``.

        The return value can be used as an argument to
        ``download_target()`` and ``find_cached_target()``.

        If ``refresh()`` has not been called before calling
        ``get_targetinfo()``, the refresh will be done implicitly.

        As a side-effect this method downloads all the additional (delegated
        targets) metadata it needs to return the target information.

        Args:
            target_path: `path-relative-URL string
                <https://url.spec.whatwg.org/#path-relative-url-string>`_
                that uniquely identifies the target within the repository.

        Raises:
            OSError: New metadata could not be written to disk
            RepositoryError: Metadata failed to verify in some way
            DownloadError: Download of a metadata file failed in some way

        Returns:
            ``TargetFile`` instance or ``None``.
        """

        if Targets.type not in self._trusted_set:
            self.refresh()
        return self._preorder_depth_first_walk(target_path)

    def find_cached_target(
        self,
        targetinfo: TargetFile,
        filepath: str | None = None,
    ) -> str | None:
        """Check whether a local file is an up to date target.

        Args:
            targetinfo: ``TargetFile`` from ``get_targetinfo()``.
            filepath: Local path to file. If ``None``, a file path is
                generated based on ``target_dir`` constructor argument.

        Raises:
            ValueError: Incorrect arguments

        Returns:
            Local file path if the file is an up to date target file.
            ``None`` if file is not found or it is not up to date.
        """

        if filepath is None:
            filepath = self._generate_target_file_path(targetinfo)

        try:
            with open(filepath, "rb") as target_file:
                targetinfo.verify_length_and_hashes(target_file)
            return filepath
        except (OSError, exceptions.LengthOrHashMismatchError):
            return None

    def download_target(
        self,
        targetinfo: TargetFile,
        filepath: str | None = None,
        target_base_url: str | None = None,
    ) -> str:
        """Download the target file specified by ``targetinfo``.

        Args:
            targetinfo: ``TargetFile`` from ``get_targetinfo()``.
            filepath: Local path to download into. If ``None``, the file is
                downloaded into directory defined by ``target_dir`` constructor
                argument using a generated filename. If file already exists,
                it is overwritten.
            target_base_url: Base URL used to form the final target
                download URL. Default is the value provided in ``Updater()``

        Raises:
            ValueError: Invalid arguments
            DownloadError: Download of the target file failed in some way
            RepositoryError: Downloaded target failed to be verified in some way
            OSError: Failed to write target to file

        Returns:
            Local path to downloaded file
        """

        if filepath is None:
            filepath = self._generate_target_file_path(targetinfo)
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)

        if target_base_url is None:
            if self._target_base_url is None:
                raise ValueError(
                    "target_base_url must be set in either "
                    "download_target() or constructor"
                )

            target_base_url = self._target_base_url
        else:
            target_base_url = _ensure_trailing_slash(target_base_url)

        target_filepath = targetinfo.path
        consistent_snapshot = self._trusted_set.root.consistent_snapshot
        if consistent_snapshot and self.config.prefix_targets_with_hash:
            hashes = list(targetinfo.hashes.values())
            dirname, sep, basename = target_filepath.rpartition("/")
            target_filepath = f"{dirname}{sep}{hashes[0]}.{basename}"
        full_url = f"{target_base_url}{target_filepath}"

        with self._fetcher.download_file(
            full_url, targetinfo.length
        ) as target_file:
            targetinfo.verify_length_and_hashes(target_file)

            target_file.seek(0)
            with open(filepath, "wb") as destination_file:
                shutil.copyfileobj(target_file, destination_file)

        logger.debug("Downloaded target %s", targetinfo.path)
        return filepath

    def _download_metadata(
        self, rolename: str, length: int, version: int | None = None
    ) -> bytes:
        """Download a metadata file and return it as bytes."""
        encoded_name = parse.quote(rolename, "")
        if version is None:
            url = f"{self._metadata_base_url}{encoded_name}.json"
        else:
            url = f"{self._metadata_base_url}{version}.{encoded_name}.json"
        return self._fetcher.download_bytes(url, length)

    def _load_local_metadata(self, rolename: str) -> bytes:
        encoded_name = parse.quote(rolename, "")
        with open(os.path.join(self._dir, f"{encoded_name}.json"), "rb") as f:
            return f.read()

    def _persist_metadata(self, rolename: str, data: bytes) -> None:
        """Write metadata to disk atomically to avoid data loss.

        Use a filename _not_ prefixed with version (e.g. "timestamp.json")
        . Encode the rolename to avoid issues with e.g. path separators
        """

        encoded_name = parse.quote(rolename, "")
        filename = os.path.join(self._dir, f"{encoded_name}.json")
        self._persist_file(filename, data)

    # Directory for rollback anchors: authenticated top-level metadata that a
    # round accepted into a candidate as an *intermediate* rollback boundary
    # but that did not become a final view (e.g. an expired timestamp or a
    # snapshot whose version does not match timestamp yet). The regular
    # unversioned cache files only contain final metadata; these anchors make
    # sure the higher version lower bounds survive both a retry on this
    # instance and a rebuild over the same metadata directory.
    #
    # A subdirectory (instead of another top-level "<role>.json" file) keeps
    # the regular metadata cache layout and its file-list expectations intact.
    _ANCHOR_DIRNAME = "rollback_anchors"

    def _anchor_file(self, rolename: str) -> str:
        encoded_name = parse.quote(rolename, "")
        return os.path.join(
            self._dir, self._ANCHOR_DIRNAME, f"{encoded_name}.json"
        )

    def _load_local_anchor(self, rolename: str) -> bytes | None:
        """Read a persisted rollback anchor for ``rolename``.

        Returns ``None`` when no anchor exists. Existence is checked with
        ``os.path.isfile`` (not by attempting ``open``) so a metadata cache
        without anchors produces no extra ``open()`` calls.
        """
        anchor = self._anchor_file(rolename)
        if not os.path.isfile(anchor):
            return None
        with open(anchor, "rb") as f:
            return f.read()

    def _persist_anchor(self, rolename: str, data: bytes) -> None:
        """Persist an authenticated intermediate rollback anchor."""
        anchor_dir = os.path.join(self._dir, self._ANCHOR_DIRNAME)
        os.makedirs(anchor_dir, exist_ok=True)
        self._persist_file(self._anchor_file(rolename), data)

    def _remove_anchor(self, rolename: str) -> None:
        """Drop a rollback anchor superseded by final metadata.

        Cleanup is best-effort: a leftover anchor cannot weaken security (it
        is re-verified on replay and can only establish an older-or-equal
        lower bound than the freshly persisted final metadata).
        """
        try:
            os.remove(self._anchor_file(rolename))
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.debug("Could not remove %s rollback anchor: %s", rolename, e)

    def _persist_root(self, version: int, data: bytes) -> None:
        """Write root metadata to disk atomically to avoid data loss.

        The metadata is stored with version prefix (e.g.
        "root_history/1.root.json").
        """
        rootdir = Path(self._dir, "root_history")
        rootdir.mkdir(exist_ok=True, parents=True)
        self._persist_file(str(rootdir / f"{version}.root.json"), data)

    def _persist_file(self, filename: str, data: bytes) -> None:
        """Write a file to disk atomically to avoid data loss."""
        temp_file_name = None

        try:
            with tempfile.NamedTemporaryFile(
                dir=self._dir, delete=False
            ) as temp_file:
                temp_file_name = temp_file.name
                temp_file.write(data)
            os.replace(temp_file.name, filename)
        except OSError as e:
            # remove tempfile if we managed to create one,
            # then let the exception happen
            if temp_file_name is not None:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(temp_file_name)
            raise e

    def _update_root_symlink(self, version: int | None = None) -> None:
        """Symlink root.json to current trusted root version in root_history/

        When ``version`` is not given, the version of the serving view is
        used. During a refresh the candidate's newest version is passed so the
        symlink advances even if the refresh later fails: every root in
        root_history has already been authenticated and the advanced symlink
        is also the correct trust anchor for ``bootstrap=None`` rebuilds.
        """
        linkname = os.path.join(self._dir, "root.json")
        if version is None:
            version = self._trusted_set.root.version
        current = os.path.join("root_history", f"{version}.root.json")
        with contextlib.suppress(FileNotFoundError):
            os.remove(linkname)
        try:
            os.symlink(current, linkname)
        except OSError as e:
            if getattr(e, "winerror", None) == 1314:
                # Fallback for NTFS "required privilege is not held by client"
                os.link(os.path.join(self._dir, current), linkname)
            else:
                raise

    def _load_root(self, trusted_set: TrustedMetadataSet | None = None) -> None:
        """Load root metadata into ``trusted_set``.

        Sequentially load newer root metadata versions. First try to load from
        local cache and if that does not work, from the remote repository.

        If metadata is loaded from remote repository, store it in local cache.
        Every accepted root is persisted immediately (by version in
        root_history), so authenticated root version advances survive both a
        failed round and an instance rebuild.

        When ``trusted_set`` is not given the serving view is used; refresh()
        always passes its private candidate.
        """
        if trusted_set is None:
            trusted_set = self._trusted_set

        # Update the root role
        lower_bound = trusted_set.root.version + 1
        upper_bound = lower_bound + self.config.max_root_rotations

        try:
            for next_version in range(lower_bound, upper_bound):
                # look for next_version in local cache
                try:
                    root_path = os.path.join(
                        self._dir, "root_history", f"{next_version}.root.json"
                    )
                    with open(root_path, "rb") as f:
                        trusted_set.update_root(f.read())
                    continue
                except (OSError, exceptions.RepositoryError) as e:
                    # this root did not exist locally or is invalid
                    logger.debug("Local root is not valid: %s", e)

                # next_version was not found locally, try remote
                try:
                    data = self._download_metadata(
                        Root.type,
                        self.config.root_max_length,
                        next_version,
                    )
                    # update_root() only accepts data that passed the previous
                    # root's threshold signatures, the version check and the
                    # new root's self-signature. A failure raises before the
                    # root is added, so the bytes below are only persisted once
                    # the version lower bound has actually advanced.
                    trusted_set.update_root(data)
                    self._persist_root(next_version, data)

                except exceptions.DownloadHTTPError as exception:
                    if exception.status_code not in {403, 404}:
                        raise
                    # 404/403 means current root is newest available
                    break
        finally:
            # Make sure the non-versioned root.json links to current version
            self._update_root_symlink(trusted_set.root.version)

    def _load_timestamp(
        self, trusted_set: TrustedMetadataSet | None = None
    ) -> None:
        """Load local and remote timestamp metadata into ``trusted_set``."""
        if trusted_set is None:
            trusted_set = self._trusted_set
        try:
            data = self._load_local_metadata(Timestamp.type)
            trusted_set.update_timestamp(data)
        except (OSError, exceptions.RepositoryError) as e:
            # Local timestamp does not exist or is invalid
            logger.debug("Local timestamp not valid as final: %s", e)

        # Replay a rollback anchor from an earlier failed round. Like the
        # regular local cache this is best-effort: an expired anchor is still
        # added as an intermediate timestamp and therefore protects against
        # rollback of both timestamp and snapshot versions.
        self._load_timestamp_anchor(trusted_set)

        # Load from remote (whether local load succeeded or not)
        data = self._download_metadata(
            Timestamp.type, self.config.timestamp_max_length
        )
        previous = trusted_set.get(Timestamp.type)
        try:
            trusted_set.update_timestamp(data)
        except exceptions.EqualVersionNumberError:
            # If the new timestamp version is the same as current, discard the
            # new timestamp. This is normal and it shouldn't raise any error.
            return
        except exceptions.RepositoryError:
            # If update_timestamp() added the new timestamp before the final
            # check failed (the only such case is an expired timestamp), the
            # candidate now holds an authenticated intermediate rollback
            # boundary. Persist it so the boundary survives a retry/rebuild;
            # verification or rollback failures happen before _add() and must
            # not be persisted.
            if trusted_set.get(Timestamp.type) is not previous:
                self._persist_anchor(Timestamp.type, data)
            raise

        self._persist_metadata(Timestamp.type, data)
        self._remove_anchor(Timestamp.type)

    def _load_timestamp_anchor(self, trusted_set: TrustedMetadataSet) -> None:
        """Best-effort replay of a persisted timestamp rollback anchor."""
        try:
            data = self._load_local_anchor(Timestamp.type)
            if data is None:
                return
            trusted_set.update_timestamp(data)
        except OSError as e:
            logger.debug("Could not read timestamp rollback anchor: %s", e)
        except exceptions.RepositoryError as e:
            # Expired anchors are still loaded as intermediate metadata by
            # TrustedMetadataSet; any other rejection just means the anchor is
            # not usable under the current root and can be ignored here.
            logger.debug("Timestamp rollback anchor not final: %s", e)

    def _load_snapshot(
        self, trusted_set: TrustedMetadataSet | None = None
    ) -> None:
        """Load local (and if needed remote) snapshot metadata."""
        if trusted_set is None:
            trusted_set = self._trusted_set
        try:
            data = self._load_local_metadata(Snapshot.type)
            trusted_set.update_snapshot(data, trusted=True)
            logger.debug("Loaded local snapshot as trusted metadata")
        except (OSError, exceptions.RepositoryError) as e:
            # Local snapshot does not exist or is invalid: the anchor replay
            # and/or the remote download below will provide the rollback state
            logger.debug("Local snapshot not valid as final: %s", e)

        # Replay a possible rollback anchor from an earlier failed round. This
        # is done even if the regular local snapshot loaded successfully: a
        # higher authenticated anchor can turn that snapshot intermediate and
        # must then force a remote update instead of being silently ignored.
        self._load_snapshot_anchor(trusted_set)

        if trusted_set.is_final_snapshot():
            logger.debug("Local snapshot is final: not downloading new one")
            return

        snapshot_meta = trusted_set.timestamp.snapshot_meta
        length = snapshot_meta.length or self.config.snapshot_max_length
        version = None
        if trusted_set.root.consistent_snapshot:
            version = snapshot_meta.version

        data = self._download_metadata(Snapshot.type, length, version)
        previous = trusted_set.get(Snapshot.type)
        try:
            trusted_set.update_snapshot(data)
        except exceptions.RepositoryError:
            # As with timestamp: if the candidate's snapshot object changed,
            # update_snapshot() accepted and added the new snapshot before a
            # final check (expiry/version match) failed. Persist those bytes
            # as a rollback anchor; failures before _add() changed nothing.
            if trusted_set.get(Snapshot.type) is not previous:
                self._persist_anchor(Snapshot.type, data)
            raise

        self._persist_metadata(Snapshot.type, data)
        self._remove_anchor(Snapshot.type)

    def _load_snapshot_anchor(self, trusted_set: TrustedMetadataSet) -> None:
        """Best-effort replay of a persisted snapshot rollback anchor."""
        try:
            data = self._load_local_anchor(Snapshot.type)
            if data is None:
                return
            # The anchor was fully verified when first accepted; load it as
            # trusted local metadata, like the regular cached snapshot.
            trusted_set.update_snapshot(data, trusted=True)
        except OSError as e:
            logger.debug("Could not read snapshot rollback anchor: %s", e)
        except exceptions.RepositoryError as e:
            # Intermediate anchors stay in the set for rollback protection;
            # anything unusable under the current timestamp is simply ignored.
            logger.debug("Snapshot rollback anchor not final: %s", e)

    def _load_targets(
        self,
        role: str,
        parent_role: str,
        trusted_set: TrustedMetadataSet | None = None,
    ) -> Targets:
        """Load local (and if needed remote) metadata for ``role``.

        When ``trusted_set`` is not given, the current serving view is used,
        which is the target-lookup path. refresh() passes its candidate to
        load the top-level targets of the view under construction.
        """
        if trusted_set is None:
            trusted_set = self._trusted_set

        # Avoid loading 'role' more than once into this view for the same
        # parent_role. The cache is per TrustedMetadataSet/view: every
        # successful refresh installs a fresh set, so metadata verified in an
        # earlier round is never reused without re-verification.
        if trusted_set.contains(role, parent_role):
            return cast("Targets", trusted_set[role])

        try:
            data = self._load_local_metadata(role)
            delegated_targets = trusted_set.update_delegated_targets(
                data, role, parent_role
            )
            logger.debug("Local %s is valid: not downloading new one", role)
            return delegated_targets
        except (OSError, exceptions.RepositoryError) as e:
            # Local 'role' does not exist or is invalid: update from remote
            logger.debug("Failed to load local %s: %s", role, e)

            snapshot = trusted_set.snapshot
            metainfo = snapshot.meta.get(f"{role}.json")
            if metainfo is None:
                raise exceptions.RepositoryError(
                    f"Role {role} was delegated but is not part of snapshot"
                ) from None

            length = metainfo.length or self.config.targets_max_length
            version = None
            if trusted_set.root.consistent_snapshot:
                version = metainfo.version

            data = self._download_metadata(role, length, version)
            delegated_targets = trusted_set.update_delegated_targets(
                data, role, parent_role
            )
            self._persist_metadata(role, data)

            return delegated_targets

    def _preorder_depth_first_walk(
        self, target_filepath: str
    ) -> TargetFile | None:
        """
        Interrogates the tree of target delegations in order of appearance
        (which implicitly order trustworthiness), and returns the matching
        target found in the most trusted role.
        """

        # List of delegations to be interrogated. A (role, parent role) pair
        # is needed to load and verify the delegated targets metadata.
        delegations_to_visit = [(Targets.type, Root.type)]
        visited_role_names: set[str] = set()

        # Preorder depth-first traversal of the graph of target delegations.
        while (
            len(visited_role_names) <= self.config.max_delegations
            and len(delegations_to_visit) > 0
        ):
            # Pop the role name from the top of the stack.
            role_name, parent_role = delegations_to_visit.pop(-1)

            # Skip any visited current role to prevent cycles.
            if role_name in visited_role_names:
                logger.debug("Skipping visited current role %s", role_name)
                continue

            # The metadata for 'role_name' must be downloaded/updated before
            # its targets, delegations, and child roles can be inspected.
            # Target lookup always operates on the current serving view: the
            # last successfully refreshed TrustedMetadataSet. Its per-view
            # cache avoids repeated loads within one view, while a successful
            # refresh replaces the whole set and thereby invalidates entries
            # verified under the previous snapshot/delegations.
            targets = self._load_targets(role_name, parent_role)

            target = targets.targets.get(target_filepath)

            if target is not None:
                logger.debug("Found target in current role %s", role_name)
                return target

            # After preorder check, add current role to set of visited roles.
            visited_role_names.add(role_name)

            if targets.delegations is not None:
                child_roles_to_visit = []
                # NOTE: This may be a slow operation if there are many
                # delegated roles.
                for (
                    child_name,
                    terminating,
                ) in targets.delegations.get_roles_for_target(target_filepath):
                    logger.debug("Adding child role %s", child_name)
                    child_roles_to_visit.append((child_name, role_name))
                    if terminating:
                        logger.debug("Not backtracking to other roles")
                        delegations_to_visit = []
                        break
                # Push 'child_roles_to_visit' in reverse order of appearance
                # onto 'delegations_to_visit'.  Roles are popped from the end of
                # the list.
                child_roles_to_visit.reverse()
                delegations_to_visit.extend(child_roles_to_visit)

        if len(delegations_to_visit) > 0:
            logger.debug(
                "%d roles left to visit, but allowed at most %d delegations",
                len(delegations_to_visit),
                self.config.max_delegations,
            )

        # If this point is reached then target is not found, return None
        return None


def _ensure_trailing_slash(url: str) -> str:
    """Return url guaranteed to end in a slash."""
    return url if url.endswith("/") else f"{url}/"
