"""Process stable media files recursively into configured barcode layouts."""

import logging
import math
import os
import stat
from collections.abc import Callable, Generator
from dataclasses import dataclass, replace
from enum import StrEnum
from os.path import abspath, normcase
from pathlib import Path
from queue import Empty, Queue
from stat import S_ISDIR, S_ISREG
from tempfile import NamedTemporaryFile
from threading import Event, Thread
from time import monotonic, sleep
from typing import Protocol

from .crop import CropMode, CropSelection
from .errors import CinebarError, FailureCategory
from .models import BatchFailure, BatchProcessingResult, BatchSkip, ProcessingResult
from .processor import SingleFileProcessor
from .progress import (
    PhaseProgress,
    ProcessingCompleted,
    ProgressPhase,
    ProgressReporter,
    ProgressState,
)

LOGGER = logging.getLogger(__name__)


_VIDEO_SUFFIXES = frozenset(
    {
        ".avi",
        ".m2ts",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mts",
        ".ts",
        ".webm",
        ".wmv",
    }
)


def _is_hidden(entry_status: os.stat_result) -> bool:
    """Return whether no-follow metadata marks an entry hidden."""
    hidden_file_attribute = getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0)
    hidden_file_flag = getattr(stat, "UF_HIDDEN", 0)
    return bool(
        getattr(entry_status, "st_file_attributes", 0) & hidden_file_attribute
        or getattr(entry_status, "st_flags", 0) & hidden_file_flag
    )


def _is_reparse_point(entry_status: os.stat_result) -> bool:
    """Return whether Windows no-follow metadata marks a reparse point."""
    reparse_point_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(
        getattr(entry_status, "st_file_attributes", 0) & reparse_point_attribute
    )


def _is_within_directory(path: Path, directory: Path, *, strict: bool) -> bool:
    """Return whether a normalized absolute path is under a directory."""
    normalized_path = normcase(abspath(path))
    normalized_directory = normcase(abspath(directory))
    if strict and normalized_path == normalized_directory:
        return False
    try:
        return (
            normcase(os.path.commonpath((normalized_path, normalized_directory)))
            == normalized_directory
        )
    except ValueError:
        return False


class OutputLayout(StrEnum):
    """Describe how batch sources map to barcode destinations."""

    BESIDE_INPUT = "beside-input"
    MIRRORED = "mirrored"
    FLAT = "flat"


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Describe settings shared by every media file in one batch.

    Args:
        height: Requested barcode height in pixels.
        workers: Process count, or ``None`` for the processor default.
        crop: Automatic, disabled, or fixed picture-region crop selection.
        overwrite: Whether existing final PNGs may be replaced.
        output_dir: Optional directory for mirrored or flat barcode destinations.
        output_layout: Explicit layout, or ``None`` for beside-input without
            ``output_dir`` and mirrored with it.
        settling_period_seconds: Positive span over which source size and
            nanosecond modification time must remain unchanged.
        readiness_timeout_seconds: Positive global limit for source readiness.
        scan_interval_seconds: Positive delay between watch directory scans.
    """

    height: int = 100
    workers: int | None = None
    crop: CropSelection = CropMode.AUTO
    overwrite: bool = False
    output_dir: Path | None = None
    output_layout: OutputLayout | None = None
    settling_period_seconds: float = 60.0
    readiness_timeout_seconds: float = 300.0
    scan_interval_seconds: float = 60.0

    def __post_init__(self) -> None:
        """Reject output directory and layout combinations without a valid meaning."""
        if (
            self.output_dir is not None
            and self.output_layout is OutputLayout.BESIDE_INPUT
        ):
            raise CinebarError(FailureCategory.INVALID_ARGUMENT)
        if self.output_dir is None and self.output_layout in (
            OutputLayout.MIRRORED,
            OutputLayout.FLAT,
        ):
            raise CinebarError(FailureCategory.INVALID_ARGUMENT)
        for duration in (
            self.settling_period_seconds,
            self.readiness_timeout_seconds,
            self.scan_interval_seconds,
        ):
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise CinebarError(FailureCategory.INVALID_ARGUMENT)


@dataclass(frozen=True, slots=True)
class _SourceVersion:
    """Record one source path and no-follow metadata observation."""

    source_path: Path
    normalized_path: str
    byte_size: int
    modification_time_ns: int


@dataclass(frozen=True, slots=True)
class _ProcessingIdentity:
    """Identify one source version and configured generation request."""

    normalized_source_path: str
    byte_size: int
    modification_time_ns: int
    normalized_output_path: str
    height: int
    workers: int | None
    crop: CropSelection
    overwrite: bool


def _processing_identity(
    source_version: _SourceVersion,
    output_path: Path,
    settings: GenerationSettings,
) -> _ProcessingIdentity:
    """Return the invocation-local identity for one generation request."""
    return _ProcessingIdentity(
        normalized_source_path=source_version.normalized_path,
        byte_size=source_version.byte_size,
        modification_time_ns=source_version.modification_time_ns,
        normalized_output_path=normcase(abspath(output_path)),
        height=settings.height,
        workers=settings.workers,
        crop=settings.crop,
        overwrite=settings.overwrite,
    )


@dataclass(frozen=True, slots=True)
class _SourcePlan:
    """Pair one source version with its output and processing identity."""

    source_version: _SourceVersion
    output_path: Path
    processing_identity: _ProcessingIdentity


@dataclass(frozen=True, slots=True)
class _SourceAttempt:
    """Record either one terminal source outcome or a changed source version."""

    outcome: ProcessingResult | BatchSkip | BatchFailure | None
    changed_version: _SourceVersion | None


type _WatchWait = Callable[[Event, float], bool]
type _WatchQueueItem = _SourcePlan | BatchFailure | BaseException


def _wait_for_watch_scan(stop_event: Event, duration: float) -> bool:
    """Wait for one watch interval or until the caller requests cancellation."""
    return stop_event.wait(duration)


def _resolve_output_path(
    root: Path,
    source_path: Path,
    settings: GenerationSettings,
) -> Path:
    """Resolve the configured barcode destination for one source path."""
    if settings.output_dir is None:
        return source_path.with_name(f"{source_path.stem}.cinebar.png")
    if settings.output_layout is OutputLayout.FLAT:
        return settings.output_dir / f"{source_path.stem}.png"
    return (
        settings.output_dir
        / source_path.parent.relative_to(root)
        / f"{source_path.stem}.cinebar.png"
    )


def _validate_root(root: Path) -> None:
    """Require one readable non-linked directory root."""
    try:
        root_status = root.stat(follow_symlinks=False)
    except FileNotFoundError:
        raise CinebarError(FailureCategory.ROOT_NOT_FOUND) from None
    except OSError:
        raise CinebarError(FailureCategory.ROOT_UNREADABLE) from None
    if _is_reparse_point(root_status) or not S_ISDIR(root_status.st_mode):
        raise CinebarError(FailureCategory.ROOT_UNREADABLE)


def _excluded_output_directory(
    root: Path,
    settings: GenerationSettings,
) -> Path | None:
    """Return the nested output tree excluded from source discovery, if any."""
    if settings.output_dir is not None and _is_within_directory(
        settings.output_dir, root, strict=True
    ):
        return settings.output_dir
    return None


def _observe_source_version(source_version: _SourceVersion) -> _SourceVersion:
    """Observe one readable regular source without following links."""
    try:
        source_status = source_version.source_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        raise CinebarError(FailureCategory.INPUT_NOT_FOUND) from None
    except OSError:
        raise CinebarError(FailureCategory.INPUT_UNREADABLE) from None

    if _is_reparse_point(source_status) or not S_ISREG(source_status.st_mode):
        raise CinebarError(FailureCategory.INPUT_UNREADABLE)

    return _SourceVersion(
        source_path=source_version.source_path,
        normalized_path=source_version.normalized_path,
        byte_size=source_status.st_size,
        modification_time_ns=source_status.st_mtime_ns,
    )


def _output_path_exists(output_path: Path) -> bool:
    """Return whether any final-path entry exists without resolving links."""
    try:
        output_path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        raise CinebarError(FailureCategory.OUTPUT_WRITE_FAILED) from None
    return True


def _reserve_unpublished_output(output_path: Path) -> Path:
    """Reserve one hidden unpublished output on the final filesystem."""
    try:
        with NamedTemporaryFile(
            delete=False,
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".unpublished",
        ) as unpublished_output:
            return Path(unpublished_output.name)
    except OSError:
        raise CinebarError(FailureCategory.OUTPUT_WRITE_FAILED) from None


def _publish_unpublished_output(
    unpublished_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    """Atomically make a complete unpublished output visible at its final path."""
    try:
        if overwrite:
            os.replace(unpublished_path, output_path)
        else:
            os.link(unpublished_path, output_path)
    except FileExistsError:
        raise CinebarError(FailureCategory.OUTPUT_EXISTS) from None
    except OSError:
        raise CinebarError(FailureCategory.OUTPUT_WRITE_FAILED) from None


def _remove_unpublished_output(unpublished_path: Path) -> None:
    """Remove one unpublished output artifact when it remains reserved."""
    try:
        unpublished_path.unlink(missing_ok=True)
    except OSError:
        raise CinebarError(FailureCategory.OUTPUT_WRITE_FAILED) from None


@dataclass(slots=True)
class _PendingSource:
    """Track one source plan until a second matching observation occurs."""

    plan: _SourcePlan
    baseline_observed_at: float


class BarcodeGenerator(Protocol):
    """Generate one barcode from one media file."""

    def process(
        self,
        media_path: Path,
        output_path: Path,
        height: int = 300,
        workers: int | None = None,
        overwrite: bool = False,
        crop: CropSelection = CropMode.AUTO,
        progress_reporter: ProgressReporter | None = None,
    ) -> ProcessingResult:
        """Generate and publish one barcode."""
        ...


class _WatchScanner:
    """Observe and settle source versions for one watch invocation."""

    def __init__(
        self,
        root: Path,
        settings: GenerationSettings,
        output_queue: Queue[_WatchQueueItem],
        stop_event: Event,
        clock: Callable[[], float],
        watch_wait: _WatchWait,
    ) -> None:
        self._root = root
        self._settings = settings
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._clock = clock
        self._watch_wait = watch_wait
        self._observed_identities: set[_ProcessingIdentity] = set()
        self._pending_candidates: dict[str, _PendingSource] = {}
        self._destination_owners: dict[str, dict[str, Path]] = {}

    def run(self) -> None:
        """Scan until stopped and forward fatal scanner exceptions to the iterator."""
        root_is_unavailable = False
        while not self._stop_event.is_set():
            try:
                self._scan()
            except CinebarError as error:
                if error.category not in {
                    FailureCategory.ROOT_NOT_FOUND,
                    FailureCategory.ROOT_UNREADABLE,
                }:
                    self._output_queue.put(error)
                    return
                if self._stop_event.is_set():
                    return
                if not root_is_unavailable:
                    LOGGER.warning(
                        "[%s] Watch root is unavailable; retrying in %g seconds.",
                        error.category.value,
                        self._settings.scan_interval_seconds,
                    )
                    root_is_unavailable = True
            except BaseException as error:
                self._output_queue.put(error)
                return
            else:
                if self._stop_event.is_set():
                    return
                if root_is_unavailable:
                    LOGGER.info("Watch root recovered; recursive discovery resumed.")
                    root_is_unavailable = False

            if self._stop_event.is_set() or self._watch_wait(
                self._stop_event,
                self._settings.scan_interval_seconds,
            ):
                return

    def _scan(self) -> None:
        """Discover every source and enqueue plans that have settled."""
        _validate_root(self._root)
        source_versions = _discover_sources(
            self._root,
            _excluded_output_directory(self._root, self._settings),
        )
        plans: list[_SourcePlan] = []
        for source_version in source_versions:
            output_path = _resolve_output_path(
                self._root,
                source_version.source_path,
                self._settings,
            )
            plans.append(
                _SourcePlan(
                    source_version=source_version,
                    output_path=output_path,
                    processing_identity=_processing_identity(
                        source_version,
                        output_path,
                        self._settings,
                    ),
                )
            )
        observed_at = self._clock()
        self._report_missing_pending_sources(plans)
        plans_by_destination: dict[str, list[_SourcePlan]] = {}
        for plan in plans:
            output_key = normcase(abspath(plan.output_path))
            plans_by_destination.setdefault(output_key, []).append(plan)
        for destination_plans in plans_by_destination.values():
            self._observe_destination_plans(destination_plans, observed_at)

    def _report_missing_pending_sources(self, plans: list[_SourcePlan]) -> None:
        """Report and discard pending versions not present in this successful scan."""
        observed_source_paths = {plan.source_version.normalized_path for plan in plans}
        missing_source_paths = sorted(
            set(self._pending_candidates) - observed_source_paths
        )
        for normalized_source_path in missing_source_paths:
            pending_source = self._pending_candidates.pop(normalized_source_path)
            self._output_queue.put(
                BatchFailure(
                    pending_source.plan.source_version.source_path,
                    FailureCategory.INPUT_NOT_FOUND,
                )
            )

    def _observe_destination_plans(
        self,
        destination_plans: list[_SourcePlan],
        observed_at: float,
    ) -> None:
        """Observe plans sharing one destination while retaining source ownership."""
        output_path = destination_plans[0].output_path
        output_key = normcase(abspath(output_path))
        source_paths = {
            plan.source_version.normalized_path: Path(
                abspath(plan.source_version.source_path)
            )
            for plan in destination_plans
        }
        destination_owners = self._destination_owners.get(output_key)
        if destination_owners is None:
            self._destination_owners[output_key] = source_paths
            if len(source_paths) >= 2:
                for plan in destination_plans:
                    self._enqueue_collision_failure(
                        plan,
                        f"Destination: {Path(abspath(output_path))}",
                    )
                return
            destination_owners = source_paths

        for plan in destination_plans:
            conflicting_source_paths = [
                source_path
                for normalized_source_path, source_path in destination_owners.items()
                if normalized_source_path != plan.source_version.normalized_path
            ]
            if conflicting_source_paths:
                safe_detail = (
                    f"Destination: {Path(abspath(output_path))}\n"
                    "Conflicting sources: "
                    + ", ".join(
                        str(source_path)
                        for source_path in sorted(
                            conflicting_source_paths,
                            key=lambda source_path: normcase(abspath(source_path)),
                        )
                    )
                )
                self._enqueue_collision_failure(plan, safe_detail)
                continue

            destination_owners.setdefault(
                plan.source_version.normalized_path,
                Path(abspath(plan.source_version.source_path)),
            )
            self._observe_plan(plan, observed_at)

    def _enqueue_collision_failure(
        self,
        plan: _SourcePlan,
        safe_detail: str,
    ) -> None:
        """Terminally report one new source version that maps to a collision."""
        if plan.processing_identity in self._observed_identities:
            return
        self._observed_identities.add(plan.processing_identity)
        self._pending_candidates.pop(plan.source_version.normalized_path, None)
        self._output_queue.put(
            BatchFailure(
                plan.source_version.source_path,
                FailureCategory.OUTPUT_COLLISION,
                safe_detail,
            )
        )

    def _observe_plan(self, plan: _SourcePlan, observed_at: float) -> None:
        """Record, settle, or supersede one current source version."""
        normalized_source_path = plan.source_version.normalized_path
        pending_source = self._pending_candidates.get(normalized_source_path)
        if pending_source is None:
            if plan.processing_identity in self._observed_identities:
                return
            self._observed_identities.add(plan.processing_identity)
            self._pending_candidates[normalized_source_path] = _PendingSource(
                plan=plan,
                baseline_observed_at=observed_at,
            )
            return

        if plan.processing_identity == pending_source.plan.processing_identity:
            if (
                observed_at - pending_source.baseline_observed_at
                >= self._settings.settling_period_seconds
            ):
                self._output_queue.put(pending_source.plan)
                del self._pending_candidates[normalized_source_path]
            return

        del self._pending_candidates[normalized_source_path]
        if plan.processing_identity in self._observed_identities:
            return
        self._observed_identities.add(plan.processing_identity)
        self._pending_candidates[normalized_source_path] = _PendingSource(
            plan=plan,
            baseline_observed_at=observed_at,
        )


class DirectoryProcessor:
    """Generate configured barcodes for stable media in one directory tree."""

    def __init__(
        self,
        barcode_generator: BarcodeGenerator | None = None,
        *,
        clock: Callable[[], float] = monotonic,
        wait: Callable[[float], None] = sleep,
        watch_wait: _WatchWait = _wait_for_watch_scan,
    ) -> None:
        """Create a directory processor with injectable generation and timing.

        Args:
            barcode_generator: Generator used for each discovered media file.
            clock: Monotonic source for readiness observation times.
            wait: Delay function used until the next batch readiness observation.
            watch_wait: Event-aware delay used until the next watch scan.
        """
        self._barcode_generator = barcode_generator or SingleFileProcessor()
        self._clock = clock
        self._wait = wait
        self._watch_wait = watch_wait
        self._observed_identities: set[_ProcessingIdentity] = set()

    def _queue_changed_source_version(
        self,
        pending_source: _PendingSource,
        current_version: _SourceVersion,
        settings: GenerationSettings,
        observed_at: float,
    ) -> bool:
        """Queue an unobserved source version for a fresh settling period."""
        output_path = pending_source.plan.output_path
        processing_identity = _processing_identity(
            current_version,
            output_path,
            settings,
        )
        if processing_identity in self._observed_identities:
            return False
        self._observed_identities.add(processing_identity)
        pending_source.plan = _SourcePlan(
            source_version=current_version,
            output_path=output_path,
            processing_identity=processing_identity,
        )
        pending_source.baseline_observed_at = observed_at
        return True

    def _process_ready_source(
        self,
        plan: _SourcePlan,
        settings: GenerationSettings,
        progress_reporter: ProgressReporter | None,
    ) -> _SourceAttempt:
        """Process one ready source without retaining readiness scheduling state."""
        source_path = plan.source_version.source_path
        output_path = plan.output_path
        unpublished_path: Path | None = None
        buffered_events: list[PhaseProgress | ProcessingCompleted] = []
        outcome: ProcessingResult | BatchSkip | BatchFailure | None = None
        changed_version: _SourceVersion | None = None
        attempt_error: BaseException | None = None
        attempt_progress_reporter: ProgressReporter | None = None

        if progress_reporter is not None:

            def buffer_attempt_progress(
                event: PhaseProgress | ProcessingCompleted,
                *,
                attempt_events: list[PhaseProgress | ProcessingCompleted] = (
                    buffered_events
                ),
                reporter: ProgressReporter = progress_reporter,
            ) -> None:
                if isinstance(event, ProcessingCompleted) or (
                    isinstance(event, PhaseProgress)
                    and event.phase is ProgressPhase.OUTPUT_PUBLICATION
                ):
                    attempt_events.append(event)
                    return
                reporter(event)

            attempt_progress_reporter = buffer_attempt_progress

        try:
            if not settings.overwrite and _output_path_exists(output_path):
                outcome = BatchSkip(source_path, output_path)
            else:
                try:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                except OSError:
                    raise CinebarError(FailureCategory.OUTPUT_WRITE_FAILED) from None
                unpublished_path = _reserve_unpublished_output(output_path)

                current_version = _observe_source_version(plan.source_version)
                if current_version != plan.source_version:
                    changed_version = current_version
                else:
                    unpublished_result = self._barcode_generator.process(
                        media_path=source_path,
                        output_path=unpublished_path,
                        height=settings.height,
                        workers=settings.workers,
                        overwrite=True,
                        crop=settings.crop,
                        progress_reporter=attempt_progress_reporter,
                    )
                    published_result = replace(
                        unpublished_result,
                        output_path=output_path,
                    )

                    if progress_reporter is not None:
                        for event in buffered_events:
                            if (
                                isinstance(event, PhaseProgress)
                                and event.state is not ProgressState.COMPLETED
                            ):
                                progress_reporter(event)

                    current_version = _observe_source_version(plan.source_version)
                    if current_version != plan.source_version:
                        changed_version = current_version
                    else:
                        _publish_unpublished_output(
                            unpublished_path,
                            output_path,
                            overwrite=settings.overwrite,
                        )
                        outcome = published_result

                        if progress_reporter is not None:
                            for event in buffered_events:
                                if (
                                    isinstance(event, PhaseProgress)
                                    and event.state is ProgressState.COMPLETED
                                ):
                                    progress_reporter(event)
                                elif isinstance(event, ProcessingCompleted):
                                    progress_reporter(
                                        replace(event, result=published_result)
                                    )
        except CinebarError as error:
            attempt_error = error
            if error.category is FailureCategory.INTERRUPTED:
                raise
            outcome = BatchFailure(source_path, error.category, error.safe_detail)
        except KeyboardInterrupt as error:
            attempt_error = error
            raise
        finally:
            if unpublished_path is not None:
                try:
                    _remove_unpublished_output(unpublished_path)
                except CinebarError as cleanup_error:
                    if cleanup_error.category is FailureCategory.INTERRUPTED:
                        raise
                    if attempt_error is None and changed_version is None:
                        outcome = BatchFailure(
                            source_path,
                            cleanup_error.category,
                            cleanup_error.safe_detail,
                        )
                    elif attempt_error is not None:
                        attempt_error.add_note(str(cleanup_error))

        if changed_version is not None:
            return _SourceAttempt(outcome=None, changed_version=changed_version)
        return _SourceAttempt(outcome=outcome, changed_version=None)

    def process_batch(
        self,
        root: Path,
        settings: GenerationSettings = GenerationSettings(),  # noqa: B008
        progress_reporter: ProgressReporter | None = None,
    ) -> BatchProcessingResult:
        """Generate one barcode for every supported media file using settings.

        Args:
            root: Directory tree containing stable source media files.
            settings: Shared generation settings for every source media file.
            progress_reporter: Optional recipient for per-file progress facts.

        Returns:
            Completed barcode results and readiness failures.
        """
        _validate_root(root)

        excluded_directory = (
            settings.output_dir
            if settings.output_dir is not None
            and _is_within_directory(settings.output_dir, root, strict=True)
            else None
        )
        source_versions = _discover_sources(root, excluded_directory)
        output_sources: dict[str, tuple[Path, dict[str, Path]]] = {}
        plans: list[_SourcePlan] = []
        discovered_identities: set[_ProcessingIdentity] = set()
        for source_version in source_versions:
            source_path = source_version.source_path
            output_path = _resolve_output_path(root, source_path, settings)
            output_key = normcase(abspath(output_path))
            source_paths = output_sources.setdefault(
                output_key,
                (Path(abspath(output_path)), {}),
            )[1]
            source_paths.setdefault(
                source_version.normalized_path,
                Path(abspath(source_path)),
            )
            processing_identity = _processing_identity(
                source_version,
                output_path,
                settings,
            )
            if (
                processing_identity in self._observed_identities
                or processing_identity in discovered_identities
            ):
                continue
            discovered_identities.add(processing_identity)
            plans.append(
                _SourcePlan(
                    source_version=source_version,
                    output_path=output_path,
                    processing_identity=processing_identity,
                )
            )

        collision_destinations = {
            output_key: destination
            for output_key, (destination, source_paths) in output_sources.items()
            if len(source_paths) >= 2
        }
        completed: list[ProcessingResult] = []
        skipped: list[BatchSkip] = []
        failures: list[BatchFailure] = []
        readiness_started_at = self._clock()
        readiness_deadline = readiness_started_at + settings.readiness_timeout_seconds
        pending_sources: list[_PendingSource] = []
        for plan in plans:
            collision_destination = collision_destinations.get(
                normcase(abspath(plan.output_path))
            )
            self._observed_identities.add(plan.processing_identity)
            if collision_destination is not None:
                failures.append(
                    BatchFailure(
                        plan.source_version.source_path,
                        FailureCategory.OUTPUT_COLLISION,
                        f"Destination: {collision_destination}",
                    )
                )
                continue
            pending_sources.append(
                _PendingSource(
                    plan=plan,
                    baseline_observed_at=readiness_started_at,
                )
            )

        if not pending_sources:
            return BatchProcessingResult(
                completed=tuple(completed),
                skipped=tuple(skipped),
                failures=tuple(failures),
            )

        while pending_sources:
            current_time = self._clock()
            if current_time > readiness_deadline:
                failures.extend(
                    BatchFailure(
                        pending_source.plan.source_version.source_path,
                        FailureCategory.READINESS_TIMEOUT,
                    )
                    for pending_source in sorted(
                        pending_sources,
                        key=lambda source: source.plan.source_version.normalized_path,
                    )
                )
                break

            next_observation_at = min(
                pending_source.baseline_observed_at + settings.settling_period_seconds
                for pending_source in pending_sources
            )
            if next_observation_at > readiness_deadline:
                failures.extend(
                    BatchFailure(
                        pending_source.plan.source_version.source_path,
                        FailureCategory.READINESS_TIMEOUT,
                    )
                    for pending_source in sorted(
                        pending_sources,
                        key=lambda source: source.plan.source_version.normalized_path,
                    )
                )
                break

            if current_time < next_observation_at:
                self._wait(next_observation_at - current_time)
                continue

            due_sources = sorted(
                (
                    pending_source
                    for pending_source in pending_sources
                    if pending_source.baseline_observed_at
                    + settings.settling_period_seconds
                    <= current_time
                ),
                key=lambda source: source.plan.source_version.normalized_path,
            )
            if not due_sources:
                continue

            ready_sources: list[_PendingSource] = []
            due_source_ids = {id(pending_source) for pending_source in due_sources}
            next_pending_sources = [
                pending_source
                for pending_source in pending_sources
                if id(pending_source) not in due_source_ids
            ]
            for pending_source in due_sources:
                source_path = pending_source.plan.source_version.source_path
                try:
                    current_version = _observe_source_version(
                        pending_source.plan.source_version
                    )
                except CinebarError as error:
                    if error.category is FailureCategory.INTERRUPTED:
                        raise
                    failures.append(
                        BatchFailure(source_path, error.category, error.safe_detail)
                    )
                    continue

                if current_version == pending_source.plan.source_version:
                    ready_sources.append(pending_source)
                    continue

                if self._queue_changed_source_version(
                    pending_source,
                    current_version,
                    settings,
                    current_time,
                ):
                    next_pending_sources.append(pending_source)

            pending_sources = next_pending_sources
            for pending_source in ready_sources:
                source_attempt = self._process_ready_source(
                    pending_source.plan,
                    settings,
                    progress_reporter,
                )
                if source_attempt.changed_version is not None:
                    if self._queue_changed_source_version(
                        pending_source,
                        source_attempt.changed_version,
                        settings,
                        self._clock(),
                    ):
                        pending_sources.append(pending_source)
                    continue
                if isinstance(source_attempt.outcome, ProcessingResult):
                    completed.append(source_attempt.outcome)
                elif isinstance(source_attempt.outcome, BatchSkip):
                    skipped.append(source_attempt.outcome)
                elif isinstance(source_attempt.outcome, BatchFailure):
                    failures.append(source_attempt.outcome)

        return BatchProcessingResult(
            completed=tuple(completed),
            skipped=tuple(skipped),
            failures=tuple(failures),
        )

    def process_watch(
        self,
        root: Path,
        settings: GenerationSettings = GenerationSettings(),  # noqa: B008
        progress_reporter: ProgressReporter | None = None,
    ) -> Generator[ProcessingResult | BatchSkip | BatchFailure, None, None]:
        """Watch a directory tree and yield each terminal source outcome.

        Args:
            root: Directory tree containing source media files to watch.
            settings: Shared generation settings for every source media file.
            progress_reporter: Optional recipient for per-file progress facts.

        Yields:
            Completed, skipped, and failed source outcomes as they become final.
        """
        output_queue: Queue[_WatchQueueItem] = Queue()
        stop_event = Event()
        scanner = _WatchScanner(
            root,
            settings,
            output_queue,
            stop_event,
            self._clock,
            self._watch_wait,
        )
        scanner_thread = Thread(target=scanner.run, daemon=False)
        scanner_thread.start()
        try:
            while True:
                try:
                    queued_item = output_queue.get(timeout=0.1)
                except Empty:
                    continue
                if isinstance(queued_item, BaseException):
                    raise queued_item
                if isinstance(queued_item, BatchFailure):
                    yield queued_item
                    continue

                source_attempt = self._process_ready_source(
                    queued_item,
                    settings,
                    progress_reporter,
                )
                if (
                    source_attempt.changed_version is None
                    and source_attempt.outcome is not None
                ):
                    yield source_attempt.outcome
        finally:
            stop_event.set()
            scanner_thread.join()


def _discover_sources(
    root: Path,
    excluded_directory: Path | None = None,
) -> list[_SourceVersion]:
    """Return supported regular files found without following links.

    Args:
        root: Directory tree to scan.
        excluded_directory: Optional directory tree omitted from discovery.
    """
    directories = [root]
    sources: list[_SourceVersion] = []

    while directories:
        directory = directories.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    try:
                        entry_status = entry.stat(follow_symlinks=False)
                    except OSError:
                        raise CinebarError(FailureCategory.ROOT_UNREADABLE) from None
                    if _is_hidden(entry_status) or _is_reparse_point(entry_status):
                        continue

                    entry_path = Path(entry.path)
                    if excluded_directory is not None and _is_within_directory(
                        entry_path,
                        excluded_directory,
                        strict=False,
                    ):
                        continue
                    if S_ISDIR(entry_status.st_mode):
                        directories.append(entry_path)
                    elif S_ISREG(entry_status.st_mode) and (
                        entry_path.suffix.casefold() in _VIDEO_SUFFIXES
                    ):
                        sources.append(
                            _SourceVersion(
                                source_path=entry_path,
                                normalized_path=normcase(abspath(entry_path)),
                                byte_size=entry_status.st_size,
                                modification_time_ns=entry_status.st_mtime_ns,
                            )
                        )
        except CinebarError:
            raise
        except OSError:
            raise CinebarError(FailureCategory.ROOT_UNREADABLE) from None

    return sorted(
        sources,
        key=lambda source: (source.normalized_path, abspath(source.source_path)),
    )
