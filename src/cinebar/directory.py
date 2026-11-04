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
from threading import Event, RLock, Thread
from time import monotonic, sleep
from typing import Literal, Protocol

from .crop import CropMode, CropSelection
from .errors import CinebarError, FailureCategory
from .models import BatchFailure, BatchProcessingResult, BatchSkip, ProcessingResult
from .processor import SingleFileProcessor
from .progress import (
    DirectoryProgress,
    DirectoryProgressReporter,
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
    """Remove an unpublished output and any nested encoder temporary files."""
    cleanup_failed = False
    temporary_pattern = f".{unpublished_path.name}.*.tmp"
    try:
        for temporary_path in unpublished_path.parent.glob(temporary_pattern):
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                cleanup_failed = True
    except OSError:
        cleanup_failed = True
    try:
        unpublished_path.unlink(missing_ok=True)
    except OSError:
        cleanup_failed = True
    if cleanup_failed:
        raise CinebarError(FailureCategory.OUTPUT_WRITE_FAILED)


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


class _DirectoryProgressTracker:
    """Maintain truthful directory-workflow state and lifecycle logging."""

    def __init__(
        self,
        command: Literal["batch", "watch"],
        reporter: DirectoryProgressReporter | None,
    ) -> None:
        """Create one thread-safe tracker for a directory command."""
        if command not in {"batch", "watch"}:
            raise ValueError("command must be 'batch' or 'watch'.")
        self._reporter = reporter
        self._lock = RLock()
        self._discovered_versions: set[tuple[str, int, int]] = set()
        self._waiting: set[_ProcessingIdentity] = set()
        self._queued: set[_ProcessingIdentity] = set()
        self._completed: set[_ProcessingIdentity] = set()
        self._skipped: set[_ProcessingIdentity] = set()
        self._failed: set[_ProcessingIdentity] = set()
        self._current_plan: _SourcePlan | None = None
        self._next_scan_seconds: float | None = None
        self._finished = False

    def discover(self, source_version: _SourceVersion) -> None:
        """Record one source version when first observed during this invocation."""
        source_key = (
            source_version.normalized_path,
            source_version.byte_size,
            source_version.modification_time_ns,
        )
        with self._lock:
            if source_key in self._discovered_versions:
                return
            self._discovered_versions.add(source_key)
            self._emit_snapshot()
        LOGGER.info("Discovered source.\nSource: %s", source_version.source_path)

    def start_waiting(self, plan: _SourcePlan) -> None:
        """Mark one discovered plan as waiting for a stable observation."""
        with self._lock:
            if self._is_terminal(plan.processing_identity):
                return
            if plan.processing_identity in self._waiting:
                return
            self._waiting.add(plan.processing_identity)
            self._emit_snapshot()

    def mark_queued(self, plan: _SourcePlan) -> None:
        """Mark one settled source plan ready for sequential processing."""
        with self._lock:
            if self._is_terminal(plan.processing_identity):
                return
            if plan.processing_identity in self._queued:
                return
            self._waiting.discard(plan.processing_identity)
            self._queued.add(plan.processing_identity)
            self._emit_snapshot()

    def start_processing(self, plan: _SourcePlan) -> None:
        """Mark one queued source plan as the active generator request."""
        with self._lock:
            if (
                self._current_plan is not None
                and self._current_plan.processing_identity == plan.processing_identity
            ):
                return
            self._waiting.discard(plan.processing_identity)
            self._queued.discard(plan.processing_identity)
            self._current_plan = plan
            self._emit_snapshot()
        LOGGER.info(
            "Processing source.\nSource: %s\nDestination: %s",
            plan.source_version.source_path,
            plan.output_path,
        )

    def record_source_change(
        self,
        plan: _SourcePlan,
        changed_version: _SourceVersion,
    ) -> None:
        """Clear stale active state and count a changed source version."""
        changed_source_key = (
            changed_version.normalized_path,
            changed_version.byte_size,
            changed_version.modification_time_ns,
        )
        with self._lock:
            state_changed = self._discard_active(plan.processing_identity)
            state_changed = (
                changed_source_key not in self._discovered_versions or state_changed
            )
            self._discovered_versions.add(changed_source_key)
            if not state_changed:
                return
            self._emit_snapshot()
        LOGGER.info(
            "Source changed; the new version must settle before processing.\n"
            "Source: %s",
            changed_version.source_path,
        )

    def record_outcome(
        self,
        plan: _SourcePlan,
        outcome: ProcessingResult | BatchSkip | BatchFailure,
    ) -> None:
        """Record one terminal source outcome and emit its lifecycle event."""
        identity = plan.processing_identity
        with self._lock:
            if self._is_terminal(identity):
                return
            self._discard_active(identity)
            if isinstance(outcome, ProcessingResult):
                self._completed.add(identity)
            elif isinstance(outcome, BatchSkip):
                self._skipped.add(identity)
            else:
                self._failed.add(identity)
            self._emit_snapshot()
        self._log_outcome(plan, outcome)

    def schedule_next_scan(self, delay_seconds: float) -> None:
        """Expose a healthy watch scan delay for idle presentation."""
        with self._lock:
            if self._next_scan_seconds == delay_seconds:
                return
            self._next_scan_seconds = delay_seconds
            self._emit_snapshot()

    def clear_next_scan(self) -> None:
        """Clear idle-watch timing while a scan cannot proceed normally."""
        with self._lock:
            if self._next_scan_seconds is None:
                return
            self._next_scan_seconds = None
            self._emit_snapshot()

    def finish(self, *, interrupted: bool) -> None:
        """Clear transient gauges and publish the final workflow snapshot."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._waiting.clear()
            self._queued.clear()
            self._current_plan = None
            self._next_scan_seconds = None
            self._emit_snapshot()
        if interrupted:
            LOGGER.info("Directory processing was interrupted; shutting down.")

    def _is_terminal(self, identity: _ProcessingIdentity) -> bool:
        """Return whether one processing identity already reached an outcome."""
        return (
            identity in self._completed
            or identity in self._skipped
            or identity in self._failed
        )

    def _discard_active(self, identity: _ProcessingIdentity) -> bool:
        """Remove one identity from transient gauges without touching newer work."""
        was_active = identity in self._waiting or identity in self._queued
        self._waiting.discard(identity)
        self._queued.discard(identity)
        if (
            self._current_plan is not None
            and self._current_plan.processing_identity == identity
        ):
            self._current_plan = None
            return True
        return was_active

    def _emit_snapshot(self) -> None:
        """Send a complete immutable snapshot while the state lock is held."""
        if self._reporter is None:
            return
        self._reporter(
            DirectoryProgress(
                discovered=len(self._discovered_versions),
                waiting=len(self._waiting),
                queued=len(self._queued),
                completed=len(self._completed),
                skipped=len(self._skipped),
                failed=len(self._failed),
                current_source_path=(
                    None
                    if self._current_plan is None
                    else self._current_plan.source_version.source_path
                ),
                next_scan_seconds=self._next_scan_seconds,
            )
        )

    def _log_outcome(
        self,
        plan: _SourcePlan,
        outcome: ProcessingResult | BatchSkip | BatchFailure,
    ) -> None:
        """Log one terminal event after its tracker snapshot is current."""
        if isinstance(outcome, ProcessingResult):
            LOGGER.info(
                "Completed source.\nSource: %s\nOutput: %s",
                plan.source_version.source_path,
                outcome.output_path,
            )
            return
        if isinstance(outcome, BatchSkip):
            LOGGER.info(
                "Skipped existing mapped output without verifying source or "
                "generation settings.\nSource: %s\nDestination: %s",
                outcome.source_path,
                outcome.output_path,
            )
            return
        detail = f"Source: {outcome.source_path}"
        if outcome.safe_detail is not None:
            detail = f"{detail}\n{outcome.safe_detail}"
        failure_error = CinebarError(outcome.category, detail)
        LOGGER.error("[%s] %s", failure_error.category.value, failure_error)


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
        progress_tracker: _DirectoryProgressTracker,
    ) -> None:
        self._root = root
        self._settings = settings
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._clock = clock
        self._watch_wait = watch_wait
        self._progress_tracker = progress_tracker
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
                self._progress_tracker.clear_next_scan()
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
                self._progress_tracker.schedule_next_scan(
                    self._settings.scan_interval_seconds
                )

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
        for plan in plans:
            self._progress_tracker.discover(plan.source_version)
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
            failure = BatchFailure(
                pending_source.plan.source_version.source_path,
                FailureCategory.INPUT_NOT_FOUND,
            )
            self._progress_tracker.record_outcome(pending_source.plan, failure)
            self._output_queue.put(failure)

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
        failure = BatchFailure(
            plan.source_version.source_path,
            FailureCategory.OUTPUT_COLLISION,
            safe_detail,
        )
        self._progress_tracker.record_outcome(plan, failure)
        self._output_queue.put(failure)

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
            self._progress_tracker.start_waiting(plan)
            return

        if plan.processing_identity == pending_source.plan.processing_identity:
            if (
                observed_at - pending_source.baseline_observed_at
                >= self._settings.settling_period_seconds
            ):
                self._progress_tracker.mark_queued(pending_source.plan)
                self._output_queue.put(pending_source.plan)
                del self._pending_candidates[normalized_source_path]
            return

        del self._pending_candidates[normalized_source_path]
        if plan.processing_identity in self._observed_identities:
            return
        self._observed_identities.add(plan.processing_identity)
        self._progress_tracker.record_source_change(
            pending_source.plan,
            plan.source_version,
        )
        self._pending_candidates[normalized_source_path] = _PendingSource(
            plan=plan,
            baseline_observed_at=observed_at,
        )
        self._progress_tracker.start_waiting(plan)


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
        progress_tracker: _DirectoryProgressTracker,
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
        previous_plan = pending_source.plan
        pending_source.plan = _SourcePlan(
            source_version=current_version,
            output_path=output_path,
            processing_identity=processing_identity,
        )
        pending_source.baseline_observed_at = observed_at
        progress_tracker.record_source_change(previous_plan, current_version)
        progress_tracker.start_waiting(pending_source.plan)
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
        directory_progress_reporter: DirectoryProgressReporter | None = None,
    ) -> BatchProcessingResult:
        """Generate one barcode for every supported media file using settings.

        Args:
            root: Directory tree containing stable source media files.
            settings: Shared generation settings for every source media file.
            progress_reporter: Optional recipient for per-file progress facts.
            directory_progress_reporter: Optional recipient for directory facts.

        Returns:
            Completed barcode results and readiness failures.
        """
        progress_tracker = _DirectoryProgressTracker(
            "batch",
            directory_progress_reporter,
        )
        interrupted = False
        completed_normally = False
        try:
            _validate_root(root)
            excluded_directory = _excluded_output_directory(root, settings)
            source_versions = _discover_sources(root, excluded_directory)
            output_sources: dict[str, tuple[Path, dict[str, Path]]] = {}
            plans: list[_SourcePlan] = []
            discovered_identities: set[_ProcessingIdentity] = set()
            for source_version in source_versions:
                progress_tracker.discover(source_version)
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
            readiness_deadline = (
                readiness_started_at + settings.readiness_timeout_seconds
            )
            pending_sources: list[_PendingSource] = []
            for plan in plans:
                collision_destination = collision_destinations.get(
                    normcase(abspath(plan.output_path))
                )
                self._observed_identities.add(plan.processing_identity)
                if collision_destination is not None:
                    failure = BatchFailure(
                        plan.source_version.source_path,
                        FailureCategory.OUTPUT_COLLISION,
                        f"Destination: {collision_destination}",
                    )
                    failures.append(failure)
                    progress_tracker.record_outcome(plan, failure)
                    continue
                pending_source = _PendingSource(
                    plan=plan,
                    baseline_observed_at=readiness_started_at,
                )
                pending_sources.append(pending_source)
                progress_tracker.start_waiting(plan)

            while pending_sources:
                current_time = self._clock()
                timed_out_sources: list[_PendingSource] = []
                if current_time > readiness_deadline:
                    timed_out_sources = sorted(
                        pending_sources,
                        key=lambda source: source.plan.source_version.normalized_path,
                    )
                else:
                    next_observation_at = min(
                        pending_source.baseline_observed_at
                        + settings.settling_period_seconds
                        for pending_source in pending_sources
                    )
                    if next_observation_at > readiness_deadline:
                        timed_out_sources = sorted(
                            pending_sources,
                            key=lambda source: (
                                source.plan.source_version.normalized_path
                            ),
                        )
                if timed_out_sources:
                    for pending_source in timed_out_sources:
                        failure = BatchFailure(
                            pending_source.plan.source_version.source_path,
                            FailureCategory.READINESS_TIMEOUT,
                        )
                        failures.append(failure)
                        progress_tracker.record_outcome(pending_source.plan, failure)
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
                        failure = BatchFailure(
                            source_path,
                            error.category,
                            error.safe_detail,
                        )
                        failures.append(failure)
                        progress_tracker.record_outcome(pending_source.plan, failure)
                        continue

                    if current_version == pending_source.plan.source_version:
                        ready_sources.append(pending_source)
                        continue

                    if self._queue_changed_source_version(
                        pending_source,
                        current_version,
                        settings,
                        current_time,
                        progress_tracker,
                    ):
                        next_pending_sources.append(pending_source)

                pending_sources = next_pending_sources
                for pending_source in ready_sources:
                    progress_tracker.mark_queued(pending_source.plan)
                for pending_source in ready_sources:
                    progress_tracker.start_processing(pending_source.plan)
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
                            progress_tracker,
                        ):
                            pending_sources.append(pending_source)
                        continue
                    if isinstance(source_attempt.outcome, ProcessingResult):
                        completed.append(source_attempt.outcome)
                        progress_tracker.record_outcome(
                            pending_source.plan,
                            source_attempt.outcome,
                        )
                    elif isinstance(source_attempt.outcome, BatchSkip):
                        skipped.append(source_attempt.outcome)
                        progress_tracker.record_outcome(
                            pending_source.plan,
                            source_attempt.outcome,
                        )
                    elif isinstance(source_attempt.outcome, BatchFailure):
                        failures.append(source_attempt.outcome)
                        progress_tracker.record_outcome(
                            pending_source.plan,
                            source_attempt.outcome,
                        )

            completed_normally = True
            return BatchProcessingResult(
                completed=tuple(completed),
                skipped=tuple(skipped),
                failures=tuple(failures),
            )
        except CinebarError as error:
            interrupted = error.category is FailureCategory.INTERRUPTED
            raise
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            progress_tracker.finish(interrupted=interrupted)
            if interrupted:
                LOGGER.info("Batch shutdown complete.")
            elif completed_normally:
                LOGGER.info("Batch processing complete.")

    def process_watch(
        self,
        root: Path,
        settings: GenerationSettings = GenerationSettings(),  # noqa: B008
        progress_reporter: ProgressReporter | None = None,
        directory_progress_reporter: DirectoryProgressReporter | None = None,
    ) -> Generator[ProcessingResult | BatchSkip | BatchFailure, None, None]:
        """Watch a directory tree and yield each terminal source outcome.

        Args:
            root: Directory tree containing source media files to watch.
            settings: Shared generation settings for every source media file.
            progress_reporter: Optional recipient for per-file progress facts.
            directory_progress_reporter: Optional recipient for directory facts.

        Yields:
            Completed, skipped, and failed source outcomes as they become final.
        """
        progress_tracker = _DirectoryProgressTracker(
            "watch",
            directory_progress_reporter,
        )
        output_queue: Queue[_WatchQueueItem] = Queue()
        stop_event = Event()
        scanner = _WatchScanner(
            root,
            settings,
            output_queue,
            stop_event,
            self._clock,
            self._watch_wait,
            progress_tracker,
        )
        scanner_thread = Thread(target=scanner.run, daemon=False)
        interrupted = False
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

                progress_tracker.start_processing(queued_item)
                source_attempt = self._process_ready_source(
                    queued_item,
                    settings,
                    progress_reporter,
                )
                if source_attempt.changed_version is not None:
                    progress_tracker.record_source_change(
                        queued_item,
                        source_attempt.changed_version,
                    )
                    continue
                if source_attempt.outcome is not None:
                    progress_tracker.record_outcome(
                        queued_item,
                        source_attempt.outcome,
                    )
                    yield source_attempt.outcome
        except CinebarError as error:
            interrupted = error.category is FailureCategory.INTERRUPTED
            raise
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            stop_event.set()
            scanner_thread.join()
            progress_tracker.finish(interrupted=interrupted)
            LOGGER.info("Watch shutdown complete.")


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
