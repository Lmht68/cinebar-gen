"""Application-owned progress facts for complete barcode processing."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from time import monotonic

from .crop import CropRectangle
from .models import ProcessingResult


class ProgressPhase(StrEnum):
    """Name one observable processing phase."""

    CROP_ANALYSIS = "crop_analysis"
    FRAME_DECODING = "frame_decoding"
    PNG_ENCODING = "png_encoding"
    OUTPUT_PUBLICATION = "output_publication"


class ProgressState(StrEnum):
    """Describe the lifecycle state of one processing phase."""

    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"


class ProgressUnit(StrEnum):
    """Name the unit counted by a processing phase."""

    SAMPLES = "samples"
    FRAMES = "frames"
    ROWS = "rows"
    FILES = "files"


_PHASE_UNITS: dict[ProgressPhase, ProgressUnit] = {
    ProgressPhase.CROP_ANALYSIS: ProgressUnit.SAMPLES,
    ProgressPhase.FRAME_DECODING: ProgressUnit.FRAMES,
    ProgressPhase.PNG_ENCODING: ProgressUnit.ROWS,
    ProgressPhase.OUTPUT_PUBLICATION: ProgressUnit.FILES,
}


def _require_nonnegative_integer(value: int, field_name: str) -> None:
    """Validate one non-negative integer progress value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")


def _require_nonnegative_number(value: float, field_name: str) -> None:
    """Validate one finite non-negative progress measurement."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite non-negative number.")
    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number.")


@dataclass(frozen=True, slots=True)
class PhaseProgress:
    """One immutable observation from an active processing phase.

    Args:
        phase: The active processing phase.
        state: Lifecycle state for ``phase``.
        completed_units: Truthful completed work in ``unit``.
        total_units: Known or estimated total work, if available.
        unit: Unit counted by ``phase``.
        fraction_complete: Trustworthy completion fraction, if available.
        is_estimate: Whether the supplied total or fraction is an estimate.
        elapsed_seconds: Monotonic time since this phase started.
        rate_per_second: Completed units per elapsed second, if measurable.
        eta_seconds: Remaining duration when the fraction supports estimation.
        worker_count: Active decoder process count for frame decoding only.

    Raises:
        ValueError: If values are malformed, inconsistent, or impossible.
    """

    phase: ProgressPhase
    state: ProgressState
    completed_units: int
    total_units: int | None
    unit: ProgressUnit
    fraction_complete: float | None
    is_estimate: bool
    elapsed_seconds: float
    rate_per_second: float | None
    eta_seconds: float | None
    worker_count: int | None

    def __post_init__(self) -> None:
        """Validate phase, count, duration, estimate, and worker invariants."""
        if not isinstance(self.phase, ProgressPhase):
            raise ValueError("phase must be a ProgressPhase.")
        if not isinstance(self.state, ProgressState):
            raise ValueError("state must be a ProgressState.")
        if not isinstance(self.unit, ProgressUnit):
            raise ValueError("unit must be a ProgressUnit.")
        if self.unit is not _PHASE_UNITS[self.phase]:
            raise ValueError("unit must match phase.")
        _require_nonnegative_integer(self.completed_units, "completed_units")
        if self.total_units is not None:
            _require_nonnegative_integer(self.total_units, "total_units")
            if self.completed_units > self.total_units:
                raise ValueError("completed_units must not exceed total_units.")
        if self.fraction_complete is not None:
            _require_nonnegative_number(self.fraction_complete, "fraction_complete")
            if self.fraction_complete > 1.0:
                raise ValueError("fraction_complete must not exceed 1.0.")
        if not isinstance(self.is_estimate, bool):
            raise ValueError("is_estimate must be a bool.")
        _require_nonnegative_number(self.elapsed_seconds, "elapsed_seconds")
        if self.rate_per_second is not None:
            _require_nonnegative_number(self.rate_per_second, "rate_per_second")
        if self.eta_seconds is not None:
            _require_nonnegative_number(self.eta_seconds, "eta_seconds")
            if self.fraction_complete is None or not 0 < self.fraction_complete < 1:
                raise ValueError(
                    "eta_seconds requires an incomplete positive fraction."
                )
        if self.state is ProgressState.COMPLETED:
            if self.fraction_complete != 1.0:
                raise ValueError("completed progress must have fraction_complete 1.0.")
            if self.eta_seconds is not None:
                raise ValueError("completed progress must not have eta_seconds.")
        if self.worker_count is not None:
            if self.phase is not ProgressPhase.FRAME_DECODING:
                raise ValueError("worker_count is only valid for frame decoding.")
            if (
                isinstance(self.worker_count, bool)
                or not isinstance(self.worker_count, int)
                or self.worker_count <= 0
            ):
                raise ValueError("worker_count must be a positive integer.")


@dataclass(frozen=True, slots=True)
class ProcessingCompleted:
    """The immutable final fact emitted after successful barcode publication.

    Args:
        result: Published barcode dimensions and final path.
        elapsed_seconds: Total monotonic processing duration.
        average_frames_per_second: Average complete-pass decode rate, if measurable.
        worker_count: Actual active decoder process count.
        accepted_crop: Crop decision accepted for the complete barcode.

    Raises:
        ValueError: If duration, rate, or worker count is invalid.
    """

    result: ProcessingResult
    elapsed_seconds: float
    average_frames_per_second: float | None
    worker_count: int
    accepted_crop: CropRectangle | None

    def __post_init__(self) -> None:
        """Validate success-summary measurements."""
        if not isinstance(self.result, ProcessingResult):
            raise ValueError("result must be a ProcessingResult.")
        _require_nonnegative_number(self.elapsed_seconds, "elapsed_seconds")
        if self.average_frames_per_second is not None:
            _require_nonnegative_number(
                self.average_frames_per_second,
                "average_frames_per_second",
            )
        if (
            isinstance(self.worker_count, bool)
            or not isinstance(self.worker_count, int)
            or self.worker_count <= 0
        ):
            raise ValueError("worker_count must be a positive integer.")
        if self.accepted_crop is not None and not isinstance(
            self.accepted_crop, CropRectangle
        ):
            raise ValueError("accepted_crop must be a CropRectangle or None.")


type ProgressEvent = PhaseProgress | ProcessingCompleted
type ProgressReporter = Callable[[ProgressEvent], None]
type UnitProgressCallback = Callable[[int], None]


@dataclass(slots=True)
class _ActivePhase:
    """Internal mutable details retained while one phase is active."""

    phase: ProgressPhase
    unit: ProgressUnit
    started_at: float
    total_units: int | None
    is_estimate: bool
    worker_count: int | None
    last_snapshot_at: float


class ProgressTracker:
    """Translate phase work facts into throttled validated reporter events."""

    _MINIMUM_SNAPSHOT_INTERVAL_SECONDS = 0.1

    def __init__(
        self,
        reporter: Callable[[PhaseProgress], None],
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Create a tracker using one reporter and monotonic clock.

        Args:
            reporter: Recipient for immutable progress events.
            clock: Monotonic source returning seconds.
        """
        self._reporter = reporter
        self._clock = clock
        self._active_phase: _ActivePhase | None = None

    def start(
        self,
        phase: ProgressPhase,
        unit: ProgressUnit,
        *,
        total_units: int | None = None,
        is_estimate: bool = False,
        worker_count: int | None = None,
    ) -> None:
        """Start one phase and emit its initial event.

        Args:
            phase: Phase becoming active.
            unit: Unit counted by ``phase``.
            total_units: Known or estimated total work, if available.
            is_estimate: Whether the total or forthcoming fraction is estimated.
            worker_count: Actual decoder workers for frame decoding only.

        Raises:
            RuntimeError: If another phase remains active.
            ValueError: If event values violate phase invariants.
        """
        if self._active_phase is not None:
            raise RuntimeError("A progress phase is already active.")
        started_at = self._clock()
        fraction_complete = 0.0 if total_units is not None else None
        event = PhaseProgress(
            phase=phase,
            state=ProgressState.STARTED,
            completed_units=0,
            total_units=total_units,
            unit=unit,
            fraction_complete=fraction_complete,
            is_estimate=is_estimate,
            elapsed_seconds=0.0,
            rate_per_second=None,
            eta_seconds=None,
            worker_count=worker_count,
        )
        self._active_phase = _ActivePhase(
            phase=phase,
            unit=unit,
            started_at=started_at,
            total_units=total_units,
            is_estimate=is_estimate,
            worker_count=worker_count,
            last_snapshot_at=started_at,
        )
        self._reporter(event)

    def update(
        self,
        completed_units: int,
        *,
        total_units: int | None = None,
        fraction_complete: float | None = None,
        is_estimate: bool | None = None,
        worker_count: int | None = None,
        estimate_eta: bool = False,
    ) -> None:
        """Optionally emit one current phase snapshot after at least 0.1 seconds.

        Args:
            completed_units: Truthful work completed by the active phase.
            total_units: Replacement total when it becomes known.
            fraction_complete: Trustworthy presentation fraction, if available.
            is_estimate: Replacement estimate marker, if it changes.
            worker_count: Replacement actual decoder worker count.
            estimate_eta: Whether ``fraction_complete`` supports an ETA.

        Raises:
            RuntimeError: If no phase is active.
            ValueError: If event values violate phase invariants.
        """
        active_phase = self._require_active_phase()
        now = self._clock()
        if (
            now - active_phase.last_snapshot_at
            < self._MINIMUM_SNAPSHOT_INTERVAL_SECONDS
        ):
            return
        self._emit_snapshot(
            active_phase,
            now,
            completed_units=completed_units,
            total_units=total_units,
            fraction_complete=fraction_complete,
            is_estimate=is_estimate,
            worker_count=worker_count,
            estimate_eta=estimate_eta,
        )

    def complete(
        self,
        completed_units: int,
        *,
        total_units: int | None = None,
        is_estimate: bool | None = None,
        worker_count: int | None = None,
    ) -> PhaseProgress:
        """Complete the active phase and emit an unconditional final snapshot.

        Args:
            completed_units: Final truthful completed work.
            total_units: Replacement total when final work establishes it.
            is_estimate: Replacement estimate marker, if it changes.
            worker_count: Replacement actual decoder worker count.

        Returns:
            The emitted completed phase fact.

        Raises:
            RuntimeError: If no phase is active.
            ValueError: If event values violate phase invariants.
        """
        active_phase = self._require_active_phase()
        now = self._clock()
        resolved_total = (
            active_phase.total_units if total_units is None else total_units
        )
        resolved_estimate = (
            active_phase.is_estimate if is_estimate is None else is_estimate
        )
        resolved_workers = (
            active_phase.worker_count if worker_count is None else worker_count
        )
        elapsed_seconds = now - active_phase.started_at
        rate_per_second = (
            completed_units / elapsed_seconds if elapsed_seconds > 0 else None
        )
        event = PhaseProgress(
            phase=active_phase.phase,
            state=ProgressState.COMPLETED,
            completed_units=completed_units,
            total_units=resolved_total,
            unit=active_phase.unit,
            fraction_complete=1.0,
            is_estimate=resolved_estimate,
            elapsed_seconds=elapsed_seconds,
            rate_per_second=rate_per_second,
            eta_seconds=None,
            worker_count=resolved_workers,
        )
        self._active_phase = None
        self._reporter(event)
        return event

    def _require_active_phase(self) -> _ActivePhase:
        """Return the one active phase or reject an invalid lifecycle transition."""
        if self._active_phase is None:
            raise RuntimeError("No progress phase is active.")
        return self._active_phase

    def _emit_snapshot(
        self,
        active_phase: _ActivePhase,
        now: float,
        *,
        completed_units: int,
        total_units: int | None,
        fraction_complete: float | None,
        is_estimate: bool | None,
        worker_count: int | None,
        estimate_eta: bool,
    ) -> None:
        """Emit one validated active-phase running snapshot."""
        resolved_total = (
            active_phase.total_units if total_units is None else total_units
        )
        resolved_fraction = fraction_complete
        if resolved_fraction is None and resolved_total is not None:
            resolved_fraction = (
                completed_units / resolved_total if resolved_total else 1.0
            )
        resolved_estimate = (
            active_phase.is_estimate if is_estimate is None else is_estimate
        )
        resolved_workers = (
            active_phase.worker_count if worker_count is None else worker_count
        )
        elapsed_seconds = now - active_phase.started_at
        rate_per_second = (
            completed_units / elapsed_seconds if elapsed_seconds > 0 else None
        )
        eta_seconds = _estimate_eta(
            elapsed_seconds,
            resolved_fraction,
            estimate_eta,
        )
        event = PhaseProgress(
            phase=active_phase.phase,
            state=ProgressState.RUNNING,
            completed_units=completed_units,
            total_units=resolved_total,
            unit=active_phase.unit,
            fraction_complete=resolved_fraction,
            is_estimate=resolved_estimate,
            elapsed_seconds=elapsed_seconds,
            rate_per_second=rate_per_second,
            eta_seconds=eta_seconds,
            worker_count=resolved_workers,
        )
        active_phase.total_units = resolved_total
        active_phase.is_estimate = resolved_estimate
        active_phase.worker_count = resolved_workers
        active_phase.last_snapshot_at = now
        self._reporter(event)


def _estimate_eta(
    elapsed_seconds: float,
    fraction_complete: float | None,
    estimate_eta: bool,
) -> float | None:
    """Calculate remaining duration only from an explicitly trustworthy fraction."""
    if (
        not estimate_eta
        or elapsed_seconds <= 0
        or fraction_complete is None
        or not 0 < fraction_complete < 1
    ):
        return None
    return elapsed_seconds * (1.0 - fraction_complete) / fraction_complete
