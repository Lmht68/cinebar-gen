"""Public orchestration for one complete barcode."""

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from stat import S_ISREG
from tempfile import NamedTemporaryFile
from time import monotonic
from typing import BinaryIO, Protocol, cast

import png

from .crop import CropMode, CropRectangle, CropSelection
from .errors import CinebarError, FailureCategory
from .extract import (
    ExtractionProgressCallback,
    FrameExtractionProgress,
    extract_frame_colors,
    sample_automatic_crop,
)
from .models import ProcessingResult
from .png import write_barcode_png
from .progress import (
    ProcessingCompleted,
    ProgressPhase,
    ProgressReporter,
    ProgressTracker,
    ProgressUnit,
    UnitProgressCallback,
)


class FrameColorExtractor(Protocol):
    """Extract complete frame colors and report parent-process facts."""

    def __call__(
        self,
        video_path: Path,
        workers: int | None,
        crop: CropRectangle | None,
        /,
        *,
        progress_callback: ExtractionProgressCallback | None,
    ) -> bytes:
        """Return packed frame colors for one complete media file."""
        ...


class AutomaticCropSelector(Protocol):
    """Sample one media file and report successful probe counts."""

    def __call__(
        self,
        video_path: Path,
        /,
        *,
        progress_callback: UnitProgressCallback | None,
    ) -> CropRectangle | None:
        """Return an accepted automatic crop or no crop decision."""
        ...


class SingleFileProcessor:
    """Process every decoded frame from one media file into a barcode PNG."""

    def __init__(
        self,
        frame_color_extractor: FrameColorExtractor = extract_frame_colors,
        automatic_crop_selector: AutomaticCropSelector = sample_automatic_crop,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Create a processor with injectable extraction and crop-selection seams.

        Args:
            frame_color_extractor: Extractor returning three RGB bytes per frame.
            automatic_crop_selector: Selector returning one sampled crop or no crop.
            clock: Monotonic source used only when callers request progress facts.
        """
        self._frame_color_extractor = frame_color_extractor
        self._automatic_crop_selector = automatic_crop_selector
        self._clock = clock

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
        """Decode one media file and atomically publish its complete barcode.

        Args:
            media_path: Local media file to decode.
            output_path: Final PNG path to publish.
            height: Requested barcode height in pixels.
            workers: Process count, or ``None`` for a safe automatic count.
            overwrite: Whether an existing final PNG may be replaced.
            crop: Automatic, disabled, or fixed picture-region crop selection.
            progress_reporter: Optional recipient for renderer-independent facts.

        Returns:
            Dimensions and path of the completed barcode PNG.

        Raises:
            CinebarError: If validation, decoding, or publication fails.
        """
        temporary_output_path: Path | None = None
        processing_started_at: float | None = None
        decoded_frame_count = 0
        actual_worker_count = 1
        average_frames_per_second: float | None = None
        resolved_crop: CropRectangle | None = None
        tracker = (
            ProgressTracker(progress_reporter, clock=self._clock)
            if progress_reporter is not None
            else None
        )
        try:
            if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
                raise CinebarError(FailureCategory.INVALID_HEIGHT)
            if workers is not None and (
                isinstance(workers, bool)
                or not isinstance(workers, int)
                or workers <= 0
            ):
                raise CinebarError(FailureCategory.INVALID_ARGUMENT)
            if not isinstance(crop, (CropMode, CropRectangle)):
                raise CinebarError(FailureCategory.INVALID_CROP)

            if not overwrite:
                _ensure_output_path_is_available(output_path)
            _validate_input_media(media_path)
            if tracker is not None:
                processing_started_at = self._clock()

            if crop is CropMode.AUTO:
                resolved_crop = self._select_automatic_crop(media_path, tracker)
            elif crop is CropMode.NONE:
                resolved_crop = None
            else:
                resolved_crop = crop

            extraction_progress_callback: ExtractionProgressCallback | None = None
            decode_phase_started = False
            if tracker is not None:

                def report_extraction(fact: FrameExtractionProgress) -> None:
                    nonlocal actual_worker_count, decode_phase_started
                    actual_worker_count = fact.worker_count
                    if not decode_phase_started:
                        tracker.start(
                            ProgressPhase.FRAME_DECODING,
                            ProgressUnit.FRAMES,
                            total_units=fact.expected_frame_count,
                            is_estimate=fact.expected_frame_count is not None,
                            worker_count=fact.worker_count,
                        )
                        decode_phase_started = True
                        return

                    fraction_complete, is_estimate, estimate_eta = (
                        _decode_progress_estimate(fact)
                    )
                    tracker.update(
                        fact.decoded_frame_count,
                        total_units=fact.expected_frame_count,
                        fraction_complete=fraction_complete,
                        is_estimate=is_estimate,
                        worker_count=fact.worker_count,
                        estimate_eta=estimate_eta,
                    )

                extraction_progress_callback = report_extraction

            compact_colors = self._frame_color_extractor(
                media_path,
                workers,
                resolved_crop,
                progress_callback=extraction_progress_callback,
            )
            if not compact_colors:
                raise CinebarError(FailureCategory.ZERO_FRAMES)
            if len(compact_colors) % 3 != 0:
                raise CinebarError(FailureCategory.CORRUPT_FRAME)

            decoded_frame_count = len(compact_colors) // 3
            if tracker is not None:
                if not decode_phase_started:
                    tracker.start(
                        ProgressPhase.FRAME_DECODING,
                        ProgressUnit.FRAMES,
                        worker_count=actual_worker_count,
                    )
                completed_decode_phase = tracker.complete(
                    decoded_frame_count,
                    total_units=decoded_frame_count,
                    is_estimate=False,
                    worker_count=actual_worker_count,
                )
                average_frames_per_second = completed_decode_phase.rate_per_second

            try:
                with NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{output_path.name}.",
                    suffix=".tmp",
                    dir=output_path.parent,
                    delete=False,
                ) as temporary_output:
                    temporary_output_path = Path(temporary_output.name)
                    png_progress_callback: UnitProgressCallback | None = None
                    if tracker is not None:
                        tracker.start(
                            ProgressPhase.PNG_ENCODING,
                            ProgressUnit.ROWS,
                            total_units=height,
                        )

                        def report_png_rows(encoded_row_count: int) -> None:
                            tracker.update(encoded_row_count)

                        png_progress_callback = report_png_rows
                    write_barcode_png(
                        cast(BinaryIO, temporary_output.file),
                        BytesIO(compact_colors),
                        decoded_frame_count,
                        height,
                        progress_callback=png_progress_callback,
                    )
                    if tracker is not None:
                        tracker.complete(height)
                        tracker.start(
                            ProgressPhase.OUTPUT_PUBLICATION,
                            ProgressUnit.FILES,
                            total_units=1,
                        )
                    temporary_output.file.flush()

                if temporary_output_path is None:
                    raise CinebarError(FailureCategory.OUTPUT_WRITE_FAILED)
                if not overwrite:
                    _ensure_output_path_is_available(output_path)
                temporary_output_path.replace(output_path)
                if tracker is not None:
                    tracker.complete(1)
            except (OSError, png.Error, ValueError):
                raise CinebarError(FailureCategory.OUTPUT_WRITE_FAILED) from None
        except KeyboardInterrupt:
            raise CinebarError(FailureCategory.INTERRUPTED) from None
        finally:
            if temporary_output_path is not None:
                try:
                    temporary_output_path.unlink(missing_ok=True)
                except OSError:
                    raise CinebarError(FailureCategory.OUTPUT_WRITE_FAILED) from None

        result = ProcessingResult(
            output_path=output_path,
            decoded_frame_count=decoded_frame_count,
            output_width=decoded_frame_count,
            output_height=height,
        )
        if progress_reporter is not None and processing_started_at is not None:
            progress_reporter(
                ProcessingCompleted(
                    result=result,
                    elapsed_seconds=self._clock() - processing_started_at,
                    average_frames_per_second=average_frames_per_second,
                    worker_count=actual_worker_count,
                    accepted_crop=resolved_crop,
                )
            )
        return result

    def _select_automatic_crop(
        self,
        media_path: Path,
        tracker: ProgressTracker | None,
    ) -> CropRectangle | None:
        """Run bounded automatic sampling and report its truthful probe count."""
        if tracker is None:
            return self._automatic_crop_selector(media_path, progress_callback=None)

        sampled_unit_count = 0
        tracker.start(
            ProgressPhase.CROP_ANALYSIS,
            ProgressUnit.SAMPLES,
            total_units=12,
        )

        def report_sampled_units(completed_units: int) -> None:
            nonlocal sampled_unit_count
            sampled_unit_count = completed_units
            tracker.update(completed_units)

        selected_crop = self._automatic_crop_selector(
            media_path,
            progress_callback=report_sampled_units,
        )
        tracker.complete(sampled_unit_count)
        return selected_crop


def _decode_progress_estimate(
    fact: FrameExtractionProgress,
) -> tuple[float | None, bool, bool]:
    """Translate extraction facts into an optional display fraction and ETA basis."""
    if fact.expected_frame_count is not None:
        if fact.decoded_frame_count > fact.expected_frame_count:
            return None, True, False
        return fact.decoded_frame_count / fact.expected_frame_count, True, False
    if (
        fact.presentation_seconds is None
        or fact.duration_seconds is None
        or fact.duration_seconds <= 0
        or fact.presentation_seconds <= 0
        or fact.presentation_seconds > fact.duration_seconds
    ):
        return None, False, False
    return fact.presentation_seconds / fact.duration_seconds, True, True


def _ensure_output_path_is_available(output_path: Path) -> None:
    """Reject an existing final-path entry without resolving symlinks."""
    try:
        output_path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        raise CinebarError(FailureCategory.OUTPUT_WRITE_FAILED) from None
    raise CinebarError(FailureCategory.OUTPUT_EXISTS)


def _validate_input_media(media_path: Path) -> None:
    """Require a readable regular source file before decoding."""
    try:
        media_stat = media_path.stat()
    except FileNotFoundError:
        raise CinebarError(FailureCategory.INPUT_NOT_FOUND) from None
    except OSError:
        raise CinebarError(FailureCategory.INPUT_UNREADABLE) from None

    if not S_ISREG(media_stat.st_mode):
        raise CinebarError(FailureCategory.INPUT_UNREADABLE)

    try:
        with media_path.open("rb"):
            pass
    except FileNotFoundError:
        raise CinebarError(FailureCategory.INPUT_NOT_FOUND) from None
    except OSError:
        raise CinebarError(FailureCategory.INPUT_UNREADABLE) from None
