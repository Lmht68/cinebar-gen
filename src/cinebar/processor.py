"""Public orchestration for one complete no-crop barcode."""

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from stat import S_ISREG
from tempfile import NamedTemporaryFile
from typing import BinaryIO, cast

import png

from .errors import CinebarError, FailureCategory
from .extract import extract_frame_colors
from .models import ProcessingResult
from .png import write_barcode_png

FrameColorExtractor = Callable[[Path, int | None], bytes]


class SingleFileProcessor:
    """Process every decoded frame from one media file into a barcode PNG."""

    def __init__(
        self,
        frame_color_extractor: FrameColorExtractor = extract_frame_colors,
    ) -> None:
        """Create a processor with an injectable frame-color extraction seam.

        Args:
            frame_color_extractor: Extractor returning three RGB bytes per frame.
        """
        self._frame_color_extractor = frame_color_extractor

    def process(
        self,
        media_path: Path,
        output_path: Path,
        height: int = 300,
        workers: int | None = None,
        overwrite: bool = False,
    ) -> ProcessingResult:
        """Decode one media file and atomically publish its complete barcode.

        Args:
            media_path: Local media file to decode.
            output_path: Final PNG path to publish.
            height: Requested barcode height in pixels.
            workers: Process count, or ``None`` for a safe automatic count.
            overwrite: Whether an existing final PNG may be replaced.

        Returns:
            Dimensions and path of the completed barcode PNG.

        Raises:
            CinebarError: If validation, decoding, or publication fails.
        """
        temporary_output_path: Path | None = None
        try:
            if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
                raise CinebarError(FailureCategory.INVALID_HEIGHT)
            if workers is not None and (
                isinstance(workers, bool)
                or not isinstance(workers, int)
                or workers <= 0
            ):
                raise CinebarError(FailureCategory.INVALID_ARGUMENT)

            if not overwrite:
                _ensure_output_path_is_available(output_path)
            _validate_input_media(media_path)

            compact_colors = self._frame_color_extractor(media_path, workers)
            if not compact_colors:
                raise CinebarError(FailureCategory.ZERO_FRAMES)
            if len(compact_colors) % 3 != 0:
                raise CinebarError(FailureCategory.CORRUPT_FRAME)

            decoded_frame_count = len(compact_colors) // 3
            try:
                with NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{output_path.name}.",
                    suffix=".tmp",
                    dir=output_path.parent,
                    delete=False,
                ) as temporary_output:
                    temporary_output_path = Path(temporary_output.name)
                    write_barcode_png(
                        cast(BinaryIO, temporary_output.file),
                        BytesIO(compact_colors),
                        decoded_frame_count,
                        height,
                    )
                    temporary_output.file.flush()

                if temporary_output_path is None:
                    raise CinebarError(FailureCategory.OUTPUT_WRITE_FAILED)
                if not overwrite:
                    _ensure_output_path_is_available(output_path)
                temporary_output_path.replace(output_path)
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

        return ProcessingResult(
            output_path=output_path,
            decoded_frame_count=decoded_frame_count,
            output_width=decoded_frame_count,
            output_height=height,
        )


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
