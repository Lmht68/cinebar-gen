"""Public orchestration for one complete no-crop barcode."""

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

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
    ) -> ProcessingResult:
        """Decode one media file and atomically publish its complete barcode.

        Args:
            media_path: Local media file to decode.
            output_path: Final PNG path to publish.
            height: Requested barcode height in pixels.
            workers: Process count, or ``None`` for a safe automatic count.

        Returns:
            Dimensions and path of the completed barcode PNG.

        Raises:
            ValueError: If dimensions or extracted frame colors are invalid.
            OSError: If temporary or output file operations fail.
        """
        if height <= 0:
            raise ValueError("height must be positive.")

        compact_colors = self._frame_color_extractor(media_path, workers)
        if not compact_colors:
            raise ValueError("No video frames were decoded.")
        if len(compact_colors) % 3 != 0:
            raise ValueError("Frame color extractor returned an incomplete RGB color.")

        decoded_frame_count = len(compact_colors) // 3
        temporary_output_path: Path | None = None
        published = False
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
                temporary_output_path,
                BytesIO(compact_colors),
                decoded_frame_count,
                height,
            )
            temporary_output_path.replace(output_path)
            published = True
        finally:
            if (
                not published
                and temporary_output_path is not None
                and temporary_output_path.exists()
            ):
                temporary_output_path.unlink()

        return ProcessingResult(
            output_path=output_path,
            decoded_frame_count=decoded_frame_count,
            output_width=decoded_frame_count,
            output_height=height,
        )
