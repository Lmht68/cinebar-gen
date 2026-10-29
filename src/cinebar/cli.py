"""Command-line interface for complete media barcode generation."""

import argparse
import logging
import math
import sys
from collections.abc import Sequence
from multiprocessing import freeze_support
from pathlib import Path
from typing import NoReturn

from .console import ConsoleProgressRenderer
from .crop import CropMode, CropRectangle, CropSelection
from .directory import DirectoryProcessor, GenerationSettings, OutputLayout
from .errors import CinebarError, FailureCategory
from .processor import SingleFileProcessor

LOGGER = logging.getLogger(__name__)


class _CinebarArgumentParser(argparse.ArgumentParser):
    """Translate argument-parser failures into Cinebar's public error model."""

    def error(self, message: str) -> NoReturn:
        """Raise an argument failure without printing parser-controlled output.

        Args:
            message: Parser-generated detail that must not reach users.
        """
        raise CinebarError(FailureCategory.INVALID_ARGUMENT)


def _add_generation_options(
    parser: argparse.ArgumentParser,
    height_help: str,
) -> None:
    """Add generation options shared by single-file and batch commands."""
    parser.add_argument(
        "--height",
        type=_positive_integer,
        default=100,
        help=height_help,
    )
    parser.add_argument(
        "--workers",
        type=_positive_worker_count,
        default=None,
        help="Worker processes. Defaults to at most 16; use 1 for sequential decode.",
    )
    parser.add_argument(
        "--crop",
        type=_crop_selection,
        default=CropMode.AUTO,
        metavar="auto|none|LEFT,TOP,RIGHT,BOTTOM",
        help=(
            "Picture region: auto, none, or LEFT,TOP,RIGHT,BOTTOM with inclusive "
            "left/top and exclusive right/bottom."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing final PNG after complete encoding.",
    )


def _build_single_file_parser() -> _CinebarArgumentParser:
    """Build the compatibility-sensitive legacy single-file parser."""
    parser = _CinebarArgumentParser(
        description="Generate a complete lossless media barcode."
    )
    parser.add_argument("media_path", type=Path, help="Local media file to decode.")
    parser.add_argument("--output", required=True, type=Path, help="Final PNG path.")
    _add_generation_options(
        parser,
        height_help="Barcode height in pixels. Defaults to 300.",
    )
    return parser


def _build_batch_parser() -> _CinebarArgumentParser:
    """Build the parser for recursive batch barcode generation."""
    parser = _CinebarArgumentParser(
        description=(
            "Generate barcodes from a stable media directory. Outputs default to "
            "beside-input, or mirrored paths under --output-dir."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory tree containing stable media files to process.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-beside-input",
        action="store_true",
        help="Write <source parent>/<stem>.cinebar.png. Default without --output-dir.",
    )
    output_group.add_argument(
        "--output-dir",
        type=Path,
        metavar="PATH",
        help=(
            "Write mirrored <relative parent>/<stem>.cinebar.png under PATH. "
            "Default layout with this option."
        ),
    )
    parser.add_argument(
        "--output-layout",
        type=OutputLayout,
        choices=(OutputLayout.MIRRORED, OutputLayout.FLAT),
        metavar="{mirrored,flat}",
        help=(
            "With --output-dir, write mirrored <relative parent>/<stem>.cinebar.png "
            "or flat <stem>.png."
        ),
    )
    parser.add_argument(
        "--settling-period",
        type=_positive_duration,
        default=5.0,
        metavar="SECONDS",
        help=(
            "Require size and nanosecond modification time to remain unchanged "
            "across observations. Defaults to 5 seconds; metadata stability is "
            "a heuristic, not proof that copying has finished."
        ),
    )
    parser.add_argument(
        "--readiness-timeout",
        type=_positive_duration,
        default=300.0,
        metavar="SECONDS",
        help="Limit readiness observations. Defaults to 300 seconds.",
    )
    _add_generation_options(
        parser,
        height_help="Barcode height in pixels. Defaults to 100.",
    )
    return parser


def _is_legacy_output_token(argument: str) -> bool:
    """Return whether one token is an accepted legacy output option spelling."""
    option = argument.partition("=")[0]
    return option.startswith("--ou") and "--output".startswith(option)


def _is_batch_command(arguments: Sequence[str]) -> bool:
    """Return whether argv unambiguously selects the batch command."""
    return (
        bool(arguments)
        and arguments[0] == "batch"
        and not any(_is_legacy_output_token(argument) for argument in arguments[1:])
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Generate barcodes from one media file or a stable directory tree.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Zero after publication, two after batch readiness failures, one after a
        fatal failure, or 130 after interruption.
    """
    freeze_support()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = list(sys.argv[1:] if argv is None else argv)
    renderer: ConsoleProgressRenderer | None = None
    try:
        if _is_batch_command(arguments):
            batch_arguments = _build_batch_parser().parse_args(arguments[1:])
            if (
                batch_arguments.output_layout is not None
                and batch_arguments.output_dir is None
            ):
                raise CinebarError(FailureCategory.INVALID_ARGUMENT)
            output_layout = (
                OutputLayout.BESIDE_INPUT
                if batch_arguments.output_beside_input
                else batch_arguments.output_layout
            )
            renderer = ConsoleProgressRenderer()
            batch_result = DirectoryProcessor().process_batch(
                root=batch_arguments.root,
                settings=GenerationSettings(
                    height=batch_arguments.height,
                    workers=batch_arguments.workers,
                    crop=batch_arguments.crop,
                    overwrite=batch_arguments.overwrite,
                    output_dir=batch_arguments.output_dir,
                    output_layout=output_layout,
                    settling_period_seconds=batch_arguments.settling_period,
                    readiness_timeout_seconds=batch_arguments.readiness_timeout,
                ),
                progress_reporter=renderer,
            )
            renderer.finish_line()
            if batch_result.failures:
                for failure in batch_result.failures:
                    failure_error = CinebarError(
                        failure.category,
                        f"Source: {failure.source_path}",
                    )
                    LOGGER.error(
                        "[%s] %s",
                        failure_error.category.value,
                        failure_error,
                    )
                return 2
        else:
            single_file_arguments = _build_single_file_parser().parse_args(arguments)
            renderer = ConsoleProgressRenderer()
            SingleFileProcessor().process(
                media_path=single_file_arguments.media_path,
                output_path=single_file_arguments.output,
                height=single_file_arguments.height,
                workers=single_file_arguments.workers,
                overwrite=single_file_arguments.overwrite,
                crop=single_file_arguments.crop,
                progress_reporter=renderer,
            )
    except CinebarError as error:
        if renderer is not None:
            renderer.finish_line()
        LOGGER.error("[%s] %s", error.category.value, error)
        return 130 if error.category is FailureCategory.INTERRUPTED else 1
    except KeyboardInterrupt:
        if renderer is not None:
            renderer.finish_line()
        interruption = CinebarError(FailureCategory.INTERRUPTED)
        LOGGER.error("[%s] %s", interruption.category.value, interruption)
        return 130
    return 0


def _positive_integer(value: str) -> int:
    """Parse one strictly positive command-line integer.

    Args:
        value: Raw command-line argument.

    Returns:
        Parsed positive integer.

    Raises:
        CinebarError: If the value is not a positive integer.
    """
    try:
        parsed_value = int(value)
    except ValueError:
        raise CinebarError(FailureCategory.INVALID_HEIGHT) from None
    if parsed_value <= 0:
        raise CinebarError(FailureCategory.INVALID_HEIGHT)
    return parsed_value


def _positive_worker_count(value: str) -> int:
    """Parse one strictly positive worker count.

    Args:
        value: Raw command-line argument.

    Returns:
        Parsed positive worker count.

    Raises:
        CinebarError: If the value is not a positive integer.
    """
    try:
        parsed_value = int(value)
    except ValueError:
        raise CinebarError(FailureCategory.INVALID_ARGUMENT) from None
    if parsed_value <= 0:
        raise CinebarError(FailureCategory.INVALID_ARGUMENT)
    return parsed_value


def _positive_duration(value: str) -> float:
    """Parse one finite strictly positive duration in seconds.

    Args:
        value: Raw command-line duration argument.

    Returns:
        Parsed finite positive seconds.

    Raises:
        CinebarError: If the value is not finite and strictly positive.
    """
    try:
        parsed_value = float(value)
    except ValueError:
        raise CinebarError(FailureCategory.INVALID_ARGUMENT) from None
    if not math.isfinite(parsed_value) or parsed_value <= 0:
        raise CinebarError(FailureCategory.INVALID_ARGUMENT)
    return parsed_value


def _crop_selection(value: str) -> CropSelection:
    """Parse one automatic, disabled, or fixed crop selection.

    Args:
        value: Raw mode token or four comma-separated decimal coordinates.

    Returns:
        One automatic or disabled mode, or a validated explicit crop rectangle.

    Raises:
        CinebarError: If the selection is malformed or has invalid bounds.
    """
    if value == "auto":
        return CropMode.AUTO
    if value == "none":
        return CropMode.NONE

    coordinate_texts = value.split(",")
    if len(coordinate_texts) != 4:
        raise CinebarError(FailureCategory.INVALID_CROP)
    try:
        left, top, right, bottom = (
            int(coordinate_text, 10) for coordinate_text in coordinate_texts
        )
    except ValueError:
        raise CinebarError(FailureCategory.INVALID_CROP) from None
    return CropRectangle(left=left, top=top, right=right, bottom=bottom)
