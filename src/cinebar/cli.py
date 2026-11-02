"""Command-line interface for complete media barcode generation."""

import argparse
import logging
import math
import signal
import sys
from collections.abc import Sequence
from multiprocessing import freeze_support
from pathlib import Path
from typing import Literal, NoReturn

from .console import ConsoleProgressRenderer
from .crop import CropMode, CropRectangle, CropSelection
from .directory import DirectoryProcessor, GenerationSettings, OutputLayout
from .errors import CinebarError, FailureCategory
from .models import BatchFailure, BatchSkip, ProcessingResult
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


def _build_directory_parser(
    command: Literal["batch", "watch"],
) -> _CinebarArgumentParser:
    """Build the parser for one recursive directory command."""
    if command == "batch":
        description = (
            "Generate barcodes from a stable media directory. Outputs default to "
            "beside-input, or mirrored paths under --output-dir."
        )
    else:
        description = (
            "Watch recursively scans in the foreground until interrupted. Outputs "
            "default to beside-input, or mirrored paths under --output-dir."
        )
    parser = _CinebarArgumentParser(description=description)
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
        default=60.0,
        metavar="SECONDS",
        help=(
            "Require size and nanosecond modification time to remain unchanged "
            "across observations. Defaults to 60 seconds; metadata stability is "
            "a heuristic, not proof that copying has finished."
        ),
    )
    if command == "batch":
        parser.add_argument(
            "--readiness-timeout",
            type=_positive_duration,
            default=300.0,
            metavar="SECONDS",
            help="Limit readiness observations. Defaults to 300 seconds.",
        )
    else:
        parser.add_argument(
            "--scan-interval",
            type=_positive_duration,
            default=60.0,
            metavar="SECONDS",
            help="Delay later scans. Defaults to 60 seconds.",
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


def _directory_command(
    arguments: Sequence[str],
) -> Literal["batch", "watch"] | None:
    """Return the unambiguous recursive directory command selected by argv."""
    if not arguments or arguments[0] not in {"batch", "watch"}:
        return None
    if any(_is_legacy_output_token(argument) for argument in arguments[1:]):
        return None
    if arguments[0] == "batch":
        return "batch"
    return "watch"


def _log_directory_outcome(
    outcome: ProcessingResult | BatchSkip | BatchFailure,
) -> bool:
    """Log one non-successful directory outcome and report whether it failed."""
    if isinstance(outcome, ProcessingResult):
        return False
    if isinstance(outcome, BatchSkip):
        LOGGER.info(
            "Skipped existing mapped output without verifying source or "
            "generation settings.\nSource: %s\nDestination: %s",
            outcome.source_path,
            outcome.output_path,
        )
        return False
    detail = f"Source: {outcome.source_path}"
    if outcome.safe_detail is not None:
        detail = f"{detail}\n{outcome.safe_detail}"
    failure_error = CinebarError(outcome.category, detail)
    LOGGER.error(
        "[%s] %s",
        failure_error.category.value,
        failure_error,
    )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Generate barcodes from one media file or a stable directory tree.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Zero after a completed directory batch or interrupted-free single-file
        run, two after any batch candidate failure, one after a fatal failure,
        or 130 after interruption.
    """
    freeze_support()
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = list(sys.argv[1:] if argv is None else argv)
    renderer: ConsoleProgressRenderer | None = None
    try:
        directory_command = _directory_command(arguments)
        if directory_command is not None:
            directory_arguments = _build_directory_parser(directory_command).parse_args(
                arguments[1:]
            )
            if (
                directory_arguments.output_layout is not None
                and directory_arguments.output_dir is None
            ):
                raise CinebarError(FailureCategory.INVALID_ARGUMENT)
            output_layout = (
                OutputLayout.BESIDE_INPUT
                if directory_arguments.output_beside_input
                else directory_arguments.output_layout
            )
            settings = GenerationSettings(
                height=directory_arguments.height,
                workers=directory_arguments.workers,
                crop=directory_arguments.crop,
                overwrite=directory_arguments.overwrite,
                output_dir=directory_arguments.output_dir,
                output_layout=output_layout,
                settling_period_seconds=directory_arguments.settling_period,
                readiness_timeout_seconds=(
                    directory_arguments.readiness_timeout
                    if directory_command == "batch"
                    else 300.0
                ),
                scan_interval_seconds=(
                    directory_arguments.scan_interval
                    if directory_command == "watch"
                    else 60.0
                ),
            )
            renderer = ConsoleProgressRenderer()
            directory_processor = DirectoryProcessor()
            if directory_command == "batch":
                batch_result = directory_processor.process_batch(
                    root=directory_arguments.root,
                    settings=settings,
                    progress_reporter=renderer,
                )
                renderer.finish_line()
                for skipped in batch_result.skipped:
                    _log_directory_outcome(skipped)
                batch_failed = False
                for failure in batch_result.failures:
                    batch_failed = _log_directory_outcome(failure) or batch_failed
                if batch_failed:
                    return 2
            else:
                for outcome in directory_processor.process_watch(
                    root=directory_arguments.root,
                    settings=settings,
                    progress_reporter=renderer,
                ):
                    if not isinstance(outcome, ProcessingResult):
                        renderer.finish_line()
                        _log_directory_outcome(outcome)
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
