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

from .console import ConsoleProgressRenderer, DirectoryConsolePresenter
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
    *,
    overwrite_help: str | None = None,
) -> None:
    """Add generation options shared by single-file and directory commands.

    Args:
        parser: Parser receiving the shared generation options.
        height_help: Command-specific height option description.
        overwrite_help: Optional directory-specific overwrite description.
    """
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
        help=overwrite_help or "Replace an existing final PNG after complete encoding.",
    )


def _build_single_file_parser() -> _CinebarArgumentParser:
    """Build the compatibility-sensitive legacy single-file parser."""
    parser = _CinebarArgumentParser(
        description=(
            "Generate a complete lossless media barcode.\n\n"
            "For recursive directory workflows, run 'cinebar batch --help' or "
            "'cinebar watch --help'."
        )
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
        epilog = """Examples:
  cinebar batch ROOT --output-beside-input
  cinebar batch ROOT --output-dir DEST
  cinebar batch ROOT --output-dir DEST --output-layout flat

Readiness requires two unchanged size and nanosecond modification-time observations.
Existing mapped outputs are skipped without source freshness checks.
Collisions list conflicting sources and remain errors with --overwrite.
Each source version receives one attempt per invocation.
Ctrl+C abandons queued work, cleans active unpublished output, and exits 130.
Exit codes: 0 complete or skipped, 1 fatal, 2 candidate failure, 130 Ctrl+C."""
    else:
        description = (
            "Watch recursively scans in the foreground until interrupted. Outputs "
            "default to beside-input, or mirrored paths under --output-dir."
        )
        epilog = """Example:
  cinebar watch ROOT --output-dir DEST --scan-interval 60

Output placement supports beside-input, mirrored, and flat layouts.
Settling requires two unchanged observations; --scan-interval schedules later scans.
One source version receives one attempt per invocation.
Temporary root outages retry without clearing known state.
Ctrl+C abandons queued work and cleans active unpublished output.
Exit codes: 1 fatal, 130 Ctrl+C."""
    parser = _CinebarArgumentParser(
        prog=f"cinebar {command}",
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        overwrite_help=(
            "Replace an existing final PNG after complete encoding. "
            "Never resolves a source collision."
        ),
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
    arguments = list(sys.argv[1:] if argv is None else argv)
    directory_command = _directory_command(arguments)
    renderer: ConsoleProgressRenderer | DirectoryConsolePresenter | None = None
    directory_presenter: DirectoryConsolePresenter | None = None
    if directory_command is not None:
        directory_presenter = DirectoryConsolePresenter()
        renderer = directory_presenter
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(message)s",
            handlers=[directory_presenter],
        )
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        if directory_command is not None:
            assert directory_presenter is not None
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
            directory_processor = DirectoryProcessor()
            if directory_command == "batch":
                batch_result = directory_processor.process_batch(
                    root=directory_arguments.root,
                    settings=settings,
                    progress_reporter=directory_presenter.report_generator_progress,
                    directory_progress_reporter=(
                        directory_presenter.report_directory_progress
                    ),
                )
                directory_presenter.finish()
                return 2 if batch_result.failures else 0
            for _ in directory_processor.process_watch(
                root=directory_arguments.root,
                settings=settings,
                progress_reporter=directory_presenter.report_generator_progress,
                directory_progress_reporter=directory_presenter.report_directory_progress,
            ):
                pass
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
        if isinstance(renderer, DirectoryConsolePresenter):
            renderer.finish()
        elif renderer is not None:
            renderer.finish_line()
        LOGGER.error("[%s] %s", error.category.value, error)
        return 130 if error.category is FailureCategory.INTERRUPTED else 1
    except KeyboardInterrupt:
        if isinstance(renderer, DirectoryConsolePresenter):
            renderer.finish()
        elif renderer is not None:
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
