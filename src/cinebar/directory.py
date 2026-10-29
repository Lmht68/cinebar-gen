"""Process stable media files recursively into beside-input barcodes."""

import os
import stat
from dataclasses import dataclass
from os.path import abspath, normcase
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import Protocol

from .crop import CropMode, CropSelection
from .errors import CinebarError, FailureCategory
from .models import ProcessingResult
from .processor import SingleFileProcessor
from .progress import ProgressReporter

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


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Describe settings shared by every media file in one batch.

    Args:
        height: Requested barcode height in pixels.
        workers: Process count, or ``None`` for the processor default.
        crop: Automatic, disabled, or fixed picture-region crop selection.
        overwrite: Whether existing final PNGs may be replaced.
    """

    height: int = 100
    workers: int | None = None
    crop: CropSelection = CropMode.AUTO
    overwrite: bool = False


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


class DirectoryProcessor:
    """Generate beside-input barcodes for stable media in one directory tree."""

    def __init__(self, barcode_generator: BarcodeGenerator | None = None) -> None:
        """Create a directory processor with an injectable barcode generator.

        Args:
            barcode_generator: Generator used for each discovered media file.
        """
        self._barcode_generator = barcode_generator or SingleFileProcessor()

    def process_batch(
        self,
        root: Path,
        settings: GenerationSettings = GenerationSettings(),  # noqa: B008
        progress_reporter: ProgressReporter | None = None,
    ) -> tuple[ProcessingResult, ...]:
        """Generate one beside-input barcode for every supported media file.

        Args:
            root: Directory tree containing stable source media files.
            settings: Shared generation settings for every source media file.
            progress_reporter: Optional recipient for per-file progress facts.

        Returns:
            Completed barcode results ordered by normalized absolute source path.
        """
        try:
            root_status = root.stat(follow_symlinks=False)
        except FileNotFoundError:
            raise CinebarError(FailureCategory.ROOT_NOT_FOUND) from None
        except OSError:
            raise CinebarError(FailureCategory.ROOT_UNREADABLE) from None
        if _is_reparse_point(root_status) or not S_ISDIR(root_status.st_mode):
            raise CinebarError(FailureCategory.ROOT_UNREADABLE)

        sources = _discover_sources(root)
        output_sources: dict[str, Path] = {}
        plans: list[tuple[Path, Path]] = []
        for source in sources:
            output_path = source.with_name(f"{source.stem}.cinebar.png")
            output_key = normcase(abspath(output_path))
            if output_key in output_sources:
                raise CinebarError(FailureCategory.OUTPUT_COLLISION)
            output_sources[output_key] = source
            plans.append((source, output_path))

        return tuple(
            self._barcode_generator.process(
                media_path=source,
                output_path=output_path,
                height=settings.height,
                workers=settings.workers,
                overwrite=settings.overwrite,
                crop=settings.crop,
                progress_reporter=progress_reporter,
            )
            for source, output_path in plans
        )


def _discover_sources(root: Path) -> list[Path]:
    """Return supported regular files found without following links."""
    directories = [root]
    sources: list[Path] = []

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
                    if S_ISDIR(entry_status.st_mode):
                        directories.append(entry_path)
                    elif S_ISREG(entry_status.st_mode) and (
                        entry_path.suffix.casefold() in _VIDEO_SUFFIXES
                    ):
                        sources.append(entry_path)
        except CinebarError:
            raise
        except OSError:
            raise CinebarError(FailureCategory.ROOT_UNREADABLE) from None

    return sorted(sources, key=lambda path: normcase(abspath(path)))
