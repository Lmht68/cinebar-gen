"""Process stable media files recursively into configured barcode layouts."""

import math
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from os.path import abspath, normcase
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from time import monotonic, sleep
from typing import Protocol

from .crop import CropMode, CropSelection
from .errors import CinebarError, FailureCategory
from .models import BatchFailure, BatchProcessingResult, ProcessingResult
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


def _path_sort_key(path: Path) -> tuple[str, str]:
    """Return a stable normalized and absolute path sort key."""
    absolute_path = abspath(path)
    return (normcase(absolute_path), absolute_path)


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
    """

    height: int = 100
    workers: int | None = None
    crop: CropSelection = CropMode.AUTO
    overwrite: bool = False
    output_dir: Path | None = None
    output_layout: OutputLayout | None = None
    settling_period_seconds: float = 5.0
    readiness_timeout_seconds: float = 300.0

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


@dataclass(slots=True)
class _PendingSource:
    """Track one source version until a second matching observation occurs."""

    source_version: _SourceVersion
    output_path: Path
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


class DirectoryProcessor:
    """Generate configured barcodes for stable media in one directory tree."""

    def __init__(
        self,
        barcode_generator: BarcodeGenerator | None = None,
        *,
        clock: Callable[[], float] = monotonic,
        wait: Callable[[float], None] = sleep,
    ) -> None:
        """Create a directory processor with injectable generation and timing.

        Args:
            barcode_generator: Generator used for each discovered media file.
            clock: Monotonic source for readiness observation times.
            wait: Delay function used until the next readiness observation.
        """
        self._barcode_generator = barcode_generator or SingleFileProcessor()
        self._clock = clock
        self._wait = wait

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
        try:
            root_status = root.stat(follow_symlinks=False)
        except FileNotFoundError:
            raise CinebarError(FailureCategory.ROOT_NOT_FOUND) from None
        except OSError:
            raise CinebarError(FailureCategory.ROOT_UNREADABLE) from None
        if _is_reparse_point(root_status) or not S_ISDIR(root_status.st_mode):
            raise CinebarError(FailureCategory.ROOT_UNREADABLE)

        excluded_directory = (
            settings.output_dir
            if settings.output_dir is not None
            and _is_within_directory(settings.output_dir, root, strict=True)
            else None
        )
        source_versions = _discover_sources(root, excluded_directory)
        output_sources: dict[str, tuple[Path, list[Path]]] = {}
        plans: list[tuple[_SourceVersion, Path]] = []
        for source_version in source_versions:
            source = source_version.source_path
            if settings.output_dir is None:
                output_path = source.with_name(f"{source.stem}.cinebar.png")
            elif settings.output_layout is OutputLayout.FLAT:
                output_path = settings.output_dir / f"{source.stem}.png"
            else:
                output_path = (
                    settings.output_dir
                    / source.parent.relative_to(root)
                    / f"{source.stem}.cinebar.png"
                )
            output_key = normcase(abspath(output_path))
            source_paths = output_sources.setdefault(
                output_key,
                (Path(abspath(output_path)), []),
            )[1]
            source_paths.append(Path(abspath(source)))
            plans.append((source_version, output_path))

        collisions = [
            (destination, source_paths)
            for destination, source_paths in output_sources.values()
            if len(source_paths) >= 2
        ]
        if collisions:
            safe_detail = "\n\n".join(
                "\n".join(
                    [
                        f"Destination: {destination}",
                        "Conflicting inputs:",
                        *(
                            f"- {source}"
                            for source in sorted(source_paths, key=_path_sort_key)
                        ),
                    ]
                )
                for destination, source_paths in sorted(
                    collisions,
                    key=lambda group: _path_sort_key(group[0]),
                )
            )
            raise CinebarError(FailureCategory.OUTPUT_COLLISION, safe_detail)

        destination_parents: dict[str, Path] = {}
        for _, output_path in plans:
            destination_parents.setdefault(
                normcase(abspath(output_path.parent)),
                output_path.parent,
            )
        try:
            for output_parent in sorted(
                destination_parents.values(),
                key=_path_sort_key,
            ):
                output_parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise CinebarError(FailureCategory.OUTPUT_WRITE_FAILED) from None

        if not plans:
            return BatchProcessingResult(completed=(), failures=())

        readiness_started_at = self._clock()
        readiness_deadline = readiness_started_at + settings.readiness_timeout_seconds
        pending_sources = [
            _PendingSource(
                source_version=source_version,
                output_path=output_path,
                baseline_observed_at=readiness_started_at,
            )
            for source_version, output_path in plans
        ]
        completed: list[ProcessingResult] = []
        failures: list[BatchFailure] = []

        while pending_sources:
            current_time = self._clock()
            if current_time > readiness_deadline:
                failures.extend(
                    BatchFailure(
                        pending_source.source_version.source_path,
                        FailureCategory.READINESS_TIMEOUT,
                    )
                    for pending_source in sorted(
                        pending_sources,
                        key=lambda source: source.source_version.normalized_path,
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
                        pending_source.source_version.source_path,
                        FailureCategory.READINESS_TIMEOUT,
                    )
                    for pending_source in sorted(
                        pending_sources,
                        key=lambda source: source.source_version.normalized_path,
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
                key=lambda source: source.source_version.normalized_path,
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
                source_path = pending_source.source_version.source_path
                try:
                    current_status = source_path.stat(follow_symlinks=False)
                except FileNotFoundError:
                    failures.append(
                        BatchFailure(source_path, FailureCategory.INPUT_NOT_FOUND)
                    )
                    continue
                except OSError:
                    failures.append(
                        BatchFailure(source_path, FailureCategory.INPUT_UNREADABLE)
                    )
                    continue

                if _is_reparse_point(current_status) or not S_ISREG(
                    current_status.st_mode
                ):
                    failures.append(
                        BatchFailure(source_path, FailureCategory.INPUT_UNREADABLE)
                    )
                    continue

                current_version = _SourceVersion(
                    source_path=source_path,
                    normalized_path=pending_source.source_version.normalized_path,
                    byte_size=current_status.st_size,
                    modification_time_ns=current_status.st_mtime_ns,
                )
                if current_version == pending_source.source_version:
                    ready_sources.append(pending_source)
                    continue

                pending_source.source_version = current_version
                pending_source.baseline_observed_at = current_time
                next_pending_sources.append(pending_source)

            pending_sources = next_pending_sources
            for pending_source in ready_sources:
                completed.append(
                    self._barcode_generator.process(
                        media_path=pending_source.source_version.source_path,
                        output_path=pending_source.output_path,
                        height=settings.height,
                        workers=settings.workers,
                        overwrite=settings.overwrite,
                        crop=settings.crop,
                        progress_reporter=progress_reporter,
                    )
                )

        return BatchProcessingResult(
            completed=tuple(completed),
            failures=tuple(failures),
        )


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
