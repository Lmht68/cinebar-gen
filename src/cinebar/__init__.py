"""Generate lossless complete-film color barcodes."""

from .crop import CropMode, CropRectangle, CropSelection
from .directory import (
    BarcodeGenerator,
    DirectoryProcessor,
    GenerationSettings,
    OutputLayout,
)
from .errors import CinebarError, FailureCategory
from .models import ProcessingResult
from .processor import SingleFileProcessor
from .progress import (
    PhaseProgress,
    ProcessingCompleted,
    ProgressEvent,
    ProgressPhase,
    ProgressReporter,
    ProgressState,
    ProgressUnit,
)

__all__ = [
    "BarcodeGenerator",
    "CinebarError",
    "CropMode",
    "CropRectangle",
    "CropSelection",
    "DirectoryProcessor",
    "FailureCategory",
    "GenerationSettings",
    "OutputLayout",
    "PhaseProgress",
    "ProcessingCompleted",
    "ProcessingResult",
    "ProgressEvent",
    "ProgressPhase",
    "ProgressReporter",
    "ProgressState",
    "ProgressUnit",
    "SingleFileProcessor",
]
