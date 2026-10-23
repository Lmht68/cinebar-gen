"""Render application-owned Cinebar progress facts to a terminal stream."""

import sys
from typing import TextIO

from .crop import CropRectangle
from .progress import (
    PhaseProgress,
    ProcessingCompleted,
    ProgressEvent,
    ProgressPhase,
    ProgressState,
)

_PHASE_LABELS: dict[ProgressPhase, str] = {
    ProgressPhase.CROP_ANALYSIS: "Analyzing crop",
    ProgressPhase.FRAME_DECODING: "Decoding frames",
    ProgressPhase.PNG_ENCODING: "Encoding PNG",
    ProgressPhase.OUTPUT_PUBLICATION: "Publishing output",
}


class ConsoleProgressRenderer:
    """Render progress events to stderr without receiving source-media paths."""

    _BAR_WIDTH = 20
    _RUNNING_UPDATE_INTERVAL_SECONDS = 30.0

    def __init__(self, stream: TextIO = sys.stderr) -> None:
        """Create a renderer for one interactive or redirected text stream.

        Args:
            stream: stderr-compatible stream that receives status text.
        """
        self._stream = stream
        self._is_interactive = stream.isatty()
        self._last_interactive_line_length = 0
        self._has_interactive_line = False
        self._last_noninteractive_update: dict[ProgressPhase, float] = {}

    def __call__(self, event: ProgressEvent) -> None:
        """Render one progress or completed-processing event.

        Args:
            event: Application-owned progress fact to render.
        """
        if isinstance(event, ProcessingCompleted):
            self._render_completed(event)
            return
        if self._is_interactive:
            self._write_interactive(self._phase_line(event))
            return
        self._render_noninteractive_phase(event)

    def finish_line(self) -> None:
        """End an in-place interactive redraw before another writer takes stderr."""
        if not self._has_interactive_line:
            return
        self._stream.write("\n")
        self._stream.flush()
        self._has_interactive_line = False
        self._last_interactive_line_length = 0

    def _render_completed(self, event: ProcessingCompleted) -> None:
        """Render the success summary after any active progress redraw."""
        if self._is_interactive:
            self.finish_line()
        summary_parts = [
            (
                f"Complete: {event.result.decoded_frame_count} frames, "
                f"{event.result.output_width}x{event.result.output_height}, "
                f"crop {_format_crop(event.accepted_crop)}, "
                f"{_format_workers(event.worker_count)}, "
                f"elapsed {_format_duration(event.elapsed_seconds)}"
            )
        ]
        if event.average_frames_per_second is not None:
            summary_parts.append(f"{event.average_frames_per_second:.1f} frames/s")
        self._write_line(" ".join(summary_parts))

    def _render_noninteractive_phase(self, event: PhaseProgress) -> None:
        """Write phase boundaries and rate-limited running updates to a log stream."""
        if event.state is ProgressState.RUNNING:
            last_update = self._last_noninteractive_update.get(event.phase)
            if (
                last_update is not None
                and event.elapsed_seconds - last_update
                < self._RUNNING_UPDATE_INTERVAL_SECONDS
            ):
                return
        self._last_noninteractive_update[event.phase] = event.elapsed_seconds
        self._write_line(self._phase_line(event))

    def _phase_line(self, event: PhaseProgress) -> str:
        """Format one complete human-readable phase status line."""
        parts = [_PHASE_LABELS[event.phase], self._bar(event), _format_work(event)]
        if event.fraction_complete is not None:
            percentage_prefix = "~" if event.is_estimate else ""
            parts.append(f"{percentage_prefix}{event.fraction_complete:.0%}")
        parts.append(f"elapsed {_format_duration(event.elapsed_seconds)}")
        if event.rate_per_second is not None:
            parts.append(f"{event.rate_per_second:.1f} {event.unit.value}/s")
        if event.worker_count is not None:
            parts.append(_format_workers(event.worker_count))
        if event.eta_seconds is not None:
            parts.append(f"ETA {_format_duration(event.eta_seconds)}")
        return " ".join(parts)

    def _bar(self, event: PhaseProgress) -> str:
        """Render one determinate or bouncing ASCII bar with exactly 20 cells."""
        if event.fraction_complete is not None:
            completed_cells = round(event.fraction_complete * self._BAR_WIDTH)
            completed_cells = min(self._BAR_WIDTH, max(0, completed_cells))
            return (
                "["
                + "=" * completed_cells
                + "-" * (self._BAR_WIDTH - completed_cells)
                + "]"
            )
        bounce_span = (self._BAR_WIDTH - 1) * 2
        bounce_position = event.completed_units % bounce_span
        if bounce_position >= self._BAR_WIDTH:
            bounce_position = bounce_span - bounce_position
        return (
            "["
            + "-" * bounce_position
            + "#"
            + "-" * (self._BAR_WIDTH - bounce_position - 1)
            + "]"
        )

    def _write_interactive(self, line: str) -> None:
        """Redraw one padded status line in place."""
        padding = " " * max(0, self._last_interactive_line_length - len(line))
        self._stream.write(f"\r{line}{padding}")
        self._stream.flush()
        self._last_interactive_line_length = len(line)
        self._has_interactive_line = True

    def _write_line(self, line: str) -> None:
        """Write and flush one complete non-interactive status line."""
        self._stream.write(f"{line}\n")
        self._stream.flush()


def _format_work(event: PhaseProgress) -> str:
    """Format completed and known work without fabricating a total."""
    if event.total_units is None:
        return f"{event.completed_units} {event.unit.value}"
    total_prefix = "~" if event.is_estimate else ""
    return (
        f"{event.completed_units}/{total_prefix}{event.total_units} {event.unit.value}"
    )


def _format_workers(worker_count: int) -> str:
    """Format one actual decoder worker count with correct plurality."""
    worker_label = "worker" if worker_count == 1 else "workers"
    return f"{worker_count} {worker_label}"


def _format_duration(seconds: float) -> str:
    """Format non-negative seconds as a fixed-width elapsed-time display."""
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3_600)
    minutes, seconds_component = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_component:02d}"


def _format_crop(crop: CropRectangle | None) -> str:
    """Format an accepted crop decision without exposing source-media identity."""
    if crop is None:
        return "none"
    return f"{crop.left},{crop.top},{crop.right},{crop.bottom}"
