"""Render application-owned Cinebar progress facts to a terminal stream."""

import logging
import sys
from os import get_terminal_size
from typing import TextIO

from .crop import CropRectangle
from .progress import (
    DirectoryProgress,
    PhaseProgress,
    ProcessingCompleted,
    ProgressEvent,
    ProgressPhase,
)

_PHASE_LABELS: dict[ProgressPhase, str] = {
    ProgressPhase.CROP_ANALYSIS: "Analyzing crop",
    ProgressPhase.FRAME_DECODING: "Decoding frames",
    ProgressPhase.PNG_ENCODING: "Encoding PNG",
    ProgressPhase.OUTPUT_PUBLICATION: "Publishing output",
}

_FALLBACK_TERMINAL_COLUMNS = 80
_SYNTHETIC_TERMINAL_COLUMNS = 1_000


def _terminal_columns(stream: TextIO) -> int:
    """Return usable terminal columns or a conservative redirected fallback."""
    try:
        columns = get_terminal_size(stream.fileno()).columns
    except OSError:
        columns = (
            _SYNTHETIC_TERMINAL_COLUMNS
            if stream.isatty()
            else _FALLBACK_TERMINAL_COLUMNS
        )
    return max(1, columns - 1)


def _fit_status_line(line: str, columns: int) -> str:
    """Fit one status update without allowing terminal line wrapping."""
    if len(line) <= columns:
        return line
    if columns <= 3:
        return "." * columns
    suffix_length = columns // 3
    prefix_length = columns - suffix_length - 3
    return f"{line[:prefix_length]}...{line[-suffix_length:]}"


def _redraw_status_line(stream: TextIO, line: str, previous_length: int) -> int:
    """Redraw one fitted status line and return its visible length."""
    columns = _terminal_columns(stream)
    fitted_line = _fit_status_line(line, columns)
    padding = " " * max(0, min(previous_length, columns) - len(fitted_line))
    stream.write(f"\r{fitted_line}{padding}")
    stream.flush()
    return len(fitted_line)


class ConsoleProgressRenderer:
    """Render progress events to stderr without receiving source-media paths."""

    _BAR_WIDTH = 20

    def __init__(self, stream: TextIO = sys.stderr) -> None:
        """Create a renderer that redraws progress on one text stream.

        Args:
            stream: stderr-compatible stream that receives status text.
        """
        self._stream = stream
        self._last_interactive_line_length = 0
        self._has_interactive_line = False

    def __call__(self, event: ProgressEvent) -> None:
        """Render one progress or completed-processing event.

        Args:
            event: Application-owned progress fact to render.
        """
        if isinstance(event, ProcessingCompleted):
            self._render_completed(event)
            return
        self._write_interactive(self._phase_line(event))

    def finish_line(self) -> None:
        """End an in-place redraw before another writer takes stderr."""
        if not self._has_interactive_line:
            return
        self._stream.write("\n")
        self._stream.flush()
        self._has_interactive_line = False
        self._last_interactive_line_length = 0

    def _render_completed(self, event: ProcessingCompleted) -> None:
        """Render the success summary after any active progress redraw."""
        self.finish_line()
        self._write_line(_format_completed(event))

    def _phase_line(self, event: PhaseProgress) -> str:
        """Format one phase status within the current terminal width."""
        return _format_phase_for_width(
            event,
            self._BAR_WIDTH,
            _terminal_columns(self._stream),
        )

    def _bar(self, event: PhaseProgress) -> str:
        """Render one determinate or bouncing ASCII bar with exactly 20 cells."""
        return _format_bar(event, self._BAR_WIDTH)

    def _write_interactive(self, line: str) -> None:
        """Redraw one padded status line without terminal wrapping."""
        self._last_interactive_line_length = _redraw_status_line(
            self._stream,
            line,
            self._last_interactive_line_length,
        )
        self._has_interactive_line = True

    def _write_line(self, line: str) -> None:
        """Write and flush one complete newline-terminated text line."""
        self._stream.write(f"{line}\n")
        self._stream.flush()


def _format_completed(event: ProcessingCompleted) -> str:
    """Format one unchanged complete-processing summary."""
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
    return " ".join(summary_parts)


def _format_phase(event: PhaseProgress, bar_width: int) -> str:
    """Format one complete human-readable phase status line."""
    parts, _ = _phase_parts(event, bar_width)
    return " ".join(parts)


def _format_phase_for_width(
    event: PhaseProgress,
    bar_width: int,
    max_width: int,
) -> str:
    """Keep core work and worker facts while removing optional overflow."""
    parts, removable_parts = _phase_parts(event, bar_width)
    for removable_part in removable_parts:
        if len(" ".join(parts)) <= max_width:
            break
        parts.remove(removable_part)
    return _fit_status_line(" ".join(parts), max_width)


def _phase_parts(
    event: PhaseProgress,
    bar_width: int,
) -> tuple[list[str], list[str]]:
    """Build ordered phase fields and lowest-priority removable fields."""
    parts = [
        _PHASE_LABELS[event.phase],
        _format_bar(event, bar_width),
        _format_work(event),
    ]
    if event.fraction_complete is not None:
        percentage_prefix = "~" if event.is_estimate else ""
        parts.append(f"{percentage_prefix}{event.fraction_complete:.0%}")
    elapsed_part: str | None = None
    rate_part: str | None = None
    if event.phase is not ProgressPhase.OUTPUT_PUBLICATION:
        elapsed_part = f"elapsed {_format_duration(event.elapsed_seconds)}"
        parts.append(elapsed_part)
        if event.rate_per_second is not None:
            rate_part = f"{event.rate_per_second:.1f} {event.unit.value}/s"
            parts.append(rate_part)
    if event.worker_count is not None:
        parts.append(_format_workers(event.worker_count))
    eta_part: str | None = None
    if event.eta_seconds is not None:
        eta_part = f"ETA {_format_duration(event.eta_seconds)}"
        parts.append(eta_part)
    removable_parts = [
        part for part in (rate_part, elapsed_part, eta_part) if part is not None
    ]
    return parts, removable_parts


def _format_bar(event: PhaseProgress, bar_width: int) -> str:
    """Render one determinate or bouncing ASCII bar with fixed-width cells."""
    if event.fraction_complete is not None:
        completed_cells = round(event.fraction_complete * bar_width)
        completed_cells = min(bar_width, max(0, completed_cells))
        return "[" + "=" * completed_cells + "-" * (bar_width - completed_cells) + "]"
    bounce_span = (bar_width - 1) * 2
    bounce_position = event.completed_units % bounce_span
    if bounce_position >= bar_width:
        bounce_position = bounce_span - bounce_position
    return (
        "["
        + "-" * bounce_position
        + "#"
        + "-" * (bar_width - bounce_position - 1)
        + "]"
    )


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


class DirectoryConsolePresenter(logging.Handler):
    """Render directory workflow facts and logging records on one stream."""

    _BAR_WIDTH = 20
    _LOG_CONTINUATION_INDENT = "  "

    def __init__(self, stream: TextIO = sys.stderr) -> None:
        """Create one coordinated presenter for a directory command.

        Args:
            stream: stderr-compatible stream that receives logs and status.
        """
        super().__init__()
        self._stream = stream
        self._directory_progress: DirectoryProgress | None = None
        self._generator_phase: PhaseProgress | None = None
        self._last_interactive_line_length = 0
        self._has_interactive_line = False
        self._stats_started = False
        self._terminal_outcome_count = 0

    def report_directory_progress(self, event: DirectoryProgress) -> None:
        """Render one authoritative directory workflow state snapshot.

        Args:
            event: Complete directory state after one lifecycle transition.
        """
        self.acquire()
        try:
            previous_source_path = (
                None
                if self._directory_progress is None
                else self._directory_progress.current_source_path
            )
            if previous_source_path != event.current_source_path:
                self._clear_status_line()
                self._generator_phase = None
            previous_outcome_count = self._terminal_outcome_count
            self._terminal_outcome_count = (
                event.completed + event.skipped + event.failed
            )
            self._directory_progress = event
            should_print_initial_stats = not self._stats_started and (
                event.current_source_path is not None
                or event.next_scan_seconds is not None
            )
            if (
                should_print_initial_stats
                or self._terminal_outcome_count > previous_outcome_count
            ):
                self._write_stats_line()
            self._render_directory_status()
        finally:
            self.release()

    def report_generator_progress(self, event: ProgressEvent) -> None:
        """Render one per-source generator event inside the directory status.

        Args:
            event: Phase update or successful completion from the generator.
        """
        self.acquire()
        try:
            if isinstance(event, ProcessingCompleted):
                self._clear_status_line()
                self._write_line(_format_completed(event))
                self._generator_phase = None
                return
            self._generator_phase = event
            self._render_directory_status()
        finally:
            self.release()

    def finish_line(self) -> None:
        """End an in-place interactive redraw before another writer uses the stream."""
        self.acquire()
        try:
            if not self._has_interactive_line:
                return
            self._stream.write("\n")
            self._stream.flush()
            self._has_interactive_line = False
            self._last_interactive_line_length = 0
        finally:
            self.release()

    def finish(self) -> None:
        """End presentation after writing any missing initial statistics."""
        self.acquire()
        try:
            if self._directory_progress is not None and not self._stats_started:
                self._write_stats_line()
            self._clear_status_line()
        finally:
            self.release()

    def emit(self, record: logging.LogRecord) -> None:
        """Clear transient status and write one complete lifecycle log record.

        Args:
            record: Standard-library logging record to render.
        """
        self.acquire()
        try:
            self._clear_status_line()
            formatted_record = self.format(record)
            if "\n" in formatted_record:
                formatted_record = formatted_record.replace(
                    "\n",
                    f"\n{self._LOG_CONTINUATION_INDENT}",
                )
            self._write_line(formatted_record)
        finally:
            self.release()

    def _render_directory_status(self) -> None:
        """Render active progress or idle watch text without repeating stats."""
        status_line = self._status_line()
        if status_line is not None:
            self._write_interactive(status_line)

    def _status_line(self) -> str | None:
        """Compose transient generator progress or idle watch context."""
        if self._directory_progress is None:
            return None
        event = self._directory_progress
        if event.current_source_path is not None:
            if self._generator_phase is None:
                return None
            columns = _terminal_columns(self._stream)
            phase_line = _format_phase_for_width(
                self._generator_phase,
                self._BAR_WIDTH,
                columns,
            )
            source_context = (
                f"Movie: {event.current_source_path.name}",
                f"Path: {event.current_source_path}",
            )
            full_status_line = " | ".join((phase_line, *source_context))
            if len(full_status_line) <= columns:
                return full_status_line
            return phase_line
        if (
            event.waiting == 0
            and event.queued == 0
            and event.next_scan_seconds is not None
        ):
            return (
                "Watching for new videos... "
                f"Next scan in {_format_duration(event.next_scan_seconds)}"
            )
        return None

    def _stats_line(self) -> str:
        """Format the latest directory counters for persistent output."""
        assert self._directory_progress is not None
        event = self._directory_progress
        return (
            f"Discovered {event.discovered} | Waiting {event.waiting} | "
            f"Queued {event.queued} | Completed {event.completed} | "
            f"Skipped {event.skipped} | Failed {event.failed}"
        )

    def _write_stats_line(self) -> None:
        """Persist one counter snapshot at a workflow milestone."""
        self._clear_status_line()
        self._write_line(self._stats_line())
        self._stats_started = True

    def _clear_status_line(self) -> None:
        """Erase one transient redraw without making it a permanent line."""
        if not self._has_interactive_line:
            return
        clear_length = min(
            self._last_interactive_line_length,
            _terminal_columns(self._stream),
        )
        self._stream.write(f"\r{' ' * clear_length}\r")
        self._stream.flush()
        self._has_interactive_line = False
        self._last_interactive_line_length = 0

    def _write_interactive(self, line: str) -> None:
        """Redraw one padded status line without terminal wrapping."""
        self._last_interactive_line_length = _redraw_status_line(
            self._stream,
            line,
            self._last_interactive_line_length,
        )
        self._has_interactive_line = True

    def _write_line(self, line: str) -> None:
        """Write and flush one complete newline-terminated text line."""
        self._stream.write(f"{line}\n")
        self._stream.flush()
