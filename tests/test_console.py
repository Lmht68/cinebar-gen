"""Tests for terminal rendering of application-owned progress facts."""

import logging
from io import StringIO
from pathlib import Path

import pytest

from cinebar import (
    DirectoryProgress,
    PhaseProgress,
    ProcessingCompleted,
    ProcessingResult,
    ProgressPhase,
    ProgressState,
    ProgressUnit,
)
from cinebar.console import ConsoleProgressRenderer, DirectoryConsolePresenter


class _InteractiveStream(StringIO):
    """In-memory stream reporting an interactive terminal state."""

    def isatty(self) -> bool:
        """Report that the synthetic stream is interactive."""
        return True


def _phase_progress(
    *,
    phase: ProgressPhase,
    state: ProgressState,
    completed_units: int,
    total_units: int | None,
    unit: ProgressUnit,
    fraction_complete: float | None,
    is_estimate: bool = False,
    elapsed_seconds: float = 0.0,
    rate_per_second: float | None = None,
    eta_seconds: float | None = None,
    worker_count: int | None = None,
) -> PhaseProgress:
    """Build one concise valid phase fact for renderer tests."""
    return PhaseProgress(
        phase=phase,
        state=state,
        completed_units=completed_units,
        total_units=total_units,
        unit=unit,
        fraction_complete=fraction_complete,
        is_estimate=is_estimate,
        elapsed_seconds=elapsed_seconds,
        rate_per_second=rate_per_second,
        eta_seconds=eta_seconds,
        worker_count=worker_count,
    )


def test_directory_progress_rejects_invalid_facts() -> None:
    """Reject invalid public directory-progress measurements."""
    with pytest.raises(ValueError, match="non-negative integer"):
        DirectoryProgress(-1, 0, 0, 0, 0, 0, None, None)
    with pytest.raises(ValueError, match="non-negative integer"):
        DirectoryProgress(0, True, 0, 0, 0, 0, None, None)
    with pytest.raises(ValueError, match="Path or None"):
        DirectoryProgress(0, 0, 0, 0, 0, 0, "Movie.mkv", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite positive"):
        DirectoryProgress(0, 0, 0, 0, 0, 0, None, 0.0)
    with pytest.raises(ValueError, match="finite positive"):
        DirectoryProgress(0, 0, 0, 0, 0, 0, None, float("inf"))


def test_interactive_renderer_shows_estimates_workers_eta_and_summary() -> None:
    """Render one ASCII in-place bar and a complete terminal summary."""
    stream = _InteractiveStream()
    renderer = ConsoleProgressRenderer(stream)

    renderer(
        _phase_progress(
            phase=ProgressPhase.FRAME_DECODING,
            state=ProgressState.STARTED,
            completed_units=0,
            total_units=100,
            unit=ProgressUnit.FRAMES,
            fraction_complete=0.0,
            is_estimate=True,
            worker_count=1,
        )
    )
    renderer(
        _phase_progress(
            phase=ProgressPhase.FRAME_DECODING,
            state=ProgressState.RUNNING,
            completed_units=50,
            total_units=100,
            unit=ProgressUnit.FRAMES,
            fraction_complete=0.5,
            is_estimate=True,
            elapsed_seconds=5.0,
            rate_per_second=10.0,
            eta_seconds=5.0,
            worker_count=1,
        )
    )
    renderer(
        ProcessingCompleted(
            result=ProcessingResult(
                output_path=Path("barcode.png"),
                decoded_frame_count=100,
                output_width=100,
                output_height=3,
            ),
            elapsed_seconds=10.0,
            average_frames_per_second=10.0,
            worker_count=1,
            accepted_crop=None,
        )
    )

    rendered = stream.getvalue()

    assert rendered.isascii()
    assert "\rDecoding frames [==========----------]" in rendered
    assert "50/~100 frames" in rendered
    assert "~50%" in rendered
    assert "1 worker" in rendered
    assert "ETA 00:00:05" in rendered
    assert "Complete: 100 frames, 100x3, crop none, 1 worker" in rendered
    assert "10.0 frames/s" in rendered


def test_interactive_renderer_bounces_without_untrustworthy_eta() -> None:
    """Render indeterminate work without inventing a percentage or ETA."""
    stream = _InteractiveStream()
    renderer = ConsoleProgressRenderer(stream)

    renderer(
        _phase_progress(
            phase=ProgressPhase.FRAME_DECODING,
            state=ProgressState.RUNNING,
            completed_units=12,
            total_units=None,
            unit=ProgressUnit.FRAMES,
            fraction_complete=None,
            elapsed_seconds=3.0,
            rate_per_second=4.0,
            worker_count=2,
        )
    )

    rendered = stream.getvalue()

    assert "Decoding frames [" in rendered
    assert "12 frames" in rendered
    assert "2 workers" in rendered
    assert "ETA" not in rendered
    assert "%" not in rendered


def test_renderer_replaces_progress_on_non_tty_stream() -> None:
    """Redraw progress in place even when the stream does not report a TTY."""
    stream = StringIO()
    renderer = ConsoleProgressRenderer(stream)

    renderer(
        _phase_progress(
            phase=ProgressPhase.PNG_ENCODING,
            state=ProgressState.STARTED,
            completed_units=0,
            total_units=3,
            unit=ProgressUnit.ROWS,
            fraction_complete=0.0,
        )
    )
    renderer(
        _phase_progress(
            phase=ProgressPhase.PNG_ENCODING,
            state=ProgressState.RUNNING,
            completed_units=1,
            total_units=3,
            unit=ProgressUnit.ROWS,
            fraction_complete=1 / 3,
            elapsed_seconds=1.0,
            rate_per_second=1.0,
        )
    )
    renderer(
        _phase_progress(
            phase=ProgressPhase.PNG_ENCODING,
            state=ProgressState.COMPLETED,
            completed_units=3,
            total_units=3,
            unit=ProgressUnit.ROWS,
            fraction_complete=1.0,
            elapsed_seconds=3.0,
            rate_per_second=1.0,
        )
    )
    renderer.finish_line()

    rendered = stream.getvalue()

    assert rendered.count("\r") == 3
    assert rendered.count("\n") == 1
    assert rendered.endswith("\n")
    assert "1/3 rows" in rendered
    assert "3/3 rows" in rendered


def test_renderer_keeps_multiworker_redraw_within_one_terminal_row() -> None:
    """Keep a detailed multi-worker update within the fallback terminal width."""
    stream = StringIO()
    renderer = ConsoleProgressRenderer(stream)

    renderer(
        _phase_progress(
            phase=ProgressPhase.FRAME_DECODING,
            state=ProgressState.RUNNING,
            completed_units=12_345,
            total_units=98_765,
            unit=ProgressUnit.FRAMES,
            fraction_complete=0.125,
            elapsed_seconds=3_723.0,
            rate_per_second=123.4,
            eta_seconds=25_000.0,
            worker_count=16,
        )
    )

    status_line = stream.getvalue().removeprefix("\r")

    assert len(status_line) <= 80


def test_finish_line_starts_error_output_on_a_clean_line() -> None:
    """Terminate an interactive redraw before a CLI error is logged."""
    stream = _InteractiveStream()
    renderer = ConsoleProgressRenderer(stream)
    renderer(
        _phase_progress(
            phase=ProgressPhase.OUTPUT_PUBLICATION,
            state=ProgressState.STARTED,
            completed_units=0,
            total_units=1,
            unit=ProgressUnit.FILES,
            fraction_complete=0.0,
        )
    )

    renderer.finish_line()
    stream.write("ERROR [decode_failed] safe message\n")

    assert stream.getvalue().endswith("\nERROR [decode_failed] safe message\n")


def test_renderer_omits_single_file_publication_rate() -> None:
    """Avoid meaningless throughput for one atomic output publication."""
    stream = StringIO()
    renderer = ConsoleProgressRenderer(stream)

    renderer(
        _phase_progress(
            phase=ProgressPhase.OUTPUT_PUBLICATION,
            state=ProgressState.COMPLETED,
            completed_units=1,
            total_units=1,
            unit=ProgressUnit.FILES,
            fraction_complete=1.0,
            elapsed_seconds=0.078,
            rate_per_second=12.8,
        )
    )

    rendered = stream.getvalue()

    assert "Publishing output" in rendered
    assert "elapsed" not in rendered
    assert "12.8 files/s" not in rendered


def test_directory_presenter_renders_active_progress_and_safe_logs() -> None:
    """Show truthful directory state around a clean interactive log line."""
    stream = _InteractiveStream()
    presenter = DirectoryConsolePresenter(stream)
    presenter.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    source_path = Path("Library") / "Movie.mkv"
    presenter.report_directory_progress(
        DirectoryProgress(4, 1, 2, 3, 4, 5, source_path, 5.0)
    )
    presenter.report_generator_progress(
        _phase_progress(
            phase=ProgressPhase.FRAME_DECODING,
            state=ProgressState.RUNNING,
            completed_units=30,
            total_units=100,
            unit=ProgressUnit.FRAMES,
            fraction_complete=0.3,
            elapsed_seconds=3.0,
            rate_per_second=10.0,
            eta_seconds=7.0,
            worker_count=1,
        )
    )
    presenter.emit(
        logging.LogRecord(
            "cinebar.directory",
            logging.INFO,
            __file__,
            0,
            "Discovered source.\nSource: %s",
            (source_path,),
            None,
        )
    )
    presenter.report_directory_progress(DirectoryProgress(4, 0, 0, 3, 4, 5, None, 5.0))

    rendered = stream.getvalue()

    assert (
        "Discovered 4 | Waiting 1 | Queued 2 | Completed 3 | Skipped 4 | Failed 5"
    ) in rendered
    assert "Movie: Movie.mkv" in rendered
    assert f"Path: {source_path}" in rendered
    assert "30/100 frames" in rendered
    assert "30%" in rendered
    assert "elapsed 00:00:03" in rendered
    assert "10.0 frames/s" in rendered
    assert "ETA 00:00:07" in rendered
    assert f"\rINFO Discovered source.\n  Source: {source_path}\n" in rendered
    assert "Watching for new videos... Next scan in 00:00:05" in rendered


def test_directory_presenter_prints_stats_at_start_and_after_each_run() -> None:
    """Persist counters once before work and once after each terminal outcome."""
    stream = StringIO()
    presenter = DirectoryConsolePresenter(stream)
    presenter.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    first_source = Path("Library") / "First.mkv"
    second_source = Path("Library") / "Second.mkv"

    presenter.report_directory_progress(
        DirectoryProgress(2, 0, 1, 0, 0, 0, first_source, None)
    )
    presenter.report_generator_progress(
        _phase_progress(
            phase=ProgressPhase.FRAME_DECODING,
            state=ProgressState.RUNNING,
            completed_units=1,
            total_units=2,
            unit=ProgressUnit.FRAMES,
            fraction_complete=0.5,
            worker_count=4,
        )
    )
    presenter.emit(
        logging.LogRecord(
            "cinebar.directory",
            logging.INFO,
            __file__,
            0,
            "Processing source.",
            (),
            None,
        )
    )
    presenter.report_directory_progress(DirectoryProgress(2, 0, 1, 1, 0, 0, None, None))
    presenter.emit(
        logging.LogRecord(
            "cinebar.directory",
            logging.INFO,
            __file__,
            0,
            "Completed source.",
            (),
            None,
        )
    )
    presenter.report_directory_progress(
        DirectoryProgress(2, 0, 0, 1, 0, 0, second_source, None)
    )
    presenter.report_directory_progress(DirectoryProgress(2, 0, 0, 2, 0, 0, None, None))

    persisted_lines = [
        line.rsplit("\r", 1)[-1] for line in stream.getvalue().split("\n")
    ]
    stats_lines = [line for line in persisted_lines if line.startswith("Discovered ")]

    assert stats_lines == [
        "Discovered 2 | Waiting 0 | Queued 1 | Completed 0 | Skipped 0 | Failed 0",
        "Discovered 2 | Waiting 0 | Queued 1 | Completed 1 | Skipped 0 | Failed 0",
        "Discovered 2 | Waiting 0 | Queued 0 | Completed 2 | Skipped 0 | Failed 0",
    ]


def test_directory_presenter_replaces_progress_on_non_tty_stream() -> None:
    """Redraw directory progress in place when the stream does not report a TTY."""
    stream = StringIO()
    presenter = DirectoryConsolePresenter(stream)
    source_path = Path("Library") / "Movie.mkv"
    presenter.report_directory_progress(
        DirectoryProgress(1, 0, 0, 0, 0, 0, source_path, 5.0)
    )
    presenter.report_generator_progress(
        _phase_progress(
            phase=ProgressPhase.FRAME_DECODING,
            state=ProgressState.STARTED,
            completed_units=0,
            total_units=2,
            unit=ProgressUnit.FRAMES,
            fraction_complete=0.0,
            worker_count=1,
        )
    )
    presenter.report_generator_progress(
        _phase_progress(
            phase=ProgressPhase.FRAME_DECODING,
            state=ProgressState.COMPLETED,
            completed_units=2,
            total_units=2,
            unit=ProgressUnit.FRAMES,
            fraction_complete=1.0,
            worker_count=1,
        )
    )
    presenter.finish()

    rendered = stream.getvalue()

    assert rendered.startswith(
        "Discovered 1 | Waiting 0 | Queued 0 | Completed 0 | Skipped 0 | Failed 0\n"
    )
    assert rendered.count("\n") == 1
    assert rendered.count("Decoding frames [") == 2
    assert "2/2 frames" in rendered
    assert "\x1b" not in rendered


def test_directory_presenter_prioritizes_multiworker_progress_on_one_row() -> None:
    """Keep the active bar visible when directory context exceeds terminal width."""
    stream = StringIO()
    presenter = DirectoryConsolePresenter(stream)
    source_path = (
        Path("Library")
        / "Deeply Nested Collection"
        / "A Movie With A Very Long Descriptive Name.mkv"
    )
    presenter.report_directory_progress(
        DirectoryProgress(12, 3, 8, 4, 1, 2, source_path, 5.0)
    )
    presenter.report_generator_progress(
        _phase_progress(
            phase=ProgressPhase.FRAME_DECODING,
            state=ProgressState.RUNNING,
            completed_units=12_345,
            total_units=98_765,
            unit=ProgressUnit.FRAMES,
            fraction_complete=0.125,
            elapsed_seconds=3_723.0,
            rate_per_second=123.4,
            eta_seconds=25_000.0,
            worker_count=16,
        )
    )

    status_line = stream.getvalue().split("\r")[-1]

    assert len(status_line) <= 80
    assert status_line.startswith("Decoding frames [")
    assert "16 workers" in status_line


def test_directory_presenter_omits_unreliable_percentages_and_eta() -> None:
    """Keep unknown frame totals and ETA absent from directory status."""
    stream = _InteractiveStream()
    presenter = DirectoryConsolePresenter(stream)
    source_path = Path("Library") / "Movie.mkv"
    presenter.report_directory_progress(
        DirectoryProgress(1, 0, 0, 0, 0, 0, source_path, 5.0)
    )
    presenter.report_generator_progress(
        _phase_progress(
            phase=ProgressPhase.FRAME_DECODING,
            state=ProgressState.RUNNING,
            completed_units=12,
            total_units=None,
            unit=ProgressUnit.FRAMES,
            fraction_complete=None,
            elapsed_seconds=3.0,
            rate_per_second=4.0,
            worker_count=2,
        )
    )
    presenter.report_generator_progress(
        ProcessingCompleted(
            result=ProcessingResult(
                output_path=Path("barcode.png"),
                decoded_frame_count=12,
                output_width=12,
                output_height=3,
            ),
            elapsed_seconds=3.0,
            average_frames_per_second=4.0,
            worker_count=2,
            accepted_crop=None,
        )
    )

    rendered = stream.getvalue()

    assert "12 frames" in rendered
    assert "2 workers" in rendered
    assert "ETA" not in rendered
    assert "%" not in rendered
    assert "Complete: 12 frames, 12x3, crop none, 2 workers" in rendered
