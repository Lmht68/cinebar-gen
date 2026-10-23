"""Tests for terminal rendering of application-owned progress facts."""

from io import StringIO
from pathlib import Path

from cinebar import (
    PhaseProgress,
    ProcessingCompleted,
    ProcessingResult,
    ProgressPhase,
    ProgressState,
    ProgressUnit,
)
from cinebar.console import ConsoleProgressRenderer


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


def test_noninteractive_renderer_rate_limits_running_updates() -> None:
    """Preserve phase boundaries while limiting redirected running status lines."""
    stream = StringIO()
    renderer = ConsoleProgressRenderer(stream)
    started = _phase_progress(
        phase=ProgressPhase.PNG_ENCODING,
        state=ProgressState.STARTED,
        completed_units=0,
        total_units=3,
        unit=ProgressUnit.ROWS,
        fraction_complete=0.0,
    )

    renderer(started)
    renderer(
        _phase_progress(
            phase=ProgressPhase.PNG_ENCODING,
            state=ProgressState.RUNNING,
            completed_units=1,
            total_units=3,
            unit=ProgressUnit.ROWS,
            fraction_complete=1 / 3,
            elapsed_seconds=29.0,
            rate_per_second=1 / 29,
        )
    )
    renderer(
        _phase_progress(
            phase=ProgressPhase.PNG_ENCODING,
            state=ProgressState.RUNNING,
            completed_units=2,
            total_units=3,
            unit=ProgressUnit.ROWS,
            fraction_complete=2 / 3,
            elapsed_seconds=30.0,
            rate_per_second=2 / 30,
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
            elapsed_seconds=31.0,
            rate_per_second=3 / 31,
        )
    )

    lines = stream.getvalue().splitlines()

    assert len(lines) == 3
    assert lines[0].startswith("Encoding PNG")
    assert "2/3 rows" in lines[1]
    assert lines[2].startswith("Encoding PNG")


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
