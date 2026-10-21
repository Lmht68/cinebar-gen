"""End-to-end tests for the installed cinebar command."""

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import png

RGB_COLORS = [(40, 50, 60), (0, 0, 0), (200, 80, 20), (9, 150, 230)]
PRESENTATION_TIMESTAMPS = [0, 1, 4, 10]


def test_installed_command_processes_all_variable_timestamp_frames(
    tmp_path: Path,
) -> None:
    """Invoke cinebar and preserve all lossless fixture colors in order."""
    media_path = tmp_path / "fixture.mkv"
    output_path = tmp_path / "barcode.png"
    _write_lossless_fixture(media_path)

    with av.open(str(media_path)) as fixture:
        selected_stream = fixture.streams.video[0]
        assert selected_stream.frames == 0

    cinebar_executable = shutil.which("cinebar")
    assert cinebar_executable is not None, "Installed cinebar executable was not found."
    completed_process = subprocess.run(
        [
            cinebar_executable,
            str(media_path),
            "--output",
            str(output_path),
            "--height",
            "3",
            "--workers",
            "1",
            "--crop",
            "none",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert _read_barcode(output_path) == (4, 3, RGB_COLORS)


def _write_lossless_fixture(media_path: Path) -> None:
    """Write a Matroska fixture with lossless bgr0 frames and uneven PTS values."""
    with av.open(str(media_path), mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=24)
        stream.width = 2
        stream.height = 2
        stream.pix_fmt = "bgr0"

        for presentation_timestamp, color in zip(
            PRESENTATION_TIMESTAMPS, RGB_COLORS, strict=True
        ):
            frame = av.VideoFrame.from_ndarray(
                np.full((2, 2, 3), color, dtype=np.uint8), format="rgb24"
            )
            frame.pts = presentation_timestamp
            frame.time_base = Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def _read_barcode(barcode_path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    """Read dimensions and first-row RGB columns from a lossless barcode PNG."""
    with barcode_path.open("rb") as barcode_file:
        width, height, rows, _ = png.Reader(file=barcode_file).read()
        pixel_rows = [list(row) for row in rows]

    first_row = pixel_rows[0]
    colors = [
        (first_row[index], first_row[index + 1], first_row[index + 2])
        for index in range(0, len(first_row), 3)
    ]
    assert pixel_rows == [first_row] * height
    return width, height, colors
