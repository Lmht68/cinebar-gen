# Cinebar

Cinebar converts every validated video frame into one lossless PNG frame strip.

## Install

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required.

```console
uv sync
```

## Generate a barcode

```console
uv run cinebar movie.mp4 --output movie-barcode.png --height 300 --crop none
```

Cinebar defaults to at most 16 worker processes.
Set `--workers N` to choose the process count or `--workers 1` to decode sequentially without frame-count metadata or random seeking.

```console
uv run cinebar movie.mp4 --output movie-barcode.png --workers 8 --crop none
```

Each frame color is OpenCV's per-channel arithmetic mean, truncated to 8-bit values and stored in RGB order.
Parallel workers process contiguous reported frame-index ranges, and the parent writes their compact colors in range order.
Parallel extraction validates seek positions, exact range lengths, adjacent raw-frame digests, and the final end of stream before publishing.
Any mismatch or worker failure prevents publication of the final PNG.

OpenCV frame-index seeking depends on the selected codec and backend.
The validation detects reported position, range, boundary, and end-of-stream mismatches, but it cannot mathematically prove frame identity for every input.
Use `--workers 1` when OpenCV cannot report a positive integer frame count or when sequential decoder behavior is required.