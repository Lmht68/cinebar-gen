# Cinebar

Cinebar converts every validated video frame into one lossless PNG frame strip.

## Install

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required.

```console
uv sync
```

## Generate a barcode

```console
uv run cinebar movie.mp4 --output movie-barcode.png --height 100
```

Cinebar refuses to replace an existing barcode by default.
Use `--overwrite` only when intentional replacement is required.
The new PNG is fully encoded to a temporary file on the destination filesystem before it replaces the final output.

```console
uv run cinebar movie.mp4 --output movie-barcode.png --height 100 --overwrite
```

Cinebar defaults to at most 16 worker processes.
Set `--workers N` to choose the process count or `--workers 1` to decode sequentially without frame-count metadata or random seeking.

```console
uv run cinebar movie.mp4 --output movie-barcode.png --workers 8 --crop none
```

## Progress reporting

Cinebar writes progress status to stderr by default.
Interactive terminals redraw an in-place ASCII bar for crop analysis, frame decoding, PNG encoding, and output publication.
Redirected stderr receives phase starts and completions plus at most one running line every 30 seconds.
Decode worker totals always describe the actual active worker count.
Parallel decode totals and percentages are approximate until range and end-of-stream validation succeeds.
Sequential decode ETA appears only when OpenCV provides finite duration metadata and monotonic in-range presentation timestamps.
Missing, non-finite, out-of-range, or regressing metadata leaves frame decoding indeterminate and suppresses ETA.

Omit `--crop` or use `--crop auto` to select one conservative automatic picture region.
Use `--crop none` to include every decoded pixel.

Automatic mode requires usable finite positive integer frame-count metadata with at least 12 positions.
It probes exactly 12 unique random frame positions, sorted only to minimize backward seeks, before one complete color-extraction pass.
Automatic selection assumes a stable encoded border across the sampled positions, not a perfect distinction between permanent black picture content and a border.

For each sample, automatic mode computes a near-black pixel mask.
A row or column qualifies when at least `ceil(line_pixel_count * 0.99)` pixels have every decoded channel at or below 16.
It scans contiguous qualifying runs from each outer edge.
A fully qualifying dimension is ambiguous, while a zero-length run is valid evidence that the edge remains uncropped.

Each edge is evaluated independently.
An edge is selected only when all sampled frames share dimensions, every informative candidate has the same coordinate, the edge removes at least one line, and its last excluded line satisfies the same near-black criterion: at least `ceil(line_pixel_count * 0.99)` pixels have every decoded channel at or below 16 in every sample.
The resulting rectangle must be non-empty.
Otherwise, that edge remains uncropped.

Automatic mode uses no crop when metadata is unusable or below 12, the probe cannot open, seek, validate a position, or decode a usable sample, sample dimensions differ, evidence is ambiguous or contradictory, a candidate boundary falls below the near-black criterion, no edge is accepted, or the combined rectangle is empty.

```console
uv run cinebar movie.mp4 --output movie-barcode.png --workers 1 --crop 140,80,1780,1000
```

An explicit crop uses zero-based `LEFT,TOP,RIGHT,BOTTOM` coordinates, with inclusive left and top edges and exclusive right and bottom edges.
The same non-empty rectangle must fit every decoded frame.
For automatic and explicit crops, every applicable last excluded row or column must satisfy the same near-black criterion: at least `ceil(line_pixel_count * 0.99)` pixels have every decoded channel at or below 16 in every complete-pass frame.
Sides that remove no pixels have no crop boundary to validate.
Genuine black pixels inside the accepted picture region remain part of the frame-color arithmetic mean.
Invalid crop bounds or a later crop boundary below the near-black criterion fails without publishing a final PNG or retaining a destination-side temporary PNG.

After the bounded automatic probe, Cinebar runs exactly one complete worker-backed color pass.
Each decoded frame contributes one frame strip from the selected picture region.

Each frame color is OpenCV's per-channel arithmetic mean, truncated to 8-bit values and stored in RGB order.
Parallel workers process contiguous reported frame-index ranges, and the parent writes their compact colors in range order.
Parallel extraction validates seek positions, exact range lengths, adjacent raw-frame digests, and the final end of stream before publishing.
Any mismatch or worker failure prevents publication of the final PNG.

OpenCV frame-index seeking depends on the selected codec and backend.
The validation detects reported position, range, boundary, and end-of-stream mismatches, but it cannot mathematically prove frame identity for every input.
Use `--workers 1` when OpenCV cannot report a positive integer frame count or when sequential decoder behavior is required.