#!/usr/bin/env python3
"""Tile a poster PDF onto A4 pages for home printing.

The first page of the input PDF is laid out across an N×M grid of A4
sheets at the requested physical size. Adjacent sheets share an overlap
strip so the printed pages can be trimmed and taped without seams.

Examples
--------
Original 36×24" size, A4 landscape, with crop marks:
    python tile_poster.py "LoRA Poster.pdf" --preset original

Shrink to a 550 mm-wide poster (2×2 = 4 sheets):
    python tile_poster.py "LoRA Poster.pdf" --preset small

Custom width:
    python tile_poster.py "LoRA Poster.pdf" --width-mm 700 --landscape

Pages are emitted top row first, left to right (row 1 col 1, row 1 col 2, ...).
Print at 100% / "Actual Size" — no "fit to page" — so the overlap regions align.
"""
import argparse
import math
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ArrayObject, DecodedStreamObject, NameObject

PT_PER_MM = 72.0 / 25.4
A4_W_PT = 210 * PT_PER_MM
A4_H_PT = 297 * PT_PER_MM

PRESETS = {
    # name      → kwargs that override defaults / CLI flags
    "small":    {"width_mm": 550,  "landscape": True},   # ~2×2 = 4 sheets
    "medium":   {"width_mm": 850,  "landscape": True},   # ~4×3 = 12 sheets
    "original": {"scale": 1.0,     "landscape": True},   # 36×24" → 4×4 = 16 sheets
}


def _crop_marks_stream(sheet_w, sheet_h, margin, overlap, c, r, cols, rows):
    """Build a PDF content stream of trim-box and overlap-cut tick marks."""
    L = 4 * PT_PER_MM  # tick length
    parts = ["q", "0 0 0 RG", "0.4 w"]
    x0, y0 = margin, margin
    x1, y1 = sheet_w - margin, sheet_h - margin

    def line(a, b, cc, d):
        parts.append(f"{a:.2f} {b:.2f} m {cc:.2f} {d:.2f} l S")

    # Trim-box L-marks at all four corners — always drawn.
    line(x0, y0, x0 + L, y0); line(x0, y0, x0, y0 + L)
    line(x1, y0, x1 - L, y0); line(x1, y0, x1, y0 + L)
    line(x0, y1, x0 + L, y1); line(x0, y1, x0, y1 - L)
    line(x1, y1, x1 - L, y1); line(x1, y1, x1, y1 - L)

    # Overlap cut-line ticks: drawn at the edge of the overlap strip on the
    # sides that have a neighbouring tile. Cut along the line between the two
    # ticks to trim the overlap strip away.
    if c > 0:                       # left neighbour exists
        x = x0 + overlap
        line(x, y0, x, y0 + L); line(x, y1, x, y1 - L)
    if c < cols - 1:                # right neighbour
        x = x1 - overlap
        line(x, y0, x, y0 + L); line(x, y1, x, y1 - L)
    if r > 0:                       # neighbour above in poster (row r-1)
        y = y1 - overlap
        line(x0, y, x0 + L, y); line(x1, y, x1 - L, y)
    if r < rows - 1:                # neighbour below in poster (row r+1)
        y = y0 + overlap
        line(x0, y, x0 + L, y); line(x1, y, x1 - L, y)

    parts.append("Q")
    return ("\n".join(parts) + "\n").encode("latin-1")


def _append_content(writer, page, data):
    """Append raw PDF operators to a page by flattening /Contents into a
    single stream. pypdf's merge_transformed_page can leave /Contents as a
    nested ArrayObject; appending naively produces an invalid structure that
    some renderers (CoreGraphics, Preview) silently drop."""
    chunks = []

    def collect(item):
        item = item.get_object() if hasattr(item, "get_object") else item
        if isinstance(item, ArrayObject):
            for sub in item:
                collect(sub)
        elif item is not None and hasattr(item, "get_data"):
            chunks.append(item.get_data())

    contents = page.get(NameObject("/Contents"))
    if contents is not None:
        collect(contents)
    chunks.append(data)

    merged = DecodedStreamObject()
    merged.set_data(b"\n".join(chunks))
    page[NameObject("/Contents")] = writer._add_object(merged)


def tile_poster(input_path, output_path, *, scale, overlap_mm, margin_mm,
                landscape, crop_marks):
    reader = PdfReader(str(input_path))
    src = reader.pages[0]
    src_w = float(src.mediabox.width)
    src_h = float(src.mediabox.height)

    poster_w = src_w * scale
    poster_h = src_h * scale

    sheet_w, sheet_h = (A4_H_PT, A4_W_PT) if landscape else (A4_W_PT, A4_H_PT)
    margin = margin_mm * PT_PER_MM
    overlap = overlap_mm * PT_PER_MM
    tile_w = sheet_w - 2 * margin
    tile_h = sheet_h - 2 * margin
    step_w = tile_w - overlap
    step_h = tile_h - overlap
    if step_w <= 0 or step_h <= 0:
        raise ValueError("overlap + 2*margin must be smaller than the sheet")

    cols = max(1, math.ceil((poster_w - overlap) / step_w))
    rows = max(1, math.ceil((poster_h - overlap) / step_h))

    writer = PdfWriter()
    for r in range(rows):
        for c in range(cols):
            page = writer.add_blank_page(width=sheet_w, height=sheet_h)
            tx = margin - c * step_w
            ty = margin - (rows - 1 - r) * step_h
            ctm = Transformation().scale(scale).translate(tx, ty)
            page.merge_transformed_page(src, ctm)
            if crop_marks:
                _append_content(
                    writer, page,
                    _crop_marks_stream(
                        sheet_w, sheet_h, margin, overlap, c, r, cols, rows,
                    ),
                )

    with open(output_path, "wb") as f:
        writer.write(f)
    return cols, rows, poster_w / PT_PER_MM, poster_h / PT_PER_MM


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="source poster PDF")
    p.add_argument("-o", "--output", type=Path, help="output PDF path")
    p.add_argument("--preset", choices=sorted(PRESETS),
                   help="named size preset (overridden by explicit flags)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--scale", type=float,
                   help="scale factor relative to source size")
    g.add_argument("--width-mm", type=float,
                   help="target printed poster width in mm")
    p.add_argument("--overlap-mm", type=float, default=10.0,
                   help="overlap between adjacent tiles in mm (default 10)")
    p.add_argument("--margin-mm", type=float, default=5.0,
                   help="unprintable margin per A4 side in mm (default 5)")
    orient = p.add_mutually_exclusive_group()
    orient.add_argument("--landscape", dest="landscape", action="store_true",
                        default=None, help="A4 tiles in landscape (default)")
    orient.add_argument("--portrait", dest="landscape", action="store_false",
                        help="A4 tiles in portrait")
    p.add_argument("--no-crop-marks", dest="crop_marks", action="store_false",
                   help="disable trim/cut tick marks")
    args = p.parse_args()

    # Resolve sizing: preset provides defaults; explicit flags override.
    cfg = dict(scale=None, width_mm=None, landscape=True)
    if args.preset:
        cfg.update(PRESETS[args.preset])
    if args.scale is not None:
        cfg["scale"], cfg["width_mm"] = args.scale, None
    if args.width_mm is not None:
        cfg["width_mm"], cfg["scale"] = args.width_mm, None
    if args.landscape is not None:
        cfg["landscape"] = args.landscape
    if cfg["scale"] is None and cfg["width_mm"] is None:
        cfg["scale"] = 1.0  # fall-through default

    reader = PdfReader(str(args.input))
    src_w_pt = float(reader.pages[0].mediabox.width)
    if cfg["width_mm"] is not None:
        scale = (cfg["width_mm"] * PT_PER_MM) / src_w_pt
    else:
        scale = cfg["scale"]

    out = args.output or args.input.with_name(args.input.stem + "_a4.pdf")
    cols, rows, w_mm, h_mm = tile_poster(
        args.input, out,
        scale=scale,
        overlap_mm=args.overlap_mm,
        margin_mm=args.margin_mm,
        landscape=cfg["landscape"],
        crop_marks=args.crop_marks,
    )
    orient = "landscape" if cfg["landscape"] else "portrait"
    print(f"Tiled {w_mm:.0f}×{h_mm:.0f} mm poster into "
          f"{cols}×{rows} = {cols * rows} A4 {orient} sheets → {out}")
    print("Pages emitted top row first, left to right.")


if __name__ == "__main__":
    main()
