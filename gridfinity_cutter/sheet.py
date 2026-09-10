"""Reference marker sheet: geometry (single source of truth) and PDF rendering.

Sheet coordinates are millimetres, origin at the top-left corner of the page,
x to the right, y downwards. Step 1 imports the geometry from here to build the
homography from photo pixels to sheet millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

PAPER_SIZES: dict[str, tuple[float, float]] = {"a4": (210.0, 297.0)}

MARGIN_MM = 10.0
MARKER_SIZE_MM = 40.0
ARUCO_DICT = cv2.aruco.DICT_4X4_50
# Marker order: top-left, top-right, bottom-right, bottom-left.
MARKER_IDS: tuple[int, int, int, int] = (0, 1, 2, 3)
MARKER_BITS = 4  # DICT_4X4
MARKER_BORDER_BITS = 1
MARKER_CELLS = MARKER_BITS + 2 * MARKER_BORDER_BITS
SCALE_BAR_MM = 100.0


@dataclass(frozen=True)
class SheetSpec:
    paper: str = "a4"
    margin_mm: float = MARGIN_MM
    marker_size_mm: float = MARKER_SIZE_MM
    marker_ids: tuple[int, int, int, int] = MARKER_IDS
    aruco_dict: int = ARUCO_DICT

    @property
    def page_mm(self) -> tuple[float, float]:
        return PAPER_SIZES[self.paper]


DEFAULT_SPEC = SheetSpec()


def aruco_dictionary(spec: SheetSpec = DEFAULT_SPEC) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(spec.aruco_dict)


def marker_bits(marker_id: int, spec: SheetSpec = DEFAULT_SPEC) -> np.ndarray:
    """Return the marker as a (MARKER_CELLS x MARKER_CELLS) bool array, True = black."""
    img = aruco_dictionary(spec).generateImageMarker(marker_id, MARKER_CELLS, MARKER_BORDER_BITS)
    return img == 0


def marker_corners_mm(spec: SheetSpec = DEFAULT_SPEC) -> dict[int, np.ndarray]:
    """Outer corners of each marker in sheet mm, ordered TL, TR, BR, BL.

    This matches the corner order returned by cv2.aruco.ArucoDetector, so the
    two can be fed to cv2.findHomography directly.
    """
    w, h = spec.page_mm
    s = spec.marker_size_mm
    m = spec.margin_mm
    origins = {
        spec.marker_ids[0]: (m, m),
        spec.marker_ids[1]: (w - m - s, m),
        spec.marker_ids[2]: (w - m - s, h - m - s),
        spec.marker_ids[3]: (m, h - m - s),
    }
    return {
        mid: np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float64)
        for mid, (x, y) in origins.items()
    }


def work_area_mm(spec: SheetSpec = DEFAULT_SPEC) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of the region between the markers where the object goes."""
    w, h = spec.page_mm
    inner = spec.margin_mm + spec.marker_size_mm
    return (inner, inner, w - inner, h - inner)


# -- rendering ----------------------------------------------------------------
#
# The sheet is described once as a list of primitives in sheet mm (origin
# top-left, y down) and rendered by two backends: PDF (reportlab, for the CLI)
# and SVG/HTML (for printing straight from the browser in the UI). Both must
# draw the same thing, so keep all layout decisions in ``sheet_elements``.

PT_PER_MM = 72 / 25.4
FONT_FAMILY_SVG = "Helvetica, Arial, sans-serif"


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float
    fill: bool = True
    stroke_gray: float | None = None  # None = no stroke
    line_width_mm: float = 0.1
    dash_mm: tuple[float, float] | None = None


@dataclass(frozen=True)
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    line_width_mm: float = 0.15


@dataclass(frozen=True)
class Text:
    x: float
    y: float  # baseline
    text: str
    size_pt: float
    anchor: str = "start"  # "start" | "middle"


Element = Rect | Line | Text


def sheet_elements(spec: SheetSpec = DEFAULT_SPEC) -> list[Element]:
    page_w, page_h = spec.page_mm
    out: list[Element] = []

    # Markers: one filled rect per black cell, no bitmap resampling.
    cell = spec.marker_size_mm / MARKER_CELLS
    for mid, corners in marker_corners_mm(spec).items():
        ox, oy = corners[0]
        bits = marker_bits(mid, spec)
        for row in range(MARKER_CELLS):
            for col in range(MARKER_CELLS):
                if bits[row, col]:
                    out.append(Rect(ox + col * cell, oy + row * cell, cell, cell))
        # Label below top markers, above bottom markers, so nothing overlaps the footer.
        label_y = oy + spec.marker_size_mm + 3 if oy < page_h / 2 else oy - 1.5
        out.append(Text(ox, label_y, f"id {mid}", 7))

    # Work area outline.
    x0, y0, x1, y1 = work_area_mm(spec)
    out.append(Rect(x0, y0, x1 - x0, y1 - y0, fill=False, stroke_gray=0.75, dash_mm=(2, 2)))

    # Scale bar along the bottom, centred between the two bottom markers.
    bar_y = page_h - spec.margin_mm - spec.marker_size_mm / 2
    bar_x = (page_w - SCALE_BAR_MM) / 2
    out.append(Line(bar_x, bar_y, bar_x + SCALE_BAR_MM, bar_y))
    for tick in range(0, int(SCALE_BAR_MM) + 1, 10):
        tx = bar_x + tick
        out.append(Line(tx, bar_y, tx, bar_y - 2))
        out.append(Text(tx, bar_y + 3, str(tick), 7, anchor="middle"))
    out.append(
        Text(
            page_w / 2,
            bar_y - 6,
            f"Scale check: this bar must measure exactly {SCALE_BAR_MM:.0f} mm",
            7,
            anchor="middle",
        )
    )

    # Footer.
    out.append(
        Text(
            page_w / 2,
            page_h - spec.margin_mm / 2,
            "gridfinity-cutter reference sheet - print at 100% / actual size, do not fit to page. "
            f"Markers: {spec.marker_ids[0]} TL, {spec.marker_ids[1]} TR, "
            f"{spec.marker_ids[2]} BR, {spec.marker_ids[3]} BL. Paper: {spec.paper.upper()}.",
            8,
            anchor="middle",
        )
    )
    return out


def render_pdf(path: str | Path, spec: SheetSpec = DEFAULT_SPEC) -> None:
    page_w, page_h = spec.page_mm
    c = canvas.Canvas(str(path), pagesize=(page_w * mm, page_h * mm))
    c.setFillGray(0)
    c.setStrokeGray(0)

    def y_pdf(y: float) -> float:
        # Sheet coordinates have their origin top-left, PDF bottom-left.
        return (page_h - y) * mm

    for el in sheet_elements(spec):
        if isinstance(el, Rect):
            if el.stroke_gray is not None:
                c.setStrokeGray(el.stroke_gray)
                c.setLineWidth(el.line_width_mm * mm)
                c.setDash(*el.dash_mm) if el.dash_mm else c.setDash()
            c.rect(
                el.x * mm,
                y_pdf(el.y + el.h),
                el.w * mm,
                el.h * mm,
                fill=int(el.fill),
                stroke=int(el.stroke_gray is not None),
            )
            c.setStrokeGray(0)
            c.setDash()
        elif isinstance(el, Line):
            c.setLineWidth(el.line_width_mm * mm)
            c.line(el.x0 * mm, y_pdf(el.y0), el.x1 * mm, y_pdf(el.y1))
        else:
            c.setFont("Helvetica", el.size_pt)
            if el.anchor == "middle":
                c.drawCentredString(el.x * mm, y_pdf(el.y), el.text)
            else:
                c.drawString(el.x * mm, y_pdf(el.y), el.text)

    c.showPage()
    c.save()


def _g(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def render_svg(spec: SheetSpec = DEFAULT_SPEC) -> str:
    """The sheet as an SVG string, 1 user unit = 1 mm, physical size set in mm."""
    from xml.sax.saxutils import escape

    page_w, page_h = spec.page_mm
    size = f'width="{_g(page_w)}mm" height="{_g(page_h)}mm" viewBox="0 0 {_g(page_w)} {_g(page_h)}"'
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" {size} font-family="{FONT_FAMILY_SVG}">']
    for el in sheet_elements(spec):
        if isinstance(el, Rect):
            attrs = f'x="{_g(el.x)}" y="{_g(el.y)}" width="{_g(el.w)}" height="{_g(el.h)}"'
            attrs += ' fill="#000"' if el.fill else ' fill="none"'
            if el.stroke_gray is not None:
                gray = round(el.stroke_gray * 255)
                attrs += (
                    f' stroke="rgb({gray},{gray},{gray})" stroke-width="{_g(el.line_width_mm)}"'
                )
                if el.dash_mm:
                    attrs += f' stroke-dasharray="{_g(el.dash_mm[0])} {_g(el.dash_mm[1])}"'
            parts.append(f"<rect {attrs}/>")
        elif isinstance(el, Line):
            parts.append(
                f'<line x1="{_g(el.x0)}" y1="{_g(el.y0)}" x2="{_g(el.x1)}" y2="{_g(el.y1)}" '
                f'stroke="#000" stroke-width="{_g(el.line_width_mm)}"/>'
            )
        else:
            parts.append(
                f'<text x="{_g(el.x)}" y="{_g(el.y)}" font-size="{_g(el.size_pt / PT_PER_MM)}" '
                f'text-anchor="{el.anchor}">{escape(el.text)}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def render_print_html(spec: SheetSpec = DEFAULT_SPEC) -> str:
    """A self-contained HTML page that prints the sheet at actual size.

    ``@page`` fixes the paper size and removes the printer margins so the
    browser's default scale of 100% puts the markers exactly where
    ``marker_corners_mm`` says they are. The UI loads this into a hidden iframe
    and calls ``print()`` on it.
    """
    page_w, page_h = spec.page_mm
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>gridfinity-cutter reference sheet ({spec.paper.upper()})</title>"
        "<style>"
        f"@page{{size:{_g(page_w)}mm {_g(page_h)}mm;margin:0}}"
        "html,body{margin:0;padding:0;background:#fff}"
        f"svg{{display:block;width:{_g(page_w)}mm;height:{_g(page_h)}mm}}"
        "</style></head><body>"
        f"{render_svg(spec)}"
        "</body></html>"
    )
