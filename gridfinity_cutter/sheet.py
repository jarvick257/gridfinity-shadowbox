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


def render_pdf(path: str | Path, spec: SheetSpec = DEFAULT_SPEC) -> None:
    page_w, page_h = spec.page_mm
    c = canvas.Canvas(str(path), pagesize=(page_w * mm, page_h * mm))

    def rect(x: float, y: float, w: float, h: float, fill: int = 1, stroke: int = 0) -> None:
        # Convert from top-left-origin sheet coordinates to PDF bottom-left origin.
        c.rect(x * mm, (page_h - y - h) * mm, w * mm, h * mm, fill=fill, stroke=stroke)

    # Markers: one filled rect per black cell, no bitmap resampling.
    c.setFillGray(0)
    cell = spec.marker_size_mm / MARKER_CELLS
    for mid, corners in marker_corners_mm(spec).items():
        ox, oy = corners[0]
        bits = marker_bits(mid, spec)
        for row in range(MARKER_CELLS):
            for col in range(MARKER_CELLS):
                if bits[row, col]:
                    rect(ox + col * cell, oy + row * cell, cell, cell)
        c.setFont("Helvetica", 7)
        # Label below top markers, above bottom markers, so nothing overlaps the footer.
        label_y = oy + spec.marker_size_mm + 3 if oy < page_h / 2 else oy - 1.5
        c.drawString(ox * mm, (page_h - label_y) * mm, f"id {mid}")

    # Work area outline.
    x0, y0, x1, y1 = work_area_mm(spec)
    c.setStrokeGray(0.75)
    c.setDash(2, 2)
    c.setLineWidth(0.3)
    rect(x0, y0, x1 - x0, y1 - y0, fill=0, stroke=1)
    c.setDash()

    # Scale bar along the bottom, centred between the two bottom markers.
    bar_y = page_h - spec.margin_mm - spec.marker_size_mm / 2
    bar_x = (page_w - SCALE_BAR_MM) / 2
    c.setStrokeGray(0)
    c.setLineWidth(0.4)
    c.line(bar_x * mm, (page_h - bar_y) * mm, (bar_x + SCALE_BAR_MM) * mm, (page_h - bar_y) * mm)
    c.setFont("Helvetica", 7)
    for tick in range(0, int(SCALE_BAR_MM) + 1, 10):
        tx = bar_x + tick
        c.line(tx * mm, (page_h - bar_y) * mm, tx * mm, (page_h - bar_y + 2) * mm)
        c.drawCentredString(tx * mm, (page_h - bar_y - 3) * mm, str(tick))
    c.drawCentredString(
        (page_w / 2) * mm,
        (page_h - bar_y + 6) * mm,
        f"Scale check: this bar must measure exactly {SCALE_BAR_MM:.0f} mm",
    )

    # Footer.
    c.setFont("Helvetica", 8)
    c.drawCentredString(
        (page_w / 2) * mm,
        (spec.margin_mm / 2) * mm,
        "gridfinity-cutter reference sheet - print at 100% / actual size, do not fit to page. "
        f"Markers: {spec.marker_ids[0]} TL, {spec.marker_ids[1]} TR, "
        f"{spec.marker_ids[2]} BR, {spec.marker_ids[3]} BL. Paper: {spec.paper.upper()}.",
    )

    c.showPage()
    c.save()
