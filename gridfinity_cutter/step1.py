"""Step 1: photo of an object on the reference sheet -> outline SVG in millimetres.

Pipeline: detect ArUco markers -> homography photo px -> sheet mm -> warp the
photo onto a flat, scaled sheet image -> crop to the work area (this discards
markers, labels and the scale bar) -> foreground mask by colour distance from
the paper -> largest external contour not touching the border -> simplify ->
SVG with ``viewBox`` in mm (1 user unit = 1 mm), one closed path, no transforms.

Known limitations
-----------------
* Parallax: a tall object photographed from close range shows its top face
  larger than its footprint. Shoot from at least ~60 cm straight down, or
  compensate with a negative clearance in step 2.
* Shiny, white or very light objects may be under-segmented. ``--threshold``
  and ``--debug DIR`` are the escape hatches; the SVG can also be hand-edited.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gridfinity_cutter import sheet
from gridfinity_cutter.sheet import DEFAULT_SPEC, SheetSpec

DEFAULT_PX_PER_MM = 10.0
DEFAULT_TOLERANCE_MM = 0.2
DEFAULT_MIN_AREA_MM2 = 100.0
PAPER_SAMPLE_STRIP_MM = 2.0
CROP_INSET_MM = 3.0  # keep the dashed work-area outline out of the crop
AUTO_THRESHOLD_MIN, AUTO_THRESHOLD_MAX, AUTO_THRESHOLD_STEP = 15.0, 90.0, 5.0
AUTO_THRESHOLD_PLATEAU = 0.01  # relative area change per step that counts as "stable"
REPROJECTION_WARN_MM = 1.5
SVG_MARGIN_MM = 5.0


class OutlineError(RuntimeError):
    """Raised when the photo cannot be turned into an outline."""


@dataclass(frozen=True)
class OutlineResult:
    polygon_mm: np.ndarray  # (N, 2) closed polygon in sheet mm
    reprojection_error_mm: float
    threshold: float

    @property
    def bbox_mm(self) -> tuple[float, float, float, float]:
        (x0, y0), (x1, y1) = self.polygon_mm.min(axis=0), self.polygon_mm.max(axis=0)
        return (float(x0), float(y0), float(x1), float(y1))

    @property
    def size_mm(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_mm
        return (x1 - x0, y1 - y0)


def detect_markers(img: np.ndarray, spec: SheetSpec = DEFAULT_SPEC) -> dict[int, np.ndarray]:
    """Return {marker id: (4, 2) float32 pixel corners, TL TR BR BL} for the sheet markers."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    detector = cv2.aruco.ArucoDetector(sheet.aruco_dictionary(spec), params)
    corners, ids, _ = detector.detectMarkers(img)
    found = {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
    missing = [mid for mid in spec.marker_ids if mid not in found]
    if missing:
        raise OutlineError(
            f"could not find marker id(s) {missing}; found {sorted(found)}. "
            "Make sure all four markers are fully visible, in focus and unobstructed."
        )
    return {mid: found[mid] for mid in spec.marker_ids}


def compute_homography(
    px_by_id: dict[int, np.ndarray], spec: SheetSpec = DEFAULT_SPEC
) -> tuple[np.ndarray, float]:
    """Homography mapping photo pixels to sheet mm, plus max reprojection error in mm."""
    mm_by_id = sheet.marker_corners_mm(spec)
    src = np.concatenate([px_by_id[mid] for mid in spec.marker_ids]).astype(np.float64)
    dst = np.concatenate([mm_by_id[mid] for mid in spec.marker_ids])
    h, _ = cv2.findHomography(src, dst)
    if h is None:
        raise OutlineError("homography estimation failed")
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h).reshape(-1, 2)
    err = float(np.linalg.norm(proj - dst, axis=1).max())
    return h, err


def warp_to_sheet(
    img: np.ndarray,
    h: np.ndarray,
    spec: SheetSpec = DEFAULT_SPEC,
    px_per_mm: float = DEFAULT_PX_PER_MM,
) -> np.ndarray:
    """Warp the photo so it covers the whole page at ``px_per_mm``, tilt removed."""
    w, hgt = spec.page_mm
    scale = np.diag([px_per_mm, px_per_mm, 1.0])
    size = (round(w * px_per_mm), round(hgt * px_per_mm))
    return cv2.warpPerspective(img, scale @ h, size, flags=cv2.INTER_LINEAR)


def _work_area_px(spec: SheetSpec, px_per_mm: float) -> tuple[int, int, int, int]:
    """Work-area crop in warped-image pixels, inset so the dashed outline is excluded."""
    x0, y0, x1, y1 = sheet.work_area_mm(spec)
    inset = CROP_INSET_MM
    vals = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
    return tuple(round(v * px_per_mm) for v in vals)  # type: ignore[return-value]


def _paper_model(lab: np.ndarray, strip_px: int) -> np.ndarray:
    """Per-pixel Lab colour of the bare paper, as a quadratic surface in x and y.

    Fitted (robustly, one outlier-rejection pass) on a strip along the crop
    border, which is assumed to be paper. A smooth surface instead of a single
    colour absorbs vignetting and uneven lighting across the sheet.
    """
    h, w = lab.shape[:2]
    border = np.zeros((h, w), bool)
    border[:strip_px] = border[-strip_px:] = True
    border[:, :strip_px] = border[:, -strip_px:] = True
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = x / w, y / h
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)

    a = design(xx[border], yy[border])
    b = lab[border]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(a), min(len(a), 50_000), replace=False)
    a, b = a[idx], b[idx]
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    resid = np.linalg.norm(b - a @ coef, axis=1)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) * 1.4826
    keep = resid <= med + 3 * mad + 1e-3
    coef, *_ = np.linalg.lstsq(a[keep], b[keep], rcond=None)
    return design(xx, yy) @ coef


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(
        flood, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 255
    )
    return (padded | cv2.bitwise_not(flood))[1:-1, 1:-1]


def _ellipse(diameter_mm: float, px_per_mm: float) -> np.ndarray:
    d = max(3, round(diameter_mm * px_per_mm) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _mask_at(dist: np.ndarray, edges: np.ndarray, threshold: float, px_per_mm: float) -> np.ndarray:
    mask = (dist > threshold).astype(np.uint8) * 255
    mask |= edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(1.5, px_per_mm))
    mask = _fill_holes(mask)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(0.7, px_per_mm))


def _touches_border(contour: np.ndarray, shape: tuple[int, ...]) -> bool:
    x, y, w, h = cv2.boundingRect(contour)
    return x <= 0 or y <= 0 or x + w >= shape[1] or y + h >= shape[0]


def _largest_area(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(
        (cv2.contourArea(c) for c in contours if not _touches_border(c, mask.shape)),
        default=0.0,
    )


def segment_object(
    warped: np.ndarray,
    spec: SheetSpec = DEFAULT_SPEC,
    px_per_mm: float = DEFAULT_PX_PER_MM,
    threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """Foreground mask (255 = object) of the work-area crop, plus the threshold used.

    Foreground is every pixel whose Lab distance from the modelled paper colour
    exceeds ``threshold``, OR'd with Canny edges so that shiny/light parts
    (e.g. a metal blade) are still enclosed; enclosed holes are then filled.

    With ``threshold=None`` the threshold is chosen automatically: the largest
    blob's area is measured for increasing thresholds and the first value where
    it stops shrinking is used. Soft shadows fade out with rising threshold
    while a hard-edged object keeps its area, so this lands just above the
    shadow.
    """
    x0, y0, x1, y1 = _work_area_px(spec, px_per_mm)
    crop = warped[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    strip = max(1, round(PAPER_SAMPLE_STRIP_MM * px_per_mm))
    dist = np.linalg.norm(lab - _paper_model(lab, strip), axis=2)

    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), _ellipse(0.3, px_per_mm))

    if threshold is None:
        steps = np.arange(AUTO_THRESHOLD_MIN, AUTO_THRESHOLD_MAX + 1, AUTO_THRESHOLD_STEP)
        areas = [_largest_area(_mask_at(dist, edges, float(t), px_per_mm)) for t in steps]
        threshold = float(steps[-1])
        for t, a0, a1 in zip(steps, areas, areas[1:]):
            if a0 > 0 and (a0 - a1) / a0 < AUTO_THRESHOLD_PLATEAU:
                threshold = float(t)
                break
    return _mask_at(dist, edges, float(threshold), px_per_mm), float(threshold)


def extract_outline(
    mask: np.ndarray,
    spec: SheetSpec = DEFAULT_SPEC,
    px_per_mm: float = DEFAULT_PX_PER_MM,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
    min_area_mm2: float = DEFAULT_MIN_AREA_MM2,
) -> np.ndarray:
    """Largest external contour not touching the crop border, simplified, in sheet mm."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for c in contours:
        if _touches_border(c, mask.shape):
            continue  # runs off the work area: shadow, clutter, the sheet edge
        area_mm2 = cv2.contourArea(c) / px_per_mm**2
        if area_mm2 >= min_area_mm2:
            candidates.append((area_mm2, c))
    if not candidates:
        raise OutlineError(
            f"no object found inside the work area (min area {min_area_mm2:g} mm^2). "
            "Try --threshold, --min-area or --debug DIR to inspect the mask."
        )
    _, best = max(candidates, key=lambda t: t[0])
    simplified = cv2.approxPolyDP(best, tolerance_mm * px_per_mm, closed=True).reshape(-1, 2)
    if len(simplified) < 3:
        raise OutlineError("outline degenerated to fewer than 3 points; lower --tolerance")
    x0, y0, _, _ = _work_area_px(spec, px_per_mm)
    return (simplified.astype(np.float64) + [x0, y0]) / px_per_mm


def write_svg(
    path: str | Path, polygon_mm: np.ndarray, comment: str | None = None
) -> tuple[float, float]:
    """Write one closed path in mm units. Returns the SVG (width, height) in mm.

    The polygon is translated so its bounding box starts at ``SVG_MARGIN_MM``;
    no ``transform`` attributes are used (see SVG contract in CLAUDE.md).
    """
    pts = np.asarray(polygon_mm, dtype=np.float64)
    lo = pts.min(axis=0) - SVG_MARGIN_MM
    pts = pts - lo
    w, h = (pts.max(axis=0) + SVG_MARGIN_MM).tolist()
    d = "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in pts) + " Z"
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    if comment:
        lines.append(f"<!-- {comment.replace('--', '-')} -->")
    lines += [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.3f}mm" height="{h:.3f}mm" '
            f'viewBox="0 0 {w:.3f} {h:.3f}">'
        ),
        f'  <path d="{d}" fill="none" stroke="black" stroke-width="0.2"/>',
        "</svg>",
        "",
    ]
    Path(path).write_text("\n".join(lines))
    return w, h


def run(
    image: str | Path,
    output: str | Path,
    *,
    spec: SheetSpec = DEFAULT_SPEC,
    px_per_mm: float = DEFAULT_PX_PER_MM,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
    threshold: float | None = None,
    min_area_mm2: float = DEFAULT_MIN_AREA_MM2,
    debug_dir: str | Path | None = None,
) -> OutlineResult:
    img = cv2.imread(str(image))
    if img is None:
        raise OutlineError(f"cannot read image {image}")
    markers = detect_markers(img, spec)
    h, err = compute_homography(markers, spec)
    if err > REPROJECTION_WARN_MM:
        print(
            f"warning: marker reprojection error {err:.2f} mm; check that the sheet was "
            "printed at 100% and lies flat",
            file=sys.stderr,
        )
    warped = warp_to_sheet(img, h, spec, px_per_mm)
    mask, thr = segment_object(warped, spec, px_per_mm, threshold)
    polygon = extract_outline(mask, spec, px_per_mm, tolerance_mm, min_area_mm2)

    if debug_dir is not None:
        dbg = Path(debug_dir)
        dbg.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg / "warp.png"), warped)
        cv2.imwrite(str(dbg / "mask.png"), mask)
        vis = warped.copy()
        cv2.polylines(vis, [(polygon * px_per_mm).astype(np.int32)], True, (0, 255, 0), 2)
        cv2.imwrite(str(dbg / "contour.png"), vis)

    comment = (
        f"gridfinity-cutter outline of {Path(image).name}; px_per_mm={px_per_mm:g} "
        f"tolerance_mm={tolerance_mm:g} threshold={thr:.1f} reprojection_error_mm={err:.2f}"
    )
    write_svg(output, polygon, comment)
    return OutlineResult(polygon_mm=polygon, reprojection_error_mm=err, threshold=thr)


if __name__ == "__main__":
    from gridfinity_cutter.cli import main

    sys.exit(main(["outline", *sys.argv[1:]]))
