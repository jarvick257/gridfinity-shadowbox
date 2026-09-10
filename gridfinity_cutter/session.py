"""Interactive session: one loaded photo, cached image stage, overlay and 3D scene.

Everything the web UI does lives here and is plain Python so it can be tested
without a browser; ``ui.py`` only wires widgets to these methods.

Cost model: marker detection, homography, warp and the
paper-colour model run once per photo (``load``). Image parameters re-run only
the threshold/contour part (``outline``, tens of ms). Geometry parameters
re-run offset + placement + extrude (``render``).
"""

from __future__ import annotations

import math
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import shapely
import trimesh
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from gridfinity_cutter import extrude, outline, sheet
from gridfinity_cutter.bins import BIN_GAP_MM, LIP_ZONE_MM
from gridfinity_cutter.params import ImageParams, Params
from gridfinity_cutter.sheet import DEFAULT_SPEC, SheetSpec

OVERLAY_MARGIN_MM = 12.0  # photo shown around the work area
DISPLAY_PX_PER_MM = 3.0
FIT_WALL_MM = 4.0  # free space kept between cutout and bin edge when auto-sizing
COLOR_OUTLINE = (40, 200, 60)  # RGB
COLOR_CLEARANCE = (170, 240, 120)
COLOR_BIN = (60, 120, 255)
COLOR_GRID = (150, 180, 255)
BIN_RGBA = (190, 190, 196, 255)


class SessionError(RuntimeError):
    """Raised when an action needs a state the session is not in."""


@dataclass(frozen=True)
class OutlineState:
    polygon_mm: np.ndarray  # sheet mm, y down (SVG plane)
    threshold: float  # threshold actually used (auto resolved)
    params: ImageParams


class Session:
    def __init__(self, spec: SheetSpec = DEFAULT_SPEC) -> None:
        self.spec = spec
        self.photo_path: Path | None = None
        self.px_per_mm: float | None = None
        self.warped: np.ndarray | None = None
        self.features: outline.PaperFeatures | None = None
        self.reprojection_error_mm: float | None = None
        self.workdir = Path(tempfile.mkdtemp(prefix="gridfinity-cutter-"))
        self._outline: OutlineState | None = None
        self._lock = threading.Lock()

    # -- image stage ----------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self.warped is not None

    def load(self, image_path: str | Path, px_per_mm: float) -> None:
        """Rectify the photo and cache everything that does not depend on parameters."""
        image_path = Path(image_path)
        with self._lock:
            if self.photo_path == image_path and self.px_per_mm == px_per_mm:
                return
            warped, err = outline.rectify(image_path, self.spec, px_per_mm)
            self.warped = warped
            self.features = outline.paper_features(warped, self.spec, px_per_mm)
            self.reprojection_error_mm = err
            self.photo_path = image_path
            self.px_per_mm = px_per_mm
            self._outline = None

    def _require_loaded(self) -> tuple[np.ndarray, outline.PaperFeatures, float]:
        if self.warped is None or self.features is None or self.px_per_mm is None:
            raise SessionError("load a photo first")
        return self.warped, self.features, self.px_per_mm

    def outline(self, image: ImageParams) -> OutlineState:
        """Outline for these image parameters; cached while they do not change."""
        with self._lock:
            _, feat, px_per_mm = self._require_loaded()
            if image.px_per_mm != px_per_mm:
                raise SessionError(
                    f"session was loaded at {px_per_mm:g} px/mm; call load() to change it"
                )
            key = (image.threshold, image.tolerance_mm, image.min_area_mm2)
            cur = self._outline
            if (
                cur is not None
                and (cur.params.threshold, cur.params.tolerance_mm, cur.params.min_area_mm2) == key
            ):
                return cur
            mask, thr = outline.mask_from_features(feat, px_per_mm, image.threshold)
            poly = outline.extract_outline(
                mask, self.spec, px_per_mm, image.tolerance_mm, image.min_area_mm2
            )
            self._outline = OutlineState(polygon_mm=poly, threshold=thr, params=image)
            return self._outline

    # -- geometry stage -------------------------------------------------------

    def offset_geometry(self, params: Params) -> Polygon | MultiPolygon:
        """Outline with clearance applied, still in the SVG plane (sheet mm, y down)."""
        state = self.outline(params.image)
        return extrude.offset_polygon(
            extrude.from_polygon(state.polygon_mm), params.geometry.clearance_mm
        )

    def fit_bin(self, params: Params) -> tuple[int, int]:
        """Smallest bin (units x, y) that holds the rotated, offset outline with a wall."""
        geom = extrude.flip_y(self.offset_geometry(params), params.geometry.mirror)
        rotated = shapely.affinity.rotate(geom, params.bin.rotation_deg, origin="center")
        minx, miny, maxx, maxy = rotated.bounds
        w = maxx - minx + 2 * FIT_WALL_MM + 2 * abs(params.bin.offset_x_mm)
        d = maxy - miny + 2 * FIT_WALL_MM + 2 * abs(params.bin.offset_y_mm)
        g = params.bin.grid_mm
        return (max(1, math.ceil(w / g)), max(1, math.ceil(d / g)))

    def build(self, params: Params) -> tuple[Polygon | MultiPolygon, trimesh.Trimesh]:
        """Placed outline in bin coordinates plus the finished bin mesh."""
        geom = self.offset_geometry(params)
        return extrude.build_in_bin(
            geom, params.geometry.height_mm, params.bin, params.geometry.mirror
        )

    def _photo_to_bin_matrix(self, params: Params) -> np.ndarray:
        """3x3 matrix from sheet mm (y down) to bin coordinates (y up)."""
        geom = self.offset_geometry(params)
        flip = np.diag([1.0, 1.0, 1.0]) if params.geometry.mirror else np.diag([1.0, -1.0, 1.0])
        place = extrude.placement_matrix(extrude.apply_matrix(geom, flip), params.bin)
        return place @ flip

    def bin_footprint_mm(self, params: Params) -> tuple[Polygon, list[LineString]]:
        """Bin rectangle and its grid lines drawn back onto the sheet (mm, y down)."""
        w, d, _ = params.bin.size_mm
        g = params.bin.grid_mm
        inv = np.linalg.inv(self._photo_to_bin_matrix(params))
        rect = extrude.apply_matrix(box(0.0, 0.0, w, d), inv)
        lines = []
        for i in range(1, params.bin.units_x):
            lines.append(extrude.apply_matrix(LineString([(i * g, 0), (i * g, d)]), inv))
        for j in range(1, params.bin.units_y):
            lines.append(extrude.apply_matrix(LineString([(0, j * g), (w, j * g)]), inv))
        return rect, lines

    # -- outputs --------------------------------------------------------------

    def _crop_mm(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = sheet.work_area_mm(self.spec)
        pw, ph = self.spec.page_mm
        m = OVERLAY_MARGIN_MM
        return (max(0.0, x0 - m), max(0.0, y0 - m), min(pw, x1 + m), min(ph, y1 + m))

    def overlay(self, params: Params, display_px_per_mm: float = DISPLAY_PX_PER_MM) -> np.ndarray:
        """RGB image of the photo around the work area with outline and bin drawn on it."""
        warped, _, px_per_mm = self._require_loaded()
        cx0, cy0, cx1, cy1 = self._crop_mm()
        crop = warped[
            round(cy0 * px_per_mm) : round(cy1 * px_per_mm),
            round(cx0 * px_per_mm) : round(cx1 * px_per_mm),
        ]
        s = display_px_per_mm
        size = (max(1, round((cx1 - cx0) * s)), max(1, round((cy1 - cy0) * s)))
        img = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        def px(coords) -> np.ndarray:
            pts = (np.asarray(coords, dtype=np.float64) - [cx0, cy0]) * s
            return np.round(pts).astype(np.int32).reshape(-1, 1, 2)

        state = self.outline(params.image)
        try:
            rect, lines = self.bin_footprint_mm(params)
            for ln in lines:
                cv2.polylines(img, [px(ln.coords)], False, COLOR_GRID, 1, cv2.LINE_AA)
            cv2.polylines(img, [px(rect.exterior.coords)], True, COLOR_BIN, 2, cv2.LINE_AA)
            if params.geometry.clearance_mm != 0.0:
                geom = self.offset_geometry(params)
                for part in getattr(geom, "geoms", [geom]):
                    cv2.polylines(
                        img, [px(part.exterior.coords)], True, COLOR_CLEARANCE, 1, cv2.LINE_AA
                    )
        except extrude.ExtrudeError:
            pass  # e.g. clearance collapsed the outline; still show the raw outline
        cv2.polylines(img, [px(state.polygon_mm)], True, COLOR_OUTLINE, 2, cv2.LINE_AA)
        return img

    def scene(self, params: Params) -> tuple[trimesh.Scene, trimesh.Trimesh]:
        """Viewer scene: the finished bin with the pocket (what the STL contains)."""
        _, mesh = self.build(params)
        solid = mesh.copy()
        solid.visual = trimesh.visual.ColorVisuals(
            solid, vertex_colors=np.tile(BIN_RGBA, (len(solid.vertices), 1))
        )
        scene = trimesh.Scene()
        scene.add_geometry(solid, geom_name="bin")
        return scene, mesh

    def render_scene(self, params: Params) -> tuple[Path, trimesh.Trimesh]:
        """Write the viewer scene as GLB (fixed path per session) and return it with the mesh."""
        scene, mesh = self.scene(params)
        path = self.workdir / "scene.glb"
        path.write_bytes(scene.export(file_type="glb"))
        return path, mesh

    def render(self, params: Params) -> Path:
        return self.render_scene(params)[0]

    def export(self, params: Params, out_dir: str | Path, stem: str) -> list[Path]:
        """Write ``<stem>.svg``, ``<stem>.stl`` (bin coordinates) and ``<stem>.params.json``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        state = self.outline(params.image)
        _, mesh = self.build(params)
        svg, stl, js = out / f"{stem}.svg", out / f"{stem}.stl", out / f"{stem}.params.json"
        photo = self.photo_path.name if self.photo_path else None
        outline.write_svg(
            svg,
            state.polygon_mm,
            f"gridfinity-cutter outline of {photo}; px_per_mm={params.image.px_per_mm:g} "
            f"tolerance_mm={params.image.tolerance_mm:g} threshold={state.threshold:.1f}",
        )
        mesh.export(str(stl))
        Params(params.image, params.geometry, params.bin, photo=photo).to_json(js)
        return [svg, stl, js]

    def status(self, params: Params, mesh: trimesh.Trimesh | None = None) -> str:
        state = self.outline(params.image)
        w, h = state.polygon_mm.max(axis=0) - state.polygon_mm.min(axis=0)
        thr = "auto" if params.image.threshold is None else "fixed"
        lines = [
            f"marker reprojection error {self.reprojection_error_mm:.2f} mm"
            + (
                " (check print scale / flatness)"
                if (self.reprojection_error_mm or 0) > outline.REPROJECTION_WARN_MM
                else ""
            ),
            (
                f"threshold {state.threshold:.0f} ({thr}), outline {w:.1f} x {h:.1f} mm, "
                f"{len(state.polygon_mm)} points"
            ),
        ]
        b = params.bin
        bw, bd, bh = b.size_mm
        top = b.pocket_top_mm
        depth = params.geometry.height_mm
        lines.append(
            f"bin {b.units_x} x {b.units_y} u ({b.grid_mm:g} mm grid), "
            f"{bw:g} x {bd:g} x {bh:g} mm excl. lip{' + lip' if b.has_lip else ''}, "
            f"pocket {top - depth:g}..{top:g} mm"
        )
        if depth > b.infill_height_mm:
            lines.append(
                "WARNING: pocket deeper than the solid part of the bin (cuts into the base)"
            )
        if self._near_wall(params):
            lines.append(
                f"WARNING: cutout within {LIP_ZONE_MM:g} mm of the bin wall (cuts the wall/lip)"
            )
        if params.geometry.mirror:
            lines.append(
                "mirror: bin footprint drawn as seen from below; +offset Y moves the object down on the photo"
            )
        if mesh is not None:
            lines.append(f"STL: {len(mesh.faces)} faces, {mesh.volume:.0f} mm^3")
        return "  \n".join(lines)  # trailing double space = Markdown line break

    def _near_wall(self, params: Params) -> bool:
        """True when the placed (offset) outline comes closer than the lip zone to the bin body."""
        geom = extrude.place_in_bin(
            extrude.flip_y(self.offset_geometry(params), params.geometry.mirror), params.bin
        )
        minx, miny, maxx, maxy = geom.bounds
        w, d, _ = params.bin.size_mm
        edge = BIN_GAP_MM / 2 + LIP_ZONE_MM
        return minx < edge or miny < edge or maxx > w - edge or maxy > d - edge
