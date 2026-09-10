"""Step 2: outline SVG in millimetres -> extruded "cutter" solid (STL).

The output is the positive shape of the object's footprint, grown by an optional
clearance, extruded from z = 0 to ``height_mm``. Subtract it from a Gridfinity bin
(or any block) in your CAD tool or slicer to get the pocket.

SVG contract
------------
Step 1 writes mm units, one closed ``M/L/Z`` path and no transforms. This module
accepts a superset so hand-drawn files (Inkscape etc.) work too:

* every closed ``<path>`` subpath plus ``rect``/``circle``/``ellipse``/``polygon``;
* curves and arcs, flattened to ``curve_tolerance_mm``;
* ``transform`` attributes and physical units (``mm``, ``in``, ``px`` at 96 dpi);
  a document without physical units is read as 1 user unit = 1 mm;
* a closed ring lying inside another ring is a hole.

Open subpaths are ignored (an error is raised if nothing closed remains).

Coordinate convention
---------------------
SVG y grows downwards, CAD/STL y grows upwards. The mesh is built with y flipped
so that looking down on the STL (from +z) shows the object exactly as it was
photographed. ``mirror=True`` skips the flip, for objects that go into the bin
upside down. The mesh is translated so its bounding box starts at (0, 0, 0).

Clearance
---------
``clearance_mm`` offsets the outline uniformly (round joins). Positive values make
a looser pocket; negative values shrink the outline, which also compensates the
parallax of a tall object photographed from close range (see the outline step).

Finger relief
-------------
``ReliefParams`` merges round scallops into the outline at its edge (full pocket
depth) so a finger can reach the object's side and lift it out. Applied after the
clearance in the SVG plane, so it rotates/mirrors with the object.

Bin placement
-------------
With ``bin=BinParams(...)`` the solid is written in *bin coordinates* instead:
the outline is rotated and moved inside a Gridfinity bin of the given size and
the pocket is sunk into the top of the bin's solid part (see
``gridfinity_shadowbox.bins``); the STL is then the finished Gridfinity bin with
the pocket cut out (``bins.geometry``).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
import trimesh
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from svgelements import SVG, Close, Line, Move
from svgelements import Path as SvgPath
from svgelements import Shape as SvgShape

from gridfinity_shadowbox.bins import BinParams

DEFAULT_CLEARANCE_MM = 0.0
DEFAULT_CURVE_TOLERANCE_MM = 0.1
MIN_AREA_MM2 = 1.0
SIMPLIFY_MM = 0.02
BUFFER_QUAD_SEGS = 16
# svgelements converts mm with the rounded factor 0.0393701 in/mm; using its exact
# reciprocal as the ppi makes 1 mm == 1 user unit with no rounding error.
_SVGELEMENTS_MM_PPI = 1.0 / 0.0393701


@dataclass(frozen=True)
class ReliefParams:
    """Round scallops merged into the outline edge so fingers can grip the object.

    ``angle_deg`` is measured from the outline's bounding-box centre in the SVG
    plane (0 = right, 90 = down on the photo); ``count`` scallops are spaced
    evenly around. ``inset_mm`` moves the circle centre from the outline edge
    towards the centre (0 = centred on the edge).
    """

    enabled: bool = False
    diameter_mm: float = 20.0
    count: int = 1
    angle_deg: float = 0.0
    inset_mm: float = 5.0


class ExtrudeError(RuntimeError):
    """Raised when the SVG cannot be turned into a solid."""


@dataclass(frozen=True)
class ExtrudeResult:
    size_mm: tuple[float, float, float]
    area_mm2: float
    volume_mm3: float
    n_faces: int
    n_parts: int


FLATTEN_STEP_MM = 0.1  # dense sampling; simplified to ``curve_tolerance_mm`` afterwards


def _flatten_segment(seg, tolerance_mm: float) -> list[tuple[float, float]]:
    """Points along ``seg`` excluding its start point.

    Straight segments contribute their end point only. Curves are sampled about
    every ``FLATTEN_STEP_MM`` and then thinned so that no point deviates more
    than ``tolerance_mm`` from the sampled curve. Only the curve itself is
    simplified; straight segments elsewhere in the path are never touched.
    """
    if isinstance(seg, (Line, Close)):
        return [(float(seg.end.x), float(seg.end.y))]
    length = seg.length(error=1e-4)
    n = max(4, math.ceil(length / FLATTEN_STEP_MM))
    pts = np.asarray(seg.npoint(np.linspace(0.0, 1.0, n + 1)), dtype=np.float64)
    if tolerance_mm > 0:
        pts = np.asarray(shapely.LineString(pts).simplify(tolerance_mm).coords)
    return [(float(x), float(y)) for x, y in pts[1:]]


def _rings_from_path(path: SvgPath, tolerance_mm: float) -> tuple[list[list[tuple]], int]:
    """Closed rings (lists of (x, y) mm) in a path, plus the number of open subpaths skipped."""
    rings: list[list[tuple]] = []
    n_open = 0
    for sub in path.as_subpaths():
        segs = list(sub)
        if not segs:
            continue
        closed = any(isinstance(s, Close) for s in segs)
        pts: list[tuple[float, float]] = []
        for s in segs:
            if isinstance(s, Move):
                pts.append((float(s.end.x), float(s.end.y)))
            else:
                pts.extend(_flatten_segment(s, tolerance_mm))
        if not closed:
            # Tolerate paths whose last point returns to the start without ``Z``.
            if len(pts) >= 4 and math.dist(pts[0], pts[-1]) < 1e-6:
                closed = True
            else:
                n_open += 1
                continue
        if len(pts) >= 3:
            rings.append(pts)
    return rings, n_open


def _assemble(rings: list[list[tuple]]) -> Polygon | MultiPolygon:
    """Rings -> geometry. A ring inside an accepted outer ring is a hole."""
    polys = []
    for r in rings:
        p = shapely.make_valid(Polygon(r))
        for part in getattr(p, "geoms", [p]):
            if isinstance(part, Polygon) and part.area >= MIN_AREA_MM2:
                polys.append(Polygon(part.exterior))
    polys.sort(key=lambda p: p.area, reverse=True)
    outers: list[Polygon] = []
    holes: list[Polygon] = []
    for p in polys:
        if any(o.contains(p) for o in outers):
            holes.append(p)
        else:
            outers.append(p)
    geom = shapely.unary_union(outers)
    if holes:
        geom = geom.difference(shapely.unary_union(holes))
    return geom


def load_polygons(
    svg_path: str | Path, curve_tolerance_mm: float = DEFAULT_CURVE_TOLERANCE_MM
) -> Polygon | MultiPolygon:
    """Read every closed shape in the SVG as shapely geometry in mm (SVG axes, y down).

    Curves are sampled densely and thinned so that no point deviates more than
    ``curve_tolerance_mm`` from the curve. Straight segments are kept verbatim.
    """
    try:
        # 1 px == 1 mm: unit-less documents are read as mm and documents with a
        # physical width/height come out correctly scaled.
        svg = SVG.parse(str(svg_path), ppi=_SVGELEMENTS_MM_PPI, reify=True)
    except Exception as e:  # svgelements raises assorted parse errors
        raise ExtrudeError(f"cannot parse {svg_path}: {e}") from e
    rings: list[list[tuple]] = []
    n_open = 0
    for el in svg.elements():
        if not isinstance(el, SvgShape):
            continue
        path = SvgPath(el)
        path.reify()
        r, o = _rings_from_path(path, curve_tolerance_mm)
        rings.extend(r)
        n_open += o
    geom = _assemble(rings) if rings else shapely.Polygon()
    if geom.is_empty:
        what = f"{n_open} open path(s)" if n_open else "no shapes"
        raise ExtrudeError(
            f"no closed outline with area >= {MIN_AREA_MM2:g} mm^2 in {svg_path} (found {what}). "
            "Close the path (Z) or check that the SVG has mm units."
        )
    return geom


def offset_polygon(geom: Polygon | MultiPolygon, clearance_mm: float) -> Polygon | MultiPolygon:
    """Grow (positive) or shrink (negative) the outline uniformly, round joins."""
    if clearance_mm != 0.0:
        n_before = len(getattr(geom, "geoms", [geom]))
        geom = geom.buffer(clearance_mm, join_style="round", quad_segs=BUFFER_QUAD_SEGS)
        if geom.is_empty:
            raise ExtrudeError(f"clearance {clearance_mm:g} mm collapsed the outline to nothing")
        n_after = len(getattr(geom, "geoms", [geom]))
        if clearance_mm < 0 and n_after > n_before:
            print(
                f"warning: clearance {clearance_mm:g} mm split the outline into {n_after} parts",
                file=sys.stderr,
            )
    return geom.simplify(SIMPLIFY_MM, preserve_topology=True)


def add_relief(geom: Polygon | MultiPolygon, relief: ReliefParams | None) -> Polygon | MultiPolygon:
    """Union finger-relief circles into the outline (SVG plane, y down)."""
    if relief is None or not relief.enabled:
        return geom
    if relief.diameter_mm <= 0 or relief.count < 1:
        raise ExtrudeError("finger relief needs a positive diameter and count >= 1")
    minx, miny, maxx, maxy = geom.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    reach = math.hypot(maxx - minx, maxy - miny)
    boundary = geom.boundary
    for k in range(relief.count):
        a = math.radians(relief.angle_deg + k * 360.0 / relief.count)
        dx, dy = math.cos(a), math.sin(a)
        ray = LineString([(cx, cy), (cx + dx * reach, cy + dy * reach)])
        hits = ray.intersection(boundary)
        pts = [g for g in getattr(hits, "geoms", [hits]) if isinstance(g, Point)]
        if not pts:
            raise ExtrudeError(f"finger relief at {math.degrees(a):g} deg does not hit the outline")
        hit = max(pts, key=lambda p: (p.x - cx) ** 2 + (p.y - cy) ** 2)
        centre = Point(hit.x - dx * relief.inset_mm, hit.y - dy * relief.inset_mm)
        geom = geom.union(centre.buffer(relief.diameter_mm / 2, quad_segs=BUFFER_QUAD_SEGS))
    return geom.simplify(SIMPLIFY_MM, preserve_topology=True)


def from_polygon(points_mm: np.ndarray) -> Polygon | MultiPolygon:
    """(N, 2) closed polygon in mm (SVG axes, y down) -> shapely geometry.

    Same validity and minimum-area rules as ``load_polygons`` so that the
    in-memory path (UI) and the SVG path (CLI) agree.
    """
    pts = [(float(x), float(y)) for x, y in np.asarray(points_mm, dtype=np.float64)]
    geom = _assemble([pts]) if len(pts) >= 3 else shapely.Polygon()
    if geom.is_empty:
        raise ExtrudeError(f"polygon has no area >= {MIN_AREA_MM2:g} mm^2")
    return geom


def flip_y(geom: Polygon | MultiPolygon, mirror: bool = False) -> Polygon | MultiPolygon:
    """SVG plane (y down) -> CAD plane (y up), unless ``mirror`` (see module docstring)."""
    if mirror:
        return geom
    return shapely.affinity.scale(geom, xfact=1.0, yfact=-1.0, origin=(0, 0))


def extrude_geometry(
    geom: Polygon | MultiPolygon, height_mm: float, z0: float = 0.0
) -> trimesh.Trimesh:
    """Extrude in place (no x/y translation) from ``z0`` to ``z0 + height_mm``."""
    if height_mm <= 0:
        raise ExtrudeError("height must be positive")
    parts = [
        trimesh.creation.extrude_polygon(p, height_mm)
        for p in getattr(geom, "geoms", [geom])
        if not p.is_empty
    ]
    if not parts:
        raise ExtrudeError("nothing to extrude")
    mesh = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
    if mesh.volume < 0:
        mesh.invert()
    if mesh.volume <= 0:
        raise ExtrudeError("extrusion produced a mesh with non-positive volume")
    if z0:
        mesh.apply_translation((0.0, 0.0, z0))
    return mesh


def to_mesh(
    geom: Polygon | MultiPolygon, height_mm: float, mirror: bool = False
) -> trimesh.Trimesh:
    """Extrude to a solid from z = 0 to ``height_mm``, bounding box starting at (0, 0)."""
    geom = flip_y(geom, mirror)
    minx, miny, _, _ = geom.bounds
    geom = shapely.affinity.translate(geom, xoff=-minx, yoff=-miny)
    return extrude_geometry(geom, height_mm)


def placement_matrix(geom_cad: Polygon | MultiPolygon, bin: BinParams) -> np.ndarray:
    """3x3 homogeneous matrix placing a CAD-plane outline inside the bin.

    Rotates by ``bin.rotation_deg`` (counter-clockwise, y up) about the
    outline's bounding-box centre, then moves that centre to bin centre +
    offset. The result does not depend on where the outline sits beforehand.
    This is the single definition of the placement; the UI overlay applies its
    inverse to draw the bin on the photo.
    """
    minx, miny, maxx, maxy = geom_cad.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    bx, by = bin.centre_mm
    tx, ty = bx + bin.offset_x_mm, by + bin.offset_y_mm
    a = math.radians(bin.rotation_deg)
    c, s = math.cos(a), math.sin(a)
    to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    to_target = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    return to_target @ rot @ to_origin


def apply_matrix(geom, m: np.ndarray):
    """Apply a 3x3 homogeneous matrix to any shapely geometry."""
    return shapely.affinity.affine_transform(
        geom, [m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[0, 2], m[1, 2]]
    )


def place_in_bin(geom_cad: Polygon | MultiPolygon, bin: BinParams) -> Polygon | MultiPolygon:
    """Move a CAD-plane outline into bin coordinates (see ``placement_matrix``)."""
    return apply_matrix(geom_cad, placement_matrix(geom_cad, bin))


def build_in_bin(
    geom: Polygon | MultiPolygon, height_mm: float, bin: BinParams, mirror: bool = False
) -> tuple[Polygon | MultiPolygon, trimesh.Trimesh]:
    """Offset outline (SVG plane) -> placed outline in bin coordinates + finished bin mesh."""
    # bins.geometry uses extrude_geometry, so it cannot be imported at module level.
    from gridfinity_shadowbox.bins.geometry import bin_with_pocket

    placed = place_in_bin(flip_y(geom, mirror), bin)
    return placed, bin_with_pocket(placed, height_mm, bin)


def _result(geom, mesh: trimesh.Trimesh) -> ExtrudeResult:
    size = tuple(float(v) for v in mesh.extents)
    return ExtrudeResult(
        size_mm=size,  # type: ignore[arg-type]
        area_mm2=float(geom.area),
        volume_mm3=float(mesh.volume),
        n_faces=len(mesh.faces),
        n_parts=len(getattr(geom, "geoms", [geom])),
    )


def run(
    svg: str | Path,
    output: str | Path,
    *,
    height_mm: float,
    clearance_mm: float = DEFAULT_CLEARANCE_MM,
    mirror: bool = False,
    curve_tolerance_mm: float = DEFAULT_CURVE_TOLERANCE_MM,
    bin: BinParams | None = None,
    relief: ReliefParams | None = None,
) -> ExtrudeResult:
    """SVG -> STL. With ``bin`` the solid is in bin coordinates (see ``bins``)."""
    geom = offset_polygon(load_polygons(svg, curve_tolerance_mm), clearance_mm)
    geom = add_relief(geom, relief)
    if bin is None:
        mesh = to_mesh(geom, height_mm, mirror=mirror)
    else:
        _, mesh = build_in_bin(geom, height_mm, bin, mirror)
    mesh.export(str(output))
    return _result(geom, mesh)


if __name__ == "__main__":
    from gridfinity_shadowbox.cli import main

    sys.exit(main(["extrude", *sys.argv[1:]]))
