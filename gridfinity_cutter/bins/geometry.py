"""The Gridfinity bin solid, built with manifold3d, and the pocket boolean.

Geometry follows the Gridfinity spec as encoded in Gridfinity Rebuilt's
``standard.scad`` (constants in :mod:`gridfinity_cutter.bins`): per-cell base
profile, bridge, solid body, optional stacking lip and base holes. Every piece is a
convex hull of rounded rectangles at two heights or a cylinder/box, so the whole
bin is a handful of booleans and renders in milliseconds. The bin (without the
pocket) is cached per option set; the pocket is subtracted per call.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache

import manifold3d as m3
import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon

from gridfinity_cutter import extrude
from gridfinity_cutter.bins import (
    BASE_HEIGHT_MM,
    BASE_PROFILE_HEIGHT_MM,
    BASE_PROFILE_MM,
    BASE_TOP_RADIUS_MM,
    BIN_GAP_MM,
    HOLE_CHAMFER_MM,
    HOLE_INSET_MM,
    LAYER_HEIGHT_MM,
    LIP_FILLET_RADIUS_MM,
    LIP_PROFILES_MM,
    MAGNET_HOLE_DEPTH_MM,
    MAGNET_HOLE_RADIUS_MM,
    SCREW_HOLE_RADIUS_MM,
    STACKING_LIP_SUPPORT_MM,
    WALL_MM,
    BinParams,
)

CIRCLE_SEGMENTS = 48  # for corner radii and holes
FILLET_SEGMENTS = 8  # points on the rounded lip tip
POCKET_OVERSHOOT_MM = 1.0  # cutter reaches above the solid top so no skin remains
CUT_OVERSHOOT_MM = 1.0  # negatives extend past the faces they open
SEAM_OVERLAP_MM = 0.5  # unioned pieces overlap by this much instead of touching
PRINTABLE_INNER_RADIUS_MM = 1.0  # bridged top of a magnet hole without a screw hole


class BinError(extrude.ExtrudeError):
    """The bin could not be built or the pocket boolean failed."""


# -- 2D helpers ------------------------------------------------------------


def rrect(w: float, d: float, r: float) -> m3.CrossSection:
    """Rounded rectangle ``w`` x ``d`` centred on the origin, corner radius ``r``."""
    r = min(r, w / 2, d / 2)
    core = m3.CrossSection.square((w - 2 * r, d - 2 * r), center=True)
    return core.offset(r, m3.JoinType.Round, circular_segments=CIRCLE_SEGMENTS) if r > 0 else core


def _points_3d(cs: m3.CrossSection, z: float) -> np.ndarray:
    pts = np.concatenate([np.asarray(poly, dtype=float) for poly in cs.to_polygons()])
    return np.column_stack([pts, np.full(len(pts), z)])


def loft(cs0: m3.CrossSection, z0: float, cs1: m3.CrossSection, z1: float) -> m3.Manifold:
    """Convex hull of two convex cross-sections at different heights (a swept chamfer)."""
    return m3.Manifold.hull_points(np.concatenate([_points_3d(cs0, z0), _points_3d(cs1, z1)]))


def prism(cs: m3.CrossSection, z0: float, z1: float) -> m3.Manifold:
    return m3.Manifold.extrude(cs, z1 - z0).translate((0, 0, z0))


def sweep(top: m3.CrossSection, profile: tuple[tuple[float, float], ...]) -> m3.Manifold:
    """Union of lofts between consecutive ``(inset, z)`` points of ``profile``.

    ``top`` is the outline at inset 0; insets are applied with round joins so
    corner radii shrink accordingly.
    """
    outlines = [
        top.offset(-inset, m3.JoinType.Round, circular_segments=CIRCLE_SEGMENTS)
        if inset > 1e-9
        else top
        for inset, _ in profile
    ]
    parts = [
        loft(outlines[i], profile[i][1], outlines[i + 1], profile[i + 1][1])
        for i in range(len(profile) - 1)
    ]
    return m3.Manifold.batch_boolean(parts, m3.OpType.Add)


# -- base -------------------------------------------------------------------


def base_cell(cell_mm: float) -> m3.Manifold:
    """One base unit (spec profile), centred on the origin, z 0..4.75."""
    top = rrect(cell_mm - BIN_GAP_MM, cell_mm - BIN_GAP_MM, BASE_TOP_RADIUS_MM)
    return sweep(top, BASE_PROFILE_MM)


def _printable_layers(
    inner_r: float, outer_r: float, top_z: float, layers: int
) -> list[m3.Manifold]:
    """Rebuilt's ``make_hole_printable``: the hole's top ``layers`` layers become
    progressively narrower bars (alternating 90 deg) ending in a square of the inner
    size, so the roof bridges without support. Returns the boxes to *cut*."""
    calc = max(layers - 1, 1)
    inner_d, outer_d = 2 * inner_r, 2 * outer_r
    step = (outer_d - inner_d) / calc
    z0 = top_z - layers * LAYER_HEIGHT_MM
    boxes = []
    for i in range(1, calc + 1):
        size = (outer_d - step * (i - 1), outer_d - step * i)
        if i % 2 == 0:
            size = (size[1], size[0])
        boxes.append(_box(size, z0 + (i - 1) * LAYER_HEIGHT_MM, LAYER_HEIGHT_MM))
    if layers > 1:
        boxes.append(_box((inner_d, inner_d), z0 + (layers - 1) * LAYER_HEIGHT_MM, LAYER_HEIGHT_MM))
    return boxes


def _box(size: tuple[float, float], z0: float, h: float) -> m3.Manifold:
    return m3.Manifold.cube((size[0], size[1], h), center=True).translate((0, 0, z0 + h / 2))


def _cylinder(r: float, z0: float, z1: float, r_top: float | None = None) -> m3.Manifold:
    return m3.Manifold.cylinder(
        z1 - z0, r, r if r_top is None else r_top, CIRCLE_SEGMENTS
    ).translate((0, 0, z0))


def hole_negative(p: BinParams) -> m3.Manifold | None:
    """One magnet/screw hole at the origin, opening downwards through z = 0."""
    parts: list[m3.Manifold] = []
    if p.magnet_holes:
        layers = 2 if p.screw_holes else 3
        depth = MAGNET_HOLE_DEPTH_MM + (layers * LAYER_HEIGHT_MM if p.printable_hole_top else 0)
        parts.append(_cylinder(MAGNET_HOLE_RADIUS_MM, -CUT_OVERSHOOT_MM, MAGNET_HOLE_DEPTH_MM))
        if p.printable_hole_top:
            inner = SCREW_HOLE_RADIUS_MM if p.screw_holes else PRINTABLE_INNER_RADIUS_MM
            bore = _cylinder(MAGNET_HOLE_RADIUS_MM, MAGNET_HOLE_DEPTH_MM, depth)
            parts += [
                box ^ bore for box in _printable_layers(inner, MAGNET_HOLE_RADIUS_MM, depth, layers)
            ]
        if p.chamfer_holes:
            parts.append(_cone(MAGNET_HOLE_RADIUS_MM, MAGNET_HOLE_DEPTH_MM))
    if p.screw_holes:
        # Rebuilt cuts holes from the base cells only, so the screw hole ends at the
        # bridge (4.75 mm) and its printable top never materialises.
        parts.append(_cylinder(SCREW_HOLE_RADIUS_MM, -CUT_OVERSHOOT_MM, BASE_PROFILE_HEIGHT_MM))
        if p.chamfer_holes:
            parts.append(_cone(SCREW_HOLE_RADIUS_MM, BASE_PROFILE_HEIGHT_MM))
    if not parts:
        return None
    return m3.Manifold.batch_boolean(parts, m3.OpType.Add)


def _cone(r: float, max_h: float) -> m3.Manifold:
    """Rebuilt's ``cone(r + 0.8, 45 deg, max_h)``: chamfer of the hole mouth at z = 0."""
    r0 = r + HOLE_CHAMFER_MM
    h = min(r0, max_h)
    return _cylinder(r0, 0.0, h, r0 - h) + _cylinder(r0, -CUT_OVERSHOOT_MM, 0.0)


def hole_centres(p: BinParams) -> list[tuple[float, float]]:
    """Hole positions in bin coordinates (from the bin's grid corner)."""
    w, d, _ = p.size_mm
    if p.only_corners or p.half_grid:
        ox, oy = (w - BIN_GAP_MM) / 2 - HOLE_INSET_MM, (d - BIN_GAP_MM) / 2 - HOLE_INSET_MM
        return [(w / 2 + sx * ox, d / 2 + sy * oy) for sx in (-1, 1) for sy in (-1, 1)]
    g = p.grid_mm
    o = (g - BIN_GAP_MM) / 2 - HOLE_INSET_MM
    return [
        (g * (i + 0.5) + sx * o, g * (j + 0.5) + sy * o)
        for i in range(p.units_x)
        for j in range(p.units_y)
        for sx in (-1, 1)
        for sy in (-1, 1)
    ]


# -- lip ---------------------------------------------------------------------

Profile = tuple[tuple[float, float], ...]


def lip_profiles(style: str) -> tuple[Profile, Profile]:
    """Inner and outer boundary of the lip material as ``(inset, z above nominal height)``.

    The spec profile ends in a knife edge at the outer wall; like Rebuilt we round
    that corner with ``LIP_FILLET_RADIUS_MM``. Both boundaries start at the bottom of
    the lip recess (``-STACKING_LIP_SUPPORT_MM``) and end at the topmost point.
    """
    line = LIP_PROFILES_MM[style]
    (i1, z1), (i2, z2) = line[-2], line[-1]
    r = LIP_FILLET_RADIUS_MM
    # corner at V = (i2, z2) between the last chamfer (towards A) and the wall (down)
    ax, az = i1 - i2, z1 - z2
    n = math.hypot(ax, az)
    ax, az = ax / n, az / n
    bx, bz = 0.0, -1.0
    cos_t = ax * bx + az * bz
    half = math.acos(max(-1.0, min(1.0, cos_t))) / 2
    t = r / math.tan(half)
    mx, mz = ax + bx, az + bz
    mn = math.hypot(mx, mz)
    cx, cz = i2 + mx / mn * r / math.sin(half), z2 + mz / mn * r / math.sin(half)
    ta = (i2 + ax * t, z2 + az * t)
    tb = (i2 + bx * t, z2 + bz * t)
    a0 = math.atan2(ta[1] - cz, ta[0] - cx)
    a1 = math.atan2(tb[1] - cz, tb[0] - cx)
    # sweep from A to B the way that passes over the top (positive z from the centre)
    if a1 < a0:
        a1 += 2 * math.pi
    if not (a0 < math.pi / 2 < a1):
        a0, a1 = a1, a0 + 2 * math.pi
    arc = [
        (
            cx + r * math.cos(a0 + (a1 - a0) * k / FILLET_SEGMENTS),
            cz + r * math.sin(a0 + (a1 - a0) * k / FILLET_SEGMENTS),
        )
        for k in range(FILLET_SEGMENTS + 1)
    ]
    arc[0], arc[-1] = ta, tb  # exact tangent points, no floating-point residue
    top_idx = max(range(len(arc)), key=lambda k: arc[k][1])
    inner_arc, outer_arc = arc[: top_idx + 1], arc[top_idx:]
    if inner_arc[0][0] < outer_arc[-1][0]:  # arc was generated wall-first
        inner_arc, outer_arc = list(reversed(outer_arc)), list(reversed(inner_arc))
    z_support = -(STACKING_LIP_SUPPORT_MM + line[0][0])  # 45 deg support below the recess
    inner = ((0.0, z_support), (line[0][0], -STACKING_LIP_SUPPORT_MM), *line[:-1], *inner_arc)
    outer = ((0.0, z_support), (0.0, tb[1]), *outer_arc)
    return tuple(inner), tuple(outer)


def _clamp(profile: Profile, z_min: float) -> Profile:
    """Move points below ``z_min`` up onto it and drop the zero-height segments."""
    out: list[tuple[float, float]] = []
    for inset, z in profile:
        pt = (inset, max(z, z_min))
        if not out or pt != out[-1]:
            out.append(pt)
    return tuple(out)


def lip_solid(
    footprint: m3.CrossSection, style: str, height_mm: float, wall_from_mm: float | None
) -> m3.Manifold:
    """The lip ring (and, if ``wall_from_mm`` is given, the outer wall below it
    starting at that z) in bin coordinates.

    Building the wall into the same sweep keeps the wall/support junction a profile
    vertex instead of a boolean intersection. On short bins the support would reach
    below the base top; like Rebuilt the profile is clamped there. The sweep starts
    ``SEAM_OVERLAP_MM`` inside the solid body so the union has no coplanar seam.
    """
    inner, outer = lip_profiles(style)
    z_support = outer[0][1]
    if wall_from_mm is not None and wall_from_mm < height_mm + z_support:
        z0 = wall_from_mm - height_mm - SEAM_OVERLAP_MM
        # the support line inset = z - z_support reaches the wall thickness here
        z_wall = z_support + WALL_MM
        inner = ((WALL_MM, z0), (WALL_MM, z_wall), *inner[1:])
        outer = ((0.0, z0), *outer[1:])
    z_min = BASE_HEIGHT_MM - height_mm
    if z_min > outer[0][1]:
        z0 = z_min - SEAM_OVERLAP_MM
        inner = ((0.0, z0), *_clamp(inner[1:], z_min))
        outer = ((0.0, z0), *_clamp(outer[1:], z_min))
    top_z = outer[-1][1]
    body = sweep(footprint, tuple((i, height_mm + z) for i, z in outer))
    air = sweep(
        footprint,
        tuple((i, height_mm + z) for i, z in inner) + ((inner[-1][0], height_mm + top_z + 1.0),),
    )
    return body - air


# -- bin ---------------------------------------------------------------------


def _bin_key(p: BinParams) -> BinParams:
    """Placement does not affect the bin itself."""
    return replace(p, offset_x_mm=0.0, offset_y_mm=0.0, rotation_deg=0.0)


@lru_cache(maxsize=8)
def _bin_solid(p: BinParams) -> m3.Manifold:
    w, d, height = p.size_mm
    g = p.grid_mm
    cell = base_cell(g)
    cells = [
        cell.translate((g * (i + 0.5), g * (j + 0.5), 0.0))
        for i in range(p.units_x)
        for j in range(p.units_y)
    ]
    footprint = rrect(w - BIN_GAP_MM, d - BIN_GAP_MM, BASE_TOP_RADIUS_MM).translate((w / 2, d / 2))
    # Overlap the pieces instead of letting them touch on coplanar faces, which
    # would leave duplicate vertices along the seams.
    cell_top = rrect(g - BIN_GAP_MM, g - BIN_GAP_MM, BASE_TOP_RADIUS_MM)
    cells += [
        prism(cell_top, BASE_PROFILE_HEIGHT_MM, BASE_HEIGHT_MM - SEAM_OVERLAP_MM).translate(
            (g * (i + 0.5), g * (j + 0.5), 0.0)
        )
        for i in range(p.units_x)
        for j in range(p.units_y)
    ]
    parts = cells + [prism(footprint, BASE_PROFILE_HEIGHT_MM, p.pocket_top_mm)]
    if p.has_lip:
        # With a lowered solid top the outer wall continues up to the lip (as in
        # Rebuilt); without a lip the solid *is* the bin and simply ends lower.
        parts.append(lip_solid(footprint, p.lip, height, p.pocket_top_mm))
    solid = m3.Manifold.batch_boolean(parts, m3.OpType.Add)

    cuts: list[m3.Manifold] = []
    hole = hole_negative(p)
    if hole is not None:
        cuts += [hole.translate((x, y, 0.0)) for x, y in hole_centres(p)]
    if cuts:
        solid = solid - m3.Manifold.batch_boolean(cuts, m3.OpType.Add)
    if solid.status() != m3.Error.NoError or solid.is_empty():
        raise BinError(f"bin construction failed ({solid.status()})")
    return solid


def bin_solid(p: BinParams) -> m3.Manifold:
    """The bin without the pocket, in bin coordinates (cached per option set)."""
    return _bin_solid(_bin_key(p))


def bin_mesh(p: BinParams) -> trimesh.Trimesh:
    return from_manifold(bin_solid(p))


# -- conversion ---------------------------------------------------------------


def to_manifold(mesh: trimesh.Trimesh, what: str) -> m3.Manifold:
    m = m3.Manifold(
        mesh=m3.Mesh(
            vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32),
        )
    )
    if m.status() != m3.Error.NoError or m.is_empty() or m.volume() <= 0:
        raise BinError(f"{what} is not a closed solid ({m.status()})")
    return m


def from_manifold(m: m3.Manifold) -> trimesh.Trimesh:
    out = m.to_mesh64()
    return trimesh.Trimesh(vertices=out.vert_properties[:, :3], faces=out.tri_verts, process=False)


# -- bin with pocket ----------------------------------------------------------


def bin_with_pocket(
    cutout: Polygon | MultiPolygon, height_mm: float, bin: BinParams
) -> trimesh.Trimesh:
    """The finished bin: the solid bin minus the object pocket.

    ``cutout`` is already placed in bin coordinates (mm, y up); the pocket is
    ``height_mm`` deep, sunk down from ``bin.pocket_top_mm``.
    """
    body = bin_solid(bin)
    top = bin.pocket_top_mm
    cutter = extrude.extrude_geometry(cutout, height_mm + POCKET_OVERSHOOT_MM, z0=top - height_mm)
    solid = body - to_manifold(cutter, "pocket cutter")
    if solid.is_empty() or not solid.volume() < body.volume():
        raise BinError("bin minus pocket produced an empty or unchanged mesh")
    return from_manifold(solid)
