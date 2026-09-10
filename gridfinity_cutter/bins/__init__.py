"""Gridfinity bin backends: build the solid that goes around (or is) the cutout.

This module is the only place that knows the Gridfinity grid constants and the
bin coordinate convention. It must not import other project modules at import
time (``extrude`` and ``params`` both import from here).

Bin coordinate convention
-------------------------
x and y are measured from the bin's bottom-left grid corner, z from the bin
bottom. A bin of ``units_x`` by ``units_y`` grid cells occupies
``[0, grid * units_x] x [0, grid * units_y]`` mm (``grid`` is 42 mm, or 21 mm
for half-grid bins; the bin body is ``BIN_GAP_MM`` smaller than the grid cell)
and is ``height_mm`` tall excluding the stacking lip. The solid part of the bin
ends at ``pocket_top_mm`` (below the lip support when a lip is present); the
cutout is placed inside the footprint by ``extrude.place_in_bin`` and the pocket
is sunk into that solid, i.e. it occupies z from ``pocket_top_mm - depth`` to
``pocket_top_mm``.

Height rules mirror the Gridfinity Rebuilt OpenSCAD library (``height()`` in
``gridfinity-rebuilt-utility.scad`` and ``new_bin()`` in ``bin.scad``); the
``openscad`` backend cross-checks them against the library on every render.

Profile constants below are the Gridfinity spec values as written in Rebuilt's
``standard.scad``; ``bins/native.py`` builds the bin from them.
"""

from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import trimesh
    from shapely.geometry import MultiPolygon, Polygon

GRID_MM = 42.0
HEIGHT_UNIT_MM = 7.0
BIN_GAP_MM = 0.5  # a 1x1 bin body is 41.5 mm wide, centred in its 42 mm cell
BASE_HEIGHT_MM = 7.0  # base profile + bridge; every bin is at least this tall
STACKING_LIP_NOMINAL_MM = 4.4  # what "external mm" heights subtract for the standard lip
STACKING_LIP_SUPPORT_MM = 1.2  # solid infill stops this far below the bin top when lipped
LIP_ZONE_MM = 2.6  # the lip support reaches this far in from the bin body edge

# Base profile (spec): per cell a 41.5 mm rounded square on top, swept down by
# 0.8 mm chamfer / 1.8 mm vertical / 2.15 mm chamfer. (inset from the top outline, z)
BASE_PROFILE_MM: tuple[tuple[float, float], ...] = (
    (2.95, 0.0),  # bottom
    (2.15, 0.8),  # after the lower 45 deg chamfer
    (2.15, 2.6),  # after the vertical part
    (0.0, 4.75),  # top of the profile
)
BASE_PROFILE_HEIGHT_MM = BASE_PROFILE_MM[-1][1]
BASE_TOP_RADIUS_MM = 3.75  # corner radius of the base top and of the bin body
CELL_TOP_MM = GRID_MM - BIN_GAP_MM  # 41.5

# Stacking lip profiles as (inset from the outer wall, z above the nominal bin
# height), from the inner tip outwards. Rebuilt's STACKING_LIP_LINE: 0.7 mm out
# at 45 deg, 1.8 mm vertical, 1.9 mm out at 45 deg. "reduced" drops the vertical
# part. Between pocket_top and the nominal height the lip zone (LIP_ZONE_MM in
# from the wall) is a 1.2 mm deep recess, as in Rebuilt's solid bins.
WALL_MM = 0.95  # outer wall above a lowered solid top (Rebuilt's d_wall)
LIP_FILLET_RADIUS_MM = 0.6  # the knife edge at the top is rounded (Rebuilt does the same)
LIP_STYLES: tuple[str, ...] = ("standard", "reduced", "none")
LIP_PROFILES_MM: dict[str, tuple[tuple[float, float], ...]] = {
    "standard": ((2.6, 0.0), (1.9, 0.7), (1.9, 2.5), (0.0, 4.4)),
    "reduced": ((2.6, 0.0), (1.9, 0.7), (0.0, 2.6)),
    "none": (),
}

# Base holes (spec + Rebuilt): centred HOLE_INSET_MM in from the edge of the base
# top on both axes (13 mm from a 42 mm cell's centre). Depths measured from z = 0.
HOLE_INSET_MM = 8.0
MAGNET_HOLE_RADIUS_MM = 6.5 / 2
MAGNET_HOLE_DEPTH_MM = 2.4  # 2 mm magnet + 2 layers
SCREW_HOLE_RADIUS_MM = 3 / 2
HOLE_CHAMFER_MM = 0.8  # extra radius at the mouth, 45 deg
LAYER_HEIGHT_MM = 0.2  # for the printable hole top

BACKEND_NAMES: tuple[str, ...] = ("none", "native", "openscad")

# gridz_define values (see BinParams).
GRIDZ_UNITS, GRIDZ_INTERNAL_MM, GRIDZ_EXTERNAL_MM, GRIDZ_EXTERNAL_WITH_LIP_MM = 0, 1, 2, 3


def lip_height_mm(style: str) -> float:
    """Height of the lip above the bin's nominal top (0 for ``none``)."""
    profile = LIP_PROFILES_MM[style]
    return profile[-1][1] if profile else 0.0


@dataclass(frozen=True)
class BinParams:
    """Bin size, base/lip options, and where the cutout goes.

    The option names follow the customizer parameters of
    ``gridfinity-rebuilt-bins.scad`` (``_mm`` suffixes added where the library
    leaves units implicit). The bin is always solid; the only cavity is our pocket.

    ``gridz`` is interpreted by ``gridz_define``: 0 = 7 mm units (incl. the 7 mm
    base, excl. lip), 1 = internal mm (excl. base and lip), 2 = external mm excl.
    lip, 3 = external mm incl. the lip (4.4 mm standard, 2.6 mm reduced).

    ``lip`` is one of ``LIP_STYLES``; any lip except ``none`` lowers the solid top
    by the 1.2 mm lip support.

    ``offset_x_mm``/``offset_y_mm`` move the cutout's centre away from the bin
    centre (x right, y towards the far edge, as seen looking down on the bin).
    ``rotation_deg`` turns the cutout counter-clockwise about its own centre.
    """

    units_x: int = 1
    units_y: int = 1
    gridz: float = 3.0
    _: KW_ONLY
    # height
    gridz_define: int = GRIDZ_UNITS
    enable_zsnap: bool = False
    height_internal_mm: float = 0.0  # >0 overrides the solid height, <=0 is subtracted
    lip: str = "standard"
    half_grid: bool = False
    # base holes
    only_corners: bool = False
    magnet_holes: bool = False
    screw_holes: bool = False
    chamfer_holes: bool = True
    printable_hole_top: bool = True
    # cutout placement
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    rotation_deg: float = 0.0
    backend: str = "none"

    def __post_init__(self) -> None:
        if self.units_x < 1 or self.units_y < 1:
            raise ValueError("bin needs at least 1 x 1 units")
        if self.gridz < 0:
            raise ValueError("gridz must not be negative")
        if self.gridz_define not in (0, 1, 2, 3):
            raise ValueError("gridz_define must be 0..3")
        if self.lip not in LIP_STYLES:
            raise ValueError(f"lip must be one of {', '.join(LIP_STYLES)}")
        if (
            self.has_lip
            and self.height_internal_mm > 0
            and self.height_internal_mm > self.height_mm - STACKING_LIP_SUPPORT_MM
        ):
            raise ValueError(
                f"height_internal_mm must be at most {self.height_mm - STACKING_LIP_SUPPORT_MM:g}"
                " mm for this bin (stacking lip support)"
            )

    @property
    def has_lip(self) -> bool:
        return self.lip != "none"

    @property
    def lip_height_mm(self) -> float:
        return lip_height_mm(self.lip)

    @property
    def grid_mm(self) -> float:
        return GRID_MM / 2 if self.half_grid else GRID_MM

    @property
    def height_mm(self) -> float:
        """Bin height excluding the stacking lip (library ``height()``)."""
        return bin_height_mm(self)

    @property
    def infill_height_mm(self) -> float:
        return infill_height_mm(self)

    @property
    def pocket_top_mm(self) -> float:
        """Top of the solid part of the bin; the pocket is cut down from here."""
        return pocket_top_mm(self)

    @property
    def size_mm(self) -> tuple[float, float, float]:
        return (self.grid_mm * self.units_x, self.grid_mm * self.units_y, self.height_mm)

    @property
    def centre_mm(self) -> tuple[float, float]:
        w, d, _ = self.size_mm
        return (w / 2, d / 2)


def bin_height_mm(p: BinParams) -> float:
    """Mirror of ``height(z, gridz_define, enable_zsnap)`` in the OpenSCAD library."""
    raw = {
        GRIDZ_UNITS: p.gridz * HEIGHT_UNIT_MM,
        GRIDZ_INTERNAL_MM: p.gridz + BASE_HEIGHT_MM,
        GRIDZ_EXTERNAL_MM: p.gridz,
        GRIDZ_EXTERNAL_WITH_LIP_MM: p.gridz - lip_height_mm(p.lip),
    }[p.gridz_define]
    if p.enable_zsnap:
        rem = math.fmod(raw, HEIGHT_UNIT_MM)  # OpenSCAD's % is fmod
        if rem != 0:
            raw = raw + HEIGHT_UNIT_MM - rem
    return max(raw, BASE_HEIGHT_MM)


def infill_height_mm(p: BinParams) -> float:
    """Mirror of ``fill_height_real`` in ``new_bin()``: height of the solid above the base."""
    calculated = bin_height_mm(p) - BASE_HEIGHT_MM
    if p.has_lip:
        calculated -= STACKING_LIP_SUPPORT_MM
    real = p.height_internal_mm if p.height_internal_mm > 0 else calculated + p.height_internal_mm
    return max(real, 0.0)


def pocket_top_mm(p: BinParams) -> float:
    return BASE_HEIGHT_MM + infill_height_mm(p)


@dataclass(frozen=True)
class BinResult:
    solid: trimesh.Trimesh  # what gets exported as STL
    context: trimesh.Trimesh | None = None  # viewer-only reference geometry
    note: str = ""  # one line for the status display (render time, cache hit, ...)


class BinBackend(Protocol):
    name: str

    def build(self, cutout: Polygon | MultiPolygon, height_mm: float, bin: BinParams) -> BinResult:
        """``cutout`` is already placed in bin coordinates (mm, y up)."""
        ...


def get_backend(name: str) -> BinBackend:
    if name == "none":
        from gridfinity_cutter.bins.none import NoneBackend

        return NoneBackend()
    if name == "native":
        from gridfinity_cutter.bins.native import NativeBackend

        return NativeBackend()
    if name == "openscad":
        from gridfinity_cutter.bins.openscad import OpenScadBackend

        return OpenScadBackend()
    raise ValueError(f"unknown bin backend {name!r}; known: {', '.join(BACKEND_NAMES)}")
