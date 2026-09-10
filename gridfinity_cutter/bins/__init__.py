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
STACKING_LIP_NOMINAL_MM = 4.4  # what "external mm" heights subtract for the lip
STACKING_LIP_SUPPORT_MM = 1.2  # solid infill stops this far below the bin top when lipped
LIP_ZONE_MM = 2.6  # the lip support reaches this far in from the bin body edge

BACKEND_NAMES: tuple[str, ...] = ("none", "openscad")

# gridz_define values (see BinParams).
GRIDZ_UNITS, GRIDZ_INTERNAL_MM, GRIDZ_EXTERNAL_MM, GRIDZ_EXTERNAL_WITH_LIP_MM = 0, 1, 2, 3
STYLE_TAB_NAMES = ("Full", "Auto", "Left", "Center", "Right", "None")
PLACE_TAB_NAMES = ("Everywhere", "Top-left division")


@dataclass(frozen=True)
class BinParams:
    """Bin size, the Gridfinity Rebuilt bin options, and where the cutout goes.

    The first block maps 1:1 onto the customizer parameters of
    ``gridfinity-rebuilt-bins.scad`` (names kept where they are self-explanatory,
    ``_mm`` suffixes added where the library leaves units implicit). Defaults are
    the library's, except ``divx``/``divy`` which default to 0 (solid bin) because
    the object pocket is our own cut.

    ``gridz`` is interpreted by ``gridz_define``: 0 = 7 mm units (incl. the 7 mm
    base, excl. lip), 1 = internal mm (excl. base and lip), 2 = external mm excl.
    lip, 3 = external mm incl. the nominal 4.4 mm lip.

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
    include_lip: bool = True
    half_grid: bool = False
    # compartments (cut by the library, in addition to our pocket)
    divx: int = 0
    divy: int = 0
    depth_mm: float = 0.0  # 0 = full depth
    cut_cylinders: bool = False
    cylinder_diameter_mm: float = 10.0
    cylinder_chamfer_mm: float = 0.5
    style_tab: int = 1
    place_tab: int = 0
    scoop: float = 1.0
    # base holes
    only_corners: bool = False
    refined_holes: bool = True
    magnet_holes: bool = False
    screw_holes: bool = False
    crush_ribs: bool = True
    chamfer_holes: bool = True
    printable_hole_top: bool = True
    enable_thumbscrew: bool = False
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
        if self.divx < 0 or self.divy < 0:
            raise ValueError("divx/divy must not be negative")
        if self.cylinder_diameter_mm <= 0 or self.cylinder_chamfer_mm < 0:
            raise ValueError("cylinder diameter must be positive and chamfer non-negative")
        if not 0 <= self.style_tab < len(STYLE_TAB_NAMES):
            raise ValueError(f"style_tab must be 0..{len(STYLE_TAB_NAMES) - 1}")
        if not 0 <= self.place_tab < len(PLACE_TAB_NAMES):
            raise ValueError(f"place_tab must be 0..{len(PLACE_TAB_NAMES) - 1}")
        if not (self.scoop == 0 or 0 < self.scoop <= 1):
            raise ValueError("scoop must be 0 or in (0, 1]")
        if self.refined_holes and self.magnet_holes:
            raise ValueError("refined_holes is not compatible with magnet_holes")
        if (
            self.include_lip
            and self.height_internal_mm > 0
            and self.height_internal_mm > self.height_mm - STACKING_LIP_SUPPORT_MM
        ):
            raise ValueError(
                f"height_internal_mm must be at most {self.height_mm - STACKING_LIP_SUPPORT_MM:g}"
                " mm for this bin (stacking lip support)"
            )

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
        GRIDZ_EXTERNAL_WITH_LIP_MM: p.gridz - STACKING_LIP_NOMINAL_MM,
    }[p.gridz_define]
    if p.enable_zsnap:
        rem = math.fmod(raw, HEIGHT_UNIT_MM)  # OpenSCAD's % is fmod
        if rem != 0:
            raw = raw + HEIGHT_UNIT_MM - rem
    return max(raw, BASE_HEIGHT_MM)


def infill_height_mm(p: BinParams) -> float:
    """Mirror of ``fill_height_real`` in ``new_bin()``: height of the solid above the base."""
    calculated = bin_height_mm(p) - BASE_HEIGHT_MM
    if p.include_lip:
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
    if name == "openscad":
        from gridfinity_cutter.bins.openscad import OpenScadBackend

        return OpenScadBackend()
    raise ValueError(f"unknown bin backend {name!r}; known: {', '.join(BACKEND_NAMES)}")
