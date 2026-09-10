"""Gridfinity bin backends: build the solid that goes around (or is) the cutout.

This module is the only place that knows the Gridfinity grid constants and the
bin coordinate convention. It must not import other project modules at import
time (``extrude`` and ``params`` both import from here).

Bin coordinate convention
-------------------------
x and y are measured from the bin's bottom-left grid corner, z from the bin
bottom. A bin of ``units_x`` by ``units_y`` by ``units_z`` occupies
``[0, 42 * units_x] x [0, 42 * units_y] x [0, 7 * units_z]`` mm (the real bin
body is ``BIN_GAP_MM`` smaller than the grid cell). The cutout is placed inside
that box by ``extrude.place_in_bin`` and the pocket is sunk into the bin top,
i.e. it occupies z from ``7 * units_z - depth`` to ``7 * units_z``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import trimesh
    from shapely.geometry import MultiPolygon, Polygon

GRID_MM = 42.0
HEIGHT_UNIT_MM = 7.0
BIN_GAP_MM = 0.5  # a 1x1 bin body is 41.5 mm wide, centred in its 42 mm cell

BACKEND_NAMES: tuple[str, ...] = ("none",)


@dataclass(frozen=True)
class BinParams:
    """Bin size in Gridfinity units and where the cutout goes inside it.

    ``offset_x_mm``/``offset_y_mm`` move the cutout's centre away from the bin
    centre (x right, y towards the far edge, as seen looking down on the bin).
    ``rotation_deg`` turns the cutout counter-clockwise about its own centre.
    """

    units_x: int = 1
    units_y: int = 1
    units_z: int = 3
    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    rotation_deg: float = 0.0
    backend: str = "none"

    @property
    def size_mm(self) -> tuple[float, float, float]:
        return (GRID_MM * self.units_x, GRID_MM * self.units_y, HEIGHT_UNIT_MM * self.units_z)

    @property
    def centre_mm(self) -> tuple[float, float]:
        w, d, _ = self.size_mm
        return (w / 2, d / 2)


@dataclass(frozen=True)
class BinResult:
    solid: trimesh.Trimesh  # what gets exported as STL
    context: trimesh.Trimesh | None = None  # viewer-only reference geometry


class BinBackend(Protocol):
    name: str

    def build(self, cutout: Polygon | MultiPolygon, height_mm: float, bin: BinParams) -> BinResult:
        """``cutout`` is already placed in bin coordinates (mm, y up)."""
        ...


def get_backend(name: str) -> BinBackend:
    if name == "none":
        from gridfinity_cutter.bins.none import NoneBackend

        return NoneBackend()
    raise ValueError(f"unknown bin backend {name!r}; known: {', '.join(BACKEND_NAMES)}")
