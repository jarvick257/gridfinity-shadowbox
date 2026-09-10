"""Backend 7a: no bin. The cutout solid alone, plus a reference block for the viewer."""

from __future__ import annotations

import trimesh
from shapely.geometry import MultiPolygon, Polygon

from gridfinity_cutter import extrude
from gridfinity_cutter.bins import BIN_GAP_MM, BinParams, BinResult

CONTEXT_RGBA = (0.75, 0.75, 0.78, 0.35)


class NoneBackend:
    """Cutout only: the STL is the pocket solid in bin coordinates, sunk into the bin top.

    The ``context`` mesh is a plain translucent block of the bin's outer size so
    the viewer shows scale and placement. It is *not* a Gridfinity profile;
    subtract the ``solid`` from a real bin in your slicer or CAD tool.
    """

    name = "none"

    def build(self, cutout: Polygon | MultiPolygon, height_mm: float, bin: BinParams) -> BinResult:
        w, d, h = bin.size_mm
        solid = extrude.extrude_geometry(cutout, height_mm, z0=bin.pocket_top_mm - height_mm)
        block = trimesh.creation.box((w - BIN_GAP_MM, d - BIN_GAP_MM, h))
        block.apply_translation((w / 2, d / 2, h / 2))
        block.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=CONTEXT_RGBA, alphaMode="BLEND", doubleSided=True
            )
        )
        return BinResult(solid=solid, context=block)
