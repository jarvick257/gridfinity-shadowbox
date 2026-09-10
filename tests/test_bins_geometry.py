"""Bin geometry (manifold3d): checked by slicing the solid, not by mesh statistics."""

import math

import numpy as np
import pytest
from shapely.geometry import box

from gridfinity_shadowbox import extrude
from gridfinity_shadowbox.bins import (
    BASE_PROFILE_HEIGHT_MM,
    LIP_STYLES,
    BinParams,
    lip_height_mm,
)
from gridfinity_shadowbox.bins import geometry as nat


def _slice_polys(p: BinParams, z: float) -> list:
    return nat.bin_solid(p).slice(z).to_polygons()


def test_rrect_and_loft():
    cs = nat.rrect(41.5, 41.5, 3.75)
    assert cs.area() == pytest.approx(41.5**2 - (4 - math.pi) * 3.75**2, rel=1e-3)
    m = nat.loft(cs, 0.0, cs.offset(-2.0), 2.0)
    assert m.status().name == "NoError"
    assert m.bounding_box()[5] == pytest.approx(2.0)
    assert m.slice(1.0).area() == pytest.approx(cs.offset(-1.0).area(), rel=1e-3)


@pytest.mark.parametrize("lip", LIP_STYLES)
def test_bin_bounds_and_lip(lip):
    p = BinParams(2, 1, 3, lip=lip)
    m = nat.bin_solid(p)
    lo = m.bounding_box()[:3]
    hi = m.bounding_box()[3:]
    assert np.allclose(lo, (0.25, 0.25, 0), atol=1e-6)
    assert np.allclose(hi[:2], (83.75, 41.75), atol=1e-6)
    if lip == "none":
        assert hi[2] == pytest.approx(21)
    else:
        # the knife edge of the spec profile is rounded, which lowers the top a bit
        assert 21 + lip_height_mm(lip) - 0.9 < hi[2] < 21 + lip_height_mm(lip)
        # the lip zone is a 1.2 mm recess above the solid top
        assert m.slice(p.pocket_top_mm + 0.6).area() < 0.3 * m.slice(p.pocket_top_mm - 0.6).area()
    assert m.slice(p.pocket_top_mm - 0.1).area() == pytest.approx(
        nat.rrect(83.5, 41.5, 3.75).area(), rel=1e-3
    )


def test_base_profile():
    p = BinParams(1, 1, 3, lip="none")
    m = nat.bin_solid(p)
    top = nat.rrect(41.5, 41.5, 3.75).area()
    assert m.slice(0.01).area() == pytest.approx(nat.rrect(35.6, 35.6, 0.8).area(), rel=2e-3)
    assert m.slice(1.5).area() == pytest.approx(nat.rrect(37.2, 37.2, 1.6).area(), rel=2e-3)
    assert m.slice(BASE_PROFILE_HEIGHT_MM + 0.1).area() == pytest.approx(top, rel=1e-3)
    # a 2x2 bin has four separate feet below the bridge and one body above
    q = BinParams(2, 2, 3, lip="none")
    assert len(_slice_polys(q, 2.0)) == 4
    assert len(_slice_polys(q, 6.0)) == 1


@pytest.mark.parametrize(
    ("kw", "n_holes", "z"),
    [
        ({"magnet_holes": True}, 4, 1.0),
        ({"screw_holes": True}, 4, 4.0),
        ({"magnet_holes": True, "screw_holes": True}, 4, 3.5),
        ({"units_x": 2, "units_y": 2, "magnet_holes": True}, 16, 1.0),
        ({"units_x": 2, "units_y": 2, "magnet_holes": True, "only_corners": True}, 4, 1.0),
        ({"units_x": 2, "units_y": 2, "half_grid": True, "magnet_holes": True}, 4, 1.0),
    ],
)
def test_holes(kw, n_holes, z):
    p = BinParams(**kw)
    polys = _slice_polys(p, z)
    n_feet = 1 if z >= BASE_PROFILE_HEIGHT_MM else p.units_x * p.units_y
    assert len(polys) == n_feet + n_holes  # outer contour(s) + one contour per hole
    plain = nat.bin_solid(BinParams(**{k: v for k, v in kw.items() if not k.endswith("holes")}))
    assert nat.bin_solid(p).volume() < plain.volume()


def test_hole_geometry():
    # magnet pocket: 6.5 mm wide, 2.4 mm deep, bridged above; screw hole 3 mm to the bridge
    p = BinParams(magnet_holes=True, screw_holes=True, chamfer_holes=False)
    m = nat.bin_solid(p)
    full = nat.bin_solid(BinParams(chamfer_holes=False))
    per_hole = lambda z: (full.slice(z).area() - m.slice(z).area()) / 4
    assert per_hole(1.0) == pytest.approx(math.pi * 3.25**2, rel=2e-2)
    assert per_hole(2.5) < math.pi * 3.25**2 * 0.7  # first bridge layer is a bar
    assert per_hole(2.7) == pytest.approx(9.0, rel=5e-2)  # 3 x 3 mm square
    assert per_hole(4.0) == pytest.approx(math.pi * 1.5**2, rel=2e-2)
    assert per_hole(5.0) == 0
    chamfered = nat.bin_solid(BinParams(magnet_holes=True, screw_holes=True))
    assert chamfered.slice(0.2).area() < m.slice(0.2).area()
    assert chamfered.slice(1.5).area() == pytest.approx(m.slice(1.5).area(), rel=1e-6)


def test_lowered_solid_keeps_wall_only_with_lip():
    low = nat.bin_solid(BinParams(height_internal_mm=5))
    assert low.slice(15.0).area() == pytest.approx(
        nat.rrect(41.5, 41.5, 3.75).area() - nat.rrect(39.6, 39.6, 2.8).area(), rel=1e-2
    )
    assert low.bounding_box()[5] > 21
    no_lip = nat.bin_solid(BinParams(height_internal_mm=5, lip="none"))
    assert no_lip.bounding_box()[5] == pytest.approx(12)


def test_short_bin_lip_does_not_protrude_below_base():
    m = nat.bin_solid(BinParams(gridz=1))
    top = nat.rrect(41.5, 41.5, 3.75).area()
    assert m.slice(4.0).area() < top
    assert m.slice(BASE_PROFILE_HEIGHT_MM + 0.1).area() == pytest.approx(top, rel=1e-3)


def test_lip_profiles_rounded_tip():
    inner, outer = nat.lip_profiles("standard")
    assert inner[0] == (0.0, -3.8) and outer[0] == (0.0, -3.8)
    assert inner[-1] == outer[0 + 2]  # both boundaries meet at the topmost point
    assert inner[-1][1] == pytest.approx(3.55, abs=0.01)
    assert nat.lip_profiles("reduced")[1][-1][1] < 2.6


def test_cache_ignores_placement():
    a = nat.bin_solid(BinParams(2, 2))
    b = nat.bin_solid(BinParams(2, 2, offset_x_mm=5, rotation_deg=30))
    assert a is b


def test_build_cuts_pocket():
    p = BinParams(2, 2, 3)
    cutout = extrude.place_in_bin(box(0, 0, 30, 50), p)
    mesh = nat.bin_with_pocket(cutout, 8.0, p)
    body = nat.bin_mesh(p)
    assert mesh.is_volume
    assert np.allclose(mesh.bounds, body.bounds, atol=1e-3)
    assert math.isclose(body.volume - mesh.volume, 30 * 50 * 8, rel_tol=1e-3)
