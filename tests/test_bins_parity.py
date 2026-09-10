"""Native bin vs. the OpenSCAD (Gridfinity Rebuilt) render of the same options.

Runs only with OpenSCAD available; the tolerances absorb Rebuilt's 0.02 mm wall
tolerance and polygonised arcs.
"""

import numpy as np
import pytest

from gridfinity_cutter.bins import BinParams
from gridfinity_cutter.bins import native as nat
from gridfinity_cutter.bins import openscad as scad

pytestmark = pytest.mark.skipif(not scad.available(), reason="needs OpenSCAD 2024+ and the library")

CASES = [
    {},
    {"lip": "none"},
    {"magnet_holes": True},
    {"magnet_holes": True, "screw_holes": True},
    {"screw_holes": True, "printable_hole_top": False, "chamfer_holes": False},
    {"units_x": 2, "units_y": 1, "magnet_holes": True, "only_corners": True},
    {"units_x": 2, "units_y": 2, "half_grid": True, "magnet_holes": True, "lip": "none"},
    {"gridz": 20, "gridz_define": 3},
    {"height_internal_mm": 5},
    {"height_internal_mm": 5, "lip": "none"},
    {"gridz": 1},
]


@pytest.fixture(autouse=True)
def _cache(tmp_path_factory, monkeypatch):
    if not (cache := __import__("os").environ.get(scad.ENV_CACHE)):
        monkeypatch.setenv(scad.ENV_CACHE, str(tmp_path_factory.mktemp("scad")))
    return cache


@pytest.mark.parametrize("kw", CASES, ids=[str(c) for c in CASES])
def test_native_matches_openscad(kw):
    p = BinParams(**kw)
    ref = scad.to_manifold(scad.load_bin(scad.render_bin(p)[0]), "ref")
    ours = nat.bin_solid(p)
    assert np.allclose(ref.bounding_box(), ours.bounding_box(), atol=0.02)
    assert ours.volume() == pytest.approx(ref.volume(), rel=3e-3)
    zs = [0.3, 1, 2, 2.5, 2.7, 3, 4, 5, 6, 6.9, 7.5, 13, p.height_mm - 0.6, p.height_mm + 0.3]
    for z in zs:
        if z < ref.bounding_box()[5]:
            assert ours.slice(z).area() == pytest.approx(ref.slice(z).area(), abs=10.0), z
