"""OpenSCAD backend against the real binary and library. Skipped when either is missing."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh
from shapely.geometry import box

manifold3d = pytest.importorskip("manifold3d")

from gridfinity_cutter import cli, extrude
from gridfinity_cutter.bins import BinParams, get_backend
from gridfinity_cutter.bins import openscad as scad
from gridfinity_cutter.params import GeometryParams, ImageParams, Params
from gridfinity_cutter.session import Session

pytestmark = pytest.mark.skipif(not scad.available(), reason=scad.unavailable_reason() or "")

FIXTURE = Path(__file__).parent / "fixtures" / "box_cutter.jpg"


@pytest.fixture(autouse=True)
def _cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv(scad.ENV_CACHE, str(tmp_path_factory.mktemp("cache")))
    scad.find_openscad.cache_clear()


def test_render_1x1_bin_in_bin_frame():
    stl, meta = scad.render_bin(BinParams(1, 1, 3))
    assert meta["cached"] is False and meta["render_s"] > 0
    mesh = scad.load_bin(stl)
    assert mesh.is_volume
    lo, hi = mesh.bounds
    assert np.allclose(lo, (0.25, 0.25, 0), atol=1e-3)
    assert np.allclose(hi[:2], (41.75, 41.75), atol=1e-3)
    assert 21 + 3.4 < hi[2] < 21 + 3.7  # stacking lip on top
    assert meta["echo"]["HEIGHT_MM"] == 21


def test_render_uses_cache(monkeypatch):
    p = BinParams(1, 2, 2, include_lip=False)
    stl, meta = scad.render_bin(p)
    assert meta["cached"] is False

    def no_run(*a, **kw):
        raise AssertionError("OpenSCAD must not run for a cached bin")

    monkeypatch.setattr(scad.subprocess, "run", no_run)
    stl2, meta2 = scad.render_bin(p)
    assert stl2 == stl and meta2["cached"] is True
    assert json.loads(stl.with_suffix(".json").read_text())["echo"] == meta2["echo"]


@pytest.mark.parametrize(
    "p",
    [
        BinParams(1, 1, 3),
        BinParams(1, 1, 10, gridz_define=1),
        BinParams(1, 1, 20, gridz_define=2),
        BinParams(1, 1, 25.4, gridz_define=3),
        BinParams(1, 1, 22, gridz_define=2, enable_zsnap=True),
        BinParams(1, 1, 3, include_lip=False),
        BinParams(1, 1, 3, height_internal_mm=5),
        BinParams(1, 1, 3, height_internal_mm=-2),
    ],
)
def test_height_rules_agree_with_library(p):
    """render_bin raises if the Python mirror of the height rules disagrees with the echo."""
    _, meta = scad.render_bin(p)
    assert meta["echo"]["HEIGHT_MM"] == pytest.approx(p.height_mm)
    assert meta["echo"]["INFILL_MM"][2] == pytest.approx(p.infill_height_mm)
    assert scad.load_bin(scad.render_bin(p)[0]).bounds[1, 2] >= p.height_mm - 1e-3


def test_half_grid_and_options():
    p = BinParams(
        2, 2, 2, half_grid=True, include_lip=False, magnet_holes=True, refined_holes=False
    )
    mesh = scad.load_bin(scad.render_bin(p)[0])
    assert np.allclose(mesh.bounds, [[0.25, 0.25, 0], [41.75, 41.75, 14]], atol=1e-3)
    p2 = BinParams(2, 1, 3, divx=2, divy=1, scoop=0.5, style_tab=0)
    with_compartments = scad.load_bin(scad.render_bin(p2)[0])
    assert with_compartments.volume < mesh.volume * 2  # compartments cut out of a 2x1
    # Compartment tops share edges with the infill top: not "watertight" for trimesh, fine for Manifold.
    assert scad.to_manifold(with_compartments, "bin").volume() > 0


def test_build_cuts_pocket():
    p = BinParams(2, 2, 3, backend="openscad")
    cutout = extrude.place_in_bin(box(0, 0, 30, 50), p)
    res = get_backend("openscad").build(cutout, 8.0, p)
    body = scad.load_bin(scad.render_bin(p)[0])
    assert res.solid.is_volume and res.context is None
    assert np.allclose(res.solid.bounds, body.bounds, atol=1e-3)
    assert math.isclose(body.volume - res.solid.volume, 30 * 50 * 8, rel_tol=1e-3)
    assert "OpenSCAD" in res.note and "faces" in res.note

    # Pocket open at the top: a probe box in the pocket meets nothing, one below its floor is solid.
    top = p.pocket_top_mm
    solid = scad.to_manifold(res.solid, "result")
    probe = manifold3d.Manifold.cube([4, 4, 2]).translate([40, 40, top - 4])
    assert (solid ^ probe).volume() == pytest.approx(0)
    below = manifold3d.Manifold.cube([4, 4, 1]).translate([40, 40, top - 8 - 1.5])
    assert (solid ^ below).volume() == pytest.approx(16, rel=1e-3)


def test_deep_pocket_and_bad_bin_errors():
    p = BinParams(1, 1, 1, backend="openscad")  # solid top at 7 mm, nothing above the base
    cutout = extrude.place_in_bin(box(0, 0, 10, 10), p)
    res = get_backend("openscad").build(cutout, 3.0, p)  # cuts into the base, still a solid
    assert res.solid.is_volume
    with pytest.raises(scad.OpenScadError, match="does not touch"):
        get_backend("openscad").build(box(100, 100, 110, 110), 3.0, p)  # outside the bin


def test_session_scene_export_and_cli(tmp_path):
    session = Session()
    session.load(FIXTURE, 5.0)
    p = Params(
        image=ImageParams(px_per_mm=5.0),
        geometry=GeometryParams(height_mm=9),
        bin=BinParams(2, 5, 3, backend="openscad"),
    )
    glb, res = session.render_scene(p)
    scene = trimesh.load(str(glb))
    assert set(scene.geometry) == {"bin"}
    assert "bin 2 x 5 u" in session.status(p, res) and "OpenSCAD" in session.status(p, res)

    svg, stl, js = session.export(p, tmp_path, "knife")
    ui_mesh = trimesh.load(str(stl))
    assert ui_mesh.is_volume and np.allclose(ui_mesh.bounds[0], (0.25, 0.25, 0), atol=1e-3)
    assert json.loads(js.read_text())["bin"]["backend"] == "openscad"

    rc = cli.main(["extrude", str(svg), "--params", str(js), "-o", str(tmp_path / "cli.stl")])
    assert rc == 0
    cli_mesh = trimesh.load(str(tmp_path / "cli.stl"))
    assert np.allclose(cli_mesh.bounds, ui_mesh.bounds, atol=0.01)
    assert abs(cli_mesh.volume - ui_mesh.volume) / ui_mesh.volume < 1e-4
