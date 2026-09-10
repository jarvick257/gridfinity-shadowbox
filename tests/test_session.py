"""Session tests: the UI logic without Gradio."""

import json
from pathlib import Path

import numpy as np
import pytest
import shapely
import trimesh

from gridfinity_shadowbox import cli, extrude
from gridfinity_shadowbox.params import (
    BinParams,
    GeometryParams,
    ImageParams,
    Params,
    ReliefParams,
)
from gridfinity_shadowbox.session import COLOR_BIN, COLOR_OUTLINE, Session, SessionError

FIXTURE = Path(__file__).parent / "fixtures" / "box_cutter.jpg"
PX_PER_MM = 5.0


@pytest.fixture(scope="module")
def session():
    s = Session()
    s.load(FIXTURE, PX_PER_MM)
    return s


def params(**kw) -> Params:
    kw_relief = kw.pop("relief", ReliefParams())
    return Params(
        ImageParams(px_per_mm=PX_PER_MM),
        GeometryParams(height_mm=kw.pop("height_mm", 12.0), clearance_mm=kw.pop("clearance", 0.0)),
        BinParams(**kw),
        kw_relief,
    )


def test_load_and_outline_cache(session):
    assert session.loaded and session.reprojection_error_mm < 1.0
    a = session.outline(ImageParams(px_per_mm=PX_PER_MM))
    b = session.outline(ImageParams(px_per_mm=PX_PER_MM))
    assert a is b
    c = session.outline(ImageParams(px_per_mm=PX_PER_MM, tolerance_mm=1.0))
    assert c is not a and len(c.polygon_mm) < len(a.polygon_mm)
    w, h = a.polygon_mm.max(axis=0) - a.polygon_mm.min(axis=0)
    assert 45 < w < 52 and 158 < h < 166
    with pytest.raises(SessionError, match="px/mm"):
        session.outline(ImageParams(px_per_mm=PX_PER_MM + 1))
    with pytest.raises(SessionError, match="load a photo"):
        Session().outline(ImageParams())


def test_overlay_draws_outline_and_bin(session):
    p = params(units_x=2, units_y=5, rotation_deg=20, offset_x_mm=5)
    img = session.overlay(p, display_px_per_mm=2.0)
    assert img.ndim == 3 and img.shape[2] == 3 and img.shape[0] > img.shape[1]
    assert (np.all(img == COLOR_OUTLINE, axis=2)).sum() > 100
    assert (np.all(img == COLOR_BIN, axis=2)).sum() > 100


def test_bin_footprint_inverts_placement(session):
    p = params(units_x=4, units_y=5, rotation_deg=35, offset_x_mm=4, offset_y_mm=-6)
    rect, lines = session.bin_footprint_mm(p)
    assert len(lines) == 3 + 4
    # Mapping the drawn footprint into bin coordinates gives the bin rectangle back.
    m = session._photo_to_bin_matrix(p)
    back = extrude.apply_matrix(rect, m)
    assert back.equals_exact(shapely.box(0, 0, 168, 210), 1e-6)
    # And the outline, mapped the same way, sits where the placement puts it.
    placed, _ = session.build(p)
    outline_in_bin = extrude.apply_matrix(session.offset_geometry(p), m)
    assert outline_in_bin.equals_exact(placed, 1e-6)
    assert shapely.box(0, 0, 168, 210).contains(placed)


def test_mirror_flips_footprint(session):
    straight, _ = session.bin_footprint_mm(params(units_x=2, units_y=5, offset_y_mm=10))
    p = params(units_x=2, units_y=5, offset_y_mm=10)
    mirrored, _ = session.bin_footprint_mm(
        Params(p.image, GeometryParams(height_mm=12, mirror=True), p.bin)
    )
    # +offset_y moves the object up on the photo (bin drawn lower); mirrored it is the reverse.
    assert straight.centroid.y > mirrored.centroid.y


def test_fit_bin(session):
    assert session.fit_bin(params()) == (2, 5)  # 48 x 163 mm knife + 4 mm walls > 168 mm
    assert session.fit_bin(params(rotation_deg=90)) == (5, 2)


def test_render_scene(session, solid_at):
    p = params(units_x=2, units_y=5, gridz=3, offset_x_mm=5, offset_y_mm=-3, height_mm=12)
    glb = session.render(p)
    scene = trimesh.load(str(glb))
    assert set(scene.geometry) == {"bin"}
    bin_mesh = scene.geometry["bin"]
    assert np.allclose(bin_mesh.bounds[0], (0.25, 0.25, 0), atol=1e-4)
    assert np.allclose(bin_mesh.bounds[1, :2], (83.75, 209.75), atol=1e-4)
    # The pocket is centred at the offset and sunk 12 mm below the lip support (21 - 1.2).
    placed, mesh = session.build(p)
    minx, miny, maxx, maxy = placed.bounds
    assert np.allclose(((minx + maxx) / 2, (miny + maxy) / 2), (42 + 5, 105 - 3), atol=1e-3)
    cx, cy = placed.centroid.x, placed.centroid.y
    assert not solid_at(mesh, cx, cy, 19.8 - 12 + 0.1) and solid_at(mesh, cx, cy, 19.8 - 12 - 0.1)
    # Rewriting with other parameters changes the file in place.
    glb2 = session.render(params(units_x=2, units_y=5, height_mm=5))
    assert glb2 == glb
    _, mesh2 = session.build(params(units_x=2, units_y=5, height_mm=5))
    assert mesh2.volume > mesh.volume


def test_export_and_cli_reproduce(session, tmp_path):
    p = params(units_x=2, units_y=5, rotation_deg=15, clearance=0.5, height_mm=9)
    files = session.export(p, tmp_path / "out", "knife")
    _, stl, js = files
    assert [f.name for f in files] == ["knife.svg", "knife.stl", "knife.params.json"]
    assert json.loads(js.read_text())["photo"] == "box_cutter.jpg"
    ui_mesh = trimesh.load(str(stl))
    assert ui_mesh.is_watertight

    rc = cli.main(["run", str(FIXTURE), "--params", str(js), "-o", str(tmp_path / "cli.stl")])
    assert rc == 0
    cli_mesh = trimesh.load(str(tmp_path / "cli.stl"))
    assert np.allclose(cli_mesh.bounds, ui_mesh.bounds, atol=0.01)
    assert abs(cli_mesh.volume - ui_mesh.volume) / ui_mesh.volume < 1e-3


def test_status_warns_on_deep_pocket(session):
    text = session.status(params(gridz=1, height_mm=12))
    assert "WARNING: pocket deeper" in text and "threshold" in text
    assert "WARNING" not in session.status(params(units_x=2, units_y=5, gridz=3, height_mm=12))
    assert "WARNING: cutout within" in session.status(params(units_x=2, units_y=4, height_mm=5))


def test_scene_with_holes_and_export(session, tmp_path):
    p = params(units_x=2, units_y=5, gridz=3, height_mm=8, magnet_holes=True)
    scene = trimesh.load(str(session.render(p)))
    assert set(scene.geometry) == {"bin"}
    bin_mesh = scene.geometry["bin"]
    assert np.allclose(bin_mesh.bounds[0], (0.25, 0.25, 0), atol=1e-4)
    assert np.allclose(bin_mesh.bounds[1, :2], (83.75, 209.75), atol=1e-4)
    assert 21 + 3.4 < bin_mesh.bounds[1, 2] < 21 + 3.7  # stacking lip with rounded tip
    assert "STL:" in session.status(p, bin_mesh)

    _, stl, js = session.export(p, tmp_path / "out", "obj")
    mesh = trimesh.load(str(stl))
    assert mesh.is_volume
    rc = cli.main(["run", str(FIXTURE), "--params", str(js), "-o", str(tmp_path / "cli.stl")])
    assert rc == 0
    assert abs(trimesh.load(str(tmp_path / "cli.stl")).volume - mesh.volume) < 1e-3 * mesh.volume


def test_relief_grows_pocket_and_reproduces(session, tmp_path):
    base = params(units_x=2, units_y=3, height_mm=8)
    relief = ReliefParams(enabled=True, diameter_mm=16, count=2, angle_deg=90, inset_mm=4)
    p = params(units_x=2, units_y=3, height_mm=8, relief=relief)
    assert session.offset_geometry(p).area > session.offset_geometry(base).area
    _, m0 = session.build(base)
    _, m1 = session.build(p)
    assert m1.volume < m0.volume and m1.is_watertight
    assert "finger relief: 2 x 16 mm" in session.status(p)
    assert session.overlay(p).shape == session.overlay(base).shape

    _, _, js = session.export(p, tmp_path / "out", "r")
    assert json.loads(js.read_text())["relief"]["enabled"] is True
    rc = cli.main(["run", str(FIXTURE), "--params", str(js), "-o", str(tmp_path / "cli.stl")])
    assert rc == 0
    cli_mesh = trimesh.load(str(tmp_path / "cli.stl"))
    assert abs(cli_mesh.volume - m1.volume) / m1.volume < 1e-3
