import math
from pathlib import Path

import numpy as np
import pytest
import shapely
import trimesh

from gridfinity_shadowbox import cli, extrude, outline
from gridfinity_shadowbox.bins import BinParams
from gridfinity_shadowbox.bins.geometry import bin_mesh
from gridfinity_shadowbox.bins.geometry import bin_with_pocket as build_bin

FIXTURE = Path(__file__).parent / "fixtures" / "box_cutter.jpg"
RECT = np.array([[30.0, 40.0], [60.0, 40.0], [60.0, 90.0], [30.0, 90.0]])  # 30 x 50 mm


def svg_file(tmp_path, body, attrs='width="100mm" height="100mm" viewBox="0 0 100 100"'):
    p = tmp_path / "in.svg"
    p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" {attrs}>{body}</svg>')
    return p


def rect_svg(tmp_path):
    p = tmp_path / "rect.svg"
    outline.write_svg(p, RECT)
    return p


def load(path) -> trimesh.Trimesh:
    return trimesh.load(str(path))


def test_rectangle_round_trip(tmp_path):
    out = tmp_path / "o.stl"
    res = extrude.run(rect_svg(tmp_path), out, height_mm=20)
    assert np.allclose(res.size_mm, (30, 50, 20), atol=1e-3)
    assert math.isclose(res.volume_mm3, 30 * 50 * 20, rel_tol=1e-6)
    mesh = load(out)
    assert mesh.is_watertight
    assert len(mesh.faces) == 12
    assert np.allclose(mesh.bounds[0], 0, atol=1e-6)


def test_positive_clearance_grows_with_round_corners(tmp_path):
    res = extrude.run(rect_svg(tmp_path), tmp_path / "o.stl", height_mm=5, clearance_mm=1.0)
    assert np.allclose(res.size_mm[:2], (32, 52), atol=1e-3)
    expected = 30 * 50 + 2 * 1.0 * (30 + 50) + math.pi * 1.0**2
    assert abs(res.area_mm2 - expected) / expected < 0.01


def test_negative_clearance_shrinks(tmp_path):
    res = extrude.run(rect_svg(tmp_path), tmp_path / "o.stl", height_mm=5, clearance_mm=-5.0)
    assert np.allclose(res.size_mm[:2], (20, 40), atol=1e-3)
    with pytest.raises(extrude.ExtrudeError, match="collapsed"):
        extrude.run(rect_svg(tmp_path), tmp_path / "o.stl", height_mm=5, clearance_mm=-20.0)


def test_circle_and_curved_path(tmp_path):
    body = '<circle cx="30" cy="30" r="10"/>'
    geom = extrude.load_polygons(svg_file(tmp_path, body), curve_tolerance_mm=0.01)
    assert abs(geom.area - math.pi * 100) / (math.pi * 100) < 0.005
    assert 40 < len(geom.exterior.coords) < 200  # flattened, but not thousands of points
    coarse = extrude.load_polygons(svg_file(tmp_path, body), curve_tolerance_mm=0.1)
    assert len(coarse.exterior.coords) < len(geom.exterior.coords)
    assert coarse.hausdorff_distance(geom) < 0.1

    # Same circle as a path with arcs, relative commands and no explicit Z endpoint.
    body = '<path d="m 60,70 a 10,10 0 1,0 20,0 a 10,10 0 1,0 -20,0 z"/>'
    geom = extrude.load_polygons(svg_file(tmp_path, body), curve_tolerance_mm=0.01)
    assert abs(geom.area - math.pi * 100) / (math.pi * 100) < 0.005
    assert np.allclose(geom.bounds, (60, 60, 80, 80), atol=0.05)

    body = '<path d="M 10 10 C 40 10, 40 40, 10 40 Z"/>'
    geom = extrude.load_polygons(svg_file(tmp_path, body))
    assert 300 < geom.area < 700  # sanity: a lobe, flattened with many points
    assert len(geom.exterior.coords) > 10


def test_units_and_transforms(tmp_path):
    body = '<rect x="100" y="100" width="300" height="500" transform="translate(50 0) scale(2)"/>'
    geom = extrude.load_polygons(
        svg_file(tmp_path, body, 'width="100mm" height="100mm" viewBox="0 0 1000 1000"')
    )
    assert np.allclose(geom.bounds, (25, 20, 85, 120), atol=1e-3)

    unitless = svg_file(
        tmp_path, '<rect x="5" y="5" width="30" height="50"/>', 'viewBox="0 0 40 60"'
    )
    assert np.allclose(extrude.load_polygons(unitless).bounds, (5, 5, 35, 55), atol=1e-6)


def test_hole(tmp_path):
    body = '<path d="M 10 10 L 50 10 L 50 50 L 10 50 Z M 20 20 L 40 20 L 40 40 L 20 40 Z"/>'
    out = tmp_path / "o.stl"
    res = extrude.run(svg_file(tmp_path, body), out, height_mm=10)
    assert math.isclose(res.volume_mm3, (40 * 40 - 20 * 20) * 10, rel_tol=1e-6)
    assert load(out).is_watertight


def test_y_flip_and_mirror(tmp_path):
    # L shape: the stub sticks out at the SVG top-left (small y).
    body = '<path d="M 0 0 L 10 0 L 10 10 L 30 10 L 30 30 L 0 30 Z"/>'
    svg = svg_file(tmp_path, body)
    mesh = extrude.to_mesh(extrude.load_polygons(svg), 5)
    top = mesh.vertices[mesh.vertices[:, 1] > 29.9]  # max-y row of the mesh
    assert top[:, 0].max() <= 10 + 1e-6  # stub is at max y after the flip
    mesh = extrude.to_mesh(extrude.load_polygons(svg), 5, mirror=True)
    bottom = mesh.vertices[mesh.vertices[:, 1] < 0.1]
    assert bottom[:, 0].max() <= 10 + 1e-6


def test_open_path_raises(tmp_path):
    svg = svg_file(tmp_path, '<path d="M 0 0 L 10 0 L 10 10"/>')
    with pytest.raises(extrude.ExtrudeError, match="1 open path"):
        extrude.load_polygons(svg)


def test_end_to_end_with_outline(tmp_path):
    svg = tmp_path / "knife.svg"
    res1 = outline.run(FIXTURE, svg, px_per_mm=5.0)
    stl = tmp_path / "knife.stl"
    res2 = extrude.run(svg, stl, height_mm=15, clearance_mm=0.5)
    w, h = res1.size_mm
    assert np.allclose(res2.size_mm, (w + 1.0, h + 1.0, 15), atol=0.1)
    assert load(stl).is_watertight


def test_cli(tmp_path, capsys):
    stl = tmp_path / "o.stl"
    rc = cli.main(
        ["extrude", str(rect_svg(tmp_path)), "--height", "10", "--clearance", "0.5", "-o", str(stl)]
    )
    assert rc == 0
    assert stl.exists()
    assert "31.0 x 51.0 x 10.0 mm" in capsys.readouterr().out


def test_cli_run_keeps_svg(tmp_path, capsys):
    stl = tmp_path / "knife.stl"
    rc = cli.main(["run", str(FIXTURE), "--height", "15", "--px-per-mm", "5", "-o", str(stl)])
    assert rc == 0
    assert stl.exists()
    assert (tmp_path / "knife.svg").exists()
    out = capsys.readouterr().out
    assert "knife.svg" in out and "knife.stl" in out
    assert load(stl).is_watertight


def test_curve_tolerance_leaves_straight_segments_alone(tmp_path):
    # A polygon with a tiny 0.3 mm notch: a 1 mm simplify pass would erase it.
    body = '<path d="M 0 0 L 20 0 L 20 10 L 10.3 10 L 10 10.3 L 9.7 10 L 0 10 Z"/>'
    fine = extrude.load_polygons(svg_file(tmp_path, body), curve_tolerance_mm=0.01)
    coarse = extrude.load_polygons(svg_file(tmp_path, body), curve_tolerance_mm=1.0)
    assert len(coarse.exterior.coords) == len(fine.exterior.coords) == 8
    assert coarse.equals_exact(fine, 1e-9)


# -- bin placement ------------------------------------------------------------


def cad_rect():
    """The 30 x 50 test rectangle in the CAD plane (y up), somewhere off-origin."""
    return extrude.flip_y(extrude.from_polygon(RECT))


def test_from_polygon_matches_svg_path(tmp_path):
    # write_svg shifts the outline so its bbox starts at the SVG margin.
    shifted = shapely.affinity.translate(
        extrude.from_polygon(RECT), outline.SVG_MARGIN_MM - 30, outline.SVG_MARGIN_MM - 40
    )
    assert shifted.equals(extrude.load_polygons(rect_svg(tmp_path)))
    with pytest.raises(extrude.ExtrudeError, match="no area"):
        extrude.from_polygon(np.array([[0, 0], [0.1, 0], [0.1, 0.1]]))


def test_extrude_geometry_z0():
    mesh = extrude.extrude_geometry(cad_rect(), 5.0, z0=16.0)
    assert np.allclose(mesh.bounds[:, 2], (16.0, 21.0))
    assert math.isclose(mesh.volume, 30 * 50 * 5, rel_tol=1e-6)


def test_place_in_bin_centre_offset_and_rotation():
    bin_ = BinParams(units_x=2, units_y=3, gridz=3, offset_x_mm=5.0, offset_y_mm=-7.0)
    placed = extrude.place_in_bin(cad_rect(), bin_)
    minx, miny, maxx, maxy = placed.bounds
    assert np.allclose(((minx + maxx) / 2, (miny + maxy) / 2), (42 + 5, 63 - 7))
    assert np.allclose((maxx - minx, maxy - miny), (30, 50))

    rot = extrude.place_in_bin(cad_rect(), BinParams(2, 3, 3, rotation_deg=90))
    minx, miny, maxx, maxy = rot.bounds
    assert np.allclose((maxx - minx, maxy - miny), (50, 30))
    assert np.allclose(((minx + maxx) / 2, (miny + maxy) / 2), (42, 63))

    # Placement ignores where the outline was beforehand.
    moved = shapely.affinity.translate(cad_rect(), 123.0, -456.0)
    assert extrude.place_in_bin(moved, bin_).equals_exact(placed, 1e-9)

    # A point at the bin corner maps back through the inverse matrix.
    m = extrude.placement_matrix(cad_rect(), bin_)
    inv = np.linalg.inv(m)
    corner = extrude.apply_matrix(shapely.Point(0, 0), inv)
    assert extrude.apply_matrix(corner, m).equals_exact(shapely.Point(0, 0), 1e-9)


def test_run_in_bin_coordinates(tmp_path, solid_at):
    out = tmp_path / "o.stl"
    bin_ = BinParams(units_x=1, units_y=2, gridz=3, lip="none")
    res = extrude.run(rect_svg(tmp_path), out, height_mm=8, bin=bin_)
    mesh = load(out)
    # The STL is the whole bin (42 x 84 mm cells, 0.5 mm gap) with the pocket cut out.
    assert np.allclose(mesh.bounds, [[0.25, 0.25, 0], [41.75, 83.75, 21]])
    assert np.allclose(res.size_mm, (41.5, 83.5, 21))
    assert mesh.is_watertight
    assert math.isclose(bin_mesh(bin_).volume - mesh.volume, 30 * 50 * 8, rel_tol=1e-3)
    # The pocket occupies the 30 x 50 mm rectangle centred in the bin, top 8 mm.
    placed = extrude.place_in_bin(cad_rect(), bin_)
    assert np.allclose(placed.bounds, (6, 17, 36, 67))
    assert not solid_at(mesh, 21, 42, 21 - 4) and solid_at(mesh, 21, 42, 21 - 12)


def test_pocket_below_lip_support(solid_at):
    bin_ = BinParams(units_x=2, units_y=1, gridz=2)
    mesh = build_bin(extrude.place_in_bin(cad_rect(), bin_), 5.0, bin_)
    # With a stacking lip the solid part stops 1.2 mm below the 14 mm bin top.
    x, y = bin_.centre_mm
    assert not solid_at(mesh, x, y, 12.8 - 5 + 0.1) and solid_at(mesh, x, y, 12.8 - 5 - 0.1)


def test_cli_bin_options(tmp_path, capsys, solid_at):
    stl = tmp_path / "o.stl"
    rc = cli.main(
        [
            "extrude",
            str(rect_svg(tmp_path)),
            "--height",
            "10",
            "--bin-units",
            "2",
            "3",
            "--bin-height",
            "28",
            "--bin-gridz-define",
            "2",
            "--bin-lip",
            "none",
            "--rotation",
            "90",
            "-o",
            str(stl),
        ]
    )
    assert rc == 0
    assert "in a 2 x 3 u bin, gridz 28 (84 x 126 x 28 mm excl. lip" in capsys.readouterr().out
    mesh = load(stl)
    assert np.allclose(mesh.bounds, [[0.25, 0.25, 0], [83.75, 125.75, 28]])
    bin_ = BinParams(2, 3, 28, gridz_define=2, lip="none")
    assert math.isclose(bin_mesh(bin_).volume - mesh.volume, 30 * 50 * 10, rel_tol=1e-3)
    # Rotated by 90 degrees the pocket is 50 mm along x.
    x, y = bin_.centre_mm
    assert not solid_at(mesh, x + 24, y, 27) and solid_at(mesh, x + 26, y, 27)


def test_add_relief_geometry():
    rect = shapely.box(0, 0, 40, 20)  # bbox centre (20, 10)
    relief = extrude.ReliefParams(enabled=True, diameter_mm=10, count=1, angle_deg=0, inset_mm=2.5)
    g = extrude.add_relief(rect, relief)
    # circle centre at x = 40 - 2.5 = 37.5, radius 5 -> reaches x = 42.5
    assert g.bounds == pytest.approx((0, 0, 42.5, 20), abs=0.05)
    assert g.area > rect.area
    assert extrude.add_relief(rect, extrude.ReliefParams()) is rect
    two = extrude.add_relief(rect, extrude.ReliefParams(True, 10, 2, 90, 0))
    assert two.bounds == pytest.approx((0, -5, 40, 25), abs=0.05)
    assert two.centroid.x == pytest.approx(20)
    with pytest.raises(extrude.ExtrudeError):
        extrude.add_relief(rect, extrude.ReliefParams(True, 0, 1))


def test_cli_relief(tmp_path, capsys):
    stl = tmp_path / "o.stl"
    args = ["extrude", str(rect_svg(tmp_path)), "--height", "5", "-o", str(stl)]
    assert cli.main(args) == 0
    plain = load(stl).volume
    assert (
        cli.main(
            args
            + ["--relief", "--relief-diameter", "10", "--relief-count", "2", "--relief-inset", "2"]
        )
        == 0
    )
    assert "finger relief 2 x 10 mm" in capsys.readouterr().out
    assert load(stl).volume > plain
