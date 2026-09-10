import json

import pytest

from gridfinity_cutter import cli
from gridfinity_cutter.params import BinParams, GeometryParams, ImageParams, Params, ParamsError


def test_json_round_trip(tmp_path):
    p = Params(
        ImageParams(px_per_mm=8.0, threshold=None, tolerance_mm=0.3, min_area_mm2=50),
        GeometryParams(height_mm=12.5, clearance_mm=-0.4, mirror=True),
        BinParams(2, 3, 4, offset_x_mm=1.5, offset_y_mm=-2.0, rotation_deg=90),
        photo="knife.jpg",
    )
    path = tmp_path / "p.json"
    p.to_json(path)
    data = json.loads(path.read_text())
    assert data["image"]["threshold"] is None and data["bin"]["lip"] == "standard"
    assert Params.from_json(path) == p


def test_defaults_and_errors(tmp_path):
    assert Params.from_dict({}) == Params()
    assert Params.from_dict({"image": {"threshold": 42}}).image.threshold == 42
    with pytest.raises(ParamsError, match="unknown top-level"):
        Params.from_dict({"images": {}})
    with pytest.raises(ParamsError, match="unknown key.*'geometry'"):
        Params.from_dict({"geometry": {"depth": 3}})
    with pytest.raises(ParamsError, match="bad 'bin' section.*lip"):
        Params.from_dict({"bin": {"lip": "huge"}})
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(ParamsError, match="invalid JSON"):
        Params.from_json(bad)
    with pytest.raises(ParamsError, match="cannot read"):
        Params.from_json(tmp_path / "missing.json")


def test_cli_defaults_match_parser_dests():
    dests = {
        a.dest for a in cli.build_parser()._subparsers._group_actions[0].choices["run"]._actions
    }
    assert set(Params().to_cli_defaults()) <= dests


def test_params_file_sets_defaults_and_flags_win(tmp_path, monkeypatch):
    pfile = tmp_path / "p.json"
    Params(
        geometry=GeometryParams(height_mm=7.0, clearance_mm=0.25),
        bin=BinParams(2, 2, 5, rotation_deg=45),
    ).to_json(pfile)
    seen = {}

    def fake_extrude(args, svg, stl):
        seen.update(vars(args))
        return 0

    monkeypatch.setattr(cli, "_run_extrude", fake_extrude)
    assert cli.main(["extrude", "in.svg", "--params", str(pfile)]) == 0
    assert seen["height"] == 7.0 and seen["clearance"] == 0.25
    assert seen["bin_units"] == [2, 2] and seen["bin_height"] == 5 and seen["rotation"] == 45
    assert cli.main(["extrude", "in.svg", "--params", str(pfile), "--height", "3"]) == 0
    assert seen["height"] == 3.0 and seen["rotation"] == 45


def test_missing_height_and_bad_params(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["extrude", "in.svg"])
    assert "--height is required" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text('{"nope": 1}')
    with pytest.raises(SystemExit):
        cli.main(["extrude", "in.svg", "--params", str(bad)])
    assert "unknown top-level" in capsys.readouterr().err
