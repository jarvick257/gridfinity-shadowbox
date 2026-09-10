"""Bin parameters, height rules and the OpenSCAD backend plumbing (no OpenSCAD needed)."""

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from gridfinity_cutter import bins
from gridfinity_cutter.bins import BinParams
from gridfinity_cutter.bins import openscad as scad

LIB_BINS = scad.LIBRARY_DIR / "gridfinity-rebuilt-bins.scad"


@pytest.fixture(autouse=True)
def _fresh_lookup():
    scad.find_openscad.cache_clear()
    yield
    scad.find_openscad.cache_clear()


@pytest.mark.parametrize(
    ("gridz", "define", "zsnap", "height"),
    [
        (3, 0, False, 21),
        (10, 1, False, 17),
        (20, 2, False, 20),
        (25.4, 3, False, 21),
        (22, 2, True, 28),  # snapped up to the next 7 mm
        (21, 2, True, 21),
        (0.5, 2, False, 7),  # never below the base height
    ],
)
def test_bin_height_mirrors_library(gridz, define, zsnap, height):
    p = BinParams(gridz=gridz, gridz_define=define, enable_zsnap=zsnap)
    assert p.height_mm == pytest.approx(height)


def test_pocket_top():
    assert BinParams(gridz=3).pocket_top_mm == pytest.approx(19.8)  # lip support 1.2 mm
    assert BinParams(gridz=3, include_lip=False).pocket_top_mm == pytest.approx(21)
    assert BinParams(gridz=3, height_internal_mm=5).pocket_top_mm == pytest.approx(12)
    assert BinParams(gridz=3, height_internal_mm=-2).pocket_top_mm == pytest.approx(17.8)
    assert BinParams(gridz=1).pocket_top_mm == pytest.approx(7)  # no solid above the base
    assert BinParams(2, 3, half_grid=True).size_mm == (42, 63, 21)


@pytest.mark.parametrize(
    "kw",
    [
        {"units_x": 0},
        {"gridz_define": 4},
        {"scoop": 1.5},
        {"scoop": -0.1},
        {"style_tab": 6},
        {"magnet_holes": True},  # with the default refined_holes
        {"height_internal_mm": 20.5},  # > 21 - 1.2 with lip
        {"cylinder_diameter_mm": 0},
    ],
)
def test_validation(kw):
    with pytest.raises(ValueError):
        BinParams(**kw)
    BinParams(magnet_holes=True, refined_holes=False)
    BinParams(height_internal_mm=20.5, include_lip=False)


def test_positional_fields_are_only_size():
    with pytest.raises(TypeError):
        BinParams(1, 1, 3, 0)  # type: ignore[misc]


def test_get_backend_names():
    assert bins.get_backend("openscad").name == "openscad"
    with pytest.raises(ValueError, match="unknown bin backend"):
        bins.get_backend("nope")


# -- OpenSCAD plumbing --------------------------------------------------------


def test_scad_defines_literals():
    d = scad.scad_defines(BinParams(2, 3, 2.5, include_lip=False, scoop=0.5, divx=2))
    assert d[:3] == ["-Dgridx=2", "-Dgridy=3", "-Dgridz=2.5"]
    assert "-Dinclude_lip=false" in d and "-Dscoop=0.5" in d and "-Ddivx=2" in d
    assert not any(a.startswith(("-Doffset", "-Drotation", "-Dbackend")) for a in d)


def test_scad_vars_match_wrapper_and_library():
    """Every mapped variable is assigned in bin.scad, and bin.scad exposes every customizer variable."""
    top_level = set(re.findall(r"^(\w+)\s*=", scad.WRAPPER.read_text(), re.MULTILINE))
    assert set(scad.SCAD_VARS.values()) <= top_level
    fields = set(scad.SCAD_VARS)
    assert fields == {f for f in BinParams.__dataclass_fields__} - {
        "offset_x_mm",
        "offset_y_mm",
        "rotation_deg",
        "backend",
    }
    if LIB_BINS.is_file():
        customizer = LIB_BINS.read_text().split("// ===== IMPLEMENTATION")[0]
        lib_vars = set(re.findall(r"^(\w+)\s*=", customizer, re.MULTILINE)) - {"hole_options"}
        assert lib_vars <= set(scad.SCAD_VARS.values())


def test_cache_key_ignores_placement():
    tool = scad.OpenScad("x", "2026.01", 2026, True)
    a = scad.cache_key(BinParams(2, 2), tool)
    assert a == scad.cache_key(
        BinParams(2, 2, offset_x_mm=5, rotation_deg=30, backend="openscad"), tool
    )
    assert a != scad.cache_key(BinParams(2, 2, gridz=4), tool)
    assert a != scad.cache_key(BinParams(2, 2), scad.OpenScad("x", "2026.02", 2026, True))


def _fake_openscad(path: Path, version: str, with_backend: bool = True) -> str:
    backend = "--backend arg" if with_backend else ""
    path.write_text(
        "#!/bin/sh\n"
        f'case "$1" in --version) echo "OpenSCAD version {version}" >&2;; '
        f'--help) echo "{backend}";; esac\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_find_openscad_versions(tmp_path, monkeypatch):
    monkeypatch.setenv(scad.ENV_OPENSCAD, _fake_openscad(tmp_path / "new", "2025.03.01"))
    tool = scad.find_openscad()
    assert tool is not None and tool.year == 2025 and tool.manifold and tool.version == "2025.03.01"

    monkeypatch.setenv(scad.ENV_OPENSCAD, _fake_openscad(tmp_path / "old", "2021.01", False))
    scad.find_openscad.cache_clear()
    assert scad.find_openscad() is None
    assert "development build" in (scad.unavailable_reason() or "")
    with pytest.raises(scad.OpenScadError, match="unavailable"):
        scad.require()


def test_parse_and_check_echo():
    text = (
        'ECHO: "GFC_HEIGHT_MM=21"\nECHO: "GFC_INFILL_MM=[39.6, 39.6, 12.8]"\n'
        'ECHO: "GFC_BBOX_MM=[41.5, 41.5, 24.5479]"\nECHO: "other"\n'
    )
    echo = scad.parse_echo(text)
    assert echo == {
        "HEIGHT_MM": 21,
        "INFILL_MM": [39.6, 39.6, 12.8],
        "BBOX_MM": [41.5, 41.5, 24.5479],
    }
    scad.check_echo(BinParams(gridz=3), echo)
    with pytest.raises(scad.OpenScadError, match="out of sync"):
        scad.check_echo(BinParams(gridz=4), echo)
    with pytest.raises(scad.OpenScadError, match="did not echo"):
        scad.check_echo(BinParams(), {})


def test_render_reports_openscad_errors(tmp_path, monkeypatch):
    monkeypatch.setenv(scad.ENV_OPENSCAD, _fake_openscad(tmp_path / "fake", "2025.01.01"))
    monkeypatch.setenv(scad.ENV_CACHE, str(tmp_path / "cache"))
    assert scad.find_openscad() is not None  # probe (and cache) before patching subprocess

    def failing_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", 'ERROR: Assertion failed: "boom"\n')

    monkeypatch.setattr(scad.subprocess, "run", failing_run)
    with pytest.raises(scad.OpenScadError, match=r"(?s)exit 1.*boom"):
        scad.render_bin(BinParams())
    assert not any(tmp_path.joinpath("cache").rglob("*.stl"))


def test_cache_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv(scad.ENV_CACHE, str(tmp_path))
    assert scad.cache_dir() == tmp_path / "bins"
    monkeypatch.delenv(scad.ENV_CACHE)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert scad.cache_dir() == tmp_path / "xdg" / "gridfinity-cutter" / "bins"
    assert os.environ.get(scad.ENV_CACHE) is None
