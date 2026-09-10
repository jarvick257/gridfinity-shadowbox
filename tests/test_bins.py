"""Bin parameters and height rules."""

import pytest

from gridfinity_cutter import bins
from gridfinity_cutter.bins import BinParams


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
    assert BinParams(gridz=3, lip="none").pocket_top_mm == pytest.approx(21)
    assert BinParams(gridz=3, lip="reduced").pocket_top_mm == pytest.approx(19.8)
    assert BinParams(gridz=3, height_internal_mm=5).pocket_top_mm == pytest.approx(12)
    assert BinParams(gridz=3, height_internal_mm=-2).pocket_top_mm == pytest.approx(17.8)
    assert BinParams(gridz=1).pocket_top_mm == pytest.approx(7)  # no solid above the base
    assert BinParams(2, 3, half_grid=True).size_mm == (42, 63, 21)


@pytest.mark.parametrize(
    "kw",
    [
        {"units_x": 0},
        {"gridz_define": 4},
        {"lip": "huge"},
        {"height_internal_mm": 20.5},  # > 21 - 1.2 with lip
    ],
)
def test_validation(kw):
    with pytest.raises(ValueError):
        BinParams(**kw)
    BinParams(magnet_holes=True, screw_holes=True)
    BinParams(height_internal_mm=20.5, lip="none")


def test_positional_fields_are_only_size():
    with pytest.raises(TypeError):
        BinParams(1, 1, 3, 0)  # type: ignore[misc]


def test_get_backend_names():
    assert bins.get_backend("native").name == "native"
    assert bins.get_backend("none").name == "none"
    with pytest.raises(ValueError, match="unknown bin backend"):
        bins.get_backend("nope")
