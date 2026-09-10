"""Gradio layer: handlers called directly, no server. Skipped without the ui extra."""

from pathlib import Path

import numpy as np
import pytest

gr = pytest.importorskip("gradio")

from gridfinity_shadowbox import ui
from gridfinity_shadowbox.bins import BinParams
from gridfinity_shadowbox.session import Session

FIXTURE = Path(__file__).parent / "fixtures" / "box_cutter.jpg"
# px_per_mm, auto, threshold, tolerance, min_area, clearance, height, mirror, relief
# (enabled, diameter, count, angle, inset), then the bin defaults.
VALUES = [5.0, True, 40, 0.2, 100, 0.5, 12.0, False, True, 15.0, 2, 90.0, 4.0] + [
    getattr(BinParams(), name) for name in ui.BIN_CONTROL_NAMES
]
IDX = {name: i for i, name in enumerate(ui.CONTROL_NAMES)}


def test_widget_specs_cover_every_bin_field():
    assert [name for name, _, _ in ui.BIN_WIDGETS].count("units_x") == 1
    assert {name for name, _, _ in ui.BIN_WIDGETS} == set(ui.BIN_CONTROL_NAMES)
    assert {g for _, g, _ in ui.BIN_WIDGETS} == {k for k, _, _ in ui.BIN_ACCORDIONS}
    assert ui.params_from_values(*VALUES).bin == BinParams()


def test_build_app():
    app = ui.build_app(FIXTURE)
    assert isinstance(app, gr.Blocks)
    assert isinstance(app.session, Session)


def test_handlers_end_to_end(tmp_path):
    session = Session()
    img, thr, glb, status, ux, uy, stem = ui.on_load(session, str(FIXTURE), *VALUES)
    assert isinstance(img, np.ndarray) and Path(glb).suffix == ".glb"
    assert thr["interactive"] is False and thr["value"] == 40.0
    assert (ux, uy) == (2, 5) and stem == "box_cutter"
    assert "threshold 40 (auto)" in status

    values = list(VALUES)
    values[1], values[2] = False, 55  # manual threshold
    img, thr, status = ui.on_image_input(session, *values)
    assert thr["interactive"] is True and "threshold 55 (fixed)" in status

    values[IDX["units_x"]], values[IDX["units_y"]], values[IDX["rotation_deg"]] = 2, 5, 30
    img, thr, glb, status = ui.on_change(session, *values)
    assert "bin 2 x 5 u (42 mm grid), 84 x 210 x 21 mm excl. lip + lip" in status

    files, status = ui.on_export(session, str(tmp_path), "knife", *values)
    assert [Path(f).name for f in files] == ["knife.svg", "knife.stl", "knife.params.json"]


def test_handlers_report_errors():
    session = Session()
    _, _, status = ui.on_image_input(session, *VALUES)
    assert status.startswith("error:") and "load a photo" in status
    bad = list(VALUES)
    bad[IDX["height_internal_mm"]] = 20.5  # above the lip support of a 21 mm bin
    assert "error: height_internal_mm" in ui.on_change(session, *bad)[3]
    assert "error: height_internal_mm" in ui.on_export(session, "", "x", *bad)[1]
    out = ui.on_load(session, None, *VALUES)
    assert "drop a photo" in out[-1]


def test_print_sheet_js_embeds_sheet():
    from gridfinity_shadowbox import sheet

    js = ui.print_sheet_js(sheet.DEFAULT_SPEC)
    assert js.startswith("() =>") and "print()" in js
    assert "@page{size:210mm 297mm;margin:0}" in js


def test_params_from_values_relief():
    r = ui.params_from_values(*VALUES).relief
    assert (r.enabled, r.diameter_mm, r.count, r.angle_deg, r.inset_mm) == (
        True,
        15.0,
        2,
        90.0,
        4.0,
    )
