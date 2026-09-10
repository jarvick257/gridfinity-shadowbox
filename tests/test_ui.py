"""Gradio layer: handlers called directly, no server. Skipped without the ui extra."""

from pathlib import Path

import numpy as np
import pytest

gr = pytest.importorskip("gradio")

from gridfinity_cutter import ui
from gridfinity_cutter.session import Session

FIXTURE = Path(__file__).parent / "fixtures" / "box_cutter.jpg"
VALUES = [5.0, True, 40, 0.2, 100, 0.5, 12.0, False, 1, 1, 3, 0.0, 0.0, 0.0]


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

    values[8], values[9], values[13] = 2, 5, 30
    img, thr, glb, status = ui.on_change(session, *values)
    assert "bin 2 x 5 x 3 u" in status

    files, status = ui.on_export(session, str(tmp_path), "knife", *values)
    assert [Path(f).name for f in files] == ["knife.svg", "knife.stl", "knife.params.json"]


def test_handlers_report_errors():
    session = Session()
    _, _, status = ui.on_image_input(session, *VALUES)
    assert status.startswith("error:") and "load a photo" in status
    out = ui.on_load(session, None, *VALUES)
    assert "drop a photo" in out[-1]
