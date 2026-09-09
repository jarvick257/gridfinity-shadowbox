import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest

from gridfinity_cutter import sheet, step1

SPEC = sheet.SheetSpec()
FIXTURE = Path(__file__).parent / "fixtures" / "box_cutter.jpg"


def render_sheet_raster(px_per_mm: float) -> np.ndarray:
    """White page with the four markers drawn at their nominal positions (grayscale)."""
    w, h = SPEC.page_mm
    img = np.full((int(h * px_per_mm), int(w * px_per_mm)), 255, np.uint8)
    size_px = int(SPEC.marker_size_mm * px_per_mm)
    for mid, corners in sheet.marker_corners_mm(SPEC).items():
        tile = np.where(sheet.marker_bits(mid, SPEC), 0, 255).astype(np.uint8)
        tile = cv2.resize(tile, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
        x, y = (corners[0] * px_per_mm).astype(int)
        img[y : y + size_px, x : x + size_px] = tile
    return img


def synthetic_photo(rect_mm, px_per_mm=4.0, seed=0):
    """Sheet raster with a dark rectangle, viewed through a mild random perspective."""
    page = cv2.cvtColor(render_sheet_raster(px_per_mm), cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = (int(v * px_per_mm) for v in rect_mm)
    page[y0:y1, x0:x1] = (60, 40, 30)
    h, w = page.shape[:2]
    rng = np.random.default_rng(seed)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-0.06, 0.06, src.shape).astype(np.float32) * [w, h]
    dst = dst - dst.min(axis=0) + 40
    out_size = (int(dst[:, 0].max()) + 40, int(dst[:, 1].max()) + 40)
    m = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return cv2.warpPerspective(page, m, out_size, borderValue=(120, 130, 140))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_synthetic_rectangle_recovered_in_mm(tmp_path, seed):
    rect = (80.0, 100.0, 110.0, 150.0)  # 30 x 50 mm
    photo = synthetic_photo(rect, seed=seed)
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), photo)

    res = step1.run(path, tmp_path / "out.svg", px_per_mm=8.0)

    assert res.reprojection_error_mm < 0.5
    assert np.allclose(res.bbox_mm, rect, atol=0.5)
    area = cv2.contourArea(res.polygon_mm.astype(np.float32))
    assert abs(area - 30 * 50) / (30 * 50) < 0.03


def test_missing_marker_raises(tmp_path):
    photo = synthetic_photo((80.0, 100.0, 110.0, 150.0))
    photo[: photo.shape[0] // 3, : photo.shape[1] // 3] = 200  # wipe the TL marker
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), photo)
    with pytest.raises(step1.OutlineError, match=r"marker id\(s\) \[0\]"):
        step1.run(path, tmp_path / "out.svg")


def test_empty_sheet_raises(tmp_path):
    page = cv2.cvtColor(render_sheet_raster(4.0), cv2.COLOR_GRAY2BGR)
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), page)
    with pytest.raises(step1.OutlineError, match="no object found"):
        step1.run(path, tmp_path / "out.svg")


def test_box_cutter_fixture(tmp_path):
    out = tmp_path / "knife.svg"
    res = step1.run(FIXTURE, out, px_per_mm=5.0)
    assert res.reprojection_error_mm < 1.0
    w, h = res.size_mm
    # TODO: tighten once the knife is measured with calipers.
    assert 45 < w < 52
    assert 158 < h < 166
    assert len(res.polygon_mm) > 20


def test_write_svg_contract(tmp_path):
    poly = np.array([[30.0, 40.0], [60.0, 40.0], [60.0, 90.0], [30.0, 90.0]])
    out = tmp_path / "o.svg"
    w, h = step1.write_svg(out, poly, comment="test -- comment")

    root = ET.parse(out).getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.get("width") == f"{w:.3f}mm" and root.get("height") == f"{h:.3f}mm"
    assert root.get("viewBox") == f"0 0 {w:.3f} {h:.3f}"
    assert np.isclose(w, 30 + 2 * step1.SVG_MARGIN_MM)
    assert np.isclose(h, 50 + 2 * step1.SVG_MARGIN_MM)
    paths = root.findall("svg:path", ns)
    assert len(paths) == 1
    assert paths[0].get("d").endswith(" Z")
    assert paths[0].get("d").startswith(f"M {step1.SVG_MARGIN_MM:.3f} {step1.SVG_MARGIN_MM:.3f}")
    assert not any("transform" in el.attrib for el in root.iter())
