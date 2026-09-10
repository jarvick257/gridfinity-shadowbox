import cv2
import numpy as np

from gridfinity_cutter import sheet

SPEC = sheet.SheetSpec()


def test_marker_geometry_inside_page_square_symmetric():
    w, h = SPEC.page_mm
    corners = sheet.marker_corners_mm(SPEC)
    assert set(corners) == set(SPEC.marker_ids)
    for c in corners.values():
        assert c.shape == (4, 2)
        assert (c >= 0).all() and (c[:, 0] <= w).all() and (c[:, 1] <= h).all()
        assert np.isclose(c[1, 0] - c[0, 0], SPEC.marker_size_mm)
        assert np.isclose(c[3, 1] - c[0, 1], SPEC.marker_size_mm)
    centres = np.array([c.mean(axis=0) for c in corners.values()])
    assert np.allclose(centres.mean(axis=0), [w / 2, h / 2])


def test_work_area_between_markers():
    x0, y0, x1, y1 = sheet.work_area_mm(SPEC)
    assert x1 > x0 and y1 > y0
    for c in sheet.marker_corners_mm(SPEC).values():
        assert not (x0 < c[0, 0] < x1 and y0 < c[0, 1] < y1)


def test_markers_detectable_at_expected_positions():
    px_per_mm = 2.0
    w, h = SPEC.page_mm
    img = np.full((int(h * px_per_mm), int(w * px_per_mm)), 255, np.uint8)
    size_px = int(SPEC.marker_size_mm * px_per_mm)
    for mid, corners in sheet.marker_corners_mm(SPEC).items():
        bits = sheet.marker_bits(mid, SPEC)
        tile = np.where(bits, 0, 255).astype(np.uint8)
        tile = cv2.resize(tile, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
        x, y = (corners[0] * px_per_mm).astype(int)
        img[y : y + size_px, x : x + size_px] = tile

    detector = cv2.aruco.ArucoDetector(sheet.aruco_dictionary(SPEC), cv2.aruco.DetectorParameters())
    found, ids, _ = detector.detectMarkers(img)
    assert ids is not None
    found_by_id = {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), found)}
    assert set(found_by_id) == set(SPEC.marker_ids)
    for mid, expected_mm in sheet.marker_corners_mm(SPEC).items():
        # Detected corners sit on pixel centres of the outer black cells, so allow ~1 px.
        assert np.allclose(found_by_id[mid], expected_mm * px_per_mm, atol=1.5)


def test_render_pdf_writes_file(tmp_path):
    out = tmp_path / "sheet.pdf"
    sheet.render_pdf(out, SPEC)
    assert out.stat().st_size > 1000
    assert out.read_bytes().startswith(b"%PDF")


def test_render_svg_matches_pdf_geometry():
    svg = sheet.render_svg(SPEC)
    w, h = SPEC.page_mm
    assert f'width="{w:g}mm" height="{h:g}mm" viewBox="0 0 {w:g} {h:g}"' in svg
    black_cells = sum(int(sheet.marker_bits(mid, SPEC).sum()) for mid in SPEC.marker_ids)
    assert svg.count('fill="#000"') == black_cells
    assert "100 mm" in svg and "do not fit to page" in svg


def test_render_print_html_fixes_page_size():
    html = sheet.render_print_html(SPEC)
    w, h = SPEC.page_mm
    assert f"@page{{size:{w:g}mm {h:g}mm;margin:0}}" in html
    assert "<svg" in html and "</svg>" in html
