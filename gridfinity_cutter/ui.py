"""Local web UI (Gradio): tune image and geometry parameters, see outline and STL live.

This module is presentation only. All state and computation live in
``session.Session``; the handlers here unpack widget values into ``Params``,
call the session and return widget updates. They take the session as their
first argument so tests can call them directly, without a server.

Event wiring notes (Gradio 6): ``.change`` also fires on programmatic updates,
so every listener here is ``.input``/``.release``/``.submit``/``.blur``/click.
That is what makes it safe to write the automatically chosen threshold back
into the threshold slider. ``trigger_mode="always_last"`` keeps the last value
of a drag even while a previous handler is still running.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import fields
from functools import partial, update_wrapper
from pathlib import Path

import gradio as gr

from gridfinity_cutter import extrude, outline, sheet
from gridfinity_cutter.bins import LIP_STYLES
from gridfinity_cutter.params import BinParams, GeometryParams, ImageParams, Params
from gridfinity_cutter.session import Session, SessionError

TITLE = "gridfinity-cutter"
Errors = (outline.OutlineError, extrude.ExtrudeError, SessionError, ValueError)

# Order of the plain widget values every handler receives after the session:
# the image/cutout controls, then one value per BinParams field.
OTHER_CONTROL_NAMES = (
    "px_per_mm",
    "auto",
    "threshold",
    "tolerance",
    "min_area",
    "clearance",
    "height",
    "mirror",
)
BIN_CONTROL_NAMES = tuple(f.name for f in fields(BinParams))
CONTROL_NAMES = OTHER_CONTROL_NAMES + BIN_CONTROL_NAMES
_BIN_TYPES = {f.name: type(getattr(BinParams(), f.name)) for f in fields(BinParams)}


def bin_from_values(v: dict[str, object]) -> BinParams:
    """Build ``BinParams`` from widget values keyed by field name (coerced to the field type)."""
    return BinParams(**{name: _BIN_TYPES[name](v[name]) for name in BIN_CONTROL_NAMES})


def params_from_values(*values) -> Params:
    v = dict(zip(CONTROL_NAMES, values, strict=True))
    return Params(
        image=ImageParams(
            px_per_mm=float(v["px_per_mm"]),
            threshold=None if v["auto"] else float(v["threshold"]),
            tolerance_mm=float(v["tolerance"]),
            min_area_mm2=float(v["min_area"]),
        ),
        geometry=GeometryParams(
            height_mm=float(v["height"]),
            clearance_mm=float(v["clearance"]),
            mirror=bool(v["mirror"]),
        ),
        bin=bin_from_values(v),
    )


def _threshold_update(session: Session, params: Params):
    """Show the auto-chosen threshold on the (disabled) slider while auto is on."""
    if params.image.threshold is None:
        return gr.update(value=session.outline(params.image).threshold, interactive=False)
    return gr.update(interactive=True)


# -- handlers -----------------------------------------------------------------


def on_image_input(session: Session, *values):
    """Image parameter moved: overlay + threshold slider + status."""
    try:
        params = params_from_values(*values)
        img = session.overlay(params)
        return img, _threshold_update(session, params), session.status(params)
    except Errors as e:
        return gr.update(), gr.update(), f"error: {e}"


def on_change(session: Session, *values):
    """Any parameter settled: overlay + threshold slider + 3D scene + status."""
    try:
        params = params_from_values(*values)
        img = session.overlay(params)
        thr = _threshold_update(session, params)
        glb, mesh = session.render_scene(params)
        return img, thr, str(glb), session.status(params, mesh)
    except Errors as e:
        return gr.update(), gr.update(), gr.update(), f"error: {e}"


def on_load(session: Session, photo: str | None, *values):
    """Photo chosen (or px/mm changed): rectify, size the bin to fit, then everything."""
    if not photo:
        return (gr.update(),) * 6 + ("drop a photo of the object on the reference sheet",)
    try:
        params = params_from_values(*values)
        session.load(photo, params.image.px_per_mm)
        ux, uy = session.fit_bin(params)
        values = list(values)
        values[CONTROL_NAMES.index("units_x")] = ux
        values[CONTROL_NAMES.index("units_y")] = uy
        img, thr, glb, status = on_change(session, *values)
        return img, thr, glb, status, ux, uy, Path(photo).stem
    except Errors as e:
        return (gr.update(),) * 6 + (f"error: {e}",)


def on_export(session: Session, out_dir: str, stem: str, *values):
    stem = (stem or "object").strip()
    try:
        params = params_from_values(*values)
        files = session.export(params, out_dir or os.getcwd(), stem)
        return [str(f) for f in files], "wrote " + ", ".join(str(f) for f in files)
    except Errors as e:
        return gr.update(), f"error: {e}"
    except OSError as e:
        return gr.update(), f"error: cannot write to {out_dir}: {e}"


def print_sheet_js(spec: sheet.SheetSpec) -> str:
    """Browser-side handler: load the sheet into a hidden iframe and open the print dialog.

    Printing the HTML/SVG version rather than the PDF keeps the browser at its
    default 100% scale (PDF viewers tend to default to "fit to page", which
    would silently shrink the markers). A fresh iframe per click makes sure the
    load event fires again. Nothing is sent to the server.
    """
    html = json.dumps(sheet.render_print_html(spec))
    return f"""() => {{
        const old = document.getElementById("gc-print-sheet");
        if (old) old.remove();
        const f = document.createElement("iframe");
        f.id = "gc-print-sheet";
        f.setAttribute("aria-hidden", "true");
        f.style.cssText = "position:fixed;right:0;bottom:0;width:0;height:0;border:0;";
        // srcdoc is set before insertion and the load handler checks for the
        // sheet: the iframe's initial about:blank document fires load too.
        f.onload = () => {{
            if (!f.contentDocument || !f.contentDocument.querySelector("svg")) return;
            f.contentWindow.focus();
            f.contentWindow.print();
        }};
        f.srcdoc = {html};
        document.body.appendChild(f);
    }}"""


# -- bin widgets ----------------------------------------------------------------

# (field, accordion, factory taking the default value). Every BinParams field
# appears exactly once; a test checks that.
BIN_WIDGETS: list[tuple[str, str, Callable[[object], gr.components.Component]]] = [
    (
        "units_x",
        "size",
        lambda d: gr.Slider(label="Units X", minimum=1, maximum=12, step=1, value=d),
    ),
    (
        "units_y",
        "size",
        lambda d: gr.Slider(label="Units Y", minimum=1, maximum=12, step=1, value=d),
    ),
    ("gridz", "size", lambda d: gr.Number(label="Height (gridz)", value=d, minimum=0, step=0.5)),
    (
        "gridz_define",
        "size",
        lambda d: gr.Dropdown(
            label="Height unit",
            choices=[
                ("7 mm units (incl. base, excl. lip)", 0),
                ("Internal mm (excl. base and lip)", 1),
                ("External mm (excl. lip)", 2),
                ("External mm (incl. lip)", 3),
            ],
            value=d,
        ),
    ),
    ("enable_zsnap", "size", lambda d: gr.Checkbox(label="Snap height to 7 mm", value=d)),
    ("lip", "size", lambda d: gr.Dropdown(label="Stacking lip", choices=list(LIP_STYLES), value=d)),
    (
        "height_internal_mm",
        "size",
        lambda d: gr.Number(label="Solid height override (mm, 0 = auto)", value=d, step=0.5),
    ),
    ("half_grid", "size", lambda d: gr.Checkbox(label="Half grid (21 mm)", value=d)),
    ("offset_x_mm", "place", lambda d: gr.Number(label="Offset X (mm)", value=d, step=0.5)),
    ("offset_y_mm", "place", lambda d: gr.Number(label="Offset Y (mm)", value=d, step=0.5)),
    ("rotation_deg", "place", lambda d: gr.Number(label="Rotation (deg, CCW)", value=d, step=5)),
    ("magnet_holes", "holes", lambda d: gr.Checkbox(label="Magnet holes (6 x 2 mm)", value=d)),
    ("screw_holes", "holes", lambda d: gr.Checkbox(label="Screw holes (M3)", value=d)),
    ("only_corners", "holes", lambda d: gr.Checkbox(label="Holes only in the corners", value=d)),
    ("chamfer_holes", "holes", lambda d: gr.Checkbox(label="Chamfered holes", value=d)),
    ("printable_hole_top", "holes", lambda d: gr.Checkbox(label="Printable hole tops", value=d)),
]
BIN_ACCORDIONS = (
    ("size", "Bin size & height", True),
    ("place", "Cutout placement", True),
    ("holes", "Base holes", False),
)


def build_bin_widgets(defaults: BinParams) -> dict[str, gr.components.Component]:
    """The Bin section: one widget per BinParams field, grouped in accordions."""
    widgets: dict[str, gr.components.Component] = {}
    for key, title, open_ in BIN_ACCORDIONS:
        with gr.Accordion(title, open=open_):
            for name, group, factory in BIN_WIDGETS:
                if group == key:
                    widgets[name] = factory(getattr(defaults, name))
    return widgets


def settle_events(w: gr.components.Component) -> list:
    """User-driven 'value settled' events per widget type (never ``.change``)."""
    if isinstance(w, gr.Slider):
        return [w.release]
    if isinstance(w, gr.Number | gr.Textbox):
        return [w.submit, w.blur]
    return [w.input]


# -- layout -------------------------------------------------------------------


def build_app(initial_photo: str | Path | None = None, output_dir: str | Path | None = None):
    session = Session()
    photo0 = str(initial_photo) if initial_photo else None
    out0 = str(output_dir) if output_dir else (str(Path(photo0).parent) if photo0 else os.getcwd())
    always = {"trigger_mode": "always_last"}

    def bind(fn):
        # partial() has no __name__, which would name every API endpoint "/partial"
        return update_wrapper(partial(fn, session), fn)

    with gr.Blocks(title=TITLE, fill_width=True) as app:
        gr.Markdown(f"## {TITLE}")
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                print_btn = gr.Button(
                    f"Print reference sheet ({session.spec.paper.upper()}, 100% scale)"
                )
                photo = gr.Image(
                    label="Photo", type="filepath", sources=["upload"], height=160, value=photo0
                )
                gr.Markdown("**Outline**")
                px_per_mm = gr.Number(label="Resolution (px/mm)", value=10.0, minimum=2, step=1)
                auto = gr.Checkbox(label="Automatic threshold", value=True)
                threshold = gr.Slider(
                    label="Threshold (Lab distance from paper)",
                    minimum=outline.AUTO_THRESHOLD_MIN,
                    maximum=120,
                    step=1,
                    value=40,
                    interactive=False,
                )
                tolerance = gr.Slider(
                    label="Simplify tolerance (mm)",
                    minimum=0.05,
                    maximum=2.0,
                    step=0.05,
                    value=outline.DEFAULT_TOLERANCE_MM,
                )
                min_area = gr.Slider(
                    label="Min blob area (mm^2)",
                    minimum=10,
                    maximum=3000,
                    step=10,
                    value=outline.DEFAULT_MIN_AREA_MM2,
                )
                gr.Markdown("**Cutout**")
                clearance = gr.Number(label="Clearance (mm, negative shrinks)", value=0.0, step=0.1)
                height = gr.Number(label="Pocket depth (mm)", value=10.0, minimum=0.1, step=0.5)
                mirror = gr.Checkbox(label="Mirror (object goes in upside down)", value=False)
                gr.Markdown("**Bin**")
                bin_widgets = build_bin_widgets(BinParams())
                gr.Markdown("**Export**")
                out_dir = gr.Textbox(label="Output folder", value=out0)
                stem = gr.Textbox(
                    label="File name (without extension)",
                    value=Path(photo0).stem if photo0 else "object",
                )
                export_btn = gr.Button("Export SVG + STL + params.json", variant="primary")
                files = gr.File(label="Written files", file_count="multiple", interactive=False)
            with gr.Column(scale=2):
                overlay = gr.Image(label="Outline on photo", interactive=False, height=720)
            with gr.Column(scale=2):
                viewer = gr.Model3D(
                    label="Finished bin (what the STL contains)",
                    height=560,
                    clear_color=(0.96, 0.96, 0.97, 1.0),
                    camera_position=(45, 45, None),
                )
                render_btn = gr.Button("Render")
                status = gr.Markdown("drop a photo of the object on the reference sheet")

        controls = [
            px_per_mm,
            auto,
            threshold,
            tolerance,
            min_area,
            clearance,
            height,
            mirror,
            *(bin_widgets[name] for name in BIN_CONTROL_NAMES),
        ]
        assert len(controls) == len(CONTROL_NAMES)
        units_x, units_y = bin_widgets["units_x"], bin_widgets["units_y"]

        load_out = [overlay, threshold, viewer, status, units_x, units_y, stem]
        photo.upload(bind(on_load), [photo, *controls], load_out, **always)
        for ev in (px_per_mm.submit, px_per_mm.blur):
            ev(bind(on_load), [photo, *controls], load_out, **always)
        if photo0:
            app.load(bind(on_load), [photo, *controls], load_out)

        img_out = [overlay, threshold, status]
        for ev in (threshold.input, tolerance.input, min_area.input, auto.input):
            ev(
                bind(on_image_input),
                controls,
                img_out,
                show_progress="hidden",
                **always,
            )

        all_out = [overlay, threshold, viewer, status]
        settle = [
            threshold.release,
            tolerance.release,
            min_area.release,
            auto.input,
            mirror.input,
            render_btn.click,
        ]
        for w in (clearance, height, *bin_widgets.values()):
            settle += settle_events(w)
        for ev in settle:
            ev(bind(on_change), controls, all_out, concurrency_limit=1, **always)

        export_btn.click(bind(on_export), [out_dir, stem, *controls], [files, status], **always)
        print_btn.click(None, js=print_sheet_js(session.spec))

    app.session = session  # type: ignore[attr-defined]
    return app


def launch(photo: str | Path | None = None, port: int = 7860, open_browser: bool = True) -> None:
    app = build_app(photo)
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=open_browser,
        show_error=True,
        quiet=True,
    )
