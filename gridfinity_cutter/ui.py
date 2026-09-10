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
from functools import partial, update_wrapper
from pathlib import Path

import gradio as gr

from gridfinity_cutter import extrude, outline, sheet
from gridfinity_cutter.bins import HEIGHT_UNIT_MM
from gridfinity_cutter.params import BinParams, GeometryParams, ImageParams, Params
from gridfinity_cutter.session import Session, SessionError

TITLE = "gridfinity-cutter"
Errors = (outline.OutlineError, extrude.ExtrudeError, SessionError, ValueError)

# Order of the plain widget values every handler receives after the session.
CONTROL_NAMES = (
    "px_per_mm",
    "auto",
    "threshold",
    "tolerance",
    "min_area",
    "clearance",
    "height",
    "mirror",
    "units_x",
    "units_y",
    "units_z",
    "offset_x",
    "offset_y",
    "rotation",
)


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
        bin=BinParams(
            units_x=int(v["units_x"]),
            units_y=int(v["units_y"]),
            units_z=int(v["units_z"]),
            offset_x_mm=float(v["offset_x"]),
            offset_y_mm=float(v["offset_y"]),
            rotation_deg=float(v["rotation"]),
        ),
    )


def _threshold_update(session: Session, params: Params):
    """Show the auto-chosen threshold on the (disabled) slider while auto is on."""
    if params.image.threshold is None:
        return gr.update(value=session.outline(params.image).threshold, interactive=False)
    return gr.update(interactive=True)


# -- handlers -----------------------------------------------------------------


def on_image_input(session: Session, *values):
    """Image parameter moved: overlay + threshold slider + status."""
    params = params_from_values(*values)
    try:
        img = session.overlay(params)
        return img, _threshold_update(session, params), session.status(params)
    except Errors as e:
        return gr.update(), gr.update(), f"error: {e}"


def on_change(session: Session, *values):
    """Any parameter settled: overlay + threshold slider + 3D scene + status."""
    params = params_from_values(*values)
    try:
        img = session.overlay(params)
        thr = _threshold_update(session, params)
        glb = session.render(params)
        _, res = session.build(params)
        return img, thr, str(glb), session.status(params, res)
    except Errors as e:
        return gr.update(), gr.update(), gr.update(), f"error: {e}"


def on_load(session: Session, photo: str | None, *values):
    """Photo chosen (or px/mm changed): rectify, size the bin to fit, then everything."""
    if not photo:
        return (gr.update(),) * 6 + ("drop a photo of the object on the reference sheet",)
    params = params_from_values(*values)
    try:
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
    params = params_from_values(*values)
    stem = (stem or "object").strip()
    try:
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
                with gr.Row():
                    units_x = gr.Slider(label="Units X", minimum=1, maximum=8, step=1, value=1)
                    units_y = gr.Slider(label="Units Y", minimum=1, maximum=8, step=1, value=1)
                units_z = gr.Slider(
                    label=f"Height units ({HEIGHT_UNIT_MM:g} mm)",
                    minimum=1,
                    maximum=12,
                    step=1,
                    value=3,
                )
                with gr.Row():
                    offset_x = gr.Number(label="Offset X (mm)", value=0.0, step=0.5)
                    offset_y = gr.Number(label="Offset Y (mm)", value=0.0, step=0.5)
                rotation = gr.Number(label="Rotation (deg, CCW)", value=0.0, step=5)
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
                    label="Cutout in bin (viewer only; STL is the cutout)",
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
            units_x,
            units_y,
            units_z,
            offset_x,
            offset_y,
            rotation,
        ]
        assert len(controls) == len(CONTROL_NAMES)

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
            units_x.release,
            units_y.release,
            units_z.release,
            render_btn.click,
        ]
        settle += [
            ev
            for w in (clearance, height, offset_x, offset_y, rotation)
            for ev in (w.submit, w.blur)
        ]
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
