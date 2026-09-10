"""Command line entry point: gridfinity-cutter sheet|outline|extrude|run|ui."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from gridfinity_cutter import extrude, outline, sheet
from gridfinity_cutter.bins import BACKEND_NAMES, BinParams
from gridfinity_cutter.params import Params, ParamsError

DEFAULT_UI_PORT = 7860


def _cmd_sheet(args: argparse.Namespace) -> int:
    spec = sheet.SheetSpec(paper=args.paper)
    sheet.render_pdf(args.output, spec)
    print(f"wrote {args.output}")
    return 0


def _run_outline(args: argparse.Namespace, image: str, svg: str) -> int:
    try:
        res = outline.run(
            image,
            svg,
            px_per_mm=args.px_per_mm,
            tolerance_mm=args.tolerance,
            threshold=args.threshold,
            min_area_mm2=args.min_area,
            debug_dir=args.debug,
        )
    except outline.OutlineError as e:
        print(f"outline: {e}", file=sys.stderr)
        return 1
    w, h = res.size_mm
    print(
        f"wrote {svg}: {len(res.polygon_mm)} points, bbox {w:.1f} x {h:.1f} mm, "
        f"marker reprojection error {res.reprojection_error_mm:.2f} mm"
    )
    return 0


def _bin_from_args(args: argparse.Namespace) -> BinParams | None:
    if args.bin_units is None:
        return None
    ux, uy = args.bin_units
    ox, oy = args.offset
    return BinParams(
        units_x=int(ux),
        units_y=int(uy),
        units_z=int(args.bin_height),
        offset_x_mm=ox,
        offset_y_mm=oy,
        rotation_deg=args.rotation,
        backend=args.bin_backend,
    )


def _run_extrude(args: argparse.Namespace, svg: str, stl: str) -> int:
    bin_params = _bin_from_args(args)
    try:
        res = extrude.run(
            svg,
            stl,
            height_mm=args.height,
            clearance_mm=args.clearance,
            mirror=args.mirror,
            curve_tolerance_mm=args.curve_tolerance,
            bin=bin_params,
        )
    except (extrude.ExtrudeError, ValueError) as e:
        print(f"extrude: {e}", file=sys.stderr)
        return 1
    w, d, h = res.size_mm
    parts = f", {res.n_parts} parts" if res.n_parts > 1 else ""
    where = ""
    if bin_params is not None:
        bw, bd, bh = bin_params.size_mm
        where = (
            f", in a {bin_params.units_x} x {bin_params.units_y} x {bin_params.units_z} u bin "
            f"({bw:g} x {bd:g} x {bh:g} mm, backend {bin_params.backend})"
        )
    print(
        f"wrote {stl}: {w:.1f} x {d:.1f} x {h:.1f} mm, {res.n_faces} faces, "
        f"clearance {args.clearance:g} mm{parts}{where}"
    )
    return 0


def _cmd_outline(args: argparse.Namespace) -> int:
    return _run_outline(args, args.image, args.output)


def _cmd_extrude(args: argparse.Namespace) -> int:
    return _run_extrude(args, args.svg, args.output)


def _cmd_run(args: argparse.Namespace) -> int:
    svg = args.svg or str(Path(args.output).with_suffix(".svg"))
    rc = _run_outline(args, args.image, svg)
    if rc:
        return rc
    return _run_extrude(args, svg, args.output)


def _cmd_ui(args: argparse.Namespace) -> int:
    try:
        from gridfinity_cutter import ui
    except ImportError as e:
        print(
            f"ui: {e}\nThe web UI needs extra packages; install them with: uv sync --extra ui",
            file=sys.stderr,
        )
        return 1
    ui.launch(photo=args.photo, port=args.port, open_browser=not args.no_browser)
    return 0


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    """Build the parser. ``defaults`` (from ``--params``) override argument defaults
    on the ``extrude`` and ``run`` subcommands; explicit flags still win."""
    p = argparse.ArgumentParser(prog="gridfinity-cutter")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sheet", help="generate the printable reference marker sheet (PDF)")
    s.add_argument("-o", "--output", default="sheet.pdf")
    s.add_argument("--paper", choices=sorted(sheet.PAPER_SIZES), default="a4")
    s.set_defaults(func=_cmd_sheet)

    o = sub.add_parser("outline", help="photo of an object on the reference sheet -> outline SVG")
    o.add_argument("image")
    o.add_argument("-o", "--output", default="object.svg")
    _add_outline_options(o)
    o.set_defaults(func=_cmd_outline)

    e = sub.add_parser("extrude", help="outline SVG (mm units) -> extruded cutter solid STL")
    e.add_argument("svg")
    e.add_argument("-o", "--output", default="object.stl")
    _add_params_option(e)
    _add_extrude_options(e)
    _add_bin_options(e)
    e.set_defaults(func=_cmd_extrude)

    r = sub.add_parser("run", help="photo -> outline SVG -> STL in one go (keeps the SVG)")
    r.add_argument("image")
    r.add_argument("-o", "--output", default="object.stl")
    r.add_argument(
        "--svg",
        metavar="PATH",
        help="where to write the intermediate outline SVG (default: OUTPUT with .svg suffix)",
    )
    _add_params_option(r)
    _add_outline_options(r)
    _add_extrude_options(r)
    _add_bin_options(r)
    r.set_defaults(func=_cmd_run)

    u = sub.add_parser("ui", help="interactive web UI on localhost (needs: uv sync --extra ui)")
    u.add_argument("photo", nargs="?", help="photo to load on start")
    u.add_argument("--port", type=int, default=DEFAULT_UI_PORT)
    u.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    u.set_defaults(func=_cmd_ui)

    if defaults:
        # argparse ignores top-level set_defaults for subparser dests, so apply
        # per subparser, after its arguments exist (later set_defaults wins).
        for sp in (e, r):
            dests = {a.dest for a in sp._actions}
            sp.set_defaults(**{k: v for k, v in defaults.items() if k in dests})
    return p


def _add_params_option(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--params",
        metavar="FILE",
        help="params.json written by the UI; sets the defaults of all options below "
        "(explicit flags still win; a mirror=true in the file cannot be switched off here)",
    )


def _add_outline_options(o: argparse.ArgumentParser) -> None:
    o.add_argument(
        "--px-per-mm",
        type=float,
        default=outline.DEFAULT_PX_PER_MM,
        help="resolution of the rectified sheet image (default %(default)s)",
    )
    o.add_argument(
        "--tolerance",
        type=float,
        default=outline.DEFAULT_TOLERANCE_MM,
        help="outline simplification tolerance in mm (default %(default)s)",
    )
    o.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Lab colour distance from paper that counts as object (default: automatic)",
    )
    o.add_argument(
        "--min-area",
        type=float,
        default=outline.DEFAULT_MIN_AREA_MM2,
        help="ignore blobs smaller than this many mm^2 (default %(default)s)",
    )
    o.add_argument("--debug", metavar="DIR", help="write warp/mask/contour images to DIR")


def _add_extrude_options(e: argparse.ArgumentParser) -> None:
    e.add_argument(
        "--height",
        type=float,
        default=None,
        help="extrusion height (pocket depth) in mm; required unless --params provides it",
    )
    e.add_argument(
        "--clearance",
        type=float,
        default=extrude.DEFAULT_CLEARANCE_MM,
        help="grow the outline by this many mm on every side (negative shrinks); "
        "use it to tune how loose the fit is (default %(default)s)",
    )
    e.add_argument(
        "--mirror",
        action="store_true",
        help="do not flip y; the STL seen from above is the mirror image of the photo",
    )
    e.add_argument(
        "--curve-tolerance",
        type=float,
        default=extrude.DEFAULT_CURVE_TOLERANCE_MM,
        help="max chord error in mm when flattening curves (default %(default)s)",
    )


def _add_bin_options(e: argparse.ArgumentParser) -> None:
    g = e.add_argument_group(
        "bin placement",
        "With --bin-units the STL is written in bin coordinates: x/y from the bin's "
        "grid corner, z from the bin bottom, pocket sunk into the bin top.",
    )
    g.add_argument(
        "--bin-units",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="bin footprint in 42 mm Gridfinity units",
    )
    g.add_argument(
        "--bin-height", type=int, default=3, metavar="U", help="bin height in 7 mm units"
    )
    g.add_argument(
        "--offset",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=[0.0, 0.0],
        help="move the cutout from the bin centre, mm (y towards the far edge)",
    )
    g.add_argument(
        "--rotation",
        type=float,
        default=0.0,
        metavar="DEG",
        help="rotate the cutout counter-clockwise (seen from above)",
    )
    g.add_argument("--bin-backend", choices=BACKEND_NAMES, default="none")


def _params_defaults(argv: list[str] | None) -> dict[str, Any]:
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--params")
    known, _ = pre.parse_known_args(argv)
    if not known.params:
        return {}
    try:
        return Params.from_json(known.params).to_cli_defaults()
    except ParamsError as e:
        print(f"params: {e}", file=sys.stderr)
        raise SystemExit(2) from e


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(_params_defaults(argv))
    args = parser.parse_args(argv)
    if args.command in ("extrude", "run") and args.height is None:
        parser.error("--height is required (or set geometry.height_mm in --params)")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
