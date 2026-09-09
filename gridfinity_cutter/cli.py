"""Command line entry point: gridfinity-cutter sheet|outline|extrude."""

from __future__ import annotations

import argparse
import sys

from gridfinity_cutter import sheet, step1


def _cmd_sheet(args: argparse.Namespace) -> int:
    spec = sheet.SheetSpec(paper=args.paper)
    sheet.render_pdf(args.output, spec)
    print(f"wrote {args.output}")
    return 0


def _cmd_outline(args: argparse.Namespace) -> int:
    try:
        res = step1.run(
            args.image,
            args.output,
            px_per_mm=args.px_per_mm,
            tolerance_mm=args.tolerance,
            threshold=args.threshold,
            min_area_mm2=args.min_area,
            debug_dir=args.debug,
        )
    except step1.OutlineError as e:
        print(f"outline: {e}", file=sys.stderr)
        return 1
    w, h = res.size_mm
    print(
        f"wrote {args.output}: {len(res.polygon_mm)} points, bbox {w:.1f} x {h:.1f} mm, "
        f"marker reprojection error {res.reprojection_error_mm:.2f} mm"
    )
    return 0


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"{args.command}: not implemented yet", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gridfinity-cutter")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sheet", help="generate the printable reference marker sheet (PDF)")
    s.add_argument("-o", "--output", default="sheet.pdf")
    s.add_argument("--paper", choices=sorted(sheet.PAPER_SIZES), default="a4")
    s.set_defaults(func=_cmd_sheet)

    o = sub.add_parser("outline", help="photo of an object on the reference sheet -> outline SVG")
    o.add_argument("image")
    o.add_argument("-o", "--output", default="object.svg")
    o.add_argument(
        "--px-per-mm",
        type=float,
        default=step1.DEFAULT_PX_PER_MM,
        help="resolution of the rectified sheet image (default %(default)s)",
    )
    o.add_argument(
        "--tolerance",
        type=float,
        default=step1.DEFAULT_TOLERANCE_MM,
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
        default=step1.DEFAULT_MIN_AREA_MM2,
        help="ignore blobs smaller than this many mm^2 (default %(default)s)",
    )
    o.add_argument("--debug", metavar="DIR", help="write warp/mask/contour images to DIR")
    o.set_defaults(func=_cmd_outline)

    e = sub.add_parser("extrude", help="outline SVG -> STL (not implemented)")
    e.add_argument("svg")
    e.add_argument("--height", type=float, required=True)
    e.add_argument("-o", "--output", default="object.stl")
    e.set_defaults(func=_not_implemented)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
