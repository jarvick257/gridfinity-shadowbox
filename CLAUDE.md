# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Python tool that turns a top-down photo of an object into a Gridfinity bin with a
shadow-board pocket shaped like that object. The **web UI is the primary
interface**: users load a photo, tune everything with live overlay and 3D preview,
and export the STL from there. The CLI covers the same pipeline headlessly and is
what the UI's `params.json` export replays, but it is the secondary path — design
new features for the UI first and make sure the CLI can reproduce them.

## User workflow

1. Print the reference sheet (the UI's print button, or `shadowbox sheet`).
2. Place the object on the sheet, photograph it straight down.
3. `shadowbox ui photo.jpg`: load the photo, adjust threshold/clearance/depth/
   relief and the bin options while watching the outline overlay and the bin
   preview, then export STL + `params.json`.
4. Headless or scripted: `outline` (photo -> outline SVG) then `extrude`
   (SVG -> STL), or `run` for both; `--params` replays a UI session.

## Layout

```
gridfinity_shadowbox/
  ui.py         # Gradio layer only (optional extra `ui`); wires widgets to session.py
  session.py    # UI logic without any web framework: cached warp, overlay, GLB scene, export
  sheet.py      # generate the printable reference sheet (PDF/SVG), known geometry
  outline.py    # photo -> outline SVG
  extrude.py    # SVG -> STL, incl. placement inside a bin
  bins/         # Gridfinity constants, BinParams, height rules
    geometry.py #   solid Gridfinity bin built with manifold3d, pocket subtracted
  params.py     # params.json schema (ImageParams/GeometryParams/BinParams), CLI defaults
  cli.py        # entry points: shadowbox sheet|outline|extrude|run|ui
tests/
```

The pipeline steps stay standalone and file-based (`outline` writes an SVG that
`extrude` reads), so users can inspect and hand-edit the intermediate SVG and so
`session.py` can drive the same code without a web framework.

## Environment and commands

Use `uv` (installed; the venv is Python 3.12):

```
uv sync --extra ui               # normal setup: deps + Gradio for the web UI
uv sync                          # headless/CLI-only setup
uv run shadowbox ui photo.jpg   # the main interface, http://127.0.0.1:7860
uv run shadowbox sheet -o sheet.pdf
uv run shadowbox outline photo.jpg -o object.svg
uv run shadowbox extrude object.svg --height 20 -o object.stl
uv run shadowbox run photo.jpg --height 20 -o object.stl   # outline + extrude, keeps object.svg
uv run shadowbox extrude object.svg --params object.params.json -o object.stl  # reproduce a UI session
uv run shadowbox extrude object.svg --height 20 --bin-units 2 2 -o bin.stl  # finished bin with pocket
uv run pytest                    # all tests (UI tests skip without the ui extra)
uv run pytest tests/test_outline.py -k calibration   # single test
uv run ruff check . && uv run ruff format .
```

Dependencies: `opencv-contrib-python-headless` (ArUco detection, perspective
transform, contours), `numpy`, `reportlab` (sheet PDF), `svgelements` (SVG
parsing incl. curves/transforms/units), `shapely` (offset + cleanup),
`trimesh` + `mapbox-earcut` (extrude + STL/GLB export), `manifold3d` (bin
construction and bin minus pocket boolean). Optional extra `ui`: `gradio` (6.x; note Gradio 6 moved
`css`/`theme` from `Blocks()` to `launch()`).

## Architecture notes

- **UI split.** `session.py` holds all state and computation and is tested without
  Gradio; `ui.py` only maps widget values to `Params` and back. Only listen to
  user events (`.input`/`.release`/`.submit`/`.blur`), never `.change`, because
  `.change` also fires on programmatic updates (the auto threshold is written back
  into its slider). Marker detection and the paper-colour model run once per
  photo (`Session.load`); parameter changes only re-run `mask_from_features` +
  `extract_outline` or the geometry stage. A new knob is not done until it exists
  in all three places: a widget in `ui.py`, a field in `params.py`, and a flag in
  `cli.py` — the UI is where it is designed, the other two are how it is reproduced.
- **Reference sheet is the single source of truth for scale and tilt.** The sheet
  generator and `outline` must share the same marker geometry (marker IDs, spacing in
  mm). Keep that in one module (`sheet.py`) and import it from `outline`; never
  hardcode marker spacing in two places. Prefer ArUco markers in the corners over
  a plain grid: they give unambiguous correspondences for the homography.
- **outline pipeline order:** detect markers -> compute homography to sheet mm
  coordinates -> warp image (this removes tilt and fixes scale in one step) ->
  mask out marker regions -> edge/contour detection -> pick the largest contour
  not touching the sheet border -> simplify -> write SVG in mm units
  (`viewBox` in mm, 1 user unit = 1 mm) with the outline as a single closed path.
- **extrude must not care about images.** It only reads an SVG whose units are mm,
  applies `--clearance` (uniform mm offset via shapely buffer, negative shrinks) and Gridfinity-related options
  (e.g. snap to 42 mm grid), and extrudes to STL. Keep it usable on hand-drawn SVGs.
- **SVG contract between steps:** mm units, one or more closed paths, no
  transforms. Document any deviation in both step modules. `extrude` accepts a
  superset (curves, basic shapes, transforms, `mm`/`px` units; unit-less = mm)
  so hand-drawn SVGs work. SVG y points down; `extrude` flips y so the STL seen
  from above matches the photo (`--mirror` disables that).
- **Finger relief.** `extrude.ReliefParams` (params.json section `relief`, CLI
  `--relief-*`, UI group "Finger relief") merges round scallops into the offset
  outline in the SVG plane (`extrude.add_relief`, right after the clearance, full
  pocket depth) so fingers can lift the object out. Everything downstream (fit,
  placement, overlay, wall warning, bin boolean) just sees a bigger outline; the
  dataclass lives in `extrude.py` because `params.py` must not be imported back.
- **Bin coordinates.** With bin options (`--bin-units`, or always in the UI) the STL
  is written in bin coordinates: x/y from the bin's grid corner, z from the bin
  bottom, pocket sunk into the top of the bin's *solid part*
  (`z = pocket_top_mm - depth .. pocket_top_mm`; with a stacking lip the solid stops
  1.2 mm below the nominal height). `extrude.placement_matrix` is the single
  definition of where the cutout goes; the UI overlay draws the bin on the photo by
  inverting that matrix, never by hand-derived sign rules. Gridfinity constants
  (42 mm, 7 mm, lip sizes) and the height rules (`bin_height_mm`, `infill_height_mm`,
  mirrors of the library's `height()`/`new_bin()`) live only in `bins/__init__.py`,
  which must not import other project modules at import time.
- **Bin geometry.** With bin options the STL is always the finished bin. Bins are
  always *solid* with our pocket as the only cavity, so `BinParams` holds only
  size/height, lip style (`standard`/`reduced`/`none`) and base hole options;
  option names follow Gridfinity Rebuilt's customizer where they exist. `bins/geometry.py` builds the bin with manifold3d from the spec constants in
  `bins/__init__.py`: every piece is a convex hull of two rounded rectangles at
  different heights (`loft`/`sweep`), a prism, or a cylinder/box, so a bin takes
  ~50 ms. The geometry was verified against Gridfinity Rebuilt 2.0.0 renders (volume
  within 0.2 %, cross-sections within Rebuilt's 0.02 mm wall tolerance); keep it that
  way when touching profiles: the lip tip is rounded (0.6 mm), the lip zone is a
  1.2 mm recess, screw holes end at the base bridge (4.75 mm), with a lowered solid
  top the outer wall (0.95 mm) continues up to the lip only when there is a lip.
  Unioned pieces must *overlap* (`SEAM_OVERLAP_MM`) or share identical outlines;
  pieces that merely touch on a face leave duplicate vertices that trimesh reports
  as non-watertight (Manifold still calls them valid). The bin without the pocket is
  cached per option set (placement fields excluded) so UI slider moves only redo the
  pocket boolean. New bin knobs go into `BinParams` (+ `cli.BIN_HELP`, `ui.BIN_WIDGETS`;
  string-valued fields also into `cli.BIN_CHOICES`); CLI flags `--bin-<field>` and
  params.json keys are generated from the dataclass.
- **params.json** (`params.py`) is the reproducibility contract between UI and CLI:
  the UI export writes it, `--params` on `extrude`/`run` reads it as defaults
  (explicit flags win). Add new knobs to the dataclasses and `to_cli_defaults`,
  not to the CLI alone.
- `tests/test_bins_geometry.py` checks the bin geometry by slicing the Manifold
  (cross-section areas and contour counts at known heights) rather than by mesh
  statistics.
- Real photos as test fixtures are large; keep a couple of small downscaled
  samples in `tests/fixtures/` and test geometry against known object dimensions
  with tolerances rather than exact pixel matches.
