"""Backend 7b: a real Gridfinity bin rendered by OpenSCAD (Gridfinity Rebuilt), pocket cut here.

OpenSCAD only ever renders the *bin* from the bin options in ``BinParams``
(``bin.scad`` next to this file wraps the vendored library). The result is
cached on disk per option set, and the object pocket is subtracted in Python
with the manifold boolean engine, so changing the cutout's placement, depth or
clearance never re-runs OpenSCAD.

Requirements: an OpenSCAD development build (2024 or newer; the library uses
syntax that 2021.01 cannot parse) on ``PATH`` as ``openscad-nightly`` or
``openscad``, or pointed to by ``$GRIDFINITY_CUTTER_OPENSCAD``; and the
``gridfinity-rebuilt-openscad`` submodule checked out.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import manifold3d
import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon

from gridfinity_cutter import extrude
from gridfinity_cutter.bins import BinParams, BinResult, bin_height_mm, infill_height_mm

HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "bin.scad"
LIBRARY_DIR = HERE / "gridfinity-rebuilt-openscad"
LIBRARY_MAIN = LIBRARY_DIR / "src" / "core" / "bin.scad"

ENV_OPENSCAD = "GRIDFINITY_CUTTER_OPENSCAD"
ENV_CACHE = "GRIDFINITY_CUTTER_CACHE"
CANDIDATES = ("openscad-nightly", "openscad")
MIN_YEAR = 2024
RENDER_TIMEOUT_S = 120.0
PROBE_TIMEOUT_S = 20.0
POCKET_OVERSHOOT_MM = 1.0  # cutter reaches above the solid top so no float32 skin remains
ECHO_TOLERANCE_MM = 1e-3

# BinParams field -> variable in bin.scad. The single source for that mapping.
SCAD_VARS: dict[str, str] = {
    "units_x": "gridx",
    "units_y": "gridy",
    "gridz": "gridz",
    "half_grid": "half_grid",
    "gridz_define": "gridz_define",
    "height_internal_mm": "height_internal",
    "enable_zsnap": "enable_zsnap",
    "include_lip": "include_lip",
    "divx": "divx",
    "divy": "divy",
    "depth_mm": "depth",
    "cut_cylinders": "cut_cylinders",
    "cylinder_diameter_mm": "cd",
    "cylinder_chamfer_mm": "c_chamfer",
    "style_tab": "style_tab",
    "place_tab": "place_tab",
    "scoop": "scoop",
    "only_corners": "only_corners",
    "refined_holes": "refined_holes",
    "magnet_holes": "magnet_holes",
    "screw_holes": "screw_holes",
    "crush_ribs": "crush_ribs",
    "chamfer_holes": "chamfer_holes",
    "printable_hole_top": "printable_hole_top",
    "enable_thumbscrew": "enable_thumbscrew",
}


class OpenScadError(extrude.ExtrudeError):
    """OpenSCAD missing, too old, failed to render, or produced unusable geometry."""


# -- parameters -> command line -------------------------------------------


def scad_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OpenScadError(f"non-finite value {value!r}")
        return repr(value)
    raise OpenScadError(f"cannot pass {value!r} to OpenSCAD")


def scad_defines(p: BinParams) -> list[str]:
    """``-D name=value`` for every bin option, in ``SCAD_VARS`` order."""
    return [f"-D{var}={scad_literal(getattr(p, field))}" for field, var in SCAD_VARS.items()]


# -- locating OpenSCAD ------------------------------------------------------


@dataclass(frozen=True)
class OpenScad:
    exe: str
    version: str
    year: int
    manifold: bool


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def probe(exe: str) -> OpenScad | None:
    """Version and capabilities of an OpenSCAD binary, ``None`` if it is not usable."""
    try:
        out = _run([exe, "--version"], PROBE_TIMEOUT_S)
        m = re.search(r"OpenSCAD version (\d{4})\.(\d\d)(\S*)", out.stdout + out.stderr)
        if not m:
            return None
        help_text = _run([exe, "--help"], PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    year = int(m.group(1))
    manifold = "--backend" in help_text.stdout + help_text.stderr
    return OpenScad(
        exe=exe, version=m.group(0).removeprefix("OpenSCAD version "), year=year, manifold=manifold
    )


@lru_cache(maxsize=1)
def find_openscad() -> OpenScad | None:
    """First usable binary: ``$GRIDFINITY_CUTTER_OPENSCAD``, ``openscad-nightly``, ``openscad``."""
    env = os.environ.get(ENV_OPENSCAD)
    candidates = [env] if env else [shutil.which(c) for c in CANDIDATES]
    for exe in candidates:
        if not exe:
            continue
        tool = probe(exe)
        if tool is not None and tool.year >= MIN_YEAR:
            return tool
    return None


def library_present() -> bool:
    return LIBRARY_MAIN.is_file()


def available() -> bool:
    return library_present() and find_openscad() is not None


def unavailable_reason() -> str | None:
    if not library_present():
        return (
            "the Gridfinity Rebuilt sources are missing; run "
            "`git submodule update --init` in the gridfinity-cutter checkout"
        )
    if find_openscad() is None:
        return (
            f"no OpenSCAD {MIN_YEAR}+ development build found (looked at ${ENV_OPENSCAD}, "
            f"{', '.join(CANDIDATES)}); install e.g. openscad-git or openscad-snapshot-appimage "
            f"from the AUR, or set ${ENV_OPENSCAD} to the binary"
        )
    return None


def require() -> OpenScad:
    reason = unavailable_reason()
    if reason:
        raise OpenScadError(f"bin backend 'openscad' unavailable: {reason}")
    tool = find_openscad()
    assert tool is not None
    return tool


# -- cache ------------------------------------------------------------------


def cache_dir() -> Path:
    if env := os.environ.get(ENV_CACHE):
        base = Path(env)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = (Path(xdg) if xdg else Path.home() / ".cache") / "gridfinity-cutter"
    return base / "bins"


@lru_cache(maxsize=1)
def library_digest() -> str:
    """Hash of every .scad file that can influence a render (submodule + wrapper)."""
    h = hashlib.sha256()
    files = sorted(LIBRARY_DIR.glob("src/**/*.scad")) + [WRAPPER]
    for f in files:
        h.update(str(f.relative_to(HERE)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def cache_key(p: BinParams, tool: OpenScad) -> str:
    payload = {"lib": library_digest(), "openscad": tool.version, "defines": scad_defines(p)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# -- rendering --------------------------------------------------------------

_ECHO_RE = re.compile(r'^ECHO: "GFC_(\w+)=(.*)"$', re.MULTILINE)


def parse_echo(text: str) -> dict[str, Any]:
    """``GFC_NAME=value`` lines echoed by ``bin.scad`` (numbers or vectors, JSON-parsable)."""
    out: dict[str, Any] = {}
    for m in _ECHO_RE.finditer(text):
        try:
            out[m.group(1)] = json.loads(m.group(2))
        except json.JSONDecodeError:
            out[m.group(1)] = m.group(2)
    return out


def _problem_lines(text: str, limit: int = 20) -> str:
    lines = [ln for ln in text.splitlines() if re.match(r"(ERROR|WARNING|TRACE|Assert)", ln)]
    if not lines:
        lines = text.splitlines()[-limit:]
    return "\n".join(lines[-limit:])


def render_command(p: BinParams, tool: OpenScad, output: Path) -> list[str]:
    cmd = [tool.exe, "-o", str(output), "--export-format", "binstl"]
    if tool.manifold:
        cmd.append("--backend=manifold")
    cmd += scad_defines(p)
    cmd.append(str(WRAPPER))
    return cmd


def check_echo(p: BinParams, echo: dict[str, Any]) -> None:
    """The Python height rules must agree with the library's, or placement would be wrong."""
    try:
        height = float(echo["HEIGHT_MM"])
        infill = float(echo["INFILL_MM"][2])
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise OpenScadError(f"bin.scad did not echo its dimensions ({e!r})") from e
    expected = ((height, bin_height_mm(p), "height"), (infill, infill_height_mm(p), "infill"))
    for got, want, what in expected:
        if abs(got - want) > ECHO_TOLERANCE_MM:
            raise OpenScadError(
                f"OpenSCAD reports a bin {what} of {got:g} mm but gridfinity_cutter.bins "
                f"computed {want:g} mm; the height rules are out of sync"
            )


def render_bin(p: BinParams) -> tuple[Path, dict[str, Any]]:
    """Path to the bin STL (bin coordinates, no pocket) and its metadata; cached on disk."""
    tool = require()
    key = cache_key(p, tool)
    d = cache_dir()
    stl, meta_path = d / f"{key}.stl", d / f"{key}.json"
    if stl.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        meta["cached"] = True
        return stl, meta

    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{key}.{os.getpid()}.tmp.stl"
    cmd = render_command(p, tool, tmp)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=RENDER_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        tmp.unlink(missing_ok=True)
        raise OpenScadError(f"OpenSCAD render timed out after {RENDER_TIMEOUT_S:g} s") from e
    except OSError as e:
        raise OpenScadError(f"cannot run OpenSCAD ({tool.exe}): {e}") from e
    render_s = time.perf_counter() - t0
    output = proc.stdout + proc.stderr
    if proc.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise OpenScadError(
            f"OpenSCAD failed (exit {proc.returncode}):\n{_problem_lines(output)}\n"
            f"command: {' '.join(cmd)}"
        )
    echo = parse_echo(output)
    try:
        check_echo(p, echo)
    except OpenScadError:
        tmp.unlink(missing_ok=True)
        raise
    meta = {
        "defines": scad_defines(p),
        "openscad": tool.version,
        "echo": echo,
        "render_s": render_s,
        "created": time.time(),
        "cached": False,
    }
    os.replace(tmp, stl)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return stl, meta


def to_manifold(mesh: trimesh.Trimesh, what: str) -> manifold3d.Manifold:
    """Manifold's own validity check is the right one: OpenSCAD output can share an edge
    between four faces (a compartment top meeting the infill top), which trimesh's
    ``is_watertight`` rejects but is a valid solid."""
    m = manifold3d.Manifold(
        mesh=manifold3d.Mesh(
            vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32),
        )
    )
    if m.status() != manifold3d.Error.NoError or m.is_empty() or m.volume() <= 0:
        raise OpenScadError(f"{what} is not a closed solid ({m.status()})")
    return m


def from_manifold(m: manifold3d.Manifold) -> trimesh.Trimesh:
    out = m.to_mesh()
    return trimesh.Trimesh(vertices=out.vert_properties[:, :3], faces=out.tri_verts, process=False)


@lru_cache(maxsize=8)
def _load_bin(path: str, mtime_ns: int) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise OpenScadError(f"OpenSCAD output {path} is empty")
    to_manifold(mesh, f"OpenSCAD output {path}")
    return mesh


def load_bin(path: Path) -> trimesh.Trimesh:
    return _load_bin(str(path), path.stat().st_mtime_ns).copy()


# -- backend ----------------------------------------------------------------


class OpenScadBackend:
    """Finished bin: Gridfinity Rebuilt bin from OpenSCAD minus the object pocket."""

    name = "openscad"

    def build(self, cutout: Polygon | MultiPolygon, height_mm: float, bin: BinParams) -> BinResult:
        stl, meta = render_bin(bin)
        body = load_bin(stl)
        top = bin.pocket_top_mm
        cutter = extrude.extrude_geometry(
            cutout, height_mm + POCKET_OVERSHOOT_MM, z0=top - height_mm
        )
        body_m = to_manifold(body, "bin")
        result = body_m - to_manifold(cutter, "pocket")
        if result.status() != manifold3d.Error.NoError or result.is_empty():
            raise OpenScadError("boolean bin minus pocket produced an empty or invalid mesh")
        if not result.volume() < body_m.volume():
            raise OpenScadError("the pocket does not touch the bin; check offset and bin size")
        solid = from_manifold(result)
        how = "cached" if meta.get("cached") else f"rendered in {meta.get('render_s', 0.0):.1f} s"
        note = f"OpenSCAD {meta.get('openscad', '?')}: bin {how}, {len(solid.faces)} faces"
        return BinResult(solid=solid, context=None, note=note)
