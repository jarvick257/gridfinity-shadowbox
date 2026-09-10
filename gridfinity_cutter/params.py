"""``params.json``: every knob of the pipeline in one reproducible file.

The UI writes this next to its exports and the CLI reads it with ``--params``,
so a session tuned in the browser can be re-run from the command line. Keys
are nested by stage; unknown keys are an error so typos do not silently fall
back to defaults. ``"threshold": null`` means automatic.

Nothing imports this module back (``outline``, ``extrude`` and ``bins`` are
its dependencies, not its clients).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from gridfinity_cutter import extrude, outline
from gridfinity_cutter.bins import BACKEND_NAMES, BinParams

__all__ = [
    "BIN_GENERIC_FIELDS",
    "BinParams",
    "GeometryParams",
    "ImageParams",
    "Params",
    "ParamsError",
]

# BinParams fields with their own CLI flags; every other field gets a generated
# ``--bin-<name>`` flag with argparse dest ``bin_<name>`` (see cli._add_bin_options).
BIN_SPECIAL_FIELDS = frozenset(
    {"units_x", "units_y", "gridz", "offset_x_mm", "offset_y_mm", "rotation_deg", "backend"}
)
BIN_GENERIC_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(BinParams) if f.name not in BIN_SPECIAL_FIELDS
)


class ParamsError(ValueError):
    """Raised for a malformed params file."""


@dataclass(frozen=True)
class ImageParams:
    px_per_mm: float = outline.DEFAULT_PX_PER_MM
    threshold: float | None = None  # None = automatic
    tolerance_mm: float = outline.DEFAULT_TOLERANCE_MM
    min_area_mm2: float = outline.DEFAULT_MIN_AREA_MM2


@dataclass(frozen=True)
class GeometryParams:
    height_mm: float = 10.0
    clearance_mm: float = extrude.DEFAULT_CLEARANCE_MM
    mirror: bool = False
    curve_tolerance_mm: float = extrude.DEFAULT_CURVE_TOLERANCE_MM


@dataclass(frozen=True)
class Params:
    image: ImageParams = field(default_factory=ImageParams)
    geometry: GeometryParams = field(default_factory=GeometryParams)
    bin: BinParams = field(default_factory=BinParams)
    photo: str | None = None  # file name of the photo the parameters were tuned on

    # -- JSON ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Params:
        if not isinstance(data, dict):
            raise ParamsError("params must be a JSON object")
        sections = {"image": ImageParams, "geometry": GeometryParams, "bin": BinParams}
        unknown = set(data) - set(sections) - {"photo"}
        if unknown:
            raise ParamsError(f"unknown top-level key(s): {sorted(unknown)}")
        kwargs: dict[str, Any] = {"photo": data.get("photo")}
        for name, klass in sections.items():
            kwargs[name] = _section(klass, name, data.get(name, {}))
        params = cls(**kwargs)
        if params.bin.backend not in BACKEND_NAMES:
            raise ParamsError(
                f"bin.backend {params.bin.backend!r} unknown; known: {', '.join(BACKEND_NAMES)}"
            )
        return params

    @classmethod
    def from_json(cls, path: str | Path) -> Params:
        try:
            data = json.loads(Path(path).read_text())
        except OSError as e:
            raise ParamsError(f"cannot read {path}: {e}") from e
        except json.JSONDecodeError as e:
            raise ParamsError(f"{path}: invalid JSON: {e}") from e
        return cls.from_dict(data)

    # -- CLI ----------------------------------------------------------------

    def to_cli_defaults(self) -> dict[str, Any]:
        """Values keyed by argparse ``dest`` names of the ``extrude``/``run`` subcommands."""
        i, g, b = self.image, self.geometry, self.bin
        return {
            "px_per_mm": i.px_per_mm,
            "threshold": i.threshold,
            "tolerance": i.tolerance_mm,
            "min_area": i.min_area_mm2,
            "height": g.height_mm,
            "clearance": g.clearance_mm,
            "mirror": g.mirror,
            "curve_tolerance": g.curve_tolerance_mm,
            "bin_units": [b.units_x, b.units_y],
            "bin_height": b.gridz,
            "offset": [b.offset_x_mm, b.offset_y_mm],
            "rotation": b.rotation_deg,
            "bin_backend": b.backend,
            **{f"bin_{name}": getattr(b, name) for name in BIN_GENERIC_FIELDS},
        }


def _section(klass: type, name: str, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ParamsError(f"{name!r} must be a JSON object")
    known = {f.name for f in fields(klass)}
    unknown = set(data) - known
    if unknown:
        raise ParamsError(f"unknown key(s) in {name!r}: {sorted(unknown)}")
    try:
        return klass(**data)
    except (TypeError, ValueError) as e:
        raise ParamsError(f"bad {name!r} section: {e}") from e
