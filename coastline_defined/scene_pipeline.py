"""Shared Mode C configuration and source adapter; no radar equations live here."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def apply_launch_profile() -> dict[str, str]:
    profile = json.loads((ROOT / "rustcore_launch_profile.json").read_text(encoding="utf-8"))
    for name, value in profile["environment"].items():
        os.environ.setdefault(name, value)
    return {name: os.environ[name] for name in profile["environment"]}


def resolve_source(simulator, output_dir, location=None, range_nm=None, radarloc_file=None,
                   prefer_source=None, verbose=True):
    if radarloc_file:
        path = Path(radarloc_file).resolve()
        if not path.is_file():
            raise ValueError(f"Location file does not exist: {path}")
        return path
    loc, features, _, _, validation = simulator._load_mode12_source(
        location, range_nm=range_nm, prefer_source=prefer_source, verbose=verbose, full_range=True)
    if not loc or not features:
        raise ValueError("No real shoreline was resolved. Select a .radarloc or check source connectivity; no synthetic coastline was substituted.")
    folder = Path(output_dir) / "resolved_locations"
    folder.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    path = folder / ("modec_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".radarloc")
    docs = []
    for index, feature in enumerate(features):
        doc = dict(feature)
        doc.setdefault("id", f"source_{index}")
        doc["points"] = [dict(x=p["x"], y=p["y"]) if isinstance(p, dict)
                         else dict(x=p[0], y=p[1]) for p in feature["points"]]
        docs.append(doc)
    path.write_text(json.dumps({"version": "1.0", "metadata": {"location_name": loc["name"],
        "center_lat": loc["lat"], "center_lon": loc["lon"],
        "range_nm": range_nm if range_nm is not None else loc["range_nm"],
        "source_type": loc.get("source_type"), "source_path": loc.get("source_path"),
        "source_validation": validation}, "coastlines": docs, "vessels": []}, indent=2), encoding="utf-8")
    return path


def run_modec(simulator, output_dir, location=None, num_scenes=1, num_rotations=1,
              range_nm=None, radarloc_file=None, prefer_source=None, verbose=True,
              options=None):
    from . import run_coastline_defined as runner
    source = resolve_source(simulator, output_dir, location, range_nm, radarloc_file,
                            prefer_source, verbose)
    args = runner.build_parser().parse_args(["--radarloc", str(source), "--output-dir", str(output_dir)])
    args.scenes, args.rotations = num_scenes, num_rotations
    args.strict_tick_raytrace = False
    if range_nm is not None:
        args.range_nm = range_nm
    for name, value in (options or {}).items():
        if not hasattr(args, name):
            raise ValueError(f"Unknown Mode C setting: {name}")
        if value is not None:
            setattr(args, name, value)
    return runner.run(args)
