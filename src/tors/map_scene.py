"""Generate a radar sweep from a map boundary and explicit scan settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon

import radar_simulator as engine


def _scene(args: argparse.Namespace) -> dict[str, Any]:
    if args.grid_scene is not None:
        data = json.loads(args.grid_scene.read_text(encoding="utf-8"))
        allowed = ("radar_config", "angle_ticks", "decay", "clutter", "_sim_seed")
        scene = {key: data[key] for key in allowed if key in data}
        if "radar_config" not in scene or "angle_ticks" not in scene:
            raise ValueError("--grid-scene must supply radar_config and angle_ticks")
    else:
        if not math.isfinite(args.range_m) or args.range_m <= 0:
            raise ValueError("--range-m must be positive and finite")
        if args.bins < 2 or not 1 <= args.spokes <= 8192:
            raise ValueError("use at least two bins and 1 to 8192 spokes")
        scene = {
            "radar_config": {
                "status": 1,
                "scale": 496,
                "range_index": 0,
                "gain": 75,
                "bins": args.bins,
                "bin_size_m": args.range_m / args.bins,
                "dead_zone_bins": min(5, args.bins - 1),
                "range_m": args.range_m,
                "ticks_per_rev": 8192,
            },
            "angle_ticks": [
                round(index * 8192 / args.spokes) for index in range(args.spokes)
            ],
            "clutter": {"noise_probability": 0.0, "noise_codes": [8, 20, 28]},
            "decay": {"ramp_up_range": [80, 200], "ramp_down_range": [40, 160]},
        }
    seed = scene.get("_sim_seed", 42) if args.seed is None else args.seed
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("seed must be an integer from 0 to 4294967295")
    scene["_sim_seed"] = seed
    scene["_sim_mode"] = engine.SIM_MODE_M1
    scene["objects"] = []
    return scene


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument(
        "--lat", type=float, required=True, help="Radar latitude in degrees"
    )
    parser.add_argument(
        "--lon", type=float, required=True, help="Radar longitude in degrees"
    )
    parser.add_argument("--range-m", type=float, default=150)
    parser.add_argument("--spokes", type=int, default=400)
    parser.add_argument("--bins", type=int, default=300)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--grid-scene",
        type=Path,
        help="Use scan settings and bearings from a scene JSON",
    )
    parser.add_argument(
        "--time", type=datetime.fromisoformat, default=datetime(2000, 1, 1)
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not -90 <= args.lat <= 90 or not -180 <= args.lon <= 180:
        raise ValueError("latitude and longitude must be in valid degree ranges")
    if args.output.exists():
        raise FileExistsError(args.output)
    scene = _scene(args)
    coastlines = engine.load_coastlines_from_geojson(
        str(args.geojson),
        args.lat,
        args.lon,
        max_range_m=scene["radar_config"]["range_m"],
    )
    if not coastlines:
        raise ValueError("the map has no shoreline inside the selected range")
    for segment in coastlines:
        polygon = Polygon(segment.points)
        if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
            raise ValueError("the map produces an invalid shoreline polygon")
    scene["coastlines"] = [
        {"name": segment.name, "points": segment.points} for segment in coastlines
    ]
    args.output.mkdir(parents=True)
    scene_path = args.output / "scene.json"
    scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    engine.MODE4_USE_OBJECT_LEARNED_MODEL = False
    paths = engine.simulate_from_scene(
        scene,
        coastlines,
        [],
        str(args.output / "radar"),
        scene_path=scene_path.name,
        num_rotations=1,
        verbose=False,
        base_time=args.time,
    )
    receipt = {
        "source_geojson_sha256": hashlib.sha256(args.geojson.read_bytes()).hexdigest(),
        "scene": str(scene_path),
        "output": paths,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
