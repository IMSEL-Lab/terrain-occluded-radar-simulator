"""Render manually arranged coastlines and vessels from a scene JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import radar_simulator as engine


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotations", type=int, default=1)
    parser.add_argument(
        "--coastline-renderer",
        choices=("ray-polygon", "filled-polygon"),
        default="ray-polygon",
    )
    parser.add_argument(
        "--time", type=datetime.fromisoformat, default=datetime(2000, 1, 1)
    )
    args = parser.parse_args(argv)
    if not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("seed must be an integer from 0 to 4294967295")
    if args.rotations < 1:
        raise ValueError("--rotations must be at least one")
    if args.output.exists():
        raise FileExistsError(args.output)
    coastlines, objects, scene = engine.load_scene_from_file(str(args.input))
    engine._prepare_scene_for_sim_mode(
        scene, objects, engine.SIM_MODE_M1, verbose=False
    )
    scene["_sim_seed"] = args.seed
    scene.pop("mode12_metadata", None)
    scene.pop("source_file", None)
    scene["coastline_render_mode"] = (
        "filled_polygon"
        if args.coastline_renderer == "filled-polygon"
        else "ray_polygon"
    )
    engine.MODE4_USE_OBJECT_LEARNED_MODEL = False
    args.output.mkdir(parents=True)
    scene_path = args.output / "scene.json"
    scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    paths = engine.simulate_from_scene(
        scene,
        coastlines,
        objects,
        str(args.output / "radar"),
        scene_path=scene_path.name,
        num_rotations=args.rotations,
        verbose=False,
        base_time=args.time,
    )
    receipt = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "seed": args.seed,
        "coastline_renderer": args.coastline_renderer,
        "outputs": paths,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
