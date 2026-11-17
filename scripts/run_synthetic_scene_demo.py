"""Reproduce a single seeded synthetic scene with explicitly specified template."""

import argparse
import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

os.environ["MODE4_USE_OBJECT_LEARNED_MODEL"] = "0"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template", type=Path, default=repo / "examples/synthetic-template.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vessels-per-class", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    template, output = args.template.resolve(), args.output.resolve()
    if not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("seed must be an integer from 0 to 4294967295")
    if not 0 <= args.vessels_per_class <= 40:
        raise ValueError("--vessels-per-class must be between 0 and 40")
    if not template.is_file():
        raise FileNotFoundError(template)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    import radar_simulator as sim

    def no_pickle(*args, **kwargs):
        raise RuntimeError("pickle loading is prohibited in scene generation")

    setattr(pickle, "load", no_pickle)  # noqa: B010
    sim.MODE4_USE_OBJECT_LEARNED_MODEL = False
    setattr(  # noqa: B010
        sim, "_discover_scene_template_files", lambda *args, **kwargs: [str(template)]
    )
    template_calls = []
    original_template = sim.generate_coastlines_from_scene_template

    def tagged_template(files, max_range_m):
        assert files == [str(template)]
        result = original_template(files, max_range_m)
        template_calls.append({"file": str(template), "features": len(result[0])})
        return result

    setattr(sim, "generate_coastlines_from_scene_template", tagged_template)  # noqa: B010

    def no_library(*args, **kwargs):
        return None

    setattr(sim, "build_coastline_library", no_library)  # noqa: B010
    original_render = sim.simulate_from_scene

    def capture_and_render(scene, coastlines, objects, *args, **kwargs):
        if not scene["coastlines"]:
            raise ValueError("the selected template generated no usable coastlines")
        (output / "scene.json").write_text(
            json.dumps(scene, indent=2, allow_nan=False) + "\n"
        )
        geometry = {
            "coordinate_frame": "x east/right, y north/up; heading clockwise from north",
            "range_m": scene["radar_config"]["range_m"],
            "coastlines": scene["coastlines"],
            "objects": scene["objects"],
            "object_representation": "center, length, width and heading are simulator inputs; no polygon hull supplied to renderer",
            "coastline_source": "seeded transform of explicitly supplied template",
        }
        (output / "geometry.json").write_text(
            json.dumps(geometry, indent=2, allow_nan=False) + "\n"
        )
        return original_render(scene, coastlines, objects, *args, **kwargs)

    setattr(sim, "simulate_from_scene", capture_and_render)  # noqa: B010
    count = args.vessels_per_class
    config = sim.SceneConfig(
        num_buoys=(0, 0),
        num_small_boats=(count, count),
        num_medium_boats=(count, count),
        num_large_boats=(count, count),
        num_barges=(count, count),
        coastline_probability=1.0,
    )
    started = time.time()
    paths = sim.simulate_scene(
        str(output / "radar"),
        config=config,
        num_rotations=1,
        base_time=datetime(2026, 11, 16, 12, 0, 0),
        verbose=True,
        seed=args.seed,
        no_implicit_capture=False,
    )
    elapsed = time.time() - started
    radar_paths = [
        Path(path)
        for path in paths
        if Path(path).parent == output / "radar" and Path(path).suffix == ".csv"
    ]
    assert len(radar_paths) == 1 and len(paths) == 3 and template_calls
    data = {
        "seed": args.seed,
        "rotations": 1,
        "base_time_synthetic": "2026-11-16T12:00:00",
        "config": asdict(config),
        "engine_sha256": sha(repo / "radar_simulator.py"),
        "object_renderer_sha256": sha(repo / "object_renderer.py"),
        "template_sha256": sha(template),
        "template_file": str(template),
        "template_description": "Explicit coastline geometry and optional aggregate echo statistics; no measured sweep is replayed",
        "template_calls": template_calls,
        "template_discovery": "explicit one-file allowlist; library fallback prohibited",
        "pickle_loading": "disabled and guarded",
        "output_csv": str(radar_paths[0]),
        "output_csv_sha256": sha(radar_paths[0]),
        "all_output_paths": paths,
        "scene_sha256": sha(output / "scene.json"),
        "geometry_sha256": sha(output / "geometry.json"),
        "elapsed_seconds": elapsed,
    }
    (output / "run_manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
