#!/usr/bin/env python3
"""TORS location-file rendering.

Loads a user-provided or newly generated ``.radarloc`` file, applies explicit
radar/noise settings, seeds random targets, and exports a
Furuno-format full-rotation CSV using the bundled location runtime.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np

try:
    from .scene_pipeline import apply_launch_profile
except ImportError:
    from scene_pipeline import apply_launch_profile

apply_launch_profile()


THIS_DIR = Path(__file__).resolve().parent
OBJECT_CREATION_DIR = THIS_DIR.parent
RUNTIME_DIR = THIS_DIR / "runtime"
DEFAULT_OUTPUT_ROOT = OBJECT_CREATION_DIR / "outputs" / "coastline_defined"
DEFAULTS_PATH = THIS_DIR / "defaults.json"

if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from radar_core import rust_bridge  # noqa: E402
from radar_sim.core.simulation import Simulation  # noqa: E402
from radar_sim.environment.attenuation import ATTENUATION_MODELS  # noqa: E402
from radar_sim.objects.vessel import Vessel, VesselType  # noqa: E402
from radar_sim.scenarios.scenario_manager import ScenarioManager  # noqa: E402


NM_TO_M = 1852.0
EARTH_RADIUS_M = 6_371_000.0

NOISE_RANGES = {
    "gain": "0.0 to 1.0",
    "sea_clutter": "0.0 to 1.0",
    "rain_clutter": "0.0 to 1.0",
    "sea_state": "0, 2, 4, 6, or 8",
    "rain_rate_mmh": "0.0 to 50.0",
    "receiver_noise_enabled": "off/on",
    "receiver_noise_level": "0.0000 to 0.0200",
}

MODE1_OBJECT_RANGES = {
    "buoy": (0, 5),
    "small_boat": (0, 4),
    "medium_boat": (0, 3),
    "large_boat": (0, 2),
    "barge": (0, 1),
}

OBJECT_SPECS = {
    "buoy": {
        "vessel_type": VesselType.BUOY,
        "name": "Buoy",
        "length": (1.0, 3.0),
        "beam": (1.0, 3.0),
        "height": (3.0, 5.0),
        "rcs": (5.0, 50.0),
        "speed": (0.0, 0.0),
    },
    "small_boat": {
        "vessel_type": VesselType.FISHING,
        "name": "Small Boat",
        "length": (4.0, 10.0),
        "beam": (2.0, 4.0),
        "height": (4.0, 7.0),
        "rcs": (5.0, 200.0),
        "speed": (0.0, 12.0),
    },
    "medium_boat": {
        "vessel_type": VesselType.FISHING,
        "name": "Medium Boat",
        "length": (10.0, 25.0),
        "beam": (4.0, 8.0),
        "height": (6.0, 10.0),
        "rcs": (50.0, 2000.0),
        "speed": (0.0, 18.0),
    },
    "large_boat": {
        "vessel_type": VesselType.CARGO,
        "name": "Large Boat",
        "length": (25.0, 80.0),
        "beam": (8.0, 20.0),
        "height": (10.0, 20.0),
        "rcs": (300.0, 5000.0),
        "speed": (2.0, 15.0),
    },
    "barge": {
        "vessel_type": VesselType.TANKER,
        "name": "Barge",
        "length": (30.0, 80.0),
        "beam": (10.0, 20.0),
        "height": (4.0, 9.0),
        "rcs": (500.0, 10000.0),
        "speed": (0.0, 6.0),
    },
}


def _load_defaults() -> dict[str, Any]:
    if DEFAULTS_PATH.is_file():
        with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_parser() -> argparse.ArgumentParser:
    defaults = _load_defaults()
    parser = argparse.ArgumentParser(
        description="Render a TORS .radarloc location file."
    )
    parser.add_argument("--radarloc", help="Path to the source .radarloc file; never re-queried")
    parser.add_argument("--location", help="Place or lat,lon to resolve when no .radarloc is supplied")
    parser.add_argument("--prefer-source", default=None)
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root or session directory (default: outputs/coastline_defined)",
    )
    parser.add_argument("--output-csv", default=None, help="Optional explicit output CSV path")
    parser.add_argument("--range-nm", type=float, default=defaults.get("range_nm"))
    parser.add_argument("--origin-xy", default=None, help="Radar origin override as local meters: x,y")
    parser.add_argument("--radar-lat", type=float, default=None, help="Radar latitude override")
    parser.add_argument("--radar-lon", type=float, default=None, help="Radar longitude override")
    parser.add_argument("--gain", type=float, default=defaults.get("gain", 0.65))
    parser.add_argument("--sea-clutter", type=float, default=defaults.get("sea_clutter", 0.0))
    parser.add_argument("--rain-clutter", type=float, default=defaults.get("rain_clutter", 0.0))
    parser.add_argument("--sea-state", type=int, default=defaults.get("sea_state", 0))
    parser.add_argument("--rain-rate-mmh", type=float, default=defaults.get("rain_rate_mmh", 0.0))
    parser.add_argument("--receiver-noise", action="store_true", default=False)
    parser.add_argument("--receiver-noise-level", type=float, default=defaults.get("receiver_noise_level", 0.0))
    parser.add_argument(
        "--attenuation-model",
        choices=sorted(ATTENUATION_MODELS),
        default=defaults.get("attenuation_model", "entry_r4_depth"),
    )
    parser.add_argument("--spokes", type=int, default=defaults.get("spokes", 1820))
    parser.add_argument("--raytrace-spokes", type=int, default=defaults.get("raytrace_spokes"))
    parser.add_argument(
        "--beam-width-deg", type=float, default=None,
        help="Override the horizontal beam width used by the coastline ray renderer.",
    )
    parser.add_argument(
        "--strict-tick-raytrace", action=argparse.BooleanOptionalAction, default=True,
        help="Ray-trace every output Furuno tick. Disable only for a lower-density preview.",
    )
    parser.add_argument(
        "--object-profile",
        choices=["mode1", "fixed", "none"],
        default=defaults.get("object_profile", "mode1"),
        help="mode1 samples class-specific count ranges; fixed uses --random-targets total; none adds no objects.",
    )
    parser.add_argument("--random-targets", type=int, default=None)
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--show-noise-ranges", action="store_true")
    parser.add_argument("--manual-object", "--add-object", action="append", default=[],
                        help="type,x,y[,rcs,length,width,heading,speed,course]; local map coordinates in meters")
    parser.add_argument("-s", "--scenes", type=int, default=1)
    parser.add_argument("-r", "--rotations", type=int, default=1)
    parser.add_argument("--rotation-period", type=float, default=2.5)
    parser.add_argument("--beam-samples", type=int, default=None)
    parser.add_argument("--beam-occlusion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--primary-occlusion", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--secondary-scale", type=float, default=None)
    parser.add_argument("--inland-scale", type=float, default=None)
    parser.add_argument("--inland-max-code", type=float, default=None)
    parser.add_argument("--occlusion-shoulder-bins", type=int, default=None)
    parser.add_argument("--training-data", action="store_true")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--waypoints", default=None)
    parser.add_argument("--wp-type", default="medium_boat")
    parser.add_argument("--wp-speed", type=float, default=8.0)
    return parser


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _parse_xy(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError("--origin-xy must be formatted as x,y")
    xy = float(parts[0]), float(parts[1])
    if not all(math.isfinite(value) for value in xy):
        raise ValueError("Radar origin coordinates must be finite")
    return xy


def _read_radarloc_metadata(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metadata", {}) or {}


def _latlon_to_local_xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    d2r = math.pi / 180.0
    x = (lon - ref_lon) * math.cos(ref_lat * d2r) * EARTH_RADIUS_M * d2r
    y = (lat - ref_lat) * EARTH_RADIUS_M * d2r
    return x, y


def _resolve_origin(args: argparse.Namespace, metadata: dict[str, Any]) -> tuple[float, float] | None:
    xy = _parse_xy(args.origin_xy)
    if xy is not None:
        return xy
    if args.radar_lat is None and args.radar_lon is None:
        return None
    if args.radar_lat is None or args.radar_lon is None:
        raise ValueError("Use both --radar-lat and --radar-lon, or neither.")
    if not (math.isfinite(args.radar_lat) and math.isfinite(args.radar_lon)
            and -90 <= args.radar_lat <= 90 and -180 <= args.radar_lon <= 180):
        raise ValueError("Radar latitude/longitude must be finite and within geographic bounds")
    ref_lat = metadata.get("center_lat")
    ref_lon = metadata.get("center_lon")
    if ref_lat is None or ref_lon is None:
        raise ValueError("Cannot use --radar-lat/--radar-lon because .radarloc metadata lacks center_lat/center_lon.")
    return _latlon_to_local_xy(args.radar_lat, args.radar_lon, float(ref_lat), float(ref_lon))


def _session_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.name.startswith("session_") and (not output_dir.exists() or not any(output_dir.iterdir())):
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session = output_dir / f"session_{stamp}"
    session.mkdir(parents=True, exist_ok=True)
    return session


def _object_counts(profile: str, random_targets: int | None, rng: random.Random) -> dict[str, int]:
    if profile == "none":
        return {k: 0 for k in MODE1_OBJECT_RANGES}
    if profile == "fixed":
        total = max(0, int(random_targets or 0))
        weights = ["buoy", "small_boat", "medium_boat", "large_boat", "barge"]
        counts = {k: 0 for k in MODE1_OBJECT_RANGES}
        for _ in range(total):
            counts[rng.choice(weights)] += 1
        return counts
    counts = {
        obj_type: rng.randint(lo, hi)
        for obj_type, (lo, hi) in MODE1_OBJECT_RANGES.items()
    }
    if random_targets is not None:
        # Preserve Mode 1 type weighting while honoring a requested total.
        expanded = []
        for obj_type, count in counts.items():
            expanded.extend([obj_type] * count)
        while len(expanded) < random_targets:
            expanded.append(rng.choice(list(MODE1_OBJECT_RANGES)))
        rng.shuffle(expanded)
        selected = expanded[: max(0, int(random_targets))]
        counts = {k: selected.count(k) for k in MODE1_OBJECT_RANGES}
    return counts


def _random_position_on_water(sim: Simulation, rng: random.Random, min_range_m: float, max_range_m: float):
    own = sim.world.own_ship
    existing = sim.world.get_targets()
    for _ in range(400):
        bearing_deg = rng.uniform(0.0, 360.0)
        range_m = rng.uniform(min_range_m, max_range_m)
        br = math.radians(bearing_deg)
        x = own.x + range_m * math.sin(br)
        y = own.y + range_m * math.cos(br)
        if not sim._point_is_water(x, y):
            continue
        if any((x - v.x) ** 2 + (y - v.y) ** 2 < 28.0 ** 2 for v in existing):
            continue
        return x, y
    return None


def _uniform_pair(rng: random.Random, pair: tuple[float, float]) -> float:
    return rng.uniform(float(pair[0]), float(pair[1]))


def _place_mode1_style_objects(sim: Simulation, profile: str, random_targets: int | None, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    counts = _object_counts(profile, random_targets, rng)
    max_range_m = sim.radar.params.max_range_m * 0.82
    min_range_m = 40.0
    manifest: list[dict[str, Any]] = []

    for obj_type, count in counts.items():
        spec = OBJECT_SPECS[obj_type]
        for idx in range(count):
            pos = _random_position_on_water(sim, rng, min_range_m, max_range_m)
            if pos is None:
                continue
            x, y = pos
            speed_range = spec["speed"]
            speed = _uniform_pair(rng, speed_range) if speed_range[1] > 0 else 0.0
            vessel = Vessel(
                id=f"coastline_defined_{obj_type}_{idx + 1}",
                name=f"{spec['name']} {idx + 1}",
                vessel_type=spec["vessel_type"],
                x=x,
                y=y,
                course=rng.uniform(0.0, 360.0) if speed > 0 else 0.0,
                speed=speed,
                length=_uniform_pair(rng, spec["length"]),
                beam=_uniform_pair(rng, spec["beam"]),
                height=_uniform_pair(rng, spec["height"]),
                rcs=_uniform_pair(rng, spec["rcs"]),
                object_creation_class=obj_type,
            )
            sim.world.add_vessel(vessel)
            manifest.append({
                "id": vessel.id,
                "object_creation_type": obj_type,
                "runtime_vessel_type": vessel.vessel_type.value,
                "x_m": round(vessel.x, 3),
                "y_m": round(vessel.y, 3),
                "range_m": round(math.hypot(vessel.x - sim.world.own_ship.x, vessel.y - sim.world.own_ship.y), 3),
                "bearing_deg": round(math.degrees(math.atan2(vessel.x - sim.world.own_ship.x, vessel.y - sim.world.own_ship.y)) % 360.0, 3),
                "rcs_m2": round(float(vessel.rcs or 0.0), 3),
                "length_m": round(vessel.length, 3),
                "beam_m": round(vessel.beam, 3),
                "height_m": round(vessel.height, 3),
                "speed_kts": round(vessel.speed, 3),
                "course_deg": round(vessel.course, 3),
            })
    return manifest


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _radarloc_summary(path: Path) -> dict:
    if rust_bridge.rust_available():
        return rust_bridge.radarloc_summary(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    coastlines = doc.get("coastlines", [])
    terrain = doc.get("terrain") or {}
    grid = terrain.get("elevations", terrain.get("grid", []))
    return {"backend": "python", "file_bytes": path.stat().st_size,
        "coastline_count": len(coastlines),
        "total_points": sum(len(c.get("points", [])) for c in coastlines),
        "max_points_per_coastline": max((len(c.get("points", [])) for c in coastlines), default=0),
        "vessel_count": len(doc.get("vessels", [])),
        "terrain_rows": terrain.get("rows", len(grid)),
        "terrain_cols": terrain.get("cols", len(grid[0]) if grid else 0),
        "range_nm": doc.get("metadata", {}).get("range_nm", 6)}


def _print_noise_ranges() -> None:
    print("CoastlineDefined selectable noise/radar ranges:")
    for key, desc in NOISE_RANGES.items():
        print(f"  {key}: {desc}")


def _stage_started(name: str) -> float:
    started = time.perf_counter()
    print(f"STAGE|{name}|started", flush=True)
    return started


def _stage_finished(name: str, started: float) -> None:
    print(f"STAGE|{name}|complete|{time.perf_counter() - started:.3f}", flush=True)


def _target_manifest(sim: Simulation) -> list[dict]:
    from radar_core.object_echo import object_class
    own = sim.world.own_ship
    result = []
    for target in sim.world.get_targets():
        dx, dy = target.x - own.x, target.y - own.y
        result.append({"id": target.id, "object_creation_type": object_class(target),
            "type": object_class(target), "width_m": target.beam,
            "size_m": f"{target.length:.1f}x{target.beam:.1f}",
            "runtime_vessel_type": target.vessel_type.value, "name": target.name,
            "x_m": target.x, "y_m": target.y, "range_m": math.hypot(dx, dy),
            "bearing_deg": math.degrees(math.atan2(dx, dy)) % 360,
            "rcs_m2": target.rcs, "length_m": target.length, "beam_m": target.beam,
            "height_m": target.height, "speed_kts": target.speed,
            "course_deg": target.course,
            "heading_deg": target.course if target.heading_deg is None else target.heading_deg})
    return result


def _apply_manual_objects(sim: Simulation, specs: list[str]) -> None:
    next_id = 1
    for index, text in enumerate(specs):
        parts = [part.strip() for part in text.split(",")]
        if not 3 <= len(parts) <= 9:
            raise ValueError(f"Invalid manual object {text!r}; expected type,x,y[,rcs,length,width,heading,speed,course]")
        kind = parts[0].lower()
        preset = OBJECT_SPECS.get(kind)
        try:
            vessel_type = preset["vessel_type"] if preset else VesselType(kind)
            values = [float(value) for value in parts[1:]]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Invalid manual object: {text}") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Object values must be finite")
        x, y = values[:2]
        defaults = [80.0, 12.0, 4.0, 0.0, 0.0, None]
        for i, value in enumerate(values[2:]):
            defaults[i] = value
        rcs, length, width, heading, speed, course = defaults
        if rcs < 0 or min(length, width) <= 0 or speed < 0:
            raise ValueError("RCS/speed must be nonnegative; object dimensions must be positive")
        while sim.world.get_vessel(f"manual_{next_id}") is not None:
            next_id += 1
        target = Vessel(id=f"manual_{next_id}", name=f"Manual {kind} {index + 1}",
            vessel_type=vessel_type, x=x, y=y, rcs=rcs, length=length, beam=width,
            height=sum(preset["height"]) / 2 if preset else 8.0,
            heading_deg=heading % 360, course=(heading if course is None else course) % 360,
            speed=speed, object_creation_class=kind if preset else None)
        sim.world.add_vessel(target)


def build_scene(args: argparse.Namespace):
    """Load once through the same manager, terrain builder and renderer as RustCore."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    env_overrides = {
        "RADARLOC_NUM_RAYS": args.beam_samples,
        "RADARLOC_SWEEP_OCCLUDE_AFTER_FIRST_LAND": args.beam_occlusion,
        "RADARLOC_OCCLUDE_AFTER_FIRST_LAND": args.primary_occlusion,
        "RADARLOC_STRICT_SECONDARY_OCCLUSION_SCALE": args.secondary_scale,
        "RADARLOC_MAPPED_INLAND_RETURN_SCALE": args.inland_scale,
        "RADARLOC_MAPPED_INLAND_RETURN_MAX_CODE": args.inland_max_code,
        "RADARLOC_SWEEP_OCCLUSION_SHOULDER_BINS": args.occlusion_shoulder_bins,
    }
    for key, value in env_overrides.items():
        if value is not None:
            os.environ[key] = str(int(value) if isinstance(value, bool) else value)
    sim = Simulation()
    sim.set_loaded_coastline_attenuation_model(args.attenuation_model)
    metadata = _read_radarloc_metadata(Path(args.radarloc))
    origin = _resolve_origin(args, metadata)
    if origin is not None:
        sim.set_temporary_own_origin(*origin)
    if not ScenarioManager().load_location_file(str(args.radarloc), sim.world, sim.radar,
                                               sim.weather, simulation=sim):
        raise ValueError(f"Failed to load .radarloc: {args.radarloc}")
    if args.range_nm is not None:
        sim.radar.set_range_scale(args.range_nm)
    if args.beam_width_deg is not None:
        sim.radar.params.horizontal_beamwidth_deg = args.beam_width_deg
        sim.radar.detection_engine._update_radar_constant()
    sim.radar.set_gain(args.gain)
    sim.radar.set_sea_clutter(args.sea_clutter)
    sim.radar.set_rain_clutter(args.rain_clutter)
    from radar_sim.environment.weather import WeatherConditions
    sim.weather.set_conditions(WeatherConditions(sea_state=args.sea_state,
        wind_speed_knots=0.0, wind_direction=0.0, rain_rate_mmh=args.rain_rate_mmh))
    sim.weather.set_noise_level(args.receiver_noise_level)
    sim.weather.set_receiver_noise_enabled(args.receiver_noise)
    _apply_manual_objects(sim, args.manual_object)
    _place_mode1_style_objects(sim, args.object_profile, args.random_targets, args.seed)
    return sim, metadata


def _validate_args(args):
    for key, lo, hi in (("gain", 0, 1), ("sea_clutter", 0, 1), ("rain_clutter", 0, 1),
        ("rain_rate_mmh", 0, 50), ("receiver_noise_level", 0, .02),
        ("spokes", 1, 8192), ("scenes", 1, 10000), ("rotations", 1, 100000),
        ("rotation_period", .001, 86400), ("range_nm", .0625, 48),
        ("beam_width_deg", .1, 15), ("beam_samples", 1, 101),
        ("secondary_scale", 0, 1), ("inland_scale", 0, 1), ("inland_max_code", 0, 252),
        ("occlusion_shoulder_bins", 0, 868), ("raytrace_spokes", 1, 8192),
        ("image_size", 64, 4096), ("val_split", 0, .99), ("wp_speed", 0, 100)):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or not lo <= value <= hi):
            raise ValueError(f"{key} must be between {lo} and {hi}")
    if args.sea_state not in (0, 2, 4, 6, 8):
        raise ValueError("sea_state must be 0, 2, 4, 6 or 8")
    if args.random_targets is not None and not 0 <= args.random_targets <= 10000:
        raise ValueError("random_targets must be between 0 and 10000")
    if args.output_csv and (args.scenes != 1 or args.rotations != 1):
        raise ValueError("--output-csv is for one rotation only; use --output-dir for a sequence")


def _training_frame(args, sim, csv_path, scene_index, session_dir):
    from PIL import Image
    import radar_simulator as legacy
    from radar_core.object_echo import object_class, radial_spread_bins
    subset = "val" if args.scenes > 1 and scene_index >= max(1, round(args.scenes * (1 - args.val_split))) else "train"
    folder = session_dir / "dataset"
    for name in ("images", "labels"):
        (folder / name / subset).mkdir(parents=True, exist_ok=True)
    Image.fromarray(legacy.render_ppi_image(str(csv_path), args.image_size)).save(folder / "images" / subset / (csv_path.stem + ".png"))
    half, radius = args.image_size / 2, args.image_size / 2 - 2
    labels = []
    own = sim.world.own_ship
    for obj in sim.world.get_targets():
        x, y = obj.x - own.x, obj.y - own.y
        distance = math.hypot(x, y)
        if not 1 <= distance <= sim.radar.params.max_range_m:
            continue
        bearing = math.atan2(x, y)
        scale = radius / sim.radar.params.max_range_m
        radial = radial_spread_bins(obj, bearing, sim.radar.params.max_range_m / sim.radar.num_range_bins) * 2 * sim.radar.params.max_range_m / sim.radar.num_range_bins
        tangent = math.radians(sim.radar.params.horizontal_beamwidth_deg) * distance / 2
        hx = (abs(math.sin(bearing)) * radial + abs(math.cos(bearing)) * tangent) * scale + 3
        hy = (abs(math.cos(bearing)) * radial + abs(math.sin(bearing)) * tangent) * scale + 3
        cx, cy = half + x * scale, half - y * scale
        x0, x1 = max(0, cx - hx), min(args.image_size, cx + hx)
        y0, y1 = max(0, cy - hy), min(args.image_size, cy + hy)
        if x1 > x0 and y1 > y0:
            labels.append(f"{list(OBJECT_SPECS).index(object_class(obj))} {(x0+x1)/2/args.image_size:.6f} {(y0+y1)/2/args.image_size:.6f} {(x1-x0)/args.image_size:.6f} {(y1-y0)/args.image_size:.6f}")
    (folder / "labels" / subset / (csv_path.stem + ".txt")).write_text("\n".join(labels), encoding="utf-8")
    for name in ("images", "labels"):
        (folder / name / "val").mkdir(parents=True, exist_ok=True)
    (folder / "dataset.yaml").write_text(f"path: {json.dumps(str(folder.resolve()))}\ntrain: images/train\nval: images/val\nnames: {json.dumps(list(OBJECT_SPECS))}\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[str]:
    _validate_args(args)
    if not args.radarloc:
        if not args.location:
            raise ValueError("Choose a .radarloc file or a location")
        import radar_simulator as legacy
        try:
            from .scene_pipeline import resolve_source
        except ImportError:
            from scene_pipeline import resolve_source
        args.radarloc = str(resolve_source(legacy, args.output_dir, args.location,
                                         args.range_nm, prefer_source=args.prefer_source))
    source = Path(args.radarloc).resolve()
    if not source.is_file():
        raise ValueError(f"Missing .radarloc: {source}")
    if args.output_csv and Path(args.output_csv).exists():
        raise FileExistsError(f"Refusing to overwrite existing CSV: {args.output_csv}")
    session = _session_dir(Path(args.output_dir))
    copied = session / "source_location.radarloc"
    if copied.exists() and copied.read_bytes() != source.read_bytes():
        raise FileExistsError(f"Session already contains a different scene: {copied}")
    if copied != source:
        shutil.copy2(source, copied)
    outputs, frames = [], []
    start_time = datetime.now().astimezone()
    from datetime import timedelta
    for scene_index in range(args.scenes):
        scene_args = argparse.Namespace(**vars(args))
        scene_args.seed = args.seed + scene_index
        stage = _stage_started("Input load")
        sim, metadata = build_scene(scene_args)
        _stage_finished("Input load", stage)
        sim.exporter.output_dir = str(session)
        route = []
        if args.waypoints:
            route = [tuple(float(v.strip()) for v in point.split(",")) for point in args.waypoints.split(";")]
            if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in route):
                raise ValueError("Waypoints must be x,y;x,y in local map meters")
            _apply_manual_objects(sim, [f"{args.wp_type},{route[0][0]},{route[0][1]},80,12,4,0,{args.wp_speed},0"])
            route_target = sim.world.get_targets()[-1]
            route_index = 1
            if len(route) > 1:
                route_target.course = math.degrees(math.atan2(route[1][0] - route[0][0], route[1][1] - route[0][1])) % 360
                route_target.heading_deg = route_target.course
        for rotation in range(args.rotations):
            timestamp = start_time + timedelta(seconds=rotation * args.rotation_period)
            stem = f"scene_{scene_index+1:03d}_rotation_{rotation+1:04d}_{start_time.strftime('%Y%m%d_%H%M%S_%f')}"
            path = Path(args.output_csv).resolve() if args.output_csv else session / (stem + ".csv")
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite: {path}")
            if path.with_suffix(".scene.json").exists():
                raise FileExistsError(f"Refusing to overwrite scene document: {path.with_suffix('.scene.json')}")
            path.parent.mkdir(parents=True, exist_ok=True)
            objects = _target_manifest(sim)
            frame = {"sequence_id": f"scene_{scene_index+1:03d}", "timestep_idx": rotation,
                "timestamp": timestamp.isoformat(), "rotation_period_sec": args.rotation_period,
                "source_radarloc": str(copied), "radar_config": {"range_nm": sim.radar.params.current_range_nm,
                "range_m": sim.radar.params.max_range_m, "gain": sim.radar.params.gain,
                "origin_xy_m": [sim.world.own_ship.x, sim.world.own_ship.y], "bins": sim.radar.num_range_bins,
                "bin_size_m": sim.radar.params.max_range_m / sim.radar.num_range_bins,
                "spokes": args.spokes, "range_index": sim.exporter.furuno_range_index_for_nm(sim.radar.params.current_range_nm),
                "beamwidth_deg": sim.radar.params.horizontal_beamwidth_deg,
                "rotation_period_sec": args.rotation_period, "ticks_per_rev": 8192, "status": 1, "scale": 496},
                "objects": objects}
            result = sim.export_furuno_rotation(str(path), spokes_per_rotation=args.spokes,
                raytrace_spokes=args.spokes if args.strict_tick_raytrace else args.raytrace_spokes)
            if not result:
                raise RuntimeError("Furuno rotation export failed")
            sidecar = session / "scene_metadata" / path.stem
            sidecar.mkdir(parents=True, exist_ok=True)
            _write_json(sidecar / (path.stem + "_metadata.json"), {"scene": frame, "radar_config": frame["radar_config"]})
            _write_json(path.with_suffix(".scene.json"), frame)
            outputs.append(str(path))
            frames.append(frame)
            if args.training_data:
                _training_frame(args, sim, path, scene_index, session)
            for target in sim.world.get_targets():
                if route and target is route_target:
                    remaining = target.speed * .514444 * args.rotation_period
                    while remaining > 0 and route_index < len(route):
                        dx, dy = route[route_index][0] - target.x, route[route_index][1] - target.y
                        distance = math.hypot(dx, dy)
                        target.course = math.degrees(math.atan2(dx, dy)) % 360
                        target.heading_deg = target.course
                        if distance <= remaining:
                            target.x, target.y = route[route_index]
                            route_index += 1
                            remaining -= distance
                        else:
                            target.x += dx * remaining / distance
                            target.y += dy * remaining / distance
                            remaining = 0
                    if route_index >= len(route):
                        target.speed = 0
                else:
                    target.update(args.rotation_period)
            sim.world.time += args.rotation_period
        run_config = {**vars(scene_args), "mode": "Mode C / CoastlineDefined", "runtime_dir": str(RUNTIME_DIR),
            "scene_pipeline": "RustCore ScenarioManager.load_location_file", "source_radarloc_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "origin_xy_m": [sim.world.own_ship.x, sim.world.own_ship.y],
            "range_nm": sim.radar.params.current_range_nm, "beam_width_deg": sim.radar.params.horizontal_beamwidth_deg,
            "raytrace_spokes": args.spokes if args.strict_tick_raytrace else args.raytrace_spokes or sim.radar.num_bearings,
            "rust_available": rust_bridge.rust_available(), "object_count": len(objects),
            "environment": apply_launch_profile(), "label_policy": "Geometric echo-footprint boxes; visibility/occlusion not certified" if args.training_data else None}
        run_config["receiver_noise_effective_level"] = sim.weather.get_effective_receiver_noise_level()
        run_config["radarloc_metadata"] = metadata
        _write_json(session / f"scene_{scene_index+1:03d}_config.json", run_config)
    _write_json(session / "coastline_defined_config.json", run_config)
    _write_json(session / "object_manifest.json", frames[0]["objects"])
    _write_json(session / "sequence_manifest.json", {"settings": vars(args), "frames": frames, "csv_files": outputs})
    _write_json(session / "radarloc_summary.json", _radarloc_summary(source))
    print(f"CoastlineDefined export complete: {len(outputs)} rotation(s)\nSession folder: {session}", flush=True)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.show_noise_ranges:
        _print_noise_ranges()
    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
