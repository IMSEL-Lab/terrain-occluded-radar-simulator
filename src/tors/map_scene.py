"""Generate a radar sweep from a map boundary and explicit scan settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
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
    config = scene.get("radar_config")
    range_m = config.get("range_m") if isinstance(config, dict) else None
    if (
        isinstance(range_m, bool)
        or not isinstance(range_m, (int, float))
        or not math.isfinite(range_m)
        or range_m <= 0
    ):
        raise ValueError("radar_config.range_m must be positive and finite")
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


def _validate_geojson(data: Any, lat: float, lon: float, range_m: float) -> dict:
    """Validate the subset that the native loader can represent without loss."""
    if not isinstance(data, dict):
        raise ValueError("GeoJSON must be an object")  # noqa: TRY004
    kind = data.get("type")
    if kind == "FeatureCollection":
        features = data.get("features")
        if not isinstance(features, list):
            raise ValueError("GeoJSON FeatureCollection must contain a features array")
    elif kind == "Feature":
        features = [data]
        data = {"type": "FeatureCollection", "features": features}
    elif kind in ("Polygon", "MultiPolygon", "LineString", "MultiLineString"):
        features = [{"type": "Feature", "geometry": data, "properties": {}}]
    else:
        raise ValueError(f"unsupported GeoJSON type: {kind!r}")

    def coordinates(value: Any, *, ring: bool) -> None:
        if not isinstance(value, list) or len(value) < (4 if ring else 2):
            raise ValueError("invalid GeoJSON ring or line: too few coordinates")
        points = []
        for coord in value:
            if (
                not isinstance(coord, list)
                or len(coord) < 2
                or any(
                    isinstance(c, bool)
                    or not isinstance(c, (int, float))
                    or not math.isfinite(c)
                    for c in coord
                )
                or not -180 <= coord[0] <= 180
                or not -90 <= coord[1] <= 90
            ):
                raise ValueError("invalid GeoJSON coordinate")
            points.append((coord[0], coord[1]))
        if any(abs(a[0] - b[0]) > 180 for a, b in pairwise(points)):
            raise ValueError(
                "GeoJSON coordinates crossing the antimeridian are unsupported"
            )
        if ring:
            if (
                points[0] != points[-1]
                or not Polygon(points).is_valid
                or Polygon(points).area <= 0
            ):
                raise ValueError("invalid GeoJSON polygon ring")
        else:
            if any(a == b for a, b in pairwise(points)):
                raise ValueError("GeoJSON line has repeated adjacent coordinates")

    def polygon(value: Any) -> None:
        if not isinstance(value, list) or not value:
            raise ValueError("invalid GeoJSON polygon: missing outer ring")
        if len(value) != 1:
            raise ValueError("GeoJSON polygon holes are unsupported")
        coordinates(value[0], ring=True)

    for feature in features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("GeoJSON features must be Feature objects")
        properties = feature.get("properties")
        if properties is not None and not isinstance(properties, dict):
            raise ValueError("GeoJSON feature properties must be an object or null")
        geom = feature.get("geometry")
        if not isinstance(geom, dict):
            raise ValueError(  # noqa: TRY004
                "GeoJSON feature geometry must be a supported non-null geometry"
            )
        geom_type = geom.get("type")
        coords = geom.get("coordinates")
        if geom_type == "Polygon":
            polygon(coords)
        elif geom_type == "LineString":
            coordinates(coords, ring=False)
        elif geom_type in ("MultiPolygon", "MultiLineString"):
            if not isinstance(coords, list) or not coords:
                raise ValueError("GeoJSON multi-geometry must contain coordinates")
            for part in coords:
                if geom_type == "MultiPolygon":
                    polygon(part)
                else:
                    coordinates(part, ring=False)
        else:
            raise ValueError(f"unsupported GeoJSON geometry: {geom_type!r}")
    return data


def _clip_lines(data: dict, lat: float, lon: float, range_m: float) -> dict:
    """Clip each line to the native loader's circular 1.1-range envelope."""
    features: list[dict[str, Any]]
    if data.get("type") == "FeatureCollection":
        features = data["features"]
    else:
        features = [{"type": "Feature", "properties": {}, "geometry": data}]
    radius = range_m * 1.1
    # Intersections fall just inside the loader's hard cutoff despite roundoff.
    inner_radius = radius * (1 - 1e-10)

    def clip(points: list) -> list:
        local = [engine.latlon_to_local_xy(p[1], p[0], lat, lon) for p in points]
        if all(math.hypot(x, y) <= radius for x, y in local):
            return [points]
        pieces: list = []
        current: list = []
        for i, (a, b) in enumerate(pairwise(local)):
            dx, dy = b[0] - a[0], b[1] - a[1]
            aa = dx * dx + dy * dy
            if aa == 0:
                raise ValueError("GeoJSON line has coincident projected coordinates")
            bb = 2 * (a[0] * dx + a[1] * dy)
            cc = a[0] * a[0] + a[1] * a[1] - inner_radius**2
            disc = bb * bb - 4 * aa * cc
            if disc <= 0:
                if current:
                    pieces.append(current)
                    current = []
                continue
            root = math.sqrt(disc)
            t0 = max(0.0, (-bb - root) / (2 * aa))
            t1 = min(1.0, (-bb + root) / (2 * aa))
            if t0 >= t1:
                if current:
                    pieces.append(current)
                    current = []
                continue

            def interpolate(t: float, index: int = i) -> list:
                if t == 0:
                    return points[index]
                if t == 1:
                    return points[index + 1]
                return [
                    points[index][j] + t * (points[index + 1][j] - points[index][j])
                    for j in (0, 1)
                ]

            start, end = interpolate(t0), interpolate(t1)
            if current and t0 == 0 and current[-1] == start:
                current.append(end)
            else:
                if current:
                    pieces.append(current)
                current = [start, end]
            if t1 < 1:
                pieces.append(current)
                current = []
        if current:
            pieces.append(current)
        return pieces

    output = []
    for feature in features:
        geom = feature["geometry"]
        if geom["type"] in ("LineString", "MultiLineString"):
            lines = (
                [geom["coordinates"]]
                if geom["type"] == "LineString"
                else geom["coordinates"]
            )
            parts = [part for line in lines for part in clip(line)]
            if not parts:
                continue
            if parts != lines:
                if geom["type"] == "LineString" and len(parts) == 1:
                    geom = {"type": "LineString", "coordinates": parts[0]}
                else:
                    geom = {"type": "MultiLineString", "coordinates": parts}
                feature = {**feature, "geometry": geom}
        output.append(feature)
    return {"type": "FeatureCollection", "features": output}


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
    source = args.geojson.read_bytes()
    data = _validate_geojson(
        json.loads(source), args.lat, args.lon, scene["radar_config"]["range_m"]
    )
    data = _clip_lines(data, args.lat, args.lon, scene["radar_config"]["range_m"])
    # Feed the validated snapshot to the frozen loader. Single Features have
    # been normalized to FeatureCollections, which the loader understands.
    with TemporaryDirectory() as temp:
        snapshot = Path(temp) / "validated.geojson"
        snapshot.write_text(json.dumps(data), encoding="utf-8")
        coastlines = engine.load_coastlines_from_geojson(
            str(snapshot),
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
        "source_geojson_sha256": hashlib.sha256(source).hexdigest(),
        "scene": str(scene_path),
        "output": paths,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
