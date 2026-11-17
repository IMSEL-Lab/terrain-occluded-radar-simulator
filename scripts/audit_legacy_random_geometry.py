"""Audit original random geometry generators without rendering radar output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"
os.environ["MODE4_USE_OBJECT_LEARNED_MODEL"] = "0"

import numpy as np
import shapely
from shapely.geometry import Point, Polygon

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import radar_simulator as legacy

SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "radar_simulator.py": _sha256(REPO / "radar_simulator.py"),
        "object_renderer.py": _sha256(REPO / "object_renderer.py"),
        "runner.py": _sha256(Path(__file__).resolve()),
    }


def _synthetic_fixture_polygon() -> list[tuple[float, float]]:
    """Define a closed radial ellipse in local XY, independent of capture data."""
    points = []
    for index in range(32):
        angle = 2 * np.pi * index / 32
        bearing_fraction = 0.5 + 0.46 * float(np.cos(angle))
        range_fraction = 0.5 + 0.38 * float(np.sin(angle))
        bearing_deg = 180.0 + (bearing_fraction - 0.5) * 24.0
        radius_m = 200.0 + range_fraction * 150.0
        bearing_rad = np.deg2rad(bearing_deg)
        points.append(
            (
                float(radius_m * np.sin(bearing_rad)),
                float(radius_m * np.cos(bearing_rad)),
            )
        )
    return points


def _synthetic_library() -> Any:
    """Build the API's normalized primitive from one frozen synthetic polygon."""
    library = legacy.CoastlineLibrary()
    points = _synthetic_fixture_polygon()
    primitive = legacy.build_primitive_from_coastline(
        points,
        {},
        "synthetic_audit_fixture",
        "synthetic-audit-primitive",
    )
    if primitive is None:
        raise ValueError("synthetic fixture did not produce a CoastlinePrimitive")
    library.add(primitive)
    return library


def _scene(range_m: float, load: int, seed: int) -> tuple[list[Any], list[Any]]:
    random.seed(seed)
    np.random.seed(seed)
    coastlines = legacy.generate_random_coastlines(
        range_m, library=_synthetic_library()
    )
    config = legacy.SceneConfig(
        num_buoys=(0, 0),
        num_small_boats=(0, 0),
        num_medium_boats=(0, 0),
        num_large_boats=(load, load),
        num_barges=(0, 0),
    )
    objects = legacy.generate_random_objects(config, range_m, coastlines)
    return coastlines, objects


def _footprint(obj: Any) -> Polygon:
    heading = np.deg2rad(float(obj.heading_deg))
    along = np.array([np.sin(heading), np.cos(heading)])
    across = np.array([np.cos(heading), -np.sin(heading)])
    center = np.array([float(obj.x), float(obj.y)])
    corners = []
    for long_sign, side_sign in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        corner = (
            center
            + long_sign * float(obj.length_m) / 2 * along
            + side_sign * float(obj.width_m) / 2 * across
        )
        corners.append((float(corner[0]), float(corner[1])))
    return Polygon(corners)


def _geometry_metrics(
    coastlines: list[Any],
    objects: list[Any],
    range_m: float,
    requested_object_count: int | None = None,
    center_range_fraction: float = 0.95,
) -> dict[str, Any]:
    closed_coastlines = [
        coast for coast in coastlines if getattr(coast, "closed", True)
    ]
    open_coastlines = [
        coast for coast in coastlines if not getattr(coast, "closed", True)
    ]
    coast_polygons = [Polygon(coast.points) for coast in closed_coastlines]
    footprints = [_footprint(obj) for obj in objects]
    invalid = sum(
        not polygon.is_valid or polygon.is_empty or polygon.area <= 0
        for polygon in coast_polygons
    )
    valid_coast_polygons = [
        polygon
        for polygon in coast_polygons
        if polygon.is_valid and not polygon.is_empty and polygon.area > 0
    ]
    coastline_out_of_range = sum(
        np.hypot(float(x), float(y)) > range_m
        for coast in coastlines
        for x, y in coast.points
    )
    overlap_coasts = sum(
        first.intersection(second).area > 0.0
        for i, first in enumerate(valid_coast_polygons)
        for second in valid_coast_polygons[i + 1 :]
    )
    center_in_valid_land = sum(
        any(
            polygon.contains(Point(float(obj.x), float(obj.y)))
            for polygon in valid_coast_polygons
        )
        for obj in objects
    )
    center_in_land = sum(
        legacy.point_in_any_coastline(float(obj.x), float(obj.y), coastlines)
        for obj in objects
    )
    footprint_land = sum(
        any(
            footprint.intersection(polygon).area > 0.0
            for polygon in valid_coast_polygons
        )
        for footprint in footprints
    )
    overlap_objects = sum(
        first.intersection(second).area > 0.0
        for i, first in enumerate(footprints)
        for second in footprints[i + 1 :]
    )
    center_out_of_range = sum(
        np.hypot(float(obj.x), float(obj.y)) > range_m for obj in objects
    )
    center_outside_generator_radius = sum(
        np.hypot(float(obj.x), float(obj.y)) > range_m * center_range_fraction
        for obj in objects
    )
    footprint_out_of_range = sum(
        max(np.hypot(x, y) for x, y in footprint.exterior.coords) > range_m
        for footprint in footprints
    )
    return {
        "coastline_count": len(coastlines),
        "coastline_closed_polygon_count": len(closed_coastlines),
        "coastline_open_segment_count": len(open_coastlines),
        "coastline_vertex_count": sum(len(coast.points) for coast in coastlines),
        "invalid_closed_polygon_count": int(invalid),
        "invalid_closed_polygons_excluded_from_intersection_counts": int(invalid),
        "coastline_vertex_outside_nominal_range_count": int(coastline_out_of_range),
        "overlapping_valid_coastline_polygon_pairs": int(overlap_coasts),
        "object_count_requested": len(objects)
        if requested_object_count is None
        else requested_object_count,
        "object_count_returned": len(objects),
        "object_count_return_shortfall": max(
            0,
            (len(objects) if requested_object_count is None else requested_object_count)
            - len(objects),
        ),
        "object_center_in_land_legacy_predicate_count": int(center_in_land),
        "object_center_in_valid_land_count": int(center_in_valid_land),
        "object_footprint_valid_land_intersection_count": int(footprint_land),
        "object_footprint_overlap_pairs": int(overlap_objects),
        "object_center_out_of_range_count": int(center_out_of_range),
        "object_center_outside_generator_radius_count": int(
            center_outside_generator_radius
        ),
        "object_footprint_out_of_range_count": int(footprint_out_of_range),
    }


def _geometry_record(coastlines: list[Any], objects: list[Any]) -> dict[str, Any]:
    return {
        "coastlines": [
            {
                "name": str(coast.name),
                "points_m": [[float(x), float(y)] for x, y in coast.points],
            }
            for coast in coastlines
        ],
        "objects": [
            {
                "x_m": float(obj.x),
                "y_m": float(obj.y),
                "obj_type": str(obj.obj_type),
                "rcs_m2": float(obj.rcs_m2),
                "length_m": float(obj.length_m),
                "width_m": float(obj.width_m),
                "heading_deg": float(obj.heading_deg),
            }
            for obj in objects
        ],
    }


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        path.write_text("\n", encoding="utf-8")
    else:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path.stat().st_size


def _create_output_dir(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)


def run(config_path: Path, output: Path) -> None:
    _create_output_dir(output)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_status": "running",
        "descriptor": "Original random geometry functions; no full rendering or capture-derived library",
        "config_file": str(config_path.resolve()),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(output / "manifest.json", manifest)
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        manifest["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        manifest["source_sha256"] = _source_hashes()
        manifest["revision"] = _git_revision()
        manifest["synthetic_input"] = {
            "scope": config["generator"]["library_scope"],
            "primitive_specification": config["generator"][
                "synthetic_library_primitive"
            ],
            "native_classification": _synthetic_library().primitives[0].classification,
        }
        manifest["environment"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "shapely": shapely.__version__,
            "cpu_count": os.cpu_count(),
            "thread_limits": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "learned_model_enabled": legacy.MODE4_USE_OBJECT_LEARNED_MODEL,
        }
        _write_json(output / "manifest.json", manifest)
        rows = []
        expected_runs = 0
        geometry_path = output / "generated_geometry.jsonl"
        with geometry_path.open("w", encoding="utf-8") as geometry_stream:
            for range_m in config["ranges_m"]:
                for load in config["object_loads"]:
                    for seed in config["replicate_seeds"]:
                        coastlines, objects = _scene(
                            float(range_m), int(load), int(seed)
                        )
                        record = _geometry_record(coastlines, objects)
                        geometry_hash = _canonical_sha(record)
                        repeat_coasts, repeat_objects = _scene(
                            float(range_m), int(load), int(seed)
                        )
                        deterministic = geometry_hash == _canonical_sha(
                            _geometry_record(repeat_coasts, repeat_objects)
                        )
                        metrics = _geometry_metrics(
                            coastlines,
                            objects,
                            float(range_m),
                            requested_object_count=int(load),
                            center_range_fraction=float(
                                config["bounds"]["object_center_range_fraction"]
                            ),
                        )
                        if not deterministic:
                            raise AssertionError(
                                f"geometry is not reproducible for range={range_m}, load={load}, seed={seed}"
                            )
                        metrics.update(
                            {
                                "range_m": float(range_m),
                                "object_load": int(load),
                                "seed": int(seed),
                                "geometry_sha256": geometry_hash,
                                "deterministic_reproduction": deterministic,
                            }
                        )
                        rows.append(metrics)
                        geometry_stream.write(
                            json.dumps(
                                {**metrics, **record},
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        expected_runs += 1
        expected = int(config["expected_scene_count"])
        if expected_runs != expected:
            raise AssertionError(
                f"frozen plan requested {expected} scenes but generated {expected_runs}"
            )
        _write_csv(output / "results.csv", rows)
        summary = {
            "scene_count": expected_runs,
            "deterministic_scenes": sum(
                row["deterministic_reproduction"] for row in rows
            ),
            "total_objects_requested": sum(
                row["object_count_requested"] for row in rows
            ),
            "total_objects_returned": sum(row["object_count_returned"] for row in rows),
            "total_object_return_shortfall": sum(
                row["object_count_return_shortfall"] for row in rows
            ),
            "total_object_pairs": sum(
                row["object_count_requested"] * (row["object_count_requested"] - 1) // 2
                for row in rows
            ),
            "total_coastline_polygons": sum(row["coastline_count"] for row in rows),
            "total_closed_coastline_polygons": sum(
                row["coastline_closed_polygon_count"] for row in rows
            ),
            "total_open_coastline_segments": sum(
                row["coastline_open_segment_count"] for row in rows
            ),
            "total_valid_closed_coastline_polygons": sum(
                row["coastline_closed_polygon_count"]
                - row["invalid_closed_polygon_count"]
                for row in rows
            ),
            "total_coastline_vertices": sum(
                row["coastline_vertex_count"] for row in rows
            ),
            "total_possible_coastline_pairs": sum(
                row["coastline_count"] * (row["coastline_count"] - 1) // 2
                for row in rows
            ),
            "total_invalid_coastline_polygons": sum(
                row["invalid_closed_polygon_count"] for row in rows
            ),
            "total_coastline_vertices_outside_nominal_range": sum(
                row["coastline_vertex_outside_nominal_range_count"] for row in rows
            ),
            "total_overlapping_valid_coastline_pairs": sum(
                row["overlapping_valid_coastline_polygon_pairs"] for row in rows
            ),
            "total_object_centers_in_land_legacy_predicate": sum(
                row["object_center_in_land_legacy_predicate_count"] for row in rows
            ),
            "total_object_centers_in_valid_land": sum(
                row["object_center_in_valid_land_count"] for row in rows
            ),
            "total_object_footprints_intersecting_valid_land": sum(
                row["object_footprint_valid_land_intersection_count"] for row in rows
            ),
            "total_object_footprint_overlap_pairs": sum(
                row["object_footprint_overlap_pairs"] for row in rows
            ),
            "total_object_centers_outside_range": sum(
                row["object_center_out_of_range_count"] for row in rows
            ),
            "total_object_centers_outside_generator_radius": sum(
                row["object_center_outside_generator_radius_count"] for row in rows
            ),
            "total_object_footprints_outside_range": sum(
                row["object_footprint_out_of_range_count"] for row in rows
            ),
            "synthetic_input_scope": "one fixed synthetic primitive reused across all scenes; results do not estimate the capture-derived library distribution",
            "coastline_range_note": "vertices beyond the nominal radar range are counted and may reflect post-clamp organic warping; this is not treated as automatic invalidity",
            "claim_scope": "generator geometry diagnostics only; range excursions and invalid closed polygons are reported counts, not automatic run failures",
        }
        _write_json(
            output / "data.json", {"schema_version": SCHEMA_VERSION, "summary": summary}
        )
        manifest["summary"] = summary
        manifest["output_files"] = {
            "results.csv_bytes": (output / "results.csv").stat().st_size,
            "generated_geometry.jsonl_bytes": geometry_path.stat().st_size,
            "data.json_bytes": (output / "data.json").stat().st_size,
        }
        manifest["run_status"] = "complete"
    except Exception as exc:
        manifest["run_status"] = "failed"
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(output / "manifest.json", manifest)


def _git_revision() -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/legacy_random_geometry_20261116.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
