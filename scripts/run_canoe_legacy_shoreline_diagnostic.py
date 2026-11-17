"""Evaluate mapped shoreline rendering against the frozen CANOE traversals."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import platform
import resource
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddmr import canoe
from ddmr.canoe import (
    PoseTable,
    RadarScan,
    _c_enuref_sensor,
    interpolate_pose,
    load_pose_table,
    load_radar_png,
    pose_time_supported,
    project_scan_with_native_range,
)
from scripts import run_canoe_osm_diagnostic as osm
from scripts.canoe_manifest import verify_source_manifest
from scripts.git_provenance import checkout_revision

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
TICKS_PER_REV = 8192
ENCODER_COUNTS_PER_REV = 5600
MAX_RANGE_M = 150.0
RADIAL_STEP_M = 0.5
SHORELINE_STRIP_WIDTH_M = 3.0
EARTH_RADIUS_M = 6_371_000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _line_parts(geometry: BaseGeometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]  # type: ignore[list-item]
    if hasattr(geometry, "geoms"):
        return [line for part in geometry.geoms for line in _line_parts(part)]
    return []


def map_enu_to_legacy_matrix(center_euler_xyz_rad: FloatArray) -> FloatArray:
    """Map ENU deltas to legacy renderer [x-right, y-forward] coordinates.

    CANOE's official sensor coordinates are [x-forward, y-right]. The pinned
    renderer's polar frame is [x-right, y-forward], so apply the axis swap P
    after inverting the full planar ENU-from-sensor matrix.
    """
    angles = np.asarray(center_euler_xyz_rad, dtype=np.float64)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("scan-center Euler angles must be finite roll, pitch, yaw")
    enu_from_sensor_xy = _c_enuref_sensor(*angles)[:2, :2]
    determinant = float(np.linalg.det(enu_from_sensor_xy))
    if abs(determinant) <= 1e-9:
        raise ValueError("scan-center planar ENU rotation is singular")
    sensor_from_enu_xy = np.linalg.inv(enu_from_sensor_xy)
    renderer_from_sensor = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    return renderer_from_sensor @ sensor_from_enu_xy


def water_boundary_scene_frame(
    water_polygon: BaseGeometry,
    center_xy_enu: FloatArray,
    center_euler_xyz_rad: FloatArray,
    *,
    max_range_m: float = MAX_RANGE_M,
) -> dict[str, Any]:
    """Encode the clipped, rotated shoreline boundary as line features.

    Temporary coordinates use an equidistant local coordinate frame solely so
    the original GeoJSON loader applies its native three-metre line buffer.
    """
    if water_polygon.is_empty or not water_polygon.is_valid:
        raise ValueError("water polygon must be nonempty and valid")
    if center_xy_enu.shape != (2,) or not np.isfinite(center_xy_enu).all():
        raise ValueError("center pose must be a finite ENU XY pair")
    map_to_legacy = map_enu_to_legacy_matrix(center_euler_xyz_rad)

    def to_scan_frame(x: float, y: float, z: float | None = None) -> tuple[float, ...]:
        east, north = x - center_xy_enu[0], y - center_xy_enu[1]
        local_x = map_to_legacy[0, 0] * east + map_to_legacy[0, 1] * north
        local_y = map_to_legacy[1, 0] * east + map_to_legacy[1, 1] * north
        return (
            (float(local_x), float(local_y))
            if z is None
            else (float(local_x), float(local_y), z)
        )

    boundary_local = osm.shapely_transform(to_scan_frame, water_polygon.boundary)
    clipping_radius_m = max_range_m * 1.1
    clipped = boundary_local.intersection(Point(0.0, 0.0).buffer(clipping_radius_m))
    degrees_per_meter = 180.0 / (math.pi * EARTH_RADIUS_M)
    features = []
    for line in _line_parts(clipped):
        coordinates = [
            [float(x) * degrees_per_meter, float(y) * degrees_per_meter]
            for x, y, *_ in line.coords
        ]
        if len(coordinates) >= 2:
            features.append(
                {
                    "type": "Feature",
                    "properties": {"name": "independent_water_boundary"},
                    "geometry": {"type": "LineString", "coordinates": coordinates},
                }
            )
    if not features:
        raise ValueError("no water-boundary segments fall within the renderer crop")
    return {"type": "FeatureCollection", "features": features}


def load_native_shoreline_strips(
    simulator: Any,
    boundary_geojson: dict[str, Any],
    temporary_geojson_path: Path,
    *,
    max_range_m: float = MAX_RANGE_M,
) -> list[Any]:
    """Apply the legacy loader's native 3 m line-to-buffer construction."""
    temporary_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_geojson_path.write_text(
        json.dumps(boundary_geojson, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    coastlines = simulator.load_coastlines_from_geojson(
        str(temporary_geojson_path), 0.0, 0.0, max_range_m=max_range_m
    )
    if not coastlines:
        raise ValueError("legacy GeoJSON loader produced no shoreline strips")
    for segment in coastlines:
        buffered = Polygon(segment.points)
        if buffered.is_empty or not buffered.is_valid or buffered.area <= 0:
            raise ValueError("legacy native buffer produced an invalid shoreline strip")
    return coastlines


def _local_boundary_distance_m(boundary_geojson: dict[str, Any]) -> float:
    factor = math.pi * EARTH_RADIUS_M / 180.0
    origin = Point(0.0, 0.0)
    distances = []
    for feature in boundary_geojson["features"]:
        coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=np.float64)
        if len(coordinates) >= 2:
            distances.append(LineString(coordinates * factor).distance(origin))
    if not distances:
        raise ValueError("no boundary segments available for ownship-distance check")
    return float(min(distances))


def explicit_segment_strip_coastlines(
    simulator: Any,
    boundary_geojson: dict[str, Any],
    *,
    width_m: float = SHORELINE_STRIP_WIDTH_M,
) -> tuple[list[Any], dict[str, int | float]]:
    """Build one square-capped 3 m rectangle per external boundary segment."""
    if not np.isfinite(width_m) or width_m <= 0:
        raise ValueError("explicit shoreline-strip width must be positive and finite")
    coordinate_to_meter = math.pi * EARTH_RADIUS_M / 180.0
    half_width = width_m / 2.0
    coastline_segments = []
    polygons: list[Polygon] = []
    source_segment_count = 0
    duplicate_or_zero_segment_count = 0
    for feature_index, feature in enumerate(boundary_geojson.get("features", [])):
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "LineString":
            raise ValueError(
                "explicit segment strips require line-only boundary inputs"
            )
        xy = np.asarray(geometry.get("coordinates", []), dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError(
                "external boundary feature has invalid or nonfinite coordinates"
            )
        xy *= coordinate_to_meter
        for segment_index, (start, end) in enumerate(itertools.pairwise(xy)):
            source_segment_count += 1
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length <= 1e-9:
                duplicate_or_zero_segment_count += 1
                continue
            tangent = delta / length
            normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
            cap = tangent * half_width
            side = normal * half_width
            corners = np.asarray(
                [
                    start - cap - side,
                    end + cap - side,
                    end + cap + side,
                    start - cap + side,
                ],
                dtype=np.float64,
            )
            if not np.isfinite(corners).all():
                raise ValueError("explicit segment strip produced nonfinite vertices")
            polygon = Polygon(corners)
            if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
                raise ValueError(
                    f"explicit segment strip is invalid at feature {feature_index}, segment {segment_index}"
                )
            polygons.append(polygon)
            coastline_segments.append(
                simulator.CoastlineSegment(
                    points=[(float(x), float(y)) for x, y in corners],
                    name=f"shoreline_segment_{feature_index:04d}_{segment_index:06d}",
                    closed=True,
                )
            )
    if not polygons:
        raise ValueError("external boundary yielded no nonzero segment strips")

    overlap_pair_count = 0
    pairwise_overlap_area_sum = 0.0
    tree = STRtree(polygons)
    pairs = tree.query(polygons, predicate="intersects")
    if pairs.ndim == 2:
        for left, right in zip(pairs[0], pairs[1], strict=False):
            if left >= right:
                continue
            area = float(polygons[int(left)].intersection(polygons[int(right)]).area)
            if area > 1e-9:
                overlap_pair_count += 1
                pairwise_overlap_area_sum += area
    return coastline_segments, {
        "source_boundary_segment_count": source_segment_count,
        "zero_or_duplicate_segment_count": duplicate_or_zero_segment_count,
        "explicit_rectangle_count": len(polygons),
        "overlapping_rectangle_pair_count": overlap_pair_count,
        "pairwise_overlap_area_sum_m2": pairwise_overlap_area_sum,
    }


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if platform.system() == "Darwin" else peak * 1024


def renderer_ticks(encoder: NDArray[np.uint16]) -> list[int]:
    """Convert CANOE encoder rows to 8192-tick renderer angles in row order."""
    if encoder.ndim != 1 or np.any(encoder >= ENCODER_COUNTS_PER_REV):
        raise ValueError("encoder values must be a 1-D sequence in [0, 5600)")
    ticks = np.rint(encoder.astype(np.float64) * TICKS_PER_REV / ENCODER_COUNTS_PER_REV)
    return [int(value) % TICKS_PER_REV for value in ticks]


def build_explicit_mode1_scene(
    simulator: Any,
    ticks: list[int],
    *,
    seed: int = 42,
    bins: int = 300,
    bin_size_m: float = RADIAL_STEP_M,
    max_range_m: float = MAX_RANGE_M,
) -> dict[str, Any]:
    """Create a strict geometry-only Mode 1 scene with no data-derived inputs."""
    if bins != round(max_range_m / bin_size_m):
        raise ValueError("renderer bins must exactly cover the frozen radial range")
    if not ticks:
        raise ValueError("at least one explicit angle tick is required")
    forbidden = {
        "coastline_profiles",
        "coastline_stats",
        "object_echo_data",
        "noise_echo_data",
        "dead_zone_echo_data",
        "mode12_metadata",
        "_mode42",
        "source_file",
    }
    scene = {
        "_sim_mode": simulator.SIM_MODE_M1,
        "_sim_seed": int(seed),
        "radar_config": {
            "status": 1,
            "scale": 496,
            "range_index": 0,
            "gain": 75,
            "bins": int(bins),
            "bin_size_m": float(bin_size_m),
            "dead_zone_bins": 5,
            "range_m": float(max_range_m),
            "ticks_per_rev": TICKS_PER_REV,
        },
        "angle_ticks": [int(value) for value in ticks],
        "clutter": {"noise_probability": 0.0, "noise_codes": [], "noise_weights": []},
        "decay": {"ramp_up_range": [80, 200], "ramp_down_range": [40, 160]},
        "objects": [],
    }
    if forbidden.intersection(scene):
        raise AssertionError(
            "explicit renderer scene contains a replay or learned field"
        )
    if scene.get("_sim_mode") != simulator.SIM_MODE_M1 or scene["objects"]:
        raise ValueError("scene must use Mode 1 and contain no rendered objects")
    return scene


def scan_motion_summary(
    scan: RadarScan, poses: PoseTable
) -> tuple[FloatArray, float, dict[str, float | int]]:
    """Return center pose and supported intra-scan translation/yaw variation."""
    center_times = np.asarray([scan.center_time_us], dtype=np.uint64)
    if not bool(pose_time_supported(poses, center_times)[0]):
        raise ValueError("scan center timestamp is outside pose-table support")
    center_pos, center_angles = interpolate_pose(poses, center_times)
    row_pos, row_angles = interpolate_pose(poses, scan.row_time_us)
    supported = scan.valid_azimuth & pose_time_supported(poses, scan.row_time_us)
    if not supported.any():
        raise ValueError("scan has no valid pose-supported azimuth rows")
    xy_delta = row_pos[supported, :2] - center_pos[0, :2]
    displacement = np.linalg.norm(xy_delta, axis=1)
    yaw_delta = (row_angles[supported, 2] - center_angles[0, 2] + math.pi) % (
        2 * math.pi
    ) - math.pi
    yaw_abs_deg = np.abs(np.rad2deg(yaw_delta))
    return (
        center_pos[0, :2].copy(),
        float(center_angles[0, 2]),
        {
            "valid_pose_supported_spokes": int(supported.sum()),
            "unsupported_or_invalid_spokes": int(
                len(scan.row_time_us) - supported.sum()
            ),
            "median_intra_scan_displacement_m": float(np.median(displacement)),
            "maximum_intra_scan_displacement_m": float(np.max(displacement)),
            "median_absolute_intra_scan_yaw_change_deg": float(np.median(yaw_abs_deg)),
            "maximum_absolute_intra_scan_yaw_change_deg": float(np.max(yaw_abs_deg)),
        },
    )


def parse_renderer_csv(
    path: Path, expected_ticks: list[int], expected_bins: int
) -> NDArray[np.float64]:
    """Read renderer echoes and enforce exact row/bearing/bin correspondence."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != ["Status", "Scale", "Range", "Gain", "Angle", "EchoValues"]:
            raise ValueError("renderer output header differs from Furuno CSV contract")
        rows = list(reader)
    if len(rows) != len(expected_ticks):
        raise ValueError("renderer output spoke count differs from requested row count")
    actual_ticks = [int(row[4]) for row in rows]
    if actual_ticks != expected_ticks:
        raise ValueError("renderer output angle rows lost their CANOE correspondence")
    if any(len(row) != 5 + expected_bins for row in rows):
        raise ValueError("renderer output has missing or extra radial bins")
    echoes = np.asarray(
        [[float(value) for value in row[5:]] for row in rows], dtype=np.float64
    )
    if not np.isfinite(echoes).all() or np.any(echoes < 0) or np.any(echoes > 252):
        raise ValueError(
            "renderer output contains nonfinite or out-of-range echo codes"
        )
    return echoes


def preserve_reference_display_255(reference_255: FloatArray) -> FloatArray:
    return reference_255.copy()


def fit_scalar_gain(
    raw_prediction: FloatArray, reference: FloatArray, valid: BoolArray
) -> float:
    if raw_prediction.shape != reference.shape or reference.shape != valid.shape:
        raise ValueError("gain inputs and mask must have identical shapes")
    if (
        not np.isfinite(raw_prediction[valid]).all()
        or not np.isfinite(reference[valid]).all()
    ):
        raise ValueError("gain calibration contains nonfinite values on valid cells")
    x, y = raw_prediction[valid], reference[valid]
    denominator = float(np.dot(x, x))
    return 0.0 if denominator <= 0 else max(0.0, float(np.dot(x, y) / denominator))


def apply_scalar_gain(raw_prediction: FloatArray, gain: float) -> FloatArray:
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("fitted display gain must be finite and nonnegative")
    return np.clip(raw_prediction * gain, 0.0, 1.0)


def fit_range_profile(
    references: list[FloatArray], masks: list[BoolArray]
) -> FloatArray:
    if not references or len(references) != len(masks):
        raise ValueError("range profile requires paired train references and masks")
    bins = references[0].shape[1]
    sums, counts = np.zeros(bins), np.zeros(bins)
    for values, valid in zip(references, masks, strict=True):
        if values.shape != valid.shape or values.shape[1] != bins:
            raise ValueError("calibration references and masks must share radial bins")
        for j in range(bins):
            selected = valid[:, j]
            sums[j] += float(values[selected, j].sum())
            counts[j] += int(selected.sum())
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def score_prediction(
    reference: FloatArray,
    prediction: FloatArray,
    valid: BoolArray,
    occupancy_threshold: float,
) -> dict[str, float | int | None]:
    if reference.shape != prediction.shape or reference.shape != valid.shape:
        raise ValueError("score arrays and mask must have identical shapes")
    if not valid.any():
        raise ValueError("no common valid CANOE cells for scoring")
    if (
        not np.isfinite(reference[valid]).all()
        or not np.isfinite(prediction[valid]).all()
    ):
        raise ValueError("nonfinite echo on a valid observed cell")
    delta = prediction[valid] - reference[valid]
    ref_positive = reference[valid] > occupancy_threshold
    pred_positive = prediction[valid] > occupancy_threshold
    intersection = int(np.count_nonzero(ref_positive & pred_positive))
    union = int(np.count_nonzero(ref_positive | pred_positive))
    return {
        "valid_cells": int(valid.sum()),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "reference_positive_cells": int(ref_positive.sum()),
        "predicted_positive_cells": int(pred_positive.sum()),
        "occupancy_intersection_cells": intersection,
        "occupancy_union_cells": union,
        "occupancy_iou": float(intersection / union) if union else None,
    }


def _load_renderer(expected_hashes: dict[str, str]) -> tuple[Any, str, str]:
    import object_renderer
    import radar_simulator as simulator

    simulator_hash = sha256_file(Path(simulator.__file__))
    renderer_hash = sha256_file(Path(object_renderer.__file__))
    if simulator_hash != expected_hashes["radar_simulator.py"]:
        raise ValueError("The evaluation configuration does not match this TORS engine")
    if renderer_hash != expected_hashes["object_renderer.py"]:
        raise ValueError(
            "The evaluation configuration does not match this object renderer"
        )
    simulator.MODE4_USE_OBJECT_LEARNED_MODEL = False
    simulator._MODE4_OBJECT_LEARNED_MODEL_CACHE.update(
        path=None, mtime=None, artifact=None
    )
    return simulator, simulator_hash, renderer_hash


def _manifest_path(manifests: Path, config: dict[str, Any], key: str) -> Path:
    return manifests / Path(config["dataset"][key]).name


def _sequence_dir(data_root: Path, role: str, sequence: str) -> Path:
    return data_root / role / sequence


def _load_records(
    config: dict[str, Any], data_root: Path, manifest_dir: Path, roles: Iterable[str]
) -> dict[str, tuple[list[dict[str, Any]], Path, PoseTable, Path]]:
    selection_path = _manifest_path(manifest_dir, config, "selection_manifest")
    source_manifest_path = _manifest_path(manifest_dir, config, "source_manifest")
    osm.verify_hash(
        selection_path,
        config["dataset"]["selection_manifest_sha256"],
        "selection manifest",
    )
    verify_source_manifest(
        source_manifest_path,
        config["dataset"]["source_manifest_sha256"],
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records = {}
    for role in roles:
        sequence = config["dataset"]["sequences"][role]
        sequence_dir = _sequence_dir(data_root, role, sequence)
        pose_path = sequence_dir / "novatel" / "radar_poses.csv"
        radar_config_path = sequence_dir / "calib" / "radar_config.yaml"
        osm.verify_hash(
            pose_path,
            config["dataset"]["pose_input_sha256"][role],
            f"{role} pose",
        )
        osm.verify_hash(
            radar_config_path,
            config["dataset"]["radar_config_sha256"],
            f"{role} radar config",
        )
        pose = load_pose_table(pose_path)
        items = selection["sequences"][role]["scans"]
        records[role] = (items, sequence_dir, pose, pose_path)
    return records


def _alignment_gates(
    config: dict[str, Any], source_root: Path, records: dict[str, Any]
) -> dict[str, Any]:
    georef = config["georeferencing"]
    maps_dir = source_root / "source_inputs"
    train_map_path = maps_dir / Path(georef["route_map_inputs"]["train"]["path"]).name
    val_map_path = (
        maps_dir / Path(georef["route_map_inputs"]["validation"]["path"]).name
    )
    osm.verify_hash(
        train_map_path, georef["route_map_inputs"]["train"]["sha256"], "train route map"
    )
    osm.verify_hash(
        val_map_path,
        georef["route_map_inputs"]["validation"]["sha256"],
        "validation route map",
    )
    train_ll = osm.load_route_markers(train_map_path)
    val_ll = osm.load_route_markers(val_map_path)
    origin = tuple(georef["local_enu_origin_lat_lon_deg"])
    train_xy = osm.latlon_to_local_enu(train_ll, origin)
    val_xy = osm.latlon_to_local_enu(val_ll, origin)
    translation = np.asarray(georef["pose_to_map_translation_m"], dtype=np.float64)
    train_poses, val_poses = records["train"][2], records["validation"][2]
    recomputed = osm.fit_route_translation(train_poses.position_enu[:, :2], train_xy)
    if float(np.linalg.norm(recomputed - translation)) > 0.02:
        raise ValueError(
            "train-only alignment differs from frozen translation by >2 cm"
        )
    results = {
        "train": osm.alignment_summary(
            train_poses.position_enu[:, :2], train_xy, translation
        ),
        "validation": osm.alignment_summary(
            val_poses.position_enu[:, :2], val_xy, translation
        ),
    }
    for role in ("train", "validation"):
        limits = georef[f"{role}_gate"]
        if (
            results[role]["median_nearest_route_residual_m"] > limits["median_max_m"]
            or results[role]["p90_nearest_route_residual_m"] > limits["p90_max_m"]
        ):
            raise ValueError(
                f"{role} route alignment gate failed; no refit is permitted"
            )
    return {
        "train": results["train"],
        "validation": results["validation"],
        "train_fit_m": recomputed.tolist(),
    }


def _render_one(
    simulator: Any,
    scan: RadarScan,
    poses: PoseTable,
    translation: FloatArray,
    water_polygon: BaseGeometry,
    output_root: Path,
    role: str,
    record: dict[str, Any],
    expected_bins: int,
    seed: int,
    geometry_method: str = "native_line_buffer",
) -> tuple[FloatArray, dict[str, Any]]:
    render_started = time.perf_counter()
    peak_rss_before = _peak_rss_bytes()
    center, _yaw, motion = scan_motion_summary(scan, poses)
    center_times = np.asarray([scan.center_time_us], dtype=np.uint64)
    _, center_angles = interpolate_pose(poses, center_times)
    center_euler_xyz_rad = center_angles[0]
    boundary_geojson = water_boundary_scene_frame(
        water_polygon, center + translation, center_euler_xyz_rad
    )
    boundary_distance_m = _local_boundary_distance_m(boundary_geojson)
    source_path = (
        output_root
        / "scene_inputs"
        / role
        / f"{Path(record['key']).stem}_boundary_local.geojson"
    )
    explicit_geometry_counts: dict[str, int | float] = {}
    if geometry_method == "native_line_buffer":
        coastlines = load_native_shoreline_strips(
            simulator, boundary_geojson, source_path
        )
    elif geometry_method == "explicit_segment_rectangles":
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(
            json.dumps(boundary_geojson, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        coastlines, explicit_geometry_counts = explicit_segment_strip_coastlines(
            simulator, boundary_geojson
        )
    else:
        raise ValueError(
            f"unsupported shoreline input geometry method: {geometry_method}"
        )
    strip_polygons = [Polygon(segment.points) for segment in coastlines]
    ownship_in_strip = any(
        polygon.covers(Point(0.0, 0.0)) for polygon in strip_polygons
    )
    ownship_allowance_m = (
        math.sqrt(2.0) * SHORELINE_STRIP_WIDTH_M / 2.0
        if geometry_method == "explicit_segment_rectangles"
        else SHORELINE_STRIP_WIDTH_M / 2.0 + 0.5
    )
    if ownship_in_strip and boundary_distance_m > ownship_allowance_m:
        raise ValueError(
            "ownship fell inside a shoreline strip outside the measured near-shore allowance"
        )
    ticks = renderer_ticks(scan.encoder)
    scene = build_explicit_mode1_scene(simulator, ticks, seed=seed, bins=expected_bins)
    scene_path = (
        output_root / "scenes" / role / f"{Path(record['key']).stem}_scene.json"
    )
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(
        json.dumps(
            {
                **scene,
                "coastlines": [
                    {"name": c.name, "points": c.points} for c in coastlines
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    render_dir = output_root / "renders" / role / Path(record["key"]).stem
    render_dir.mkdir(parents=True, exist_ok=False)
    base_time = dt.datetime.fromtimestamp(
        scan.center_time_us / 1e6, tz=dt.timezone.utc
    ).replace(tzinfo=None)
    generated = simulator.simulate_from_scene(
        scene,
        coastlines,
        [],
        str(render_dir),
        scene_path=scene_path.name,
        num_rotations=1,
        verbose=False,
        base_time=base_time,
    )
    csv_paths = [
        Path(path)
        for path in generated
        if str(path).lower().endswith(".csv") and "detection" not in str(path).lower()
    ]
    if len(csv_paths) != 1:
        raise RuntimeError(f"expected one rendered CSV; got {len(csv_paths)}")
    raw_codes = parse_renderer_csv(csv_paths[0], ticks, expected_bins)
    metadata_path = (
        csv_paths[0].parent
        / "scene_metadata"
        / csv_paths[0].stem
        / f"{csv_paths[0].stem}_metadata.json"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError("renderer omitted its run metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_cfg = metadata.get("scene", {}).get("radar_config", {})
    expected_config = {
        "status": 1,
        "scale": 496,
        "range_index": 0,
        "gain": 75,
        "bins": expected_bins,
        "bin_size_m": RADIAL_STEP_M,
        "dead_zone_bins": 5,
        "range_m": MAX_RANGE_M,
        "ticks_per_rev": TICKS_PER_REV,
    }
    for key, expected in expected_config.items():
        if actual_cfg.get(key) != expected:
            raise ValueError(
                f"renderer metadata {key}={actual_cfg.get(key)!r}, expected {expected!r}"
            )
    elapsed = time.perf_counter() - render_started
    provenance = {
        "scan_key": record["key"],
        "scan_sha256": record["sha256"],
        "row_count": len(scan.encoder),
        "echo_bins": expected_bins,
        "renderer_angle_ticks_sha256": hashlib.sha256(
            np.asarray(ticks, dtype="<u2").tobytes()
        ).hexdigest(),
        "render_csv": str(csv_paths[0].relative_to(output_root)),
        "render_csv_sha256": sha256_file(csv_paths[0]),
        "scene_json": str(scene_path.relative_to(output_root)),
        "scene_json_sha256": sha256_file(scene_path),
        "boundary_local_geojson": str(source_path.relative_to(output_root)),
        "boundary_local_geojson_sha256": sha256_file(source_path),
        "shoreline_input_geometry_method": geometry_method,
        "shoreline_strip_count": len(coastlines),
        **explicit_geometry_counts,
        "shoreline_strip_width_m": SHORELINE_STRIP_WIDTH_M,
        "shoreline_strip_smoothing": (
            "none; one explicit rectangle per nonzero source segment"
            if geometry_method == "explicit_segment_rectangles"
            else "native line buffer; two Chaikin iterations and polygon smoothing"
        ),
        "ownship_distance_to_source_boundary_m": boundary_distance_m,
        "ownship_within_buffered_land_strip": ownship_in_strip,
        "near_shore_exception": bool(ownship_in_strip),
        "static_pose_yaw_motion_summary": motion,
        "scan_center_euler_xyz_rad": center_euler_xyz_rad.tolist(),
        "map_enu_to_legacy_renderer_xy_matrix": map_enu_to_legacy_matrix(
            center_euler_xyz_rad
        ).tolist(),
        "render_runtime_s": elapsed,
        "renderer_spoke_iterations": len(ticks),
        "renderer_ms_per_spoke": 1000.0 * elapsed / max(1, len(ticks)),
        "peak_rss_bytes_before_render": peak_rss_before,
        "peak_rss_bytes_after_render": _peak_rss_bytes(),
        "renderer_actual_radar_config": actual_cfg,
        "renderer_metadata_sha256": sha256_file(metadata_path),
    }
    return raw_codes / 252.0, provenance


def _project_reference(
    scan: RadarScan, poses: PoseTable, native_max_range_m: float, range_offset_m: float
) -> tuple[FloatArray, BoolArray]:
    projected = project_scan_with_native_range(
        scan,
        poses,
        native_max_range_m,
        MAX_RANGE_M,
        range_offset_m=range_offset_m,
        min_range_m=2.5,
        radial_step_m=RADIAL_STEP_M,
    )
    if not np.isfinite(projected.echo[projected.valid]).all():
        raise ValueError("reference echoes must be finite on common valid cells")
    return preserve_reference_display_255(projected.echo), projected.valid


def _render_record(
    simulator: Any,
    role: str,
    record: dict[str, Any],
    sequence_dir: Path,
    poses: PoseTable,
    translation: FloatArray,
    water_polygon: BaseGeometry,
    output_dir: Path,
    expected_hashes: dict[str, str],
    seed: int,
    geometry_method: str = "native_line_buffer",
) -> tuple[FloatArray, FloatArray, BoolArray, dict[str, Any]]:
    scan_path = sequence_dir / "radar" / Path(record["key"]).name
    source_key = (
        f"{record['sequence']}/radar/{scan_path.name}" if "sequence" in record else None
    )
    expected = expected_hashes.get(source_key or record["key"])
    if expected is None:
        expected = record.get("sha256")
    if expected is None:
        raise ValueError(f"selected source manifest has no hash for {record['key']}")
    osm.verify_hash(scan_path, expected, f"CANOE scan {record['key']}")
    scan = load_radar_png(scan_path)
    native_max_range, range_offset = osm._range_config(sequence_dir, sequence_dir.name)
    reference, valid = _project_reference(scan, poses, native_max_range, range_offset)
    raw, render_provenance = _render_one(
        simulator,
        scan,
        poses,
        translation,
        water_polygon,
        output_dir,
        role,
        {**record, "sha256": expected},
        reference.shape[1],
        seed,
        geometry_method,
    )
    if raw.shape != reference.shape:
        raise ValueError(
            f"renderer grid {raw.shape} does not match CANOE grid {reference.shape}"
        )
    render_provenance.update(
        {
            "source_scan": str(scan_path),
            "source_scan_sha256": expected,
            "native_max_range_m": native_max_range,
            "range_offset_m": range_offset,
            "valid_spokes": int(
                (
                    scan.valid_azimuth & pose_time_supported(poses, scan.row_time_us)
                ).sum()
            ),
            "valid_cells": int(valid.sum()),
        }
    )
    return raw, reference, valid, render_provenance


def runtime_manifest(config_path: Path) -> dict[str, Any]:
    return {
        "git_revision": checkout_revision(REPO_ROOT),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "canoe_utility_sha256": sha256_file(Path(canoe.__file__)),
        "osm_runner_sha256": sha256_file(
            REPO_ROOT / "scripts" / "run_canoe_osm_diagnostic.py"
        ),
        "uv_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
    }


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(
    config_path: Path,
    data_root: Path,
    source_root: Path,
    output_dir: Path,
    *,
    score_requested: bool,
    smoke_train: bool,
    manifest_dir: Path,
) -> None:
    if score_requested == smoke_train:
        raise ValueError("choose exactly one of --score or --smoke-train")
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "run_status.json"
    status: dict[str, Any] = {
        "status": "started",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "smoke_train_only_no_scoring"
        if smoke_train
        else "frozen_retrospective_scoring",
        "data_root": str(data_root),
        "source_root": str(source_root),
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "uv_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
    }
    _write_status(status_path, status)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        status["provenance"] = runtime_manifest(config_path)
        status["config_input_hashes"] = {
            "selection_manifest": config["dataset"]["selection_manifest_sha256"],
            "source_manifest": config["dataset"]["source_manifest_sha256"],
            "osm_geometry": config["independent_geometry"]["geometry_sha256"],
            "pose_by_role": config["dataset"]["pose_input_sha256"],
            "radar_config": config["dataset"]["radar_config_sha256"],
        }
        _write_status(status_path, status)
        status["status"] = "validating_inputs"
        _write_status(status_path, status)
        roles = ("train", "validation", "test")
        records = _load_records(config, data_root, manifest_dir, roles)
        status["config_input_hashes"]["source_manifest"] = sha256_file(
            _manifest_path(manifest_dir, config, "source_manifest")
        )
        alignment = _alignment_gates(config, source_root, records)
        status["alignment_gates"] = alignment
        _write_status(status_path, status)

        geometry_path = (
            source_root
            / "source_inputs"
            / Path(config["independent_geometry"]["geometry_path"]).name
        )
        osm.verify_hash(
            geometry_path,
            config["independent_geometry"]["geometry_sha256"],
            "OSM geometry",
        )
        origin = tuple(config["georeferencing"]["local_enu_origin_lat_lon_deg"])
        water_polygon = osm.load_water_polygon(geometry_path, origin)
        translation = np.asarray(
            config["georeferencing"]["pose_to_map_translation_m"], dtype=np.float64
        )

        simulator, original_simulator_sha, original_renderer_sha = _load_renderer(
            config["renderer"]["source_sha256"],
        )
        status["original_simulator_sha256"] = original_simulator_sha
        status["original_object_renderer_sha256"] = original_renderer_sha
        status["status"] = "renderer_loaded"
        _write_status(status_path, status)

        source_manifest_path = _manifest_path(manifest_dir, config, "source_manifest")
        expected_hashes = osm._expected_scan_hashes(source_manifest_path)
        train_items, train_dir, train_poses, _ = records["train"]
        calibration = [
            item for item in train_items if item["subset"] == "train_calibration"
        ]
        if len(calibration) != config["dataset"]["frozen_scans"]["train_calibration"]:
            raise ValueError("train calibration count differs from frozen split")
        if smoke_train:
            entry = calibration[0]
            scan_path = train_dir / "radar" / Path(entry["key"]).name
            osm.verify_hash(
                scan_path, expected_hashes[entry["key"]], "smoke train scan"
            )
            scan = load_radar_png(scan_path)
            native_max_range, range_offset = osm._range_config(
                train_dir, train_dir.name
            )
            smoke_reference, smoke_valid = _project_reference(
                scan, train_poses, native_max_range, range_offset
            )
            rendered, render_provenance = _render_one(
                simulator,
                scan,
                train_poses,
                translation,
                water_polygon,
                output_dir,
                "train_smoke",
                {**entry, "sha256": expected_hashes[entry["key"]]},
                int(config["renderer"]["radar"]["bins"]),
                int(config["renderer"]["seed"]),
                config["renderer"].get("input_geometry_method", "native_line_buffer"),
            )
            if rendered.shape != (
                len(scan.encoder),
                int(config["renderer"]["radar"]["bins"]),
            ):
                raise ValueError(
                    "smoke render shape does not match native CANOE spoke/bin shape"
                )
            if smoke_reference.shape != rendered.shape or not smoke_valid.any():
                raise ValueError(
                    "smoke observation support does not match the 300-bin renderer grid"
                )
            if not np.isfinite(rendered[smoke_valid]).all():
                raise ValueError(
                    "smoke render is missing predictions on a valid observed cell"
                )
            render_provenance.update(
                {
                    "native_max_range_m": native_max_range,
                    "range_offset_m": range_offset,
                    "valid_cells": int(smoke_valid.sum()),
                    "valid_spokes": int(
                        (
                            scan.valid_azimuth
                            & pose_time_supported(train_poses, scan.row_time_us)
                        ).sum()
                    ),
                    "prediction_finite_on_all_valid_cells": True,
                    "observed_display_divisor": 255,
                    "renderer_echo_divisor": 252,
                }
            )
            status.update(
                {
                    "status": "complete_smoke_no_scores",
                    "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "smoke_scan": render_provenance,
                    "rendered_shape": list(rendered.shape),
                    "row_angle_correspondence": "verified exactly",
                    "output_note": "Only one train calibration scan was rendered. No echo comparison or score was computed.",
                }
            )
            _write_status(status_path, status)
            return

        status["status"] = "rendering_train_calibration"
        _write_status(status_path, status)
        cal_raw: list[FloatArray] = []
        cal_ref: list[FloatArray] = []
        cal_masks: list[BoolArray] = []
        calibration_render_provenance: list[dict[str, Any]] = []
        for entry in calibration:
            raw, reference, valid, info = _render_record(
                simulator,
                "train_calibration",
                entry,
                train_dir,
                train_poses,
                translation,
                water_polygon,
                output_dir,
                expected_hashes,
                int(config["renderer"]["seed"]),
                config["renderer"].get("input_geometry_method", "native_line_buffer"),
            )
            cal_raw.append(raw)
            cal_ref.append(reference)
            cal_masks.append(valid)
            calibration_render_provenance.append(info)
        gain = fit_scalar_gain(
            np.concatenate(cal_raw, axis=0),
            np.concatenate(cal_ref, axis=0),
            np.concatenate(cal_masks, axis=0),
        )
        range_profile = fit_range_profile(cal_ref, cal_masks)
        status.update(
            {
                "calibration_scans": len(calibration),
                "training_only_scalar_gain": gain,
                "training_only_range_profile_bins": int(range_profile.size),
                "status": "rendering_retrospective_splits",
            }
        )
        _write_status(status_path, status)

        rows: list[dict[str, Any]] = []
        all_render_provenance = list(calibration_render_provenance)
        for role in ("validation", "test"):
            items, sequence_dir, poses, _ = records[role]
            for index, entry in enumerate(items):
                raw, reference, valid, info = _render_record(
                    simulator,
                    role,
                    entry,
                    sequence_dir,
                    poses,
                    translation,
                    water_polygon,
                    output_dir,
                    expected_hashes,
                    int(config["renderer"]["seed"]),
                    config["renderer"].get(
                        "input_geometry_method", "native_line_buffer"
                    ),
                )
                all_render_provenance.append(info)
                methods = {
                    "legacy_mode1_raw_divided_by_252": raw,
                    "legacy_mode1_train30_scalar_gain": apply_scalar_gain(raw, gain),
                    "zero_return": np.zeros_like(reference),
                    "train30_range_profile_mean": np.broadcast_to(
                        range_profile[None, :], reference.shape
                    ),
                }
                gain_unclipped = raw * gain
                clipping_fraction = (
                    float(
                        np.count_nonzero(gain_unclipped[valid] > 1.0) / int(valid.sum())
                    )
                    if valid.any()
                    else 0.0
                )
                for method, prediction in methods.items():
                    metrics = score_prediction(
                        reference,
                        prediction,
                        valid,
                        float(
                            config["calibration_and_baselines"]["occupancy_threshold"][
                                "value"
                            ]
                        ),
                    )
                    rows.append(
                        {
                            "split": role,
                            "scan_index": index,
                            "scan_file": Path(entry["key"]).name,
                            "method": method,
                            "calibration_gain": gain
                            if method == "legacy_mode1_train30_scalar_gain"
                            else "",
                            "gain_prediction_clipping_fraction": clipping_fraction
                            if method == "legacy_mode1_train30_scalar_gain"
                            else "",
                            "valid_cells": metrics["valid_cells"],
                            "mae": metrics["mae"],
                            "rmse": metrics["rmse"],
                            "reference_positive_cells": metrics[
                                "reference_positive_cells"
                            ],
                            "predicted_positive_cells": metrics[
                                "predicted_positive_cells"
                            ],
                            "occupancy_intersection_cells": metrics[
                                "occupancy_intersection_cells"
                            ],
                            "occupancy_union_cells": metrics["occupancy_union_cells"],
                            "occupancy_iou": metrics["occupancy_iou"],
                            "valid_spokes": info["valid_spokes"],
                            "valid_fraction_of_source_spokes": info["valid_spokes"]
                            / info["row_count"],
                            "render_runtime_s": info["render_runtime_s"],
                        }
                    )
        result_path = output_dir / "canoe_legacy_shoreline_per_scan_metrics.csv"
        with result_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        aggregate: dict[str, Any] = {}
        for role in ("validation", "test"):
            aggregate[role] = {}
            for method in sorted(
                {str(row["method"]) for row in rows if row["split"] == role}
            ):
                group = [
                    row
                    for row in rows
                    if row["split"] == role and row["method"] == method
                ]
                aggregate[role][method] = {
                    "frame_count": len(group),
                    "equal_weight_mean_frame_mae": float(
                        np.mean([float(row["mae"]) for row in group])
                    ),
                    "equal_weight_mean_frame_rmse": float(
                        np.mean([float(row["rmse"]) for row in group])
                    ),
                    "equal_weight_mean_frame_occupancy_iou": (
                        float(
                            np.mean(
                                [
                                    float(row["occupancy_iou"])
                                    for row in group
                                    if row["occupancy_iou"] is not None
                                ]
                            )
                        )
                        if any(row["occupancy_iou"] is not None for row in group)
                        else None
                    ),
                }
        summary = {
            "experiment": config["experiment"],
            "status": "retrospective validation and test scores computed",
            "provenance": status["provenance"],
            "original_simulator_sha256": original_simulator_sha,
            "original_object_renderer_sha256": original_renderer_sha,
            "alignment_gates": alignment,
            "independent_geometry_sha256": config["independent_geometry"][
                "geometry_sha256"
            ],
            "independent_geometry_indexed_date": config["independent_geometry"][
                "osm_indexed_date"
            ],
            "renderer": config["renderer"],
            "train_calibration_scan_count": len(calibration),
            "train_only_scalar_gain": gain,
            "occupancy_threshold_from_existing_train_only_diagnostic": config[
                "calibration_and_baselines"
            ]["occupancy_threshold"]["value"],
            "train_only_range_profile_fit": "mean normalized observed echo by radial bin over the same 30 calibration scans and valid support",
            "scored_scans": {
                role: sum(row["split"] == role for row in rows) // 4
                for role in ("validation", "test")
            },
            "aggregate_metrics": aggregate,
            "render_inputs_and_hashes": all_render_provenance,
            "per_scan_metrics_csv": result_path.name,
            "limitations": config["interpretation_limits"],
        }
        summary_path = output_dir / "canoe_legacy_shoreline_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        status.update(
            {
                "status": "complete",
                "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "summary_sha256": sha256_file(summary_path),
                "per_scan_metrics_sha256": sha256_file(result_path),
                "render_count": len(all_render_provenance),
            }
        )
        _write_status(status_path, status)
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_status(status_path, status)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/canoe-map-evaluation.json",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke-train",
        action="store_true",
        help="Render one frozen train calibration scan without scoring.",
    )
    mode.add_argument(
        "--score",
        action="store_true",
        help="Explicitly run the frozen retrospective train-fit/validation/test protocol.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(
            args.config,
            args.data_root,
            args.source_root,
            args.output_dir,
            score_requested=args.score,
            smoke_train=args.smoke_train,
            manifest_dir=args.manifests,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"CANOE legacy shoreline diagnostic stopped: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
