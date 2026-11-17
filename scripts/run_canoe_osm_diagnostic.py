"""Map loading and pose alignment for the CANOE shoreline evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from ddmr.canoe import (
    PoseTable,
    RadarScan,
    _c_enuref_sensor,
    interpolate_pose,
    pose_time_supported,
)

FloatArray = NDArray[np.float64]
EARTH_A_M = 6_378_137.0
EARTH_E2 = 6.69437999014e-3
ROUTE_MARKER_RE = re.compile(
    r"L\.marker\(\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]"
)


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


def load_route_markers(path: Path) -> FloatArray:
    """Return ordered route-map markers as latitude/longitude degrees."""
    html = path.read_text(encoding="utf-8", errors="replace")
    markers = np.asarray(
        [[float(lat), float(lon)] for lat, lon in ROUTE_MARKER_RE.findall(html)],
        dtype=np.float64,
    )
    if markers.ndim != 2 or markers.shape[0] < 10 or markers.shape[1] != 2:
        raise ValueError(f"route map has too few parsable GPS markers: {path}")
    if not np.isfinite(markers).all() or np.any(np.abs(markers[:, 0]) > 90):
        raise ValueError(f"route map contains invalid GPS markers: {path}")
    return markers


def _ecef(lat_deg: float, lon_deg: float) -> FloatArray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    normal = EARTH_A_M / math.sqrt(1.0 - EARTH_E2 * math.sin(lat) ** 2)
    return np.asarray(
        [
            normal * math.cos(lat) * math.cos(lon),
            normal * math.cos(lat) * math.sin(lon),
            normal * (1.0 - EARTH_E2) * math.sin(lat),
        ],
        dtype=np.float64,
    )


def latlon_to_local_enu(
    lat_lon_deg: FloatArray, origin_lat_lon_deg: tuple[float, float]
) -> FloatArray:
    """Project WGS84 latitude/longitude coordinates into a local ENU tangent."""
    points = np.asarray(lat_lon_deg, dtype=np.float64)
    one_point = points.ndim == 1
    points = np.atleast_2d(points)
    origin = _ecef(*origin_lat_lon_deg)
    delta = np.stack([_ecef(float(lat), float(lon)) - origin for lat, lon in points])
    phi, lam = map(math.radians, origin_lat_lon_deg)
    east = -math.sin(lam) * delta[:, 0] + math.cos(lam) * delta[:, 1]
    north = (
        -math.sin(phi) * math.cos(lam) * delta[:, 0]
        - math.sin(phi) * math.sin(lam) * delta[:, 1]
        + math.cos(phi) * delta[:, 2]
    )
    result = np.column_stack((east, north))
    return result[0] if one_point else result


def _nearest_path_residuals(
    pose_xy: FloatArray, route_xy: FloatArray, translation_xy: FloatArray
) -> FloatArray:
    distances, _ = KDTree(route_xy).query(pose_xy + translation_xy, k=1)
    return np.asarray(distances, dtype=np.float64)


def fit_route_translation(pose_xy: FloatArray, route_xy: FloatArray) -> FloatArray:
    """Fit a single route-minus-pose translation using TRAIN trajectory only."""
    if pose_xy.ndim != 2 or pose_xy.shape[1] != 2 or len(pose_xy) < 2:
        raise ValueError("pose trajectory must be an N-by-2 array")
    if route_xy.ndim != 2 or route_xy.shape[1] != 2 or len(route_xy) < 10:
        raise ValueError("route map must contain at least ten 2-D markers")
    translation = np.median(route_xy, axis=0) - np.median(pose_xy, axis=0)
    tree = KDTree(route_xy)
    for _ in range(20):
        _, nearest = tree.query(pose_xy + translation, k=1)
        updated = np.median(route_xy[nearest] - pose_xy, axis=0)
        if float(np.linalg.norm(updated - translation)) < 1e-9:
            translation = updated
            break
        translation = updated
    return translation


def alignment_summary(
    pose_xy: FloatArray, route_xy: FloatArray, translation: FloatArray
) -> dict[str, float]:
    residual = _nearest_path_residuals(pose_xy, route_xy, translation)
    return {
        "median_nearest_route_residual_m": float(np.quantile(residual, 0.5)),
        "p90_nearest_route_residual_m": float(np.quantile(residual, 0.9)),
        "maximum_nearest_route_residual_m": float(np.max(residual)),
    }


def _transform_geometry_latlon(
    geometry: BaseGeometry, origin_lat_lon_deg: tuple[float, float]
) -> BaseGeometry:
    def project(lon: Any, lat: Any, z: Any = None) -> Any:
        lon_array, lat_array = np.asarray(lon), np.asarray(lat)
        pts = np.column_stack((lat_array.reshape(-1), lon_array.reshape(-1)))
        xy = latlon_to_local_enu(pts, origin_lat_lon_deg)
        east = xy[:, 0].reshape(lat_array.shape)
        north = xy[:, 1].reshape(lat_array.shape)
        if np.ndim(lon) == 0:
            return (
                (float(east), float(north))
                if z is None
                else (float(east), float(north), z)
            )
        return (east, north) if z is None else (east, north, z)

    return shapely_transform(project, geometry)


def load_water_polygon(
    path: Path, origin_lat_lon_deg: tuple[float, float]
) -> BaseGeometry:
    source = json.loads(path.read_text(encoding="utf-8"))
    feature = source.get("feature", source)
    geometry = shape(feature["geometry"])
    if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("OSM input must contain a nonempty Polygon or MultiPolygon")
    if not geometry.is_valid:
        raise ValueError("OSM water polygon is invalid; geometry repair is prohibited")
    local = _transform_geometry_latlon(geometry, origin_lat_lon_deg)
    if local.is_empty or not local.is_valid:
        raise ValueError("projected OSM polygon is empty or invalid; no repair applied")
    return local


def _geometry_points(geometry: BaseGeometry) -> list[Point]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [geometry]
    if isinstance(geometry, MultiPoint):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [point for child in geometry.geoms for point in _geometry_points(child)]
    if geometry.geom_type in {"LineString", "LinearRing"}:
        coords = list(geometry.coords)
        if not coords:
            return []
        return [Point(coords[0]), Point(coords[-1])]
    if hasattr(geometry, "geoms"):
        return [point for child in geometry.geoms for point in _geometry_points(child)]
    return []


def first_positive_intersection_range(
    origin_xy: tuple[float, float],
    direction_xy_per_range: tuple[float, float],
    max_range_m: float,
    boundary: BaseGeometry,
) -> float | None:
    """Return nearest positive slant range to a 2-D shoreline boundary."""
    direction = np.asarray(direction_xy_per_range, dtype=np.float64)
    horizontal_norm = float(np.linalg.norm(direction))
    if horizontal_norm <= 1e-9:
        return None
    start = np.asarray(origin_xy, dtype=np.float64)
    end = start + direction * max_range_m
    ray = LineString((start, end))
    intersections = ray.intersection(boundary)
    candidates: list[float] = []
    for point in _geometry_points(intersections):
        distance_xy = float(point.distance(Point(*origin_xy)))
        slant_range = distance_xy / horizontal_norm
        if 1e-6 < slant_range <= max_range_m + 1e-9:
            candidates.append(slant_range)
    return min(candidates) if candidates else None


def sensor_ray_direction_enu_xy(
    encoder_count: int, roll_pitch_yaw: tuple[float, float, float]
) -> FloatArray:
    """Map one CANOE clockwise encoder bearing into the ENU horizontal plane."""
    bearing = float(encoder_count) * (2.0 * math.pi / 5600.0)
    # Official CANOE sensor coordinates are x-forward, y-right.
    local_ray = np.asarray([math.cos(bearing), math.sin(bearing), 0.0])
    return _c_enuref_sensor(*roll_pitch_yaw)[:2, :] @ local_ray


def shoreline_ranges(
    scan: RadarScan,
    poses: PoseTable,
    translation_xy: FloatArray,
    polygon: BaseGeometry,
    max_range_m: float,
) -> tuple[FloatArray, NDArray[np.bool_], NDArray[np.bool_]]:
    """Compute ray-boundary ranges and per-spoke pose and containment masks."""
    supported = pose_time_supported(poses, scan.row_time_us)
    positions, angles = interpolate_pose(poses, scan.row_time_us)
    ranges = np.full(scan.echo.shape[0], np.nan, dtype=np.float64)
    inside = np.zeros(scan.echo.shape[0], dtype=bool)
    boundary = polygon.boundary
    for row in np.flatnonzero(scan.valid_azimuth & supported):
        x, y = positions[row, :2] + translation_xy
        inside[row] = polygon.covers(Point(float(x), float(y)))
        attitude: tuple[float, float, float] = (
            float(angles[row, 0]),
            float(angles[row, 1]),
            float(angles[row, 2]),
        )
        direction_xy = sensor_ray_direction_enu_xy(int(scan.encoder[row]), attitude)
        ranges[row] = (
            first_positive_intersection_range(
                (float(x), float(y)),
                (float(direction_xy[0]), float(direction_xy[1])),
                max_range_m,
                boundary,
            )
            or np.nan
        )
    return ranges, supported, inside


def gaussian_return_profile(
    shoreline_range_m: FloatArray,
    bin_centers_m: FloatArray,
    sigma_m: float,
) -> FloatArray:
    if sigma_m <= 0:
        raise ValueError("Gaussian radial standard deviation must be positive")
    valid = np.isfinite(shoreline_range_m)
    prediction = np.zeros(
        (len(shoreline_range_m), len(bin_centers_m)), dtype=np.float64
    )
    delta = bin_centers_m[None, :] - shoreline_range_m[valid, None]
    prediction[valid] = np.exp(-0.5 * (delta / sigma_m) ** 2)
    return prediction


def fit_gain(
    prediction: FloatArray, reference: FloatArray, valid: NDArray[np.bool_]
) -> float:
    mask = valid & np.isfinite(reference)
    x, y = prediction[mask], reference[mask]
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        return 0.0
    return float(np.clip(np.dot(x, y) / denominator, 0.0, 1.0))


def score(
    reference: FloatArray,
    prediction: FloatArray,
    valid: NDArray[np.bool_],
    threshold: float,
) -> dict[str, float | int | None]:
    if reference.shape != prediction.shape or reference.shape != valid.shape:
        raise ValueError("reference, prediction, and valid mask must share one shape")
    if not np.isfinite(reference[valid]).all():
        raise ValueError("valid observed cells contain nonfinite reference echoes")
    if not np.isfinite(prediction[valid]).all():
        raise ValueError("model prediction is nonfinite on a valid observed cell")
    ref, pred = reference[valid], prediction[valid]
    if not ref.size:
        raise ValueError("no common valid polar cells for scoring")
    ref_active, pred_active = ref > threshold, pred > threshold
    intersection = int(np.count_nonzero(ref_active & pred_active))
    union = int(np.count_nonzero(ref_active | pred_active))
    predicted_count, reference_count = int(pred_active.sum()), int(ref_active.sum())
    delta = pred - ref
    return {
        "valid_cells": int(ref.size),
        "reference_active_cells": reference_count,
        "predicted_active_cells": predicted_count,
        "occupancy_intersection_cells": intersection,
        "occupancy_union_cells": union,
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "occupancy_iou": float(intersection / union) if union else None,
        "occupancy_precision": (
            float(intersection / predicted_count) if predicted_count else None
        ),
        "occupancy_recall": (
            float(intersection / reference_count) if reference_count else None
        ),
    }


def _range_config(sequence_dir: Path, sequence: str) -> tuple[float, float]:
    text = (sequence_dir / "calib" / "radar_config.yaml").read_text(encoding="utf-8")
    match = re.search(
        rf"^\s+{re.escape(sequence)}:\s*([0-9.]+)\s*$", text, re.MULTILINE
    )
    offset = re.search(r"^range_offset:\s*([-+0-9.eE]+)", text, re.MULTILINE)
    if match is None or offset is None:
        raise ValueError(f"cannot read CANOE range config for {sequence}")
    return float(match.group(1)), float(offset.group(1))


def _expected_scan_hashes(source_manifest_path: Path) -> dict[str, str]:
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    return {entry["key"]: entry["sha256"] for entry in source["selected_scans"]}
