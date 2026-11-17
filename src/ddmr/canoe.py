"""Small, auditable CANOE radar and pose utilities for a map projection test."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class RadarScan:
    """One decoded PNG scan. Echo data exclude the eleven metadata columns."""

    path: str
    center_time_us: int
    row_time_us: NDArray[np.uint64]
    encoder: NDArray[np.uint16]
    valid_azimuth: BoolArray
    echo: NDArray[np.uint8]


@dataclass(frozen=True)
class PoseTable:
    """Provided sensor states expressed in the fixed ENU reference frame."""

    time_us: NDArray[np.float64]
    position_enu: FloatArray
    euler_xyz: FloatArray


@dataclass(frozen=True)
class EmpiricalMap:
    """Sparse-observation raster with maximum normalized intensity per cell."""

    origin_xy: tuple[float, float]
    resolution_m: float
    values: FloatArray


@dataclass(frozen=True)
class ProjectedScan:
    """Range-aggregated observation and its world coordinates per polar cell."""

    echo: FloatArray
    valid: BoolArray
    x_enu: FloatArray
    y_enu: FloatArray
    pose_supported: BoolArray


def load_radar_png(path: str | Path) -> RadarScan:
    """Decode public CANOE/Oxford radar PNG metadata and preserve its row mask."""
    source = Path(path)
    image = np.asarray(Image.open(source))
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("radar PNG must decode as a 2-D uint8 grayscale image")
    if image.shape[1] <= 11:
        raise ValueError("radar PNG is missing the eleven metadata columns")
    rows = image.shape[0]
    # The official pycanoe decoder views the metadata bytes as native uint64 and
    # uint16 values. CANOE compute hosts are little-endian, matching its PNGs.
    row_time = np.frombuffer(image[:, :8].copy().tobytes(), dtype="<u8", count=rows)
    encoder = np.frombuffer(image[:, 8:10].copy().tobytes(), dtype="<u2", count=rows)
    valid = image[:, 10] == 255
    if np.any(encoder >= 5600):
        raise ValueError("encoder value exceeds the 5600-count CANOE revolution")
    if np.any(row_time == 0):
        raise ValueError("azimuth timestamps must be nonzero UTC microseconds")
    center = int(source.stem)
    return RadarScan(
        path=str(source),
        center_time_us=center,
        row_time_us=row_time,
        encoder=encoder,
        valid_azimuth=valid,
        echo=image[:, 11:].copy(),
    )


def load_pose_table(path: str | Path) -> PoseTable:
    """Read the documented radar_poses CSV; require its full field names."""
    expected = (
        "utc_micro",
        "x",
        "y",
        "z",
        "ang_x",
        "ang_y",
        "ang_z",
    )
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(expected).issubset(reader.fieldnames):
            raise ValueError(
                "radar_poses CSV does not match the documented CANOE schema"
            )
        rows = list(reader)
    values = np.asarray(
        [[float(row[name]) for name in expected] for row in rows], dtype=np.float64
    )
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("radar_poses CSV must contain at least two finite states")
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    times = values[:, 0]
    if np.any(np.diff(times) <= 0):
        raise ValueError("radar_poses timestamps must be strictly increasing")
    angles = np.unwrap(values[:, 4:7], axis=0)
    return PoseTable(time_us=times, position_enu=values[:, 1:4], euler_xyz=angles)


def radar_range_resolution_m(max_range_m: float, range_bin_count: int) -> float:
    """Use the official CANOE loader convention, maximum range / image bins."""
    if not np.isfinite(max_range_m) or max_range_m <= 0 or range_bin_count <= 0:
        raise ValueError("maximum range and radial bin count must be positive")
    return float(max_range_m / range_bin_count)


def interpolate_pose(
    table: PoseTable, time_us: NDArray[np.uint64]
) -> tuple[FloatArray, FloatArray]:
    """Interpolate within pose support and clamp values for unsupported rows.

    Call ``pose_time_supported`` and mask unsupported azimuth rows before
    mapping or scoring; endpoint values only keep the coordinate arrays finite.
    """
    query = np.asarray(time_us, dtype=np.float64)
    query = np.clip(query, table.time_us[0], table.time_us[-1])
    position = np.column_stack(
        [
            np.interp(query, table.time_us, table.position_enu[:, axis])
            for axis in range(3)
        ]
    )
    angles = np.column_stack(
        [np.interp(query, table.time_us, table.euler_xyz[:, axis]) for axis in range(3)]
    )
    return position, angles


def pose_time_supported(table: PoseTable, time_us: NDArray[np.uint64]) -> BoolArray:
    """Return azimuth times covered by the provided pose table."""
    query = np.asarray(time_us, dtype=np.float64)
    return (query >= table.time_us[0]) & (query <= table.time_us[-1])


def _c_enuref_sensor(roll: float, pitch: float, yaw: float) -> FloatArray:
    """Match pycanoe's C1(roll) @ C2(pitch) @ C3(yaw) convention."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    c1 = np.array([[1.0, 0.0, 0.0], [0.0, cr, sr], [0.0, -sr, cr]])
    c2 = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    c3 = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return c1 @ c2 @ c3


def scan_positive_returns(
    scan: RadarScan,
    poses: PoseTable,
    range_resolution_m: float,
    range_offset_m: float = -0.407910,
    point_limit: int = 4167,
    min_range_m: float = 2.5,
) -> tuple[FloatArray, FloatArray]:
    """Return a uniform deterministic sample of positive returns in ENU XY."""
    if range_resolution_m <= 0 or point_limit <= 0:
        raise ValueError("range resolution and point limit must be positive")
    row_ids, bin_ids = np.nonzero((scan.echo > 0) & scan.valid_azimuth[:, None])
    if row_ids.size == 0:
        return np.empty((0, 2)), np.empty(0)
    if row_ids.size > point_limit:
        keep = np.linspace(0, row_ids.size - 1, point_limit, dtype=np.int64)
        row_ids, bin_ids = row_ids[keep], bin_ids[keep]
    row_times = scan.row_time_us[row_ids]
    pose_supported = pose_time_supported(poses, row_times)
    row_ids, bin_ids = row_ids[pose_supported], bin_ids[pose_supported]
    row_times = row_times[pose_supported]
    if row_ids.size == 0:
        return np.empty((0, 2)), np.empty(0)
    positions, angles = interpolate_pose(poses, row_times)
    ranges = (bin_ids.astype(np.float64) + 0.5) * range_resolution_m + range_offset_m
    keep = ranges >= min_range_m
    row_ids, ranges, positions, angles = (
        row_ids[keep],
        ranges[keep],
        positions[keep],
        angles[keep],
    )
    encoder_angle = scan.encoder[row_ids].astype(np.float64) * (2.0 * np.pi / 5600.0)
    # The official CANOE sensor basis is x-forward, y-right. The encoder is
    # zero-forward and increases clockwise, so local rays are
    # [r cos(a), r sin(a), 0].
    local_xy = np.column_stack(
        (ranges * np.cos(encoder_angle), ranges * np.sin(encoder_angle))
    )
    world_xy = np.empty_like(local_xy)
    for index, (roll, pitch, yaw) in enumerate(angles):
        world_xy[index] = (
            _c_enuref_sensor(roll, pitch, yaw)[:2, :2] @ local_xy[index]
            + positions[index, :2]
        )
    intensity = scan.echo[row_ids, bin_ids[keep]].astype(np.float64) / 255.0
    return world_xy, intensity


def build_empirical_map(
    point_sets: list[tuple[FloatArray, FloatArray]], resolution_m: float = 0.5
) -> EmpiricalMap:
    """Rasterize training returns, retaining maximum intensity per ENU voxel."""
    if resolution_m <= 0 or not point_sets:
        raise ValueError("map resolution and training point sets must be nonempty")
    nonempty = [(xy, v) for xy, v in point_sets if len(xy) and len(v)]
    if not nonempty:
        raise ValueError("training returns contain no positive map points")
    points = np.concatenate([xy for xy, _ in nonempty], axis=0)
    intensity = np.concatenate([v for _, v in nonempty], axis=0)
    origin = np.floor(points.min(axis=0) / resolution_m) * resolution_m
    maximum = np.floor(points.max(axis=0) / resolution_m) * resolution_m
    shape = np.floor((maximum - origin) / resolution_m).astype(np.int64) + 1
    if np.any(shape > 30000):
        raise ValueError("empirical map extent exceeds the 30,000-cell safety bound")
    cells = np.floor((points - origin) / resolution_m).astype(np.int64)
    grid = np.full(tuple(shape), -np.inf, dtype=np.float64)
    np.maximum.at(grid, (cells[:, 0], cells[:, 1]), intensity)
    grid[~np.isfinite(grid)] = np.nan
    return EmpiricalMap((float(origin[0]), float(origin[1])), resolution_m, grid)


def map_lookup(
    map_grid: EmpiricalMap,
    x: FloatArray,
    y: FloatArray,
    interpolation: str = "bilinear",
) -> tuple[FloatArray, BoolArray]:
    """Sample a map at ENU points, returning values and map-support mask."""
    gx = (
        np.asarray(x, dtype=np.float64) - map_grid.origin_xy[0]
    ) / map_grid.resolution_m
    gy = (
        np.asarray(y, dtype=np.float64) - map_grid.origin_xy[1]
    ) / map_grid.resolution_m
    out = np.full(np.broadcast_shapes(gx.shape, gy.shape), np.nan, dtype=np.float64)
    if interpolation == "nearest":
        ix, iy = np.rint(gx).astype(np.int64), np.rint(gy).astype(np.int64)
        inside = (
            (ix >= 0)
            & (iy >= 0)
            & (ix < map_grid.values.shape[0])
            & (iy < map_grid.values.shape[1])
        )
        out[inside] = map_grid.values[ix[inside], iy[inside]]
        return out, np.isfinite(out)
    if interpolation != "bilinear":
        raise ValueError("interpolation must be nearest or bilinear")
    x0, y0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    dx, dy = gx - x0, gy - y0
    weight_sum = np.zeros_like(out)
    for ox, oy, weight in (
        (0, 0, (1 - dx) * (1 - dy)),
        (1, 0, dx * (1 - dy)),
        (0, 1, (1 - dx) * dy),
        (1, 1, dx * dy),
    ):
        ix, iy = x0 + ox, y0 + oy
        inside = (
            (ix >= 0)
            & (iy >= 0)
            & (ix < map_grid.values.shape[0])
            & (iy < map_grid.values.shape[1])
        )
        values = np.full_like(out, np.nan)
        values[inside] = map_grid.values[ix[inside], iy[inside]]
        valid = np.isfinite(values)
        out[valid] = np.nan_to_num(out[valid]) + values[valid] * weight[valid]
        weight_sum[valid] += weight[valid]
    support = weight_sum > 0
    out[support] /= weight_sum[support]
    out[~support] = np.nan
    return out, support


def project_scan_to_common_grid(
    scan: RadarScan,
    poses: PoseTable,
    range_resolution_m: float,
    *,
    range_offset_m: float = -0.407910,
    min_range_m: float = 2.5,
    max_range_m: float = 150.0,
    radial_step_m: float = 0.5,
) -> ProjectedScan:
    """Area-aggregate native range bins and transform each azimuth to ENU."""
    if min(range_resolution_m, max_range_m, radial_step_m) <= 0:
        raise ValueError("range and grid resolutions must be positive")
    ranges = (
        np.arange(scan.echo.shape[1], dtype=np.float64) + 0.5
    ) * range_resolution_m + range_offset_m
    radial_edges = np.arange(0.0, max_range_m + radial_step_m, radial_step_m)
    bins = len(radial_edges) - 1
    radial_ids = np.floor(ranges / radial_step_m).astype(np.int64)
    radial_keep = (
        (ranges >= min_range_m)
        & (ranges < radial_edges[-1])
        & (radial_ids >= 0)
        & (radial_ids < bins)
    )
    source_ids = np.flatnonzero(radial_keep)
    echo = np.full((scan.echo.shape[0], bins), np.nan, dtype=np.float64)
    if source_ids.size:
        labels = radial_ids[source_ids]
        for row in np.flatnonzero(scan.valid_azimuth):
            sums = np.bincount(
                labels, weights=scan.echo[row, source_ids], minlength=bins
            )
            counts = np.bincount(labels, minlength=bins)
            present = counts > 0
            echo[row, present] = sums[present] / counts[present] / 255.0
    pose_supported = pose_time_supported(poses, scan.row_time_us)
    valid = np.broadcast_to(
        (scan.valid_azimuth & pose_supported)[:, None], echo.shape
    ).copy() & np.isfinite(echo)
    positions, angles = interpolate_pose(poses, scan.row_time_us)
    bearing = scan.encoder.astype(np.float64) * (2.0 * np.pi / 5600.0)
    range_centers = (np.arange(bins, dtype=np.float64) + 0.5) * radial_step_m
    local_x = np.cos(bearing)[:, None] * range_centers[None, :]
    local_y = np.sin(bearing)[:, None] * range_centers[None, :]
    x_enu = np.full_like(echo, np.nan)
    y_enu = np.full_like(echo, np.nan)
    for row in range(angles.shape[0]):
        roll, pitch, yaw = angles[row, 0], angles[row, 1], angles[row, 2]
        rotation = _c_enuref_sensor(roll, pitch, yaw)
        x_enu[row] = (
            positions[row, 0]
            + rotation[0, 0] * local_x[row]
            + rotation[0, 1] * local_y[row]
        )
        y_enu[row] = (
            positions[row, 1]
            + rotation[1, 0] * local_x[row]
            + rotation[1, 1] * local_y[row]
        )
    return ProjectedScan(echo, valid, x_enu, y_enu, pose_supported)


def project_scan_with_native_range(
    scan: RadarScan,
    poses: PoseTable,
    native_max_range_m: float,
    analysis_max_range_m: float,
    *,
    range_offset_m: float = -0.407910,
    min_range_m: float = 2.5,
    radial_step_m: float = 0.5,
) -> ProjectedScan:
    """Project native bins using sensor mode, then apply a separate range crop."""
    resolution = radar_range_resolution_m(native_max_range_m, scan.echo.shape[1])
    return project_scan_to_common_grid(
        scan,
        poses,
        resolution,
        range_offset_m=range_offset_m,
        min_range_m=min_range_m,
        max_range_m=analysis_max_range_m,
        radial_step_m=radial_step_m,
    )
