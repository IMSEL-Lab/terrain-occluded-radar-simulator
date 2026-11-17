"""Optional ctypes bridge to the Rust radar core.

The bridge is intentionally fail-soft. If the DLL is missing, blocked, or
loaded on the wrong architecture, callers should keep using the Python/NumPy
fallbacks instead of crashing the simulator.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable

import numpy as np


_LIB = None
_LOAD_ERROR: str | None = None


class RadarLocSummary(ctypes.Structure):
    _fields_ = [
        ("file_bytes", ctypes.c_uint64),
        ("coastline_count", ctypes.c_uint64),
        ("total_points", ctypes.c_uint64),
        ("max_points_per_coastline", ctypes.c_uint64),
        ("vessel_count", ctypes.c_uint64),
        ("terrain_rows", ctypes.c_uint64),
        ("terrain_cols", ctypes.c_uint64),
        ("range_nm", ctypes.c_double),
    ]


def _candidate_library_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    release_dir = root / "radar_core_rs" / "target" / "release"
    debug_dir = root / "radar_core_rs" / "target" / "debug"
    names = [
        "radar_core_rs.dll",
        "libradar_core_rs.so",
        "libradar_core_rs.dylib",
    ]
    return [folder / name for folder in (release_dir, debug_dir) for name in names]


def _load_library():
    global _LIB, _LOAD_ERROR
    if os.environ.get("RADAR_CORE_DISABLE_RUST", "").lower() in {"1", "true", "yes"}:
        _LOAD_ERROR = "disabled by RADAR_CORE_DISABLE_RUST"
        return None
    if _LIB is not None:
        return _LIB
    for path in _candidate_library_paths():
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            _configure_signatures(lib)
            _LIB = lib
            _LOAD_ERROR = None
            return _LIB
        except OSError as exc:
            _LOAD_ERROR = f"{path}: {exc}"
    if _LOAD_ERROR is None:
        _LOAD_ERROR = "radar_core_rs library not found"
    return None


def _configure_signatures(lib) -> None:
    double_p = ctypes.POINTER(ctypes.c_double)
    lib.radar_core_rs_available.argtypes = []
    lib.radar_core_rs_available.restype = ctypes.c_uint32

    lib.radar_core_peak_furuno_code.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ]
    lib.radar_core_peak_furuno_code.restype = ctypes.c_double

    lib.radar_core_apply_stc_f64.argtypes = [
        double_p, double_p, ctypes.c_size_t, double_p,
    ]
    lib.radar_core_apply_stc_f64.restype = ctypes.c_int32

    lib.radar_core_log_compress_f64.argtypes = [
        double_p, ctypes.c_size_t, ctypes.c_double, double_p,
    ]
    lib.radar_core_log_compress_f64.restype = ctypes.c_int32

    lib.radar_core_quantize_f64.argtypes = [
        double_p, ctypes.c_size_t, ctypes.c_uint32, double_p,
    ]
    lib.radar_core_quantize_f64.restype = ctypes.c_int32

    lib.radar_core_process_signal_chain_f64.argtypes = [
        double_p, double_p, ctypes.c_size_t, ctypes.c_double,
        ctypes.c_uint32, double_p,
    ]
    lib.radar_core_process_signal_chain_f64.restype = ctypes.c_int32

    lib.radar_core_entry_r4_depth_attenuation_f64.argtypes = [
        double_p, ctypes.c_size_t, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        double_p,
    ]
    lib.radar_core_entry_r4_depth_attenuation_f64.restype = ctypes.c_int32

    lib.radar_core_generate_furuno_angle_ticks_u16.argtypes = [
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t,
    ]
    lib.radar_core_generate_furuno_angle_ticks_u16.restype = ctypes.c_int32

    lib.radar_core_quantize_sweep_to_furuno_codes_u16.argtypes = [
        double_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.radar_core_quantize_sweep_to_furuno_codes_u16.restype = ctypes.c_int32

    lib.radar_core_paint_object_echo_f64.argtypes = [
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, ctypes.c_double,
        double_p, ctypes.c_size_t,
    ]
    lib.radar_core_paint_object_echo_f64.restype = ctypes.c_int32

    lib.radar_core_radarloc_summary_utf8.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.POINTER(RadarLocSummary),
    ]
    lib.radar_core_radarloc_summary_utf8.restype = ctypes.c_int32

    lib.radar_core_polar_intersection_returns_f64.argtypes = [
        double_p, double_p, ctypes.c_size_t,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_size_t,
        ctypes.c_double, ctypes.c_size_t, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_uint32, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_uint32, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        double_p,
    ]
    lib.radar_core_polar_intersection_returns_f64.restype = ctypes.c_int32

    lib.radar_core_polar_intersection_batch_returns_f64.argtypes = [
        double_p, double_p,
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
        double_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_uint32, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_uint32, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        double_p,
    ]
    lib.radar_core_polar_intersection_batch_returns_f64.restype = ctypes.c_int32


def rust_available() -> bool:
    lib = _load_library()
    if lib is None:
        return False
    try:
        return bool(lib.radar_core_rs_available())
    except Exception:
        return False


def rust_load_error() -> str | None:
    _load_library()
    return _LOAD_ERROR


def _array(values: Iterable[float]) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _out_like(values: np.ndarray) -> np.ndarray:
    return np.empty(len(values), dtype=np.float64)


def peak_furuno_code(
    rcs_m2: float,
    length_m: float,
    width_m: float,
    heading_deg: float,
    range_m: float,
    beam_gain: float,
) -> float:
    lib = _load_library()
    if lib is None:
        raise RuntimeError(_LOAD_ERROR or "Rust radar core unavailable")
    return float(lib.radar_core_peak_furuno_code(
        float(rcs_m2), float(length_m), float(width_m), float(heading_deg),
        float(range_m), float(beam_gain),
    ))


def apply_stc(sweep_linear, stc_curve) -> np.ndarray:
    lib = _load_library()
    if lib is None:
        raise RuntimeError(_LOAD_ERROR or "Rust radar core unavailable")
    sweep = _array(sweep_linear)
    stc = _array(stc_curve)
    if len(stc) < len(sweep):
        stc = np.pad(stc, (0, len(sweep) - len(stc)))
    elif len(stc) > len(sweep):
        stc = stc[:len(sweep)]
    out = _out_like(sweep)
    code = lib.radar_core_apply_stc_f64(
        sweep.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        stc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(len(sweep)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if code != 0:
        raise RuntimeError(f"radar_core_apply_stc_f64 failed: {code}")
    return out


def log_compress(sweep_linear, dynamic_range_db: float) -> np.ndarray:
    lib = _load_library()
    if lib is None:
        raise RuntimeError(_LOAD_ERROR or "Rust radar core unavailable")
    sweep = _array(sweep_linear)
    out = _out_like(sweep)
    code = lib.radar_core_log_compress_f64(
        sweep.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(len(sweep)),
        ctypes.c_double(float(dynamic_range_db)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if code != 0:
        raise RuntimeError(f"radar_core_log_compress_f64 failed: {code}")
    return out


def quantize(normalized_sweep, bits: int) -> np.ndarray:
    lib = _load_library()
    if lib is None:
        raise RuntimeError(_LOAD_ERROR or "Rust radar core unavailable")
    sweep = _array(normalized_sweep)
    out = _out_like(sweep)
    code = lib.radar_core_quantize_f64(
        sweep.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(len(sweep)),
        ctypes.c_uint32(int(bits)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if code != 0:
        raise RuntimeError(f"radar_core_quantize_f64 failed: {code}")
    return out


def process_signal_chain(sweep_linear, stc_curve, dynamic_range_db: float,
                         bits: int) -> np.ndarray:
    lib = _load_library()
    if lib is None:
        raise RuntimeError(_LOAD_ERROR or "Rust radar core unavailable")
    sweep = _array(sweep_linear)
    stc = _array(stc_curve)
    if len(stc) < len(sweep):
        stc = np.pad(stc, (0, len(sweep) - len(stc)))
    elif len(stc) > len(sweep):
        stc = stc[:len(sweep)]
    out = _out_like(sweep)
    code = lib.radar_core_process_signal_chain_f64(
        sweep.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        stc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(len(sweep)),
        ctypes.c_double(float(dynamic_range_db)),
        ctypes.c_uint32(int(bits)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if code != 0:
        raise RuntimeError(f"radar_core_process_signal_chain_f64 failed: {code}")
    return out


def entry_r4_depth_attenuation(bin_ranges_m, entry_range_m: float,
                               config_values: tuple[float, ...]) -> np.ndarray:
    if len(config_values) != 8:
        raise ValueError("config_values must contain 8 floats")
    lib = _load_library()
    if lib is None:
        raise RuntimeError(_LOAD_ERROR or "Rust radar core unavailable")
    ranges = _array(bin_ranges_m)
    out = _out_like(ranges)
    code = lib.radar_core_entry_r4_depth_attenuation_f64(
        ranges.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(len(ranges)),
        ctypes.c_double(float(entry_range_m)),
        *[ctypes.c_double(float(value)) for value in config_values],
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if code != 0:
        raise RuntimeError(f"radar_core_entry_r4_depth_attenuation_f64 failed: {code}")
    return out


def generate_furuno_angle_ticks(spokes_per_rotation: int) -> np.ndarray:
    lib = _load_library()
    if lib is None:
        raise RuntimeError(_LOAD_ERROR or "Rust radar core unavailable")
    spokes = max(1, int(spokes_per_rotation))
    out = np.empty(spokes, dtype=np.uint16)
    code = lib.radar_core_generate_furuno_angle_ticks_u16(
        ctypes.c_size_t(spokes),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ctypes.c_size_t(len(out)),
    )
    if code != 0:
        raise RuntimeError(f"radar_core_generate_furuno_angle_ticks_u16 failed: {code}")
    return out


def quantize_sweep_to_furuno_codes(echo_values, dead_zone_bins: int) -> np.ndarray:
    lib = _load_library()
    if lib is None:
        raise RuntimeError(_LOAD_ERROR or "Rust radar core unavailable")
    echoes = _array(echo_values)
    out = np.empty(len(echoes), dtype=np.uint16)
    code = lib.radar_core_quantize_sweep_to_furuno_codes_u16(
        echoes.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(len(echoes)),
        ctypes.c_size_t(max(0, int(dead_zone_bins))),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    if code != 0:
        raise RuntimeError(f"radar_core_quantize_sweep_to_furuno_codes_u16 failed: {code}")
    return out


def paint_object_echo(center_bin: int, spread_bins: int, peak_code: float,
                      dynamic_range_db: float, num_bins: int,
                      existing=None) -> np.ndarray:
    lib = _load_library()
    if lib is None:
        raise RuntimeError(_LOAD_ERROR or "Rust radar core unavailable")
    if existing is None:
        out = np.zeros(max(0, int(num_bins)), dtype=np.float64)
    else:
        out = _array(existing).copy()
    code = lib.radar_core_paint_object_echo_f64(
        ctypes.c_size_t(max(0, int(center_bin))),
        ctypes.c_size_t(max(1, int(spread_bins))),
        ctypes.c_double(float(peak_code)),
        ctypes.c_double(float(dynamic_range_db)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(len(out)),
    )
    if code != 0:
        raise RuntimeError(f"radar_core_paint_object_echo_f64 failed: {code}")
    return out


def polar_intersection_returns(
    points_xy,
    *,
    own_x: float,
    own_y: float,
    bearing_deg: float,
    beamwidth_deg: float,
    max_range_m: float,
    num_bins: int,
    reflectivity: float,
    num_rays: int,
    edge_factor: float,
    leading_edge_boost: float,
    far_range_floor: float,
    range_decay_strength: float,
    core_depth_scale: float,
    interior_decay_scale: float,
    attenuation_model_entry_r4: bool,
    continuous_land_return: bool,
    continuous_return_threshold: float,
    continuous_return_depth_m: float,
    continuous_texture_strength: float,
    continuous_texture_floor: float,
    origin_range_attenuation: bool,
    origin_range_ref_m: float,
    origin_range_power: float,
    origin_range_floor: float,
    occlude_after_first_land: bool,
    secondary_return_scale: float,
    secondary_return_max_code: float,
    attenuation_config_values: tuple[float, ...],
) -> np.ndarray:
    if len(attenuation_config_values) != 8:
        raise ValueError("attenuation_config_values must contain 8 floats")
    lib = _load_library()
    if lib is None:
        raise RuntimeError(_LOAD_ERROR or "Rust radar core unavailable")
    points = np.ascontiguousarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must be an Nx2 array")
    xs = np.ascontiguousarray(points[:, 0], dtype=np.float64)
    ys = np.ascontiguousarray(points[:, 1], dtype=np.float64)
    bins = max(0, int(num_bins))
    out = np.zeros(bins, dtype=np.float64)
    code = lib.radar_core_polar_intersection_returns_f64(
        xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ys.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(len(points)),
        ctypes.c_double(float(own_x)),
        ctypes.c_double(float(own_y)),
        ctypes.c_double(float(bearing_deg)),
        ctypes.c_double(float(beamwidth_deg)),
        ctypes.c_double(float(max_range_m)),
        ctypes.c_size_t(bins),
        ctypes.c_double(float(reflectivity)),
        ctypes.c_size_t(max(3, int(num_rays))),
        ctypes.c_double(float(edge_factor)),
        ctypes.c_double(float(leading_edge_boost)),
        ctypes.c_double(float(far_range_floor)),
        ctypes.c_double(float(range_decay_strength)),
        ctypes.c_double(float(core_depth_scale)),
        ctypes.c_double(float(interior_decay_scale)),
        ctypes.c_uint32(1 if attenuation_model_entry_r4 else 0),
        ctypes.c_uint32(1 if continuous_land_return else 0),
        ctypes.c_double(float(continuous_return_threshold)),
        ctypes.c_double(float(continuous_return_depth_m)),
        ctypes.c_double(float(continuous_texture_strength)),
        ctypes.c_double(float(continuous_texture_floor)),
        ctypes.c_uint32(1 if origin_range_attenuation else 0),
        ctypes.c_double(float(origin_range_ref_m)),
        ctypes.c_double(float(origin_range_power)),
        ctypes.c_double(float(origin_range_floor)),
        ctypes.c_uint32(1 if occlude_after_first_land else 0),
        ctypes.c_double(float(secondary_return_scale)),
        ctypes.c_double(float(secondary_return_max_code)),
        *[ctypes.c_double(float(value)) for value in attenuation_config_values],
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if code != 0:
        raise RuntimeError(f"radar_core_polar_intersection_returns_f64 failed: {code}")
    return out


def polar_intersection_batch_returns(
    flat_points_xy,
    starts,
    lengths,
    reflectivities,
    *,
    own_x: float,
    own_y: float,
    bearing_deg: float,
    beamwidth_deg: float,
    max_range_m: float,
    num_bins: int,
    num_rays: int,
    edge_factor: float,
    leading_edge_boost: float,
    far_range_floor: float,
    range_decay_strength: float,
    core_depth_scale: float,
    interior_decay_scale: float,
    attenuation_model_entry_r4: bool,
    continuous_land_return: bool,
    continuous_return_threshold: float,
    continuous_return_depth_m: float,
    continuous_texture_strength: float,
    continuous_texture_floor: float,
    origin_range_attenuation: bool,
    origin_range_ref_m: float,
    origin_range_power: float,
    origin_range_floor: float,
    occlude_after_first_land: bool,
    secondary_return_scale: float,
    secondary_return_max_code: float,
    attenuation_config_values: tuple[float, ...],
) -> np.ndarray:
    if len(attenuation_config_values) != 8:
        raise ValueError("attenuation_config_values must contain 8 floats")
    lib = _load_library()
    if lib is None:
        raise RuntimeError(_LOAD_ERROR or "Rust radar core unavailable")
    points = np.ascontiguousarray(flat_points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("flat_points_xy must be an Nx2 array")
    starts_arr = np.ascontiguousarray(starts, dtype=np.uintp)
    lengths_arr = np.ascontiguousarray(lengths, dtype=np.uintp)
    reflect_arr = np.ascontiguousarray(reflectivities, dtype=np.float64)
    if not (len(starts_arr) == len(lengths_arr) == len(reflect_arr)):
        raise ValueError("starts, lengths, and reflectivities must have matching length")
    xs = np.ascontiguousarray(points[:, 0], dtype=np.float64)
    ys = np.ascontiguousarray(points[:, 1], dtype=np.float64)
    bins = max(0, int(num_bins))
    out = np.zeros(bins, dtype=np.float64)
    code = lib.radar_core_polar_intersection_batch_returns_f64(
        xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ys.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        starts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        lengths_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        reflect_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(len(starts_arr)),
        ctypes.c_size_t(len(points)),
        ctypes.c_double(float(own_x)),
        ctypes.c_double(float(own_y)),
        ctypes.c_double(float(bearing_deg)),
        ctypes.c_double(float(beamwidth_deg)),
        ctypes.c_double(float(max_range_m)),
        ctypes.c_size_t(bins),
        ctypes.c_size_t(max(3, int(num_rays))),
        ctypes.c_double(float(edge_factor)),
        ctypes.c_double(float(leading_edge_boost)),
        ctypes.c_double(float(far_range_floor)),
        ctypes.c_double(float(range_decay_strength)),
        ctypes.c_double(float(core_depth_scale)),
        ctypes.c_double(float(interior_decay_scale)),
        ctypes.c_uint32(1 if attenuation_model_entry_r4 else 0),
        ctypes.c_uint32(1 if continuous_land_return else 0),
        ctypes.c_double(float(continuous_return_threshold)),
        ctypes.c_double(float(continuous_return_depth_m)),
        ctypes.c_double(float(continuous_texture_strength)),
        ctypes.c_double(float(continuous_texture_floor)),
        ctypes.c_uint32(1 if origin_range_attenuation else 0),
        ctypes.c_double(float(origin_range_ref_m)),
        ctypes.c_double(float(origin_range_power)),
        ctypes.c_double(float(origin_range_floor)),
        ctypes.c_uint32(1 if occlude_after_first_land else 0),
        ctypes.c_double(float(secondary_return_scale)),
        ctypes.c_double(float(secondary_return_max_code)),
        *[ctypes.c_double(float(value)) for value in attenuation_config_values],
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if code != 0:
        raise RuntimeError(f"radar_core_polar_intersection_batch_returns_f64 failed: {code}")
    return out


def radarloc_summary(
    path: str | os.PathLike,
    *,
    max_file_bytes: int = 256 * 1024 * 1024,
    max_total_points: int = 5_000_000,
    max_coastlines: int = 100_000,
    max_terrain_cells: int = 4_000_000,
) -> dict:
    lib = _load_library()
    if lib is None:
        raise RuntimeError(_LOAD_ERROR or "Rust radar core unavailable")
    path_bytes = os.fsencode(Path(path))
    path_arr = (ctypes.c_uint8 * len(path_bytes)).from_buffer_copy(path_bytes)
    summary = RadarLocSummary()
    code = lib.radar_core_radarloc_summary_utf8(
        path_arr,
        ctypes.c_size_t(len(path_bytes)),
        ctypes.c_uint64(int(max_file_bytes)),
        ctypes.c_uint64(int(max_total_points)),
        ctypes.c_uint64(int(max_coastlines)),
        ctypes.c_uint64(int(max_terrain_cells)),
        ctypes.byref(summary),
    )
    if code != 0:
        raise RuntimeError(f"radar_core_radarloc_summary_utf8 failed: {code}")
    return {
        "file_bytes": int(summary.file_bytes),
        "coastline_count": int(summary.coastline_count),
        "total_points": int(summary.total_points),
        "max_points_per_coastline": int(summary.max_points_per_coastline),
        "vessel_count": int(summary.vessel_count),
        "terrain_rows": int(summary.terrain_rows),
        "terrain_cols": int(summary.terrain_cols),
        "range_nm": float(summary.range_nm),
    }
