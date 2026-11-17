"""Shared object echo renderer for Object Creation.

This module centralizes object support generation and per-bin intensity
synthesis so the simulator can use one object-rendering path across
reference-informed and native synthetic modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

try:
    import pandas as _pd
    _HAS_PANDAS = True
except ImportError:
    _pd = None
    _HAS_PANDAS = False


DEFAULT_REF_RANGE_M = 50.0
DEFAULT_REF_RCS_M2 = 350.0
DEFAULT_REF_CODE = 190.0
DEFAULT_COMPRESSION = 55.0
DEFAULT_SUPPORT_FLOOR_CODE = 8.0

OBJECT_TYPE_DB_CORRECTIONS = {
    "buoy": 5.5,
    "small_boat": 2.0,
    "medium_boat": 1.5,
    "large_boat": 3.0,
    "barge": 3.5,
}

OBJECT_TYPE_PROFILE_CFG = {
    "buoy": {"plateau_frac": 0.00, "gamma": 2.8, "sigma_scale": 0.45},
    "small_boat": {"plateau_frac": 0.10, "gamma": 2.2, "sigma_scale": 0.55},
    "medium_boat": {"plateau_frac": 0.18, "gamma": 1.8, "sigma_scale": 0.70},
    "large_boat": {"plateau_frac": 0.28, "gamma": 1.5, "sigma_scale": 0.85},
    "barge": {"plateau_frac": 0.38, "gamma": 1.3, "sigma_scale": 1.00},
}

OBJECT_TYPE_SPREAD_CAPS = {
    "buoy": 2,
    "small_boat": 5,
    "medium_boat": 8,
    "large_boat": 12,
    "barge": 15,
}

OBJECT_TYPE_SUPPORT_LIMITS = {
    "buoy": (1, 9),
    "small_boat": (3, 21),
    "medium_boat": (5, 33),
    "large_boat": (7, 49),
    "barge": (9, 61),
}


@dataclass
class ObjectRenderPriors:
    ref_range_m: float = DEFAULT_REF_RANGE_M
    ref_rcs_m2: float = DEFAULT_REF_RCS_M2
    ref_code: float = DEFAULT_REF_CODE
    compression: float = DEFAULT_COMPRESSION
    spread_bins: Optional[float] = None
    support_length_bins: Optional[int] = None
    support_floor_code: float = DEFAULT_SUPPORT_FLOOR_CODE
    reference_pattern: Optional[list[float]] = None
    reference_center_bin: Optional[int] = None
    reference_profile_blend: float = 0.0


@dataclass
class ObjectSupportSegment:
    start_bin: int
    end_bin: int
    values: Optional[list[int]] = None

    @property
    def length_bins(self) -> int:
        return max(0, int(self.end_bin) - int(self.start_bin) + 1)


@dataclass
class ObjectRenderResult:
    echoes: np.ndarray
    support_source: str
    support_segments_used: int
    rendered_bins: int
    peak_db: Optional[float]


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _safe_ratio_float(numer: float, denom: float, default: float = 0.0) -> float:
    denom = float(denom)
    if abs(denom) < 1e-9:
        return float(default)
    return float(numer) / denom


def display_db_from_code_value(value: float) -> float:
    safe = max(0.0, float(value))
    return 20.0 * math.log10((safe + 1.0) / 252.0)


def code_value_from_display_db(display_db: float) -> float:
    return _clamp(252.0 * (10.0 ** (float(display_db) / 20.0)) - 1.0, 0.0, 252.0)


def shape_fraction_drop_db(fraction: float) -> float:
    return 20.0 * math.log10(max(1e-6, float(fraction)))


def _round_clamped_code(value: float) -> int:
    return int(min(252, max(0, round(float(value)))))


def _normalized_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    return (float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0


def _projected_object_extents(length_m: float, width_m: float, aspect_deg: float) -> tuple[float, float]:
    aspect_rad = math.radians(float(aspect_deg))
    radial_extent = abs(float(length_m) * math.cos(aspect_rad)) + abs(float(width_m) * math.sin(aspect_rad))
    cross_extent = abs(float(length_m) * math.sin(aspect_rad)) + abs(float(width_m) * math.cos(aspect_rad))
    return float(radial_extent), float(cross_extent)


def _object_support_phase_label(offset: int, seg_len: int) -> str:
    if seg_len <= 2:
        return "short"
    if offset <= 1:
        return "entry"
    if offset >= seg_len - 2:
        return "exit"
    return "interior"


def _object_peak_correction_db(obj_type: str, range_m: float) -> float:
    type_db = OBJECT_TYPE_DB_CORRECTIONS.get(obj_type, OBJECT_TYPE_DB_CORRECTIONS["medium_boat"])
    if range_m < 35.0:
        range_db = 2.0
    elif range_m < 80.0:
        range_db = 0.75
    elif range_m < 140.0:
        range_db = -0.5
    elif range_m < 175.0:
        range_db = 0.5
    else:
        range_db = 2.0
    return type_db + range_db


def _object_support_correction_db(obj_type: str, support_len_bins: int, rel_pos: float) -> float:
    if support_len_bins <= 2:
        return -2.0
    edge_db = 1.6 if (rel_pos <= 0.2 or rel_pos >= 0.8) else 0.0
    if obj_type == "buoy" and edge_db > 0.0:
        edge_db += 0.5
    return edge_db


def _estimate_geometry_spread_bins(obj, scene_bin_size: float) -> float:
    aspect_deg = _normalized_angle_diff_deg(
        math.degrees(math.atan2(float(obj.x), float(obj.y))) % 360.0,
        float(obj.heading_deg),
    )
    projected_radial_extent_m, _ = _projected_object_extents(
        float(obj.length_m),
        float(obj.width_m),
        aspect_deg,
    )
    spread_cap = OBJECT_TYPE_SPREAD_CAPS.get(getattr(obj, "obj_type", "medium_boat"), 8)
    return float(max(2, min(spread_cap, int(projected_radial_extent_m / max(float(scene_bin_size), 1e-6) / 6.0))))


def _normalize_reference_pattern(pattern: Optional[Sequence[float]], target_len: int) -> Optional[np.ndarray]:
    if not pattern or target_len <= 0:
        return None
    arr = np.asarray([max(0.0, float(v)) for v in pattern], dtype=np.float64)
    if arr.size == 0:
        return None
    peak = float(np.max(arr))
    if peak <= 0.0:
        return None
    arr = arr / peak
    if arr.size == target_len:
        return arr
    x_src = np.linspace(0.0, 1.0, arr.size)
    x_dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_dst, x_src, arr)


def _build_support_profile(obj_type: str, spread_bins: float, length_bins: int,
                           reference_pattern: Optional[Sequence[float]] = None,
                           reference_profile_blend: float = 0.0) -> np.ndarray:
    n = max(1, int(length_bins))
    if n == 1:
        return np.ones(1, dtype=np.float64)

    cfg = OBJECT_TYPE_PROFILE_CFG.get(obj_type, OBJECT_TYPE_PROFILE_CFG["medium_boat"])
    x = np.linspace(0.0, 1.0, n)
    plateau_frac = float(_clamp(cfg["plateau_frac"] + 0.02 * min(float(spread_bins), 10.0), 0.0, 0.55))
    left_end = max(0.05, 0.5 - plateau_frac / 2.0)
    right_start = min(0.95, 0.5 + plateau_frac / 2.0)
    gamma = max(1.1, float(cfg["gamma"]))

    profile = np.ones(n, dtype=np.float64)
    left_mask = x < left_end
    right_mask = x > right_start
    if np.any(left_mask):
        profile[left_mask] = np.power(np.clip(x[left_mask] / left_end, 0.0, 1.0), gamma)
    if np.any(right_mask):
        profile[right_mask] = np.power(
            np.clip((1.0 - x[right_mask]) / max(1e-6, 1.0 - right_start), 0.0, 1.0),
            gamma,
        )

    sigma = max(0.18, min(0.95, float(cfg["sigma_scale"]) * float(spread_bins) / max(2.0, n)))
    gaussian = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
    profile = 0.55 * profile + 0.45 * gaussian

    ref_profile = _normalize_reference_pattern(reference_pattern, n)
    blend = _clamp(reference_profile_blend, 0.0, 1.0)
    if ref_profile is not None and blend > 0.0:
        profile = (1.0 - blend) * profile + blend * ref_profile

    profile = np.clip(profile, 0.0, None)
    peak = float(np.max(profile))
    return profile / peak if peak > 0 else np.ones(n, dtype=np.float64)


def _estimate_support_length_bins(obj, scene_bin_size: float, spread_bins: float,
                                  support_length_bins: Optional[int],
                                  reference_pattern: Optional[Sequence[float]]) -> int:
    obj_type = getattr(obj, "obj_type", "medium_boat")
    min_bins, max_bins = OBJECT_TYPE_SUPPORT_LIMITS.get(obj_type, OBJECT_TYPE_SUPPORT_LIMITS["medium_boat"])
    if support_length_bins is not None and support_length_bins > 0:
        candidate = int(round(float(support_length_bins)))
    else:
        candidate = int(round(4.0 * float(spread_bins) + 1.0))
    if reference_pattern:
        candidate = max(candidate, len(reference_pattern))
    return int(max(min_bins, min(max_bins, candidate)))


def _normalize_support_segments(support_segments: Optional[Sequence[Any]],
                                scene_bins: int,
                                scene_dead_zone: int) -> list[ObjectSupportSegment]:
    normalized: list[ObjectSupportSegment] = []
    for seg in support_segments or []:
        start_bin: Optional[int] = None
        end_bin: Optional[int] = None
        values: Optional[list[int]] = None
        if isinstance(seg, dict):
            try:
                start_bin = int(seg.get("start_bin", 0))
            except (TypeError, ValueError):
                start_bin = None
            raw_vals = seg.get("values")
            if isinstance(raw_vals, (list, tuple)):
                values = [int(v) for v in raw_vals]
            try:
                if seg.get("end_bin") is not None:
                    end_bin = int(seg.get("end_bin"))
            except (TypeError, ValueError):
                end_bin = None
        elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
            try:
                start_bin = int(seg[0])
            except (TypeError, ValueError):
                start_bin = None
            if isinstance(seg[1], (list, tuple)):
                values = [int(v) for v in seg[1]]
            if len(seg) >= 3:
                try:
                    end_bin = int(seg[2])
                except (TypeError, ValueError):
                    end_bin = None
        if start_bin is None:
            continue
        if end_bin is None:
            seg_len = len(values or [])
            if seg_len <= 0:
                continue
            end_bin = start_bin + seg_len - 1
        start_bin = max(scene_dead_zone, int(start_bin))
        end_bin = min(scene_bins - 1, int(end_bin))
        if end_bin < start_bin:
            continue
        normalized.append(ObjectSupportSegment(start_bin=start_bin, end_bin=end_bin, values=values))
    return normalized


def _build_geometry_support_segments(obj, priors: ObjectRenderPriors, scene_bins: int,
                                     scene_bin_size: float, scene_dead_zone: int) -> list[ObjectSupportSegment]:
    spread_bins = float(priors.spread_bins) if priors.spread_bins is not None else _estimate_geometry_spread_bins(obj, scene_bin_size)
    support_len = _estimate_support_length_bins(
        obj,
        scene_bin_size,
        spread_bins,
        priors.support_length_bins,
        priors.reference_pattern,
    )
    obj_range = math.hypot(float(obj.x), float(obj.y))
    center_bin = int(obj_range / max(float(scene_bin_size), 1e-6))
    start_bin = center_bin - support_len // 2
    end_bin = start_bin + support_len - 1
    if end_bin < scene_dead_zone or start_bin >= scene_bins:
        return []
    start_bin = max(scene_dead_zone, start_bin)
    end_bin = min(scene_bins - 1, end_bin)
    if end_bin < start_bin:
        return []
    return [ObjectSupportSegment(start_bin=start_bin, end_bin=end_bin, values=None)]


def _build_segment_context(obj, raw_obj: Optional[dict], bearing_deg: float,
                           obj_bearing_deg: float, obj_range_m: float, beam_offset_deg: float,
                           seg_len: int, start_bin: int, end_bin: int, scene_bin_size: float,
                           baseline_segment_peak_code: float, baseline_segment_peak_db: float) -> dict:
    raw = raw_obj or {}
    obj_type = raw.get("type") or raw.get("geom_type") or getattr(obj, "obj_type", "unknown")
    geom_type = raw.get("geom_type", obj_type)
    material_proxy = raw.get("material", geom_type)
    length_m = float(getattr(obj, "length_m", raw.get("length_m", 0.0)) or 0.0)
    width_m = float(getattr(obj, "width_m", raw.get("width_m", 0.0)) or 0.0)
    radial_bins = float(raw.get("radial_bins", 0.0) or 0.0)
    angular_spokes = float(raw.get("angular_spokes", 0.0) or 0.0)
    obj_heading_deg = float(getattr(obj, "heading_deg", raw.get("heading_deg", 0.0)) or 0.0)
    aspect_to_spoke_deg = _normalized_angle_diff_deg(bearing_deg, obj_heading_deg)
    aspect_to_radar_deg = _normalized_angle_diff_deg(obj_bearing_deg, obj_heading_deg)
    projected_radial_extent_m, projected_cross_range_extent_m = _projected_object_extents(
        length_m,
        width_m,
        aspect_to_spoke_deg,
    )
    support_len_m = float(seg_len) * float(scene_bin_size)
    support_density = _safe_ratio_float(
        float(seg_len),
        max(1.0, radial_bins * max(1.0, angular_spokes)),
        default=0.0,
    )
    return {
        "object_type": obj_type,
        "geom_type": geom_type,
        "material_proxy": material_proxy,
        "object_range_m": float(obj_range_m),
        "object_bearing_deg": float(obj_bearing_deg),
        "beam_offset_deg": float(beam_offset_deg),
        "abs_beam_offset_deg": abs(float(beam_offset_deg)),
        "object_heading_deg": obj_heading_deg,
        "aspect_to_spoke_deg": aspect_to_spoke_deg,
        "aspect_to_radar_deg": aspect_to_radar_deg,
        "rcs_m2": float(getattr(obj, "rcs_m2", raw.get("rcs_m2", 0.0)) or 0.0),
        "length_m": length_m,
        "width_m": width_m,
        "length_width_ratio": _safe_ratio_float(length_m, max(width_m, 0.1), default=0.0),
        "radial_bins": radial_bins,
        "angular_spokes": angular_spokes,
        "support_len_bins": float(seg_len),
        "support_len_m": support_len_m,
        "support_density": support_density,
        "projected_radial_extent_m": projected_radial_extent_m,
        "projected_cross_range_extent_m": projected_cross_range_extent_m,
        "projected_radial_bins": _safe_ratio_float(projected_radial_extent_m, scene_bin_size, default=0.0),
        "projected_cross_range_bins": _safe_ratio_float(projected_cross_range_extent_m, scene_bin_size, default=0.0),
        "segment_start_bin": float(start_bin),
        "segment_end_bin": float(max(start_bin, end_bin - 1)),
        "segment_center_bin": 0.5 * float(start_bin + max(start_bin, end_bin - 1)),
        "baseline_segment_peak_code": float(baseline_segment_peak_code),
        "baseline_segment_peak_db": float(baseline_segment_peak_db),
    }


def _predict_peak_delta_db(artifact: Optional[dict], feature_row: dict) -> float:
    if not artifact or not _HAS_PANDAS:
        return 0.0
    model = artifact.get("peak_model")
    if model is None:
        return 0.0
    limit = float(artifact.get("peak_correction_db_limit", 8.0))
    try:
        pred = float(model.predict(_pd.DataFrame([feature_row]))[0])
    except Exception:
        return 0.0
    return float(_clamp(pred, -limit, limit))


def _predict_shape_delta_db(artifact: Optional[dict], feature_rows: list[dict]) -> np.ndarray:
    if not artifact or not feature_rows or not _HAS_PANDAS:
        return np.zeros(len(feature_rows), dtype=np.float64)
    model = artifact.get("shape_model")
    if model is None:
        return np.zeros(len(feature_rows), dtype=np.float64)
    limit = float(artifact.get("shape_correction_db_limit", 6.0))
    try:
        pred = np.asarray(model.predict(_pd.DataFrame(feature_rows)), dtype=np.float64)
    except Exception:
        return np.zeros(len(feature_rows), dtype=np.float64)
    return np.clip(pred, -limit, limit)


def render_object_echo(obj, bearing_rad: float, priors: ObjectRenderPriors,
                       scene_bins: int, scene_bin_size: float, scene_dead_zone: int,
                       beamwidth_deg: float,
                       support_segments: Optional[Sequence[Any]] = None,
                       learned_artifact: Optional[dict] = None,
                       raw_obj: Optional[dict] = None,
                       beam_convention: str = "legacy_2p77") -> ObjectRenderResult:
    echoes = np.zeros(scene_bins, dtype=np.float64)

    obj_range = math.hypot(float(obj.x), float(obj.y))
    if obj_range < scene_dead_zone * scene_bin_size or float(obj.rcs_m2) <= 0.0 or obj_range <= 0.0:
        return ObjectRenderResult(echoes, "none", 0, 0, None)

    obj_bearing = math.atan2(float(obj.x), float(obj.y))
    d_angle = (float(bearing_rad) - obj_bearing + math.pi) % (2.0 * math.pi) - math.pi
    d_angle_deg = math.degrees(d_angle)
    half_beam = float(beamwidth_deg) / 2.0
    if abs(d_angle_deg) > half_beam + 1.0:
        return ObjectRenderResult(echoes, "none", 0, 0, None)

    if beam_convention == "half_power":
        # This value is passed through a 20*log10 amplitude-code conversion.
        # Take the square root of the power response so the half-power point
        # appears as a 3.01 dB amplitude-code drop.
        normalized_offset = d_angle_deg / max(half_beam, 1e-6)
        beam_gain = math.exp(-0.5 * math.log(2.0) * normalized_offset**2)
    elif beam_convention in ("legacy_2p77", "legacy_12db"):
        normalized_offset = d_angle_deg / max(half_beam, 1e-6)
        beam_gain = math.exp(-2.77 * normalized_offset**2)
    elif beam_convention == "none":
        beam_gain = 1.0
    else:
        raise ValueError(
            "beam_convention must be half_power, legacy_2p77, legacy_12db, or none"
        )
    intensity_db = (
        10.0 * math.log10(max(float(obj.rcs_m2), 1e-6) / max(float(priors.ref_rcs_m2), 1e-6))
        - 40.0 * math.log10(obj_range / max(float(priors.ref_range_m), 1e-6))
    )
    peak_code = float(priors.ref_code) * (10.0 ** (intensity_db / max(float(priors.compression), 1e-6)))
    peak_db = (
        display_db_from_code_value(peak_code)
        + shape_fraction_drop_db(beam_gain)
        + _object_peak_correction_db(getattr(obj, "obj_type", "medium_boat"), obj_range)
    )
    if code_value_from_display_db(peak_db) < 4.0:
        return ObjectRenderResult(echoes, "none", 0, 0, peak_db)

    normalized_support = _normalize_support_segments(support_segments, scene_bins, scene_dead_zone)
    if normalized_support:
        source = "scene_support"
        active_segments = normalized_support
    else:
        source = "geometry_support"
        active_segments = _build_geometry_support_segments(
            obj,
            priors,
            scene_bins,
            scene_bin_size,
            scene_dead_zone,
        )
    if not active_segments:
        return ObjectRenderResult(echoes, source, 0, 0, peak_db)

    spread_bins = float(priors.spread_bins) if priors.spread_bins is not None else _estimate_geometry_spread_bins(obj, scene_bin_size)
    floor_db = display_db_from_code_value(float(priors.support_floor_code))
    bearing_deg = math.degrees(float(bearing_rad)) % 360.0
    obj_bearing_deg = math.degrees(obj_bearing) % 360.0
    rendered_bins = 0

    for seg in active_segments:
        seg_len = seg.length_bins
        if seg_len <= 0:
            continue
        start_bin = int(seg.start_bin)
        end_bin = int(seg.end_bin)
        profile = _build_support_profile(
            getattr(obj, "obj_type", "medium_boat"),
            spread_bins,
            seg_len,
            reference_pattern=priors.reference_pattern,
            reference_profile_blend=priors.reference_profile_blend,
        )
        segment_bins: list[int] = []
        baseline_db_rows: list[float] = []
        feature_rows: list[dict] = []

        for offset, b in enumerate(range(start_bin, end_bin + 1)):
            if b < scene_dead_zone or b >= scene_bins:
                continue
            rel_pos = float(offset) / float(max(1, seg_len - 1)) if seg_len > 1 else 0.5
            baseline_db = max(
                floor_db,
                peak_db
                + shape_fraction_drop_db(float(profile[offset]))
                + _object_support_correction_db(getattr(obj, "obj_type", "medium_boat"), seg_len, rel_pos),
            )
            baseline_val = code_value_from_display_db(baseline_db)
            baseline_code = float(_round_clamped_code(baseline_val))
            baseline_db_quantized = display_db_from_code_value(baseline_code)
            segment_bins.append(b)
            baseline_db_rows.append(baseline_db_quantized)
            feature_rows.append({
                "baseline_db": baseline_db_quantized,
                "baseline_code": baseline_code,
                "range_m": float(b) * float(scene_bin_size),
                "log_range_m": math.log(max(float(b) * float(scene_bin_size), 1.0)),
                "bin_idx": float(b),
                "support_len_bins": float(seg_len),
                "support_len_m": float(seg_len) * float(scene_bin_size),
                "depth_from_entry_bins": float(offset),
                "depth_to_exit_bins": float(max(0, seg_len - 1 - offset)),
                "edge_distance_bins": float(min(offset, max(0, seg_len - 1 - offset))),
                "rel_pos": rel_pos,
                "phase": _object_support_phase_label(offset, seg_len),
                "normalized_center_distance": abs(rel_pos - 0.5) * 2.0,
                "is_short_support": 1.0 if seg_len <= 2 else 0.0,
                "is_edge_bin": 1.0 if offset == 0 or offset == seg_len - 1 else 0.0,
            })

        if not segment_bins:
            continue

        baseline_segment_peak_db = max(baseline_db_rows)
        baseline_segment_peak_code = max(
            float(_round_clamped_code(code_value_from_display_db(val_db)))
            for val_db in baseline_db_rows
        )
        segment_context = _build_segment_context(
            obj,
            raw_obj,
            bearing_deg,
            obj_bearing_deg,
            obj_range,
            d_angle_deg,
            seg_len,
            start_bin,
            end_bin + 1,
            scene_bin_size,
            baseline_segment_peak_code,
            baseline_segment_peak_db,
        )
        peak_delta_db = _predict_peak_delta_db(learned_artifact, segment_context)
        for row in feature_rows:
            row.update(segment_context)
            row["baseline_drop_from_segment_peak_db"] = baseline_segment_peak_db - float(row["baseline_db"])
        shape_delta_db = _predict_shape_delta_db(learned_artifact, feature_rows)

        for idx, b in enumerate(segment_bins):
            final_db = baseline_db_rows[idx] + peak_delta_db + float(shape_delta_db[idx])
            val = code_value_from_display_db(final_db)
            if val >= 6.0:
                echoes[b] = max(echoes[b], val)
                rendered_bins += 1

    return ObjectRenderResult(
        echoes=echoes,
        support_source=source,
        support_segments_used=len(active_segments),
        rendered_bins=rendered_bins,
        peak_db=peak_db,
    )
