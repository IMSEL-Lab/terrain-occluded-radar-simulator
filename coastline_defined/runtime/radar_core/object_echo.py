"""Compact object echo model used by the radar sweep generator.

This is the Object Creation-style object painter, extracted into the core so
future C++/Rust ports have one small equation-focused file to reproduce.
"""
from __future__ import annotations

import math
from typing import Any

from . import rust_bridge


REF_RANGE_M = 50.0
REF_RCS_M2 = 350.0
REF_CODE = 190.0
COMPRESSION_DB = 55.0
MAX_FURUNO_CODE = 252.0

SPREAD_CAPS_BY_CLASS = {
    "buoy": 3,
    "small_boat": 6,
    "medium_boat": 9,
    "large_boat": 13,
    "barge": 15,
}

# The compact target painter is intentionally bounded so near-range buoys do
# not become island-sized returns. These small class boosts restore enough PPI
# contrast for placed targets to remain visible against bright coastline fill.
VISIBILITY_BOOST_BY_CLASS = {
    "buoy": 1.80,
    "small_boat": 1.70,
    "medium_boat": 1.60,
    "large_boat": 1.35,
    "barge": 1.25,
}
VISIBILITY_RANGE_LIFT_MAX = 2.20
VISIBILITY_RANGE_LIFT_SCALE_M = 650.0


def object_class(target: Any) -> str:
    """Map simulator vessels to the compact Object Creation object classes."""
    explicit = getattr(target, "object_creation_class", None)
    if explicit in SPREAD_CAPS_BY_CLASS:
        return explicit
    vessel_type = getattr(target, "vessel_type", None)
    value = getattr(vessel_type, "value", str(vessel_type or "")).lower()
    length = float(getattr(target, "length", 0.0) or 0.0)

    if "buoy" in value:
        return "buoy"
    if any(token in value for token in ("sailing", "pilot", "tug")):
        return "small_boat"
    if "fishing" in value:
        return "small_boat" if length <= 18.0 else "medium_boat"
    if any(token in value for token in ("cargo", "tanker", "passenger")):
        return "large_boat"
    if length <= 4.0:
        return "buoy"
    if length <= 18.0:
        return "small_boat"
    if length <= 35.0:
        return "medium_boat"
    return "large_boat"


def visibility_boost(target: Any, range_m: float) -> float:
    """Return bounded object-only display lift for PPI readability.

    The base peak still follows the R^-4 radar trend. This lift prevents
    mid-range small craft from collapsing into the same Furuno codes as the
    speckle floor while keeping near-range objects bounded below saturation.
    """
    target_class = object_class(target)
    class_boost = VISIBILITY_BOOST_BY_CLASS.get(target_class, 1.0)
    range_m = max(0.0, float(range_m))
    range_lift = 1.0 + (VISIBILITY_RANGE_LIFT_MAX - 1.0) * (
        1.0 - math.exp(-range_m / VISIBILITY_RANGE_LIFT_SCALE_M)
    )
    return class_boost * range_lift


def peak_furuno_code(target: Any, range_m: float, beam_gain: float) -> float:
    """Compute a bounded Furuno-like object peak code.

    I_code = REF_CODE * 10^(((10 log10(RCS/REF_RCS)
             - 40 log10(R/REF_RANGE)) / COMPRESSION_DB)) * beam_gain

    The bounded code domain prevents near-range targets from exploding into
    island-sized returns while preserving the radar-equation range trend.
    """
    if range_m <= 0.0 or beam_gain <= 0.0:
        return 0.0

    rcs_m2 = max(0.01, float(getattr(target, "rcs", 0.01) or 0.01))
    length_m = max(0.01, float(getattr(target, "length", 0.01) or 0.01))
    width_m = max(0.01, float(getattr(target, "beam", 0.01) or 0.01))
    heading = getattr(target, "heading_deg", None)
    heading_deg = float(getattr(target, "course", 0.0) if heading is None else heading)
    boost = visibility_boost(target, range_m)

    if rust_bridge.rust_available():
        try:
            base_code = rust_bridge.peak_furuno_code(
                rcs_m2, length_m, width_m, heading_deg, range_m, beam_gain)
            return max(0.0, min(MAX_FURUNO_CODE, base_code * boost))
        except Exception:
            pass
    range_m = max(1.0, float(range_m))
    intensity_db = (
        10.0 * math.log10(rcs_m2 / REF_RCS_M2)
        - 40.0 * math.log10(range_m / REF_RANGE_M)
    )
    code = REF_CODE * (10.0 ** (intensity_db / COMPRESSION_DB))
    return max(0.0, min(MAX_FURUNO_CODE, code * float(beam_gain) * boost))


def radial_spread_bins(target: Any, bearing_rad: float, bin_size_m: float) -> int:
    """Return compact radial target width in bins."""
    bin_size_m = max(0.01, float(bin_size_m))
    length_m = max(0.01, float(getattr(target, "length", bin_size_m) or bin_size_m))
    width_m = max(0.01, float(getattr(target, "beam", bin_size_m) or bin_size_m))
    heading = getattr(target, "heading_deg", None)
    heading_deg = float(getattr(target, "course", 0.0) if heading is None else heading)

    relative_angle = bearing_rad - math.radians(heading_deg)
    radial_extent_m = (
        abs(length_m * math.cos(relative_angle))
        + abs(width_m * math.sin(relative_angle))
    )
    radial_extent_m = max(radial_extent_m, bin_size_m)

    spread_cap = SPREAD_CAPS_BY_CLASS.get(object_class(target), 5)
    spread = math.ceil(radial_extent_m / bin_size_m / 5.0)
    return max(3, min(spread_cap, spread))


def furuno_code_to_linear_power(code: float, dynamic_range_db: float) -> float:
    """Convert a target display code back to pre-log linear power."""
    normalized = max(0.0, min(1.0, float(code) / MAX_FURUNO_CODE))
    dynamic_range_db = max(1.0, float(dynamic_range_db))
    db_value = normalized * dynamic_range_db - dynamic_range_db
    return 10.0 ** (db_value / 10.0)


def paint_object_echo_profile(center_bin: int, spread_bins: int, peak_code: float,
                              dynamic_range_db: float, num_bins: int):
    """Paint a deterministic object echo profile into a linear-power sweep.

    The live simulator still adds Python-side scintillation/jitter for visual
    texture. This helper exists so bulk deterministic echo creation can move
    into Rust without changing the quality-sensitive live path prematurely.
    """
    if rust_bridge.rust_available():
        try:
            return rust_bridge.paint_object_echo(
                center_bin, spread_bins, peak_code, dynamic_range_db, num_bins)
        except Exception:
            pass
    import numpy as np

    sweep = np.zeros(max(0, int(num_bins)), dtype=np.float64)
    if not (0 <= center_bin < len(sweep)) or peak_code < 4.0:
        return sweep
    sigma = max(spread_bins * 0.5, 0.5)
    start_bin = max(0, center_bin - spread_bins * 2)
    end_bin = min(len(sweep), center_bin + spread_bins * 2 + 1)
    for bin_idx in range(start_bin, end_bin):
        dist_bins = abs(bin_idx - center_bin)
        profile = math.exp(-0.5 * (dist_bins / sigma) ** 2)
        code = peak_code * profile
        if code >= 6.0:
            sweep[bin_idx] = max(
                sweep[bin_idx],
                furuno_code_to_linear_power(code, dynamic_range_db),
            )
    return sweep
