"""Loaded-coastline attenuation models isolated from UI and scenario code."""
from __future__ import annotations

import os
from typing import Mapping

import numpy as np

from . import rust_bridge


ATTENUATION_CURRENT = "current"
ATTENUATION_ENTRY_R4_DEPTH = "entry_r4_depth"
ATTENUATION_MODELS = {ATTENUATION_CURRENT, ATTENUATION_ENTRY_R4_DEPTH}


def normalize_attenuation_model(value: str | None,
                                default: str = ATTENUATION_CURRENT) -> str:
    """Normalize CLI/UI/environment aliases to a supported model name."""
    model = (value or default or ATTENUATION_CURRENT).strip().lower().replace("-", "_")
    aliases = {
        "legacy": ATTENUATION_CURRENT,
        "existing": ATTENUATION_CURRENT,
        "default": ATTENUATION_CURRENT,
        "r4": ATTENUATION_ENTRY_R4_DEPTH,
        "r4_depth": ATTENUATION_ENTRY_R4_DEPTH,
        "entry_r4": ATTENUATION_ENTRY_R4_DEPTH,
        "ray": ATTENUATION_ENTRY_R4_DEPTH,
        "physical": ATTENUATION_ENTRY_R4_DEPTH,
        "ray_attenuation": ATTENUATION_ENTRY_R4_DEPTH,
    }
    model = aliases.get(model, model)
    return model if model in ATTENUATION_MODELS else default


def env_attenuation_model(default: str = ATTENUATION_CURRENT) -> str:
    """Read the requested loaded-coastline attenuation model from the shell."""
    return normalize_attenuation_model(
        os.environ.get("RADARLOC_ATTENUATION_MODEL"),
        default=default,
    )


def _float_config(
    config: Mapping[str, object] | None,
    key: str,
    env_key: str,
    default: float,
) -> float:
    if config and key in config:
        try:
            return float(config[key])
        except (TypeError, ValueError):
            return default
    raw = os.environ.get(env_key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def entry_r4_depth_attenuation(
    bin_ranges_m: np.ndarray,
    entry_range_m: float,
    *,
    config: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Return attenuation after the first shoreline entry along a ray.

    I_b = I_entry * (R_entry / R_b)^4 * exp(-depth_m / L_penetration)

    Nearby shoreline entries are allowed a longer penetration length than
    distant entries, but every bin still follows the same range-equation trend.
    """
    if len(bin_ranges_m) == 0:
        return np.zeros(0, dtype=np.float32)

    r_min = max(1.0, _float_config(config, "r_min_m", "RADARLOC_RAY_ATTEN_R_MIN_M", 20.0))
    l_ref = max(0.1, _float_config(config, "penetration_ref_m", "RADARLOC_RAY_ATTEN_LREF_M", 8.0))
    r_ref = max(1.0, _float_config(config, "penetration_ref_range_m", "RADARLOC_RAY_ATTEN_RREF_M", 200.0))
    power = max(0.0, _float_config(config, "penetration_power", "RADARLOC_RAY_ATTEN_POWER", 0.5))
    l_min = max(0.1, _float_config(config, "penetration_min_m", "RADARLOC_RAY_ATTEN_LMIN_M", 2.0))
    l_max = max(l_min, _float_config(config, "penetration_max_m", "RADARLOC_RAY_ATTEN_LMAX_M", 20.0))
    core_m = max(0.0, _float_config(config, "core_depth_m", "RADARLOC_RAY_ATTEN_CORE_M", 1.5))
    floor = max(0.0, min(1.0, _float_config(config, "floor", "RADARLOC_RAY_ATTEN_FLOOR", 0.0)))
    if rust_bridge.rust_available():
        try:
            result = rust_bridge.entry_r4_depth_attenuation(
                bin_ranges_m,
                entry_range_m,
                (r_min, l_ref, r_ref, power, l_min, l_max, core_m, floor),
            )
            return result.astype(np.float32)
        except Exception:
            pass

    entry_range = max(float(entry_range_m), r_min)
    ranges = np.maximum(np.asarray(bin_ranges_m, dtype=np.float32), entry_range)

    penetration_m = np.clip(l_ref * (r_ref / entry_range) ** power, l_min, l_max)
    depth_m = np.maximum(0.0, ranges - entry_range)
    interior_depth_m = np.maximum(0.0, depth_m - core_m)
    range_decay = np.power(entry_range / ranges, 4.0)
    depth_decay = np.exp(-interior_depth_m / penetration_m)
    attenuation = range_decay * depth_decay
    if floor > 0.0:
        attenuation = np.maximum(floor, attenuation)
    return np.clip(attenuation, 0.0, 1.0).astype(np.float32)
