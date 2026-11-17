"""Shared sweep signal chain: STC, log compression, and quantization."""
from __future__ import annotations

from typing import List, Union

import numpy as np

from . import rust_bridge


ArrayLike = Union[List[float], np.ndarray]


def apply_stc(sweep_linear: ArrayLike, stc_curve: ArrayLike,
              return_numpy: bool = False) -> ArrayLike:
    """Apply sensitivity-time-control attenuation to linear-power bins."""
    if rust_bridge.rust_available():
        try:
            result = rust_bridge.apply_stc(sweep_linear, stc_curve)
            return result if return_numpy else result.tolist()
        except Exception:
            pass
    sweep = np.asarray(sweep_linear)
    stc = np.asarray(stc_curve)
    attenuation = np.ones(len(sweep))
    attenuation[:min(len(sweep), len(stc))] = 1.0 - stc[:min(len(sweep), len(stc))]
    result = sweep * attenuation
    return result if return_numpy else result.tolist()


def log_compress(sweep_linear: ArrayLike, dynamic_range_db: float = 80.0,
                 return_numpy: bool = False) -> ArrayLike:
    """Map linear power to normalized display intensity with log compression."""
    if rust_bridge.rust_available():
        try:
            result = rust_bridge.log_compress(sweep_linear, dynamic_range_db)
            return result if return_numpy else result.tolist()
        except Exception:
            pass
    sweep = np.asarray(sweep_linear)
    floor = 10.0 ** (-float(dynamic_range_db) / 10.0)
    db = 10.0 * np.log10(np.maximum(sweep, floor))
    normalized = (db + float(dynamic_range_db)) / float(dynamic_range_db)
    result = np.clip(normalized, 0.0, 1.0)
    return result if return_numpy else result.tolist()


def quantize(normalized_sweep: ArrayLike, bits: int = 8,
             return_numpy: bool = False) -> ArrayLike:
    """Quantize normalized display values to an N-bit display grid."""
    if rust_bridge.rust_available():
        try:
            result = rust_bridge.quantize(normalized_sweep, bits)
            return result if return_numpy else result.tolist()
        except Exception:
            pass
    sweep = np.clip(np.asarray(normalized_sweep), 0.0, 1.0)
    levels = (1 << int(bits)) - 1
    result = np.round(sweep * levels) / levels
    return result if return_numpy else result.tolist()


def process_signal_chain(sweep_linear: ArrayLike, stc_curve: ArrayLike,
                         dynamic_range_db: float = 80.0, bits: int = 8,
                         return_numpy: bool = False) -> ArrayLike:
    """Apply STC, log compression, and quantization in one call."""
    if rust_bridge.rust_available():
        try:
            result = rust_bridge.process_signal_chain(
                sweep_linear, stc_curve, dynamic_range_db, bits)
            return result if return_numpy else result.tolist()
        except Exception:
            pass
    stc = apply_stc(sweep_linear, stc_curve, return_numpy=True)
    compressed = log_compress(stc, dynamic_range_db, return_numpy=True)
    return quantize(compressed, bits, return_numpy=return_numpy)
