"""Compatibility wrapper for the split radar-core signal chain."""
from typing import List, Union

import numpy as np

from radar_core.signal_chain import (
    apply_stc as _core_apply_stc,
    log_compress,
    process_signal_chain as _core_process_signal_chain,
    quantize,
)


ArrayLike = Union[List[float], np.ndarray]


def apply_stc(sweep_db: ArrayLike, stc_curve: ArrayLike,
              max_range_m: float, return_numpy: bool = False) -> ArrayLike:
    """Apply STC; max_range_m is kept for the old radar_sim API."""
    return _core_apply_stc(sweep_db, stc_curve, return_numpy=return_numpy)


def process_signal_chain(sweep_db: ArrayLike, stc_curve: ArrayLike,
                         max_range_m: float, dynamic_range_db: float = 80.0,
                         bits: int = 8, return_numpy: bool = False) -> ArrayLike:
    """Apply STC, log compression, and quantization in one core call."""
    return _core_process_signal_chain(
        sweep_db, stc_curve, dynamic_range_db, bits, return_numpy=return_numpy)
