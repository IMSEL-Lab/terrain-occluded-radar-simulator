"""Compatibility wrapper for the split radar core object echo model."""
from radar_core.object_echo import (
    MAX_FURUNO_CODE as OBJECT_CREATION_MAX_CODE,
    SPREAD_CAPS_BY_CLASS as OBJECT_CREATION_SPREAD_CAPS,
    furuno_code_to_linear_power as object_creation_code_to_linear,
    object_class as object_creation_type,
    peak_furuno_code as object_creation_peak_code,
    radial_spread_bins as object_creation_spread_bins,
)

