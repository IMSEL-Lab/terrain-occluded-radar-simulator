"""Check the location-file response options documented in the paper."""

from __future__ import annotations

import numpy as np

from coastline_defined import run_coastline_defined as location


def test_entry_range_and_depth_model_matches_equation():
    from radar_core.shoreline_attenuation import entry_r4_depth_attenuation

    ranges = np.asarray([100.0, 101.5, 109.5, 117.5], dtype=np.float32)
    config = {
        "r_min_m": 20.0,
        "penetration_ref_m": 8.0,
        "penetration_ref_range_m": 100.0,
        "penetration_power": 0.0,
        "penetration_min_m": 2.0,
        "penetration_max_m": 20.0,
        "core_depth_m": 1.5,
        "floor": 0.0,
    }
    expected = (100.0 / ranges) ** 4 * np.exp(-np.maximum(0.0, ranges - 101.5) / 8.0)
    actual = entry_r4_depth_attenuation(ranges, 100.0, config=config)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_secondary_returns_follow_cumulative_scale_and_cap(monkeypatch):
    monkeypatch.setenv("RADARLOC_SWEEP_OCCLUDE_AFTER_FIRST_LAND", "1")
    monkeypatch.setenv("RADARLOC_SWEEP_OCCLUSION_THRESHOLD_CODE", "20")
    monkeypatch.setenv("RADARLOC_SWEEP_OCCLUSION_GAP_BINS", "6")
    monkeypatch.setenv("RADARLOC_SWEEP_OCCLUSION_SHOULDER_BINS", "2")
    monkeypatch.setenv("RADARLOC_SWEEP_SECONDARY_RETURN_SCALE", "0.05")
    monkeypatch.setenv("RADARLOC_SWEEP_SECONDARY_RETURN_MAX_CODE", "28")
    source = np.zeros(40, dtype=np.float32)
    source[2:4], source[16:18], source[30:32] = 0.5, 0.7, 0.9
    output = location.Simulation()._apply_combined_coastline_occlusion(source)
    np.testing.assert_allclose(output[2:4], 0.5)
    np.testing.assert_allclose(output[16:18], min(0.7 * 0.05, 28 / 252))
    np.testing.assert_allclose(output[30:32], min(0.9 * 0.05**2, 28 / 252 * 0.05))


def test_location_controls_accept_manual_vessels_and_attenuation():
    args = location.build_parser().parse_args(
        [
            "--radarloc",
            "examples/synthetic-location.radarloc",
            "--attenuation-model",
            "entry_r4_depth",
            "--manual-object",
            "small_boat,-40,50,30,7,3,20,2,0",
            "--receiver-noise",
            "--receiver-noise-level",
            "0.002",
        ]
    )
    assert args.attenuation_model == "entry_r4_depth"
    assert args.receiver_noise and args.manual_object
