"""Focused checks for the isolated CANOE adapter."""

import inspect
import unittest
from copy import deepcopy
from typing import Any

import numpy as np

import extract_scene
from ddmr.canoe_capture import (
    extract_support,
    mask_from_runs,
    render_support,
    support_runs,
)


class CanoeCaptureTests(unittest.TestCase):
    def setUp(self):
        self.profile: dict[str, Any] = {
            "grid": {"shape": [48, 50], "spacing_m": 0.5},
            "extraction": {
                "sigma_spokes": 1.0,
                "sigma_bins": 2.0,
                "support_threshold_display": 0.15,
                "seed_threshold_display": 0.20,
            },
            "texture": {
                "quantile_probabilities": [0, 0.5, 1],
                "range_centers_m": [0, 25],
                "background": {
                    "quantiles": [[0.0, 0.1, 0.2], [0.0, 0.12, 0.25]],
                    "angular_lag1_correlation": 0.2,
                    "radial_lag1_correlation": 0.3,
                },
                "foreground": {
                    "quantiles": [[0.2, 0.4, 0.6], [0.2, 0.45, 0.7]],
                    "angular_lag1_correlation": 0.4,
                    "radial_lag1_correlation": 0.4,
                },
            },
        }
        self.valid = np.ones((48, 50), dtype=bool)
        self.valid[:, :5] = False
        self.values = np.full((48, 50), 0.05)
        self.values[12:23, 16:37] = 0.5
        self.ticks = np.arange(48) * 170

    def test_original_detector_gate_restored_and_support_detected(self):
        original = extract_scene.COASTLINE_SATURATION_GATE
        support, polygons = extract_support(
            self.values, self.valid, self.ticks, self.profile
        )
        self.assertGreater(support.sum(), 30)
        self.assertTrue(polygons)
        self.assertFalse(np.any(support & ~self.valid))
        self.assertEqual(original, extract_scene.COASTLINE_SATURATION_GATE)
        self.assertTrue(
            np.array_equal(
                support, mask_from_runs(support_runs(support), support.shape)
            )
        )

    def test_renderer_reproducible_nonreplay_and_uses_support(self):
        support, _ = extract_support(self.values, self.valid, self.ticks, self.profile)
        a = render_support(support, self.valid, self.profile, 0)
        self.assertTrue(
            np.array_equal(a, render_support(support, self.valid, self.profile, 0))
        )
        self.assertFalse(
            np.array_equal(a, render_support(support, self.valid, self.profile, 1))
        )
        self.assertFalse(np.array_equal(a, self.values))
        self.assertNotIn("values", inspect.signature(render_support).parameters)
        shifted = np.roll(support, 10, axis=0) & self.valid
        self.assertFalse(
            np.array_equal(a, render_support(shifted, self.valid, self.profile, 0))
        )
        self.assertTrue(np.all(a[~self.valid] == 0))

    def test_bad_profile_and_mask_fail(self):
        bad = deepcopy(self.profile)
        bad["extraction"]["support_threshold_display"] = 0.0
        with self.assertRaises(ValueError):
            extract_support(self.values, self.valid, self.ticks, bad)
        mask = np.zeros_like(self.valid)
        mask[0, 0] = True
        with self.assertRaises(ValueError):
            render_support(mask, self.valid, self.profile, 0)


if __name__ == "__main__":
    unittest.main()
