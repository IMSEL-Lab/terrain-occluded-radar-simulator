"""CANOE adaptation of capture-conditioned extraction and resynthesis.

The original extractor supplies support geometry; this module supplies a
CANOE-specific continuous-display renderer. Render inputs exclude the source
intensity grid and contain only segmented support and acquisition geometry.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr

SCORE_SCALE = 1_000_000


def extract_support(values, valid, ticks, profile):
    """Call the original run/component detector on a smoothed display image."""
    import extract_scene

    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or values.shape != tuple(profile["grid"]["shape"]):
        raise ValueError("grid and valid mask must match the trained CANOE shape")
    sigma = (profile["extraction"]["sigma_spokes"], profile["extraction"]["sigma_bins"])
    weights = gaussian_filter(
        valid.astype(float), sigma, mode=("wrap", "constant"), cval=0.0
    )
    total = gaussian_filter(
        np.where(valid, values, 0.0), sigma, mode=("wrap", "constant"), cval=0.0
    )
    smooth = np.divide(total, weights, out=np.zeros_like(total), where=weights > 1e-12)
    scores = np.rint(np.where(valid, smooth, 0.0) * SCORE_SCALE).astype(np.int32)
    low = round(profile["extraction"]["support_threshold_display"] * SCORE_SCALE)
    high = round(profile["extraction"]["seed_threshold_display"] * SCORE_SCALE)
    if low <= 0 or high < low:
        raise ValueError("training thresholds must satisfy 0 < support <= seed")
    spokes = [
        {"angle_tick": int(tick), "echoes": score.tolist()}
        for tick, score in zip(ticks, scores, strict=True)
    ]
    mask, polygons, _, _, _ = extract_scene.detect_coastlines(
        spokes,
        values.shape[1],
        profile["grid"]["spacing_m"],
        land_threshold=low,
        min_run=6,
        peak_gate=high,
    )
    if np.any(mask & ~valid):
        raise AssertionError("invalid samples entered the extracted support")
    return mask, polygons


def _sigma_from_lag1(rho):
    # A Gaussian-filtered independent field has lag-one correlation
    # approximately exp(-1/(4 sigma^2)); nonpositive correlations use iid.
    rho = float(np.clip(rho, 0.0, 0.99))
    return 0.0 if rho == 0 else float(np.sqrt(-1.0 / (4.0 * np.log(rho))))


def _normal_field(rng, shape, angular_rho, radial_rho):
    field = rng.standard_normal(shape)
    angular_rho = float(angular_rho)
    if abs(angular_rho) >= 0.5:
        raise ValueError(
            "angular lag correlation must be strictly between -0.5 and 0.5"
        )
    if angular_rho:
        a = 2 * angular_rho / (1 + np.sqrt(1 - 4 * angular_rho * angular_rho))
        field = (field + a * np.roll(field, 1, axis=0)) / np.sqrt(1 + a * a)
    sigma = _sigma_from_lag1(radial_rho)
    if sigma > 0:
        field = gaussian_filter(field, (0, sigma), mode=("wrap", "reflect"))
    std = float(field.std())
    if not std:
        raise ValueError("degenerate random field")
    return (field - field.mean()) / std


def render_support(support, valid, profile, seed=0):
    """Generate display samples using support and training-only distributions."""
    support = np.asarray(support, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if support.shape != valid.shape or support.shape != tuple(profile["grid"]["shape"]):
        raise ValueError("support and valid grids must match the trained CANOE shape")
    if np.any(support & ~valid):
        raise ValueError("support cannot extend outside valid cells")
    texture = profile["texture"]
    probabilities = np.asarray(texture["quantile_probabilities"], dtype=float)
    centers = np.asarray(texture["range_centers_m"], dtype=float)
    if (
        probabilities[0] != 0
        or probabilities[-1] != 1
        or np.any(np.diff(probabilities) <= 0)
    ):
        raise ValueError("quantile probabilities must increase strictly from 0 to 1")
    if np.any(np.diff(centers) <= 0):
        raise ValueError("range centers must increase strictly")
    rng = np.random.default_rng(seed)
    result = np.zeros(support.shape, dtype=float)
    radii = (np.arange(support.shape[1]) + 0.5) * float(profile["grid"]["spacing_m"])
    for name, region in (
        ("background", ~support & valid),
        ("foreground", support & valid),
    ):
        params = texture[name]
        quantiles = np.asarray(params["quantiles"], dtype=float)
        if (
            quantiles.shape != (len(centers), len(probabilities))
            or not np.isfinite(quantiles).all()
        ):
            raise ValueError(f"invalid {name} range-conditional quantiles")
        field = _normal_field(
            rng,
            support.shape,
            params["angular_lag1_correlation"],
            params["radial_lag1_correlation"],
        )
        u = ndtr(field)
        for j, radius in enumerate(radii):
            selected = region[:, j]
            if not selected.any():
                continue
            # Linear interpolation in range removes artificial band boundaries.
            q = np.array(
                [
                    np.interp(radius, centers, quantiles[:, k])
                    for k in range(len(probabilities))
                ]
            )
            result[selected, j] = np.interp(u[selected, j], probabilities, q)
    return np.clip(result, 0.0, 1.0)


def support_runs(mask):
    result = []
    for row in np.asarray(mask, dtype=bool):
        changes = np.diff(np.r_[False, row, False].astype(np.int8))
        result.append(
            [
                [int(a), int(b)]
                for a, b in zip(
                    np.flatnonzero(changes == 1),
                    np.flatnonzero(changes == -1),
                    strict=True,
                )
            ]
        )
    return result


def mask_from_runs(runs, shape):
    result = np.zeros(shape, dtype=bool)
    if len(runs) != shape[0]:
        raise ValueError("incorrect spoke count in support")
    for i, segments in enumerate(runs):
        for start, end in segments:
            if not (0 <= start < end <= shape[1]):
                raise ValueError("invalid support segment")
            result[i, start:end] = True
    return result


def load_spokes(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    return records


def write_acquisition_csv(path, values, records):
    """Store measured encoder and timestamp with newly generated 8-bit codes."""
    codes = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp_us", "encoder_count", "renderer_tick"]
            + [f"bin_{i}" for i in range(codes.shape[1])]
        )
        for record, row in zip(records, codes, strict=True):
            writer.writerow(
                [
                    record["source_row_time_us"],
                    record["encoder_count_0_to_5599"],
                    record["renderer_tick_0_to_8191"],
                    *row.tolist(),
                ]
            )


def read_profile(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
