"""Fit a CANOE display-code sensor profile on the existing 30 training scans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label

CALIBRATION_SEQUENCE = "canoe-2025-08-28-14-13"
TARGET = "canoe-2025-08-28-14-50/radar/1756392754126501.png"
SIGMAS = (1.0, 2.0)
SPOKES, BINS = 400, 300
BIN_M = 0.5
MIN_VALID_M, MAX_M, OFFSET_M = 2.5, 150.0, -0.407910
MIN_RADIAL_RUN, MIN_COMPONENT_CELLS = 6, 30
SCORE_THRESHOLD = 0.24491978609625667
PROBABILITIES = np.r_[0.0, 0.01, np.arange(1, 20) / 20, 0.99, 0.995, 0.999, 1.0]
VERIFIED_SHA256 = {
    "canoe_selected_scans.json": "a3debf098153b605690ed1f5db29d60d47b728dc002361b38dddfe5e58636445",
    "canoe_source_manifest.json": "443f83491158ef116f7dd4928c3541058c7e8b96374a4564f8604e53568f0e27",
    "radar_config.yaml": "ba1ffad53a45d486fbc18da9b4a9ed1fbc535168fab32662b4bda160ff0ed661",
    "radar_poses.csv": "b88ee71e5d7e46a2a815430d9b7f48ba8b6320ebc9e8920097556ed51abe4b7e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_hash(path: Path, expected: str) -> str:
    digest = sha256(path)
    if digest != expected:
        raise ValueError(f"sha256 mismatch: {path}")
    return digest


def masked_smooth(echo: np.ndarray, valid: np.ndarray) -> np.ndarray:
    numerator = gaussian_filter(
        np.where(valid, echo, 0), SIGMAS, mode=("wrap", "constant")
    )
    denominator = gaussian_filter(
        valid.astype(float), SIGMAS, mode=("wrap", "constant")
    )
    return np.divide(
        numerator, denominator, where=denominator > 0, out=np.zeros_like(echo)
    )


def foreground_mask(
    smooth: np.ndarray, valid: np.ndarray, support: float, seed: float
) -> np.ndarray:
    micro_units = np.rint(smooth * 1_000_000).astype(np.int32)
    support_code, seed_code = np.rint(np.array([support, seed]) * 1_000_000).astype(int)
    candidate = valid & (micro_units >= support_code)
    mask = np.zeros_like(valid)
    for row in np.flatnonzero(np.any(candidate, axis=1)):
        changes = np.diff(candidate[row], prepend=False, append=False)
        starts, ends = np.flatnonzero(changes).reshape(-1, 2).T
        for start, end in zip(starts, ends, strict=True):
            if (
                end - start >= MIN_RADIAL_RUN
                and np.max(micro_units[row, start:end]) >= seed_code
            ):
                mask[row, start:end] = True
    labels, count = label(mask, structure=np.ones((3, 3), dtype=bool))
    roots = np.arange(count + 1)

    def find(component: int) -> int:
        while roots[component] != component:
            roots[component] = roots[roots[component]]
            component = int(roots[component])
        return component

    for rad in np.flatnonzero(mask[0]):
        for last_row_label in labels[-1, max(0, rad - 1) : rad + 2]:
            if last_row_label > 0:
                roots[find(int(last_row_label))] = find(int(labels[0, rad]))
    lookup = np.array([find(i) for i in range(count + 1)])
    merged = lookup[labels]
    counts = np.bincount(merged.ravel(), minlength=count + 1)
    return mask & (counts[merged] >= MIN_COMPONENT_CELLS)


def class_profile(
    echoes: np.ndarray, mask: np.ndarray, centers: np.ndarray
) -> dict[str, object]:
    counts = np.array([mask[..., i : i + 5].sum() for i in range(5, BINS, 5)])
    observed = counts > 0
    if not np.any(observed):
        raise ValueError("train split lacks one labelled class")
    tables = np.full((centers.size, PROBABILITIES.size), np.nan)
    for group in np.flatnonzero(observed):
        start = 5 + 5 * group
        sl = slice(start, start + 5)
        tables[group] = np.quantile(echoes[..., sl][mask[..., sl]], PROBABILITIES)
    for i in range(PROBABILITIES.size):
        tables[:, i] = np.interp(centers, centers[observed], tables[observed, i])
    median_index = int(np.flatnonzero(PROBABILITIES == 0.5)[0])
    radial_centers = BIN_M * (np.arange(BINS) + 0.5)
    median = np.interp(radial_centers, centers, tables[:, median_index])
    residual = echoes - median
    radial_pairs = mask[..., :-1] & mask[..., 1:]
    angular_pairs = mask & np.roll(mask, -1, axis=1)
    radial_a = residual[..., :-1][radial_pairs]
    radial_b = residual[..., 1:][radial_pairs]
    angular_a = residual[angular_pairs]
    angular_b = np.roll(residual, -1, axis=1)[angular_pairs]
    return {
        "quantiles": tables.tolist(),
        "samples_per_bin": counts.tolist(),
        "radial_lag1_correlation": float(np.corrcoef(radial_a, radial_b)[0, 1]),
        "angular_lag1_correlation": float(np.corrcoef(angular_a, angular_b)[0, 1]),
        "radial_pair_count": int(radial_a.size),
        "angular_pair_count": int(angular_a.size),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)

    from ddmr import canoe
    from ddmr.canoe import (
        load_pose_table,
        load_radar_png,
        project_scan_with_native_range,
    )

    selection_file = arguments.manifests / "canoe_selected_scans.json"
    source_file = arguments.manifests / "canoe_source_manifest.json"
    data = arguments.dataset / "train" / CALIBRATION_SEQUENCE
    pose_file = data / "novatel/radar_poses.csv"
    radar_config_file = data / "calib/radar_config.yaml"
    based_on = {
        path.name: verified_hash(path, VERIFIED_SHA256[path.name])
        for path in [selection_file, source_file, pose_file, radar_config_file]
    }
    frozen = json.loads(selection_file.read_text())
    training = frozen["sequences"]["train"]
    if training["sequence"] != CALIBRATION_SEQUENCE:
        raise ValueError("unexpected training traversal")
    scans = [
        item["key"]
        for item in training["scans"]
        if item["subset"] == "train_calibration"
    ]
    if len(set(scans)) != 30 or any(
        not key.startswith(CALIBRATION_SEQUENCE + "/radar/") for key in scans
    ):
        raise ValueError("wrong frozen list")
    if TARGET in scans or not any(
        item["key"] == TARGET for item in frozen["sequences"]["validation"]["scans"]
    ):
        raise ValueError("target split is inconsistent")
    sources = {
        entry["key"]: entry["sha256"]
        for entry in json.loads(source_file.read_text())["selected_scans"]
    }
    poses = load_pose_table(pose_file)
    scan_echoes, scan_valid, smoothed, records = [], [], [], []
    for key in scans:
        path = data / "radar" / Path(key).name
        digest = verified_hash(path, sources[key])
        scan = load_radar_png(path)
        projected = project_scan_with_native_range(
            scan,
            poses,
            MAX_M,
            MAX_M,
            range_offset_m=OFFSET_M,
            min_range_m=MIN_VALID_M,
            radial_step_m=BIN_M,
        )
        if projected.echo.shape != (SPOKES, BINS):
            raise ValueError(f"unexpected shape for {key}: {projected.echo.shape}")
        # Check the shared analysis crop against actual decoded support.
        if np.any(projected.valid[:, :5]) or not np.any(projected.valid[:, 5]):
            raise ValueError(f"unexpected valid near range for {key}")
        scan_echoes.append(projected.echo)
        scan_valid.append(projected.valid)
        smoothed.append(masked_smooth(projected.echo, projected.valid))
        records.append(
            {
                "source_key": key,
                "sha256": digest,
                "normalized_grid_sha256_nan_to_zero_f64le": hashlib.sha256(
                    np.where(projected.valid, projected.echo, 0).astype("<f8").tobytes()
                ).hexdigest(),
                "valid_mask_sha256_u8": hashlib.sha256(
                    projected.valid.astype("uint8").tobytes()
                ).hexdigest(),
                "source_echo_shape": list(scan.echo.shape),
                "valid_azimuth_rows": int(scan.valid_azimuth.sum()),
                "valid_analysis_cells": int(projected.valid.sum()),
                "first_source_row_time_us": int(scan.row_time_us[0]),
                "last_source_row_time_us": int(scan.row_time_us[-1]),
                "observed_spoke_span_s": int(np.ptp(scan.row_time_us)) / 1e6,
                "code_quantiles_display": np.quantile(
                    projected.echo[projected.valid], [0.5, 0.9, 0.95, 0.99, 1.0]
                ).tolist(),
            }
        )
    echoes, valid, smooth = map(np.stack, [scan_echoes, scan_valid, smoothed])
    support, seed = np.quantile(smooth[valid], [0.90, 0.95])
    foreground = np.stack(
        [
            foreground_mask(smooth_scan, valid_scan, support, seed)
            for smooth_scan, valid_scan in zip(smooth, valid, strict=True)
        ]
    )
    for record, mask in zip(records, foreground, strict=True):
        record["foreground_support_cells"] = int(mask.sum())
    range_centers = np.arange(MIN_VALID_M + 1.25, MAX_M, 2.5)
    raw_q95 = float(np.quantile(echoes[valid], 0.95))
    if abs(raw_q95 - SCORE_THRESHOLD) > 2e-12:
        raise ValueError(f"original pooled train q95 changed: {raw_q95}")
    profile = {
        "schema_version": 1,
        "grid": {
            "shape": [SPOKES, BINS],
            "spacing_m": BIN_M,
            "min_valid_range_m": MIN_VALID_M,
            "max_range_m": MAX_M,
            "encoder_counts_per_revolution": 5600,
        },
        "extraction": {
            "sigma_spokes": SIGMAS[0],
            "sigma_bins": SIGMAS[1],
            "support_threshold_display": float(support),
            "seed_threshold_display": float(seed),
            "min_radial_run_bins": MIN_RADIAL_RUN,
            "min_component_cells": MIN_COMPONENT_CELLS,
            "score_foreground_threshold_display": SCORE_THRESHOLD,
            "foreground_definition": "Smoothed training q90 support and q95 seed, at least one seed in each consecutive radial run of at least six bins, wrapped eight-neighbor components of at least 30 cells. Both samples and thresholds round to integer micro-units for extraction.",
        },
        "texture": {
            "quantile_probabilities": PROBABILITIES.tolist(),
            "range_centers_m": range_centers.tolist(),
            "background": class_profile(echoes, valid & ~foreground, range_centers),
            "foreground": class_profile(echoes, valid & foreground, range_centers),
            "correlation_definition": "Pearson correlation of display amplitudes centered by interpolated range-and-class median, using only same-class valid cell pairs. Adjacent spokes wrap.",
            "distribution_definition": "Measured display /255 quantiles per 2.5 m range group and class. For any empty class-range group, interpolate quantile values across observed groups and clamp at the ends.",
        },
        "provenance": {
            "source_attribution": "CANOE dataset, CC BY 4.0. https://registry.opendata.aws/canoe/",
            "excluded_validation_target": TARGET,
            "source_sequence": CALIBRATION_SEQUENCE,
            "source_sequence_range_m": MAX_M,
            "source_sequence_range_offset_m": OFFSET_M,
            "frozen_subset": "train_calibration",
            "input_hashes": based_on,
            "decoder_sha256": sha256(Path(canoe.__file__)),
            "calibration_script_sha256": sha256(Path(__file__)),
            "pooled_valid_cells": int(valid.sum()),
            "pooled_foreground_cells": int(foreground.sum()),
            "pooled_raw_q95_recomputed": raw_q95,
            "training_spoke_span_s_median": float(
                np.median([item["observed_spoke_span_s"] for item in records])
            ),
            "scans": records,
        },
    }
    arguments.output_dir.mkdir(exist_ok=True, parents=True)
    output_file = arguments.output_dir / "sensor_profile.json"
    with output_file.open("x") as output:
        json.dump(profile, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Profile {output_file} sha256={sha256(output_file)}")
    print(json.dumps(profile["extraction"], indent=2))
    print(
        f"Training scans={len(records)}, valid cells={valid.sum()}, foreground cells={foreground.sum()}"
    )


if __name__ == "__main__":
    main()
