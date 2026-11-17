"""Run the opt-in CANOE capture-conditioned reconstruction experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from ddmr.canoe_capture import (
    extract_support,
    load_spokes,
    mask_from_runs,
    read_profile,
    render_support,
    support_runs,
    write_acquisition_csv,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    results = args.output / "results"
    results.mkdir()
    inp = args.input
    source_path = inp / "canoe_display_norm_0_1_over_255.csv"
    valid_path = inp / "common_valid_mask.csv"
    spokes_path = inp / "spokes.csv"
    values = np.loadtxt(source_path, delimiter=",")
    valid = np.loadtxt(valid_path, delimiter=",").astype(bool)
    records = load_spokes(spokes_path)
    ticks = [int(record["renderer_tick_0_to_8191"]) for record in records]
    profile = read_profile(args.profile)
    mask, polygons = extract_support(values, valid, ticks, profile)
    np.savetxt(results / "support.csv", mask.astype(np.uint8), fmt="%d", delimiter=",")
    scene = {
        "method": "CANOE capture-conditioned extraction and training-profile intensity sampling",
        "shape": list(values.shape),
        "range_bin_spacing_m": float(profile["grid"]["spacing_m"]),
        "angle_ticks": ticks,
        "encoder_counts": [int(row["encoder_count_0_to_5599"]) for row in records],
        "timestamps_us": [int(row["source_row_time_us"]) for row in records],
        "valid_runs": support_runs(valid),
        "support_runs": support_runs(mask),
        "support_labels": "radar-derived high-intensity support; semantic coastline classification is not established",
        "support_cells": int(mask.sum()),
        "component_polygon_count": len(polygons),
        "polygons": [
            {
                k: v
                for k, v in p.items()
                if k in ("name", "points", "num_spokes", "size_bins", "parent_label")
            }
            for p in polygons
        ],
        "source_intensity_samples_stored": False,
    }
    (results / "scene_support.json").write_text(
        json.dumps(scene, indent=2, allow_nan=False) + "\n"
    )
    # Reconstruct solely from the saved, amplitude-free support and profile.
    support = mask_from_runs(scene["support_runs"], tuple(scene["shape"]))
    valid_render = mask_from_runs(scene["valid_runs"], tuple(scene["shape"]))
    assert np.array_equal(support, mask) and np.array_equal(valid_render, valid)
    for seed in (0, 1, 2):
        sample = render_support(support, valid_render, profile, seed)
        sample = np.rint(sample * 255).astype(np.uint8).astype(float) / 255
        name = (
            "reconstructed_display_norm_0_1.csv"
            if seed == 0
            else f"reconstructed_seed{seed}_display_norm_0_1.csv"
        )
        np.savetxt(results / name, sample, delimiter=",", fmt="%.9f")
        write_acquisition_csv(
            results / f"reconstructed_seed{seed}_acquisition.csv", sample, records
        )
    # Spoke-circular shift of support, fixed at 1/4 revolution; valid grid stays fixed.
    shuffled = np.roll(support, shift=len(support) // 4, axis=0) & valid_render
    control = render_support(shuffled, valid_render, profile, 0)
    control = np.rint(control * 255).astype(np.uint8).astype(float) / 255
    np.savetxt(
        results / "shifted_support_control_display_norm_0_1.csv",
        control,
        delimiter=",",
        fmt="%.9f",
    )
    np.savetxt(
        results / "shifted_support_control.csv",
        shuffled.astype(np.uint8),
        delimiter=",",
        fmt="%d",
    )
    assert np.array_equal(
        render_support(support, valid_render, profile, 0),
        render_support(support, valid_render, profile, 0),
    )
    manifest = {
        "selected_scan_png_sha256": digest(inp / "source_canoe_radar_png.png"),
        "source_grid_sha256": digest(source_path),
        "valid_mask_sha256": digest(valid_path),
        "spokes_sha256": digest(spokes_path),
        "training_profile_sha256": digest(args.profile),
        "support_scene_sha256": digest(results / "scene_support.json"),
        "support_cells": int(mask.sum()),
        "valid_cells": int(valid.sum()),
        "support_fraction_of_valid": float(mask.sum() / valid.sum()),
        "component_polygon_count": len(polygons),
        "seeds": [0, 1, 2],
        "control": "spoke-circular shift by 100 indices of extracted support, same valid mask and seed 0",
        "source_intensity_to_renderer": False,
        "renderer_intensity": "independent seeded spatial Gaussian fields with train-derived radial and angular lag structure, transformed to interpolated train-only range-conditional class distributions and quantized to nearest CANOE 8-bit display code; marginal Pearson correlations are approximate",
        "synthetic_acquisition_fields": [],
        "timestamp_span_us": int(records[-1]["source_row_time_us"])
        - int(records[0]["source_row_time_us"]),
        "estimated_full_sweep_period_us": float(
            int(records[-1]["source_row_time_us"])
            - int(records[0]["source_row_time_us"])
            + np.median(np.diff([int(row["source_row_time_us"]) for row in records]))
        ),
        "estimated_full_sweep_definition": "first-to-last timestamp span plus median inter-spoke timestamp interval; estimate, not a measured closing interval",
        "extractor_sha256": digest(ROOT / "extract_scene.py"),
        "adapter_sha256": digest(
            Path(sys.modules[extract_support.__module__].__file__ or "")
        ),
        "note": "The extractor computes support from the target scan. CANOE profile and amplitude distributions are trained on other scans. Output does not replay target per-cell values.",
    }
    (results / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
