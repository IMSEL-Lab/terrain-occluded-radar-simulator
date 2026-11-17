"""Verify the frozen profile against the original detector and retain its fit inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.calibrate_canoe import (
    BIN_M,
    BINS,
    CALIBRATION_SEQUENCE,
    MAX_M,
    MIN_RADIAL_RUN,
    MIN_VALID_M,
    OFFSET_M,
    TARGET,
    foreground_mask,
    masked_smooth,
    sha256,
    verified_hash,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    root = arguments.output_dir
    profile_path = root / "sensor_profile.json"
    profile = json.loads(profile_path.read_text())
    provenance = profile["provenance"]
    records = provenance["scans"]
    keys = [r["source_key"] for r in records]
    if len(set(keys)) != 30 or TARGET in keys:
        raise ValueError(
            "the calibration scan list must contain 30 unique training scans"
        )
    extraction = profile["extraction"]
    support = extraction["support_threshold_display"]
    seed = extraction["seed_threshold_display"]

    import extract_scene
    from ddmr.canoe import (
        load_pose_table,
        load_radar_png,
        project_scan_with_native_range,
    )

    data = arguments.dataset / "train" / CALIBRATION_SEQUENCE
    pose_path = data / "novatel/radar_poses.csv"
    verified_hash(pose_path, provenance["input_hashes"]["radar_poses.csv"])
    poses = load_pose_table(pose_path)
    inputs, valid_masks = [], []
    verified = []
    for record in records:
        key = record["source_key"]
        path = data / "radar" / Path(key).name
        verified_hash(path, record["sha256"])
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
        smooth = masked_smooth(projected.echo, projected.valid)
        rounded = np.rint(np.where(projected.valid, smooth, 0) * 1_000_000).astype(
            np.int32
        )
        ticks = np.rint(scan.encoder * (8192 / 5600)).astype(int) % 8192
        spokes = [
            {"angle_tick": int(tick), "echoes": row.tolist()}
            for tick, row in zip(ticks, rounded, strict=True)
        ]
        legacy_mask, *_ = extract_scene.detect_coastlines(
            spokes,
            BINS,
            BIN_M,
            land_threshold=round(support * 1_000_000),
            min_run=MIN_RADIAL_RUN,
            peak_gate=round(seed * 1_000_000),
        )
        fitted_mask = foreground_mask(smooth, projected.valid, support, seed)
        np.testing.assert_array_equal(fitted_mask, legacy_mask)
        if int(legacy_mask.sum()) != record["foreground_support_cells"]:
            raise ValueError(f"foreground support count drifted for {key}")
        verified.append({"source_key": key, "foreground_cells": int(legacy_mask.sum())})
        inputs.append(np.where(projected.valid, projected.echo, 0).astype("<f8"))
        valid_masks.append(projected.valid)

    arrays = root / "training_grids.npz"
    np.savez_compressed(
        arrays,
        normalized_display=np.stack(inputs),
        valid=np.stack(valid_masks),
        source_keys=np.array(keys),
    )
    result = {
        "sensor_profile_sha256": sha256(profile_path),
        "original_extractor_sha256": sha256(Path(extract_scene.__file__)),
        "training_grids_sha256": sha256(arrays),
        "verification": "The fitted class mask matches the original wrapped detector cell for cell with the CANOE thresholds and smoothing applied.",
        "comparison_scan_pixels_loaded": False,
        "all_training_masks_match": True,
        "verification_records": verified,
    }
    with (root / "calibration_verification.json").open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(verified)} original-detector masks and training scan hashes")


if __name__ == "__main__":
    main()
