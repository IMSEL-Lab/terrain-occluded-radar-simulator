"""Tests for complete-mask scoring and canonical rotation preservation."""

import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from scripts.run_furuno_conditioned_diagnostic import (
    _canonicalize_rotation,
    _guarded_simulator_command,
    _write_capture_averages,
    _write_metrics_table,
    run,
    score_pair,
)


def test_score_keeps_all_source_observed_angles() -> None:
    source = np.asarray([[100.0, 0.0], [20.0, 40.0], [50.0, 80.0]])
    rendered = np.asarray([[0.0, 0.0], [22.0, 36.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="missing 1 source-observed angle"):
        score_pair(
            source,
            rendered,
            np.asarray([True, True, False]),
            np.asarray([False, True, True]),
        )


def test_score_rejects_nonfinite_prediction_at_observed_angle() -> None:
    source = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    rendered = np.asarray([[1.0, np.nan], [3.0, 4.0]])
    with pytest.raises(ValueError, match="render contains nonfinite"):
        score_pair(
            source,
            rendered,
            np.asarray([True, True]),
            np.asarray([True, True]),
        )


def test_score_computes_zero_baseline_and_empty_foreground() -> None:
    source = np.zeros((2, 3), dtype=np.float32)
    rendered = np.zeros_like(source)
    result = score_pair(
        source,
        rendered,
        np.asarray([True, False]),
        np.asarray([True, True]),
    )
    assert result["common_observed_angles"] == 1
    assert result["paired_valid_bins"] == 3
    assert result["normalized_mae"] == 0.0
    assert result["normalized_rmse"] == 0.0
    assert result["source_positive_normalized_mae"] is None


def test_full_grid_rmse_and_capture_averages(tmp_path) -> None:
    source = np.array([[0.0, 0.0], [100.0, 100.0], [252.0, 252.0]])
    rendered = np.array([[0.0, 6.0], [108.0, 100.0], [0.0, 0.0]])
    scores = score_pair(
        source, rendered, np.array([True, True, False]), np.ones(3, dtype=bool)
    )
    assert scores["normalized_mae"] == pytest.approx(3.5 / 252)
    assert scores["normalized_rmse"] == pytest.approx(5.0 / 252)
    assert scores["source_positive_normalized_mae"] == pytest.approx(4.0 / 252)
    rows = [
        {"stratum": "capture", "mode": "mode", "seed": 0, **scores},
        {
            "stratum": "capture",
            "mode": "mode",
            "seed": 1,
            **(scores | {"normalized_mae": 0.0, "normalized_rmse": 0.0}),
        },
    ]
    _write_metrics_table(tmp_path, rows)
    averages = _write_capture_averages(tmp_path, rows)
    assert averages[0]["run_count"] == 2
    assert averages[0]["normalized_mae"] == pytest.approx(1.75 / 252)
    assert averages[0]["normalized_rmse"] == pytest.approx(2.5 / 252)
    with (tmp_path / "metrics_table.csv").open(newline="") as stream:
        saved_rows = list(csv.DictReader(stream))
    assert float(saved_rows[0]["normalized_rmse"]) == pytest.approx(5.0 / 252)
    assert float(saved_rows[0]["normalized_mae"]) == pytest.approx(3.5 / 252)
    with (tmp_path / "capture_averages.csv").open(newline="") as stream:
        saved_averages = list(csv.DictReader(stream))
    assert float(saved_averages[0]["normalized_rmse"]) == pytest.approx(2.5 / 252)


def test_canonical_rotation_preserves_rows_across_metadata_segmentation(
    tmp_path,
) -> None:
    raw = tmp_path / "raw.csv"
    canonical = tmp_path / "canonical.csv"
    with raw.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Status", "Scale", "Range", "Gain", "Angle", "EchoValues"])
        writer.writerow([1, 496, 5, 91, 100, 0, 12, 0])
        writer.writerow([1, 496, 5, 92, 200, 0, 13, 0])
        writer.writerow([1, 496, 5, 92, 300, 4, 0, 5])
    grid, info, provenance = _canonicalize_rotation(raw, canonical, 1, 3)
    assert info["selected_source_rows"] == 2
    assert info["canonical_rotation_count"] == 1
    assert [row["angle_ticks"] for row in provenance] == [200, 300]
    assert grid.angular_valid.sum() == 2
    assert (
        info["canonical_echo_roundtrip"] == "exact metadata, angle, and echo equality"
    )


def test_child_guard_blocks_pickle_loading(tmp_path) -> None:
    probe = tmp_path / "pickle_probe.py"
    probe.write_text("import pickle\npickle.loads(b'not a pickle')\n", encoding="utf-8")
    result = subprocess.run(
        _guarded_simulator_command([sys.executable, str(probe)]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "pickle loading forbidden in E2 run" in result.stderr


def test_run_persists_failed_status_on_input_error(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "new_run"
    config_path.write_text(
        json.dumps(
            {
                "protocol": "furuno-source-conditioned-diagnostic-v1",
                "range_bins": 3,
                "seed_values": [0],
                "simulations": ["current_mode4"],
                "captures": [
                    {
                        "relative_path": "missing.csv",
                        "sha256": "0" * 64,
                        "stratum": "test",
                    }
                ],
                "display_divisor": 252,
                "range_profile": "observed-sdk-range-index",
                "radial_mapping": "linear_full_range_unverified_assumption",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        run(config_path, output_dir, tmp_path, False)
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["run_status"] == "failed"
    assert provenance["failure"]["type"] == "FileNotFoundError"


def test_score_rejects_empty_observed_source() -> None:
    grid = np.zeros((2, 3))
    with pytest.raises(ValueError, match="no observed angles"):
        score_pair(grid, grid, np.zeros(2, dtype=bool), np.ones(2, dtype=bool))
