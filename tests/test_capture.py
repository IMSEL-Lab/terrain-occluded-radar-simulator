import csv
import json
from typing import cast

import numpy as np

from ddmr.capture import (
    CaptureConfig,
    grid_rotation,
    load_capture,
    save_capture_npz,
    save_capture_summary_json,
    save_grid_npz,
    segment_rotations,
)


def write_capture(path, rows, bins=4):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Status", "Scale", "Range", "Gain", "Angle", "EchoValues"])
        for row in rows:
            metadata = row[:5]
            echoes = row[5] if len(row) > 5 else [0] * bins
            writer.writerow([*metadata, *echoes])


def spoke(angle, echoes=None, status=1, scale=496, range_code=2, gain=30):
    return [status, scale, range_code, gain, angle, echoes or [angle % 256, 2, 3, 4]]


def test_parser_keeps_metadata_and_counts_malformed_rows(tmp_path):
    path = tmp_path / "capture.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Status", "Scale", "Range", "Gain", "Angle", "EchoValues"])
        writer.writerow([1, 496, 2, 30, 10, 3, 4, 5, 6])
        writer.writerow([1, 496, 2, 30, "bad", 3, 4, 5, 6])
        writer.writerow([1, 496, 2, 30, 11, 3, 4])
        writer.writerow([4, 500, 4, 31, 12, 9, 8, 7, 6])
    capture = load_capture(path, CaptureConfig(expected_echo_bins=4))
    assert len(capture.spokes) == 2
    assert capture.spokes[0].scale == 496
    assert capture.spokes[0].range_code == 2
    assert capture.spokes[1].status == 4
    assert capture.spokes[1].range_code == 4
    assert capture.diagnostics.rows_seen == 4
    assert capture.diagnostics.rows_parsed == 2
    assert capture.diagnostics.malformed_rows == 2
    assert capture.diagnostics.malformed_reasons == {
        "invalid_number": 1,
        "echo_bin_count_mismatch": 1,
    }
    assert capture.diagnostics.excluded_status_rows == 1


def test_partial_rotation_keeps_missing_azimuths_invalid_and_uses_last_duplicate(
    tmp_path,
):
    path = tmp_path / "partial.csv"
    write_capture(
        path,
        [
            spoke(100, [1, 1, 1, 1]),
            spoke(100, [7, 7, 7, 7]),
            spoke(102, [2, 2, 2, 2]),
        ],
    )
    cfg = CaptureConfig(
        expected_echo_bins=4, azimuth_bins=8192, range_mapping_nm={2: 0.5}
    )
    capture = load_capture(path, cfg)
    rotation = segment_rotations(capture, cfg)[0]
    grid = grid_rotation(rotation, cfg)
    assert grid.angular_valid.sum() == 2
    assert np.isnan(grid.echoes[99]).all()
    assert grid.echoes[100].tolist() == [7, 7, 7, 7]
    assert grid.source_spoke_count[100] == 2
    assert grid.range_max_m == 926.0


def test_wrap_and_metadata_changes_create_segment_boundaries(tmp_path):
    path = tmp_path / "segments.csv"
    write_capture(
        path,
        [
            spoke(8000),
            spoke(100),
            spoke(200, range_code=4),
            spoke(300, status=4, range_code=4),
        ],
    )
    cfg = CaptureConfig(expected_echo_bins=4, min_wrap_drop_ticks=4000)
    rotations = segment_rotations(load_capture(path, cfg), cfg)
    assert [rotation.boundary_before for rotation in rotations] == [
        "capture_start",
        "angular_wrap",
        "range_change",
        "status_change",
    ]
    assert [(r.range_code, r.status) for r in rotations] == [
        (2, 1),
        (2, 1),
        (4, 1),
        (4, 4),
    ]


def test_stalled_angle_block_is_preserved_but_excluded_from_grid_and_rotation(tmp_path):
    path = tmp_path / "stalled.csv"
    rows = [spoke(77, [i, 1, 1, 1]) for i in range(5)] + [spoke(78)]
    write_capture(path, rows)
    cfg = CaptureConfig(expected_echo_bins=4, max_consecutive_same_angle=4)
    capture = load_capture(path, cfg)
    assert len(capture.spokes) == 6
    assert not any(s.usable for s in capture.spokes[:5])
    assert capture.diagnostics.stalled_angle_blocks == 1
    assert capture.diagnostics.stalled_rows == 5
    rotations = segment_rotations(capture, cfg)
    assert [s.angle_ticks for s in rotations[0].spokes] == [78]


def test_archive_and_summary_helpers_preserve_audit_data(tmp_path):
    path = tmp_path / "archive.csv"
    write_capture(path, [spoke(1), spoke(2)])
    cfg = CaptureConfig(expected_echo_bins=4)
    capture = load_capture(path, cfg)
    rotation = segment_rotations(capture, cfg)[0]
    grid = grid_rotation(rotation, cfg)
    raw_file = tmp_path / "raw.npz"
    grid_file = tmp_path / "grid.npz"
    summary_file = tmp_path / "summary.json"
    save_capture_npz(raw_file, capture)
    save_grid_npz(grid_file, grid)
    save_capture_summary_json(summary_file, capture)
    with np.load(raw_file) as raw:
        assert raw["echoes"].shape == (2, 4)
        assert raw["range_code"].tolist() == [2, 2]
    with np.load(grid_file) as saved:
        assert saved["angular_valid"].sum() == 2
        assert np.isnan(saved["echoes"][3]).all()
    summary = json.loads(summary_file.read_text())
    assert summary["spoke_count"] == 2
    assert summary["status_counts"] == {"1": 2}
    assert summary["diagnostics"]["rows_parsed"] == 2


def test_echo_values_must_fit_unsigned_byte_payload(tmp_path):
    path = tmp_path / "invalid-amplitude.csv"
    write_capture(
        path,
        [
            spoke(1, [-1, 0, 1, 255]),
            spoke(2, [0, 1, 256, 255]),
            spoke(3, [0, 1, 2, 255]),
        ],
    )
    capture = load_capture(path, CaptureConfig(expected_echo_bins=4))
    assert [s.angle_ticks for s in capture.spokes] == [3]
    assert capture.diagnostics.malformed_reasons == {"echo_value_out_of_range": 2}


def test_range_mapping_requires_integer_codes_and_finite_positive_ranges():
    import pytest

    with pytest.raises(ValueError, match="finite and positive"):
        CaptureConfig(range_mapping_nm={2: float("nan")})
    with pytest.raises(ValueError, match="finite and positive"):
        CaptureConfig(range_mapping_nm={2: float("inf")})
    with pytest.raises(ValueError, match="integers"):
        CaptureConfig(range_mapping_nm=cast(dict[int, float], {2.5: 0.5}))
    with pytest.raises(ValueError, match="integers"):
        CaptureConfig(range_mapping_nm={-1: 0.5})
