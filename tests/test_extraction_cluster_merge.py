"""Order and output parity for accelerated aligned-cluster merging."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import extract_scene


def _records(specs, bearing_overrides=None):
    spokes_count, num_bins = 360, 256
    spokes = [
        {"angle_tick": int(index * extract_scene.TICKS_PER_REV / spokes_count)}
        for index in range(spokes_count)
    ]
    for index, tick in (bearing_overrides or {}).items():
        spokes[index]["angle_tick"] = tick

    labels = np.zeros((spokes_count, num_bins), dtype=np.int32)
    echo_map = np.zeros((spokes_count, num_bins), dtype=np.int32)
    for label, (spoke_index, start_bin) in enumerate(specs, start=1):
        bins = np.arange(start_bin, start_bin + 3)
        labels[spoke_index, bins] = label
        echo_map[spoke_index, bins] = [180, 220, 200]

    land_mask = np.zeros_like(labels, dtype=bool)
    object_mask = labels > 0
    records = []
    for label in range(1, len(specs) + 1):
        pixels = np.argwhere(labels == label)
        record = extract_scene._build_object_cluster_record(
            label,
            pixels,
            spokes,
            num_bins,
            1.0,
            land_mask,
            object_mask,
            echo_map,
        )
        assert record is not None
        records.append(record)
    return records, spokes, echo_map


def _legacy_merge(
    clusters, spokes, bin_size_m, echo_map, bearing_tol_deg=2.0, range_overlap_m=10.0
):
    """Literal pre-optimization scan order using the real record merger."""
    merged = list(clusters)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a = merged[i]
                b = merged[j]
                d_brg = abs(a["obj_bearing"] - b["obj_bearing"])
                d_brg = min(d_brg, 360.0 - d_brg)
                if d_brg > bearing_tol_deg:
                    continue
                a_min = a["min_bin"] * bin_size_m
                a_max = (a["max_bin"] + 1) * bin_size_m
                b_min = b["min_bin"] * bin_size_m
                b_max = (b["max_bin"] + 1) * bin_size_m
                range_gap = max(a_min, b_min) - min(a_max, b_max)
                if range_gap > range_overlap_m:
                    continue
                merged[i] = extract_scene._merge_cluster_records(
                    a, b, spokes, bin_size_m, echo_map
                )
                merged.pop(j)
                changed = True
                break
            if changed:
                break
    return merged


def _compare_with_legacy(
    specs, *, bearing_tol_deg=2.0, range_overlap_m=10.0, bearing_overrides=None
):
    records, spokes, echo_map = _records(specs, bearing_overrides)
    expected = _legacy_merge(
        copy.deepcopy(records), spokes, 1.0, echo_map, bearing_tol_deg, range_overlap_m
    )
    actual = extract_scene._merge_aligned_clusters(
        copy.deepcopy(records),
        spokes,
        1.0,
        echo_map,
        bearing_tol_deg,
        range_overlap_m,
    )
    assert actual == expected
    return actual


def test_random_nonempty_records_match_original_scan_order():
    rng = np.random.default_rng(951)
    for _ in range(15):
        spoke_indices = rng.choice(360, size=45, replace=False)
        specs = [(int(spoke), int(rng.integers(35, 205))) for spoke in spoke_indices]
        _compare_with_legacy(specs)


def test_chained_merge_rechecks_earlier_pair_before_later_candidate():
    result = _compare_with_legacy(
        [(10, 10), (11, 30), (12, 20)],
        bearing_tol_deg=2.1,
        range_overlap_m=7.0,
    )
    assert len(result) == 1
    assert result[0]["labels"] == [1, 3, 2]


@pytest.mark.parametrize(
    "specs,bearing_tol,range_tol,overrides",
    [
        (
            [(20, 10), (21, 20)],
            0.0,
            10.0,
            {21: 20 * extract_scene.TICKS_PER_REV // 360},
        ),
        ([(20, 10), (21, 20)], -0.1, 10.0, None),
        ([(359, 10), (1, 20)], 3.5, 10.0, None),
        (
            [(20, 20), (21, 21)],
            0.0,
            -1.0,
            {21: 20 * extract_scene.TICKS_PER_REV // 360},
        ),
    ],
)
def test_nondefault_tolerances_and_wrap_match_original(
    specs, bearing_tol, range_tol, overrides
):
    _compare_with_legacy(
        specs,
        bearing_tol_deg=bearing_tol,
        range_overlap_m=range_tol,
        bearing_overrides=overrides,
    )


def test_missing_merge_inputs_returns_original_sequence():
    clusters = [{"obj_bearing": 10.0}, {"obj_bearing": 10.1}]
    assert extract_scene._merge_aligned_clusters(clusters) is clusters


def test_zero_tolerance_canonicalizes_360_degree_bearing():
    records, spokes, echo_map = _records([(359, 10), (0, 20)])
    records[0]["obj_bearing"] = 360.0
    records[1]["obj_bearing"] = 0.0
    expected = _legacy_merge(copy.deepcopy(records), spokes, 1.0, echo_map, 0.0, 10.0)
    actual = extract_scene._merge_aligned_clusters(
        copy.deepcopy(records), spokes, 1.0, echo_map, 0.0, 10.0
    )
    assert actual == expected
    assert len(actual) == 1


def test_tiny_positive_tolerance_uses_parity_preserving_fallback():
    records, spokes, echo_map = _records([(20, 10), (21, 20)])
    records[0]["obj_bearing"] = records[1]["obj_bearing"]
    tolerance = np.nextafter(0.0, 1.0)
    expected = _legacy_merge(
        copy.deepcopy(records), spokes, 1.0, echo_map, tolerance, 10.0
    )
    actual = extract_scene._merge_aligned_clusters(
        copy.deepcopy(records), spokes, 1.0, echo_map, tolerance, 10.0
    )
    assert actual == expected
