"""Contract tests for grouped component-pixel extraction."""

from __future__ import annotations

import numpy as np
import pytest

import extract_scene


@pytest.mark.parametrize(
    "shape,density,label_values",
    [
        ((1, 1), 0.0, [0, 4, 17]),
        ((17, 23), 0.08, [0, 1, 7, 31]),
        ((31, 19), 0.82, [0, 2, 3, 90]),
        ((24, 33), 0.35, [0, 5, 101, 9001]),
    ],
)
def test_grouped_pixels_match_argwhere_for_label_arrays(shape, density, label_values):
    rng = np.random.default_rng(439)
    labels = rng.choice(label_values, size=shape, p=None)
    if density < 0.5:
        labels[rng.random(shape) >= density] = 0
    actual = list(extract_scene._iter_component_pixels_by_label(labels))
    expected = [
        (int(label), np.argwhere(labels == label))
        for label in np.unique(labels[labels > 0])
    ]
    assert [label for label, _ in actual] == [label for label, _ in expected]
    for (_, actual_pixels), (_, expected_pixels) in zip(actual, expected):
        np.testing.assert_array_equal(actual_pixels, expected_pixels)


def test_grouped_pixels_handles_empty_and_wrapped_border_labels():
    assert (
        list(
            extract_scene._iter_component_pixels_by_label(
                np.zeros((0, 8), dtype=np.int32)
            )
        )
        == []
    )
    labels = np.zeros((7, 9), dtype=np.int32)
    labels[0, 3:5] = 8
    labels[-1, 3:5] = 8
    labels[2, 1] = 2
    actual = list(extract_scene._iter_component_pixels_by_label(labels))
    assert [label for label, _ in actual] == [2, 8]
    np.testing.assert_array_equal(actual[0][1], np.argwhere(labels == 2))
    np.testing.assert_array_equal(actual[1][1], np.argwhere(labels == 8))


def _old_argwhere_collector(labels):
    """Reference the pre-change collection contract for end-to-end parity."""
    for label in range(1, int(labels.max()) + 1):
        pixels = np.argwhere(labels == label)
        if pixels.size:
            yield label, pixels


def _detection_fixture():
    spokes_count, bins = 16, 96
    spokes = [
        {"angle_tick": int(index * extract_scene.TICKS_PER_REV / spokes_count)}
        for index in range(spokes_count)
    ]
    land_mask = np.zeros((spokes_count, bins), dtype=bool)
    echo_map = np.zeros((spokes_count, bins), dtype=np.int32)
    # Three distinct returns, including one connected across the spoke wrap.
    echo_map[2:5, 42:45] = np.array([[180, 190, 200], [200, 210, 220], [220, 230, 240]])
    echo_map[[15, 0, 1], 58:61] = 205
    echo_map[8:11, 35:38] = 195
    return spokes, bins, 5.0, land_mask, echo_map


def test_detect_objects_matches_previous_argwhere_collection(monkeypatch):
    fixture = _detection_fixture()
    optimized = extract_scene.detect_objects(*fixture)
    assert optimized[0], "parity fixture must contain detected objects"
    assert optimized[1].any()
    assert optimized[2]
    monkeypatch.setattr(
        extract_scene, "_iter_component_pixels_by_label", _old_argwhere_collector
    )
    reference = extract_scene.detect_objects(*fixture)
    assert optimized[0] == reference[0]
    np.testing.assert_array_equal(optimized[1], reference[1])
    assert optimized[2] == reference[2]
