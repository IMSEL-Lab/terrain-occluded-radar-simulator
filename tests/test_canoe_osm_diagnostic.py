"""Unit checks for the external-map geometry diagnostic primitives."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from ddmr.canoe import PoseTable, RadarScan
from scripts.run_canoe_osm_diagnostic import (
    first_positive_intersection_range,
    fit_gain,
    gaussian_return_profile,
    latlon_to_local_enu,
    score,
    sensor_ray_direction_enu_xy,
    shoreline_ranges,
)


def test_wgs84_local_enu_cardinals() -> None:
    origin = (43.7725429111, -79.4626013556)
    east = latlon_to_local_enu(
        np.asarray([[origin[0], origin[1] + 0.00001]], dtype=np.float64), origin
    )[0]
    north = latlon_to_local_enu(
        np.asarray([[origin[0] + 0.00001, origin[1]]], dtype=np.float64), origin
    )[0]
    zero = latlon_to_local_enu(np.asarray(origin, dtype=np.float64), origin)

    assert np.allclose(zero, [0.0, 0.0], atol=1e-9)
    assert east[0] > 0 and abs(east[1]) < 1e-4
    assert north[1] > 0 and abs(north[0]) < 1e-4


def test_first_positive_intersection_uses_polygon_holes() -> None:
    water = Polygon(
        [(-10, -10), (10, -10), (10, 10), (-10, 10)],
        holes=[[(-1, 5), (1, 5), (1, 6), (-1, 6)]],
    )

    result = first_positive_intersection_range(
        (0.0, 0.0), (0.0, 1.0), 20.0, water.boundary
    )

    assert result is not None
    assert np.isclose(result, 5.0)


def test_canoe_encoder_cardinal_directions() -> None:
    level_installation = (0.0, 0.0, 0.0)
    # The helper returns M_enu_sensor @ [forward, right].
    expected = ([1, 0], [0, 1], [-1, 0], [0, -1])

    for encoder, direction in zip((0, 1400, 2800, 4200), expected, strict=True):
        assert np.allclose(
            sensor_ray_direction_enu_xy(encoder, level_installation),
            direction,
            atol=1e-12,
        )


def test_no_boundary_intersection_is_unmodeled() -> None:
    water = Polygon([(5, 5), (8, 5), (8, 8), (5, 8)])

    assert (
        first_positive_intersection_range((0, 0), (-1, 0), 20, water.boundary) is None
    )


def test_pose_unsupported_spokes_are_not_given_geometry_ranges() -> None:
    scan = RadarScan(
        path="fixture.png",
        center_time_us=150,
        row_time_us=np.asarray([150, 250], dtype=np.uint64),
        encoder=np.asarray([1400, 1400], dtype=np.uint16),
        valid_azimuth=np.asarray([True, True], dtype=np.bool_),
        echo=np.zeros((2, 8), dtype=np.uint8),
    )
    poses = PoseTable(
        time_us=np.asarray([100.0, 200.0]),
        position_enu=np.zeros((2, 3)),
        euler_xyz=np.zeros((2, 3)),
    )
    water = Polygon([(-5, 5), (5, 5), (5, 10), (-5, 10)])

    ranges, supported, inside = shoreline_ranges(scan, poses, np.zeros(2), water, 20.0)

    assert supported.tolist() == [True, False]
    assert np.isclose(ranges[0], 5.0)
    assert np.isnan(ranges[1])
    assert inside.tolist() == [False, False]


def test_gaussian_gain_fit_is_scalar_and_bounded() -> None:
    distances = np.asarray([5.0])
    centers = np.arange(20, dtype=np.float64) + 0.5
    unit = gaussian_return_profile(distances, centers, 2.0)
    reference = 0.4 * unit
    gain = fit_gain(unit, reference, np.ones_like(unit, dtype=bool))

    assert np.isclose(gain, 0.4)
    assert 0.0 <= gain <= 1.0


def test_score_fails_on_nonfinite_prediction_in_valid_cells() -> None:
    reference = np.asarray([[0.2, 0.4]])
    prediction = np.asarray([[0.2, np.nan]])
    valid = np.asarray([[True, True]], dtype=np.bool_)

    with pytest.raises(ValueError, match="nonfinite"):
        score(reference, prediction, valid, threshold=0.3)


def test_undefined_occupancy_ratios_are_none_with_counts() -> None:
    reference = np.asarray([[0.0, 0.8]])
    prediction = np.zeros_like(reference)
    valid = np.ones_like(reference, dtype=np.bool_)
    result = score(reference, prediction, valid, threshold=0.5)

    assert result["predicted_active_cells"] == 0
    assert result["reference_active_cells"] == 1
    assert result["occupancy_precision"] is None
    assert result["occupancy_recall"] == 0.0
    assert result["occupancy_iou"] == 0.0


def test_undefined_recall_is_none_when_no_reference_cells_active() -> None:
    result = score(
        np.zeros((1, 2)),
        np.asarray([[0.0, 0.8]]),
        np.ones((1, 2), dtype=np.bool_),
        threshold=0.5,
    )

    assert result["predicted_active_cells"] == 1
    assert result["reference_active_cells"] == 0
    assert result["occupancy_precision"] == 0.0
    assert result["occupancy_recall"] is None
    assert result["occupancy_iou"] == 0.0
