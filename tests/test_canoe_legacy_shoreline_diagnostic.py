"""Focused guards for the bounded original-renderer CANOE adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Polygon

from ddmr.canoe import PoseTable, RadarScan
from scripts.run_canoe_legacy_shoreline_diagnostic import (
    apply_scalar_gain,
    build_explicit_mode1_scene,
    explicit_segment_strip_coastlines,
    fit_scalar_gain,
    load_native_shoreline_strips,
    map_enu_to_legacy_matrix,
    parse_renderer_csv,
    preserve_reference_display_255,
    renderer_ticks,
    scan_motion_summary,
    score_prediction,
    water_boundary_scene_frame,
)


def _fake_simulator() -> SimpleNamespace:
    return SimpleNamespace(SIM_MODE_M1=1)


def test_boundary_scene_contains_lines_only_and_uses_full_pose_bridge() -> None:
    water = Polygon([(-2.0, 10.0), (2.0, 10.0), (2.0, 12.0), (-2.0, 12.0)])

    north_facing = water_boundary_scene_frame(
        water, np.asarray([0.0, 0.0]), np.asarray([0.0, 0.0, 0.0])
    )
    rotated = water_boundary_scene_frame(
        water, np.asarray([0.0, 0.0]), np.asarray([0.0, 0.0, np.pi / 2.0])
    )

    assert all(
        feature["geometry"]["type"] == "LineString"
        for feature in north_facing["features"]
    )
    factor = np.pi * 6_371_000.0 / 180.0
    north_points = np.asarray(
        [
            [point[0] * factor, point[1] * factor]
            for feature in north_facing["features"]
            for point in feature["geometry"]["coordinates"]
        ]
    )
    rotated_points = np.asarray(
        [
            [point[0] * factor, point[1] * factor]
            for feature in rotated["features"]
            for point in feature["geometry"]["coordinates"]
        ]
    )
    # With zero yaw, the official sensor's forward axis points east and its
    # right axis points north; the legacy PPI frame swaps those axes.
    assert np.isclose(np.min(north_points[:, 0]), 10.0, atol=1e-5)
    assert np.isclose(np.max(north_points[:, 0]), 12.0, atol=1e-5)
    assert np.min(rotated_points[:, 1]) < -11.9
    assert np.max(rotated_points[:, 1]) > -10.1
    assert np.max(np.abs(rotated_points[:, 0])) < 2.1
    assert all(
        feature["geometry"]["type"] != "Polygon" for feature in rotated["features"]
    )


def test_native_loader_receives_only_boundary_lines_and_frozen_clip(
    tmp_path: Path,
) -> None:
    class FakeSimulator:
        def load_coastlines_from_geojson(
            self, path: str, ref_lat: float, ref_lon: float, *, max_range_m: float
        ):
            source = json.loads(Path(path).read_text())
            assert ref_lat == 0.0 and ref_lon == 0.0
            assert max_range_m == 150.0
            assert all(
                item["geometry"]["type"] == "LineString" for item in source["features"]
            )
            return [
                SimpleNamespace(points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
            ]

    feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [0, 1e-5]]},
            }
        ],
    }
    out = load_native_shoreline_strips(
        FakeSimulator(), feature, tmp_path / "input.json"
    )

    assert len(out) == 1
    assert out[0].points[1] == (1.0, 0.0)


def test_explicit_segment_rectangles_are_valid_three_meter_square_capped_strips() -> (
    None
):
    class FakeSimulator:
        @staticmethod
        def CoastlineSegment(*, points, name, closed):
            return SimpleNamespace(points=points, name=name, closed=closed)

    degrees_per_10m = 10.0 * 180.0 / (np.pi * 6_371_000.0)
    boundary = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [0.0, 0.0],
                        [degrees_per_10m, 0.0],
                        [degrees_per_10m, degrees_per_10m],
                    ],
                },
            }
        ],
    }

    segments, counts = explicit_segment_strip_coastlines(FakeSimulator(), boundary)
    polygons = [Polygon(segment.points) for segment in segments]

    assert counts["source_boundary_segment_count"] == 2
    assert counts["explicit_rectangle_count"] == 2
    assert counts["zero_or_duplicate_segment_count"] == 0
    assert all(segment.closed for segment in segments)
    assert all(polygon.is_valid and polygon.area > 0 for polygon in polygons)
    assert all(polygon.area == pytest.approx(39.0, abs=1e-6) for polygon in polygons)
    # A 10 m segment receives 1.5 m extension at both ends and 3 m width.
    assert all(polygon.length == pytest.approx(32.0, abs=1e-6) for polygon in polygons)
    assert counts["overlapping_rectangle_pair_count"] == 1


def test_explicit_segment_rectangles_skip_duplicates_and_reject_bad_inputs() -> None:
    class FakeSimulator:
        @staticmethod
        def CoastlineSegment(*, points, name, closed):
            return SimpleNamespace(points=points, name=name, closed=closed)

    boundary = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0.0, 0.0], [0.0, 0.0], [1e-5, 0.0]],
                },
            }
        ],
    }
    segments, counts = explicit_segment_strip_coastlines(FakeSimulator(), boundary)
    assert len(segments) == 1
    assert counts["zero_or_duplicate_segment_count"] == 1

    invalid = json.loads(json.dumps(boundary))
    invalid["features"][0]["geometry"]["coordinates"][1] = [float("nan"), 0.0]
    with pytest.raises(ValueError, match="nonfinite"):
        explicit_segment_strip_coastlines(FakeSimulator(), invalid)

    with pytest.raises(ValueError, match="width"):
        explicit_segment_strip_coastlines(FakeSimulator(), boundary, width_m=0.0)


def test_explicit_mode1_scene_has_no_source_or_replay_payload() -> None:
    scene = build_explicit_mode1_scene(_fake_simulator(), [0, 123, 123, 400], seed=42)

    assert scene["_sim_mode"] == 1
    assert scene["_sim_seed"] == 42
    assert scene["angle_ticks"] == [0, 123, 123, 400]
    assert scene["radar_config"]["bins"] == 300
    assert scene["radar_config"]["bin_size_m"] == 0.5
    assert scene["radar_config"]["range_m"] == 150.0
    assert scene["clutter"]["noise_probability"] == 0.0
    assert scene["objects"] == []
    assert "coastline_profiles" not in scene
    assert "object_echo_data" not in scene
    assert "_mode42" not in scene


def test_map_to_legacy_bridge_roundtrips_full_official_pose_matrix() -> None:
    angles = np.asarray([3.114505933424574, -0.00025324389721649365, 1.982704549616565])
    matrix = map_enu_to_legacy_matrix(angles)
    # Official sensor [forward,right] and renderer [right,forward] differ by P.
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    from ddmr.canoe import _c_enuref_sensor

    enu_from_sensor = _c_enuref_sensor(*angles)[:2, :2]
    for legacy_ray in (
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 0.0]),
        np.asarray([1.0, -1.0]),
    ):
        official_sensor_ray = swap @ legacy_ray
        enu_delta = enu_from_sensor @ official_sensor_ray
        recovered = matrix @ enu_delta
        np.testing.assert_allclose(recovered, legacy_ray, atol=1e-12)


def test_encoder_to_tick_conversion_preserves_order_and_duplicates() -> None:
    encoder = np.asarray([0, 1400, 1400, 4200], dtype=np.uint16)

    assert renderer_ticks(encoder) == [0, 2048, 2048, 6144]


def test_renderer_csv_requires_exact_rows_angles_and_radial_bins(
    tmp_path: Path,
) -> None:
    path = tmp_path / "render.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Status", "Scale", "Range", "Gain", "Angle", "EchoValues"])
        writer.writerow([1, 496, 0, 75, 2, 10, 20])
        writer.writerow([1, 496, 0, 75, 2, 30, 40])

    assert np.array_equal(parse_renderer_csv(path, [2, 2], 2), [[10, 20], [30, 40]])
    with pytest.raises(ValueError, match="angle"):
        parse_renderer_csv(path, [2, 3], 2)
    with pytest.raises(ValueError, match="spoke count"):
        parse_renderer_csv(path, [2], 2)


def test_motion_summary_uses_scan_center_and_supported_rows() -> None:
    scan = RadarScan(
        path="fixture.png",
        center_time_us=200,
        row_time_us=np.asarray([100, 200, 300], dtype=np.uint64),
        encoder=np.asarray([0, 0, 0], dtype=np.uint16),
        valid_azimuth=np.asarray([True, True, True], dtype=np.bool_),
        echo=np.zeros((3, 2), dtype=np.uint8),
    )
    poses = PoseTable(
        time_us=np.asarray([100.0, 200.0, 300.0]),
        position_enu=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        euler_xyz=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.3]]),
    )

    center, yaw, summary = scan_motion_summary(scan, poses)

    assert np.allclose(center, [1.0, 0.0])
    assert yaw == pytest.approx(0.1)
    assert summary["median_intra_scan_displacement_m"] == pytest.approx(1.0)
    assert summary["maximum_intra_scan_displacement_m"] == pytest.approx(2.0)
    assert summary["maximum_absolute_intra_scan_yaw_change_deg"] == pytest.approx(
        np.rad2deg(0.2)
    )


def test_reference_scale_conversion_gain_fit_and_shared_mask() -> None:
    source_display = np.asarray([[255.0, 127.5, 0.0]]) / 255.0
    reference = preserve_reference_display_255(source_display)
    assert np.array_equal(reference, source_display)
    raw = np.asarray([[0.2, 0.4, 0.6]])
    target = raw * 0.5
    mask = np.asarray([[True, True, False]], dtype=np.bool_)
    assert fit_scalar_gain(raw, target, mask) == pytest.approx(0.5)
    scored = score_prediction(target, raw * 0.5, mask, occupancy_threshold=0.15)
    assert scored["valid_cells"] == 2
    assert scored["reference_positive_cells"] == 1
    assert scored["predicted_positive_cells"] == 1
    assert scored["occupancy_iou"] == 1.0

    unbounded = fit_scalar_gain(
        np.asarray([[0.5]]), np.asarray([[1.0]]), np.asarray([[True]], dtype=np.bool_)
    )
    assert unbounded == pytest.approx(2.0)
    assert np.array_equal(
        apply_scalar_gain(np.asarray([[0.5, 0.75]]), unbounded), [[1.0, 1.0]]
    )


def test_score_fails_if_valid_prediction_is_missing() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        score_prediction(
            np.asarray([[0.0, 0.2]]),
            np.asarray([[0.0, np.nan]]),
            np.asarray([[True, True]], dtype=np.bool_),
            occupancy_threshold=0.1,
        )
