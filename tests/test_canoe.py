import csv

import numpy as np
import pytest
from PIL import Image

from ddmr.canoe import (
    PoseTable,
    RadarScan,
    build_empirical_map,
    interpolate_pose,
    load_pose_table,
    load_radar_png,
    map_lookup,
    pose_time_supported,
    project_scan_to_common_grid,
    project_scan_with_native_range,
    radar_range_resolution_m,
    scan_positive_returns,
)


def test_radar_png_decodes_embedded_fields_and_validity_mask(tmp_path):
    raw = np.zeros((3, 15), dtype=np.uint8)
    raw[:, :8] = np.frombuffer(
        np.asarray([100, 200, 300], dtype="<u8").tobytes(), dtype=np.uint8
    ).reshape(3, 8)
    raw[:, 8:10] = np.frombuffer(
        np.asarray([0, 2800, 5599], dtype="<u2").tobytes(), dtype=np.uint8
    ).reshape(3, 2)
    raw[:, 10] = [255, 0, 255]
    raw[:, 11:] = [[1, 2, 3, 4], [5, 6, 7, 8], [0, 9, 10, 11]]
    path = tmp_path / "1234567890123456.png"
    Image.fromarray(raw).save(path)

    scan = load_radar_png(path)

    np.testing.assert_array_equal(scan.row_time_us, [100, 200, 300])
    np.testing.assert_array_equal(scan.encoder, [0, 2800, 5599])
    np.testing.assert_array_equal(scan.valid_azimuth, [True, False, True])
    np.testing.assert_array_equal(scan.echo, raw[:, 11:])
    assert scan.center_time_us == 1234567890123456


def test_pose_csv_schema_and_interpolation_are_bounded(tmp_path):
    path = tmp_path / "radar_poses.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["utc_micro", "x", "y", "z", "ang_x", "ang_y", "ang_z"])
        writer.writerow([100, 0, 0, 0, 0, 0, 3.1])
        writer.writerow([200, 10, 20, 0, 0, 0, -3.1])
    table = load_pose_table(path)
    position, angles = interpolate_pose(table, np.asarray([150], dtype=np.uint64))
    np.testing.assert_allclose(position, [[5, 10, 0]])
    assert angles[0, 2] == pytest.approx(np.pi)
    before, _ = interpolate_pose(table, np.asarray([0], dtype=np.uint64))
    np.testing.assert_allclose(before, [[0, 0, 0]])
    assert not pose_time_supported(table, np.asarray([0], dtype=np.uint64))[0]


def test_train_return_map_and_support_aware_lookup():
    points = np.asarray([[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]])
    intensities = np.asarray([0.2, 0.4, 0.6, 0.8])
    grid = build_empirical_map([(points, intensities)], resolution_m=0.5)
    values, support = map_lookup(
        grid, np.asarray([0.25, 20.0]), np.asarray([0.25, 20.0]), "bilinear"
    )
    assert support.tolist() == [True, False]
    assert values[0] == pytest.approx(0.5)
    assert np.isnan(values[1])


def test_positive_return_transform_uses_valid_rows_and_encoder_convention():
    scan = load_radar_png_for_test()
    table = PoseTable(
        time_us=np.asarray([100, 200], dtype=np.float64),
        position_enu=np.zeros((2, 3), dtype=np.float64),
        euler_xyz=np.zeros((2, 3), dtype=np.float64),
    )
    xy, echo = scan_positive_returns(
        scan, table, range_resolution_m=1.0, range_offset_m=0.0
    )
    assert xy.shape == (1, 2)
    np.testing.assert_allclose(xy[0], [3.5, 0], atol=1e-6)
    assert echo[0] == pytest.approx(100 / 255)


def test_official_forward_right_sensor_basis_cardinal_encoder_projection():
    scan = RadarScan(
        path="synthetic-cardinals",
        center_time_us=50,
        row_time_us=np.asarray([50, 50, 50, 50], dtype=np.uint64),
        encoder=np.asarray([0, 1400, 2800, 4200], dtype=np.uint16),
        valid_azimuth=np.ones(4, dtype=np.bool_),
        echo=np.zeros((4, 1), dtype=np.uint8),
    )
    poses = PoseTable(
        time_us=np.asarray([0, 100], dtype=np.float64),
        position_enu=np.zeros((2, 3), dtype=np.float64),
        euler_xyz=np.zeros((2, 3), dtype=np.float64),
    )

    projected = project_scan_to_common_grid(
        scan,
        poses,
        range_resolution_m=100.0,
        range_offset_m=0.0,
        min_range_m=1.0,
        max_range_m=100.0,
        radial_step_m=100.0,
    )

    np.testing.assert_allclose(
        np.column_stack((projected.x_enu[:, 0], projected.y_enu[:, 0])),
        [[50.0, 0.0], [0.0, 50.0], [-50.0, 0.0], [0.0, -50.0]],
        atol=1e-6,
    )


@pytest.mark.parametrize("native_max_range", [150.0, 1000.0])
def test_native_range_modes_crop_to_same_physical_range(native_max_range):
    bins = 3424
    resolution = radar_range_resolution_m(native_max_range, bins)
    target_range_m = 100.0
    source_bin = round((target_range_m + 0.407910) / resolution - 0.5)
    echo = np.zeros((2, bins), dtype=np.uint8)
    echo[0, source_bin] = 200
    scan = RadarScan(
        path="synthetic",
        center_time_us=150,
        row_time_us=np.asarray([100, 200], dtype=np.uint64),
        encoder=np.asarray([0, 0], dtype=np.uint16),
        valid_azimuth=np.array([True, False], dtype=np.bool_),
        echo=echo,
    )
    poses = PoseTable(
        time_us=np.asarray([100, 200], dtype=np.float64),
        position_enu=np.zeros((2, 3), dtype=np.float64),
        euler_xyz=np.zeros((2, 3), dtype=np.float64),
    )
    projected = project_scan_with_native_range(
        scan,
        poses,
        native_max_range_m=native_max_range,
        analysis_max_range_m=150,
        range_offset_m=-0.407910,
        radial_step_m=0.5,
    )
    source_center = (source_bin + 0.5) * resolution - 0.407910
    target_bin = int(source_center / 0.5)
    assert projected.echo.shape == (2, 300)
    assert projected.echo[0, target_bin] > 0
    assert not projected.valid[1].any()


def test_official_range_resolution_matches_documented_modes():
    assert radar_range_resolution_m(150.0, 3424) == pytest.approx(0.0438084, rel=1e-5)
    assert radar_range_resolution_m(1000.0, 3424) == pytest.approx(0.292056, rel=1e-5)


def load_radar_png_for_test():
    from ddmr.canoe import RadarScan

    return RadarScan(
        path="synthetic",
        center_time_us=150,
        row_time_us=np.asarray([100, 200], dtype=np.uint64),
        encoder=np.asarray([0, 2800], dtype=np.uint16),
        valid_azimuth=np.array([True, False], dtype=np.bool_),
        echo=np.asarray([[0, 0, 0, 100], [0, 0, 0, 255]], dtype=np.uint8),
    )
