"""Public GeoJSON validation protects the frozen renderer from lossy inputs."""

import json
from argparse import Namespace

import pytest

from tors.map_scene import _clip_lines, _scene, _validate_geojson, main


def feature(geometry):
    return {"type": "Feature", "properties": {}, "geometry": geometry}


def line(points):
    return {"type": "LineString", "coordinates": points}


def test_single_feature_renders(tmp_path):
    source = tmp_path / "line.geojson"
    source.write_text(json.dumps(feature(line([[0.0001, 0], [0.0001, 0.0001]]))))
    output = tmp_path / "output"
    main(
        ["--geojson", str(source), "--lat", "0", "--lon", "0", "--output", str(output)]
    )
    assert (output / "run_manifest.json").is_file()
    assert len(json.loads((output / "scene.json").read_text())["coastlines"]) == 1


@pytest.mark.parametrize("geometry", [None, {"type": "Point", "coordinates": [0, 0]}])
def test_mixed_collection_rejects_unsupported_geometry(geometry):
    data = {
        "type": "FeatureCollection",
        "features": [feature(line([[0, 0], [0.0001, 0]])), feature(geometry)],
    }
    with pytest.raises(ValueError, match="geometry"):
        _validate_geojson(data, 0, 0, 150)


@pytest.mark.parametrize(
    "rings",
    [
        [
            [[0, 0], [0.001, 0], [0, 0.001], [0, 0]],
            [[0, 0], [0.0001, 0], [0, 0.0001], [0, 0]],
        ],
        [[[0, 0], [0.001, 0], [0, 0.001]]],
        [[[0, 0], [0.001, 0.001], [0, 0.001], [0.001, 0], [0, 0]]],
        [],
    ],
)
def test_invalid_or_unsupported_polygons_are_rejected(rings):
    with pytest.raises(ValueError, match="polygon|ring"):
        _validate_geojson({"type": "Polygon", "coordinates": rings}, 0, 0, 150)


def test_crossing_line_with_both_endpoints_outside_renders(tmp_path):
    source = tmp_path / "crossing.geojson"
    # Latitude 0.001 degrees is approximately 111 m north of the radar.
    source.write_text(json.dumps(line([[-0.003, 0.001], [0.003, 0.001]])))
    output = tmp_path / "crossing"
    main(
        ["--geojson", str(source), "--lat", "0", "--lon", "0", "--output", str(output)]
    )
    coasts = json.loads((output / "scene.json").read_text())["coastlines"]
    assert len(coasts) == 1
    assert (
        min(p[0] for p in coasts[0]["points"])
        < 0
        < max(p[0] for p in coasts[0]["points"])
    )


def test_outside_feature_does_not_hide_in_range_feature():
    in_range = line([[0, 0], [0.0001, 0]])
    data = {
        "type": "FeatureCollection",
        "features": [feature(line([[0.01, 0], [0.02, 0]])), feature(in_range)],
    }
    result = _clip_lines(_validate_geojson(data, 0, 0, 150), 0, 0, 150)
    assert len(result["features"]) == 1
    assert result["features"][0]["geometry"] == in_range


def test_line_exit_reentry_produces_separate_pieces():
    points = [[0, 0], [0.003, 0], [0.003, 0.001], [0, 0.001]]
    result = _clip_lines(_validate_geojson(line(points), 0, 0, 150), 0, 0, 150)
    parts = result["features"][0]["geometry"]["coordinates"]
    assert len(parts) == 2
    assert parts[0][0] == points[0] and parts[1][-1] == points[-1]
    assert parts[0][-1] != parts[1][0]


def test_grid_scene_missing_range_is_a_clear_error(tmp_path):
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"radar_config": {}, "angle_ticks": [0]}))
    with pytest.raises(ValueError, match="radar_config.range_m"):
        _scene(Namespace(grid_scene=grid, seed=None))
