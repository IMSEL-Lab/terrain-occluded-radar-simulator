"""Small contract checks for the original random geometry audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.audit_legacy_random_geometry as audit

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_plan_is_100_seeded_original_generator_scenes() -> None:
    config = json.loads(
        (ROOT / "configs/legacy_random_geometry_20261116.json").read_text()
    )
    assert (
        len(config["ranges_m"])
        * len(config["object_loads"])
        * len(config["replicate_seeds"])
        == 100
    )
    assert config["object_type"] == "large_boat"
    assert config["generator"]["coastline_function"].startswith(
        "generate_random_coastlines"
    )
    assert config["generator"]["object_function"].startswith("generate_random_objects")
    assert "synthetic" in config["generator"]["synthetic_library_primitive"]


def test_synthetic_primitive_is_closed_and_capture_free() -> None:
    points = audit._synthetic_fixture_polygon()
    library = audit._synthetic_library()
    primitive = library.primitives[0]
    assert len(primitive.normalized_points) == 32
    assert primitive.source_pattern == "synthetic_audit_fixture"
    assert primitive.echo_stats == {}
    assert audit.Polygon(points).is_valid
    assert primitive.classification == audit.legacy.classify_coastline(points)[0]
    assert primitive.original_num_points == len(points)
    assert all(
        0 <= bearing <= 1 and 0 <= range_offset <= 1
        for bearing, range_offset in primitive.normalized_points
    )


def test_geometry_metrics_report_land_and_object_intersections() -> None:
    coast = audit.legacy.CoastlineSegment(
        points=[(-10, -10), (10, -10), (10, 10), (-10, 10)], name="land"
    )
    first = SimpleNamespace(x=0.0, y=0.0, heading_deg=0.0, length_m=4.0, width_m=2.0)
    second = SimpleNamespace(x=0.5, y=0.0, heading_deg=0.0, length_m=4.0, width_m=2.0)
    metrics = audit._geometry_metrics(
        [coast], [first, second], 100.0, requested_object_count=3
    )
    assert metrics["invalid_closed_polygon_count"] == 0
    assert metrics["coastline_closed_polygon_count"] == 1
    assert metrics["coastline_open_segment_count"] == 0
    assert metrics["object_center_in_land_legacy_predicate_count"] == 2
    assert metrics["object_center_in_valid_land_count"] == 2
    assert metrics["object_footprint_valid_land_intersection_count"] == 2
    assert metrics["object_footprint_overlap_pairs"] == 1
    assert metrics["object_count_requested"] == 3
    assert metrics["object_count_returned"] == 2
    assert metrics["object_count_return_shortfall"] == 1


def test_output_refuses_overwrite_and_source_hashes_cover_generator(
    tmp_path: Path,
) -> None:
    output = tmp_path / "legacy_geometry_overwrite_test"
    audit._create_output_dir(output)
    with pytest.raises(FileExistsError):
        audit._create_output_dir(output)
    hashes = audit._source_hashes()
    assert {"radar_simulator.py", "object_renderer.py", "runner.py"} <= set(hashes)
    assert all(len(digest) == 64 for digest in hashes.values())


def test_manifest_hashes_inputs_and_marks_exceptions_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads(
        (ROOT / "configs/legacy_random_geometry_20261116.json").read_text()
    )
    config.update(
        {
            "ranges_m": [250.0],
            "object_loads": [1],
            "replicate_seeds": [7],
            "expected_scene_count": 1,
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(audit, "_scene", lambda range_m, load, seed: ([], []))
    output = tmp_path / "output"
    audit.run(config_path, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["run_status"] == "complete"
    assert (
        manifest["config_sha256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
    )

    failed_output = tmp_path / "failed"
    monkeypatch.setattr(
        audit,
        "_scene",
        lambda *args: (_ for _ in ()).throw(ValueError("audit failure")),
    )
    with pytest.raises(ValueError, match="audit failure"):
        audit.run(config_path, failed_output)
    failed_manifest = json.loads((failed_output / "manifest.json").read_text())
    assert failed_manifest["run_status"] == "failed"
    assert failed_manifest["failure"]["type"] == "ValueError"
