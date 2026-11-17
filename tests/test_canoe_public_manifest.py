"""The portable source manifest is accepted only for the frozen CANOE pin."""

import hashlib
import json

import pytest

from scripts import canoe_manifest
from scripts import run_canoe_legacy_shoreline_diagnostic as diagnostic
from scripts import run_canoe_osm_diagnostic as osm


def digest(content):
    return hashlib.sha256(content).hexdigest()


def test_historical_and_portable_pins_are_narrow(tmp_path, monkeypatch):
    original = b"historical manifest"
    portable = b"portable manifest"
    path = tmp_path / "source.json"
    monkeypatch.setattr(canoe_manifest, "HISTORICAL_SOURCE_SHA256", digest(original))
    monkeypatch.setattr(canoe_manifest, "PORTABLE_SOURCE_SHA256", digest(portable))
    for content in (original, portable):
        path.write_bytes(content)
        assert canoe_manifest.verify_source_manifest(path, digest(original)) == digest(
            content
        )
    path.write_bytes(b"modified manifest")
    with pytest.raises(ValueError, match="source manifest SHA-256 mismatch"):
        canoe_manifest.verify_source_manifest(path, digest(original))
    path.write_bytes(portable)
    with pytest.raises(ValueError, match="source manifest SHA-256 mismatch"):
        canoe_manifest.verify_source_manifest(path, digest(b"custom pin"))


def test_map_reader_checks_manifest_and_retains_scan_hash_check(tmp_path, monkeypatch):
    scan = tmp_path / "scan.png"
    scan.write_bytes(b"original scan")
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {"selected_scans": [{"key": "scan", "sha256": digest(scan.read_bytes())}]}
        )
    )
    historical = digest(b"historical manifest")
    monkeypatch.setattr(canoe_manifest, "HISTORICAL_SOURCE_SHA256", historical)
    monkeypatch.setattr(
        canoe_manifest, "PORTABLE_SOURCE_SHA256", digest(source.read_bytes())
    )
    selection = tmp_path / "selection.json"
    selection.write_text('{"sequences": {}}')
    config = {
        "dataset": {
            "source_manifest": source.name,
            "source_manifest_sha256": historical,
            "selection_manifest": selection.name,
            "selection_manifest_sha256": digest(selection.read_bytes()),
        }
    }
    assert diagnostic._load_records(config, tmp_path, tmp_path, ()) == {}
    expected = osm._expected_scan_hashes(source)["scan"]
    osm.verify_hash(scan, expected, "scan")
    scan.write_bytes(b"changed scan")
    with pytest.raises(ValueError, match="scan SHA-256 mismatch"):
        osm.verify_hash(scan, expected, "scan")
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="source manifest SHA-256 mismatch"):
        diagnostic._load_records(config, tmp_path, tmp_path, ())
