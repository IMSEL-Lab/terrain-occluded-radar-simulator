"""Regression checks for the installed public workflows."""

import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import radar_simulator as engine
from tors.cli import main
from tors.map_scene import _scene

ROOT = Path(__file__).resolve().parents[1]


def test_version_through_public_entrypoint(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "1.0.1"


def test_bundled_template_survives_generator_filters():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(42)
        coastlines, _ = engine.generate_coastlines_from_scene_template(
            [str(ROOT / "examples/synthetic-template.json")], engine.RANGE_M
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    assert len(coastlines) >= 3
    assert all(len(coast.points) >= 3 for coast in coastlines)


def test_map_scan_configuration_rejects_bad_seed_and_excludes_capture_payload(tmp_path):
    path = tmp_path / "grid.json"
    path.write_text(
        json.dumps(
            {
                "radar_config": {"bins": 2, "range_m": 1.0},
                "angle_ticks": [0, 4096],
                "_sim_seed": 42,
                "objects": [{"type": "boat"}],
                "noise_profiles": [[250, 250]],
                "_mode42": True,
            }
        )
    )
    args = Namespace(grid_scene=path, seed=None)
    scene = _scene(args)
    assert scene["angle_ticks"] == [0, 4096]
    assert scene["objects"] == []
    assert "noise_profiles" not in scene and "_mode42" not in scene
    args.seed = -1
    with pytest.raises(ValueError, match="seed"):
        _scene(args)
