"""Run the frozen, source-conditioned Furuno reconstruction diagnostic.

The diagnostic reuses a selected capture as scene input. Its scores measure
reconstruction of supplied-source content and are not independent validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ddmr import capture
from ddmr.capture import CaptureConfig, grid_rotation, load_capture, segment_rotations


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    """Source hashes remain authoritative for archives without Git metadata."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def score_pair(
    source: np.ndarray,
    rendered: np.ndarray,
    source_valid: np.ndarray,
    rendered_valid: np.ndarray,
    divisor: float = 252.0,
) -> dict[str, int | float | None]:
    """Score every source-observed azimuth; missing rendered rows fail closed."""
    if divisor != 252.0:
        raise ValueError("normalized display scores use the fixed divisor 252")
    if source.ndim != 2 or rendered.ndim != 2 or source.shape != rendered.shape:
        raise ValueError("source and rendered grids must have identical 2-D shapes")
    if (
        source_valid.shape != source.shape[:1]
        or rendered_valid.shape != source.shape[:1]
    ):
        raise ValueError("valid-angle masks must match the grid azimuth dimension")
    if not np.any(source_valid):
        raise ValueError("source has no observed angles")
    if np.any(source_valid & ~rendered_valid):
        missing = int(np.count_nonzero(source_valid & ~rendered_valid))
        raise ValueError(f"render is missing {missing} source-observed angles")
    if np.any(~np.isfinite(source[source_valid])):
        raise ValueError("source contains nonfinite values at observed angles")
    if np.any(~np.isfinite(rendered[source_valid])):
        raise ValueError("render contains nonfinite values at source-observed angles")
    paired = np.broadcast_to(source_valid[:, None], source.shape).copy()
    source_values = source[paired].astype(np.float64)
    rendered_values = rendered[paired].astype(np.float64)
    if (
        np.any(source_values < 0)
        or np.any(source_values > 252)
        or np.any(rendered_values < 0)
        or np.any(rendered_values > 252)
    ):
        raise ValueError(
            "display echo values must remain in the verified [0, 252] code range"
        )
    absolute_error = np.abs(rendered_values - source_values)
    positive_source = source_values > 0
    return {
        "common_observed_angles": int(source_valid.sum()),
        "paired_valid_bins": int(paired.sum()),
        "source_positive_bins": int(positive_source.sum()),
        "rendered_positive_bins": int((rendered_values > 0).sum()),
        "source_positive_fraction": float(positive_source.mean()),
        "rendered_positive_fraction": float((rendered_values > 0).mean()),
        "normalized_mae": float(absolute_error.mean() / divisor),
        "source_positive_normalized_mae": (
            float(absolute_error[positive_source].mean() / divisor)
            if np.any(positive_source)
            else None
        ),
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol") != "furuno-source-conditioned-diagnostic-v1":
        raise ValueError("unsupported or modified diagnostic protocol")
    if float(config.get("display_divisor", 0)) != 252.0:
        raise ValueError("the verified simulator display divisor is fixed at 252")
    return config


def _canonicalize_rotation(
    raw_path: Path, canonical_path: Path, rotation_index: int, bins: int
) -> tuple[Any, dict[str, Any], list[dict[str, int]]]:
    """Write precisely the parser-selected usable rotation and verify its round trip."""
    raw_capture = load_capture(
        raw_path, CaptureConfig(expected_echo_bins=bins, radial_bins=bins)
    )
    rotations = segment_rotations(raw_capture)
    if not 0 <= rotation_index < len(rotations):
        raise ValueError(
            f"rotation {rotation_index} unavailable in {raw_path}; found {len(rotations)}"
        )
    rotation = rotations[rotation_index]
    rows: list[dict[str, int]] = []
    with canonical_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Status", "Scale", "Range", "Gain", "Angle", "EchoValues"])
        for output_row, spoke in enumerate(rotation.spokes, start=2):
            writer.writerow(
                [
                    spoke.status,
                    spoke.scale,
                    spoke.range_code,
                    spoke.gain,
                    spoke.angle_ticks,
                    *spoke.echoes,
                ]
            )
            rows.append(
                {
                    "output_row": output_row,
                    "source_row": spoke.row_number,
                    "angle_ticks": spoke.angle_ticks,
                }
            )
    canonical_capture = load_capture(
        canonical_path, CaptureConfig(expected_echo_bins=bins, radial_bins=bins)
    )
    canonical_rotations = segment_rotations(canonical_capture)
    if len(canonical_rotations) != 1:
        raise ValueError(
            f"canonical rotation must parse as one segment, found {len(canonical_rotations)}"
        )
    canonical_rotation = canonical_rotations[0]
    if len(canonical_rotation.spokes) != len(rotation.spokes):
        raise ValueError("canonical rotation changed usable spoke count")
    for original, copied in zip(
        rotation.spokes, canonical_rotation.spokes, strict=True
    ):
        if (
            original.status,
            original.scale,
            original.range_code,
            original.gain,
            original.angle_ticks,
            original.echoes,
        ) != (
            copied.status,
            copied.scale,
            copied.range_code,
            copied.gain,
            copied.angle_ticks,
            copied.echoes,
        ):
            raise ValueError(
                "canonical rotation did not preserve source metadata and echoes exactly"
            )
    grid = grid_rotation(
        canonical_rotation, CaptureConfig(expected_echo_bins=bins, radial_bins=bins)
    )
    values = grid.echoes[np.isfinite(grid.echoes)]
    if values.size == 0 or np.any(values < 0) or np.any(values > 252):
        raise ValueError(
            "source echoes are empty or outside the [0, 252] display-code range"
        )
    info = {
        "raw_rows_seen": raw_capture.diagnostics.rows_seen,
        "raw_rows_parsed": raw_capture.diagnostics.rows_parsed,
        "raw_malformed_rows": raw_capture.diagnostics.malformed_rows,
        "raw_exact_duplicate_rows": raw_capture.diagnostics.exact_duplicate_rows,
        "raw_repeated_angle_rows": raw_capture.diagnostics.repeated_angle_rows,
        "raw_stalled_angle_blocks": raw_capture.diagnostics.stalled_angle_blocks,
        "selected_rotation_index": rotation_index,
        "selected_rotation_boundary": rotation.boundary_before,
        "selected_source_rows": len(rotation.spokes),
        "selected_rotation_start_row": rotation.start_row,
        "selected_rotation_end_row": rotation.end_row,
        "canonical_rotation_count": len(canonical_rotations),
        "canonical_echo_roundtrip": "exact metadata, angle, and echo equality",
        "range_code": rotation.range_code,
        "gain": rotation.gain,
        "observed_angle_bins": int(grid.angular_valid.sum()),
        "positive_source_bins": int(np.count_nonzero(values > 0)),
        "source_positive_fraction_on_observed_angles": float(
            np.count_nonzero(values > 0) / values.size
        ),
        "echo_value_min": int(values.min()),
        "echo_value_max": int(values.max()),
    }
    return grid, info, rows


def _run(command: list[str], env: dict[str, str], log_path: Path) -> float:
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.monotonic() - started


def _render_csv(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*.csv"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one rendered radar CSV in {output_dir}, found {len(candidates)}"
        )
    return candidates[0]


def _save_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    target = output_dir / "provenance.json"
    temporary = output_dir / "provenance.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def _write_metrics_table(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    path = output_dir / "metrics_table.csv"
    fields = [
        "stratum",
        "mode",
        "seed",
        "render_csv",
        "common_observed_angles",
        "paired_valid_bins",
        "source_positive_bins",
        "rendered_positive_bins",
        "source_positive_fraction",
        "rendered_positive_fraction",
        "normalized_mae",
        "source_positive_normalized_mae",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _guarded_simulator_command(command: list[str]) -> list[str]:
    """Run the simulator after installing a child-process pickle load trap."""
    guard = (
        "import pickle, runpy, sys; "
        "def_block=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('pickle loading forbidden in E2 run')); "
        "pickle.load=def_block; pickle.loads=def_block; "
        "class_block=type('BlockedUnpickler',(),{'__init__':def_block,'load':def_block,'loads':def_block}); "
        "pickle.Unpickler=class_block; "
        "script=sys.argv[1]; sys.argv=[script,*sys.argv[2:]]; runpy.run_path(script,run_name='__main__')"
    )
    return [command[0], "-c", guard, *command[1:]]


def run(
    config_path: Path, output_dir: Path, source_root: Path | None, smoke_only: bool
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    config = _load_config(config_path)
    if source_root is None:
        raise ValueError("provide --source-root for the permitted capture files")
    source_base = source_root.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    bins = int(config["range_bins"])
    seed_values = [int(value) for value in config["seed_values"]]
    simulations = list(config["simulations"])
    if smoke_only:
        seed_values, simulations = seed_values[:1], simulations[:1]
    captures = config["captures"][:1] if smoke_only else config["captures"]

    env = os.environ.copy()
    env.update(
        {
            "MODE4_USE_OBJECT_LEARNED_MODEL": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    code_files = [
        REPO_ROOT / name
        for name in (
            "extract_scene.py",
            "radar_simulator.py",
            "object_renderer.py",
            "scripts/run_furuno_conditioned_diagnostic.py",
            "uv.lock",
        )
    ]
    manifest: dict[str, Any] = {
        "protocol": config["protocol"],
        "run_status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_root": str(source_base),
        "code_sha256": {
            **{
                str(path.relative_to(REPO_ROOT)): sha256_file(path)
                for path in code_files
            },
            "ddmr/capture.py": sha256_file(Path(capture.__file__)),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "environment_controls": {
            key: env[key]
            for key in (
                "MODE4_USE_OBJECT_LEARNED_MODEL",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "assumptions": {
            "range_profile": config["range_profile"],
            "radial_mapping": config["radial_mapping"],
            "physical_range_calibration": None,
            "claim_scope": "source-conditioned reconstruction diagnostic; not independent physical validation",
        },
        "smoke_only": smoke_only,
        "runner_command": [sys.executable, *sys.argv],
        "runs": [],
        "metrics_table": [],
    }
    _save_manifest(output_dir, manifest)

    try:
        for capture_spec in captures:
            relative = Path(capture_spec["relative_path"])
            source_path = (source_base / relative).resolve()
            if source_base not in source_path.parents:
                raise ValueError(f"capture path escapes source root: {relative}")
            actual_hash = sha256_file(source_path)
            if actual_hash != capture_spec["sha256"]:
                raise ValueError(
                    f"source SHA-256 mismatch for {source_path}: {actual_hash}"
                )
            capture_dir = output_dir / capture_spec["stratum"]
            capture_dir.mkdir()
            canonical_path = capture_dir / "source_rotation.csv"
            source_grid, capture_info, row_provenance = _canonicalize_rotation(
                source_path, canonical_path, int(capture_spec["rotation_index"]), bins
            )
            if len(row_provenance) != int(capture_spec["expected_rows"]):
                raise ValueError(
                    f"selected rotation row count changed for {source_path}: "
                    f"expected {capture_spec['expected_rows']}, got {len(row_provenance)}"
                )
            scene_path = capture_dir / "source_scene.json"
            extraction_command = [
                sys.executable,
                str(REPO_ROOT / "extract_scene.py"),
                str(canonical_path),
                "--output",
                str(scene_path),
                "--rotation",
                "0",
                "--range-profile",
                config["range_profile"],
                "--assume-linear-full-range",
            ]
            extract_sec = _run(extraction_command, env, capture_dir / "extract.log")
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            expected_angles = [row["angle_ticks"] for row in row_provenance]
            if scene.get("angle_ticks") != expected_angles:
                raise ValueError(
                    "extracted scene angle sequence differs from canonical selected rotation"
                )
            capture_manifest: dict[str, Any] = {
                "stratum": capture_spec["stratum"],
                "selection_reason": capture_spec["selection_reason"],
                "relative_path": relative.as_posix(),
                "source_path": str(source_path),
                "source_sha256": actual_hash,
                "source_bytes": source_path.stat().st_size,
                "canonical_rotation_path": str(canonical_path),
                "canonical_rotation_sha256": sha256_file(canonical_path),
                "canonical_rotation_bytes": canonical_path.stat().st_size,
                "row_provenance": row_provenance,
                "audit_coverage_bins_of_400": capture_spec[
                    "audit_coverage_bins_of_400"
                ],
                "source_capture_diagnostics": capture_info,
                "extracted_scene_angles_exact": True,
                "source_scene_sha256": sha256_file(scene_path),
                "extract_command": extraction_command,
                "extract_wall_seconds": extract_sec,
                "renders": [],
                "zero_baseline": {},
            }
            manifest["runs"].append(capture_manifest)
            source_zero = np.zeros_like(source_grid.echoes)
            zero_metrics = score_pair(
                source_grid.echoes,
                source_zero,
                source_grid.angular_valid,
                source_grid.angular_valid,
                float(config["display_divisor"]),
            )
            capture_manifest["zero_baseline"] = zero_metrics
            manifest["metrics_table"].append(
                {
                    "stratum": capture_spec["stratum"],
                    "mode": "zero_baseline",
                    "seed": "",
                    "render_csv": "",
                    **zero_metrics,
                }
            )
            _write_metrics_table(output_dir, manifest["metrics_table"])
            _save_manifest(output_dir, manifest)
            for seed in seed_values:
                for mode in simulations:
                    run_dir = capture_dir / f"seed_{seed:02d}" / mode
                    run_dir.mkdir(parents=True)
                    command = [
                        sys.executable,
                        str(REPO_ROOT / "radar_simulator.py"),
                        "--scene-file",
                        str(scene_path),
                        "--sim-mode",
                        mode,
                        "--output",
                        str(run_dir),
                        "--rotations",
                        "1",
                        "--quiet",
                    ]
                    child_env = env | {"SIM_RANDOM_SEED": str(seed)}
                    guarded_command = _guarded_simulator_command(command)
                    elapsed = _run(guarded_command, child_env, run_dir / "render.log")
                    rendered_path = _render_csv(run_dir)
                    rendered_capture = load_capture(
                        rendered_path,
                        CaptureConfig(expected_echo_bins=bins, radial_bins=bins),
                    )
                    rendered_rotations = segment_rotations(rendered_capture)
                    if not rendered_rotations:
                        raise ValueError(
                            f"render has no usable rotation: {rendered_path}"
                        )
                    rendered_grid = grid_rotation(
                        rendered_rotations[0],
                        CaptureConfig(expected_echo_bins=bins, radial_bins=bins),
                    )
                    rendered_values = rendered_grid.echoes[
                        np.isfinite(rendered_grid.echoes)
                    ]
                    if (
                        rendered_values.size == 0
                        or np.any(rendered_values < 0)
                        or np.any(rendered_values > 252)
                    ):
                        raise ValueError(
                            f"rendered echoes fall outside the [0, 252] display-code range: {rendered_path}"
                        )
                    metrics = score_pair(
                        source_grid.echoes,
                        rendered_grid.echoes,
                        source_grid.angular_valid,
                        rendered_grid.angular_valid,
                        float(config["display_divisor"]),
                    )
                    capture_manifest["renders"].append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "render_csv": str(rendered_path),
                            "render_sha256": sha256_file(rendered_path),
                            "render_bytes": rendered_path.stat().st_size,
                            "wall_seconds": elapsed,
                            "render_command": guarded_command,
                            "pickle_load_guard": "pickle.load, pickle.loads, and pickle.Unpickler blocked in child process",
                            "metrics": metrics,
                        }
                    )
                    manifest["metrics_table"].append(
                        {
                            "stratum": capture_spec["stratum"],
                            "mode": mode,
                            "seed": seed,
                            "render_csv": str(rendered_path),
                            **metrics,
                        }
                    )
                    _write_metrics_table(output_dir, manifest["metrics_table"])
                    _save_manifest(output_dir, manifest)
            if sha256_file(source_path) != actual_hash:
                raise ValueError(f"source changed during run: {source_path}")
        manifest["run_status"] = "complete"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _save_manifest(output_dir, manifest)
        return manifest
    except Exception as error:
        manifest["run_status"] = "failed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        _save_manifest(output_dir, manifest)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/furuno-folly-beach.json",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="root directory of the capture files",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="run only the first capture, seed 0, current_mode4",
    )
    args = parser.parse_args(argv)
    result = run(args.config, args.output_dir, args.source_root, args.smoke_only)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "runs": len(result["runs"]),
                "smoke_only": args.smoke_only,
            }
        )
    )


if __name__ == "__main__":
    main()
