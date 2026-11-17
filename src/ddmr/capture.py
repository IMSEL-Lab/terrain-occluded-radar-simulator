"""Furuno spoke capture parsing, segmentation, and polar gridding.

The CSV stores five scalar fields followed by a variable-length echo vector.
Raw, well-formed rows remain available even when quality rules make them
ineligible for segmentation or gridding.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

TICKS_PER_REVOLUTION = 8192


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Parsing and canonical-grid settings.

    ``range_mapping_nm`` is deliberately opt-in. It maps a raw Range code to
    a verified full-range value in nautical miles for the specific exporter.
    Without it, grid data has no physical radial calibration.
    """

    expected_echo_bins: int | None = 868
    azimuth_bins: int = TICKS_PER_REVOLUTION
    radial_bins: int | None = None
    accepted_statuses: tuple[int, ...] = (1,)
    range_mapping_nm: Mapping[int, float] | None = None
    max_consecutive_same_angle: int = 64
    min_wrap_drop_ticks: int = TICKS_PER_REVOLUTION // 2
    split_on_range_change: bool = True
    split_on_gain_change: bool = True
    split_on_status_change: bool = True
    duplicate_angle_policy: Literal["last"] = "last"

    def __post_init__(self) -> None:
        if self.expected_echo_bins is not None and self.expected_echo_bins < 1:
            raise ValueError("expected_echo_bins must be positive or None")
        if self.azimuth_bins < 1:
            raise ValueError("azimuth_bins must be positive")
        if self.radial_bins is not None and self.radial_bins < 1:
            raise ValueError("radial_bins must be positive or None")
        if self.max_consecutive_same_angle < 2:
            raise ValueError("max_consecutive_same_angle must be at least 2")
        if not 0 < self.min_wrap_drop_ticks <= TICKS_PER_REVOLUTION:
            raise ValueError("min_wrap_drop_ticks must be in [1, 8192]")
        if self.duplicate_angle_policy != "last":
            raise ValueError("only the explicit 'last' duplicate policy is supported")
        if self.range_mapping_nm is not None:
            for code, value in self.range_mapping_nm.items():
                if not isinstance(code, Integral) or isinstance(code, bool) or code < 0:
                    raise ValueError("range mapping keys must be nonnegative integers")
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("mapped ranges must be finite and positive")


@dataclass(frozen=True, slots=True)
class Spoke:
    """One parsed source row, including its unmodified acquisition metadata."""

    row_number: int
    status: int
    scale: int
    range_code: int
    gain: int
    angle_ticks: int
    echoes: tuple[int, ...]
    usable: bool = True
    quality_flags: tuple[str, ...] = ()


@dataclass(slots=True)
class CaptureDiagnostics:
    rows_seen: int = 0
    rows_parsed: int = 0
    malformed_rows: int = 0
    malformed_reasons: dict[str, int] = field(default_factory=dict)
    exact_duplicate_rows: int = 0
    repeated_angle_rows: int = 0
    stalled_angle_blocks: int = 0
    stalled_rows: int = 0
    excluded_status_rows: int = 0

    def reject(self, reason: str) -> None:
        self.malformed_rows += 1
        self.malformed_reasons[reason] = self.malformed_reasons.get(reason, 0) + 1


@dataclass(frozen=True, slots=True)
class Capture:
    source: str
    spokes: tuple[Spoke, ...]
    diagnostics: CaptureDiagnostics
    echo_bin_count: int


@dataclass(frozen=True, slots=True)
class Rotation:
    index: int
    spokes: tuple[Spoke, ...]
    start_row: int
    end_row: int
    boundary_before: str

    @property
    def status(self) -> int:
        return self.spokes[0].status

    @property
    def range_code(self) -> int:
        return self.spokes[0].range_code

    @property
    def gain(self) -> int:
        return self.spokes[0].gain


@dataclass(frozen=True, slots=True)
class PolarGrid:
    """A canonical array and an angular mask marking actually observed bins."""

    echoes: NDArray[np.float32]
    angular_valid: NDArray[np.bool_]
    source_spoke_count: NDArray[np.int32]
    range_code: int
    status: int
    gain: int
    range_max_m: float | None


def _is_header(row: Sequence[str]) -> bool:
    if not row:
        return False
    normalized = [cell.strip().lower() for cell in row[:5]]
    return normalized == ["status", "scale", "range", "gain", "angle"]


def _parse_int(value: str) -> int:
    # Accept integer-valued CSV floats (for example "3.0") without silently
    # truncating genuinely fractional metadata or echo samples.
    try:
        return int(value.strip())
    except ValueError:
        pass
    number = float(value.strip())
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError("not_integer")
    return int(number)


def _row_key(spoke: Spoke) -> tuple[object, ...]:
    return (
        spoke.status,
        spoke.scale,
        spoke.range_code,
        spoke.gain,
        spoke.angle_ticks,
        spoke.echoes,
    )


def _mark_stalled_blocks(
    spokes: list[Spoke], threshold: int, diagnostics: CaptureDiagnostics
) -> list[Spoke]:
    result = list(spokes)
    start = 0
    while start < len(result):
        end = start + 1
        angle = result[start].angle_ticks
        while end < len(result) and result[end].angle_ticks == angle:
            end += 1
        run_length = end - start
        if run_length > threshold:
            diagnostics.stalled_angle_blocks += 1
            diagnostics.stalled_rows += run_length
            for i in range(start, end):
                old = result[i]
                result[i] = Spoke(
                    row_number=old.row_number,
                    status=old.status,
                    scale=old.scale,
                    range_code=old.range_code,
                    gain=old.gain,
                    angle_ticks=old.angle_ticks,
                    echoes=old.echoes,
                    usable=False,
                    quality_flags=(*old.quality_flags, "stalled_angle_block"),
                )
        start = end
    return result


def load_capture(path: str | Path, config: CaptureConfig | None = None) -> Capture:
    """Read a Furuno CSV while retaining every valid row and counting rejects."""

    cfg = config or CaptureConfig()
    source = Path(path)
    diagnostics = CaptureDiagnostics()
    spokes: list[Spoke] = []
    expected_bins = cfg.expected_echo_bins
    prior_row_key: tuple[object, ...] | None = None
    previous_angle: int | None = None

    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        for row_number, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if diagnostics.rows_seen == 0 and _is_header(row):
                continue
            diagnostics.rows_seen += 1
            if len(row) < 6:
                diagnostics.reject("too_few_fields")
                continue
            try:
                status, scale, range_code, gain, angle = (
                    _parse_int(cell) for cell in row[:5]
                )
                echoes = tuple(_parse_int(cell) for cell in row[5:])
            except ValueError as exc:
                diagnostics.reject(
                    str(exc) if str(exc) == "not_integer" else "invalid_number"
                )
                continue
            if not echoes:
                diagnostics.reject("empty_echo_payload")
                continue
            if any(value < 0 or value > 255 for value in echoes):
                diagnostics.reject("echo_value_out_of_range")
                continue
            if expected_bins is None:
                expected_bins = len(echoes)
            if len(echoes) != expected_bins:
                diagnostics.reject("echo_bin_count_mismatch")
                continue
            if angle < 0 or angle >= TICKS_PER_REVOLUTION:
                diagnostics.reject("angle_out_of_range")
                continue
            if status < 0 or scale < 0 or range_code < 0 or gain < 0:
                diagnostics.reject("negative_metadata")
                continue
            spoke = Spoke(
                row_number=row_number,
                status=status,
                scale=scale,
                range_code=range_code,
                gain=gain,
                angle_ticks=angle,
                echoes=echoes,
            )
            key = _row_key(spoke)
            if key == prior_row_key:
                diagnostics.exact_duplicate_rows += 1
            if previous_angle == angle:
                diagnostics.repeated_angle_rows += 1
            prior_row_key = key
            previous_angle = angle
            if status not in cfg.accepted_statuses:
                diagnostics.excluded_status_rows += 1
            spokes.append(spoke)
            diagnostics.rows_parsed += 1

    spokes = _mark_stalled_blocks(spokes, cfg.max_consecutive_same_angle, diagnostics)
    return Capture(
        source=str(source),
        spokes=tuple(spokes),
        diagnostics=diagnostics,
        echo_bin_count=expected_bins or 0,
    )


def _metadata_change(
    previous: Spoke, current: Spoke, config: CaptureConfig
) -> str | None:
    changes: list[str] = []
    if config.split_on_range_change and previous.range_code != current.range_code:
        changes.append("range_change")
    if config.split_on_gain_change and previous.gain != current.gain:
        changes.append("gain_change")
    if config.split_on_status_change and previous.status != current.status:
        changes.append("status_change")
    return "+".join(changes) if changes else None


def segment_rotations(
    capture: Capture, config: CaptureConfig | None = None
) -> tuple[Rotation, ...]:
    """Split on substantial falling-angle wraps and configured metadata changes."""

    cfg = config or CaptureConfig()
    usable = [spoke for spoke in capture.spokes if spoke.usable]
    if not usable:
        return ()
    groups: list[tuple[str, list[Spoke]]] = []
    current: list[Spoke] = []
    boundary = "capture_start"
    previous: Spoke | None = None
    for spoke in usable:
        next_boundary: str | None = None
        if previous is not None:
            change = _metadata_change(previous, spoke, cfg)
            if change:
                next_boundary = change
            elif previous.angle_ticks - spoke.angle_ticks >= cfg.min_wrap_drop_ticks:
                next_boundary = "angular_wrap"
        if next_boundary and current:
            groups.append((boundary, current))
            current = []
            boundary = next_boundary
        current.append(spoke)
        previous = spoke
    if current:
        groups.append((boundary, current))
    return tuple(
        Rotation(
            index=index,
            spokes=tuple(group),
            start_row=group[0].row_number,
            end_row=group[-1].row_number,
            boundary_before=why,
        )
        for index, (why, group) in enumerate(groups)
    )


def _range_max_m(rotation: Rotation, config: CaptureConfig) -> float | None:
    if config.range_mapping_nm is None:
        return None
    nautical_miles = config.range_mapping_nm.get(rotation.range_code)
    return None if nautical_miles is None else float(nautical_miles * 1852.0)


def grid_rotation(rotation: Rotation, config: CaptureConfig | None = None) -> PolarGrid:
    """Grid selected spokes without representing missing azimuths as zero data.

    The canonical shape is ``(azimuth_bins, radial_bins)``. Missing azimuth
    rows contain NaN and have ``angular_valid=False``. If quantized angles map
    to one bin more than once, the last usable spoke wins; source counts retain
    the multiplicity for audit.
    """

    cfg = config or CaptureConfig()
    selected = [
        spoke
        for spoke in rotation.spokes
        if spoke.usable and spoke.status in cfg.accepted_statuses
    ]
    radial_bins = cfg.radial_bins or (len(selected[0].echoes) if selected else 0)
    if radial_bins < 1:
        raise ValueError("rotation has no eligible spokes to grid")
    grid = np.full((cfg.azimuth_bins, radial_bins), np.nan, dtype=np.float32)
    valid = np.zeros(cfg.azimuth_bins, dtype=np.bool_)
    counts = np.zeros(cfg.azimuth_bins, dtype=np.int32)
    for spoke in selected:
        azimuth = round(spoke.angle_ticks * cfg.azimuth_bins / TICKS_PER_REVOLUTION)
        azimuth %= cfg.azimuth_bins
        samples = np.asarray(spoke.echoes, dtype=np.float32)
        if len(samples) != radial_bins:
            old_x = np.linspace(0.0, 1.0, len(samples), dtype=np.float64)
            new_x = np.linspace(0.0, 1.0, radial_bins, dtype=np.float64)
            samples = np.interp(new_x, old_x, samples).astype(np.float32)
        # Iteration is in source order, so assignment implements last-record-wins.
        grid[azimuth] = samples
        counts[azimuth] += 1
        valid[azimuth] = True
    return PolarGrid(
        echoes=grid,
        angular_valid=valid,
        source_spoke_count=counts,
        range_code=rotation.range_code,
        status=rotation.status,
        gain=rotation.gain,
        range_max_m=_range_max_m(rotation, cfg),
    )


def save_capture_npz(path: str | Path, capture: Capture) -> None:
    """Save all parsed rows, metadata, and quality flags in a compressed NPZ."""

    width = capture.echo_bin_count
    echoes = np.asarray([spoke.echoes for spoke in capture.spokes], dtype=np.int32)
    if not capture.spokes:
        echoes = np.empty((0, width), dtype=np.int32)
    np.savez_compressed(
        path,
        row_number=np.asarray([s.row_number for s in capture.spokes], dtype=np.int64),
        status=np.asarray([s.status for s in capture.spokes], dtype=np.int32),
        scale=np.asarray([s.scale for s in capture.spokes], dtype=np.int32),
        range_code=np.asarray([s.range_code for s in capture.spokes], dtype=np.int32),
        gain=np.asarray([s.gain for s in capture.spokes], dtype=np.int32),
        angle_ticks=np.asarray([s.angle_ticks for s in capture.spokes], dtype=np.int32),
        echoes=echoes,
        usable=np.asarray([s.usable for s in capture.spokes], dtype=np.bool_),
    )


def save_grid_npz(path: str | Path, grid: PolarGrid) -> None:
    """Save gridded echoes and validity/count arrays in compressed NPZ format."""

    np.savez_compressed(
        path,
        echoes=grid.echoes,
        angular_valid=grid.angular_valid,
        source_spoke_count=grid.source_spoke_count,
        range_code=np.asarray(grid.range_code),
        status=np.asarray(grid.status),
        gain=np.asarray(grid.gain),
        range_max_m=np.asarray(
            np.nan if grid.range_max_m is None else grid.range_max_m
        ),
    )


def save_capture_summary_json(path: str | Path, capture: Capture) -> None:
    """Write a compact, machine-readable capture summary and diagnostics."""

    summary = {
        "source": capture.source,
        "echo_bin_count": capture.echo_bin_count,
        "spoke_count": len(capture.spokes),
        "usable_spoke_count": sum(spoke.usable for spoke in capture.spokes),
        "status_counts": _counts(spoke.status for spoke in capture.spokes),
        "range_code_counts": _counts(spoke.range_code for spoke in capture.spokes),
        "gain_counts": _counts(spoke.gain for spoke in capture.spokes),
        "diagnostics": asdict(capture.diagnostics),
    }
    Path(path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _counts(values: Iterable[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts
