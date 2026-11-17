"""Data export functionality for radar returns.

Supports two CSV formats:
1. radar_plotter format (default) - Compatible with the Rust radar_plotter visualizer
2. detailed format - Full metadata with labeled columns
"""
import csv
import os
import shutil
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass
from radar_core import rust_bridge

ANGLE_TICKS_PER_DEGREE = 8192.0 / 360.0
NM_TO_METERS = 1852.0
FURUNO_STATUS = 1
FURUNO_SCALE = 496
FURUNO_TICKS_PER_REV = 8192
FURUNO_DEAD_ZONE_BINS = 31
FURUNO_DEFAULT_SPOKES_PER_ROTATION = 1820
FURUNO_RANGE_SCALES_NM = [0.0625, 0.125, 0.25, 0.5, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0]
FURUNO_RANGE_INDEX_OFFSET = 0
FURUNO_GAIN_CODES = [0, 8, 20, 28, 36, 44, 56, 64, 76, 88, 104, 120, 140, 164, 188, 220, 252]


@dataclass
class RadarSweepRecord:
    timestamp: float
    bearing_deg: float
    range_scale_nm: float
    gain: float
    sea_clutter: float
    rain_clutter: float
    num_bins: int
    echo_values: List[float]


class RadarDataExporter:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.getcwd()
        self.mirror_dir: Optional[str] = None
        self.records: List[RadarSweepRecord] = []
        self.is_recording = False
        self.session_id: Optional[str] = None
        self.max_records = 36000
        os.makedirs(self.output_dir, exist_ok=True)

    def start_recording(self) -> str:
        self.records.clear()
        self.is_recording = True
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.session_id

    def stop_recording(self) -> str:
        self.is_recording = False
        if self.records:
            return self.save_to_csv()
        return ""

    def add_sweep(self, timestamp: float, bearing_deg: float,
                  range_scale_nm: float, gain: float,
                  sea_clutter: float, rain_clutter: float,
                  echo_values: List[float]) -> None:
        if not self.is_recording:
            return
        if len(self.records) >= self.max_records:
            self.save_to_csv()
            self.records.clear()
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        record = RadarSweepRecord(
            timestamp=timestamp, bearing_deg=bearing_deg,
            range_scale_nm=range_scale_nm, gain=gain,
            sea_clutter=sea_clutter, rain_clutter=rain_clutter,
            num_bins=len(echo_values), echo_values=echo_values.copy())
        self.records.append(record)

    def save_to_csv(self, filename: str = None, format: str = "radar_plotter") -> str:
        if not self.records:
            return ""
        if filename is None:
            filename = f"radar_data_{self.session_id}.csv"
        filepath = os.path.join(self.output_dir, filename)
        if format == "radar_plotter":
            return self._save_radar_plotter_format(filepath)
        else:
            return self._save_detailed_format(filepath)

    def _save_radar_plotter_format(self, filepath: str) -> str:
        num_bins = self.records[0].num_bins
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp', 'unused', 'range_m', 'gain_code', 'angle_ticks']
            for i in range(num_bins):
                header.append(f'bin_{i}')
            writer.writerow(header)
            for record in self.records:
                angle_ticks = record.bearing_deg * ANGLE_TICKS_PER_DEGREE
                gain_code = int(record.gain * 255)
                range_m = int(record.range_scale_nm * NM_TO_METERS)
                row = [f"{record.timestamp:.3f}", 0, range_m, gain_code,
                       f"{angle_ticks:.2f}"]
                for val in record.echo_values:
                    # Clamp noise floor to zero so visualizer doesn't
                    # render empty bins as faint returns
                    clamped = val if val >= 0.03 else 0.0
                    row.append(f"{clamped:.4f}")
                writer.writerow(row)
        self._mirror_file(filepath)
        return filepath

    def _save_detailed_format(self, filepath: str) -> str:
        num_bins = self.records[0].num_bins
        range_scale = self.records[0].range_scale_nm
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp_s', 'bearing_deg', 'range_scale_nm',
                      'gain', 'sea_clutter', 'rain_clutter']
            bin_size_nm = range_scale / num_bins
            for i in range(num_bins):
                range_nm = (i + 0.5) * bin_size_nm
                header.append(f'echo_{range_nm:.3f}nm')
            writer.writerow(header)
            for record in self.records:
                row = [f"{record.timestamp:.3f}", f"{record.bearing_deg:.1f}",
                       f"{record.range_scale_nm:.2f}", f"{record.gain:.3f}",
                       f"{record.sea_clutter:.3f}", f"{record.rain_clutter:.3f}"]
                for val in record.echo_values:
                    clamped = val if val >= 0.03 else 0.0
                    row.append(f"{clamped:.4f}")
                writer.writerow(row)
        self._mirror_file(filepath)
        return filepath

    def export_single_sweep(self, timestamp: float, bearing_deg: float,
                           range_scale_nm: float, gain: float,
                           sea_clutter: float, rain_clutter: float,
                           echo_values: List[float],
                           filename: str = None,
                           format: str = "radar_plotter") -> str:
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sweep_{ts}_bearing{int(bearing_deg)}.csv"
        filepath = os.path.join(self.output_dir, filename)
        num_bins = len(echo_values)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp', 'unused', 'range_m', 'gain_code', 'angle_ticks']
            for i in range(num_bins):
                header.append(f'bin_{i}')
            writer.writerow(header)
            angle_ticks = bearing_deg * ANGLE_TICKS_PER_DEGREE
            gain_code = int(gain * 255)
            range_m = int(range_scale_nm * NM_TO_METERS)
            row = [f"{timestamp:.3f}", 0, range_m, gain_code, f"{angle_ticks:.2f}"]
            for val in echo_values:
                clamped = val if val >= 0.03 else 0.0
                row.append(f"{clamped:.4f}")
            writer.writerow(row)
        self._mirror_file(filepath)
        return filepath

    def _mirror_file(self, filepath: str) -> None:
        """Copy file to mirror directory if configured."""
        if self.mirror_dir and filepath:
            try:
                os.makedirs(self.mirror_dir, exist_ok=True)
                shutil.copy2(filepath, self.mirror_dir)
            except OSError:
                pass

    def furuno_range_index_for_nm(self, range_scale_nm: float) -> int:
        """Map range scale to the direct Furuno CSV range code."""
        closest_idx = min(
            range(len(FURUNO_RANGE_SCALES_NM)),
            key=lambda idx: abs(FURUNO_RANGE_SCALES_NM[idx] - range_scale_nm),
        )
        return FURUNO_RANGE_INDEX_OFFSET + closest_idx

    def generate_furuno_angle_ticks(
        self, spokes_per_rotation: int = FURUNO_DEFAULT_SPOKES_PER_ROTATION
    ) -> List[int]:
        """Generate monotonic Furuno-style angle ticks for one rotation."""
        if rust_bridge.rust_available():
            try:
                return [
                    int(value)
                    for value in rust_bridge.generate_furuno_angle_ticks(
                        spokes_per_rotation)
                ]
            except Exception:
                pass
        spokes = max(1, int(spokes_per_rotation))
        ticks: List[int] = []
        for i in range(spokes):
            tick = int(round(i * FURUNO_TICKS_PER_REV / spokes))
            tick = max(0, min(FURUNO_TICKS_PER_REV - 1, tick))
            if ticks and tick <= ticks[-1]:
                tick = min(FURUNO_TICKS_PER_REV - 1, ticks[-1] + 1)
            ticks.append(tick)
        return ticks

    def quantize_sweep_to_furuno_codes(
        self, echo_values: List[float], dead_zone_bins: int = FURUNO_DEAD_ZONE_BINS
    ) -> List[int]:
        """Convert normalized sweep values to nearest Furuno display codes."""
        if rust_bridge.rust_available():
            try:
                return [
                    int(value)
                    for value in rust_bridge.quantize_sweep_to_furuno_codes(
                        echo_values, dead_zone_bins)
                ]
            except Exception:
                pass
        codes: List[int] = [0] * len(echo_values)
        for idx, val in enumerate(echo_values):
            if idx < dead_zone_bins or val < 0.03:
                codes[idx] = 0
                continue
            raw = max(0.0, min(1.0, float(val))) * 252.0
            nearest = min(FURUNO_GAIN_CODES, key=lambda code: abs(code - raw))
            codes[idx] = int(nearest)
        return codes

    def save_furuno_rotation(self, rows: List[dict], filename: str) -> str:
        """Write exact Furuno-style CSV rows: Status,Scale,Range,Gain,Angle,EchoValues."""
        filepath = filename if os.path.isabs(filename) else os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(filepath) or self.output_dir, exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Status', 'Scale', 'Range', 'Gain', 'Angle', 'EchoValues'])
            for row in rows:
                writer.writerow([
                    row['status'],
                    row['scale'],
                    row['range'],
                    row['gain'],
                    row['angle'],
                    *row['echoes'],
                ])
        self._mirror_file(filepath)
        return filepath

    def get_record_count(self) -> int:
        return len(self.records)

    def is_active(self) -> bool:
        return self.is_recording
