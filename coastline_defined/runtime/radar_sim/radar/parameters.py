"""Radar system parameters based on Furuno marine radar specifications."""
import math
from dataclasses import dataclass
from enum import Enum
from typing import List

DISPLAY_RANGE_SCALES_NM = [0.0625, 0.125, 0.25, 0.5, 0.75, 1.5, 3, 6, 12, 24, 48]
DEFAULT_RANGE_SCALES_NM = list(DISPLAY_RANGE_SCALES_NM)
FURUNO_COASTAL_RANGE_SCALES_NM = list(DISPLAY_RANGE_SCALES_NM)
DEFAULT_PULSE_LENGTHS_US = [0.07, 0.15, 0.5, 0.8, 1.2]
FURUNO_COASTAL_PULSE_LENGTHS_US = [0.07, 0.15, 0.3, 0.5, 0.8, 1.2]

class RadarBand(Enum):
    X_BAND = "X"  # 9.3-9.5 GHz
    S_BAND = "S"  # 2.9-3.1 GHz

@dataclass
class RadarParameters:
    """Configuration parameters for the radar system."""

    # Radar identification
    name: str = "Furuno FAR-2xx7"
    band: RadarBand = RadarBand.X_BAND

    # Transmitter
    peak_power_kw: float = 25.0  # Peak transmit power in kW
    frequency_ghz: float = 9.41  # Operating frequency
    pulse_lengths_us: List[float] = None  # Available pulse lengths
    pulse_length_idx: int = 2  # Index into pulse_lengths_us for current pulse
    prf_hz: float = 3000  # Pulse repetition frequency

    # Antenna
    antenna_gain_db: float = 28.0
    horizontal_beamwidth_deg: float = 1.2
    vertical_beamwidth_deg: float = 22.0
    rotation_rpm: float = 24.0  # Antenna rotation speed
    antenna_height_m: float = 15.0  # Height above sea level

    # Receiver
    noise_figure_db: float = 5.0
    min_detectable_signal_dbm: float = -100.0

    # Range settings
    range_scales_nm: List[float] = None  # Available range scales
    current_range_nm: float = 6.0  # Currently selected range

    # Processing
    gain: float = 0.8  # 0-1, receiver gain
    sea_clutter: float = 0.3  # 0-1, sea clutter suppression
    rain_clutter: float = 0.3  # 0-1, rain clutter suppression
    interference_rejection: bool = True

    # Signal chain parameters
    wavelength_m: float = 0.0319  # ~3.19 cm at 9.41 GHz
    system_losses_db: float = 8.0  # Cable, connector, radome losses
    display_log_range_db: float = 80.0  # Dynamic range for log compression
    output_bits: int = 8  # Quantization bits

    # Sweep/display geometry
    sweep_spokes_per_rotation: int = 360
    display_num_range_bins: int = 512

    def __post_init__(self):
        if self.pulse_lengths_us is None:
            self.pulse_lengths_us = list(DEFAULT_PULSE_LENGTHS_US)
        if self.range_scales_nm is None:
            self.range_scales_nm = list(DEFAULT_RANGE_SCALES_NM)

    @property
    def current_pulse_length_us(self) -> float:
        """Currently selected pulse length in microseconds."""
        idx = max(0, min(self.pulse_length_idx, len(self.pulse_lengths_us) - 1))
        return self.pulse_lengths_us[idx]

    @property
    def rotation_period_s(self) -> float:
        """Time for one complete antenna rotation in seconds."""
        return 60.0 / self.rotation_rpm

    @property
    def range_resolution_m(self) -> float:
        """Range resolution based on current pulse length."""
        pulse_s = self.current_pulse_length_us * 1e-6
        c = 299792458  # Speed of light
        return c * pulse_s / 2

    @property
    def max_range_m(self) -> float:
        """Maximum range in meters based on current scale."""
        return self.current_range_nm * 1852  # nm to meters

    def set_range_scale(self, range_nm: float) -> None:
        """Set the range scale to the nearest available value."""
        closest = min(self.range_scales_nm, key=lambda x: abs(x - range_nm))
        self.current_range_nm = closest

    def apply_standard_profile(self, range_nm: float = None) -> None:
        """Use the simulator's original generic marine-radar profile."""
        self.name = "Terrain-Occluded Marine Radar"
        self.band = RadarBand.X_BAND
        self.frequency_ghz = 9.41
        self.pulse_lengths_us = list(DEFAULT_PULSE_LENGTHS_US)
        self.pulse_length_idx = min(2, len(self.pulse_lengths_us) - 1)
        self.horizontal_beamwidth_deg = 1.2
        self.vertical_beamwidth_deg = 22.0
        self.rotation_rpm = 24.0
        self.display_log_range_db = 80.0
        self.gain = 0.8
        self.sea_clutter = 0.3
        self.rain_clutter = 0.3
        self.range_scales_nm = list(DEFAULT_RANGE_SCALES_NM)
        self.sweep_spokes_per_rotation = 360
        self.display_num_range_bins = 512
        target_range = self.current_range_nm if range_nm is None else range_nm
        self.set_range_scale(target_range)

    def apply_furuno_coastal_profile(self, range_nm: float = None) -> None:
        """Use a Furuno-style coastal-display profile for .radarloc scenes.

        This stays compatible with the terrain simulator's host signal chain,
        but moves sweep geometry closer to the Object Creation / Furuno dialect.
        """
        self.name = "Furuno Hybrid Coastal"
        self.band = RadarBand.X_BAND
        self.frequency_ghz = 9.41
        self.pulse_lengths_us = list(FURUNO_COASTAL_PULSE_LENGTHS_US)
        self.pulse_length_idx = min(3, len(self.pulse_lengths_us) - 1)
        self.horizontal_beamwidth_deg = 3.1
        self.vertical_beamwidth_deg = 22.0
        self.rotation_rpm = 24.0
        self.display_log_range_db = 45.0
        self.gain = 0.65
        self.sea_clutter = 0.85
        self.rain_clutter = 1.0
        self.range_scales_nm = list(FURUNO_COASTAL_RANGE_SCALES_NM)
        self.sweep_spokes_per_rotation = 720
        self.display_num_range_bins = 868
        target_range = self.current_range_nm if range_nm is None else range_nm
        self.set_range_scale(target_range)

    def compute_stc_curve(self, num_bins: int = 512) -> List[float]:
        """Compute STC attenuation curve from sea_clutter knob setting.

        Maps the sea_clutter knob (0-1) to a range-dependent attenuation
        curve that suppresses near-range clutter.

        Args:
            num_bins: Number of range bins.

        Returns:
            List of attenuation values (0-1) per bin, where 1 = full suppression.
        """
        if self.sea_clutter <= 0.0:
            return [0.0] * num_bins

        # STC affects the first ~25% of range proportional to knob setting
        max_range = self.max_range_m
        stc_range_m = max_range * 0.25 * self.sea_clutter
        bin_size = max_range / num_bins

        curve = []
        for i in range(num_bins):
            range_m = (i + 0.5) * bin_size
            if range_m < stc_range_m and stc_range_m > 0:
                # Attenuation strongest at close range, tapering to zero
                ratio = range_m / stc_range_m
                atten = self.sea_clutter * (1.0 - ratio) ** 2
                curve.append(atten)
            else:
                curve.append(0.0)
        return curve
