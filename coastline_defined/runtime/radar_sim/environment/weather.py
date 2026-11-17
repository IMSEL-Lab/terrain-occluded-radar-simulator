"""Weather effects for radar simulation.

Optimized with NumPy vectorization.
"""
from dataclasses import dataclass
from typing import List, Union
import numpy as np
from .noise import NoiseGenerator
from .clutter import SeaClutter, RainClutter

# Type alias
ArrayLike = Union[List[float], np.ndarray]


@dataclass
class WeatherConditions:
    """Current weather conditions."""
    sea_state: int = 3  # 0-9 Douglas scale
    wind_speed_knots: float = 10.0
    wind_direction: float = 0.0  # degrees
    rain_rate_mmh: float = 0.0
    visibility_nm: float = 10.0


class WeatherEffects:
    """Combines all weather-related effects on radar.

    Optimized with NumPy for vectorized operations.
    """

    def __init__(self):
        self.conditions = WeatherConditions()
        self.noise_gen = NoiseGenerator()
        self.sea_clutter = SeaClutter()
        self.rain_clutter = RainClutter()
        self.receiver_noise_enabled = False

    def set_conditions(self, conditions: WeatherConditions) -> None:
        """Set weather conditions."""
        self.conditions = conditions
        self.sea_clutter.set_sea_state(conditions.sea_state)
        self.sea_clutter.set_wind(conditions.wind_direction)
        self.rain_clutter.set_rain_rate(conditions.rain_rate_mmh)

    def set_sea_state(self, state: int) -> None:
        """Set sea state."""
        self.conditions.sea_state = state
        self.sea_clutter.set_sea_state(state)

    def set_wind(self, direction: float, speed_knots: float) -> None:
        """Set wind conditions."""
        self.conditions.wind_direction = direction
        self.conditions.wind_speed_knots = speed_knots
        self.sea_clutter.set_wind(direction, speed_knots)

    def set_rain(self, rate_mmh: float) -> None:
        """Set rain rate."""
        self.conditions.rain_rate_mmh = rate_mmh
        self.rain_clutter.set_rain_rate(rate_mmh)

    def set_noise_level(self, level: float) -> None:
        """Set receiver thermal noise level used in sweep generation."""
        self.noise_gen.set_noise_level(level)

    def set_receiver_noise_enabled(self, enabled: bool) -> None:
        """Enable or disable explicit receiver noise in sweep generation."""
        self.receiver_noise_enabled = bool(enabled)

    def get_receiver_noise_level(self) -> float:
        """Return the configured receiver-noise level, even when disabled."""
        return float(self.noise_gen.thermal_noise_level)

    def get_effective_receiver_noise_level(self) -> float:
        """Return the noise level currently applied to exported/radar sweeps."""
        if not self.receiver_noise_enabled:
            return 0.0
        return self.get_receiver_noise_level()

    def apply_to_sweep(self, sweep_data: ArrayLike, bearing: float,
                      max_range_m: float, sea_suppression: float = 0.0,
                      rain_suppression: float = 0.0,
                      return_numpy: bool = False) -> ArrayLike:
        """Apply all weather effects to sweep data.

        Vectorized with NumPy for ~5-10x speedup.

        Uses power addition (not max) for physically correct combination,
        and applies two-way rain attenuation per bin.
        """
        sweep_arr = np.asarray(sweep_data)
        num_bins = len(sweep_arr)
        bin_size = max_range_m / num_bins

        # Generate clutter (now returns numpy arrays directly)
        sea = self.sea_clutter.generate_clutter(
            num_bins, bearing, max_range_m, sea_suppression, return_numpy=True
        )
        rain = self.rain_clutter.generate_clutter(
            num_bins, bearing, max_range_m, rain_suppression, return_numpy=True
        )

        # Power addition: combined = signal + sea + rain
        combined = sweep_arr + sea + rain

        # Rain attenuation: Marshall-Palmer at X-band
        # 0.01 * R^1.21 dB/km (R = rain rate in mm/h)
        rain_rate = self.conditions.rain_rate_mmh
        if rain_rate > 0:
            rain_atten_db_per_km = 0.01 * (rain_rate ** 1.21)

            # Vectorized range-dependent attenuation
            ranges_km = (np.arange(num_bins) + 0.5) * bin_size / 1000.0
            atten_db = rain_atten_db_per_km * ranges_km * 2.0  # two-way
            atten_linear = 10.0 ** (-atten_db / 10.0)
            combined *= atten_linear

        # Add receiver noise only when the explicit UI/config toggle enables it.
        noise_level = self.get_effective_receiver_noise_level()
        if noise_level > 0.0:
            result = self.noise_gen.add_noise_to_sweep(
                combined, noise_level=noise_level, return_numpy=True)
        else:
            result = combined

        return result if return_numpy else result.tolist()

    def get_attenuation_factor(self, range_m: float) -> float:
        """Calculate atmospheric attenuation based on weather."""
        rain_rate = self.conditions.rain_rate_mmh
        if rain_rate <= 0:
            return 1.0
        rain_atten_db_per_km = 0.01 * (rain_rate ** 1.21)
        range_km = range_m / 1000.0
        total_atten_db = rain_atten_db_per_km * range_km * 2
        return 10 ** (-total_atten_db / 10)
