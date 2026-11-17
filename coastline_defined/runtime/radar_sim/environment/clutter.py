"""Sea and rain clutter simulation with K-distribution sea clutter."""
import math
from typing import List, Union
import numpy as np

# Type alias for array-like returns
ArrayLike = Union[List[float], np.ndarray]


class SeaClutter:
    """Simulates sea surface clutter returns using K-distribution.

    Optimized with NumPy vectorization for ~10-50x speedup over pure Python.
    """

    def __init__(self):
        self.sea_state = 3  # 0-9 Douglas sea scale
        self.wind_direction = 0.0  # degrees
        self.base_intensity = 0.1

    def set_sea_state(self, state: int) -> None:
        """Set sea state (0-9 Douglas scale)."""
        self.sea_state = max(0, min(9, state))

    def set_wind(self, direction: float, speed_knots: float = None) -> None:
        """Set wind direction (and optionally estimate sea state from wind)."""
        self.wind_direction = direction % 360
        if speed_knots is not None:
            self.sea_state = min(9, int(speed_knots / 5))

    def generate_clutter(self, num_range_bins: int, bearing: float,
                        max_range_m: float, suppression: float = 0.0,
                        return_numpy: bool = False) -> ArrayLike:
        """Generate K-distribution sea clutter for a sweep.

        K-distribution = gamma(nu) * exponential, where nu (shape) decreases
        with sea state to produce spikier clutter.

        Vectorized with NumPy for performance.
        """
        bin_size = max_range_m / num_range_bins

        # K-distribution shape parameter: lower = spikier
        nu = max(0.5, 10.0 - self.sea_state * 1.2)

        # Wind direction modulation
        wind_factor = 1.0 + 0.3 * math.cos(math.radians(bearing - self.wind_direction))

        # Base clutter power from sea state
        base = self.base_intensity * (self.sea_state / 5.0)

        # Vectorized range calculation
        ranges = (np.arange(num_range_bins) + 0.5) * bin_size

        # Vectorized range-dependent falloff: R^-2.5
        range_factors = np.where(ranges < 100, 1.0, (100.0 / ranges) ** 2.5)

        # Vectorized K-distribution sampling
        # gamma(nu, 1/nu) * exponential(1) - mean = 1.0
        gamma_samples = np.random.gamma(nu, 1.0 / nu, num_range_bins)
        exp_samples = np.random.exponential(1.0, num_range_bins)
        k_samples = gamma_samples * exp_samples

        # Compute clutter values
        clutter = base * k_samples * range_factors * wind_factor
        clutter = np.minimum(1.0, clutter)
        clutter *= (1.0 - suppression)
        clutter = np.maximum(0.0, clutter)

        return clutter if return_numpy else clutter.tolist()


class RainClutter:
    """Simulates rain/precipitation clutter.

    Optimized with NumPy vectorization.
    """

    def __init__(self):
        self.rain_rate_mmh = 0.0  # mm/hour
        self.rain_cells = []  # List of (range, bearing, radius, intensity)

    def set_rain_rate(self, rate_mmh: float) -> None:
        """Set uniform rain rate in mm/hour."""
        self.rain_rate_mmh = max(0.0, rate_mmh)

    def add_rain_cell(self, range_m: float, bearing: float,
                     radius_m: float, intensity: float) -> None:
        """Add a localized rain cell."""
        self.rain_cells.append((range_m, bearing, radius_m, intensity))

    def clear_rain_cells(self) -> None:
        """Remove all rain cells."""
        self.rain_cells.clear()

    def generate_clutter(self, num_range_bins: int, bearing: float,
                        max_range_m: float, suppression: float = 0.0,
                        return_numpy: bool = False) -> ArrayLike:
        """Generate rain clutter for a sweep (vectorized)."""
        bin_size = max_range_m / num_range_bins
        ranges = (np.arange(num_range_bins) + 0.5) * bin_size

        # Initialize clutter array
        clutter = np.zeros(num_range_bins)

        # Uniform rain - vectorized
        if self.rain_rate_mmh > 0:
            uniform_level = min(0.5, self.rain_rate_mmh / 100.0)
            clutter = uniform_level * np.random.uniform(0.5, 1.0, num_range_bins)

        # Rain cells - partially vectorized
        for cell_range, cell_bearing, cell_radius, cell_intensity in self.rain_cells:
            bearing_diff = abs((bearing - cell_bearing + 180) % 360 - 180)
            if bearing_diff > 30:
                continue

            # Vectorized distance calculations
            in_range_mask = np.abs(ranges - cell_range) < cell_radius
            if not np.any(in_range_mask):
                continue

            angular_dist = ranges * math.radians(bearing_diff)
            dist_to_center = np.sqrt((ranges - cell_range)**2 + angular_dist**2)
            in_cell_mask = dist_to_center < cell_radius

            # Apply cell intensity where in range
            edge_factor = 1.0 - (dist_to_center / cell_radius)
            cell_values = cell_intensity * edge_factor * np.random.uniform(0.7, 1.0, num_range_bins)
            clutter = np.where(in_cell_mask, np.maximum(clutter, cell_values), clutter)

        # Apply suppression
        clutter *= (1.0 - suppression)

        return clutter if return_numpy else clutter.tolist()
