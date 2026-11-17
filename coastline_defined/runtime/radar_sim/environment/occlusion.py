"""Terrain occlusion engine — LOS checks and terrain radar returns.

Optimized with NumPy vectorization for ray-marching and elevation sampling.
"""
import math
from typing import List, Optional, Union, Tuple
import numpy as np
from .terrain import HeightMap
from .attenuation import (
    ATTENUATION_CURRENT,
    ATTENUATION_ENTRY_R4_DEPTH,
    entry_r4_depth_attenuation,
    normalize_attenuation_model,
)

# Type alias
ArrayLike = Union[List[float], np.ndarray]


class OcclusionEngine:
    """Computes line-of-sight occlusion and terrain radar returns.

    Optimized with NumPy for ~5-20x speedup on ray-marching operations.
    """

    def __init__(self, terrain_maps: List[HeightMap],
                 antenna_height_m: float = 15.0):
        """
        Args:
            terrain_maps: List of HeightMap objects in the scene.
            antenna_height_m: Radar antenna height above sea level.
        """
        self.terrain_maps = terrain_maps
        self.antenna_height_m = antenna_height_m
        # Cache for occlusion results: (origin_x, origin_y, target_x, target_y) -> bool
        self._occlusion_cache = {}
        self._cache_max_size = 10000

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _sample_elevation(self, wx: float, wy: float) -> float:
        """Get the highest terrain elevation at a world point across all maps."""
        best = 0.0
        for hm in self.terrain_maps:
            e = hm.get_elevation(wx, wy)
            if e > best:
                best = e
        return best

    def _sample_elevations_batch(self, wx_arr: np.ndarray, wy_arr: np.ndarray) -> np.ndarray:
        """Batch sample elevations at multiple world points (vectorized).

        Args:
            wx_arr: Array of world X coordinates.
            wy_arr: Array of world Y coordinates.

        Returns:
            Array of highest elevation values at each point.
        """
        n = len(wx_arr)
        elevations = np.zeros(n)

        for hm in self.terrain_maps:
            # Get elevations from this heightmap (batch if supported, else loop)
            if hasattr(hm, 'get_elevations_batch'):
                e = hm.get_elevations_batch(wx_arr, wy_arr)
            else:
                # Fallback to per-point sampling
                e = np.array([hm.get_elevation(wx_arr[i], wy_arr[i]) for i in range(n)])
            elevations = np.maximum(elevations, e)

        return elevations

    def clear_cache(self) -> None:
        """Clear the occlusion result cache."""
        self._occlusion_cache.clear()

    # ------------------------------------------------------------------
    # Core algorithms (NumPy optimized)
    # ------------------------------------------------------------------
    def compute_occlusion_profile(self, origin_x: float, origin_y: float,
                                  bearing_deg: float, max_range_m: float,
                                  step_m: float = 50.0) -> List[float]:
        """Ray-march along a bearing and return the max elevation angle seen so far
        at each step. Vectorized with NumPy.

        Returns:
            List of max-elevation-angle (degrees) at each sample point.
        """
        ray_rad = math.radians(bearing_deg)
        dx = math.sin(ray_rad)
        dy = math.cos(ray_rad)

        # Vectorized distance array
        num_steps = int(max_range_m / step_m)
        distances = np.arange(1, num_steps + 1) * step_m

        # Batch compute world coordinates
        wx_arr = origin_x + dx * distances
        wy_arr = origin_y + dy * distances

        # Batch sample elevations
        elevations = self._sample_elevations_batch(wx_arr, wy_arr)

        # Vectorized angle calculation
        angles = np.degrees(np.arctan2(elevations - self.antenna_height_m, distances))

        # Compute cumulative maximum (running max of angles seen so far)
        profile = np.maximum.accumulate(angles)

        return profile.tolist()

    def is_target_occluded(self, origin_x: float, origin_y: float,
                           target_x: float, target_y: float,
                           target_height_m: float = 10.0,
                           step_m: float = 50.0) -> bool:
        """Check whether terrain blocks line-of-sight to a target.

        Vectorized with NumPy for batch elevation sampling.

        Args:
            origin_x, origin_y: Radar antenna position.
            target_x, target_y: Target position.
            target_height_m: Height of target above sea level.
            step_m: Ray-march step size.

        Returns:
            True if the target is occluded by terrain.
        """
        dx = target_x - origin_x
        dy = target_y - origin_y
        target_range = math.sqrt(dx * dx + dy * dy)
        if target_range < step_m:
            return False

        # Check cache first
        cache_key = (round(origin_x, 1), round(origin_y, 1),
                     round(target_x, 1), round(target_y, 1))
        if cache_key in self._occlusion_cache:
            return self._occlusion_cache[cache_key]

        # Angle from antenna to top of target
        target_angle = math.degrees(
            math.atan2(target_height_m - self.antenna_height_m, target_range)
        )

        # March along the ray with vectorized operations
        bearing_rad = math.atan2(dx, dy)
        ux = math.sin(bearing_rad)
        uy = math.cos(bearing_rad)

        # Create distance array
        num_steps = int(target_range / step_m)
        if num_steps == 0:
            return False

        distances = np.arange(1, num_steps + 1) * step_m

        # Batch compute world coordinates
        wx_arr = origin_x + ux * distances
        wy_arr = origin_y + uy * distances

        # Batch sample elevations
        elevations = self._sample_elevations_batch(wx_arr, wy_arr)

        # Vectorized angle calculation
        angles = np.degrees(np.arctan2(elevations - self.antenna_height_m, distances))

        # Find max terrain angle
        max_terrain_angle = np.max(angles) if len(angles) > 0 else -90.0

        result = max_terrain_angle > target_angle

        # Cache result (with size limit)
        if len(self._occlusion_cache) < self._cache_max_size:
            self._occlusion_cache[cache_key] = result

        return result

    def generate_terrain_returns(self, origin_x: float, origin_y: float,
                                 bearing_deg: float, beamwidth_deg: float,
                                 max_range_m: float, num_bins: int,
                                 return_numpy: bool = False,
                                 furuno_hybrid: bool = False,
                                 attenuation_model: str = ATTENUATION_CURRENT,
                                 attenuation_config: Optional[dict] = None) -> ArrayLike:
        """Generate radar return intensities from terrain along a bearing.

        Vectorized with NumPy. Casts multiple sub-rays across the beamwidth.

        Returns:
            Array of intensities (0-1) for each range bin.
        """
        returns = np.zeros(num_bins)
        bin_size = max_range_m / num_bins
        half_beam = beamwidth_deg / 2.0
        attenuation_model = normalize_attenuation_model(attenuation_model)
        num_rays = 5
        if furuno_hybrid and attenuation_model == ATTENUATION_ENTRY_R4_DEPTH:
            # Loaded coastline R4 mode needs denser beam sampling so land fill
            # does not appear as separated radial strokes in Furuno CSV output.
            requested_rays = int((attenuation_config or {}).get("continuous_num_rays", 7))
            num_rays = max(5, min(13, requested_rays))

        # Pre-compute distances for all bins (shared across rays)
        distances = (np.arange(num_bins) + 0.5) * bin_size

        for ray_idx in range(num_rays):
            if num_rays > 1:
                offset = -half_beam + (2 * half_beam * ray_idx / (num_rays - 1))
            else:
                offset = 0.0
            ray_bearing = bearing_deg + offset
            ray_weight = 1.0 - 0.3 * abs(offset) / (half_beam + 0.01)

            ray_rad = math.radians(ray_bearing)
            ux = math.sin(ray_rad)
            uy = math.cos(ray_rad)

            # Vectorized world coordinates
            wx_arr = origin_x + ux * distances
            wy_arr = origin_y + uy * distances

            # Batch sample elevations
            elevations = self._sample_elevations_batch(wx_arr, wy_arr)

            # Vectorized angle calculation
            angles = np.degrees(np.arctan2(elevations - self.antenna_height_m, distances))

            land_mask = elevations > 0.0
            if furuno_hybrid and attenuation_model == ATTENUATION_ENTRY_R4_DEPTH:
                max_gap_m = float((attenuation_config or {}).get("continuous_gap_close_m", 120.0))
                max_gap_bins = max(1, int(round(max_gap_m / max(1.0, bin_size))))
                idx = 0
                while idx < num_bins:
                    if land_mask[idx]:
                        idx += 1
                        continue
                    gap_start = idx
                    while idx + 1 < num_bins and not land_mask[idx + 1]:
                        idx += 1
                    gap_end = idx
                    surrounded_by_land = (
                        gap_start > 0
                        and gap_end < num_bins - 1
                        and land_mask[gap_start - 1]
                        and land_mask[gap_end + 1]
                    )
                    if surrounded_by_land and (gap_end - gap_start + 1) <= max_gap_bins:
                        land_mask[gap_start:gap_end + 1] = True
                    idx += 1

            # Process visibility with running maximum
            visible_mask = np.zeros(num_bins, dtype=bool)
            max_angle = -90.0
            for b in range(num_bins):
                if not land_mask[b]:
                    continue

                if angles[b] >= max_angle:
                    max_angle = angles[b]
                    visible_mask[b] = True
                    if not furuno_hybrid:
                        # Visible terrain — produce a return
                        reflectivity = 0.85
                        intensity = reflectivity * ray_weight * min(1.0, elevations[b] / 50.0)
                        returns[b] = max(returns[b], intensity)

            if not furuno_hybrid:
                continue

            core_depth_bins = max(1, min(4, int(round(num_bins / 220.0))))
            interior_decay_bins = max(2.0, num_bins / 110.0)
            far_range_floor = 0.35
            range_decay_strength = 0.85
            leading_edge_boost = 1.12
            edge_factor = 0.78

            b = 0
            while b < num_bins:
                if not land_mask[b]:
                    b += 1
                    continue

                start = b
                while b + 1 < num_bins and land_mask[b + 1]:
                    b += 1
                end = b

                if not np.any(visible_mask[start:end + 1]):
                    b += 1
                    continue

                first_visible = start + int(np.argmax(visible_mask[start:end + 1]))
                segment_bins = np.arange(first_visible, end + 1)
                if len(segment_bins) == 0:
                    b += 1
                    continue

                segment_ranges = distances[segment_bins]
                elevation_factor = np.clip(
                    0.65 + 0.35 * np.minimum(elevations[segment_bins] / 25.0, 1.0),
                    0.65, 1.0
                )
                if attenuation_model == ATTENUATION_ENTRY_R4_DEPTH:
                    depth_m = segment_ranges - segment_ranges[0]
                    decay_m = float((attenuation_config or {}).get("continuous_decay_m", 520.0))
                    floor = float((attenuation_config or {}).get("continuous_floor", 0.34))
                    range_power = float((attenuation_config or {}).get("continuous_range_power", 0.18))
                    depth_decay = floor + (1.0 - floor) * np.exp(-depth_m / max(1.0, decay_m))
                    range_decay = np.power(
                        segment_ranges[0] / np.maximum(segment_ranges[0], segment_ranges),
                        max(0.0, range_power),
                    )
                    attenuation = np.clip(depth_decay * range_decay, 0.0, 1.0)
                    intensities = 0.85 * ray_weight * leading_edge_boost * attenuation * elevation_factor
                else:
                    range_ratio = np.clip(segment_ranges / max(1.0, max_range_m), 0.0, 1.0)
                    range_decay = np.maximum(
                        far_range_floor,
                        1.0 - range_decay_strength * np.power(range_ratio, 1.35),
                    )

                    depth_from_entry = np.arange(len(segment_bins), dtype=np.float32)
                    profile = np.empty(len(segment_bins), dtype=np.float32)
                    core_mask = depth_from_entry < core_depth_bins
                    if np.any(core_mask):
                        profile[core_mask] = (
                            leading_edge_boost
                            - (depth_from_entry[core_mask] / max(1, core_depth_bins))
                            * (leading_edge_boost - 1.0)
                        )
                    if np.any(~core_mask):
                        interior_depth = depth_from_entry[~core_mask] - core_depth_bins + 1.0
                        profile[~core_mask] = np.exp(-interior_depth / interior_decay_bins)

                    intensities = 0.85 * ray_weight * range_decay * profile * elevation_factor
                intensities[0] *= edge_factor
                if len(intensities) > 1 and attenuation_model != ATTENUATION_ENTRY_R4_DEPTH:
                    intensities[-1] *= edge_factor

                intensities = np.clip(intensities, 0.0, 1.0)
                returns[segment_bins] = np.maximum(returns[segment_bins], intensities)
                b += 1

        if furuno_hybrid and attenuation_model == ATTENUATION_ENTRY_R4_DEPTH:
            # Lightly close range-bin pinholes while preserving bright leading
            # edges. This affects exported echoes, not just the UI preview.
            kernel = np.array([0.16, 0.68, 0.16], dtype=np.float32)
            smoothed = np.convolve(returns.astype(np.float32), kernel, mode="same")
            returns = np.maximum(returns, smoothed * 0.94)

        return returns if return_numpy else returns.tolist()

    def get_occlusion_mask(self, origin_x: float, origin_y: float,
                           bearing_deg: float, max_range_m: float,
                           num_bins: int) -> List[bool]:
        """Return a boolean shadow mask for a single bearing.

        Vectorized with NumPy for elevation sampling.

        True means the range bin is in shadow (occluded).
        """
        bin_size = max_range_m / num_bins
        ray_rad = math.radians(bearing_deg)
        ux = math.sin(ray_rad)
        uy = math.cos(ray_rad)

        # Vectorized distance and coordinate calculation
        distances = (np.arange(num_bins) + 0.5) * bin_size
        wx_arr = origin_x + ux * distances
        wy_arr = origin_y + uy * distances

        # Batch sample elevations
        elevations = self._sample_elevations_batch(wx_arr, wy_arr)

        # Vectorized angle calculation
        angles = np.degrees(np.arctan2(elevations - self.antenna_height_m, distances))

        # Build mask with running maximum tracking
        mask = np.zeros(num_bins, dtype=bool)
        max_angle = -90.0

        for b in range(num_bins):
            elev = elevations[b]
            angle = angles[b]

            if elev > 0 and angle >= max_angle:
                max_angle = angle
                # This bin has visible terrain — not shadowed
            elif max_angle > -90.0:
                # Check if this point is below the shadow line
                shadow_elev = self.antenna_height_m + distances[b] * math.tan(math.radians(max_angle))
                if elev < shadow_elev:
                    mask[b] = True

        return mask.tolist()
