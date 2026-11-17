"""Coastline and landmass simulation for radar returns.

Optimized with NumPy vectorization for ray-polygon intersections.
"""
import math
import os
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
import numpy as np
from radar_core import rust_bridge
from .attenuation import (
    ATTENUATION_CURRENT,
    ATTENUATION_ENTRY_R4_DEPTH,
    entry_r4_depth_attenuation,
    normalize_attenuation_model,
)

# Type alias
ArrayLike = Union[List[float], np.ndarray]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def apply_coastline_edge_profile(returns: ArrayLike) -> np.ndarray:
    """Optionally tighten loaded-coastline bodies and add a low-code halo.

    This is intentionally a post-process so the Python and Rust coastline
    renderers keep the same geometric ray-intersection behavior. The launcher
    enables it for .radarloc exports; leaving the environment flag unset keeps
    prior behavior available.
    """
    arr = np.asarray(returns, dtype=np.float32).copy()
    if not _env_bool("RADARLOC_COASTLINE_EDGE_PROFILE", False) or arr.size == 0:
        return arr

    active_threshold = max(
        0.0,
        min(1.0, _env_float("RADARLOC_COASTLINE_ACTIVE_THRESHOLD", 0.02)),
    )
    body_core_bins = max(0, _env_int("RADARLOC_COASTLINE_BODY_CORE_BINS", 2))
    body_decay_bins = max(0.1, _env_float("RADARLOC_COASTLINE_BODY_DECAY_BINS", 5.0))
    body_max = max(
        0.0,
        min(1.0, _env_float("RADARLOC_COASTLINE_BODY_MAX_CODE", 190.0) / 252.0),
    )
    body_floor = max(
        0.0,
        min(1.0, _env_float("RADARLOC_COASTLINE_BODY_FLOOR_CODE", 0.0) / 252.0),
    )
    halo_bins = max(0, _env_int("RADARLOC_COASTLINE_HALO_BINS", 1))
    halo_scale = max(0.0, min(1.0, _env_float("RADARLOC_COASTLINE_HALO_SCALE", 0.35)))
    halo_max = max(
        0.0,
        min(1.0, _env_float("RADARLOC_COASTLINE_HALO_MAX_CODE", 24.0) / 252.0),
    )

    active = arr >= active_threshold
    if not np.any(active):
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    padded = np.concatenate(([False], active, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(edges[0::2], edges[1::2]):
        segment = arr[start:end]
        if segment.size == 0:
            continue

        peak_window = segment[: max(1, min(segment.size, body_core_bins + 1))]
        entry_peak = min(float(np.max(peak_window)), body_max)
        arr[start:end] = np.minimum(segment, body_max)

        if body_core_bins < segment.size:
            tail_idx = np.arange(body_core_bins, segment.size, dtype=np.float32)
            allowed = entry_peak * np.exp(
                -((tail_idx - body_core_bins + 1.0) / body_decay_bins)
            )
            if body_floor > 0.0:
                allowed = np.maximum(body_floor, allowed)
            global_tail = start + tail_idx.astype(np.int32)
            arr[global_tail] = np.minimum(arr[global_tail], allowed.astype(np.float32))

        if halo_bins > 0 and halo_scale > 0.0 and halo_max > 0.0:
            left_edge = float(arr[start])
            right_edge = float(arr[end - 1])
            for distance in range(1, halo_bins + 1):
                falloff = halo_scale ** distance
                left_idx = start - distance
                if left_idx >= 0 and not active[left_idx]:
                    arr[left_idx] = max(arr[left_idx], min(halo_max, left_edge * falloff))
                right_idx = end - 1 + distance
                if right_idx < arr.size and not active[right_idx]:
                    arr[right_idx] = max(arr[right_idx], min(halo_max, right_edge * falloff))

    return np.clip(arr, 0.0, 1.0).astype(np.float32)


@dataclass
class CoastlinePoint:
    """A point defining the coastline."""
    x: float  # meters, East positive
    y: float  # meters, North positive


@dataclass
class CoastlineEchoProfile:
    """Controls Furuno-style coastline rendering for the hybrid loader path."""
    num_rays: int = 9
    edge_factor: float = 0.78
    leading_edge_boost: float = 1.12
    far_range_floor: float = 0.35
    range_decay_strength: float = 0.85
    core_depth_scale: float = 220.0
    interior_decay_scale: float = 110.0
    attenuation_model: str = ATTENUATION_CURRENT
    ray_penetration_ref_m: float = 8.0
    ray_penetration_ref_range_m: float = 200.0
    ray_penetration_power: float = 0.5
    ray_penetration_min_m: float = 2.0
    ray_penetration_max_m: float = 20.0
    ray_core_depth_m: float = 1.5
    ray_floor: float = 0.0
    continuous_land_return: bool = True
    continuous_return_threshold: float = 0.06
    continuous_return_depth_m: float = 800.0
    continuous_return_decay_m: float = 1500.0
    continuous_return_floor: float = 0.22
    continuous_texture_strength: float = 0.30
    continuous_texture_floor: float = 0.48
    origin_range_attenuation: bool = True
    origin_range_ref_m: float = 6000.0
    origin_range_power: float = 4.0
    origin_range_floor: float = 0.04
    occlude_after_first_land: bool = True
    secondary_return_scale: float = 0.08
    secondary_return_max_code: float = 36.0


class Coastline:
    """Represents a coastline that generates radar returns.

    Optimized with NumPy for vectorized ray-polygon intersections.
    """

    def __init__(self, points: List[Tuple[float, float]] = None):
        """Initialize coastline with a list of (x, y) points."""
        self.points: List[CoastlinePoint] = []
        self._points_array: Optional[np.ndarray] = None  # Cached numpy array
        self._edge_start: Optional[np.ndarray] = None
        self._edge_end: Optional[np.ndarray] = None
        self._edge_dx: Optional[np.ndarray] = None
        self._edge_dy: Optional[np.ndarray] = None
        if points:
            self.set_points(points)

        # Radar reflection characteristics
        self.reflectivity = 0.9  # High reflectivity for land
        self.roughness = 0.3    # Surface roughness affects return spread
        self.render_mode = "simple"
        self.echo_profile = CoastlineEchoProfile()

    def _update_points_array(self) -> None:
        """Update cached numpy arrays for points and edges."""
        if self.points:
            self._points_array = np.array([[p.x, p.y] for p in self.points], dtype=np.float32)
            if len(self._points_array) >= 2:
                self._edge_start = self._points_array[:-1]
                self._edge_end = self._points_array[1:]
                self._edge_dx = self._edge_end[:, 0] - self._edge_start[:, 0]
                self._edge_dy = self._edge_end[:, 1] - self._edge_start[:, 1]
            else:
                self._edge_start = None
                self._edge_end = None
                self._edge_dx = None
                self._edge_dy = None
        else:
            self._points_array = None
            self._edge_start = None
            self._edge_end = None
            self._edge_dx = None
            self._edge_dy = None

    def _invalidate_geometry_cache(self) -> None:
        """Invalidate cached geometry arrays after editing points."""
        self._points_array = None
        self._edge_start = None
        self._edge_end = None
        self._edge_dx = None
        self._edge_dy = None

    def set_points(self, points: List[Tuple[float, float]]) -> None:
        """Replace all coastline points in one bulk operation."""
        self.points = [CoastlinePoint(float(x), float(y)) for x, y in points]
        self._update_points_array()

    def add_point(self, x: float, y: float) -> None:
        """Add a point to the coastline."""
        self.points.append(CoastlinePoint(x, y))
        self._invalidate_geometry_cache()

    def clear(self) -> None:
        """Clear all coastline points."""
        self.points.clear()
        self._invalidate_geometry_cache()

    def configure_rendering(self, mode: str = "simple",
                            echo_profile: Optional[Dict[str, Any]] = None) -> None:
        """Configure how this coastline paints radar returns.

        The terrain-occluded simulator now supports a Furuno-inspired hybrid
        path for .radarloc-loaded coastlines. Unknown profile keys are ignored
        so loader workflows remain forward-compatible.
        """
        self.render_mode = mode or "simple"
        if echo_profile:
            for key, value in echo_profile.items():
                if hasattr(self.echo_profile, key):
                    setattr(self.echo_profile, key, value)
        self.echo_profile.attenuation_model = normalize_attenuation_model(
            getattr(self.echo_profile, "attenuation_model", ATTENUATION_CURRENT)
        )

    def _attenuation_config(self) -> Dict[str, float]:
        profile = self.echo_profile
        return {
            "penetration_ref_m": profile.ray_penetration_ref_m,
            "penetration_ref_range_m": profile.ray_penetration_ref_range_m,
            "penetration_power": profile.ray_penetration_power,
            "penetration_min_m": profile.ray_penetration_min_m,
            "penetration_max_m": profile.ray_penetration_max_m,
            "core_depth_m": profile.ray_core_depth_m,
            "floor": profile.ray_floor,
        }

    def _rust_attenuation_config_values(self) -> Tuple[float, ...]:
        def env_float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        profile = self.echo_profile
        return (
            env_float("RADARLOC_RAY_ATTEN_R_MIN_M", 20.0),
            profile.ray_penetration_ref_m,
            profile.ray_penetration_ref_range_m,
            profile.ray_penetration_power,
            profile.ray_penetration_min_m,
            profile.ray_penetration_max_m,
            profile.ray_core_depth_m,
            profile.ray_floor,
        )

    def _continuous_entry_bins(self, start_bin: int, entry_dist: float,
                               bin_size: float, num_bins: int,
                               end_bin: Optional[int] = None) -> np.ndarray:
        """Return consecutive land-body bins behind a shoreline entry."""
        if start_bin >= num_bins:
            return np.zeros(0, dtype=np.int32)

        depth_m = max(0.0, float(self.echo_profile.continuous_return_depth_m))
        max_bins = max(1, int(math.ceil(depth_m / max(1e-6, bin_size))))
        stop_bin = min(num_bins - 1, start_bin + max_bins)
        if end_bin is not None:
            stop_bin = min(stop_bin, max(start_bin, int(end_bin)))
        stop_bin = min(num_bins, stop_bin + 1)
        return np.arange(start_bin, stop_bin, dtype=np.int32)

    def _continuous_entry_attenuation(self, bin_ranges: np.ndarray,
                                      entry_dist: float) -> np.ndarray:
        """Strict radar range-equation attenuation behind the shoreline entry.

        I_b = I_entry * (R_entry / R_b)^4
        """
        profile = self.echo_profile
        entry_range = max(1.0, float(entry_dist))
        bin_ranges = np.maximum(entry_range, np.asarray(bin_ranges, dtype=np.float32))
        attenuation = np.power(entry_range / bin_ranges, 4.0)
        threshold = max(0.0, float(profile.continuous_return_threshold))
        attenuation = np.where(attenuation >= threshold, attenuation, 0.0)
        return np.clip(attenuation, 0.0, 1.0).astype(np.float32)

    def _origin_range_attenuation(self, bin_ranges: np.ndarray) -> np.ndarray:
        """Return own-ship-to-bin range loss for display-calibrated coastlines."""
        profile = self.echo_profile
        if not profile.origin_range_attenuation:
            return np.ones_like(bin_ranges, dtype=np.float32)
        ref_m = max(1.0, float(profile.origin_range_ref_m))
        power = max(0.0, float(profile.origin_range_power))
        floor = max(0.0, min(1.0, float(profile.origin_range_floor)))
        ranges = np.maximum(ref_m, np.asarray(bin_ranges, dtype=np.float32))
        attenuation = np.power(ref_m / ranges, power)
        if floor > 0.0:
            attenuation = np.maximum(floor, attenuation)
        return np.clip(attenuation, 0.0, 1.0).astype(np.float32)

    def _apply_secondary_land_occlusion(
        self, intensities: np.ndarray, is_secondary_land_interval: bool
    ) -> np.ndarray:
        """Suppress land intervals that sit behind the first shoreline hit."""
        profile = self.echo_profile
        if not (profile.occlude_after_first_land and is_secondary_land_interval):
            return intensities
        scale = max(0.0, min(1.0, float(profile.secondary_return_scale)))
        max_return = max(0.0, min(1.0, float(profile.secondary_return_max_code) / 252.0))
        return np.minimum(intensities * scale, max_return).astype(np.float32)

    def _apply_continuous_radar_texture(self, returns: np.ndarray,
                                        bearing: float) -> np.ndarray:
        """Break up flat polygon fills with deterministic Furuno-like texture."""
        profile = self.echo_profile
        if not (
            profile.attenuation_model == ATTENUATION_ENTRY_R4_DEPTH
            and profile.continuous_land_return
            and np.any(returns > 0.0)
        ):
            return returns

        idx = np.arange(len(returns), dtype=np.float32)
        # Deterministic pseudo-noise, smoothed along range so it reads as radar
        # phosphor texture instead of random salt-and-pepper.
        raw = np.sin(idx * 12.9898 + float(bearing) * 78.233) * 43758.5453
        noise = raw - np.floor(raw)
        noise = np.convolve(
            noise.astype(np.float32),
            np.array([0.08, 0.18, 0.48, 0.18, 0.08], dtype=np.float32),
            mode="same",
        )
        slow_variation = 0.88 + 0.12 * np.sin(idx * 0.075 + float(bearing) * 0.031)
        texture_floor = max(0.0, min(1.0, float(profile.continuous_texture_floor)))
        texture = (texture_floor + (1.0 - texture_floor) * noise) * slow_variation
        strength = max(0.0, min(1.0, float(profile.continuous_texture_strength)))
        textured = returns * ((1.0 - strength) + strength * texture)

        # Soften hard one-bin cliffs while keeping bright shoreline edges.
        smoothed = np.convolve(
            textured.astype(np.float32),
            np.array([0.08, 0.18, 0.48, 0.18, 0.08], dtype=np.float32),
            mode="same",
        )
        return np.maximum(textured * 0.88, smoothed * 0.96).astype(np.float32)

    def generate_returns(self, own_x: float, own_y: float,
                        bearing: float, beamwidth: float,
                        max_range: float, num_bins: int,
                        return_numpy: bool = False) -> ArrayLike:
        """Generate radar returns for the coastline at a given bearing.

        Uses filled-polygon ray-casting with even-odd rule to produce
        solid landmass returns matching real radar output.

        Optimized with NumPy for vectorized bin filling.

        Args:
            own_x, own_y: Own ship position
            bearing: Current antenna bearing in degrees
            beamwidth: Antenna beamwidth in degrees
            max_range: Maximum radar range in meters
            num_bins: Number of range bins
            return_numpy: If True, return numpy array

        Returns:
            Array of intensities for each range bin
        """
        if self.render_mode == "furuno_hybrid":
            returns = self._generate_furuno_hybrid_returns(
                own_x, own_y, bearing, beamwidth, max_range, num_bins)
        elif self.render_mode == "polar_intersection":
            returns = self._generate_polar_intersection_returns(
                own_x, own_y, bearing, beamwidth, max_range, num_bins)
        else:
            returns = self._generate_simple_returns(
                own_x, own_y, bearing, beamwidth, max_range, num_bins)

        returns = apply_coastline_edge_profile(returns)
        return returns if return_numpy else returns.tolist()

    def _generate_simple_returns(self, own_x: float, own_y: float,
                                 bearing: float, beamwidth: float,
                                 max_range: float, num_bins: int) -> np.ndarray:
        """Legacy coastline painter retained for non-.radarloc scenarios."""
        returns = np.zeros(num_bins)
        bin_size = max_range / num_bins
        half_beam = beamwidth / 2

        if len(self.points) < 3:
            return returns

        if self._points_array is None:
            self._update_points_array()

        num_rays = 5
        for ray_idx in range(num_rays):
            offset = -half_beam + (2 * half_beam * ray_idx / (num_rays - 1)) if num_rays > 1 else 0
            ray_bearing = bearing + offset
            ray_weight = 1.0 - 0.3 * abs(offset) / (half_beam + 0.01)
            hits = self._ray_polygon_intersections_vectorized(own_x, own_y, ray_bearing, max_range)
            if len(hits) == 0:
                continue

            hits = np.sort(hits)
            i = 0
            while i < len(hits):
                enter_dist = hits[i]
                exit_dist = hits[i + 1] if i + 1 < len(hits) else max_range
                i += 2

                start_bin = max(0, int(enter_dist / bin_size))
                end_bin = min(num_bins, int(exit_dist / bin_size) + 1)
                if start_bin < end_bin:
                    bin_range = np.arange(start_bin, end_bin)
                    intensities = np.full(len(bin_range), self.reflectivity * ray_weight)
                    if len(bin_range) > 0:
                        intensities[0] *= 0.75
                    if len(bin_range) > 1:
                        intensities[-1] *= 0.75
                    returns[bin_range] = np.maximum(returns[bin_range], intensities)

        return returns

    def _generate_furuno_hybrid_returns(self, own_x: float, own_y: float,
                                        bearing: float, beamwidth: float,
                                        max_range: float, num_bins: int) -> np.ndarray:
        """Furuno-inspired filled-polygon shoreline rendering.

        This keeps the terrain simulator's signal chain and loader workflow,
        but gives .radarloc coastlines the smoother beam-broadened shoreline
        behavior used in Object Creation's Mode C.
        """
        returns = np.zeros(num_bins, dtype=np.float32)
        if len(self.points) < 3:
            return returns

        if self._points_array is None:
            self._update_points_array()

        profile = self.echo_profile
        if (
            os.environ.get("RADAR_CORE_ENABLE_PER_COASTLINE_RUST", "").lower()
            in {"1", "true", "yes"}
            and os.environ.get("RADAR_CORE_DISABLE_RUST_COASTLINE", "").lower()
            not in {"1", "true", "yes"}
            and rust_bridge.rust_available()
        ):
            try:
                return rust_bridge.polar_intersection_returns(
                    self._points_array,
                    own_x=own_x,
                    own_y=own_y,
                    bearing_deg=bearing,
                    beamwidth_deg=beamwidth,
                    max_range_m=max_range,
                    num_bins=num_bins,
                    reflectivity=self.reflectivity,
                    num_rays=profile.num_rays,
                    edge_factor=profile.edge_factor,
                    leading_edge_boost=profile.leading_edge_boost,
                    far_range_floor=profile.far_range_floor,
                    range_decay_strength=profile.range_decay_strength,
                    core_depth_scale=profile.core_depth_scale,
                    interior_decay_scale=profile.interior_decay_scale,
                    attenuation_model_entry_r4=(
                        profile.attenuation_model == ATTENUATION_ENTRY_R4_DEPTH
                    ),
                    continuous_land_return=profile.continuous_land_return,
                    continuous_return_threshold=profile.continuous_return_threshold,
                    continuous_return_depth_m=profile.continuous_return_depth_m,
                    continuous_texture_strength=profile.continuous_texture_strength,
                    continuous_texture_floor=profile.continuous_texture_floor,
                    origin_range_attenuation=profile.origin_range_attenuation,
                    origin_range_ref_m=profile.origin_range_ref_m,
                    origin_range_power=profile.origin_range_power,
                    origin_range_floor=profile.origin_range_floor,
                    occlude_after_first_land=profile.occlude_after_first_land,
                    secondary_return_scale=profile.secondary_return_scale,
                    secondary_return_max_code=profile.secondary_return_max_code,
                    attenuation_config_values=self._rust_attenuation_config_values(),
                ).astype(np.float32)
            except Exception:
                # Rust is an optional speed path; keep the Python renderer authoritative.
                pass
        bin_size = max_range / max(1, num_bins)
        half_beam = beamwidth / 2.0
        num_rays = max(1, int(profile.num_rays))
        core_depth_bins = max(1, min(4, int(round(num_bins / max(1.0, profile.core_depth_scale)))))
        interior_decay_bins = max(2.0, num_bins / max(1.0, profile.interior_decay_scale))

        for ray_idx in range(num_rays):
            offset = (-half_beam + (2.0 * half_beam * ray_idx / (num_rays - 1))
                      if num_rays > 1 else 0.0)
            ray_bearing = bearing + offset
            ray_weight = 1.0 - 0.3 * abs(offset) / (half_beam + 0.01)
            hits = self._ray_polygon_intersections_vectorized(
                own_x, own_y, ray_bearing, max_range)
            if len(hits) == 0:
                continue

            hits = np.sort(hits)
            land_already_hit = False
            for i in range(0, len(hits), 2):
                enter_dist = hits[i]
                exit_dist = hits[i + 1] if i + 1 < len(hits) else max_range

                start_bin = max(0, int(enter_dist / bin_size))
                end_bin = min(num_bins - 1, int(exit_dist / bin_size))
                if end_bin < start_bin:
                    continue

                if (
                    profile.attenuation_model == ATTENUATION_ENTRY_R4_DEPTH
                    and profile.continuous_land_return
                ):
                    bin_indices = self._continuous_entry_bins(
                        start_bin, enter_dist, bin_size, num_bins, end_bin)
                else:
                    bin_indices = np.arange(start_bin, end_bin + 1)
                if len(bin_indices) == 0:
                    continue

                is_secondary_land_interval = land_already_hit
                bin_ranges = np.maximum(1.0, (bin_indices + 0.5) * bin_size)
                if profile.attenuation_model == ATTENUATION_ENTRY_R4_DEPTH:
                    if profile.continuous_land_return:
                        attenuation = self._continuous_entry_attenuation(
                            bin_ranges, enter_dist)
                    else:
                        attenuation = entry_r4_depth_attenuation(
                            bin_ranges,
                            enter_dist,
                            config=self._attenuation_config(),
                        )
                    intensities = (
                        self.reflectivity
                        * ray_weight
                        * profile.leading_edge_boost
                        * attenuation
                    )
                else:
                    depth_from_entry = np.arange(len(bin_indices), dtype=np.float32)
                    range_ratio = np.clip(bin_ranges / max(1.0, max_range), 0.0, 1.0)
                    range_decay = np.maximum(
                        profile.far_range_floor,
                        1.0 - profile.range_decay_strength * np.power(range_ratio, 1.35),
                    )

                    intensities = np.empty(len(bin_indices), dtype=np.float32)
                    core_mask = depth_from_entry < core_depth_bins
                    if np.any(core_mask):
                        core_depth = max(1, core_depth_bins)
                        intensities[core_mask] = (
                            profile.leading_edge_boost
                            - (depth_from_entry[core_mask] / core_depth)
                            * (profile.leading_edge_boost - 1.0)
                        )
                    if np.any(~core_mask):
                        interior_depth = depth_from_entry[~core_mask] - core_depth_bins + 1.0
                        intensities[~core_mask] = np.exp(-interior_depth / interior_decay_bins)

                    intensities *= self.reflectivity * ray_weight * range_decay
                if len(intensities) > 0:
                    intensities[0] *= profile.edge_factor
                if len(intensities) > 1 and not (
                    profile.attenuation_model == ATTENUATION_ENTRY_R4_DEPTH
                    and profile.continuous_land_return
                ):
                    intensities[-1] *= profile.edge_factor

                intensities *= self._origin_range_attenuation(bin_ranges)
                intensities = self._apply_secondary_land_occlusion(
                    intensities,
                    is_secondary_land_interval,
                )
                intensities = np.clip(intensities, 0.0, 1.0)
                returns[bin_indices] = np.maximum(returns[bin_indices], intensities)
                land_already_hit = True

        return returns

    def _generate_polar_intersection_returns(self, own_x: float, own_y: float,
                                             bearing: float, beamwidth: float,
                                             max_range: float, num_bins: int) -> np.ndarray:
        """Polygon-intersection shoreline renderer for loaded .radarloc scenes.

        The loaded polygon coastline remains the authoritative geometry.
        We cast several sub-rays across the beam, collect entry/exit
        intersections directly from the polygon, smooth their combined return
        profile, and apply inland/range dropoff after the first shoreline hit.

        This keeps the existing signal chain intact while avoiding the support
        terrain grid painting the shoreline a second time.
        """
        returns = np.zeros(num_bins, dtype=np.float32)
        if len(self.points) < 3:
            return returns

        if self._points_array is None:
            self._update_points_array()

        profile = self.echo_profile
        bin_size = max_range / max(1, num_bins)
        half_beam = beamwidth / 2.0
        num_rays = max(3, int(profile.num_rays))
        core_depth_bins = max(1, min(4, int(round(num_bins / max(1.0, profile.core_depth_scale)))))
        interior_decay_bins = max(2.0, num_bins / max(1.0, profile.interior_decay_scale))

        ray_returns = []
        for ray_idx in range(num_rays):
            offset = (-half_beam + (2.0 * half_beam * ray_idx / (num_rays - 1))
                      if num_rays > 1 else 0.0)
            ray_bearing = bearing + offset
            ray_weight = 1.0 - 0.3 * abs(offset) / (half_beam + 0.01)
            hits = self._ray_polygon_intersections_vectorized(
                own_x, own_y, ray_bearing, max_range)
            if len(hits) == 0:
                continue

            hits = np.sort(hits)
            ray_bins = np.zeros(num_bins, dtype=np.float32)
            land_already_hit = False

            for i in range(0, len(hits), 2):
                entry_dist = float(hits[i])
                exit_dist = float(hits[i + 1]) if i + 1 < len(hits) else max_range
                if exit_dist <= entry_dist:
                    continue

                start_bin = max(0, int(entry_dist / bin_size))
                end_bin = min(num_bins - 1, int(exit_dist / bin_size))
                if end_bin < start_bin:
                    continue

                if (
                    profile.attenuation_model == ATTENUATION_ENTRY_R4_DEPTH
                    and profile.continuous_land_return
                ):
                    bin_indices = self._continuous_entry_bins(
                        start_bin, entry_dist, bin_size, num_bins, end_bin)
                else:
                    bin_indices = np.arange(start_bin, end_bin + 1)
                if len(bin_indices) == 0:
                    continue

                is_secondary_land_interval = land_already_hit
                bin_ranges = np.maximum(1.0, (bin_indices + 0.5) * bin_size)
                if profile.attenuation_model == ATTENUATION_ENTRY_R4_DEPTH:
                    if profile.continuous_land_return:
                        attenuation = self._continuous_entry_attenuation(
                            bin_ranges, entry_dist)
                    else:
                        attenuation = entry_r4_depth_attenuation(
                            bin_ranges,
                            entry_dist,
                            config=self._attenuation_config(),
                        )
                    intensities = (
                        self.reflectivity
                        * ray_weight
                        * profile.leading_edge_boost
                        * attenuation
                    )
                else:
                    depth_from_entry = np.arange(len(bin_indices), dtype=np.float32)
                    range_ratio = np.clip(bin_ranges / max(1.0, max_range), 0.0, 1.0)
                    range_decay = np.maximum(
                        profile.far_range_floor,
                        1.0 - profile.range_decay_strength * np.power(range_ratio, 1.35),
                    )

                    intensities = np.empty(len(bin_indices), dtype=np.float32)
                    core_mask = depth_from_entry < core_depth_bins
                    if np.any(core_mask):
                        core_depth = max(1, core_depth_bins)
                        intensities[core_mask] = (
                            profile.leading_edge_boost
                            - (depth_from_entry[core_mask] / core_depth)
                            * (profile.leading_edge_boost - 1.0)
                        )
                    if np.any(~core_mask):
                        interior_depth = depth_from_entry[~core_mask] - core_depth_bins + 1.0
                        intensities[~core_mask] = np.exp(-interior_depth / interior_decay_bins)

                    intensities *= self.reflectivity * ray_weight * range_decay
                intensities[0] *= profile.edge_factor
                if len(intensities) > 1 and not (
                    profile.attenuation_model == ATTENUATION_ENTRY_R4_DEPTH
                    and profile.continuous_land_return
                ):
                    intensities[-1] *= profile.edge_factor
                intensities *= self._origin_range_attenuation(bin_ranges)
                intensities = self._apply_secondary_land_occlusion(
                    intensities,
                    is_secondary_land_interval,
                )
                intensities = np.clip(intensities, 0.0, 1.0)
                ray_bins[bin_indices] = np.maximum(ray_bins[bin_indices], intensities)
                land_already_hit = True

            if np.any(ray_bins > 0):
                ray_returns.append(ray_bins)

        if not ray_returns:
            return returns

        stacked = np.stack(ray_returns, axis=0)
        returns = stacked.max(axis=0)

        # Gentle shoreline smoothing between adjacent intersection-derived
        # points so narrow gaps do not create a dotted edge in the PPI.
        kernel = np.array([0.18, 0.64, 0.18], dtype=np.float32)
        smoothed = np.convolve(returns, kernel, mode="same")
        returns = np.maximum(returns, smoothed * 0.96)
        returns = self._apply_continuous_radar_texture(returns, bearing)
        return np.clip(returns, 0.0, 1.0).astype(np.float32)

    def _ray_polygon_intersections(self, ox: float, oy: float,
                                    bearing: float, max_range: float) -> List[float]:
        """Find all distances where a ray intersects the polygon edges (legacy)."""
        return self._ray_polygon_intersections_vectorized(ox, oy, bearing, max_range).tolist()

    def _ray_polygon_intersections_vectorized(self, ox: float, oy: float,
                                               bearing: float, max_range: float) -> np.ndarray:
        """Vectorized ray-polygon intersection (NumPy optimized)."""
        if self._points_array is None:
            self._update_points_array()
        if self._points_array is None or len(self._points_array) < 2:
            return np.array([], dtype=np.float32)

        ray_rad = math.radians(bearing)
        dx = math.sin(ray_rad)
        dy = math.cos(ray_rad)

        if self._edge_start is None or self._edge_dx is None or self._edge_dy is None:
            self._update_points_array()
        if self._edge_start is None or self._edge_dx is None or self._edge_dy is None:
            return np.array([], dtype=np.float32)

        # Denominator for intersection calculation
        denom = dx * self._edge_dy - dy * self._edge_dx

        # Avoid division by zero
        valid = np.abs(denom) > 1e-10

        if not np.any(valid):
            return np.array([], dtype=np.float32)

        # Calculate intersection parameters only for valid edges
        p1_valid = self._edge_start[valid]
        sx_valid = self._edge_dx[valid]
        sy_valid = self._edge_dy[valid]
        denom_valid = denom[valid]

        t = ((p1_valid[:, 0] - ox) * sy_valid - (p1_valid[:, 1] - oy) * sx_valid) / denom_valid
        u = ((p1_valid[:, 0] - ox) * dy - (p1_valid[:, 1] - oy) * dx) / denom_valid

        # Filter valid intersections
        hit_mask = (t > 0) & (u >= 0) & (u <= 1) & (t <= max_range)

        return t[hit_mask]

    def is_point_on_land(self, x: float, y: float) -> bool:
        """Check if a point is on land (inside the coastline polygon)."""
        if len(self.points) < 3:
            return False

        # Ray casting algorithm
        inside = False
        j = len(self.points) - 1

        for i in range(len(self.points)):
            pi = self.points[i]
            pj = self.points[j]

            if ((pi.y > y) != (pj.y > y) and
                x < (pj.x - pi.x) * (y - pi.y) / (pj.y - pi.y) + pi.x):
                inside = not inside
            j = i

        return inside


def create_harbor_coastline(center_x: float = 0, center_y: float = 8000,
                           width: float = 15000, depth: float = 4000) -> Coastline:
    """Create a harbor-shaped coastline ahead of own ship."""
    coastline = Coastline()

    points = [
        (center_x - width/2, center_y - depth),
        (center_x - width/2, center_y),
        (center_x - width/6, center_y),
        (center_x - width/6, center_y + depth/2),
        (center_x + width/6, center_y + depth/2),
        (center_x + width/6, center_y),
        (center_x + width/2, center_y),
        (center_x + width/2, center_y - depth),
    ]

    for x, y in points:
        coastline.add_point(x, y)

    coastline.add_point(center_x + width/2, center_y + depth * 2)
    coastline.add_point(center_x - width/2, center_y + depth * 2)
    coastline.add_point(center_x - width/2, center_y - depth)

    return coastline


def create_island_coastline(center_x: float, center_y: float,
                           radius: float, num_points: int = 24) -> Coastline:
    """Create a roughly circular island."""
    import random
    coastline = Coastline()

    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        r = radius * (0.8 + 0.4 * random.random())
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        coastline.add_point(x, y)

    if coastline.points:
        coastline.add_point(coastline.points[0].x, coastline.points[0].y)

    return coastline


def create_straight_coastline(start_x: float, start_y: float,
                             end_x: float, end_y: float,
                             land_depth: float = 5000) -> Coastline:
    """Create a straight coastline with land behind it."""
    coastline = Coastline()

    dx = end_x - start_x
    dy = end_y - start_y
    length = math.sqrt(dx*dx + dy*dy)

    px = -dy / length
    py = dx / length

    coastline.add_point(start_x, start_y)
    coastline.add_point(end_x, end_y)
    coastline.add_point(end_x + px * land_depth, end_y + py * land_depth)
    coastline.add_point(start_x + px * land_depth, start_y + py * land_depth)
    coastline.add_point(start_x, start_y)

    return coastline
