"""Main simulation engine with terrain occlusion support."""
import math
import os
import random
import time
from typing import Optional, Callable, List
import numpy as np
from radar_core import rust_bridge
from .world import World
from ..radar.system import RadarSystem
from ..radar.signal_processing import process_signal_chain
from ..environment.weather import WeatherEffects, WeatherConditions
from ..environment.coastline import (
    Coastline,
    apply_coastline_edge_profile,
    create_harbor_coastline,
    create_island_coastline,
)
from ..environment.terrain import HeightMap
from ..environment.occlusion import OcclusionEngine
from ..environment.attenuation import (
    ATTENUATION_ENTRY_R4_DEPTH,
    env_attenuation_model,
    normalize_attenuation_model,
)
from ..data_export import RadarDataExporter

class Simulation:
    """Main simulation controller."""

    def __init__(self):
        self.world = World()
        self.radar = RadarSystem()
        self.weather = WeatherEffects()
        self.exporter = RadarDataExporter()

        # Coastline/landmass
        self.coastlines: List[Coastline] = []
        self.coastline_enabled = False

        # Terrain / occlusion
        self.terrain_maps: List[HeightMap] = []
        self.occlusion_engine: Optional[OcclusionEngine] = None
        # Match the launcher and location loader for every loaded coastline.
        self.loaded_coastline_attenuation_model = env_attenuation_model(ATTENUATION_ENTRY_R4_DEPTH)

        # Timing
        self.time_scale = 1.0  # 1.0 = real-time
        self.target_fps = 60
        self.dt = 1.0 / self.target_fps

        # State
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0

        # Annotation collection (populated each rotation for export)
        self._current_annotations: List[dict] = []
        self._scene_echo_preview_cache: Optional[List[tuple]] = None
        self._scene_echo_preview_key = None
        self._scene_echo_preview_version = 0
        self.loaded_location_path: Optional[str] = None
        self.temporary_own_origin: Optional[tuple[float, float]] = None

        # Callbacks
        self.on_update: Optional[Callable[[float], None]] = None

    def set_temporary_own_origin(self, x: float, y: float) -> None:
        """Set a session-only radar origin override for loaded .radarloc files."""
        self.temporary_own_origin = (float(x), float(y))

    # ------------------------------------------------------------------
    # Coastlines
    # ------------------------------------------------------------------
    def add_coastline(self, coastline: Coastline) -> None:
        """Add a coastline to the simulation."""
        if getattr(coastline, "render_mode", "simple") in ("furuno_hybrid", "polar_intersection"):
            coastline.echo_profile.attenuation_model = self.loaded_coastline_attenuation_model
        self.coastlines.append(coastline)
        self.coastline_enabled = True
        self._invalidate_scene_echo_preview()

    def clear_coastlines(self) -> None:
        """Remove all coastlines."""
        self.coastlines.clear()
        self.coastline_enabled = False
        self._invalidate_scene_echo_preview()

    def setup_harbor_coastline(self) -> None:
        """Set up a default harbor coastline ahead of own ship."""
        self.clear_coastlines()
        harbor = create_harbor_coastline(
            center_x=0, center_y=8000, width=12000, depth=3000
        )
        self.add_coastline(harbor)
        island = create_island_coastline(
            center_x=4000, center_y=5000, radius=500, num_points=16
        )
        self.add_coastline(island)

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------
    def add_terrain(self, terrain: HeightMap) -> None:
        """Add a terrain height map and rebuild the occlusion engine."""
        self.terrain_maps.append(terrain)
        self._rebuild_occlusion_engine()
        self._invalidate_scene_echo_preview()

    def clear_terrain(self) -> None:
        """Remove all terrain maps and disable occlusion."""
        self.terrain_maps.clear()
        self.occlusion_engine = None
        self.radar.detection_engine.occlusion_engine = None
        self._invalidate_scene_echo_preview()

    def move_terrain(self, index: int, new_cx: float, new_cy: float) -> None:
        """Move a terrain map so its center is at (new_cx, new_cy)."""
        if 0 <= index < len(self.terrain_maps):
            self.terrain_maps[index].move_center(new_cx, new_cy)
            self._rebuild_occlusion_engine()
            self._invalidate_scene_echo_preview()

    def remove_terrain(self, index: int) -> None:
        """Remove a single terrain map by index."""
        if 0 <= index < len(self.terrain_maps):
            self.terrain_maps.pop(index)
            self._rebuild_occlusion_engine()
            self._invalidate_scene_echo_preview()

    def add_vessel_at(self, vessel_type_str: str, x: float, y: float) -> None:
        """Add a new vessel of the given type at (x, y)."""
        from ..objects.vessel import Vessel, VesselType

        type_map = {
            "cargo": (VesselType.CARGO, "Cargo Ship", 150, 20, 25),
            "tanker": (VesselType.TANKER, "Tanker", 200, 30, 20),
            "fishing": (VesselType.FISHING, "Fishing Boat", 25, 6, 8),
            "sailing": (VesselType.SAILING, "Sailing Yacht", 15, 4, 15),
            "tug": (VesselType.TUG, "Tug", 30, 10, 12),
            "passenger": (VesselType.PASSENGER, "Ferry", 100, 20, 25),
            "pilot": (VesselType.PILOT, "Pilot Boat", 15, 5, 6),
            "buoy": (VesselType.BUOY, "Buoy", 3, 3, 4),
        }

        vtype, name, length, beam, height = type_map.get(
            vessel_type_str, (VesselType.CARGO, "Ship", 100, 15, 20))

        # Generate unique id
        existing_ids = set(self.world.vessels.keys())
        idx = 1
        while f"placed_{vessel_type_str}_{idx}" in existing_ids:
            idx += 1
        vid = f"placed_{vessel_type_str}_{idx}"

        vessel = Vessel(
            id=vid, name=f"{name} {idx}",
            vessel_type=vtype, x=x, y=y,
            course=0, speed=0,
            length=length, beam=beam, height=height)
        self.world.add_vessel(vessel)

    def remove_vessel(self, vessel_id: str) -> None:
        """Remove a vessel by ID (not own ship)."""
        vessel = self.world.get_vessel(vessel_id)
        if vessel and vessel is not self.world.own_ship:
            self.world.remove_vessel(vessel_id)

    def _point_is_water(self, wx: float, wy: float) -> bool:
        """Return True when a point lies on water for the current loaded scene."""
        if not self.terrain_maps:
            return True

        checked_extent = False
        for terrain in self.terrain_maps:
            min_x, min_y, max_x, max_y = terrain.world_extent
            if min_x <= wx <= max_x and min_y <= wy <= max_y:
                checked_extent = True
                if terrain.get_elevation(wx, wy) > 0.5:
                    return False
        return checked_extent

    def populate_random_targets(
        self,
        count: int = 6,
        *,
        seed: Optional[int] = None,
        min_range_m: float = 40.0,
        max_range_m: Optional[float] = None,
    ) -> List[str]:
        """Populate water-only random targets around own ship."""
        from ..objects.vessel import Vessel, VesselType

        own_ship = self.world.own_ship
        if own_ship is None:
            return []

        rng = random.Random(seed)
        max_range = max_range_m or (self.radar.params.max_range_m * 0.82)
        target_specs = [
            (VesselType.CARGO, "Cargo Ship", 150, 20, 25, (0.0, 12.0)),
            (VesselType.TANKER, "Tanker", 200, 30, 22, (0.0, 10.0)),
            (VesselType.FISHING, "Fishing Boat", 25, 6, 8, (0.0, 8.0)),
            (VesselType.SAILING, "Sailing Yacht", 15, 4, 15, (0.0, 6.0)),
            (VesselType.PASSENGER, "Ferry", 100, 20, 25, (0.0, 16.0)),
            (VesselType.TUG, "Tug", 30, 10, 12, (0.0, 10.0)),
            (VesselType.PILOT, "Pilot Boat", 15, 5, 6, (0.0, 18.0)),
            (VesselType.BUOY, "Buoy", 3, 3, 4, (0.0, 0.0)),
        ]

        created_ids: List[str] = []
        existing_targets = self.world.get_targets()
        min_sep_m = 28.0

        for created_idx in range(max(0, int(count))):
            placed = False
            for _ in range(300):
                bearing_deg = rng.uniform(0.0, 360.0)
                range_m = rng.uniform(min_range_m, max_range)
                bearing_rad = math.radians(bearing_deg)
                x = own_ship.x + range_m * math.sin(bearing_rad)
                y = own_ship.y + range_m * math.cos(bearing_rad)

                if not self._point_is_water(x, y):
                    continue
                if any((x - v.x) ** 2 + (y - v.y) ** 2 < min_sep_m ** 2 for v in existing_targets):
                    continue

                vtype, base_name, length, beam, height, speed_range = rng.choice(target_specs)
                speed = rng.uniform(*speed_range) if speed_range[1] > 0.0 else 0.0
                course = rng.uniform(0.0, 360.0)
                vessel_id = f"random_{vtype.value}_{len(existing_targets) + 1}"
                vessel = Vessel(
                    id=vessel_id,
                    name=f"{base_name} {created_idx + 1}",
                    vessel_type=vtype,
                    x=x,
                    y=y,
                    course=course,
                    speed=speed,
                    length=length,
                    beam=beam,
                    height=height,
                )
                self.world.add_vessel(vessel)
                existing_targets.append(vessel)
                created_ids.append(vessel_id)
                placed = True
                break
            if not placed:
                break

        return created_ids

    def _invalidate_scene_echo_preview(self) -> None:
        """Invalidate cached scene-view shoreline preview sweeps."""
        self._scene_echo_preview_version += 1
        self._scene_echo_preview_cache = None
        self._scene_echo_preview_key = None

    def _rebuild_occlusion_engine(self) -> None:
        """(Re)create the OcclusionEngine from current terrain maps."""
        if self.terrain_maps:
            self.occlusion_engine = OcclusionEngine(
                self.terrain_maps,
                antenna_height_m=self.radar.params.antenna_height_m,
            )
            self.radar.detection_engine.occlusion_engine = self.occlusion_engine
        else:
            self.occlusion_engine = None
            self.radar.detection_engine.occlusion_engine = None

    def set_loaded_coastline_attenuation_model(self, model: str) -> str:
        """Set shoreline attenuation for loaded-location coastline renderers."""
        normalized = normalize_attenuation_model(model)
        self.loaded_coastline_attenuation_model = normalized
        for coastline in self.coastlines:
            if getattr(coastline, "render_mode", "simple") in ("furuno_hybrid", "polar_intersection"):
                coastline.echo_profile.attenuation_model = normalized
        self._invalidate_scene_echo_preview()
        return normalized

    def _loaded_coastline_attenuation_config(self) -> dict:
        def env_float(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        def env_int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return int(float(raw))
            except (TypeError, ValueError):
                return default

        return {
            "penetration_ref_m": env_float("RADARLOC_RAY_ATTEN_LREF_M", 8.0),
            "penetration_ref_range_m": env_float("RADARLOC_RAY_ATTEN_RREF_M", 200.0),
            "penetration_power": env_float("RADARLOC_RAY_ATTEN_POWER", 0.5),
            "penetration_min_m": env_float("RADARLOC_RAY_ATTEN_LMIN_M", 2.0),
            "penetration_max_m": env_float("RADARLOC_RAY_ATTEN_LMAX_M", 20.0),
            "core_depth_m": env_float("RADARLOC_RAY_ATTEN_CORE_M", 1.5),
            "floor": env_float("RADARLOC_RAY_ATTEN_FLOOR", 0.0),
            "continuous_decay_m": env_float("RADARLOC_SOLIDIFY_DECAY_M", 420.0),
            "continuous_floor": env_float("RADARLOC_SOLIDIFY_FLOOR", 0.24),
            "continuous_range_power": env_float("RADARLOC_SOLIDIFY_RANGE_POWER", 0.22),
            "continuous_num_rays": env_int("RADARLOC_SOLIDIFY_NUM_RAYS", 9),
            "continuous_gap_close_m": env_float("RADARLOC_SOLIDIFY_GAP_CLOSE_M", 180.0),
            "continuous_azimuth_fill": env_float("RADARLOC_SOLIDIFY_AZIMUTH_FILL", 0.70),
            "continuous_azimuth_passes": env_int("RADARLOC_SOLIDIFY_AZIMUTH_PASSES", 2),
        }

    def _should_solidify_loaded_coastline(self) -> bool:
        if self.loaded_coastline_attenuation_model != ATTENUATION_ENTRY_R4_DEPTH:
            return False
        if self._use_coastline_terrain_hybrid():
            return True
        return self.coastline_enabled and any(
            getattr(coastline, "render_mode", "simple") == "polar_intersection"
            for coastline in self.coastlines
        )

    def _angular_solidify_rows(self, sweep_rows: list) -> list:
        """Bridge only bounded bearing gaps in loaded coastline R4 output.

        A display beam softens a shoreline across adjacent bearings, but it
        does not clone a return outward from one bearing.  Two-sided bridging
        removes raster stair steps while preserving narrow water lanes and
        avoids the broad one-sided smears produced by a neighbor maximum.
        """
        if not self._should_solidify_loaded_coastline() or len(sweep_rows) < 3:
            return sweep_rows

        arr = np.asarray(sweep_rows, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return sweep_rows

        config = self._loaded_coastline_attenuation_config()
        fill = float(config.get("continuous_azimuth_fill", 0.82))
        passes = max(1, min(4, int(config.get("continuous_azimuth_passes", 2))))
        try:
            threshold_code = float(os.environ.get("RADARLOC_AZIMUTH_SMOOTH_THRESHOLD_CODE", 8.0))
        except (TypeError, ValueError):
            threshold_code = 8.0
        threshold = max(0.0, min(1.0, threshold_code / 252.0))
        for _ in range(passes):
            left = np.roll(arr, 1, axis=0)
            right = np.roll(arr, -1, axis=0)
            # Geometric mean stays below both neighbors and therefore cannot
            # turn a narrow one-sided fragment into a new coastline body.
            bridged = np.sqrt(np.maximum(left, 0.0) * np.maximum(right, 0.0)) * fill
            gap = (arr < threshold) & (left >= threshold) & (right >= threshold)
            arr = np.where(gap, np.maximum(arr, bridged), arr)

        return np.clip(arr, 0.0, 1.0).tolist()

    def _suppress_isolated_outer_returns(self, sweep_rows: list) -> list:
        """Remove small detached post-coastline echo islands.

        This cleanup runs after the ordinary shoreline equations, halo, and
        angular fill. It does not alter the main coastline body. Instead, it
        looks past the first strong coastline layer on each bearing and removes
        compact components that are detached from that layer, which are usually
        tiny water-gap or polygon-fragment returns that real radar would not see
        after an occluding shoreline.
        """
        raw_enabled = os.environ.get("RADARLOC_SUPPRESS_OUTER_ISOLATED_RETURNS", "0")
        if raw_enabled.strip().lower() not in {"1", "true", "yes", "on"}:
            return sweep_rows
        if not self._should_solidify_loaded_coastline() or len(sweep_rows) < 3:
            return sweep_rows

        arr = np.asarray(sweep_rows, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return sweep_rows

        def env_float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        def env_int(name: str, default: int) -> int:
            try:
                return int(float(os.environ.get(name, default)))
            except (TypeError, ValueError):
                return default

        first_layer_threshold = max(
            0.0,
            min(1.0, env_float("RADARLOC_OUTER_COMPONENT_FIRST_LAYER_CODE", 88.0) / 252.0),
        )
        component_threshold = max(
            0.0,
            min(1.0, env_float("RADARLOC_OUTER_COMPONENT_THRESHOLD_CODE", 20.0) / 252.0),
        )
        min_range_fraction = max(
            0.0,
            min(1.0, env_float("RADARLOC_OUTER_COMPONENT_MIN_RANGE_FRACTION", 0.65)),
        )
        shoulder_bins = max(0, env_int("RADARLOC_OUTER_COMPONENT_SHOULDER_BINS", 8))
        max_area = max(1, env_int("RADARLOC_OUTER_COMPONENT_MAX_AREA", 900))
        max_bearings = max(1, env_int("RADARLOC_OUTER_COMPONENT_MAX_BEARINGS", 45))
        max_range_bins = max(1, env_int("RADARLOC_OUTER_COMPONENT_MAX_RANGE_BINS", 80))

        rows, cols = arr.shape
        candidate = np.zeros((rows, cols), dtype=bool)
        for row_idx in range(rows):
            strong = np.flatnonzero(arr[row_idx] >= first_layer_threshold)
            if strong.size == 0:
                continue

            first_start = int(strong[0])
            first_end = first_start
            last = first_start
            for col_idx in strong[1:]:
                col_idx = int(col_idx)
                if col_idx - last > shoulder_bins:
                    break
                first_end = col_idx
                last = col_idx

            tail_start = min(cols, first_end + shoulder_bins + 1)
            tail_start = max(tail_start, int(round(cols * min_range_fraction)))
            if tail_start < cols:
                candidate[row_idx, tail_start:] = arr[row_idx, tail_start:] >= component_threshold

        if not np.any(candidate):
            return arr.tolist()

        visited = np.zeros_like(candidate, dtype=bool)
        cleaned = arr.copy()

        for start_row, start_col in np.argwhere(candidate & ~visited):
            stack = [(int(start_row), int(start_col))]
            visited[start_row, start_col] = True
            pixels = []
            min_row = max_row = int(start_row)
            min_col = max_col = int(start_col)

            while stack:
                row_idx, col_idx = stack.pop()
                pixels.append((row_idx, col_idx))
                min_row = min(min_row, row_idx)
                max_row = max(max_row, row_idx)
                min_col = min(min_col, col_idx)
                max_col = max(max_col, col_idx)

                for next_row, next_col in (
                    ((row_idx - 1) % rows, col_idx),
                    ((row_idx + 1) % rows, col_idx),
                    (row_idx, col_idx - 1),
                    (row_idx, col_idx + 1),
                ):
                    if next_col < 0 or next_col >= cols:
                        continue
                    if candidate[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))

            area = len(pixels)
            bearing_span = max_row - min_row + 1
            range_span = max_col - min_col + 1
            if area <= max_area and bearing_span <= max_bearings and range_span <= max_range_bins:
                for row_idx, col_idx in pixels:
                    cleaned[row_idx, col_idx] = 0.0

        return np.clip(cleaned, 0.0, 1.0).tolist()

    def _use_coastline_terrain_hybrid(self) -> bool:
        """Whether terrain should own the shoreline-return model.

        For .radarloc-loaded scenes without DEM, the loader synthesizes a
        land/water height grid from shoreline polygons. In those cases the
        terrain grid is the authoritative land mask, so we let terrain paint
        Furuno-style shoreline returns and suppress separate polygon returns.
        """
        return bool(self.terrain_maps) and self.coastline_enabled and all(
            getattr(terrain, "generated_from_coastline", False)
            for terrain in self.terrain_maps
        ) and all(
            getattr(coastline, "render_mode", "simple") == "furuno_hybrid"
            for coastline in self.coastlines
        )

    def _suppress_support_terrain_returns(self) -> bool:
        """Whether support terrain should be kept for scene/occlusion only.

        Loaded .radarloc scenes synthesize a terrain grid from coastline
        polygons so the scene view has land/water semantics and vessel
        occlusion still works. When coastlines are rendered directly from the
        polygon intersections, that support terrain must not also paint radar
        land returns or the shoreline gets double-counted.
        """
        return bool(self.terrain_maps) and self.coastline_enabled and all(
            getattr(terrain, "generated_from_coastline", False)
            for terrain in self.terrain_maps
        ) and any(
            getattr(coastline, "render_mode", "simple") != "furuno_hybrid"
            for coastline in self.coastlines
        )

    def _classify_primary_coastlines(self) -> set:
        """Split loaded coastlines into primary (own-ship water) vs secondary.

        Primary coastlines border the water body the own ship floats on: the
        main lake/harbor ring and any islands inside it. Secondary coastlines
        are everything else -- ponds, creeks, and basins that sit behind the
        primary shoreline. Real radar cannot see those at all, so they are
        subject to strict occlusion instead of the soft secondary cap.

        Classification: flood-fill the own-ship water region on the
        coastline-derived terrain grid, then mark a coastline primary when any
        of its vertices lies within a couple of cells of that region.
        Returns a set of indices into self.coastlines.
        """
        cache_key = (id(self.coastlines), len(self.coastlines))
        cached = getattr(self, "_primary_coastline_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        primary = set(range(len(self.coastlines)))  # fail-open: all primary
        terrain = None
        for candidate in (self.terrain_maps or []):
            if getattr(candidate, "generated_from_coastline", False):
                terrain = candidate
                break
        if terrain is None and self.terrain_maps:
            terrain = self.terrain_maps[0]

        if terrain is not None and self.coastlines:
            try:
                from collections import deque
                grid = terrain.grid
                rows, cols = grid.shape
                cell = float(terrain.config.cell_size)
                gx0 = float(terrain.config.origin_x)
                gy0 = float(terrain.config.origin_y)
                water = grid <= 0.5

                def to_rc(x: float, y: float):
                    c = int((x - gx0) / cell)
                    r = int((y - gy0) / cell)
                    return r, c

                # Flood own-ship water from the scene origin (radar position
                # in .radarloc frames). Search outward briefly if the origin
                # cell itself rasterized onto shoreline.
                seed = None
                r0, c0 = to_rc(0.0, 0.0)
                for radius in range(0, 8):
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            rr, cc = r0 + dr, c0 + dc
                            if 0 <= rr < rows and 0 <= cc < cols and water[rr, cc]:
                                seed = (rr, cc)
                                break
                        if seed:
                            break
                    if seed:
                        break

                if seed is not None:
                    connected = np.zeros_like(water, dtype=bool)
                    queue = deque([seed])
                    connected[seed] = True
                    while queue:
                        r, c = queue.popleft()
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and water[nr, nc] and not connected[nr, nc]:
                                connected[nr, nc] = True
                                queue.append((nr, nc))

                    # Dilate the connected-water test by 2 cells so shoreline
                    # vertices (which sit ON the land/water edge) qualify.
                    reach = connected.copy()
                    for _ in range(2):
                        grown = reach.copy()
                        grown[1:, :] |= reach[:-1, :]
                        grown[:-1, :] |= reach[1:, :]
                        grown[:, 1:] |= reach[:, :-1]
                        grown[:, :-1] |= reach[:, 1:]
                        reach = grown

                    primary = set()
                    for idx, coastline in enumerate(self.coastlines):
                        pts = coastline.points
                        if not pts:
                            continue
                        stride = max(1, len(pts) // 200)
                        for point in pts[::stride]:
                            r, c = to_rc(float(point.x), float(point.y))
                            if 0 <= r < rows and 0 <= c < cols and reach[r, c]:
                                primary.add(idx)
                                break
            except Exception:
                primary = set(range(len(self.coastlines)))

        self._primary_coastline_cache = (cache_key, primary)
        return primary

    @staticmethod
    def _first_land_body_cap_start(arr: np.ndarray, threshold: float,
                                   gap_bins: int, shoulder_bins: int) -> Optional[int]:
        """Return the bin index just past the first land body, or None."""
        active = np.flatnonzero(arr >= threshold)
        if active.size == 0:
            return None
        first_end = int(active[0])
        last_active = first_end
        for idx in active[1:]:
            idx = int(idx)
            if idx - last_active > gap_bins:
                break
            first_end = idx
            last_active = idx
        return min(arr.size, first_end + shoulder_bins + 1)

    def _apply_own_ship_range_falloff(self, sweep: np.ndarray, num_bins: int) -> np.ndarray:
        """Second attenuation stage: own-ship r^4-style falloff to zero.

        The per-coastline origin-range attenuation follows (ref/r)^power but is
        floored, so distant returns survive and the log-compression stage then
        re-lifts them into visible echoes. Real displays lose shoreline near
        maximum range entirely. This multiplies the combined environment sweep
        by a fade that tracks the r^4 curve and reaches exactly zero at the
        range edge. Applied at the sweep level so the Rust batch path gets the
        identical treatment.
        """
        raw_enabled = os.environ.get("RADARLOC_ORIGIN_EDGE_FADE", "1")
        if raw_enabled.strip().lower() not in {"1", "true", "yes", "on"}:
            return sweep

        try:
            fade_start = float(os.environ.get("RADARLOC_ORIGIN_EDGE_FADE_START", 0.78))
        except (TypeError, ValueError):
            fade_start = 0.78
        fade_start = max(0.05, min(0.98, fade_start))

        cache = getattr(self, "_edge_fade_cache", None)
        if cache is not None and cache[0] == (num_bins, fade_start):
            fade = cache[1]
        else:
            positions = (np.arange(num_bins, dtype=np.float32) + 0.5) / float(num_bins)
            fade = np.ones(num_bins, dtype=np.float32)
            ramp = positions >= fade_start
            t = (positions[ramp] - fade_start) / max(1e-6, 1.0 - fade_start)
            # Cosine ramp: smooth like the r^4 knee, and exactly 0.0 at max range.
            fade[ramp] = 0.5 * (1.0 + np.cos(np.pi * np.clip(t, 0.0, 1.0)))
            fade[-1] = 0.0
            self._edge_fade_cache = ((num_bins, fade_start), fade)

        return (np.asarray(sweep, dtype=np.float32) * fade).astype(np.float32)

    def _apply_combined_coastline_occlusion(self, coastline_returns: np.ndarray) -> np.ndarray:
        """Cap coastline returns behind the first combined shoreline body.

        Individual .radarloc polygons can be split into many small rings. A
        later polygon on the same bearing should not become a fresh bright
        shoreline if the ray already crossed a nearer land body.
        """
        raw_enabled = os.environ.get("RADARLOC_SWEEP_OCCLUDE_AFTER_FIRST_LAND", "1")
        if raw_enabled.strip().lower() not in {"1", "true", "yes", "on"}:
            return coastline_returns

        arr = np.asarray(coastline_returns, dtype=np.float32).copy()
        if arr.size == 0 or not np.any(arr > 0.0):
            return arr

        def env_float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        def env_int(name: str, default: int) -> int:
            try:
                return int(float(os.environ.get(name, default)))
            except (TypeError, ValueError):
                return default

        threshold = max(0.0, min(1.0, env_float("RADARLOC_SWEEP_OCCLUSION_THRESHOLD_CODE", 20.0) / 252.0))
        gap_bins = max(0, env_int("RADARLOC_SWEEP_OCCLUSION_GAP_BINS", 6))
        shoulder_bins = max(0, env_int("RADARLOC_SWEEP_OCCLUSION_SHOULDER_BINS", 2))
        secondary_scale = max(0.0, min(1.0, env_float("RADARLOC_SWEEP_SECONDARY_RETURN_SCALE", 0.035)))
        secondary_cap = max(0.0, min(1.0, env_float("RADARLOC_SWEEP_SECONDARY_RETURN_MAX_CODE", 20.0) / 252.0))

        active = np.flatnonzero(arr >= threshold)
        if active.size == 0:
            return arr

        # Cumulative multi-body occlusion: each successive shoreline body a
        # bearing crosses multiplies the attenuation for everything behind it.
        # One body behind the first shoreline stays a faint differentiator;
        # anything behind several coastlines vanishes, as on a real display.
        # Bodies are detected on the ORIGINAL profile so later bodies are not
        # missed merely because an earlier cap dimmed them.
        original = arr.copy()
        cursor = 0
        bodies_crossed = 0
        for _ in range(16):  # bounded: more than 16 bodies never happens
            active = np.flatnonzero(original[cursor:] >= threshold)
            if active.size == 0:
                break
            body_start = cursor + int(active[0])
            body_end = body_start
            last_active = body_start
            for idx in active[1:]:
                idx = cursor + int(idx)
                if idx - last_active > gap_bins:
                    break
                body_end = idx
                last_active = idx
            cap_start = min(arr.size, body_end + shoulder_bins + 1)
            if cap_start >= arr.size:
                break
            bodies_crossed += 1
            scale_k = secondary_scale ** bodies_crossed
            cap_k = secondary_cap * (secondary_scale ** (bodies_crossed - 1))
            arr[cap_start:] = np.minimum(original[cap_start:] * scale_k, cap_k)
            cursor = cap_start
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def _build_environment_linear_sweep(
        self, own_x: float, own_y: float, bearing: float, num_bins: int
    ) -> list:
        """Build coastline/terrain returns in linear power domain only."""
        sweep_data = [0.0] * num_bins
        use_coastline_terrain_hybrid = self._use_coastline_terrain_hybrid()

        if self.coastline_enabled and not use_coastline_terrain_hybrid:
            primary_indices = self._classify_primary_coastlines()
            mapped_inland_classes = {
                "water", "river", "canal", "stream", "pond", "lake", "basin",
                "reservoir", "harbour", "harbor", "lagoon", "bay", "fairway",
                "strait", "wetland", "marsh", "saltmarsh", "tidalflat", "reedbed",
                "swamp", "wet_meadow", "mangrove", "bog", "fen",
            }
            # Three tiers, split by index so membership stays O(1) and never
            # depends on Coastline equality semantics:
            #   primary          - own-ship water boundary + islands in it
            #   mapped_inland    - tertiary: mapped water/marsh behind the shore
            #   strict_secondary - land features behind the shore (ponds' banks,
            #                      creek edges) that real radar cannot see
            primary_group, mapped_inland_group, strict_secondary_group = [], [], []
            for idx, coastline in enumerate(self.coastlines):
                if idx in primary_indices:
                    primary_group.append(coastline)
                elif str(getattr(coastline, "source_feature_class", "")).strip().lower() in mapped_inland_classes:
                    mapped_inland_group.append(coastline)
                else:
                    strict_secondary_group.append(coastline)
            secondary_group = mapped_inland_group + strict_secondary_group

            def render_group(group) -> np.ndarray:
                if not group:
                    return np.zeros(num_bins, dtype=np.float32)
                batch = self._try_rust_batched_coastline_sweep(
                    own_x, own_y, bearing, num_bins, coastlines=group)
                if batch is not None:
                    return apply_coastline_edge_profile(batch).astype(np.float32)
                combined = np.zeros(num_bins, dtype=np.float32)
                for coastline in group:
                    coast_returns = coastline.generate_returns(
                        own_x,
                        own_y,
                        bearing,
                        self.radar.params.horizontal_beamwidth_deg,
                        self.radar.params.max_range_m,
                        num_bins,
                    )
                    combined = np.maximum(
                        combined, np.asarray(coast_returns, dtype=np.float32))
                return combined

            # Primary shorelines (own-ship water boundary + islands in it)
            # bypass strict occlusion: they get the entry->interior ramp from
            # generation plus the soft combined cap for far-side land.
            primary_sweep = self._apply_combined_coastline_occlusion(
                render_group(primary_group))

            # Secondary features (ponds/creeks behind the primary shoreline)
            # are strictly occluded past the first primary land hit: real
            # radar never sees new objects surface behind a coastline.
            secondary_sweep = render_group(strict_secondary_group)
            mapped_inland_sweep = render_group(mapped_inland_group)
            if secondary_group and (np.any(secondary_sweep > 0.0) or np.any(mapped_inland_sweep > 0.0)):
                def env_float(name: str, default: float) -> float:
                    try:
                        return float(os.environ.get(name, default))
                    except (TypeError, ValueError):
                        return default

                threshold = max(0.0, min(1.0, env_float(
                    "RADARLOC_SWEEP_OCCLUSION_THRESHOLD_CODE", 20.0) / 252.0))
                gap_bins = max(0, int(env_float("RADARLOC_SWEEP_OCCLUSION_GAP_BINS", 6)))
                shoulder_bins = max(0, int(env_float("RADARLOC_SWEEP_OCCLUSION_SHOULDER_BINS", 2)))
                strict_scale = max(0.0, min(1.0, env_float(
                    "RADARLOC_STRICT_SECONDARY_OCCLUSION_SCALE", 0.0)))

                cap_start = self._first_land_body_cap_start(
                    primary_sweep, threshold, gap_bins, shoulder_bins)
                if cap_start is not None and cap_start < secondary_sweep.size:
                    # Within the interior-fill band behind the first
                    # shoreline, mapped ponds/marsh still texture the
                    # return (dim); beyond it they are strictly occluded.
                    fill_depth_m = env_float("RADARLOC_CONTINUOUS_RETURN_DEPTH_M", 1100.0)
                    near_scale = max(strict_scale, min(1.0, env_float(
                        "RADARLOC_SECONDARY_INBAND_SCALE", 0.30)))
                    bin_m = self.radar.params.max_range_m / max(1, num_bins)
                    fill_bins = int(max(0.0, fill_depth_m) / max(bin_m, 1.0))
                    band_end = min(secondary_sweep.size, cap_start + fill_bins)
                    secondary_sweep = secondary_sweep.copy()
                    secondary_sweep[cap_start:band_end] *= near_scale
                    secondary_sweep[band_end:] *= strict_scale
                    # Ray-reached marsh and water detail can remain faintly
                    # visible inland of the first coast, without becoming a
                    # fresh bright shoreline behind it.
                    mapped_scale = max(0.0, min(1.0, env_float(
                        "RADARLOC_MAPPED_INLAND_RETURN_SCALE", 0.035)))
                    mapped_cap = max(0.0, min(1.0, env_float(
                        "RADARLOC_MAPPED_INLAND_RETURN_MAX_CODE", 18.0) / 252.0))
                    mapped_inland_sweep = mapped_inland_sweep.copy()
                    mapped_inland_sweep[cap_start:band_end] *= near_scale
                    mapped_inland_sweep[band_end:] = np.minimum(
                        mapped_inland_sweep[band_end:] * mapped_scale, mapped_cap
                    )
                else:
                    # No usable primary land on this bearing -- either none was
                    # crossed, or the only primary body sits at the very edge
                    # (cap_start beyond the array). Either way, fall back to the
                    # soft combined occlusion so the nearest secondary body
                    # reads as one shoreline instead of the whole chain lighting
                    # up at full strength. (Previously the edge case silently
                    # skipped occlusion, leaving secondary features full-bright.)
                    secondary_sweep = self._apply_combined_coastline_occlusion(
                        secondary_sweep)
                    mapped_inland_sweep = self._apply_combined_coastline_occlusion(
                        mapped_inland_sweep)

            combined_coastline = np.maximum(
                primary_sweep, np.maximum(secondary_sweep, mapped_inland_sweep)
            )
            combined_coastline = self._apply_own_ship_range_falloff(
                combined_coastline, num_bins)
            sweep_data = np.maximum(
                np.asarray(sweep_data, dtype=np.float32),
                combined_coastline * self.radar.params.gain,
            ).tolist()

        if self.occlusion_engine is not None and not self._suppress_support_terrain_returns():
            terrain_returns = self.occlusion_engine.generate_terrain_returns(
                own_x,
                own_y,
                bearing,
                self.radar.params.horizontal_beamwidth_deg,
                self.radar.params.max_range_m,
                num_bins,
                furuno_hybrid=use_coastline_terrain_hybrid,
                attenuation_model=self.loaded_coastline_attenuation_model,
                attenuation_config=self._loaded_coastline_attenuation_config(),
            )
            for i, val in enumerate(terrain_returns):
                sweep_data[i] = max(sweep_data[i], val * self.radar.params.gain)

        return sweep_data

    def _try_rust_batched_coastline_sweep(
        self, own_x: float, own_y: float, bearing: float, num_bins: int,
        coastlines: Optional[list] = None,
    ) -> Optional[np.ndarray]:
        """Render compatible loaded coastlines through one Rust call."""
        if coastlines is None:
            coastlines = self.coastlines
        if not coastlines or not rust_bridge.rust_available():
            return None
        if os.environ.get("RADAR_CORE_DISABLE_RUST_COASTLINE", "").lower() in {
            "1", "true", "yes",
        }:
            return None
        if any(getattr(c, "render_mode", "simple") != "polar_intersection"
               for c in coastlines):
            return None

        def profile_key(profile):
            return (
                profile.num_rays,
                profile.edge_factor,
                profile.leading_edge_boost,
                profile.far_range_floor,
                profile.range_decay_strength,
                profile.core_depth_scale,
                profile.interior_decay_scale,
                profile.attenuation_model,
                profile.continuous_land_return,
                profile.continuous_return_threshold,
                profile.continuous_return_depth_m,
                profile.continuous_texture_strength,
                profile.continuous_texture_floor,
                profile.origin_range_attenuation,
                profile.origin_range_ref_m,
                profile.origin_range_power,
                profile.origin_range_floor,
                profile.occlude_after_first_land,
                profile.secondary_return_scale,
                profile.secondary_return_max_code,
                profile.ray_penetration_ref_m,
                profile.ray_penetration_ref_range_m,
                profile.ray_penetration_power,
                profile.ray_penetration_min_m,
                profile.ray_penetration_max_m,
                profile.ray_core_depth_m,
                profile.ray_floor,
            )

        grouped_coastlines = {}
        for coastline in coastlines:
            if len(coastline.points) < 3:
                continue
            grouped_coastlines.setdefault(
                profile_key(coastline.echo_profile), []
            ).append(coastline)
        if not grouped_coastlines:
            return None

        combined = np.zeros(num_bins, dtype=np.float32)
        try:
            for group in grouped_coastlines.values():
                first = group[0].echo_profile
                starts = []
                lengths = []
                reflectivities = []
                point_arrays = []
                cursor = 0
                for coastline in group:
                    if coastline._points_array is None:
                        coastline._update_points_array()
                    if coastline._points_array is None:
                        continue
                    points = np.ascontiguousarray(coastline._points_array, dtype=np.float64)
                    starts.append(cursor)
                    lengths.append(len(points))
                    reflectivities.append(float(coastline.reflectivity))
                    point_arrays.append(points)
                    cursor += len(points)
                if not point_arrays:
                    continue

                flat_points = np.ascontiguousarray(np.vstack(point_arrays), dtype=np.float64)
                group_returns = rust_bridge.polar_intersection_batch_returns(
                    flat_points,
                    starts,
                    lengths,
                    reflectivities,
                    own_x=own_x,
                    own_y=own_y,
                    bearing_deg=bearing,
                    beamwidth_deg=self.radar.params.horizontal_beamwidth_deg,
                    max_range_m=self.radar.params.max_range_m,
                    num_bins=num_bins,
                    num_rays=first.num_rays,
                    edge_factor=first.edge_factor,
                    leading_edge_boost=first.leading_edge_boost,
                    far_range_floor=first.far_range_floor,
                    range_decay_strength=first.range_decay_strength,
                    core_depth_scale=first.core_depth_scale,
                    interior_decay_scale=first.interior_decay_scale,
                    attenuation_model_entry_r4=(
                        first.attenuation_model == ATTENUATION_ENTRY_R4_DEPTH
                    ),
                    continuous_land_return=first.continuous_land_return,
                    continuous_return_threshold=first.continuous_return_threshold,
                    continuous_return_depth_m=first.continuous_return_depth_m,
                    continuous_texture_strength=first.continuous_texture_strength,
                    continuous_texture_floor=first.continuous_texture_floor,
                    origin_range_attenuation=first.origin_range_attenuation,
                    origin_range_ref_m=first.origin_range_ref_m,
                    origin_range_power=first.origin_range_power,
                    origin_range_floor=first.origin_range_floor,
                    occlude_after_first_land=first.occlude_after_first_land,
                    secondary_return_scale=first.secondary_return_scale,
                    secondary_return_max_code=first.secondary_return_max_code,
                    attenuation_config_values=group[0]._rust_attenuation_config_values(),
                )
                combined = np.maximum(combined, group_returns.astype(np.float32))
            return combined
        except Exception:
            return None

    def _postprocess_sweep(
        self,
        sweep_data: list,
        bearing: float,
        *,
        include_weather: bool,
    ) -> list:
        """Apply shared post-processing to a sweep."""
        processed = list(sweep_data)

        if include_weather:
            processed = self.weather.apply_to_sweep(
                processed,
                bearing,
                self.radar.params.max_range_m,
                self.radar.params.sea_clutter,
                self.radar.params.rain_clutter,
            )

        stc_curve = self.radar.params.compute_stc_curve(len(processed))
        return process_signal_chain(
            processed,
            stc_curve,
            self.radar.params.max_range_m,
            self.radar.params.display_log_range_db,
            self.radar.params.output_bits,
        )

    def get_scene_echo_preview_sweeps(self, num_bearings: Optional[int] = None) -> List[tuple]:
        """Return cached coastline/terrain-only sweeps for scene-view overlay.

        The preview mirrors the active shoreline renderer but excludes moving
        targets and weather clutter so the scene view stays readable.
        """
        own_ship = self.world.own_ship
        if own_ship is None or (not self.coastline_enabled and self.occlusion_engine is None):
            return []

        if num_bearings is None:
            num_bearings = max(90, min(self.radar.num_bearings, 240))

        render_modes = tuple(
            getattr(coastline, "render_mode", "simple")
            for coastline in self.coastlines
        )
        cache_key = (
            round(own_ship.x, 1),
            round(own_ship.y, 1),
            round(self.radar.params.current_range_nm, 3),
            self.radar.num_range_bins,
            round(self.radar.params.horizontal_beamwidth_deg, 3),
            round(self.radar.params.gain, 4),
            round(self.radar.params.display_log_range_db, 3),
            self.radar.params.output_bits,
            num_bearings,
            render_modes,
            self.loaded_coastline_attenuation_model,
            self._scene_echo_preview_version,
        )
        if self._scene_echo_preview_cache is not None and self._scene_echo_preview_key == cache_key:
            return self._scene_echo_preview_cache

        sweeps: List[tuple] = []
        step_deg = 360.0 / float(num_bearings)
        for idx in range(num_bearings):
            bearing = (idx * step_deg) % 360.0
            linear_sweep = self._build_environment_linear_sweep(
                own_ship.x, own_ship.y, bearing, self.radar.num_range_bins
            )
            if max(linear_sweep, default=0.0) <= 0.0:
                continue
            sweeps.append(
                (bearing, self._postprocess_sweep(linear_sweep, bearing, include_weather=False))
            )

        if self._should_solidify_loaded_coastline() and sweeps:
            bearings = [bearing for bearing, _ in sweeps]
            rows = self._angular_solidify_rows([row for _, row in sweeps])
            sweeps = list(zip(bearings, rows))

        self._scene_echo_preview_cache = sweeps
        self._scene_echo_preview_key = cache_key
        return sweeps

    # ------------------------------------------------------------------
    # Default scenario
    # ------------------------------------------------------------------
    def setup_default_scenario(self) -> None:
        """Set up a default scenario with own ship and some targets."""
        from ..objects.vessel import Vessel, VesselType

        self.radar.apply_standard_preset(range_nm=6.0)

        own_ship = Vessel(
            id="own_ship", name="Own Ship",
            vessel_type=VesselType.OWN_SHIP,
            x=0, y=0, course=0, speed=10,
            length=100, beam=15, height=20
        )
        self.world.add_vessel(own_ship)

        targets = [
            Vessel(id="target_1", name="Cargo Ship", vessel_type=VesselType.CARGO,
                  x=3000, y=5000, course=225, speed=12, length=150, beam=20, height=25),
            Vessel(id="target_2", name="Tanker", vessel_type=VesselType.TANKER,
                  x=-4000, y=3000, course=90, speed=8, length=200, beam=30, height=20),
            Vessel(id="target_3", name="Fishing Boat", vessel_type=VesselType.FISHING,
                  x=2000, y=-2000, course=315, speed=6, length=25, beam=6, height=8),
            Vessel(id="target_4", name="Sailing Yacht", vessel_type=VesselType.SAILING,
                  x=-1500, y=-4000, course=45, speed=5, length=15, beam=4, height=15),
        ]
        for target in targets:
            self.world.add_vessel(target)

        self.weather.set_conditions(WeatherConditions(
            sea_state=3, wind_speed_knots=15, wind_direction=45,
            rain_rate_mmh=0, visibility_nm=10
        ))

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    def collect_annotations(self) -> List[dict]:
        """Collect annotation data for all current targets."""
        own_ship = self.world.own_ship
        if own_ship is None:
            return []

        import math
        annotations = []
        for vessel in self.world.get_targets():
            dx = vessel.x - own_ship.x
            dy = vessel.y - own_ship.y
            range_m = math.sqrt(dx * dx + dy * dy)
            bearing_deg = math.degrees(math.atan2(dx, dy)) % 360

            occluded = False
            if self.occlusion_engine is not None:
                occluded = self.occlusion_engine.is_target_occluded(
                    own_ship.x, own_ship.y, vessel.x, vessel.y,
                    target_height_m=vessel.height
                )

            annotations.append({
                'vessel_id': vessel.id,
                'vessel_type': vessel.vessel_type.value,
                'range_m': range_m,
                'bearing_deg': bearing_deg,
                'rcs': vessel.rcs,
                'length': vessel.length,
                'beam': vessel.beam,
                'height': vessel.height,
                'occluded': occluded,
                'x': vessel.x,
                'y': vessel.y,
                'course': vessel.course,
                'speed': vessel.speed,
            })
        self._current_annotations = annotations
        return annotations

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------
    def update(self, dt: float = None) -> None:
        """Update simulation by one time step."""
        if self.is_paused:
            return

        dt = dt or self.dt
        scaled_dt = dt * self.time_scale

        self.world.update(scaled_dt)
        self.radar.update(self.world, scaled_dt)

        self.frame_count += 1
        if self.on_update:
            self.on_update(scaled_dt)

    def get_radar_sweep_data(self, bearing: float) -> list:
        """Get radar sweep data for a bearing, including terrain returns.

        Applies full signal chain: weather → STC → log compression → quantization.
        """
        own_ship = self.world.own_ship
        if own_ship is None:
            return [0.0] * self.radar.num_range_bins

        # Generate fresh target sweep data for this requested bearing. Live UI
        # calls usually match the antenna state already, but full-rotation
        # export asks for many bearings without advancing the antenna.
        targets = self.world.get_targets()
        own_x = own_ship.x
        own_y = own_ship.y

        previous_bearing = self.radar.antenna.get_bearing()
        self.radar.antenna.set_bearing(bearing)
        try:
            target_sweep = self.radar.detection_engine.generate_sweep_data(
                targets, own_x, own_y, self.world.time,
                self.radar.num_range_bins
            )
        finally:
            self.radar.antenna.set_bearing(previous_bearing)
        environment_sweep = self._build_environment_linear_sweep(
            own_x, own_y, bearing, self.radar.num_range_bins
        )
        sweep_data = [
            max(target_val, env_val)
            for target_val, env_val in zip(target_sweep, environment_sweep)
        ]

        return self._postprocess_sweep(sweep_data, bearing, include_weather=True)

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------
    def set_time_scale(self, scale: float) -> None:
        self.time_scale = max(0.1, min(10.0, scale))

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False

    def toggle_pause(self) -> bool:
        self.is_paused = not self.is_paused
        return self.is_paused

    def reset(self) -> None:
        self.world.clear()
        self.radar.clear_sweep_buffer()
        self.clear_coastlines()
        self.clear_terrain()
        self.frame_count = 0
        self.is_paused = False
        self._current_annotations = []
        self._invalidate_scene_echo_preview()

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------
    def start_recording(self) -> str:
        return self.exporter.start_recording()

    def stop_recording(self) -> str:
        return self.exporter.stop_recording()

    def is_recording(self) -> bool:
        return self.exporter.is_recording

    def get_record_count(self) -> int:
        return self.exporter.get_record_count()

    def export_current_sweep(self, bearing: float) -> str:
        sweep_data = self.get_radar_sweep_data(bearing)
        return self.exporter.export_single_sweep(
            timestamp=self.world.time,
            bearing_deg=bearing,
            range_scale_nm=self.radar.params.current_range_nm,
            gain=self.radar.params.gain,
            sea_clutter=self.radar.params.sea_clutter,
            rain_clutter=self.radar.params.rain_clutter,
            echo_values=sweep_data,
        )

    def export_furuno_rotation(
        self,
        filename: Optional[str] = None,
        *,
        spokes_per_rotation: int = 1820,
        raytrace_spokes: Optional[int] = None,
    ) -> str:
        """Export one full rotation as an exact Furuno-style CSV.

        The scene is ray-traced at `raytrace_spokes` bearings, then resampled
        onto the requested Furuno angle-tick sequence. This keeps the export
        aligned with the active shoreline renderer while avoiding prohibitively
        slow full-brute-force tracing at every Furuno spoke.
        """
        if self.world.own_ship is None:
            return ""

        print("STAGE|Sweep render|started", flush=True)
        render_started = time.perf_counter()
        rows = []
        export_ticks = self.exporter.generate_furuno_angle_ticks(spokes_per_rotation)
        base_spokes = max(1, int(raytrace_spokes or self.radar.num_bearings))
        base_ticks = self.exporter.generate_furuno_angle_ticks(base_spokes)
        base_sweeps = []
        for tick in base_ticks:
            bearing_deg = tick * 360.0 / 8192.0
            base_sweeps.append(self.get_radar_sweep_data(bearing_deg))
        base_sweeps = self._angular_solidify_rows(base_sweeps)
        base_sweeps = self._suppress_isolated_outer_returns(base_sweeps)

        gain_code = max(0, min(100, int(round(self.radar.params.gain * 100.0))))
        range_index = self.exporter.furuno_range_index_for_nm(self.radar.params.current_range_nm)

        for tick in export_ticks:
            bearing_deg = tick * 360.0 / 8192.0
            base_position = (bearing_deg / 360.0) * base_spokes
            left_idx = int(math.floor(base_position)) % base_spokes
            frac = base_position - math.floor(base_position)
            right_idx = (left_idx + 1) % base_spokes

            if frac <= 1e-6:
                sweep_data = base_sweeps[left_idx]
            else:
                left_sweep = base_sweeps[left_idx]
                right_sweep = base_sweeps[right_idx]
                sweep_data = [
                    left_val * (1.0 - frac) + right_val * frac
                    for left_val, right_val in zip(left_sweep, right_sweep)
                ]
            furuno_echoes = self.exporter.quantize_sweep_to_furuno_codes(sweep_data)
            rows.append({
                "status": 1,
                "scale": 496,
                "range": range_index,
                "gain": gain_code,
                "angle": tick,
                "echoes": furuno_echoes,
            })

        print(f"STAGE|Sweep render|complete|{time.perf_counter() - render_started:.3f}", flush=True)
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"furuno_rotation_{timestamp}.csv"

        print("STAGE|CSV write|started", flush=True)
        write_started = time.perf_counter()
        result = self.exporter.save_furuno_rotation(rows, filename)
        print(f"STAGE|CSV write|complete|{time.perf_counter() - write_started:.3f}", flush=True)
        return result
