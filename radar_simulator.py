"""
Geometric Radar Simulator — Furuno DRS4DNXT at 0.125 NM
Generates synthetic radar CSV files matching real Furuno capture format.

Output format: Status,Scale,Range,Gain,Angle,EchoValues (868 echo columns)
  Status=1, Scale=496, Range=2 (0.125 NM index), Gain=75
  Angle = 0-8191 (8192 ticks per 360 degrees)
  EchoValues = 868 integers 0-252

DRS4DNXT sensor characteristics at 0.125 NM:
  Frequency: 9.41 GHz (X-band)
  Beamwidth: 3.9 degrees horizontal
  Range: 231.5 m (0.125 NM)
  Range bins: 868
  Bin resolution: 0.2668 m
  Dead zone: bins 0-30 (~8.3 m, transmit/receive switchover)
  Rotation: ~24 RPM
  Spokes per rotation: ~1820
"""

import math
import random
import sys
import numpy as np
import os
import csv
import json
import hashlib
import pickle
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional
import time as _time_module
import re as _re_module
from object_renderer import (
    ObjectRenderPriors,
    render_object_echo as _render_object_echo_core,
)

try:
    import pandas as _pd
    _HAS_PANDAS = True
except ImportError:
    _pd = None
    _HAS_PANDAS = False

try:
    import requests as _requests_lib
    _HAS_REQUESTS = True
except ImportError:
    _requests_lib = None
    _HAS_REQUESTS = False

try:
    from scipy.spatial import cKDTree as _cKDTree
    from scipy.stats import ks_2samp as _ks_2samp
    _HAS_SCIPY_VALIDATION = True
except ImportError:
    _cKDTree = None
    _ks_2samp = None
    _HAS_SCIPY_VALIDATION = False


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEC_CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache", "mode_c")
MODEC_SHORELINE_CACHE_DIR = os.path.join(MODEC_CACHE_DIR, "shorelines")
SCENE_REFERENCE_DIR = os.path.join(SCRIPT_DIR, "references", "scenes")
SCENE_TEMPLATE_DIR = os.path.join(SCENE_REFERENCE_DIR, "templates")
SCENE_CAPTURE_DIR = os.path.join(SCENE_REFERENCE_DIR, "captures")
JSON_CONFIG_DIR = os.path.join(SCRIPT_DIR, "references", "json_configs")
LIBRARY_CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache", "library")
MODEC_SOURCE_DIRS = [
    ("NOAA ENC", os.path.join(SCRIPT_DIR, "shoreline_sources", "noaa_enc")),
    ("NOAA Shoreline", os.path.join(SCRIPT_DIR, "shoreline_sources", "noaa_shoreline")),
    ("USGS Hydrography", os.path.join(SCRIPT_DIR, "shoreline_sources", "usgs")),
    ("Local GeoJSON", os.path.join(SCRIPT_DIR, "shoreline_sources", "geojson")),
]
MODEC_GEOJSON_SEARCH_ROOTS = [SCRIPT_DIR] + [path for _, path in MODEC_SOURCE_DIRS]
MODEC_SKIP_DIRS = {".git", "__pycache__", ".cache", "outputs"}
_GNIS_FIND_URL = "https://carto.nationalmap.gov/arcgis/rest/services/geonames/MapServer/find"
_MODEC_ARCGIS_TIMEOUT = 8
_MODEC_GNIS_TIMEOUT = 6
_MODEC_NOMINATIM_TIMEOUT = 6
_MODEC_OVERPASS_TIMEOUT = 20
_MODEC_OVERPASS_RETRY_DELAY_S = 5
_MODEC_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
_MODEC_OVERPASS_HEADERS = {
    "User-Agent": "TORS/1.0 (Furuno coastline generation)",
    "Accept": "application/json",
    "Content-Type": "text/plain; charset=utf-8",
}
_NOAA_ENC_LINE_LAYERS = [
    ("Harbor", "https://encdirect.noaa.gov/arcgis/rest/services/encdirect/enc_harbour/MapServer/84"),
    ("Approach", "https://encdirect.noaa.gov/arcgis/rest/services/encdirect/enc_approach/MapServer/88"),
    ("Coastal", "https://encdirect.noaa.gov/arcgis/rest/services/encdirect/enc_coastal/MapServer/70"),
    ("General", "https://encdirect.noaa.gov/arcgis/rest/services/encdirect/enc_general/MapServer/58"),
]
_USGS_3DHP_WATERBODY_URL = "https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer/60"
_USGS_3DHP_FLOWLINE_URL = "https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer/50"

SIM_MODE_CURRENT_MODE4 = "current_mode4"
SIM_MODE_M1 = "m1"
SIM_MODE_M4G = "m4g"
SIM_MODE_M4P = "m4p"
SIM_MODE_M42 = "m42"
SIM_MODE_CHOICES = [
    SIM_MODE_CURRENT_MODE4,
    SIM_MODE_M1,
    SIM_MODE_M4G,
    SIM_MODE_M4P,
    SIM_MODE_M42,
]
SIM_MODE_LABELS = {
    SIM_MODE_CURRENT_MODE4: "Current Mode 4 baseline",
    SIM_MODE_M1: "Synthetic baseline (Mode 1)",
    SIM_MODE_M4G: "Preview M4G",
    SIM_MODE_M4P: "Preview M4P",
    SIM_MODE_M42: "Replay ceiling (Mode 42)",
}
MODE4_CALIBRATION_PATH = os.path.join(SCRIPT_DIR, "mode4_calibration.json")
MODE4_OBJECT_LEARNED_MODEL_PATH = os.environ.get(
    "MODE4_OBJECT_LEARNED_MODEL_PATH",
    os.path.join(SCRIPT_DIR, "mode4_object_learned_model.pkl"),
)
MODE4_USE_OBJECT_LEARNED_MODEL = os.environ.get("MODE4_USE_OBJECT_LEARNED_MODEL", "1").strip().lower() not in {
    "0", "false", "no", "off"
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


MODE4_OBJECT_PEAK_CORRECTION_LIMIT_DB = _env_float("MODE4_OBJECT_PEAK_CORRECTION_LIMIT_DB", 8.0)
MODE4_OBJECT_SHAPE_CORRECTION_LIMIT_DB = _env_float("MODE4_OBJECT_SHAPE_CORRECTION_LIMIT_DB", 6.0)
_MODE4_OBJECT_LEARNED_MODEL_CACHE: dict[str, object] = {
    "path": None,
    "mtime": None,
    "artifact": None,
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ============================================================
# DRS4DNXT Constants at 0.125 NM
# ============================================================
STATUS = 1
SCALE = 496
RANGE_INDEX = 2         # Furuno range index for 0.125 NM
GAIN = 75               # Default gain (0-100)
RANGE_NM = 0.125
RANGE_M = RANGE_NM * 1852.0   # 231.5 m
NUM_BINS = 868
BIN_SIZE_M = RANGE_M / NUM_BINS  # ~0.267 m
DEAD_ZONE_BINS = 31     # Bins 0-30 always zero (Tx/Rx switchover)
TICKS_PER_REV = 8192    # Angle ticks per full rotation
BEAMWIDTH_DEG = 3.9     # DRS4DNXT horizontal beamwidth
SPOKES_PER_REV = 1820   # Typical spoke count per rotation
ROTATION_DT = 2.5       # Seconds per rotation (~24 RPM)

# Furuno range-code lookup used when custom simulations override the range.
# Preserve the legacy RANGE_INDEX value for the default range, but emit the
# nearest standard code for explicitly requested ranges such as 1.5 nm.
FURUNO_RANGE_NM_OPTIONS = [0.125, 0.25, 0.5, 0.75, 1.5, 3.0, 6.0, 12.0,
                           24.0, 48.0, 72.0, 96.0]


def furuno_range_index_for_nm(range_nm: float) -> int:
    if abs(float(range_nm) - RANGE_NM) < 1e-9:
        return RANGE_INDEX
    return min(
        range(len(FURUNO_RANGE_NM_OPTIONS)),
        key=lambda i: abs(FURUNO_RANGE_NM_OPTIONS[i] - float(range_nm)),
    )

# Furuno DRS4DNXT quantized gain code levels (17 discrete values from log amplifier)
FURUNO_GAIN_CODES = np.array([0, 8, 20, 28, 36, 44, 56, 64, 76, 88,
                              104, 120, 140, 164, 188, 220, 252], dtype=int)
FURUNO_GAIN_SET = set(FURUNO_GAIN_CODES.tolist())

# Mode 4 coastline intensity tuning.
# These values are intentionally centralized so reference-calibrated
# brightness can be tuned against known real captures in a controlled way.
MODE4_COASTLINE_CORE_DEPTH_MIN_BINS = _env_int("MODE4_COASTLINE_CORE_DEPTH_MIN_BINS", 4)
MODE4_COASTLINE_CORE_DEPTH_MAX_BINS = _env_int("MODE4_COASTLINE_CORE_DEPTH_MAX_BINS", 10)
MODE4_COASTLINE_DECAY_DB_DIVISOR = _env_float("MODE4_COASTLINE_DECAY_DB_DIVISOR", 50.0)
MODE4_COASTLINE_TERRAIN_ABSORB_PER_BIN = _env_float("MODE4_COASTLINE_TERRAIN_ABSORB_PER_BIN", 0.08)
MODE4_NOISE_FIELD_SECTORS = 24
MODE4_NOISE_FIELD_BANDS = 8
MODE4_NOISE_NEAR_LAND_BINS = 12
MODE4_NOISE_OPEN_WATER_MAX_CODE = 88
MODE4_NOISE_MAX_EXISTING_ECHO = 20
OBJECT_TYPE_DB_CORRECTIONS = {
    "buoy": 5.5,
    "small_boat": 2.0,
    "medium_boat": 1.5,
    "large_boat": 3.0,
    "barge": 3.5,
}


# ============================================================
# Object Definitions
# ============================================================
@dataclass
class RadarObject:
    """A detectable object in the scene with motion state and optional waypoints.

    Waypoint navigation:
        Set `waypoints` to a list of (x, y) tuples in meters. The object will
        steer toward each waypoint in order, advancing to the next when within
        `waypoint_radius_m`. After all waypoints are reached the object holds
        position (or loops if `waypoints_loop` is True).

        This enables scripted scenarios like "enter harbor from SE bearing,
        follow channel, turn to port, approach dock."
    """
    x: float              # meters from radar (east positive)
    y: float              # meters from radar (north positive)
    obj_type: str         # "buoy", "small_boat", "medium_boat", "large_boat", "barge"
    rcs_m2: float         # Radar cross section in m²
    length_m: float       # Physical length along major axis
    width_m: float        # Physical width
    heading_deg: float    # Orientation 0=north, clockwise
    speed_kts: float = 0.0    # Speed in knots (0 = stationary)
    course_deg: float = 0.0   # Course over ground, 0=north, clockwise
    turn_rate_dps: float = 0.0  # Turn rate in degrees per second (gentle course changes)
    label: str = ""

    # Waypoint navigation
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    waypoint_idx: int = field(default=0, repr=False)
    waypoint_radius_m: float = 5.0   # Arrival threshold
    waypoints_loop: bool = False      # Loop back to first waypoint when done

    # Track history for metadata
    _initial_x: float = field(default=0.0, init=False, repr=False)
    _initial_y: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        self._initial_x = self.x
        self._initial_y = self.y

    @property
    def speed_mps(self) -> float:
        """Speed in meters per second."""
        return self.speed_kts * 0.514444

    @property
    def is_moving(self) -> bool:
        return self.speed_kts > 0.1

    @property
    def motion_label(self) -> str:
        if not self.is_moving:
            return "stationary"
        elif self.speed_kts < 3:
            return "drifting"
        elif self.speed_kts < 8:
            return "slow"
        elif self.speed_kts < 15:
            return "moderate"
        else:
            return "fast"

    @property
    def has_active_waypoint(self) -> bool:
        """True if there are waypoints remaining to follow."""
        return bool(self.waypoints) and self.waypoint_idx < len(self.waypoints)

    def _steer_toward_waypoint(self):
        """Update course_deg to steer toward the current waypoint."""
        if not self.has_active_waypoint:
            return
        wx, wy = self.waypoints[self.waypoint_idx]
        dx = wx - self.x
        dy = wy - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < self.waypoint_radius_m:
            # Arrived — advance to next waypoint
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.waypoints) and self.waypoints_loop:
                self.waypoint_idx = 0
            if self.has_active_waypoint:
                self._steer_toward_waypoint()
            return
        target_deg = math.degrees(math.atan2(dx, dy)) % 360
        # Smooth turn toward target (max 5 deg/s equivalent)
        diff = (target_deg - self.course_deg + 180) % 360 - 180
        max_turn = 15.0  # degrees per update step
        if abs(diff) > max_turn:
            diff = max_turn if diff > 0 else -max_turn
        self.course_deg = (self.course_deg + diff) % 360

    def update_position(self, dt_seconds: float,
                        coastlines: List['CoastlineSegment'] = None,
                        max_range_m: float = RANGE_M) -> None:
        """Advance position by dt seconds along current course.
        If waypoints are set, steers toward them in order.
        Respects coastline boundaries and max range."""
        # Waypoint steering (overrides turn_rate when active)
        if self.has_active_waypoint:
            self._steer_toward_waypoint()
            # Stop when all waypoints reached (non-looping)
            if not self.has_active_waypoint and not self.waypoints_loop:
                self.speed_kts = 0.0
                return
        elif not self.is_moving:
            return
        else:
            # Free-running: apply turn rate
            self.course_deg = (self.course_deg + self.turn_rate_dps * dt_seconds) % 360

        if not self.is_moving:
            return

        # Align heading with course for moving vessels
        heading_diff = (self.course_deg - self.heading_deg + 180) % 360 - 180
        self.heading_deg = (self.heading_deg + heading_diff * 0.3) % 360

        # Compute displacement
        course_rad = math.radians(self.course_deg)
        dx = self.speed_mps * math.sin(course_rad) * dt_seconds
        dy = self.speed_mps * math.cos(course_rad) * dt_seconds

        new_x = self.x + dx
        new_y = self.y + dy

        # Check bounds: stay within radar range
        new_range = math.sqrt(new_x**2 + new_y**2)
        if new_range > max_range_m * 0.95:
            if self.has_active_waypoint:
                # Waypoint is out of range — skip it
                self.waypoint_idx += 1
            else:
                self.course_deg = (self.course_deg + 180) % 360
            return

        # Check coastline collision
        if coastlines and point_in_any_coastline(new_x, new_y, coastlines):
            if self.has_active_waypoint:
                self.waypoint_idx += 1
            else:
                self.course_deg = (self.course_deg + 90) % 360
            return

        self.x = new_x
        self.y = new_y

    def distance_traveled(self) -> float:
        """Distance from initial position in meters."""
        return math.sqrt((self.x - self._initial_x)**2 +
                         (self.y - self._initial_y)**2)


@dataclass
class CoastlineSegment:
    """A coastline polygon defined by vertices in (x, y) meters."""
    points: List[Tuple[float, float]]
    name: str = "coastline"
    closed: bool = True


@dataclass
class CoastlinePrimitive:
    """A reusable coastline shape extracted from real radar captures."""
    primitive_id: str              # "pattern11_coastline_0"
    classification: str            # "arc", "headland", "bay", "thin_linear", "island", "large_mass"
    source_pattern: str            # "pattern11"
    normalized_points: list        # shape in [0,1] x [0,1] bearing/range space
    original_bearing_span_deg: float
    original_range_span_m: float
    original_min_range_m: float
    original_center_bearing_deg: float
    original_num_points: int
    aspect_ratio: float
    convexity: float
    fourier_descriptors: list      # first 8 FDs for shape matching
    echo_stats: dict = field(default_factory=dict)


class CoastlineLibrary:
    """Collection of coastline primitives indexed by classification."""

    def __init__(self):
        self.primitives = []
        self.by_classification = {}

    def add(self, primitive):
        self.primitives.append(primitive)
        cls = primitive.classification
        if cls not in self.by_classification:
            self.by_classification[cls] = []
        self.by_classification[cls].append(primitive)


@dataclass
class SceneConfig:
    """Configuration for a randomly generated scene."""
    num_buoys: Tuple[int, int] = (0, 5)
    num_small_boats: Tuple[int, int] = (0, 4)
    num_medium_boats: Tuple[int, int] = (0, 3)
    num_large_boats: Tuple[int, int] = (0, 2)
    num_barges: Tuple[int, int] = (0, 1)
    coastline_probability: float = 0.95
    noise_floor: int = 0
    gain: int = 75


# ============================================================
# Object Type Parameters (based on DRS4DNXT X-band returns)
# ============================================================
OBJECT_PARAMS = {
    "buoy": {
        "rcs_range": (5, 50),
        "length_range": (1, 3),
        "width_range": (1, 3),
        "echo_peak": (140, 220),
        "echo_spread_bins": (1, 3),
        "speed_range": (0, 0),
        "turn_rate_range": (0, 0),
    },
    "small_boat": {
        "rcs_range": (5, 200),
        "length_range": (4, 10),
        "width_range": (2, 4),
        "echo_peak": (120, 220),
        "echo_spread_bins": (3, 12),
        "speed_range": (0, 12),
        "turn_rate_range": (-3, 3),
    },
    "medium_boat": {
        "rcs_range": (50, 2000),
        "length_range": (10, 25),
        "width_range": (4, 8),
        "echo_peak": (160, 252),
        "echo_spread_bins": (8, 30),
        "speed_range": (0, 18),
        "turn_rate_range": (-2, 2),
    },
    "large_boat": {
        "rcs_range": (300, 5000),
        "length_range": (25, 80),
        "width_range": (8, 20),
        "echo_peak": (200, 252),
        "echo_spread_bins": (15, 60),
        "speed_range": (2, 15),
        "turn_rate_range": (-1, 1),
    },
    "barge": {
        "rcs_range": (500, 10000),
        "length_range": (30, 80),
        "width_range": (10, 20),
        "echo_peak": (220, 252),
        "echo_spread_bins": (20, 80),
        "speed_range": (0, 6),
        "turn_rate_range": (-0.5, 0.5),
    },
}


# ============================================================
# Coastline Noise & Utility Functions
# ============================================================
def fractal_noise_1d(n, beta=1.5, amplitude=1.0, seed=None):
    """Generate 1D fractal noise via spectral synthesis (1/f^beta).

    Produces continuous multi-scale noise suitable for coastline edges:
      beta=1.2-1.3: rough (rocky islands)
      beta=1.5:     natural coastline roughness
      beta=1.8-2.0: smooth (engineered structures, outer edges)
      beta=2.5:     very smooth (depth profiles)
    Output is inherently periodic (FFT-based), so closed polygons wrap smoothly.
    """
    rng = np.random.RandomState(seed) if seed else np.random
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid division by zero
    power = 1.0 / np.power(freqs, beta / 2.0)
    power[0] = 0.0  # zero mean
    phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
    spectrum = power * np.exp(1j * phases)
    noise = np.fft.irfft(spectrum, n=n)
    rms = np.sqrt(np.mean(noise**2))
    return noise * (amplitude / rms) if rms > 0 else noise


def variable_depth_profile(n, base_depth, min_depth=None, max_depth=None):
    """Generate smooth depth variation along a coastline.

    Uses fractal noise with beta=2.5 for slow undulations, clamped to range.
    """
    if min_depth is None:
        min_depth = base_depth * 0.5
    if max_depth is None:
        max_depth = base_depth * 1.5
    noise = fractal_noise_1d(n, beta=2.5, amplitude=(max_depth - min_depth) * 0.3)
    depth = base_depth + noise
    return np.clip(depth, min_depth, max_depth)


def adaptive_point_count(arc_span_deg, range_m):
    """Compute vertex count: ~1 point per 0.7m of arc length, clamped [80, 2000]."""
    arc_length_m = math.radians(arc_span_deg) * range_m
    return max(80, min(2000, int(arc_length_m / 0.7)))


# ============================================================
# Marine Coordinate Transforms
# ============================================================
EARTH_RADIUS_M = 6_371_000.0  # WGS-84 mean radius in meters


def latlon_to_local_xy(lat: float, lon: float,
                       ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """Convert (lat, lon) to local (x, y) meters relative to a reference point.

    Uses equirectangular projection:
      x = (lon - ref_lon) * cos(ref_lat) * R * pi/180   (east positive)
      y = (lat - ref_lat) * R * pi/180                   (north positive)

    Sub-meter accuracy within 50 km of the reference point, far exceeding
    the DRS4DNXT's 231.5 m maximum range.

    Args:
        lat: Point latitude in decimal degrees
        lon: Point longitude in decimal degrees
        ref_lat: Radar position latitude (projection center)
        ref_lon: Radar position longitude (projection center)

    Returns:
        (x, y) in meters — east and north offsets from radar
    """
    d2r = math.pi / 180.0
    x = (lon - ref_lon) * math.cos(ref_lat * d2r) * EARTH_RADIUS_M * d2r
    y = (lat - ref_lat) * EARTH_RADIUS_M * d2r
    return x, y


def load_coastlines_from_geojson(filepath: str, ref_lat: float, ref_lon: float,
                                  max_range_m: float = RANGE_M
                                  ) -> List['CoastlineSegment']:
    """Load coastline polygons from a GeoJSON file.

    Supports:
      - Polygon: outer ring converted to CoastlineSegment
      - MultiPolygon: each outer ring
      - LineString: buffered into a thin polygon (3 m width, same pattern as
        breakwater/jetty generators)
      - MultiLineString: each line buffered independently

    Coordinates are converted from (lon, lat) to local (x, y) via
    latlon_to_local_xy(). Points beyond max_range_m are clipped.
    Features with fewer than 3 remaining points are discarded.

    Args:
        filepath: Path to GeoJSON file
        ref_lat: Radar latitude (projection center)
        ref_lon: Radar longitude (projection center)
        max_range_m: Maximum range for clipping (default: RANGE_M)

    Returns:
        List of CoastlineSegment objects
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    features = data.get('features', [])
    if not features and data.get('type') in ('Polygon', 'MultiPolygon',
                                              'LineString', 'MultiLineString'):
        features = [{'type': 'Feature', 'geometry': data, 'properties': {}}]

    coastlines = []

    def _convert_ring(coords, name):
        """Convert a coordinate ring to a CoastlineSegment, clipping to range."""
        pts = []
        for coord in coords:
            lon, lat = coord[0], coord[1]
            x, y = latlon_to_local_xy(lat, lon, ref_lat, ref_lon)
            pts.append((x, y))
        clipped = _clip_closed_polygon_with_boundary_arc(pts, max_range_m)
        if len(clipped) >= 3:
            pts = clipped
            pts = _chaikin_smooth(pts, iterations=2)
            pts = _smooth_polygon_points(pts)
            coastlines.append(CoastlineSegment(points=pts, name=name, closed=True))

    def _buffer_line(coords, name, width=3.0):
        """Buffer a LineString into a thin polygon, like breakwater/jetty."""
        pts = []
        for coord in coords:
            lon, lat = coord[0], coord[1]
            x, y = latlon_to_local_xy(lat, lon, ref_lat, ref_lon)
            if math.sqrt(x * x + y * y) <= max_range_m * 1.1:
                pts.append((x, y))
        if len(pts) < 2:
            return
        left_pts = []
        right_pts = []
        half_w = width / 2.0
        for i in range(len(pts)):
            if i < len(pts) - 1:
                dx = pts[i + 1][0] - pts[i][0]
                dy = pts[i + 1][1] - pts[i][1]
            else:
                dx = pts[i][0] - pts[i - 1][0]
                dy = pts[i][1] - pts[i - 1][1]
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-9:
                continue
            nx, ny = -dy / length, dx / length
            left_pts.append((pts[i][0] + nx * half_w, pts[i][1] + ny * half_w))
            right_pts.append((pts[i][0] - nx * half_w, pts[i][1] - ny * half_w))
        if len(left_pts) >= 2:
            polygon = left_pts + list(reversed(right_pts))
            polygon = _chaikin_smooth(polygon, iterations=2)
            polygon = _smooth_polygon_points(polygon)
            coastlines.append(CoastlineSegment(points=polygon, name=name, closed=True))

    for feature in features:
        geom = feature.get('geometry', {})
        props = feature.get('properties', {})
        name = props.get('name', 'geojson_coastline') if props else 'geojson_coastline'
        geom_type = geom.get('type', '')
        coords = geom.get('coordinates', [])

        if geom_type == 'Polygon':
            _convert_ring(coords[0], name)
        elif geom_type == 'MultiPolygon':
            for poly_coords in coords:
                _convert_ring(poly_coords[0], name)
        elif geom_type == 'LineString':
            _buffer_line(coords, name)
        elif geom_type == 'MultiLineString':
            for line_coords in coords:
                _buffer_line(line_coords, name)

    return coastlines


def fetch_noaa_coastline_stub(radar_lat: float, radar_lon: float,
                               buffer_radius_m: float = 500.0
                               ) -> Optional[str]:
    """Placeholder for NOAA shoreline data download.

    When implemented, this function will:
    1. Query the NOAA Shoreline dataset via ArcGIS REST API:
       https://coast.noaa.gov/arcgis/rest/services/
         FloodExposureMapper/CFEM_CoastalFloodHazardComposite/MapServer/
         3/query?geometry=<lon-buffer>,<lat-buffer>,<lon+buffer>,<lat+buffer>
         &geometryType=esriGeometryEnvelope&spatialRel=esriSpatialRelIntersects
         &outFields=*&f=geojson
    2. Download the GeoJSON response to a temporary file
    3. Return the temp file path for load_coastlines_from_geojson()

    Args:
        radar_lat: Radar position latitude
        radar_lon: Radar position longitude
        buffer_radius_m: Search radius around radar position (meters)

    Returns:
        None (stub — real implementation returns path to temp GeoJSON file)
    """
    return None


def _ensure_modec_cache_dirs():
    """Create repo-local cache directories for Mode C runtime artifacts."""
    os.makedirs(MODEC_SHORELINE_CACHE_DIR, exist_ok=True)
    os.makedirs(LIBRARY_CACHE_DIR, exist_ok=True)


def _scene_template_search_dirs(scene_dir=None):
    """Return scene-template directories in priority order."""
    base_dir = scene_dir or SCRIPT_DIR
    dirs = [
        base_dir,
        SCENE_TEMPLATE_DIR,
        SCENE_CAPTURE_DIR,
    ]
    seen = []
    for path in dirs:
        norm = os.path.normcase(os.path.abspath(path))
        if norm not in seen and os.path.isdir(path):
            seen.append(norm)
    return [path for path in dirs if os.path.isdir(path)]


def _find_scene_template_files(scene_dir=None):
    """Discover template scene JSON files across organized reference folders."""
    files = []
    for base_dir in _scene_template_search_dirs(scene_dir):
        try:
            for name in os.listdir(base_dir):
                if name.startswith("pattern") and name.endswith("_scene.json"):
                    files.append(os.path.join(base_dir, name))
        except OSError:
            continue
    return sorted(set(files))


def _normalize_location_key(text: str) -> str:
    """Normalize location text for alias lookup and filename matching."""
    if not text:
        return ""
    text = text.lower()
    text = text.replace("harbour", "harbor")
    text = text.replace("&", " and ")
    text = _re_module.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(part for part in text.split() if part)


def _extract_state_token(location_name: str) -> Optional[str]:
    """Extract a likely state/territory token from a user query."""
    if not location_name:
        return None
    tokens = [tok.strip(" ,") for tok in location_name.split() if tok.strip(" ,")]
    if not tokens:
        return None
    tail = tokens[-1]
    if len(tail) == 2 and tail.isalpha():
        return tail.upper()
    if len(tokens) >= 2 and len(tokens[-2]) == 2 and tokens[-2].isalpha() and tail.upper() in {"USA", "US"}:
        return tokens[-2].upper()
    return None


def _webmercator_to_latlon(x: float, y: float) -> Tuple[float, float]:
    """Convert EPSG:3857 meters to geographic coordinates."""
    lon = (x / 6378137.0) * (180.0 / math.pi)
    lat = (2.0 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2.0) * (180.0 / math.pi)
    return lat, lon


def _meters_to_degree_envelope(center_lat: float, center_lon: float, radius_m: float):
    """Build a WGS84 envelope around a center point."""
    lat_delta = radius_m / 111_320.0
    lon_scale = max(0.2, math.cos(math.radians(center_lat)))
    lon_delta = radius_m / (111_320.0 * lon_scale)
    return (
        center_lon - lon_delta,
        center_lat - lat_delta,
        center_lon + lon_delta,
        center_lat + lat_delta,
    )


def _is_us_bbox(center_lat: float, center_lon: float) -> bool:
    """Coarse filter for U.S. and nearby coastal waters."""
    return 17.0 <= center_lat <= 72.0 and -170.0 <= center_lon <= -60.0


def _remote_modec_cache_path(prefix: str, loc_name: str, lat: float, lon: float, radius_m: float) -> str:
    """Stable repo-local cache path for remote shoreline extracts."""
    _ensure_modec_cache_dirs()
    slug = _normalize_location_key(loc_name).replace(" ", "_")[:60] or "location"
    digest = hashlib.sha1(f"{prefix}|{lat:.6f}|{lon:.6f}|{radius_m:.1f}".encode("utf-8")).hexdigest()[:10]
    return os.path.join(MODEC_SHORELINE_CACHE_DIR, f"{prefix}_{slug}_{digest}.geojson")


def _merge_feature_collections(collections):
    """Merge multiple GeoJSON feature collections into one."""
    features = []
    for collection in collections:
        if not collection:
            continue
        for feat in collection.get("features", []):
            if feat and feat.get("geometry"):
                features.append(feat)
    return {"type": "FeatureCollection", "features": features}


def _query_arcgis_geojson(base_url: str, center_lat: float, center_lon: float,
                          radius_m: float, where: str = "1=1",
                          timeout: int = _MODEC_ARCGIS_TIMEOUT):
    """Query an ArcGIS Feature/MapServer layer and return GeoJSON."""
    if not _HAS_REQUESTS:
        return None
    xmin, ymin, xmax, ymax = _meters_to_degree_envelope(center_lat, center_lon, radius_m)
    params = {
        "where": where,
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": 4326,
        "f": "geojson",
    }
    resp = _requests_lib.get(base_url.rstrip("/") + "/query", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _fetch_remote_noaa_enc_geojson(loc, search_radius_m, verbose=False):
    """Fetch chart-derived shoreline lines from NOAA ENC services and cache them."""
    if not _HAS_REQUESTS or not _is_us_bbox(loc["lat"], loc["lon"]):
        return None, None

    cache_path = _remote_modec_cache_path("noaa_enc", loc["name"], loc["lat"], loc["lon"], search_radius_m)
    if os.path.isfile(cache_path):
        return cache_path, "NOAA ENC Remote Cache"

    collections = []
    for label, base_url in _NOAA_ENC_LINE_LAYERS:
        try:
            data = _query_arcgis_geojson(base_url, loc["lat"], loc["lon"], search_radius_m)
        except Exception as exc:
            if verbose:
                print(f"  NOAA ENC {label} query failed: {exc}")
            continue
        if data and data.get("features"):
            collections.append(data)

    merged = _merge_feature_collections(collections)
    if not merged.get("features"):
        return None, None

    merged["properties"] = {
        "name": loc["name"],
        "source": "NOAA ENC",
        "center_lat": loc["lat"],
        "center_lon": loc["lon"],
        "query_radius_m": search_radius_m,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(merged, f)
    return cache_path, "NOAA ENC Remote"


def _fetch_remote_usgs_geojson(loc, search_radius_m, verbose=False):
    """Fetch inland shoreline geometry from USGS 3DHP and cache it locally."""
    if not _HAS_REQUESTS or not _is_us_bbox(loc["lat"], loc["lon"]):
        return None, None

    cache_path = _remote_modec_cache_path("usgs_3dhp", loc["name"], loc["lat"], loc["lon"], search_radius_m)
    if os.path.isfile(cache_path):
        return cache_path, "USGS 3DHP Remote Cache"

    collections = []
    # Waterbodies first for lakes/reservoirs; flowlines add riverbanks/channels.
    for label, base_url, where in [
        ("Waterbody", _USGS_3DHP_WATERBODY_URL, "1=1"),
        ("Flowline", _USGS_3DHP_FLOWLINE_URL, "FType in (460, 558, 334, 336)"),
    ]:
        try:
            data = _query_arcgis_geojson(base_url, loc["lat"], loc["lon"], search_radius_m, where=where)
        except Exception as exc:
            if verbose:
                print(f"  USGS 3DHP {label} query failed: {exc}")
            continue
        if data and data.get("features"):
            collections.append(data)

    merged = _merge_feature_collections(collections)
    if not merged.get("features"):
        return None, None

    merged["properties"] = {
        "name": loc["name"],
        "source": "USGS 3DHP",
        "center_lat": loc["lat"],
        "center_lon": loc["lon"],
        "query_radius_m": search_radius_m,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(merged, f)
    return cache_path, "USGS 3DHP Remote"


def _fetch_remote_authoritative_modec_geojson(loc, search_radius_m, verbose=False):
    """Bootstrap authoritative shoreline data for first-run open-source users."""
    category = (loc.get("category") or "").lower()
    if category in {"lake", "river", "reservoir"}:
        path, label = _fetch_remote_usgs_geojson(loc, search_radius_m, verbose=verbose)
        if path:
            return path, label
        path, label = _fetch_remote_noaa_enc_geojson(loc, search_radius_m, verbose=verbose)
        return path, label

    path, label = _fetch_remote_noaa_enc_geojson(loc, search_radius_m, verbose=verbose)
    if path:
        return path, label
    path, label = _fetch_remote_usgs_geojson(loc, search_radius_m, verbose=verbose)
    return path, label


def _gather_modec_source_candidates(loc, search_radius_m, radarloc_file=None, verbose=False):
    """Collect all available Mode C coastline candidates before choosing one."""
    _ensure_modec_cache_dirs()
    candidates = []

    if radarloc_file:
        try:
            features, _ = _load_radarloc_file(radarloc_file, verbose=verbose)
            if features:
                candidates.append(_modec_make_candidate(
                    candidate_id="radarloc_explicit",
                    label="radarloc",
                    source_kind="radarloc",
                    features=features,
                    source_path=os.path.abspath(radarloc_file),
                    explicit=True,
                    notes=["User-specified .radarloc coastline source"],
                ))
        except Exception as exc:
            if verbose:
                print(f"  Failed to load .radarloc candidate {radarloc_file}: {exc}")
        return candidates

    local_candidates = _gather_local_modec_candidates(
        loc.get("name"), loc["lat"], loc["lon"], search_radius_m, verbose=verbose)
    candidates.extend(local_candidates)

    category = (loc.get("category") or "").lower()
    if category in {"lake", "river", "reservoir"}:
        remote_order = [
            ("usgs", _fetch_remote_usgs_geojson, "usgs"),
            ("noaa", _fetch_remote_noaa_enc_geojson, "noaa"),
        ]
    else:
        remote_order = [
            ("noaa", _fetch_remote_noaa_enc_geojson, "noaa"),
            ("usgs", _fetch_remote_usgs_geojson, "usgs"),
        ]

    for candidate_id, fetch_fn, kind_prefix in remote_order:
        try:
            remote_path, remote_label = fetch_fn(loc, search_radius_m, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  Remote shoreline fetch failed for {candidate_id}: {exc}")
            continue
        if not remote_path:
            continue
        try:
            features = _load_geojson_features_near_location(
                remote_path, loc["lat"], loc["lon"], search_radius_m)
        except Exception as exc:
            if verbose:
                print(f"  Failed to load cached remote shoreline {remote_path}: {exc}")
            continue
        if not features:
            continue
        source_kind = f"{kind_prefix}_{'cache' if remote_label and 'Cache' in remote_label else 'remote'}"
        candidates.append(_modec_make_candidate(
            candidate_id=f"{candidate_id}_{len(candidates)}",
            label=remote_label or candidate_id.upper(),
            source_kind=source_kind,
            features=features,
            source_path=remote_path,
        ))

    if _HAS_REQUESTS:
        osm_features = _osm_query_water_features(
            loc["lat"], loc["lon"], search_radius_m, simplify_epsilon=2.0)
        if osm_features:
            candidates.append(_modec_make_candidate(
                candidate_id="osm_fallback",
                label="OSM",
                source_kind="osm",
                features=osm_features,
                source_path=None,
                notes=["Live OSM/Overpass fallback candidate"],
            ))

    return candidates


def prefetch_modec_source(location=None, range_nm=None, verbose=True, prefer_source=None):
    """Resolve and cache the best available Mode C shoreline source.

    This is a lightweight bootstrap path for first-run users. It does not run
    the simulator; it only resolves the location and returns the shoreline
    source that would be used for Mode C.
    """
    loc = _resolve_mode12_location(location, range_nm, verbose=verbose)
    if loc is None:
        return None

    selection_radius_m = min(loc["range_m"], RANGE_M * 3.0)
    candidates = _gather_modec_source_candidates(loc, selection_radius_m, verbose=verbose)
    chosen, validation = _select_modec_candidate(
        candidates, loc.get("category"), prefer_source=prefer_source)
    if chosen:
        result = {
            "location": loc,
            "source_type": chosen["label"],
            "source_path": chosen.get("source_path"),
            "feature_count": chosen.get("feature_count", 0),
            "cached": bool(chosen.get("source_path") and "Cache" in chosen["label"]),
            "coastline_validation": validation,
        }
        if verbose:
            print(f"Prefetch: selected shoreline source ({chosen['label']})")
            if chosen.get("source_path"):
                print(f"  Path: {chosen['source_path']}")
            print(f"  Features: {chosen.get('feature_count', 0)}")
        return result

    if verbose:
        print("Prefetch: no usable shoreline candidates found; Mode C would fall back to library coastlines.")
    return {
        "location": loc,
        "source_type": "none",
        "source_path": None,
        "feature_count": 0,
        "cached": False,
        "coastline_validation": validation,
    }


def resolve_modec_location_only(location=None, range_nm=None, verbose=True):
    """Resolve a Mode C location without running shoreline fetch or simulation."""
    loc = _resolve_mode12_location(location, range_nm, verbose=verbose)
    if loc is None:
        return None
    if verbose:
        print("Mode C location resolved:")
        print(f"  Name: {loc['name']}")
        print(f"  Coordinates: ({loc['lat']:.6f}, {loc['lon']:.6f})")
        print(f"  Category: {loc.get('category', 'unknown')}")
        print(f"  Resolution: {loc.get('resolution_method', 'unknown')}")
    return loc


def _candidate_score(query: str, candidate_name: str) -> float:
    """Score a candidate name against the user's query."""
    q = _normalize_location_key(query)
    c = _normalize_location_key(candidate_name)
    if not q or not c:
        return 0.0
    score = 0.0
    if q == c:
        score += 100.0
    elif q in c or c in q:
        score += 70.0
    q_tokens = set(q.split())
    c_tokens = set(c.split())
    if q_tokens and c_tokens:
        score += 20.0 * (len(q_tokens & c_tokens) / max(1, len(q_tokens)))
    score += 25.0 * SequenceMatcher(None, q, c).ratio()
    return score


def _search_local_named_locations(location_name: str, limit: int = 8):
    """Search the built-in location catalog for likely matches."""
    results = []
    seen = set()
    for loc in COASTAL_LOCATIONS:
        score = _candidate_score(location_name, loc["name"])
        base = loc["name"].split(",")[0].strip()
        score = max(score, _candidate_score(location_name, base))
        if score < 25.0:
            continue
        key = (round(loc["lat"], 6), round(loc["lon"], 6), loc["name"])
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "name": loc["name"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "category": loc.get("category", "unknown"),
            "resolution_method": "catalog",
            "score": score,
        })
    results.sort(key=lambda item: (-item["score"], item["name"]))
    return results[:limit]


def _query_gnis_candidates(location_name: str, limit: int = 8, verbose=False):
    """Query GNIS for multiple location candidates."""
    if not _HAS_REQUESTS:
        return []
    state = _extract_state_token(location_name)
    params = {
        "searchText": location_name,
        "searchFields": "FEATURE_NAME",
        "layers": "all",
        "returnGeometry": "true",
        "f": "json",
        "sr": "4326",
    }
    try:
        resp = _requests_lib.get(_GNIS_FIND_URL, params=params, timeout=_MODEC_GNIS_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        if verbose:
            print(f"  GNIS candidate search failed: {exc}")
        return []

    hydro_terms = ("lake", "bay", "harbor", "river", "channel", "sound", "strait", "creek", "reservoir", "marina")
    candidates = []
    for result in payload.get("results", []):
        attrs = result.get("attributes", {}) or {}
        geometry = result.get("geometry") or {}
        lat = geometry.get("y")
        lon = geometry.get("x")
        if lat is None or lon is None:
            lat = attrs.get("PRIM_LAT_DEC") or attrs.get("prim_lat_dec")
            lon = attrs.get("PRIM_LONG_DEC") or attrs.get("prim_long_dec")
        if lat is None or lon is None:
            continue
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            continue
        spatial_ref = geometry.get("spatialReference", {}) if isinstance(geometry, dict) else {}
        wkid = spatial_ref.get("wkid")
        if wkid == 3857 or abs(lat) > 90 or abs(lon) > 180:
            lat, lon = _webmercator_to_latlon(lon, lat)

        name = attrs.get("FEATURE_NAME") or result.get("value") or location_name
        score = _candidate_score(location_name, name)
        state_alpha = (attrs.get("STATE_ALPHA") or attrs.get("state_alpha") or "").upper()
        feat_class = (attrs.get("FEATURE_CLASS") or attrs.get("feature_class") or "").lower()
        if state and state == state_alpha:
            score += 20.0
        if any(term in feat_class for term in hydro_terms):
            score += 15.0
        candidates.append({
            "name": name if not state_alpha else f"{name}, {state_alpha}",
            "lat": lat,
            "lon": lon,
            "category": feat_class or "geocoded",
            "resolution_method": "gnis",
            "score": score,
        })

    candidates.sort(key=lambda item: (-item["score"], item["name"]))
    return candidates[:limit]


def _query_nominatim_candidates(location_name: str, limit: int = 8, verbose=False):
    """Query Nominatim for multiple location candidates."""
    if not _HAS_REQUESTS:
        return []
    _nominatim_rate_limit()
    try:
        resp = _requests_lib.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location_name, "format": "json", "limit": limit},
            headers={"User-Agent": _NOMINATIM_USER_AGENT},
            timeout=_MODEC_NOMINATIM_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json()
    except Exception as exc:
        if verbose:
            print(f"  Nominatim candidate search failed: {exc}")
        return []

    candidates = []
    for row in rows:
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        name = row.get("display_name", location_name)
        candidates.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "category": row.get("type", "geocoded"),
            "resolution_method": "nominatim",
            "score": _candidate_score(location_name, name),
        })
    candidates.sort(key=lambda item: (-item["score"], item["name"]))
    return candidates[:limit]


def search_modec_location_candidates(location=None, limit: int = 8, verbose=True, output_path=None):
    """Return ranked candidate matches for a typed Mode C location."""
    if not location:
        return []

    combined = []
    combined.extend(_search_local_named_locations(location, limit=limit))
    combined.extend(_query_gnis_candidates(location, limit=limit, verbose=verbose))
    combined.extend(_query_nominatim_candidates(location, limit=limit, verbose=verbose))

    deduped = []
    seen = set()
    for item in sorted(combined, key=lambda cand: (-cand["score"], cand["name"])):
        key = (round(item["lat"], 5), round(item["lon"], 5), _normalize_location_key(item["name"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(deduped, 1):
                f.write(f"{idx}\t{item['name']}\t{item['lat']:.6f}\t{item['lon']:.6f}\t{item['resolution_method']}\n")

    if verbose and deduped:
        print("Mode C location candidates:")
        for idx, item in enumerate(deduped, 1):
            print(f"  {idx}. {item['name']} ({item['lat']:.6f}, {item['lon']:.6f}) [{item['resolution_method']}]")
    elif verbose and not deduped:
        print("No location candidates found.")

    return deduped


def _build_modec_location_aliases():
    """Build a deterministic alias map for known locations and common variants."""
    alias_map = {}
    for loc in COASTAL_LOCATIONS:
        canonical = dict(loc)
        canonical["resolution_method"] = "catalog"
        keys = {
            _normalize_location_key(loc["name"]),
            _normalize_location_key(loc["name"].split(",")[0]),
        }
        if "," in loc["name"]:
            parts = [part.strip() for part in loc["name"].split(",") if part.strip()]
            if len(parts) >= 2:
                keys.add(_normalize_location_key(parts[0] + " " + parts[-1]))
        for key in keys:
            if key:
                alias_map[key] = canonical

    # High-value aliases for existing validation locations and common shorthand.
    extra_aliases = {
        "lake murray": "Lake Murray, SC",
        "lake murray sc": "Lake Murray, SC",
        "lake murray south carolina": "Lake Murray, SC",
        "lake murray lexington sc": "Lake Murray, SC",
        "lake murray lexington south carolina": "Lake Murray, SC",
        "lake norman": "Lake Norman, NC",
        "lake norman nc": "Lake Norman, NC",
        "lake norman north carolina": "Lake Norman, NC",
        "lake norman sc": "Lake Norman, NC",
        "lake norman south carolina": "Lake Norman, NC",
        "charleston harbor": "Charleston Harbor, SC",
        "new york harbor": "New York Harbor, NY",
        "san francisco bay": "San Francisco Bay, CA",
    }
    by_name = {loc["name"]: dict(loc, resolution_method="catalog") for loc in COASTAL_LOCATIONS}
    for alias, target in extra_aliases.items():
        if target in by_name:
            alias_map[_normalize_location_key(alias)] = by_name[target]
    return alias_map


MODEC_LOCATION_ALIASES = {}


def _lookup_local_named_location(location_name: str):
    """Resolve a user-entered location against the built-in catalog first."""
    normalized = _normalize_location_key(location_name)
    if not normalized:
        return None
    loc = MODEC_LOCATION_ALIASES.get(normalized)
    if loc:
        return dict(loc)

    # Last-resort fuzzy match against known aliases to tolerate punctuation.
    for key, candidate in MODEC_LOCATION_ALIASES.items():
        if normalized == key or normalized in key or key in normalized:
            return dict(candidate)
    return None


def _iter_modec_geojson_candidates():
    """Yield candidate GeoJSON paths for authoritative shoreline lookup."""
    seen = set()
    labeled_dirs = {os.path.normcase(path): label for label, path in MODEC_SOURCE_DIRS}
    for root in MODEC_GEOJSON_SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in MODEC_SKIP_DIRS]
            for filename in filenames:
                if not filename.lower().endswith(".geojson"):
                    continue
                path = os.path.join(current_root, filename)
                norm = os.path.normcase(path)
                if norm in seen:
                    continue
                seen.add(norm)
                label = "Local GeoJSON"
                current_norm = os.path.normcase(current_root)
                for candidate_root, candidate_label in labeled_dirs.items():
                    if current_norm.startswith(candidate_root):
                        label = candidate_label
                        break
                yield label, path


def _score_modec_geojson_candidate(filepath: str, location_name: str) -> int:
    """Score a local GeoJSON candidate against a requested location name."""
    if not location_name:
        return 0
    wanted = _normalize_location_key(location_name)
    if not wanted:
        return 0

    score = 0
    base = _normalize_location_key(os.path.splitext(os.path.basename(filepath))[0])
    if base == wanted:
        score += 300
    elif wanted in base or base in wanted:
        score += 200

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except Exception:
        return score

    top_props = data.get("properties", {})
    names_to_check = []
    if isinstance(top_props, dict):
        names_to_check.extend([
            top_props.get("name", ""),
            top_props.get("description", ""),
            top_props.get("location_name", ""),
        ])
    for feat in data.get("features", [])[:10]:
        props = feat.get("properties", {})
        if isinstance(props, dict):
            names_to_check.append(props.get("name", ""))

    for name in names_to_check:
        normalized = _normalize_location_key(name)
        if not normalized:
            continue
        if normalized == wanted:
            score += 250
        elif wanted in normalized or normalized in wanted:
            score += 150
    return score


def _coastline_segments_to_feature_dicts(coastlines):
    """Convert CoastlineSegment objects into the Mode C feature dict format."""
    features = []
    for idx, coast in enumerate(coastlines):
        features.append({
            "id": f"geojson_{idx}",
            "name": coast.name,
            "points": list(coast.points),
            "closed": getattr(coast, "closed", True),
        })
    return features


def _load_geojson_features_near_location(filepath, center_lat, center_lon, radius_m):
    """Load a GeoJSON shoreline file into local feature dicts around a location."""
    coastlines = load_coastlines_from_geojson(filepath, center_lat, center_lon, radius_m)
    return _coastline_segments_to_feature_dicts(coastlines)


def _modec_flatten_feature_points(features):
    """Flatten feature dicts into a list of local (x, y) shoreline points."""
    points = []
    for feat in features or []:
        for pt in feat.get("points", []):
            if len(pt) >= 2:
                points.append((float(pt[0]), float(pt[1])))
    return points


def _modec_sample_points(points, max_points=600):
    """Downsample point clouds deterministically for source-comparison metrics."""
    if len(points) <= max_points:
        return list(points)
    step = max(1, int(math.ceil(len(points) / max_points)))
    return list(points[::step])[:max_points]


def _modec_nearest_distance_stats(points_a, points_b):
    """Compute nearest-point distance summary from A to B."""
    if points_a and isinstance(points_a[0], dict):
        points_a = _modec_flatten_feature_points(points_a)
    if points_b and isinstance(points_b[0], dict):
        points_b = _modec_flatten_feature_points(points_b)
    if not points_a or not points_b:
        return None
    sampled_a = _modec_sample_points(points_a)
    sampled_b = _modec_sample_points(points_b)
    dists = []
    for ax, ay in sampled_a:
        best = min(math.hypot(ax - bx, ay - by) for bx, by in sampled_b)
        dists.append(best)
    if not dists:
        return None
    ordered = sorted(dists)
    p95_idx = max(0, min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1))
    return {
        "count_a": len(sampled_a),
        "count_b": len(sampled_b),
        "mean_m": float(sum(dists) / len(dists)),
        "median_m": float(ordered[len(ordered) // 2]),
        "p95_m": float(ordered[p95_idx]),
        "max_m": float(ordered[-1]),
    }


def _modec_feature_metrics(features):
    """Basic geometric metrics for a coastline candidate."""
    points = _modec_flatten_feature_points(features)
    if not points:
        return {
            "feature_count": 0,
            "point_count": 0,
            "min_range_m": None,
            "max_range_m": None,
        }
    radial = [math.hypot(x, y) for x, y in points]
    return {
        "feature_count": len(features),
        "point_count": len(points),
        "min_range_m": float(min(radial)),
        "max_range_m": float(max(radial)),
    }


def _modec_source_preference_score(source_kind, category, explicit=False):
    """Base priority for choosing among coastline sources."""
    if explicit and source_kind == "radarloc":
        return 1000.0

    category = (category or "").lower()
    inland = category in {"lake", "river", "reservoir"}
    source_kind = (source_kind or "").lower()

    if inland:
        table = {
            "local_geojson": 110.0,
            "usgs_remote": 100.0,
            "usgs_cache": 98.0,
            "radarloc": 96.0,
            "osm": 60.0,
            "noaa_remote": 25.0,
            "noaa_cache": 25.0,
        }
    else:
        table = {
            "local_geojson": 110.0,
            "noaa_remote": 100.0,
            "noaa_cache": 98.0,
            "radarloc": 96.0,
            "osm": 70.0,
            "usgs_remote": 55.0,
            "usgs_cache": 55.0,
        }
    return table.get(source_kind, 40.0)


def _modec_source_kind_matches_preference(source_kind, preference):
    """Return True when a candidate source kind satisfies a user preference."""
    if not preference:
        return True
    pref = preference.strip().lower()
    kind = (source_kind or "").lower()
    if pref == "auto":
        return True
    if pref == "local_geojson":
        return kind == "local_geojson"
    if pref == "radarloc":
        return kind == "radarloc"
    if pref == "osm":
        return kind == "osm"
    if pref == "usgs":
        return kind.startswith("usgs_")
    if pref == "noaa":
        return kind.startswith("noaa_")
    return kind == pref


def _modec_make_candidate(candidate_id, label, source_kind, features,
                          source_path=None, notes=None, explicit=False):
    """Build a normalized coastline-source candidate record."""
    metrics = _modec_feature_metrics(features)
    candidate = {
        "id": candidate_id,
        "label": label,
        "source_kind": source_kind,
        "source_path": source_path,
        "features": features or [],
        "explicit": bool(explicit),
        "notes": notes or [],
        "validation": {},
    }
    candidate.update(metrics)
    return candidate


def _gather_local_modec_candidates(location_name, center_lat, center_lon,
                                   search_radius_m, verbose=False):
    """Load all matching local GeoJSON shoreline candidates for a location."""
    gathered = []
    candidates = list(_iter_modec_geojson_candidates())
    if location_name:
        candidates.sort(key=lambda item: _score_modec_geojson_candidate(item[1], location_name), reverse=True)

    for idx, (label, path) in enumerate(candidates):
        try:
            features = _load_geojson_features_near_location(path, center_lat, center_lon, search_radius_m)
        except Exception as exc:
            if verbose:
                print(f"  Skipping shoreline source {path}: {exc}")
            continue
        if not features:
            continue
        if verbose:
            print(f"  Loaded local shoreline candidate from {label}: {path}")
        gathered.append(_modec_make_candidate(
            candidate_id=f"local_{idx}",
            label=label,
            source_kind="local_geojson",
            features=features,
            source_path=path,
        ))
    return gathered


def _validate_modec_candidates(candidates):
    """Annotate source candidates with pairwise agreement metrics."""
    for idx, candidate in enumerate(candidates):
        other_medians = []
        for jdx, other in enumerate(candidates):
            if idx == jdx:
                continue
            stats = _modec_nearest_distance_stats(candidate["features"], other["features"])
            if stats:
                candidate["validation"][other["id"]] = stats
                other_medians.append(stats["median_m"])
        if other_medians:
            candidate["agreement_median_m"] = float(sum(other_medians) / len(other_medians))
            candidate["agreement_count"] = len(other_medians)
        else:
            candidate["agreement_median_m"] = None
            candidate["agreement_count"] = 0
    return candidates


def _score_modec_candidate(candidate, category):
    """Compute final selection score for a coastline-source candidate."""
    score = _modec_source_preference_score(
        candidate.get("source_kind"),
        category,
        explicit=candidate.get("explicit", False),
    )
    score += min(candidate.get("feature_count", 0), 40) * 0.6
    score += min(candidate.get("point_count", 0), 2000) / 2000.0 * 8.0

    min_range = candidate.get("min_range_m")
    if min_range is not None:
        if min_range <= RANGE_M * 1.1:
            score += 12.0
        elif min_range <= RANGE_M * 3.0:
            score += 6.0

    agreement = candidate.get("agreement_median_m")
    if agreement is not None:
        score += max(0.0, 30.0 - min(agreement, 300.0)) / 10.0

    candidate["selection_score"] = float(score)
    return candidate["selection_score"]


def _build_modec_validation_summary(candidates, chosen_id=None):
    """Serialize candidate comparison data for scene metadata."""
    summary = {
        "chosen_source_id": chosen_id,
        "candidates": [],
    }
    for candidate in candidates:
        summary["candidates"].append({
            "id": candidate["id"],
            "label": candidate["label"],
            "source_kind": candidate["source_kind"],
            "source_path": candidate.get("source_path"),
            "feature_count": candidate.get("feature_count", 0),
            "point_count": candidate.get("point_count", 0),
            "min_range_m": candidate.get("min_range_m"),
            "max_range_m": candidate.get("max_range_m"),
            "selection_score": candidate.get("selection_score"),
            "agreement_median_m": candidate.get("agreement_median_m"),
            "agreement_count": candidate.get("agreement_count", 0),
            "notes": list(candidate.get("notes", [])),
            "validation": candidate.get("validation", {}),
        })
    return summary


def _print_modec_validation_summary(validation_summary):
    """Pretty-print coastline source comparison results for Mode C."""
    if not validation_summary:
        print("  Coastline source comparison: no candidates")
        return

    chosen_id = validation_summary.get("chosen_source_id")
    candidates = validation_summary.get("candidates", [])
    print(f"  Coastline candidates evaluated: {len(candidates)}")

    for cand in candidates:
        marker = "*" if cand.get("id") == chosen_id else " "
        label = cand.get("label", cand.get("id", "unknown"))
        score = cand.get("selection_score")
        score_text = f"{score:.1f}" if isinstance(score, (int, float)) else "n/a"
        feature_count = cand.get("feature_count", 0)
        point_count = cand.get("point_count", 0)
        agreement = cand.get("agreement_median_m")
        agreement_text = f"{agreement:.2f} m" if isinstance(agreement, (int, float)) else "n/a"
        source_path = cand.get("source_path")

        print(
            f"   {marker} {label}: "
            f"score={score_text}, "
            f"features={feature_count}, "
            f"points={point_count}, "
            f"agreement={agreement_text}"
        )
        if source_path:
            print(f"      path: {source_path}")

    winner = next((c for c in candidates if c.get("id") == chosen_id), None)
    print(f"  Winner: {winner.get('label', 'none') if winner else 'none'}")


def _select_modec_candidate(candidates, category, prefer_source=None):
    """Choose the best coastline candidate after multi-source validation."""
    if not candidates:
        return None, _build_modec_validation_summary([], None)

    _validate_modec_candidates(candidates)
    for candidate in candidates:
        _score_modec_candidate(candidate, category)

    pool = candidates
    if prefer_source and str(prefer_source).strip().lower() != "auto":
        preferred = [
            candidate for candidate in candidates
            if _modec_source_kind_matches_preference(candidate.get("source_kind"), prefer_source)
        ]
        if preferred:
            for candidate in preferred:
                candidate.setdefault("notes", []).append(
                    f"Matched preferred source override: {prefer_source}"
                )
            pool = preferred
        else:
            for candidate in candidates:
                if candidate.get("id") == candidates[0].get("id"):
                    candidate.setdefault("notes", []).append(
                        f"Preferred source override '{prefer_source}' found no candidates; using auto selection"
                    )
                    break

    chosen = max(pool, key=lambda item: item.get("selection_score", 0.0))
    return chosen, _build_modec_validation_summary(candidates, chosen.get("id"))


def _load_authoritative_modec_features(location_name, center_lat, center_lon,
                                       search_radius_m, verbose=False):
    """Prefer local authoritative shoreline files before live OSM fallback."""
    gathered = _gather_local_modec_candidates(
        location_name, center_lat, center_lon, search_radius_m, verbose=verbose)
    for candidate in gathered:
        return candidate["features"], candidate["label"], candidate["source_path"]
    return [], None, None


def _query_gnis_location(location_name, verbose=False):
    """Resolve a U.S. geographic name through the official USGS GNIS service."""
    if not _HAS_REQUESTS:
        return None

    state = _extract_state_token(location_name)
    params = {
        "searchText": location_name,
        "searchFields": "FEATURE_NAME",
        "layers": "all",
        "returnGeometry": "true",
        "f": "json",
        "sr": "4326",
    }
    try:
        resp = _requests_lib.get(_GNIS_FIND_URL, params=params, timeout=_MODEC_GNIS_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        if verbose:
            print(f"  GNIS lookup failed: {exc}")
        return None

    results = payload.get("results", [])
    if not results:
        return None

    wanted = _normalize_location_key(location_name)
    hydro_terms = ("lake", "bay", "harbor", "harbour", "river", "channel", "sound", "strait", "creek", "reservoir", "marina")

    def _score_result(result):
        attrs = result.get("attributes", {}) or {}
        value = attrs.get("FEATURE_NAME") or result.get("value") or ""
        layer_name = (result.get("layerName") or "").lower()
        feat_class = (attrs.get("FEATURE_CLASS") or attrs.get("feature_class") or "").lower()
        state_alpha = (attrs.get("STATE_ALPHA") or attrs.get("state_alpha") or "").upper()
        score = 0
        normalized_value = _normalize_location_key(value)
        if normalized_value == wanted:
            score += 300
        elif wanted in normalized_value or normalized_value in wanted:
            score += 180
        if state and state_alpha == state:
            score += 120
        if any(term in layer_name for term in ("hydro", "stream")):
            score += 80
        if any(term in feat_class for term in hydro_terms):
            score += 80
        if any(term in wanted for term in hydro_terms) and any(term in normalized_value for term in hydro_terms):
            score += 40
        return score

    best = max(results, key=_score_result)
    attrs = best.get("attributes", {}) or {}
    geometry = best.get("geometry") or {}
    lat = geometry.get("y")
    lon = geometry.get("x")
    if lat is None or lon is None:
        lat = attrs.get("PRIM_LAT_DEC") or attrs.get("prim_lat_dec")
        lon = attrs.get("PRIM_LONG_DEC") or attrs.get("prim_long_dec")
    if lat is None or lon is None:
        return None

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None

    spatial_ref = geometry.get("spatialReference", {}) if isinstance(geometry, dict) else {}
    wkid = spatial_ref.get("wkid")
    if wkid == 3857 or abs(lat) > 90 or abs(lon) > 180:
        lat, lon = _webmercator_to_latlon(lon, lat)

    result = {
        "lat": lat,
        "lon": lon,
        "display_name": attrs.get("FEATURE_NAME") or best.get("value") or location_name,
        "state_alpha": attrs.get("STATE_ALPHA") or attrs.get("state_alpha"),
        "feature_class": attrs.get("FEATURE_CLASS") or attrs.get("feature_class"),
        "feature_id": attrs.get("FEATURE_ID") or attrs.get("feature_id"),
        "source": "GNIS",
    }
    if verbose:
        print(f"  GNIS matched {result['display_name']} ({lat:.6f}, {lon:.6f})")
    return result


# ============================================================
# Mode 12: Real-Coastline Location Database & OSM Functions
# ============================================================

COASTAL_LOCATIONS = [
    # US East Coast
    {"name": "Charleston Harbor, SC", "lat": 32.7765, "lon": -79.9311, "range_nm": 3.0, "category": "harbor"},
    {"name": "Chesapeake Bay Entrance, VA", "lat": 36.9631, "lon": -76.0133, "range_nm": 6.0, "category": "bay"},
    {"name": "Boston Harbor, MA", "lat": 42.3541, "lon": -71.0488, "range_nm": 3.0, "category": "harbor"},
    {"name": "New York Harbor, NY", "lat": 40.6892, "lon": -74.0445, "range_nm": 6.0, "category": "harbor"},
    {"name": "Miami Beach, FL", "lat": 25.7907, "lon": -80.1300, "range_nm": 3.0, "category": "open_coast"},
    {"name": "Savannah River Entrance, GA", "lat": 32.0333, "lon": -80.8833, "range_nm": 3.0, "category": "river"},
    {"name": "Cape Cod Bay, MA", "lat": 41.8500, "lon": -70.2000, "range_nm": 6.0, "category": "bay"},
    {"name": "Narragansett Bay, RI", "lat": 41.5800, "lon": -71.3800, "range_nm": 6.0, "category": "bay"},
    # US West Coast
    {"name": "San Francisco Bay, CA", "lat": 37.8044, "lon": -122.2712, "range_nm": 6.0, "category": "bay"},
    {"name": "Puget Sound, WA", "lat": 47.6062, "lon": -122.3321, "range_nm": 6.0, "category": "channel"},
    {"name": "San Diego Bay, CA", "lat": 32.6859, "lon": -117.1831, "range_nm": 3.0, "category": "bay"},
    {"name": "Columbia River Mouth, OR", "lat": 46.2500, "lon": -124.0500, "range_nm": 6.0, "category": "river"},
    {"name": "Monterey Bay, CA", "lat": 36.8000, "lon": -121.9000, "range_nm": 6.0, "category": "bay"},
    # US Gulf Coast
    {"name": "Galveston Bay, TX", "lat": 29.3013, "lon": -94.7977, "range_nm": 6.0, "category": "bay"},
    {"name": "Mobile Bay, AL", "lat": 30.4383, "lon": -87.9664, "range_nm": 6.0, "category": "bay"},
    {"name": "Tampa Bay, FL", "lat": 27.7600, "lon": -82.5500, "range_nm": 6.0, "category": "bay"},
    # US Lakes & Rivers
    {"name": "Lake Murray, SC", "lat": 34.0637, "lon": -81.2229, "range_nm": 3.0, "category": "lake"},
    {"name": "Lake Norman, NC", "lat": 35.4815, "lon": -80.9054, "range_nm": 3.0, "category": "lake"},
    {"name": "Lake Michigan, Chicago, IL", "lat": 41.8827, "lon": -87.6127, "range_nm": 6.0, "category": "lake"},
    {"name": "Lake Erie, Cleveland, OH", "lat": 41.5088, "lon": -81.6913, "range_nm": 6.0, "category": "lake"},
    {"name": "Lake Tahoe, CA", "lat": 39.0968, "lon": -120.0324, "range_nm": 3.0, "category": "lake"},
    {"name": "Mississippi River, New Orleans, LA", "lat": 29.9511, "lon": -90.0715, "range_nm": 3.0, "category": "river"},
    # International
    {"name": "Tokyo Bay, Japan", "lat": 35.4000, "lon": 139.7700, "range_nm": 6.0, "category": "bay"},
    {"name": "Sydney Harbour, Australia", "lat": -33.8523, "lon": 151.2108, "range_nm": 3.0, "category": "harbor"},
    {"name": "Hong Kong Victoria Harbour", "lat": 22.2855, "lon": 114.1577, "range_nm": 3.0, "category": "harbor"},
    {"name": "Singapore Strait", "lat": 1.2600, "lon": 103.8300, "range_nm": 6.0, "category": "channel"},
    {"name": "Rotterdam Europoort, Netherlands", "lat": 51.9600, "lon": 4.1300, "range_nm": 6.0, "category": "harbor"},
    {"name": "English Channel, Dover, UK", "lat": 51.1300, "lon": 1.3100, "range_nm": 6.0, "category": "channel"},
    {"name": "Gibraltar Strait", "lat": 35.9867, "lon": -5.6100, "range_nm": 6.0, "category": "channel"},
    {"name": "Panama Canal, Pacific Entrance", "lat": 8.9500, "lon": -79.5600, "range_nm": 3.0, "category": "channel"},
    {"name": "Suez Canal Entrance, Egypt", "lat": 29.9167, "lon": 32.5667, "range_nm": 3.0, "category": "channel"},
    {"name": "Bergen, Norway", "lat": 60.3913, "lon": 5.3221, "range_nm": 3.0, "category": "fjord"},
    {"name": "Geiranger Fjord, Norway", "lat": 62.1000, "lon": 7.0900, "range_nm": 3.0, "category": "fjord"},
    {"name": "Piraeus, Greece", "lat": 37.9475, "lon": 23.6370, "range_nm": 3.0, "category": "harbor"},
    {"name": "Istanbul Bosphorus, Turkey", "lat": 41.0082, "lon": 29.0050, "range_nm": 3.0, "category": "channel"},
    {"name": "Rio de Janeiro, Brazil", "lat": -22.9068, "lon": -43.1729, "range_nm": 6.0, "category": "bay"},
    {"name": "Cape Town, South Africa", "lat": -33.9249, "lon": 18.4241, "range_nm": 6.0, "category": "open_coast"},
    {"name": "Mumbai Harbour, India", "lat": 18.9388, "lon": 72.8354, "range_nm": 3.0, "category": "harbor"},
    {"name": "Vancouver Harbour, BC", "lat": 49.2906, "lon": -123.1046, "range_nm": 3.0, "category": "harbor"},
    # Islands
    {"name": "Key West, FL", "lat": 24.5551, "lon": -81.7800, "range_nm": 3.0, "category": "island"},
    {"name": "Catalina Island, CA", "lat": 33.3872, "lon": -118.4165, "range_nm": 3.0, "category": "island"},
    {"name": "Bermuda", "lat": 32.3078, "lon": -64.7505, "range_nm": 3.0, "category": "island"},
]

MODEC_LOCATION_ALIASES = _build_modec_location_aliases()


# ---- Nominatim Geocoding (rate-limited) ----

_NOMINATIM_LAST_REQUEST = 0.0
_NOMINATIM_USER_AGENT = "DRS4DNXT_RadarSimulator/1.0"


def _nominatim_rate_limit():
    """Enforce 1 request/sec rate limit for Nominatim (OSM usage policy)."""
    global _NOMINATIM_LAST_REQUEST
    elapsed = _time_module.time() - _NOMINATIM_LAST_REQUEST
    if elapsed < 1.0:
        _time_module.sleep(1.0 - elapsed)
    _NOMINATIM_LAST_REQUEST = _time_module.time()


def _geocode_location(location_name, verbose=False):
    """Convert a location name to lat/lon via Nominatim fallback."""
    if not _HAS_REQUESTS:
        return None
    _nominatim_rate_limit()
    try:
        resp = _requests_lib.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location_name, "format": "json", "limit": 1},
            headers={"User-Agent": _NOMINATIM_USER_AGENT},
            timeout=_MODEC_NOMINATIM_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        r = results[0]
        return {
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "display_name": r.get("display_name", location_name),
            "source": "Nominatim",
        }
    except Exception as exc:
        if verbose:
            print(f"  Nominatim lookup failed: {exc}")
        return None


# ---- OSM Query Functions (adapted from marine-radar-location-generator) ----

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _post_overpass_query(query_text, verbose=False):
    """Submit an Overpass query with resilient headers and endpoint fallback."""
    if not _HAS_REQUESTS:
        return None

    last_error = None
    for endpoint in _MODEC_OVERPASS_ENDPOINTS:
        # Try classic form-encoded first for compatibility, then raw plain text
        attempts = (
            ("form", {"data": query_text}),
            ("plain", query_text.encode("utf-8")),
        )
        for mode, payload in attempts:
            try:
                if mode == "form":
                    resp = _requests_lib.post(
                        endpoint,
                        data=payload,
                        headers=_MODEC_OVERPASS_HEADERS,
                        timeout=_MODEC_OVERPASS_TIMEOUT,
                    )
                else:
                    resp = _requests_lib.post(
                        endpoint,
                        data=payload,
                        headers=_MODEC_OVERPASS_HEADERS,
                        timeout=_MODEC_OVERPASS_TIMEOUT,
                    )

                if resp.status_code == 429:
                    if verbose:
                        print(f"    Overpass rate-limited at {endpoint}, retrying in {_MODEC_OVERPASS_RETRY_DELAY_S}s...")
                    _time_module.sleep(_MODEC_OVERPASS_RETRY_DELAY_S)
                    continue

                resp.raise_for_status()
                data = resp.json()
                if verbose:
                    print(f"    Overpass query succeeded via {endpoint} ({mode})")
                return data
            except Exception as exc:
                last_error = f"{endpoint} [{mode}] -> {exc}"
                if verbose:
                    print(f"    Overpass attempt failed: {last_error}")
                continue
    raise RuntimeError(last_error or "Overpass query failed")


def _osm_douglas_peucker(points, epsilon):
    """Simplify a polyline using the Douglas-Peucker algorithm.

    Args:
        points: List of (x, y) tuples in local meters.
        epsilon: Maximum allowed deviation in meters.

    Returns:
        Simplified list of (x, y) tuples.
    """
    if len(points) <= 2:
        return points

    start, end = points[0], points[-1]
    max_dist = 0.0
    max_idx = 0
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    line_len_sq = dx * dx + dy * dy

    for i in range(1, len(points) - 1):
        px, py = points[i][0] - start[0], points[i][1] - start[1]
        if line_len_sq > 0:
            t = max(0.0, min(1.0, (px * dx + py * dy) / line_len_sq))
            proj_x, proj_y = t * dx, t * dy
        else:
            proj_x, proj_y = 0.0, 0.0
        dist = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        left = _osm_douglas_peucker(points[:max_idx + 1], epsilon)
        right = _osm_douglas_peucker(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def _osm_assemble_multipolygon(way_segments, center_lat, center_lon,
                                nodes, epsilon):
    """Assemble way segments into closed polygons by connecting endpoints.

    OSM multipolygon relations have multiple ways that connect end-to-end
    to form closed rings. Joins them, converts to local XY, and simplifies.

    Returns list of (points, closed) tuples — points as (x, y) in meters.
    """
    if not way_segments:
        return []

    segments = []
    for way_id, way_nodes in way_segments:
        if len(way_nodes) < 2:
            continue
        points = []
        for nid in way_nodes:
            if nid in nodes:
                lat, lon = nodes[nid]
                x, y = latlon_to_local_xy(lat, lon, center_lat, center_lon)
                points.append((x, y))
        if len(points) >= 2:
            segments.append({
                'start_node': way_nodes[0],
                'end_node': way_nodes[-1],
                'points': points,
            })

    if not segments:
        return []

    assembled = []
    used = set()

    while len(used) < len(segments):
        ring_points = []
        for i, seg in enumerate(segments):
            if i not in used:
                ring_points = list(seg['points'])
                current_end = seg['end_node']
                start_node = seg['start_node']
                used.add(i)
                break

        max_iterations = len(segments) * 2
        for _ in range(max_iterations):
            found = False
            for i, seg in enumerate(segments):
                if i in used:
                    continue
                if seg['start_node'] == current_end:
                    ring_points.extend(seg['points'][1:])
                    current_end = seg['end_node']
                    used.add(i)
                    found = True
                    break
                elif seg['end_node'] == current_end:
                    ring_points.extend(reversed(seg['points'][:-1]))
                    current_end = seg['start_node']
                    used.add(i)
                    found = True
                    break

            if not found or current_end == start_node:
                break

        is_closed = (current_end == start_node) and len(ring_points) >= 3
        simplified = _osm_douglas_peucker(ring_points, epsilon)
        if len(simplified) >= 3:
            assembled.append((simplified, is_closed))

    return assembled


def _osm_query_water_features(center_lat, center_lon, radius_m,
                               simplify_epsilon=2.0):
    """Query Overpass API for water boundaries near a location.

    Fetches coastlines, lakes, reservoirs, and riverbanks within radius.
    Assembles multipolygon relations. Points returned as (x, y) tuples
    in local meters (relative to center_lat/center_lon).

    Uses simplify_epsilon=2.0 (not 50.0) because DRS4DNXT range is small.

    Returns list of dicts with keys: id, name, points [(x,y)...], closed.
    """
    if not _HAS_REQUESTS:
        return []

    query = f"""
    [out:json][timeout:60];
    (
      way["natural"="coastline"](around:{radius_m},{center_lat},{center_lon});
      way["natural"="water"](around:{radius_m},{center_lat},{center_lon});
      relation["natural"="water"](around:{radius_m},{center_lat},{center_lon});
      way["waterway"="riverbank"](around:{radius_m},{center_lat},{center_lon});
    );
    out body;
    >;
    out skel qt;
    """

    try:
        data = _post_overpass_query(query, verbose=False)
    except Exception as e:
        print(f"    Overpass API error: {e}")
        return []

    # Build node lookup
    nodes = {}
    for el in data.get("elements", []):
        if el["type"] == "node":
            nodes[el["id"]] = (el["lat"], el["lon"])

    # Build way lookup
    way_data = {}
    for el in data.get("elements", []):
        if el["type"] == "way" and "nodes" in el:
            way_data[el["id"]] = {
                'nodes': el["nodes"],
                'tags': el.get("tags", {}),
            }

    # Track which ways belong to relations
    relation_way_ids = set()

    results = []

    # Process relations first — assemble multipolygons
    for el in data.get("elements", []):
        if el["type"] != "relation":
            continue

        tags = el.get("tags", {})
        rel_name = tags.get("name", "")
        rel_type = tags.get("water", tags.get("natural", "water"))

        outer_ways = []
        for member in el.get("members", []):
            if member["type"] == "way" and member.get("role") in ("outer", ""):
                way_id = member["ref"]
                relation_way_ids.add(way_id)
                if way_id in way_data:
                    outer_ways.append((way_id, way_data[way_id]['nodes']))

        assembled = _osm_assemble_multipolygon(outer_ways, center_lat, center_lon,
                                                nodes, simplify_epsilon)
        for i, (points, is_closed) in enumerate(assembled):
            in_range = any(
                math.sqrt(p[0] ** 2 + p[1] ** 2) <= radius_m * 1.2
                for p in points
            )
            if not in_range:
                continue
            results.append({
                "id": f"relation_{el['id']}_{i}",
                "name": rel_name or f"{rel_type}_{el['id']}",
                "points": points,
                "closed": is_closed,
            })

    # Process standalone ways
    for way_id, wd in way_data.items():
        if way_id in relation_way_ids:
            continue

        way_nodes = wd['nodes']
        raw_points = []
        for nid in way_nodes:
            if nid in nodes:
                lat, lon = nodes[nid]
                x, y = latlon_to_local_xy(lat, lon, center_lat, center_lon)
                raw_points.append((x, y))
        if len(raw_points) < 3:
            continue

        closed = (way_nodes[0] == way_nodes[-1]) and len(way_nodes) > 3
        simplified = _osm_douglas_peucker(raw_points, simplify_epsilon)
        if len(simplified) < 3:
            continue

        in_range = any(
            math.sqrt(p[0] ** 2 + p[1] ** 2) <= radius_m * 1.2
            for p in simplified
        )
        if not in_range:
            continue

        tags = wd['tags']
        name = tags.get("name", "")
        water_type = tags.get("water", tags.get("natural", "shoreline"))

        results.append({
            "id": f"way_{way_id}",
            "name": name or f"{water_type}_{way_id}",
            "points": simplified,
            "closed": closed,
        })

    return results


def _clip_polygon_to_range(pts, max_range_m):
    """Clip a polygon to a circular range boundary with interpolation.

    Instead of simply dropping points beyond range (which loses edge shape),
    this interpolates where each polygon edge crosses the range circle,
    preserving the coastline contour at the boundary.

    Args:
        pts: List of (x, y) tuples forming the polygon
        max_range_m: Clipping radius in meters

    Returns:
        List of (x, y) tuples forming the clipped polygon
    """
    clip_r = max_range_m * 1.1
    n = len(pts)
    if n < 2:
        return pts

    clipped = []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        d1 = math.sqrt(x1 * x1 + y1 * y1)
        d2 = math.sqrt(x2 * x2 + y2 * y2)

        if d1 <= clip_r:
            clipped.append((x1, y1))
            if d2 > clip_r and d2 - d1 > 0.01:
                # Edge exits range — interpolate exit point
                t = (clip_r - d1) / (d2 - d1)
                t = max(0.0, min(1.0, t))
                clipped.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        elif d2 <= clip_r and d1 - d2 > 0.01:
            # Edge enters range — interpolate entry point
            t = (d1 - clip_r) / (d1 - d2)
            t = max(0.0, min(1.0, t))
            clipped.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))

    return clipped


def _append_boundary_arc(points, max_range_m, clockwise=False, steps=24):
    """Close a clipped polygon along the radar boundary instead of a straight chord.

    When a real closed shoreline polygon is clipped to the circular radar range,
    connecting the clipped endpoints directly creates an artificial straight wall
    across the view. This helper reconnects the endpoints along the range circle
    so any synthetic closure stays on the outer boundary instead.
    """
    if len(points) < 2:
        return points

    clip_r = max_range_m * 1.1
    start = points[-1]
    end = points[0]

    start_r = math.hypot(start[0], start[1])
    end_r = math.hypot(end[0], end[1])
    if abs(start_r - clip_r) > 2.0 or abs(end_r - clip_r) > 2.0:
        return points

    start_angle = math.atan2(start[1], start[0])
    end_angle = math.atan2(end[1], end[0])
    if clockwise:
        while end_angle >= start_angle:
            end_angle -= math.tau
    else:
        while end_angle <= start_angle:
            end_angle += math.tau

    delta = end_angle - start_angle
    if abs(delta) < math.radians(3):
        return points

    arc_points = []
    for i in range(1, max(2, steps)):
        t = i / max(2, steps)
        ang = start_angle + delta * t
        arc_points.append((clip_r * math.cos(ang), clip_r * math.sin(ang)))

    return points + arc_points


def _clip_closed_polygon_with_boundary_arc(pts, max_range_m):
    """Clip a closed polygon and, if clipped, reconnect it along the range arc."""
    clip_r = max_range_m * 1.1
    had_outside = any(math.hypot(x, y) > clip_r for x, y in pts)
    clipped = _clip_polygon_to_range(pts, max_range_m)
    if not clipped:
        return clipped
    if had_outside and len(clipped) >= 3:
        clockwise = _polygon_area(pts) < 0
        clipped = _append_boundary_arc(clipped, max_range_m, clockwise=clockwise)
    return clipped


def _osm_features_to_coastline_segments(features, max_range_m):
    """Convert OSM query results to CoastlineSegment objects.

    Closed polygons are clipped to radar range with boundary interpolation
    (not simple point-dropping). Open ways are buffered into thin 3m-width
    polygons. All segments receive Chaikin + Gaussian smoothing.

    Args:
        features: List of dicts from _osm_query_water_features()
        max_range_m: Maximum range for clipping

    Returns:
        List of CoastlineSegment objects
    """
    coastlines = []

    for feat in features:
        pts = feat["points"]
        name = feat.get("name", "osm_coastline")
        is_closed = feat.get("closed", False)

        # Check if ANY part of this feature intersects the range
        has_inside = any(math.sqrt(p[0] ** 2 + p[1] ** 2) <= max_range_m * 1.1
                         for p in pts)
        if not has_inside:
            continue

        # Closed rings reconnect along the boundary arc after clipping so they
        # do not form a synthetic straight closure wall across the view.
        clipped = (_clip_closed_polygon_with_boundary_arc(pts, max_range_m)
                   if is_closed else _clip_polygon_to_range(pts, max_range_m))

        if len(clipped) >= 3:
            smoothed = _chaikin_smooth(clipped, iterations=2)
            smoothed = _smooth_polygon_points(smoothed)
            coastlines.append(CoastlineSegment(points=smoothed, name=name, closed=is_closed))
        elif len(clipped) >= 2:
            # Too few points for a polygon — buffer into thin strip
            width = 3.0
            left_pts = []
            right_pts = []
            half_w = width / 2.0
            for i in range(len(clipped)):
                if i < len(clipped) - 1:
                    dx = clipped[i + 1][0] - clipped[i][0]
                    dy = clipped[i + 1][1] - clipped[i][1]
                else:
                    dx = clipped[i][0] - clipped[i - 1][0]
                    dy = clipped[i][1] - clipped[i - 1][1]
                length = math.sqrt(dx * dx + dy * dy)
                if length < 1e-9:
                    continue
                nx, ny = -dy / length, dx / length
                left_pts.append((clipped[i][0] + nx * half_w, clipped[i][1] + ny * half_w))
                right_pts.append((clipped[i][0] - nx * half_w, clipped[i][1] - ny * half_w))
            if len(left_pts) >= 2:
                polygon = left_pts + list(reversed(right_pts))
                polygon = _chaikin_smooth(polygon, iterations=2)
                polygon = _smooth_polygon_points(polygon)
                coastlines.append(CoastlineSegment(points=polygon, name=name, closed=True))

    return coastlines


def _reposition_radar_near_coastline(features, sim_range_m, verbose=False):
    """Find optimal radar position near the closest coastline feature.

    OSM features are in local (x, y) meters relative to the location center.
    The DRS4DNXT range is only 231.5m, but location centers are often km
    from the nearest shoreline. This function finds the nearest coastline
    point and calculates an offset to place the radar ~60% of range from it.

    Args:
        features: List of dicts from _osm_query_water_features(), each with
                  'points' as list of (x, y) tuples in local meters
        sim_range_m: Simulator range (RANGE_M = 231.5m)
        verbose: Print repositioning info

    Returns:
        (offset_x, offset_y): Offset to subtract from all feature coordinates
        to reposition origin near coastline. (0, 0) if no repositioning needed.
    """
    # Scan all feature points to find the nearest one to center
    min_dist = float('inf')
    nearest_pt = None

    for feat in features:
        for pt in feat["points"]:
            d = math.sqrt(pt[0] ** 2 + pt[1] ** 2)
            if d < min_dist:
                min_dist = d
                nearest_pt = pt

    if nearest_pt is None:
        return (0.0, 0.0)

    # If coastline is already within range, no repositioning needed
    if min_dist <= sim_range_m * 0.8:
        if verbose:
            print(f"    Nearest coastline at {min_dist:.0f} m — already within range")
        return (0.0, 0.0)

    # Position radar so coastline is at ~60% of range (leaves room for objects
    # on the water side and coastline extends into the outer 40%)
    target_coast_dist = sim_range_m * 0.6  # ~139m from radar to coastline

    # Move from center toward nearest coastline, stopping target_coast_dist
    # short of it. Direction: center → nearest point (center is on water).
    move_dist = min_dist - target_coast_dist
    dx = nearest_pt[0] / min_dist
    dy = nearest_pt[1] / min_dist
    offset_x = dx * move_dist
    offset_y = dy * move_dist

    if verbose:
        print(f"    Nearest coastline at {min_dist:.0f} m from location center")
        print(f"    Repositioning radar {move_dist:.0f} m toward coastline")
        print(f"    Coastline will be ~{target_coast_dist:.0f} m from radar")

    return (offset_x, offset_y)


def _load_radarloc_file(filepath, verbose=False):
    """Load a .radarloc file and return OSM-compatible feature list + location info.

    .radarloc files are JSON documents with metadata (lat/lon/range) and
    coastline polygons in local (x, y) meters. This converts them to the
    same format as _osm_query_water_features() output.

    Args:
        filepath: Path to .radarloc file
        verbose: Print loading info

    Returns:
        (features, loc_info) where:
          features: list of dicts with id, name, points [(x,y)...], closed
          loc_info: dict with lat, lon, name, range_nm, range_m, category
    """
    with open(filepath, 'r') as f:
        doc = json.load(f)

    meta = doc.get("metadata", {})
    lat = meta.get("center_lat", 0.0)
    lon = meta.get("center_lon", 0.0)
    name = meta.get("location_name", os.path.basename(filepath))
    range_nm = meta.get("range_nm", 3.0)

    loc_info = {
        "lat": lat, "lon": lon,
        "name": name, "range_nm": range_nm,
        "range_m": range_nm * 1852.0, "category": "radarloc",
        "resolution_method": "radarloc",
        "query_text": filepath,
    }

    features = []
    for coast in doc.get("coastlines", []):
        pts_raw = coast.get("points", [])
        if len(pts_raw) < 3:
            continue
        # .radarloc points are dicts with x/y keys
        if isinstance(pts_raw[0], dict):
            points = [(p["x"], p["y"]) for p in pts_raw]
        else:
            points = [(p[0], p[1]) for p in pts_raw]
        features.append({
            "id": coast.get("id", f"radarloc_{len(features)}"),
            "name": coast.get("name", "coastline"),
            "points": points,
            "closed": coast.get("closed", True),
        })

    if verbose:
        print(f"  Loaded .radarloc: {filepath}")
        print(f"  Location: {name} ({lat:.4f}, {lon:.4f})")
        print(f"  Range: {range_nm:.1f} NM, Features: {len(features)}")

    return features, loc_info


def _load_mode12_source(location, range_nm=None, radarloc_file=None, verbose=False,
                        prefer_source=None, local_radarloc_production=False, full_range=False):
    """Resolve the real-world source for Mode 12/C coastline generation.

    Returns:
        (loc, osm_features, loc_range_nm, loc_range_m, validation_summary)
    """
    osm_features = None
    validation_summary = None
    _ensure_modec_cache_dirs()
    selection_radius_m = RANGE_M * 3.0
    if radarloc_file:
        osm_features, loc = _load_radarloc_file(radarloc_file, verbose=verbose)
        if range_nm is not None:
            loc["range_nm"] = range_nm
            loc["range_m"] = range_nm * 1852.0
        selection_radius_m = loc["range_m"] if full_range else min(loc["range_m"], RANGE_M * 3.0)
        if local_radarloc_production:
            # A selected local scene is already the production geometry. Do
            # not re-rank sources or trigger a network-backed rebuild.
            loc["source_type"] = "radarloc"
            loc["source_path"] = os.path.abspath(radarloc_file)
            validation_summary = {
                "mode": "local_radarloc_production",
                "network_queries": 0,
                "source_candidates_considered": 0,
            }
        else:
            candidates = _gather_modec_source_candidates(
                loc, selection_radius_m, radarloc_file=radarloc_file, verbose=verbose)
            chosen, validation_summary = _select_modec_candidate(
                candidates, loc.get("category"), prefer_source=prefer_source)
            if chosen:
                osm_features = chosen["features"]
                loc["source_type"] = chosen["label"]
                loc["source_path"] = chosen.get("source_path")
            else:
                loc["source_type"] = "radarloc"
                loc["source_path"] = os.path.abspath(radarloc_file)
    else:
        loc = _resolve_mode12_location(location, range_nm, verbose=verbose)
        if loc is None:
            return None, None, None, None, None
        selection_radius_m = loc["range_m"] if full_range else min(loc["range_m"], RANGE_M * 3.0)

        candidates = _gather_modec_source_candidates(loc, selection_radius_m, verbose=verbose)
        chosen, validation_summary = _select_modec_candidate(
            candidates, loc.get("category"), prefer_source=prefer_source)
        if chosen:
            osm_features = chosen["features"]
            loc["source_type"] = chosen["label"]
            loc["source_path"] = chosen.get("source_path")
        else:
            if not _HAS_REQUESTS:
                print("ERROR: No shoreline candidates found and the 'requests' library is unavailable.")
                print("  Install with: pip install requests")
                print("  Or place NOAA/USGS/GeoJSON shoreline files under shoreline_sources/.")
                return None, None, None, None, None

    return loc, osm_features, loc["range_nm"], loc["range_m"], validation_summary


def _shift_mode12_features_near_coastline(osm_features, loc, sim_range_m, verbose=False):
    """Reposition the local radar origin so real coastlines fall within DRS4DNXT range."""
    if not osm_features:
        return loc, osm_features

    offset_x, offset_y = _reposition_radar_near_coastline(
        osm_features, sim_range_m, verbose=verbose)
    if offset_x == 0 and offset_y == 0:
        return loc, osm_features

    shifted_features = []
    for feat in osm_features:
        shifted = dict(feat)
        shifted["points"] = [(p[0] - offset_x, p[1] - offset_y)
                             for p in feat["points"]]
        shifted_features.append(shifted)

    shifted_loc = dict(loc)
    d2r = math.pi / 180.0
    shifted_loc["lat"] += offset_y / (EARTH_RADIUS_M * d2r)
    shifted_loc["lon"] += offset_x / (math.cos(shifted_loc["lat"] * d2r) * EARTH_RADIUS_M * d2r)

    if verbose:
        print(f"  Radar repositioned to ({shifted_loc['lat']:.6f}, {shifted_loc['lon']:.6f})")

    return shifted_loc, shifted_features


def _prepare_mode12_coastlines(location=None, range_nm=None, radarloc_file=None,
                               library=None, verbose=False, prefer_source=None):
    """Prepare Mode 12/C coastline inputs from authoritative data, OSM, or `.radarloc`.

    Returns:
        dict with loc, coastlines, echo_stats_map, osm_features, metadata,
        sim_range_m, sim_range_nm, loc_range_m, loc_range_nm
    """
    loc, osm_features, loc_range_nm, loc_range_m, validation_summary = _load_mode12_source(
        location, range_nm=range_nm, radarloc_file=radarloc_file,
        verbose=verbose, prefer_source=prefer_source)
    if loc is None:
        return None

    sim_range_m = RANGE_M
    sim_range_nm = RANGE_NM
    loc, osm_features = _shift_mode12_features_near_coastline(
        osm_features, loc, sim_range_m, verbose=verbose)

    coastlines = []
    echo_stats_map = {}
    use_fractal_fallback = False

    if osm_features:
        coastlines = _osm_features_to_coastline_segments(osm_features, sim_range_m)
        if verbose:
            print(f"  Source features: {len(osm_features)} -> {len(coastlines)} coastline segments")
        if coastlines and library and library.primitives:
            echo_stats_map = assign_echo_profiles_to_geojson_coastlines(coastlines, library)
            if verbose:
                print(f"  Echo profiles matched: {len(echo_stats_map)}/{len(coastlines)}")
    else:
        use_fractal_fallback = True

    if not coastlines:
        use_fractal_fallback = True

    if use_fractal_fallback:
        if verbose:
            print("  No usable real coastline geometry found -> falling back to library coastlines")
        if library and library.primitives:
            coastlines, echo_stats_map = generate_random_coastlines_from_library(library, sim_range_m)
        else:
            scene_template_files = _discover_scene_template_files()
            if scene_template_files:
                coastlines, echo_stats_map = generate_coastlines_from_scene_template(
                    scene_template_files, sim_range_m)
            else:
                if verbose:
                    print("  No library or templates available -> skipping coastlines")
                coastlines = []
                echo_stats_map = {}

    metadata = {
        "location_name": loc["name"],
        "lat": loc["lat"],
        "lon": loc["lon"],
        "category": loc.get("category", "unknown"),
        "resolution_method": loc.get("resolution_method", "unknown"),
        "query_text": loc.get("query_text"),
        "source_type": loc.get("source_type", "unknown"),
        "source_path": loc.get("source_path"),
        "query_radius_m": loc_range_m,
        "selection_radius_m": RANGE_M * 3.0 if loc_range_m > RANGE_M * 3.0 else loc_range_m,
        "source_features_found": len(osm_features) if osm_features else 0,
        "coastline_segments": len(coastlines),
        "fractal_fallback": use_fractal_fallback,
        "coastline_validation": validation_summary,
    }

    return {
        "loc": loc,
        "osm_features": osm_features,
        "coastlines": coastlines,
        "echo_stats_map": echo_stats_map,
        "metadata": metadata,
        "sim_range_m": sim_range_m,
        "sim_range_nm": sim_range_nm,
        "loc_range_m": loc_range_m,
        "loc_range_nm": loc_range_nm,
    }


def _build_mode12_scene(config, coastlines, objects, sim_range_m, sim_range_nm,
                        metadata=None, echo_stats_map=None):
    """Build the geometric scene dict used by Mode 12/C flows."""
    scene = {
        "radar_config": {
            "status": STATUS, "scale": SCALE, "range_index": RANGE_INDEX,
            "gain": config.gain, "bins": NUM_BINS, "bin_size_m": BIN_SIZE_M,
            "dead_zone_bins": DEAD_ZONE_BINS, "range_m": sim_range_m,
            "range_nm": sim_range_nm, "ticks_per_rev": TICKS_PER_REV,
        },
        "clutter": {
            "noise_probability": config.gain * 0.0001,
            "noise_codes": [8, 20, 28],
        },
        "decay": {
            "ramp_up_range": [80, 200],
            "ramp_down_range": [40, 160],
        },
        "coastlines": [
            {"name": c.name, "points": [[p[0], p[1]] for p in c.points],
             "closed": getattr(c, "closed", True)}
            for c in coastlines
        ],
        "objects": [
            {"type": obj.obj_type, "x": obj.x, "y": obj.y,
             "rcs_m2": obj.rcs_m2, "length_m": obj.length_m, "width_m": obj.width_m,
             "heading_deg": obj.heading_deg, "speed_kts": obj.speed_kts,
             "course_deg": obj.course_deg, "turn_rate_dps": obj.turn_rate_dps}
            for obj in objects
        ],
    }
    if metadata:
        scene["mode12_metadata"] = metadata
    if echo_stats_map:
        scene["coastline_stats"] = echo_stats_map
    return scene


def _resolve_mode12_location(location_str, range_nm, verbose=False):
    """Resolve a Mode 12 location specification.

    Three cases:
      1. None → pick random from COASTAL_LOCATIONS
      2. Matches 'lat,lon' regex → use directly
      3. Name string → local catalog → GNIS → Nominatim fallback

    Returns dict with: lat, lon, name, range_nm, range_m
    """
    NM_TO_M = 1852.0

    if location_str is None:
        # Random location from database
        loc = random.choice(COASTAL_LOCATIONS)
        rnm = range_nm if range_nm is not None else loc["range_nm"]
        return {
            "lat": loc["lat"], "lon": loc["lon"],
            "name": loc["name"], "range_nm": rnm,
            "range_m": rnm * NM_TO_M, "category": loc.get("category", "unknown"),
        }

    # Try lat,lon pattern
    m = _re_module.match(r'^([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)$', location_str.strip())
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        rnm = range_nm if range_nm is not None else 3.0
        return {
            "lat": lat, "lon": lon,
            "name": f"({lat:.4f}, {lon:.4f})", "range_nm": rnm,
            "range_m": rnm * NM_TO_M, "category": "custom",
            "resolution_method": "coordinates",
            "query_text": location_str,
        }

    # Known project locations and curated aliases come first.
    local_match = _lookup_local_named_location(location_str)
    if local_match:
        rnm = range_nm if range_nm is not None else local_match["range_nm"]
        return {
            "lat": local_match["lat"], "lon": local_match["lon"],
            "name": local_match["name"], "range_nm": rnm,
            "range_m": rnm * NM_TO_M, "category": local_match.get("category", "known"),
            "resolution_method": local_match.get("resolution_method", "catalog"),
            "query_text": location_str,
        }

    # Prefer the official GNIS service for U.S. place names.
    result = _query_gnis_location(location_str, verbose=verbose)
    if result:
        rnm = range_nm if range_nm is not None else 3.0
        return {
            "lat": result["lat"], "lon": result["lon"],
            "name": result.get("display_name", location_str), "range_nm": rnm,
            "range_m": rnm * NM_TO_M, "category": "geocoded",
            "resolution_method": "gnis",
            "query_text": location_str,
            "state_alpha": result.get("state_alpha"),
            "feature_class": result.get("feature_class"),
            "feature_id": result.get("feature_id"),
        }

    # Final fallback for global/open-ended names.
    result = _geocode_location(location_str, verbose=verbose)
    if result:
        rnm = range_nm if range_nm is not None else 3.0
        return {
            "lat": result["lat"], "lon": result["lon"],
            "name": result.get("display_name", location_str), "range_nm": rnm,
            "range_m": rnm * NM_TO_M, "category": "geocoded",
            "resolution_method": "nominatim",
            "query_text": location_str,
        }

    print(f"    ERROR: Could not resolve location '{location_str}' to coordinates.")
    print("    Try a known location name, a more specific state-qualified name, or 'lat,lon'.")
    return None


# ============================================================
# Coastline Generators
# ============================================================
def generate_arc_coastline(center_bearing_deg: float, arc_span_deg: float,
                           min_range_m: float, max_range_m: float,
                           num_points: int = None) -> CoastlineSegment:
    """Generate an arc-shaped coastline (harbor wall, island shore, etc.)."""
    if num_points is None:
        num_points = adaptive_point_count(arc_span_deg, max_range_m)
    points = []
    start = center_bearing_deg - arc_span_deg / 2
    end = center_bearing_deg + arc_span_deg / 2
    base_depth = max_range_m - min_range_m
    depth_profile = variable_depth_profile(num_points, base_depth,
                                           min_depth=base_depth * 0.6,
                                           max_depth=base_depth * 1.4)
    inner_noise = fractal_noise_1d(num_points, amplitude=random.uniform(3.0, 6.0), beta=2.0)
    outer_noise = fractal_noise_1d(num_points, amplitude=2.0, beta=2.0)
    angle_noise = fractal_noise_1d(num_points, amplitude=random.uniform(0.08, 0.15), beta=2.2)

    # Inner edge (closer to radar)
    for i in range(num_points):
        angle_deg = start + (end - start) * i / (num_points - 1) + angle_noise[i]
        angle_rad = math.radians(angle_deg)
        r = min_range_m + inner_noise[i]
        x = r * math.sin(angle_rad)
        y = r * math.cos(angle_rad)
        points.append((x, y))

    # Outer edge (farther from radar), reversed
    for i in range(num_points - 1, -1, -1):
        angle_deg = start + (end - start) * i / (num_points - 1) + angle_noise[i]
        angle_rad = math.radians(angle_deg)
        r = min_range_m + depth_profile[i] + outer_noise[i]
        x = r * math.sin(angle_rad)
        y = r * math.cos(angle_rad)
        points.append((x, y))

    points = _warp_coastline_organic(points, warp_amplitude=random.uniform(0.10, 0.20))
    points = _smooth_polygon_points(points)
    return CoastlineSegment(points=points, name="arc_coastline")


def generate_island_coastline(cx: float, cy: float, radius_m: float,
                               irregularity: float = 0.3,
                               num_points: int = None) -> CoastlineSegment:
    """Generate an irregular island polygon with fractal edges."""
    if num_points is None:
        num_points = max(40, int(2 * math.pi * radius_m / 1.0))
    radius_noise = fractal_noise_1d(num_points, amplitude=0.4, beta=1.3)
    angle_noise = fractal_noise_1d(num_points, amplitude=random.uniform(0.01, 0.03), beta=2.0)
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points + angle_noise[i]
        r = radius_m * (1 + radius_noise[i])
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        points.append((x, y))
    points.append(points[0])  # close polygon (FFT periodicity ensures smooth join)
    points = _warp_coastline_organic(points, warp_amplitude=random.uniform(0.10, 0.20))
    points = _smooth_polygon_points(points)
    return CoastlineSegment(points=points, name="island")


def generate_linear_coastline(start_bearing_deg: float, end_bearing_deg: float,
                               range_m: float, depth_m: float = None,
                               num_points: int = None) -> CoastlineSegment:
    """Generate a roughly linear shoreline at a given range with fractal edges."""
    if depth_m is None:
        depth_m = random.uniform(40, 100)
    span_deg = abs(end_bearing_deg - start_bearing_deg)
    if num_points is None:
        num_points = adaptive_point_count(span_deg, range_m)
    depth_profile = variable_depth_profile(num_points, depth_m,
                                           min_depth=depth_m * 0.5,
                                           max_depth=depth_m * 1.5)
    inner_noise = fractal_noise_1d(num_points, amplitude=random.uniform(3.0, 7.0), beta=2.0)
    outer_noise = fractal_noise_1d(num_points, amplitude=2.0, beta=2.0)
    angle_noise = fractal_noise_1d(num_points, amplitude=random.uniform(0.08, 0.15), beta=2.2)

    inner_pts = []
    outer_pts = []
    # Inner edge
    for i in range(num_points):
        frac = i / (num_points - 1)
        bearing_deg = start_bearing_deg + (end_bearing_deg - start_bearing_deg) * frac + angle_noise[i]
        bearing_rad = math.radians(bearing_deg)
        r = range_m + inner_noise[i]
        x = r * math.sin(bearing_rad)
        y = r * math.cos(bearing_rad)
        inner_pts.append((x, y))
    # Outer edge
    for i in range(num_points):
        frac = i / (num_points - 1)
        bearing_deg = start_bearing_deg + (end_bearing_deg - start_bearing_deg) * frac + angle_noise[i]
        bearing_rad = math.radians(bearing_deg)
        r = range_m + depth_profile[i] + outer_noise[i]
        x = r * math.sin(bearing_rad)
        y = r * math.cos(bearing_rad)
        outer_pts.append((x, y))

    # Build polygon: inner edge → end cap → outer edge (reversed) → start cap
    # End caps use 5 interpolated points to create curved connections
    # instead of straight "bow string" lines
    def _arc_cap(p1, p2, n_cap=5):
        """Interpolate n_cap points along a slight outward arc between p1 and p2."""
        cap = []
        cx = (p1[0] + p2[0]) / 2
        cy = (p1[1] + p2[1]) / 2
        # Bulge outward from radar center
        dist = math.sqrt(cx * cx + cy * cy)
        if dist > 1e-3:
            bulge = random.uniform(0.1, 0.3) * math.sqrt(
                (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            bx = cx + (cx / dist) * bulge
            by = cy + (cy / dist) * bulge
        else:
            bx, by = cx, cy
        for j in range(1, n_cap + 1):
            t = j / (n_cap + 1)
            # Quadratic bezier: P1 → bulge midpoint → P2
            x = (1 - t) ** 2 * p1[0] + 2 * (1 - t) * t * bx + t ** 2 * p2[0]
            y = (1 - t) ** 2 * p1[1] + 2 * (1 - t) * t * by + t ** 2 * p2[1]
            cap.append((x, y))
        return cap

    points = list(inner_pts)
    points.extend(_arc_cap(inner_pts[-1], outer_pts[-1]))
    points.extend(reversed(outer_pts))
    points.extend(_arc_cap(outer_pts[0], inner_pts[0]))

    points = _warp_coastline_organic(points, warp_amplitude=random.uniform(0.10, 0.20))
    points = _smooth_polygon_points(points)
    return CoastlineSegment(points=points, name="shoreline")


def generate_headland_coastline(center_bearing_deg: float, arc_span_deg: float,
                                 base_range_m: float,
                                 protrusion_m: float = 25.0) -> CoastlineSegment:
    """Generate a headland — protruding landmass that bulges toward the radar.

    Inner edge has a Gaussian bump profile (closer to radar at center) with
    fractal noise layered on top. Outer edge is smooth at base_range + depth.
    """
    num_points = adaptive_point_count(arc_span_deg, base_range_m)
    start = center_bearing_deg - arc_span_deg / 2
    end = center_bearing_deg + arc_span_deg / 2
    depth_m = random.uniform(50, 120)
    sigma = arc_span_deg * 0.3

    inner_noise = fractal_noise_1d(num_points, amplitude=random.uniform(3.0, 6.0), beta=2.0)
    outer_noise = fractal_noise_1d(num_points, amplitude=1.5, beta=2.0)
    angle_noise = fractal_noise_1d(num_points, amplitude=random.uniform(0.06, 0.12), beta=2.2)

    points = []
    # Inner edge — bulges toward radar at center (lower range = closer)
    for i in range(num_points):
        frac = i / (num_points - 1)
        angle_deg = start + (end - start) * frac + angle_noise[i]
        angle_rad = math.radians(angle_deg)
        # Gaussian bump: subtract protrusion at center
        offset_deg = angle_deg - center_bearing_deg
        bump = protrusion_m * math.exp(-0.5 * (offset_deg / sigma) ** 2)
        r = base_range_m - bump + inner_noise[i]
        x = r * math.sin(angle_rad)
        y = r * math.cos(angle_rad)
        points.append((x, y))
    # Outer edge, reversed — smooth at base_range + depth
    for i in range(num_points - 1, -1, -1):
        frac = i / (num_points - 1)
        angle_deg = start + (end - start) * frac + angle_noise[i]
        angle_rad = math.radians(angle_deg)
        r = base_range_m + depth_m + outer_noise[i]
        x = r * math.sin(angle_rad)
        y = r * math.cos(angle_rad)
        points.append((x, y))
    points = _warp_coastline_organic(points, warp_amplitude=random.uniform(0.10, 0.20))
    points = _smooth_polygon_points(points)
    return CoastlineSegment(points=points, name="headland")


def generate_bay_coastline(center_bearing_deg: float, arc_span_deg: float,
                            base_range_m: float, indent_m: float = 15.0,
                            is_harbor: bool = False) -> CoastlineSegment:
    """Generate a bay or harbor — concave indentation away from radar.

    Inner edge recedes from radar at center (inverse Gaussian).
    Harbor variant is narrower and deeper than a natural bay.
    Fractal noise is reduced inside the bay (calmer water reflections).
    """
    num_points = adaptive_point_count(arc_span_deg, base_range_m)
    start = center_bearing_deg - arc_span_deg / 2
    end = center_bearing_deg + arc_span_deg / 2
    depth_m = random.uniform(40, 90)

    if is_harbor:
        sigma = arc_span_deg * 0.15
        noise_amp = 2.0  # calmer, engineered edges
    else:
        sigma = arc_span_deg * 0.35
        noise_amp = 4.0

    inner_noise = fractal_noise_1d(num_points, amplitude=random.uniform(3.0, 6.0) if not is_harbor else noise_amp, beta=2.0)
    outer_noise = fractal_noise_1d(num_points, amplitude=1.5, beta=2.0)
    angle_noise = fractal_noise_1d(num_points, amplitude=random.uniform(0.06, 0.12), beta=2.2)

    points = []
    # Inner edge — recedes from radar at center (higher range = farther)
    for i in range(num_points):
        frac = i / (num_points - 1)
        angle_deg = start + (end - start) * frac + angle_noise[i]
        angle_rad = math.radians(angle_deg)
        offset_deg = angle_deg - center_bearing_deg
        indent = indent_m * math.exp(-0.5 * (offset_deg / sigma) ** 2)
        # Reduce noise amplitude inside the bay
        noise_scale = 1.0 - 0.5 * math.exp(-0.5 * (offset_deg / sigma) ** 2)
        r = base_range_m + indent + inner_noise[i] * noise_scale
        x = r * math.sin(angle_rad)
        y = r * math.cos(angle_rad)
        points.append((x, y))
    # Outer edge, reversed
    for i in range(num_points - 1, -1, -1):
        frac = i / (num_points - 1)
        angle_deg = start + (end - start) * frac + angle_noise[i]
        angle_rad = math.radians(angle_deg)
        r = base_range_m + depth_m + indent_m + outer_noise[i]
        x = r * math.sin(angle_rad)
        y = r * math.cos(angle_rad)
        points.append((x, y))
    points = _warp_coastline_organic(points, warp_amplitude=random.uniform(0.10, 0.20))
    points = _smooth_polygon_points(points)
    name = "harbor" if is_harbor else "bay"
    return CoastlineSegment(points=points, name=name)


def generate_breakwater_coastline(start_bearing_deg: float, start_range_m: float,
                                    length_m: float = 50.0,
                                    orientation_deg: float = 0.0,
                                    width_m: float = 3.5) -> CoastlineSegment:
    """Generate a breakwater/seawall — thin linear engineered structure.

    2-5m wide, 20-80m long. Very mild fractal noise (engineered structure).
    Built as left_side + reversed(right_side) offset from centerline.
    """
    num_points = max(40, int(length_m / 0.5))
    noise = fractal_noise_1d(num_points, amplitude=random.uniform(0.5, 1.5), beta=2.0)

    # Centerline in Cartesian: start point + direction
    start_rad = math.radians(start_bearing_deg)
    sx = start_range_m * math.sin(start_rad)
    sy = start_range_m * math.cos(start_rad)

    orient_rad = math.radians(orientation_deg)
    dir_x = math.sin(orient_rad)
    dir_y = math.cos(orient_rad)
    # Perpendicular for width offset
    perp_x = math.cos(orient_rad)
    perp_y = -math.sin(orient_rad)

    half_w = width_m / 2.0
    left_pts = []
    right_pts = []
    for i in range(num_points):
        frac = i / (num_points - 1)
        cx = sx + dir_x * length_m * frac + perp_x * noise[i] * 0.3
        cy = sy + dir_y * length_m * frac + perp_y * noise[i] * 0.3
        # Width varies slightly with noise
        w = half_w + noise[i] * 0.1
        left_pts.append((cx + perp_x * w, cy + perp_y * w))
        right_pts.append((cx - perp_x * w, cy - perp_y * w))

    points = left_pts + list(reversed(right_pts))
    points = _smooth_polygon_points(points, window=5)
    return CoastlineSegment(points=points, name="breakwater")


def generate_jetty_coastline(base_bearing_deg: float, base_range_m: float,
                               extension_m: float = 25.0,
                               width_m: float = 3.0) -> CoastlineSegment:
    """Generate a jetty — thin feature extending radially from shore toward radar.

    2-4m wide, 10-40m long. Very low noise (engineered structure).
    Built as thin strip along the bearing direction toward radar.
    """
    num_points = max(40, int(extension_m / 0.5))
    noise = fractal_noise_1d(num_points, amplitude=random.uniform(0.3, 1.0), beta=2.0)

    bearing_rad = math.radians(base_bearing_deg)
    dir_x = math.sin(bearing_rad)
    dir_y = math.cos(bearing_rad)
    # Perpendicular for width
    perp_x = math.cos(bearing_rad)
    perp_y = -math.sin(bearing_rad)

    half_w = width_m / 2.0
    left_pts = []
    right_pts = []
    for i in range(num_points):
        frac = i / (num_points - 1)
        # Extend from base_range toward radar (decreasing range)
        r = base_range_m - extension_m * frac
        cx = r * dir_x + perp_x * noise[i] * 0.2
        cy = r * dir_y + perp_y * noise[i] * 0.2
        w = half_w + noise[i] * 0.05
        left_pts.append((cx + perp_x * w, cy + perp_y * w))
        right_pts.append((cx - perp_x * w, cy - perp_y * w))

    points = left_pts + list(reversed(right_pts))
    points = _smooth_polygon_points(points, window=5)
    return CoastlineSegment(points=points, name="jetty")


def generate_composite_coastlines(max_range_m: float) -> List[CoastlineSegment]:
    """Generate spatially coherent multi-feature coastline scenes.

    Picks an anchor bearing/range, then places related features relative to it.
    Six templates (weighted random):
      - Harbor scene:       Bay + 1-3 jetties + breakwater at entrance
      - Headland scene:     Headland + flanking bays + 0-2 offshore rocks
      - Marina scene:       Narrow harbor + 2-5 parallel jetties
      - River mouth:        Two converging linear coastlines + optional sandbar
      - Island group:       2-5 clustered islands (1 large + smaller)
      - Continental shelf:  Large arc (90-180°) + 1-2 overlaid features
    """
    coastlines = []
    anchor_bearing = random.uniform(0, 360)
    anchor_range = random.uniform(max_range_m * 0.25, max_range_m * 0.65)

    template = random.choices(
        ["harbor", "headland", "marina", "river_mouth", "island_group",
         "continental_shelf", "dense_shoreline", "dam_scene"],
        weights=[15, 15, 12, 12, 12, 12, 12, 10],
        k=1,
    )[0]

    if template == "harbor":
        # Bay + 1-3 jetties + breakwater at entrance
        bay_span = random.uniform(20, 45)
        bay_indent = random.uniform(15, 40)
        coastlines.append(generate_bay_coastline(
            anchor_bearing, bay_span, anchor_range, bay_indent, is_harbor=True))

        num_jetties = random.randint(1, 3)
        for j in range(num_jetties):
            offset = (j - (num_jetties - 1) / 2) * random.uniform(3, 8)
            jetty_bearing = anchor_bearing + offset
            jetty_range = anchor_range + random.uniform(-5, 10)
            extension = random.uniform(10, 30)
            width = random.uniform(2, 4)
            coastlines.append(generate_jetty_coastline(
                jetty_bearing, jetty_range, extension, width))

        # Breakwater at harbor entrance
        bw_bearing = anchor_bearing + random.uniform(-bay_span * 0.4, bay_span * 0.4)
        bw_range = anchor_range - bay_indent * 0.6 + random.uniform(-5, 5)
        bw_length = random.uniform(20, 50)
        bw_orient = anchor_bearing + 90 + random.uniform(-30, 30)
        coastlines.append(generate_breakwater_coastline(
            bw_bearing, bw_range, bw_length, bw_orient, random.uniform(2, 4)))

    elif template == "headland":
        # Headland + flanking bays + 0-2 offshore rocks
        hl_span = random.uniform(30, 70)
        protrusion = random.uniform(15, 40)
        coastlines.append(generate_headland_coastline(
            anchor_bearing, hl_span, anchor_range, protrusion))

        # Flanking bays on either side
        for side in [-1, 1]:
            bay_bearing = anchor_bearing + side * (hl_span * 0.5 + random.uniform(10, 25))
            bay_span = random.uniform(15, 35)
            bay_range = anchor_range + random.uniform(0, 15)
            bay_indent = random.uniform(8, 20)
            coastlines.append(generate_bay_coastline(
                bay_bearing, bay_span, bay_range, bay_indent, is_harbor=False))

        # Optional offshore rocks
        num_rocks = random.randint(0, 2)
        for _ in range(num_rocks):
            rock_bearing_rad = math.radians(anchor_bearing + random.uniform(-20, 20))
            rock_dist = anchor_range - protrusion - random.uniform(15, 40)
            if rock_dist < max_range_m * 0.1:
                continue
            rx = rock_dist * math.sin(rock_bearing_rad)
            ry = rock_dist * math.cos(rock_bearing_rad)
            rock_radius = random.uniform(2, 8)
            coastlines.append(generate_island_coastline(rx, ry, rock_radius,
                                                         irregularity=0.4))

    elif template == "marina":
        # Narrow harbor + 2-5 parallel jetties
        harbor_span = random.uniform(12, 25)
        harbor_indent = random.uniform(20, 45)
        coastlines.append(generate_bay_coastline(
            anchor_bearing, harbor_span, anchor_range, harbor_indent, is_harbor=True))

        num_jetties = random.randint(2, 5)
        jetty_spacing = harbor_span / (num_jetties + 1)
        for j in range(num_jetties):
            jb = anchor_bearing - harbor_span / 2 + jetty_spacing * (j + 1)
            j_range = anchor_range + random.uniform(-3, 8)
            extension = random.uniform(12, 25)
            width = random.uniform(1.5, 3)
            coastlines.append(generate_jetty_coastline(jb, j_range, extension, width))

    elif template == "river_mouth":
        # Two converging linear coastlines + optional sandbar island
        mouth_width_deg = random.uniform(8, 20)
        coast_span = random.uniform(25, 60)
        depth = random.uniform(15, 35)

        # Left bank
        left_start = anchor_bearing - mouth_width_deg / 2 - coast_span
        left_end = anchor_bearing - mouth_width_deg / 2
        coastlines.append(generate_linear_coastline(
            left_start, left_end, anchor_range, depth))

        # Right bank
        right_start = anchor_bearing + mouth_width_deg / 2
        right_end = anchor_bearing + mouth_width_deg / 2 + coast_span
        coastlines.append(generate_linear_coastline(
            right_start, right_end, anchor_range, depth))

        # Optional sandbar island in the mouth
        if random.random() < 0.5:
            sb_bearing_rad = math.radians(anchor_bearing)
            sb_dist = anchor_range - random.uniform(10, 25)
            if sb_dist > max_range_m * 0.1:
                sx = sb_dist * math.sin(sb_bearing_rad)
                sy = sb_dist * math.cos(sb_bearing_rad)
                sb_radius = random.uniform(3, 10)
                coastlines.append(generate_island_coastline(
                    sx, sy, sb_radius, irregularity=0.2))

    elif template == "island_group":
        # 2-5 clustered islands (1 large + smaller)
        num_islands = random.randint(2, 5)
        # Large main island
        main_bearing_rad = math.radians(anchor_bearing)
        mx = anchor_range * math.sin(main_bearing_rad)
        my = anchor_range * math.cos(main_bearing_rad)
        main_radius = random.uniform(15, 35)
        coastlines.append(generate_island_coastline(mx, my, main_radius,
                                                     irregularity=0.3))

        # Smaller satellite islands
        for _ in range(num_islands - 1):
            offset_angle = random.uniform(0, 360)
            offset_dist = main_radius + random.uniform(10, 40)
            offset_rad = math.radians(offset_angle)
            ix = mx + offset_dist * math.cos(offset_rad)
            iy = my + offset_dist * math.sin(offset_rad)
            if math.sqrt(ix * ix + iy * iy) > max_range_m * 0.95:
                continue
            i_radius = random.uniform(3, main_radius * 0.6)
            coastlines.append(generate_island_coastline(ix, iy, i_radius,
                                                         irregularity=0.35))

    elif template == "continental_shelf":
        # Large arc (90-210°) + 1-3 overlaid features (jetty/breakwater)
        arc_span = random.uniform(90, 210)
        min_r = anchor_range
        max_r = min(min_r + random.uniform(20, 60), max_range_m * 0.98)
        coastlines.append(generate_arc_coastline(anchor_bearing, arc_span, min_r, max_r))

        # Overlaid features
        num_overlaid = random.randint(1, 2)
        for _ in range(num_overlaid):
            feat_bearing = anchor_bearing + random.uniform(-arc_span * 0.3, arc_span * 0.3)
            if random.random() < 0.5:
                # Jetty extending from the coastline
                ext = random.uniform(10, 30)
                w = random.uniform(2, 4)
                coastlines.append(generate_jetty_coastline(
                    feat_bearing, min_r, ext, w))
            else:
                # Breakwater in front of the coastline
                bw_range = min_r - random.uniform(10, 30)
                if bw_range > max_range_m * 0.1:
                    bw_len = random.uniform(20, 60)
                    bw_orient = feat_bearing + random.uniform(60, 120)
                    coastlines.append(generate_breakwater_coastline(
                        feat_bearing, bw_range, bw_len, bw_orient, random.uniform(2, 5)))

    elif template == "dense_shoreline":
        # Dense multi-layer shoreline (modeled on Lake Murray captures)
        # 3-5 large arcs at different ranges covering 200-360° total
        # Plus many small fill segments
        num_major_arcs = random.randint(3, 5)
        total_target_span = random.uniform(200, 350)
        current_bearing = anchor_bearing

        for i in range(num_major_arcs):
            arc_span = total_target_span / num_major_arcs + random.uniform(-20, 20)
            arc_span = max(30, min(arc_span, 150))
            # Multi-scale: alternate near and mid range
            if i % 2 == 0:
                min_r = random.uniform(max_range_m * 0.03, max_range_m * 0.15)
            else:
                min_r = random.uniform(max_range_m * 0.15, max_range_m * 0.45)
            max_r = min_r + random.uniform(max_range_m * 0.05, max_range_m * 0.3)
            max_r = min(max_r, max_range_m * 0.98)
            coastlines.append(generate_arc_coastline(current_bearing, arc_span, min_r, max_r))
            current_bearing += arc_span + random.uniform(-5, 10)

        # Add 2-3 headlands/bays at mid range
        for _ in range(random.randint(2, 3)):
            bearing = random.uniform(0, 360)
            if random.random() < 0.5:
                span = random.uniform(20, 60)
                base_range = random.uniform(max_range_m * 0.2, max_range_m * 0.5)
                protrusion = random.uniform(10, 35)
                coastlines.append(generate_headland_coastline(
                    bearing, span, base_range, protrusion))
            else:
                span = random.uniform(15, 45)
                base_range = random.uniform(max_range_m * 0.2, max_range_m * 0.6)
                indent = random.uniform(8, 25)
                coastlines.append(generate_bay_coastline(
                    bearing, span, base_range, indent, is_harbor=False))

        # Add 8-15 small fill segments clustered near existing features
        num_fill = random.randint(8, 15)
        coastlines.extend(_generate_fill_segments(max_range_m, num_fill,
                                                   anchor_coastlines=coastlines))

    elif template == "dam_scene":
        # Dam/seawall scene (modeled on Lake Murray Dam captures)
        # Very close large arc (dam wall) + distant shoreline fragments
        dam_bearing = anchor_bearing
        dam_span = random.uniform(100, 200)
        dam_min_r = random.uniform(max_range_m * 0.03, max_range_m * 0.08)
        dam_max_r = dam_min_r + random.uniform(max_range_m * 0.02, max_range_m * 0.08)
        coastlines.append(generate_arc_coastline(dam_bearing, dam_span, dam_min_r, dam_max_r))

        # Secondary near-field feature (adjacent wall section)
        adj_bearing = dam_bearing + dam_span / 2 + random.uniform(5, 20)
        adj_span = random.uniform(40, 100)
        adj_min_r = random.uniform(max_range_m * 0.03, max_range_m * 0.12)
        adj_max_r = adj_min_r + random.uniform(max_range_m * 0.05, max_range_m * 0.35)
        coastlines.append(generate_arc_coastline(adj_bearing, adj_span, adj_min_r, adj_max_r))

        # One large mid-range arc (opposite shore)
        opp_bearing = dam_bearing + random.uniform(150, 210)
        opp_span = random.uniform(60, 120)
        opp_min_r = random.uniform(max_range_m * 0.15, max_range_m * 0.45)
        opp_max_r = opp_min_r + random.uniform(max_range_m * 0.1, max_range_m * 0.3)
        opp_max_r = min(opp_max_r, max_range_m * 0.98)
        coastlines.append(generate_arc_coastline(opp_bearing, opp_span, opp_min_r, opp_max_r))

        # 1-2 headlands at mid range
        for _ in range(random.randint(1, 2)):
            bearing = random.uniform(0, 360)
            span = random.uniform(25, 60)
            base_range = random.uniform(max_range_m * 0.4, max_range_m * 0.7)
            protrusion = random.uniform(15, 40)
            coastlines.append(generate_headland_coastline(
                bearing, span, base_range, protrusion))

        # Many fill segments at far range clustered near features
        num_fill = random.randint(10, 18)
        coastlines.extend(_generate_fill_segments(max_range_m, num_fill,
                                                   anchor_coastlines=coastlines))

    # Add fill segments to all templates for realistic feature density
    # (dense_shoreline and dam_scene already added their own fill above)
    if template not in ("dense_shoreline", "dam_scene"):
        num_fill = random.randint(3, 10)
        coastlines.extend(_generate_fill_segments(max_range_m, num_fill,
                                                   anchor_coastlines=coastlines))

    # Filter out unnatural shapes and heavy overlaps
    coastlines = [c for c in coastlines if _coastline_looks_natural(c.points)]
    coastlines = _filter_overlapping_coastlines(coastlines)

    return coastlines


# ============================================================
# Coastline Primitive Library
# ============================================================
def _polygon_area(points):
    """Compute area of polygon using shoelace formula."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def _convex_hull_area(points):
    """Compute convex hull area using Andrew's monotone chain algorithm."""
    pts = sorted(set((round(p[0], 6), round(p[1], 6)) for p in points))
    if len(pts) < 3:
        return _polygon_area(points)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return _polygon_area(hull)


def _coastline_looks_natural(pts):
    """Check whether a coastline polygon looks natural based on shape metrics.

    Rejects shapes that are:
      - Too thin/elongated (high aspect ratio + low fill ratio)
      - Over-dense (too many points for their size → noise-like)
      - Dominated by long perfectly-straight segments
      - Mid-size wispy shapes that look like noise on PPI
      - Rectangular artifacts
    Returns True if the coastline passes all checks.
    """
    n = len(pts)
    if n < 3:
        return False

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    max_span = max(span_x, span_y)
    min_span = min(span_x, span_y)

    # Very tiny coastlines (< 1m) are noise
    if max_span < 1.0:
        return False

    aspect = max_span / min_span if min_span > 0.01 else 999.0

    # Shoelace area and fill ratio
    area = _polygon_area(pts)
    bbox_area = span_x * span_y
    fill_ratio = area / bbox_area if bbox_area > 0.01 else 0.0

    # --- Rule 1: Elongated shapes ---
    # Small/mid features with high aspect look like streak artifacts.
    # Large features (arcs, shorelines) are naturally elongated — allow higher aspect.
    if max_span < 25.0:
        if aspect > 4.5 and fill_ratio < 0.75:
            return False
    else:
        # Large features: only reject extreme elongation
        if aspect > 12.0 and fill_ratio < 0.10:
            return False

    # --- Rule 2: Over-dense point count ---
    # Real coastlines have ~10-30 pts per meter of span; artifacts have 50+
    if max_span > 2.0:
        pts_per_meter = n / max_span
        if pts_per_meter > 40 and fill_ratio < 0.35:
            return False

    # --- Rule 3: Rectangular shapes ---
    # High fill ratio + few points = rectangle artifact
    if fill_ratio > 0.85 and n <= 12:
        return False
    if fill_ratio > 0.92 and n <= 20:
        return False

    # --- Rule 4: Mid-size thin wispy shapes ---
    # Shapes 8-25m span with very low fill AND moderate elongation
    # look like noise/artifacts on PPI, not natural coastlines
    if 8.0 < max_span < 25.0:
        if aspect > 3.0 and fill_ratio < 0.50:
            return False

    # --- Rule 5: Straightness (single segment dominates perimeter) ---
    seg_lengths = [
        math.sqrt((pts[(i+1) % n][0] - pts[i][0])**2 +
                  (pts[(i+1) % n][1] - pts[i][1])**2)
        for i in range(n)]
    perimeter = sum(seg_lengths)
    if perimeter > 0 and max_span > 5.0:
        max_seg = max(seg_lengths)
        # If one segment > 40% of perimeter → "bow string" straight line artifact
        if max_seg / perimeter > 0.40:
            return False
        # Also check top 2 segments (both closing edges of a bow shape)
        sorted_segs = sorted(seg_lengths, reverse=True)
        if len(sorted_segs) >= 2 and (sorted_segs[0] + sorted_segs[1]) / perimeter > 0.50:
            return False

    # --- Rule 6: Near-circular shapes (ring artifacts) ---
    # Isoperimetric quotient: perfect circle = 1.0, natural coastlines < 0.92
    if perimeter > 0 and max_span > 8.0:
        iq = 4 * math.pi * area / (perimeter * perimeter)
        if iq > 0.92 and aspect < 1.3:
            return False

    # --- Rule 7: Angular/geometric shapes (triangles, diamonds) ---
    # Only catch extreme cases: very few points with many sharp vertices
    if 6 <= n < 25 and max_span > 8.0:
        sharp_corners = 0
        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
            dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]
            len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
            len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
            if len1 > 0.01 and len2 > 0.01:
                cos_a = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
                cos_a = max(-1.0, min(1.0, cos_a))
                angle = math.degrees(math.acos(cos_a))
                turn = 180.0 - angle
                if turn > 45:
                    sharp_corners += 1
        # Shapes with >20% sharp corners AND few points = geometric artifact
        if sharp_corners >= 3 and sharp_corners / n > 0.20:
            return False

    return True


def _filter_overlapping_coastlines(coastlines):
    """Remove coastlines that are nearly fully contained within a larger one.

    Uses a high threshold (90% bbox containment AND similar size ratio < 3x)
    to avoid removing intentional companion features in clusters.
    Only removes the smaller shape when it's essentially a duplicate
    sitting on top of a larger feature, not a distinct nearby feature.
    """
    if len(coastlines) <= 1:
        return coastlines

    # Precompute bounding boxes and areas
    info = []
    for c in coastlines:
        xs = [p[0] for p in c.points]
        ys = [p[1] for p in c.points]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        info.append((bbox, max(bbox_area, 0.001)))

    remove = set()
    for i in range(len(coastlines)):
        if i in remove:
            continue
        for j in range(i + 1, len(coastlines)):
            if j in remove:
                continue
            # Determine smaller/larger by bbox area
            if info[i][1] >= info[j][1]:
                big, small, si = i, j, j
            else:
                big, small, si = j, i, i

            bx = info[big][0]
            sx = info[small][0]

            # Intersection of bounding boxes
            ix0 = max(bx[0], sx[0])
            iy0 = max(bx[1], sx[1])
            ix1 = min(bx[2], sx[2])
            iy1 = min(bx[3], sx[3])
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            inter_area = (ix1 - ix0) * (iy1 - iy0)
            small_area = info[small][1]

            # Only remove if nearly fully contained (>90%) AND similar size
            # (size ratio < 3x means it's a near-duplicate, not a small detail)
            size_ratio = info[big][1] / small_area if small_area > 0 else 999
            if small_area > 0 and inter_area / small_area > 0.90 and size_ratio < 3.0:
                remove.add(si)

    return [c for i, c in enumerate(coastlines) if i not in remove]


def _bearing_span_and_start(bearings):
    """Compute angular span and start bearing, handling wraparound."""
    if len(bearings) < 2:
        return 0.0, bearings[0] if bearings else 0.0
    sorted_b = sorted(bearings)
    max_gap = 0.0
    gap_end_idx = 0
    for i in range(len(sorted_b) - 1):
        gap = sorted_b[i + 1] - sorted_b[i]
        if gap > max_gap:
            max_gap = gap
            gap_end_idx = i + 1
    wrap_gap = 360.0 - sorted_b[-1] + sorted_b[0]
    if wrap_gap > max_gap:
        max_gap = wrap_gap
        gap_end_idx = 0
    bearing_span = 360.0 - max_gap
    start_bearing = sorted_b[gap_end_idx]
    return bearing_span, start_bearing


def _fix_ring_and_spoke_arrangements(coastlines, max_range_m):
    """Detect and fix ring-like or spoke-like arrangements by nudging positions.

    Ring: all features at similar range spanning wide bearing → nudge ranges apart.
    Spoke: multiple features at same bearing at different ranges → nudge bearings apart.
    Returns corrected coastlines (modifies in place).
    """
    if len(coastlines) < 5:
        return coastlines

    # Compute centroid polar coords for each coastline
    centroids = []
    for c in coastlines:
        cx = sum(p[0] for p in c.points) / len(c.points)
        cy = sum(p[1] for p in c.points) / len(c.points)
        brg = math.degrees(math.atan2(cx, cy)) % 360
        rng = math.sqrt(cx * cx + cy * cy)
        centroids.append((brg, rng))

    ranges = [r for _, r in centroids]
    bearings = [b for b, _ in centroids]
    mean_range = sum(ranges) / len(ranges)
    range_std = (sum((r - mean_range) ** 2 for r in ranges) / len(ranges)) ** 0.5
    range_cv = range_std / mean_range if mean_range > 0 else 0

    bearing_span, _ = _bearing_span_and_start(bearings)

    # Ring detection: low range variation + wide bearing coverage
    if range_cv < 0.15 and bearing_span > 270 and len(coastlines) >= 5:
        # Nudge 30% of features' ranges to break the ring
        indices = list(range(len(coastlines)))
        random.shuffle(indices)
        n_nudge = max(2, len(coastlines) // 3)
        for idx in indices[:n_nudge]:
            nudge_m = random.uniform(-25, 25)
            pts = coastlines[idx].points
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            r_orig = math.sqrt(cx * cx + cy * cy)
            if r_orig > 1e-3:
                scale = max(0.3, (r_orig + nudge_m) / r_orig)
                coastlines[idx].points = [(x * scale, y * scale) for x, y in pts]

    # Spoke detection: 3+ features within 4° bearing at different ranges
    bearings_sorted = sorted(enumerate(bearings), key=lambda x: x[1])
    spoke_groups = []
    current_group = [bearings_sorted[0]]
    for i in range(1, len(bearings_sorted)):
        if abs(bearings_sorted[i][1] - current_group[-1][1]) < 4.0:
            current_group.append(bearings_sorted[i])
        else:
            if len(current_group) >= 3:
                spoke_groups.append(current_group)
            current_group = [bearings_sorted[i]]
    if len(current_group) >= 3:
        spoke_groups.append(current_group)

    for group in spoke_groups:
        # Nudge all but the first feature in the spoke by 8-15° each direction
        for k, (idx, _) in enumerate(group[1:], 1):
            nudge_deg = random.uniform(8, 15) * (1 if k % 2 == 0 else -1)
            nudge_rad = math.radians(nudge_deg)
            cos_n, sin_n = math.cos(nudge_rad), math.sin(nudge_rad)
            pts = coastlines[idx].points
            coastlines[idx].points = [
                (x * cos_n - y * sin_n, x * sin_n + y * cos_n)
                for x, y in pts]

    return coastlines


def classify_coastline(points, coastline_stats=None):
    """Classify coastline shape using geometric decision tree.

    Returns (classification, metrics_dict) where classification is one of:
    'arc', 'headland', 'bay', 'thin_linear', 'island', 'large_mass'.
    """
    if len(points) < 3:
        return "arc", {"bearing_span_deg": 0, "range_span_m": 0,
                       "aspect_ratio": 1.0, "convexity": 1.0}

    # Convert to polar
    polar = []
    for p in points:
        x, y = p[0], p[1]
        r = math.sqrt(x * x + y * y)
        b = math.degrees(math.atan2(x, y)) % 360
        polar.append((b, r))

    bearings = [p[0] for p in polar]
    ranges = [p[1] for p in polar]

    bearing_span, start_bearing = _bearing_span_and_start(bearings)
    min_range = min(ranges)
    max_range = max(ranges)
    range_span = max_range - min_range

    # Centroid
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    # Aspect ratio: bearing arc length / range span
    mean_range = (min_range + max_range) / 2.0
    arc_length = math.radians(bearing_span) * mean_range if mean_range > 0 else 0
    aspect_ratio = arc_length / range_span if range_span > 0.1 else 10.0

    # Convexity
    poly_area = _polygon_area(points)
    hull_area = _convex_hull_area(points)
    convexity = poly_area / hull_area if hull_area > 0 else 1.0
    convexity = min(1.0, convexity)

    # Check if closed (first pt ~ last pt within 2m)
    d_close = math.sqrt((points[0][0] - points[-1][0])**2 +
                        (points[0][1] - points[-1][1])**2)
    is_closed = d_close < 2.0

    # Compute total length of the coastline
    total_length = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        total_length += math.sqrt(dx * dx + dy * dy)

    metrics = {
        "bearing_span_deg": bearing_span,
        "range_span_m": range_span,
        "min_range_m": min_range,
        "center_bearing_deg": (start_bearing + bearing_span / 2) % 360,
        "aspect_ratio": aspect_ratio,
        "convexity": convexity,
        "is_closed": is_closed,
        "total_length_m": total_length,
        "num_points": len(points),
    }

    # Decision tree
    if bearing_span > 150 and range_span > 20:
        return "large_mass", metrics
    if range_span < 8 and total_length > 20:
        return "thin_linear", metrics
    if is_closed and bearing_span > 300:
        return "island", metrics

    # Headland vs bay: check if center third is closer/farther than edges
    bearing_offsets = [((b - start_bearing) % 360, r) for b, r in polar]
    bearing_offsets.sort(key=lambda x: x[0])
    n = len(bearing_offsets)
    third = max(1, n // 3)
    edge_ranges = ([r for _, r in bearing_offsets[:third]] +
                   [r for _, r in bearing_offsets[-third:]])
    center_ranges = [r for _, r in bearing_offsets[third:n - third]]

    if center_ranges and edge_ranges:
        center_avg = sum(center_ranges) / len(center_ranges)
        edge_avg = sum(edge_ranges) / len(edge_ranges)
        if center_avg < edge_avg - 3:  # center closer to radar
            return "headland", metrics
        if center_avg > edge_avg + 3:  # center farther from radar
            return "bay", metrics

    return "arc", metrics


def normalize_coastline_shape(points):
    """Convert (x,y) polygon to normalized [0,1] x [0,1] polar space.

    Returns (normalized_points, bearing_span, range_span, min_range, center_bearing).
    """
    polar = []
    for p in points:
        x, y = p[0], p[1]
        r = math.sqrt(x * x + y * y)
        b = math.degrees(math.atan2(x, y)) % 360
        polar.append((b, r))

    bearings = [p[0] for p in polar]
    ranges = [p[1] for p in polar]

    bearing_span, start_bearing = _bearing_span_and_start(bearings)
    min_range = min(ranges)
    max_range = max(ranges)
    range_span = max_range - min_range

    if bearing_span < 0.01:
        bearing_span = 1.0
    if range_span < 0.01:
        range_span = 1.0

    center_bearing = (start_bearing + bearing_span / 2) % 360

    normalized = []
    for b, r in polar:
        b_offset = (b - start_bearing) % 360
        b_norm = max(0.0, min(1.0, b_offset / bearing_span))
        r_norm = max(0.0, min(1.0, (r - min_range) / range_span))
        normalized.append([b_norm, r_norm])

    return normalized, bearing_span, range_span, min_range, center_bearing


def compute_fourier_descriptors(points, n=8):
    """Compute first n normalized Fourier descriptors of the shape.

    Uses centroid-distance function FFT. Returns first n normalized
    harmonic magnitudes for shape similarity matching.
    """
    if len(points) < 4:
        return [0.0] * n

    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    distances = [math.sqrt((p[0] - cx)**2 + (p[1] - cy)**2) for p in points]

    fft = np.fft.fft(distances)
    magnitudes = np.abs(fft)

    if magnitudes[0] > 0:
        magnitudes = magnitudes / magnitudes[0]

    descriptors = magnitudes[1:n + 1].tolist()
    while len(descriptors) < n:
        descriptors.append(0.0)
    return descriptors


def build_primitive_from_coastline(points, stats, pattern_name, prim_id):
    """Build a CoastlinePrimitive from raw coastline data.

    Pipeline: classify -> normalize -> compute FDs -> assemble primitive.
    """
    if len(points) < 4:
        return None

    pts = [(p[0], p[1]) for p in points]

    classification, metrics = classify_coastline(pts, stats)
    normalized, bearing_span, range_span, min_range, center_bearing = \
        normalize_coastline_shape(pts)
    fds = compute_fourier_descriptors(pts)

    return CoastlinePrimitive(
        primitive_id=prim_id,
        classification=classification,
        source_pattern=pattern_name,
        normalized_points=normalized,
        original_bearing_span_deg=bearing_span,
        original_range_span_m=range_span,
        original_min_range_m=min_range,
        original_center_bearing_deg=center_bearing,
        original_num_points=len(pts),
        aspect_ratio=metrics["aspect_ratio"],
        convexity=metrics["convexity"],
        fourier_descriptors=fds,
        echo_stats=stats if stats else {},
    )


def build_coastline_library(scene_dir=None, cache_path=None, verbose=False):
    """Auto-discover pattern*_scene.json files and build a library of primitives.

    Caches to coastline_library_cache.json. On subsequent runs, loads from
    cache if scene files haven't changed (mod-time check).
    """
    if scene_dir is None:
        scene_dir = os.path.dirname(os.path.abspath(__file__))
    if cache_path is None:
        _ensure_modec_cache_dirs()
        cache_path = os.path.join(LIBRARY_CACHE_DIR, "coastline_library_cache.json")

    # Find all scene files
    scene_files = _find_scene_template_files(scene_dir)

    if not scene_files:
        # Auto-build: try extracting scene JSONs from source CSVs
        build_script = os.path.join(scene_dir, "build_library.py")
        if os.path.isfile(build_script):
            if verbose:
                print("  No scene JSONs found — auto-extracting from source CSVs...")
            try:
                import subprocess
                subprocess.run(
                    [sys.executable, build_script, "--extract"],
                    cwd=scene_dir, timeout=300)
                scene_files = _find_scene_template_files(scene_dir)
            except Exception as e:
                if verbose:
                    print(f"  Auto-extract failed: {e}")
        if not scene_files:
            if verbose:
                print("  Coastline library: no pattern*_scene.json files found")
            return CoastlineLibrary()

    # Check cache validity
    current_mtimes = {}
    for sf in scene_files:
        current_mtimes[os.path.basename(sf)] = os.path.getmtime(sf)

    if os.path.isfile(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            cached_mtimes = cache.get("scene_mtimes", {})
            if cached_mtimes == current_mtimes:
                library = CoastlineLibrary()
                for pd in cache.get("primitives", []):
                    prim = CoastlinePrimitive(
                        primitive_id=pd["primitive_id"],
                        classification=pd["classification"],
                        source_pattern=pd["source_pattern"],
                        normalized_points=pd["normalized_points"],
                        original_bearing_span_deg=pd["original_bearing_span_deg"],
                        original_range_span_m=pd["original_range_span_m"],
                        original_min_range_m=pd["original_min_range_m"],
                        original_center_bearing_deg=pd["original_center_bearing_deg"],
                        original_num_points=pd["original_num_points"],
                        aspect_ratio=pd["aspect_ratio"],
                        convexity=pd["convexity"],
                        fourier_descriptors=pd["fourier_descriptors"],
                        echo_stats=pd.get("echo_stats", {}),
                    )
                    library.add(prim)
                if verbose:
                    print(f"  Coastline library: {len(library.primitives)} primitives (cached)")
                    for cls, prims in sorted(library.by_classification.items()):
                        print(f"    {cls}: {len(prims)}")
                return library
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Build from scene files
    library = CoastlineLibrary()
    for scene_path in scene_files:
        pattern_name = os.path.basename(scene_path).replace("_scene.json", "")
        try:
            with open(scene_path, 'r') as f:
                scene_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        coastlines_data = scene_data.get("coastlines", [])
        stats_dict = scene_data.get("coastline_stats", {})

        for coast in coastlines_data:
            name = coast.get("name", "unknown")
            pts = coast.get("points", [])
            if len(pts) < 4:
                continue
            stats = stats_dict.get(name, {})
            prim_id = f"{pattern_name}_{name}"
            prim = build_primitive_from_coastline(pts, stats, pattern_name, prim_id)
            if prim:
                library.add(prim)

    # Save cache
    cache_data = {
        "scene_mtimes": current_mtimes,
        "primitives": [
            {
                "primitive_id": p.primitive_id,
                "classification": p.classification,
                "source_pattern": p.source_pattern,
                "normalized_points": p.normalized_points,
                "original_bearing_span_deg": p.original_bearing_span_deg,
                "original_range_span_m": p.original_range_span_m,
                "original_min_range_m": p.original_min_range_m,
                "original_center_bearing_deg": p.original_center_bearing_deg,
                "original_num_points": p.original_num_points,
                "aspect_ratio": p.aspect_ratio,
                "convexity": p.convexity,
                "fourier_descriptors": p.fourier_descriptors,
                "echo_stats": p.echo_stats,
            }
            for p in library.primitives
        ],
    }
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
    except IOError:
        pass

    if verbose:
        print(f"  Coastline library: {len(library.primitives)} primitives "
              f"from {len(scene_files)} scenes")
        for cls, prims in sorted(library.by_classification.items()):
            print(f"    {cls}: {len(prims)}")

    return library


def _discover_scene_template_files(scene_dir=None):
    """Find all pattern*_scene.json files for scene-template generation.

    If none are found, attempts auto-build via build_library.py --extract.
    """
    if scene_dir is None:
        scene_dir = os.path.dirname(os.path.abspath(__file__))
    files = _find_scene_template_files(scene_dir)

    if not files:
        # Auto-build: extract scene JSONs from source CSVs
        build_script = os.path.join(scene_dir, "build_library.py")
        if os.path.isfile(build_script):
            print("  No pattern*_scene.json found — auto-building from source CSVs...")
            try:
                import subprocess
                subprocess.run(
                    [sys.executable, build_script, "--extract", "--clean"],
                    cwd=scene_dir, timeout=300)
                # Re-scan
                files = _find_scene_template_files(scene_dir)
                if files:
                    print(f"  Auto-built {len(files)} scene templates")
            except Exception as e:
                print(f"  Auto-build failed: {e}")

    return files


def generate_coastlines_from_scene_template(scene_template_files, max_range_m):
    """Generate coastlines by loading a real scene JSON as a whole template.

    Picks a random scene, applies a rigid transform (rotation + optional mirror
    + slight scale) to all coastlines as a group, randomly drops 0-30% for
    variety, clips to the radar range circle, and returns calibrated echo stats.

    Returns:
        (List[CoastlineSegment], dict) — coastlines and echo_stats_map.
        On failure returns ([], {}).
    """
    if not scene_template_files:
        return [], {}

    scene_path = random.choice(scene_template_files)
    try:
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return [], {}

    raw_coastlines = scene_data.get("coastlines", [])
    raw_stats = scene_data.get("coastline_stats", {})
    if not raw_coastlines:
        return [], {}

    # --- Rigid transform parameters ---
    theta = random.uniform(0, 2 * math.pi)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    mirror = random.random() < 0.5
    scale = random.uniform(0.85, 1.15)

    # --- Randomly drop 0-30% of coastlines (keep at least 1) ---
    if len(raw_coastlines) > 1:
        keep_frac = random.uniform(0.70, 1.0)
        k = max(1, int(len(raw_coastlines) * keep_frac))
        raw_coastlines = random.sample(raw_coastlines, k)

    range_sq = max_range_m * max_range_m
    coastlines = []
    echo_stats_map = {}

    for rc in raw_coastlines:
        orig_name = rc.get("name", "coastline")
        tmpl_name = f"tmpl_{orig_name}"
        raw_pts = rc.get("points", [])
        if len(raw_pts) < 3:
            continue

        # Apply transform: scale, mirror, rotate
        transformed = []
        for x, y in raw_pts:
            x2, y2 = x * scale, y * scale
            if mirror:
                x2 = -x2
            rx = x2 * cos_t - y2 * sin_t
            ry = x2 * sin_t + y2 * cos_t
            transformed.append((rx, ry))

        # Clip: keep only points inside the radar range circle
        clipped = [(x, y) for x, y in transformed if x * x + y * y <= range_sq]
        if len(clipped) < 3:
            continue

        # Filter out unnatural-looking coastlines
        if not _coastline_looks_natural(clipped):
            continue

        coastlines.append(CoastlineSegment(points=clipped, name=tmpl_name))

        # Map echo stats with the new name
        if orig_name in raw_stats:
            echo_stats_map[tmpl_name] = raw_stats[orig_name]

    # Remove coastlines that heavily overlap with larger ones
    coastlines = _filter_overlapping_coastlines(coastlines)
    # Update echo_stats_map to only include surviving coastlines
    kept_names = set(c.name for c in coastlines)
    echo_stats_map = {k: v for k, v in echo_stats_map.items() if k in kept_names}

    return coastlines, echo_stats_map


def denormalize_coastline_shape(normalized_points, target_bearing_deg,
                                 target_range_m, target_bearing_span_deg,
                                 target_range_span_m, mirror=False):
    """Reverse of normalize: scale [0,1] shape to target position/size.

    Args:
        normalized_points: Points in [0,1] x [0,1] (bearing_offset, range_offset)
        target_bearing_deg: Center bearing for the output coastline
        target_range_m: Minimum range for the output coastline
        target_bearing_span_deg: Angular span of the output
        target_range_span_m: Radial depth of the output
        mirror: If True, flip the bearing axis (horizontal mirror)

    Returns:
        List of (x, y) tuples in meters
    """
    start_bearing = target_bearing_deg - target_bearing_span_deg / 2
    points = []
    for pt in normalized_points:
        b_norm, r_norm = pt[0], pt[1]
        if mirror:
            b_norm = 1.0 - b_norm
        bearing_deg = start_bearing + b_norm * target_bearing_span_deg
        range_m = target_range_m + r_norm * target_range_span_m
        bearing_rad = math.radians(bearing_deg)
        x = range_m * math.sin(bearing_rad)
        y = range_m * math.cos(bearing_rad)
        points.append((x, y))
    return points


def _smooth_polygon_points(points, window=None):
    """Smooth polygon vertices with a Gaussian-weighted moving average.

    Preserves overall shape while removing small-scale jitter from radar
    measurement noise in the source captures. Window scales with point count.
    """
    n = len(points)
    if window is None:
        window = max(3, min(15, n // 40))
    if n < window * 3:
        return points

    # Gaussian kernel for natural-looking smoothing
    half = window // 2
    kernel = np.exp(-0.5 * np.linspace(-2, 2, window) ** 2)
    kernel = kernel / kernel.sum()

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    # Pad edges to avoid boundary artifacts
    xs_pad = np.concatenate([xs[:half][::-1], xs, xs[-half:][::-1]])
    ys_pad = np.concatenate([ys[:half][::-1], ys, ys[-half:][::-1]])

    xs_smooth = np.convolve(xs_pad, kernel, mode='same')[half:half + n]
    ys_smooth = np.convolve(ys_pad, kernel, mode='same')[half:half + n]

    return [(float(xs_smooth[i]), float(ys_smooth[i])) for i in range(n)]


def _chaikin_smooth(points, iterations=2):
    """Chaikin corner-cutting subdivision for smooth curves from angular polygons.

    Doubles vertex count per iteration by cutting corners at 25/75% positions.
    Applied to external coastlines (GeoJSON, OSM) before Gaussian smoothing to
    bring sparse vertex density up to ray-tracing resolution.
    """
    for _ in range(iterations):
        new_pts = []
        for i in range(len(points) - 1):
            p0, p1 = points[i], points[i + 1]
            new_pts.append((0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]))
            new_pts.append((0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]))
        if points[0] == points[-1]:  # closed polygon
            new_pts.append(new_pts[0])
        else:
            new_pts.append(points[-1])
        points = new_pts
    return points


def _warp_coastline_organic(points, warp_amplitude=0.15):
    """Apply large-scale radial deformation to make geometric shapes look natural.

    Converts to polar coords relative to centroid, applies fractal radial warp,
    converts back, then Chaikin-smooths corners.
    """
    n = len(points)
    if n < 10:
        return points
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    polar = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        r = math.sqrt(dx * dx + dy * dy)
        theta = math.atan2(dy, dx)
        polar.append((theta, r))
    mean_r = sum(r for _, r in polar) / n
    radial_warp = fractal_noise_1d(n, amplitude=warp_amplitude * mean_r, beta=1.5)
    warped = []
    for i, (theta, r) in enumerate(polar):
        new_r = max(0.5, r + radial_warp[i])
        warped.append((cx + new_r * math.cos(theta), cy + new_r * math.sin(theta)))
    return _chaikin_smooth(warped, iterations=1)


def generate_coastline_from_primitive(primitive, target_bearing_deg, target_range_m,
                                       target_bearing_span=None, target_range_span=None,
                                       max_range_m=RANGE_M):
    """Generate a CoastlineSegment from a library primitive.

    Denormalizes to target position/size, smooths edges, adds gentle fractal
    perturbation, 50% chance of horizontal mirror. Returns CoastlineSegment
    or None.
    """
    if target_bearing_span is None:
        target_bearing_span = primitive.original_bearing_span_deg
    if target_range_span is None:
        target_range_span = primitive.original_range_span_m

    mirror = random.random() < 0.5

    points = denormalize_coastline_shape(
        primitive.normalized_points, target_bearing_deg,
        target_range_m, target_bearing_span, target_range_span, mirror)

    if len(points) < 4:
        return None

    # Random rotation to break source geometry (prevents triangular/diamond artifacts)
    rotation_deg = random.uniform(-25, 25)
    if abs(rotation_deg) > 1.0:
        rot_rad = math.radians(rotation_deg)
        cos_r, sin_r = math.cos(rot_rad), math.sin(rot_rad)
        rcx = sum(p[0] for p in points) / len(points)
        rcy = sum(p[1] for p in points) / len(points)
        points = [(rcx + (x - rcx) * cos_r - (y - rcy) * sin_r,
                   rcy + (x - rcx) * sin_r + (y - rcy) * cos_r)
                  for x, y in points]

    # Smooth raw shape to remove radar measurement jitter from source captures
    points = _smooth_polygon_points(points)

    # Add gentle perturbation: beta=2.2 gives smooth undulations that preserve
    # the real coastline character without adding jagged high-frequency noise
    n = len(points)
    noise_amplitude = random.uniform(0.10, 0.20) * target_range_span
    noise = fractal_noise_1d(n, beta=2.2, amplitude=noise_amplitude)

    perturbed = []
    for i, (x, y) in enumerate(points):
        r = math.sqrt(x * x + y * y)
        if r < 1e-6:
            perturbed.append((x, y))
            continue
        new_r = max(1.0, min(max_range_m, r + noise[i]))
        scale = new_r / r
        perturbed.append((x * scale, y * scale))

    # Final smoothing pass to blend perturbation with the shape
    perturbed = _smooth_polygon_points(perturbed, window=5)
    perturbed = _warp_coastline_organic(perturbed, warp_amplitude=random.uniform(0.10, 0.20))
    perturbed = _smooth_polygon_points(perturbed, window=5)

    # Second organic warp pass (40% chance) to further break geometric source shapes
    if random.random() < 0.4:
        perturbed = _warp_coastline_organic(perturbed, warp_amplitude=random.uniform(0.06, 0.12))
        perturbed = _smooth_polygon_points(perturbed, window=3)

    name = f"lib_{primitive.classification}"
    return CoastlineSegment(points=perturbed, name=name)


def generate_random_coastlines_from_library(library, max_range_m):
    """Generate coastlines using library primitives as templates.

    Drop-in replacement for fractal generation. Uses same feature type weights,
    maps to library classifications, falls back if type unavailable.
    Handles 40% composite scenes using library primitives.

    Returns (coastlines, echo_stats_map).
    """
    coastlines = []
    echo_stats_map = {}

    type_to_class = {
        "arc": "arc", "linear": "arc",
        "island": "island", "headland": "headland",
        "bay": "bay", "harbor": "bay",
        "breakwater": "thin_linear", "jetty": "thin_linear",
    }

    # 50% composite scenes with multi-scale layering
    if random.random() < 0.5:
        # Near-field anchor (large features close to radar)
        near_bearing = random.uniform(0, 360)
        near_range = random.uniform(max_range_m * 0.05, max_range_m * 0.25)
        # Mid-field anchor — ensure at least 30° separation from near
        mid_offset = random.uniform(-60, 60)
        if abs(mid_offset) < 30:
            mid_offset = 30.0 * (1 if mid_offset >= 0 else -1)
        mid_bearing = near_bearing + mid_offset
        mid_range = random.uniform(max_range_m * 0.25, max_range_m * 0.55)
        # Far-field anchor — ensure at least 30° from both near and mid
        far_offset = random.uniform(-120, 120)
        far_bearing = near_bearing + far_offset
        while (abs((far_bearing - near_bearing + 180) % 360 - 180) < 30 or
               abs((far_bearing - mid_bearing + 180) % 360 - 180) < 30):
            far_offset = random.uniform(-120, 120)
            far_bearing = near_bearing + far_offset
        far_range = random.uniform(max_range_m * 0.55, max_range_m * 0.85)

        num_near = random.randint(2, 5)
        num_mid = random.randint(2, 5)
        num_far = random.randint(2, 5)

        # Pre-filter: large primitives (span > 20°) for anchor features
        large_prims = [p for p in library.primitives
                       if p.original_bearing_span_deg > 20]
        if not large_prims:
            large_prims = library.primitives

        # Per-layer bearing jitter: near=tight, mid=moderate, far=wide
        layer_jitter = [40, 70, 90]

        for layer, (anchor_bearing, anchor_range, n) in enumerate([
            (near_bearing, near_range, num_near),
            (mid_bearing, mid_range, num_mid),
            (far_bearing, far_range, num_far),
        ]):
            jitter = layer_jitter[layer]
            for i in range(n):
                bearing = anchor_bearing + random.uniform(-jitter, jitter)
                base_range = anchor_range + random.uniform(-15, 15)
                # Add asymmetric range spread for mid/far layers
                if layer > 0:
                    base_range += random.uniform(-max_range_m * 0.06, max_range_m * 0.06)
                base_range = max(max_range_m * 0.03, min(base_range, max_range_m * 0.85))

                # First feature per layer uses large primitive; rest random
                if i == 0:
                    primitive = random.choice(large_prims)
                else:
                    primitive = random.choice(library.primitives)
                bearing_span = primitive.original_bearing_span_deg * random.uniform(0.6, 1.5)
                range_span = primitive.original_range_span_m * random.uniform(0.6, 1.5)

                if base_range + range_span > max_range_m * 0.98:
                    range_span = max(5.0, max_range_m * 0.98 - base_range)

                coastline = generate_coastline_from_primitive(
                    primitive, bearing, base_range,
                    target_bearing_span=bearing_span,
                    target_range_span=range_span,
                    max_range_m=max_range_m)
                if coastline:
                    coastline.name = f"lib_{primitive.classification}_{len(coastlines)}"
                    coastlines.append(coastline)
                    if primitive.echo_stats:
                        echo_stats_map[coastline.name] = primitive.echo_stats

        # Add fill segments clustered near existing features
        num_fill = random.randint(4, 12)
        fill = _generate_fill_segments(max_range_m, num_fill,
                                        anchor_coastlines=coastlines)
        coastlines.extend(fill)

        # Filter out unnatural shapes and heavy overlaps
        coastlines = [c for c in coastlines if _coastline_looks_natural(c.points)]
        coastlines = _filter_overlapping_coastlines(coastlines)
        coastlines = _fix_ring_and_spoke_arrangements(coastlines, max_range_m)
        kept_names = set(c.name for c in coastlines)
        echo_stats_map = {k: v for k, v in echo_stats_map.items() if k in kept_names}

        return coastlines, echo_stats_map

    # Independent features (increased from 1-5 to 3-10)
    num_features = random.randint(3, 10)
    feature_types = random.choices(
        ["arc", "island", "linear", "headland", "bay", "harbor", "breakwater", "jetty"],
        weights=[15, 10, 20, 15, 15, 8, 10, 7],
        k=num_features,
    )

    for feat_idx, feature_type in enumerate(feature_types):
        cls = type_to_class.get(feature_type, "arc")
        candidates = library.by_classification.get(cls, [])

        if not candidates:
            candidates = library.primitives
        if not candidates:
            continue

        # For the first 2 features, prefer larger primitives (span > 15°)
        if feat_idx < 2:
            large_cands = [p for p in candidates if p.original_bearing_span_deg > 15]
            if large_cands:
                candidates = large_cands

        primitive = random.choice(candidates)

        bearing = random.uniform(0, 360)
        base_range = random.uniform(max_range_m * 0.05, max_range_m * 0.75)
        bearing_span = primitive.original_bearing_span_deg * random.uniform(0.7, 1.5)
        range_span = primitive.original_range_span_m * random.uniform(0.7, 1.5)

        if base_range + range_span > max_range_m * 0.98:
            range_span = max(5.0, max_range_m * 0.98 - base_range)

        coastline = generate_coastline_from_primitive(
            primitive, bearing, base_range,
            target_bearing_span=bearing_span,
            target_range_span=range_span,
            max_range_m=max_range_m)
        if coastline:
            coastline.name = f"lib_{primitive.classification}_{len(coastlines)}"
            coastlines.append(coastline)
            if primitive.echo_stats:
                echo_stats_map[coastline.name] = primitive.echo_stats

    # Add fill segments clustered near existing features
    num_fill = random.randint(4, 12)
    fill = _generate_fill_segments(max_range_m, num_fill,
                                    anchor_coastlines=coastlines)
    coastlines.extend(fill)

    # Filter out unnatural shapes and heavy overlaps
    coastlines = [c for c in coastlines if _coastline_looks_natural(c.points)]
    coastlines = _filter_overlapping_coastlines(coastlines)
    coastlines = _fix_ring_and_spoke_arrangements(coastlines, max_range_m)
    kept_names = set(c.name for c in coastlines)
    echo_stats_map = {k: v for k, v in echo_stats_map.items() if k in kept_names}

    return coastlines, echo_stats_map


def find_best_matching_primitive(query_points, library, preferred_type=None):
    """Find the library primitive most similar to query_points.

    Uses weighted composite similarity:
    aspect_ratio (0.3) + convexity (0.3) + point_density (0.2) + fourier (0.2).
    """
    if not library or not library.primitives:
        return None

    pts = [(p[0], p[1]) for p in query_points]
    classification, metrics = classify_coastline(pts)
    query_fds = compute_fourier_descriptors(pts)

    if preferred_type and preferred_type in library.by_classification:
        candidates = library.by_classification[preferred_type]
    elif classification in library.by_classification:
        candidates = library.by_classification[classification]
    else:
        candidates = library.primitives

    if not candidates:
        candidates = library.primitives

    best_score = -1
    best_prim = None

    for prim in candidates:
        ar_sim = 1.0 - min(1.0, abs(metrics["aspect_ratio"] - prim.aspect_ratio) /
                           max(prim.aspect_ratio, 0.01))
        cx_sim = 1.0 - min(1.0, abs(metrics["convexity"] - prim.convexity))

        pd_query = len(pts) / max(metrics["bearing_span_deg"], 1.0)
        pd_prim = prim.original_num_points / max(prim.original_bearing_span_deg, 1.0)
        pd_sim = 1.0 - min(1.0, abs(pd_query - pd_prim) / max(pd_prim, 0.01))

        fd_dist = sum((query_fds[i] - prim.fourier_descriptors[i]) ** 2
                      for i in range(min(len(query_fds), len(prim.fourier_descriptors))))
        fd_sim = 1.0 / (1.0 + math.sqrt(fd_dist))

        score = 0.3 * ar_sim + 0.3 * cx_sim + 0.2 * pd_sim + 0.2 * fd_sim
        if score > best_score:
            best_score = score
            best_prim = prim

    return best_prim


def assign_echo_profiles_to_geojson_coastlines(coastlines, library):
    """Match GeoJSON coastlines to closest library primitives' echo stats.

    For each coastline, finds the best matching primitive by shape similarity
    and returns its echo_stats for rendering.
    """
    if not library or not library.primitives:
        return {}

    echo_stats = {}
    for coastline in coastlines:
        prim = find_best_matching_primitive(coastline.points, library)
        if prim and prim.echo_stats:
            echo_stats[coastline.name] = prim.echo_stats
    return echo_stats


def attach_echo_stats_to_scene(scene, coastlines, echo_stats_map):
    """Inject per-coastline echo stats into scene['coastline_stats'].

    Merges library-inherited echo stats with any existing stats in the scene.
    The rendering engine reads from this dict, so no rendering changes needed.
    """
    if not echo_stats_map:
        return
    existing_stats = scene.get("coastline_stats", {})
    for coastline in coastlines:
        if coastline.name in echo_stats_map and coastline.name not in existing_stats:
            existing_stats[coastline.name] = echo_stats_map[coastline.name]
    scene["coastline_stats"] = existing_stats


def _generate_fill_segments(max_range_m: float, num_segments: int,
                            exclude_bearings: List[Tuple[float, float]] = None,
                            anchor_coastlines: List[CoastlineSegment] = None,
                            ) -> List[CoastlineSegment]:
    """Generate small fill segments that cluster near existing features.

    When anchor_coastlines are provided, 70% of fill segments are placed
    near existing features (within ±15° bearing, ±20m range) for natural
    spatial clustering.  30% are placed randomly for coverage.

    Size varies: mix of small (2-5° span) and very small (1-3°) segments
    to match the varied size distribution seen in real scenes.
    """
    segments = []

    # Compute anchor bearings/ranges for clustering
    anchors = []
    if anchor_coastlines:
        for c in anchor_coastlines:
            cx = sum(p[0] for p in c.points) / len(c.points)
            cy = sum(p[1] for p in c.points) / len(c.points)
            brg = math.degrees(math.atan2(cx, cy)) % 360
            rng = math.sqrt(cx * cx + cy * cy)
            anchors.append((brg, rng))

    for idx in range(num_segments):
        # 70% cluster near anchors, 30% random
        if anchors and random.random() < 0.70:
            anchor_brg, anchor_rng = random.choice(anchors)
            bearing = anchor_brg + random.uniform(-15, 15)
            range_at = anchor_rng + random.uniform(-20, 20)
            range_at = max(max_range_m * 0.03, min(range_at, max_range_m * 0.95))
        else:
            bearing = random.uniform(0, 360)
            # Skip bearings that overlap with existing features
            if exclude_bearings:
                overlaps = False
                for b_start, b_end in exclude_bearings:
                    if b_start <= bearing <= b_end or (b_end > 360 and bearing <= b_end - 360):
                        overlaps = True
                        break
                if overlaps:
                    continue
            band = random.choices(["near", "mid", "far"], weights=[30, 40, 30], k=1)[0]
            if band == "near":
                range_at = random.uniform(max_range_m * 0.05, max_range_m * 0.25)
            elif band == "mid":
                range_at = random.uniform(max_range_m * 0.25, max_range_m * 0.55)
            else:
                range_at = random.uniform(max_range_m * 0.55, max_range_m * 0.85)

        # Varied sizes: minimum 3° to prevent spoke-like thin lines
        span = random.choice([
            random.uniform(3, 6),   # small
            random.uniform(5, 10),  # medium-small
            random.uniform(8, 18),  # medium
        ])
        start_b = bearing - span / 2
        end_b = bearing + span / 2
        depth = random.uniform(8, max_range_m * 0.18)
        seg = generate_linear_coastline(start_b, end_b, range_at, depth)
        # Add organic curvature to prevent straight-edged artifacts
        seg = CoastlineSegment(
            points=_warp_coastline_organic(seg.points, warp_amplitude=random.uniform(0.12, 0.25)),
            name=f"fill_segment_{len(segments)}")
        segments.append(seg)
    return segments


def generate_random_coastlines(max_range_m: float, library=None) -> List[CoastlineSegment]:
    """Generate a natural-looking set of coastline features using spatial clustering.

    Coastlines are placed in 2-4 spatial clusters (like real shoreline sectors),
    each with a large anchor feature and smaller companion features nearby.
    This produces scenes where coastlines appear in tandem with irregular,
    rounded edges at varied sizes — matching real radar observations.

    When a coastline library is available, uses real shapes 80% of the time.
    50% chance of generating a spatially coherent composite scene instead.

    Scene composition (informed by user-curated natural scenes):
      - 2-4 spatial clusters at different bearings/ranges
      - Each cluster: 1 large anchor (15-40m) + 2-6 smaller companions (2-12m)
      - Fill segments clustered near existing features (not random scatter)
      - Size distribution: ~25% small (<3m), ~30% medium (3-8m), ~25% mid (8-15m), ~20% large (15m+)
      - Range distribution: ~45% near, ~40% mid, ~15% far
    """
    if library and library.primitives:
        # Always prefer library when available — fractal path suppressed
        coastlines, _ = generate_random_coastlines_from_library(library, max_range_m)
        return coastlines

    # No library available — try to build one
    fallback_library = build_coastline_library(verbose=False)
    if fallback_library and fallback_library.primitives:
        coastlines, _ = generate_random_coastlines_from_library(fallback_library, max_range_m)
        return coastlines

    # No library possible — try scene templates
    scene_template_files = _discover_scene_template_files()
    if scene_template_files:
        coastlines, _ = generate_coastlines_from_scene_template(scene_template_files, max_range_m)
        return coastlines

    # Absolute last resort — return empty (fractal suppressed)
    return []


# ============================================================
# Scene Generator
# ============================================================
def generate_random_objects(config: SceneConfig, max_range_m: float,
                            coastlines: List[CoastlineSegment]) -> List[RadarObject]:
    """Generate random radar objects avoiding coastline interiors."""
    objects = []

    def random_position():
        for _ in range(50):
            r = random.uniform(DEAD_ZONE_BINS * BIN_SIZE_M + 2, max_range_m * 0.95)
            bearing = random.uniform(0, 2 * math.pi)
            x = r * math.sin(bearing)
            y = r * math.cos(bearing)
            if not point_in_any_coastline(x, y, coastlines):
                return x, y
        r = random.uniform(max_range_m * 0.3, max_range_m * 0.6)
        bearing = random.uniform(0, 2 * math.pi)
        return r * math.sin(bearing), r * math.cos(bearing)

    def make_object(obj_type):
        params = OBJECT_PARAMS[obj_type]
        x, y = random_position()
        rcs = random.uniform(*params["rcs_range"])
        length = random.uniform(*params["length_range"])
        width = random.uniform(*params["width_range"])
        heading = random.uniform(0, 360)

        speed_min, speed_max = params["speed_range"]
        if speed_max == 0:
            speed = 0.0
            course = 0.0
            turn_rate = 0.0
        else:
            if obj_type in ("small_boat", "barge") and random.random() < 0.3:
                speed = 0.0
                course = 0.0
                turn_rate = 0.0
            else:
                speed = random.uniform(speed_min, speed_max)
                course = random.uniform(0, 360)
                turn_rate = random.uniform(*params["turn_rate_range"])

        return RadarObject(x=x, y=y, obj_type=obj_type, rcs_m2=rcs,
                          length_m=length, width_m=width, heading_deg=heading,
                          speed_kts=speed, course_deg=course,
                          turn_rate_dps=turn_rate)

    type_counts = [
        ("buoy", random.randint(*config.num_buoys)),
        ("small_boat", random.randint(*config.num_small_boats)),
        ("medium_boat", random.randint(*config.num_medium_boats)),
        ("large_boat", random.randint(*config.num_large_boats)),
        ("barge", random.randint(*config.num_barges)),
    ]

    for obj_type, count in type_counts:
        for _ in range(count):
            objects.append(make_object(obj_type))

    return objects


def point_in_polygon(px: float, py: float,
                     polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def point_in_any_coastline(x: float, y: float,
                           coastlines: List[CoastlineSegment]) -> bool:
    for c in coastlines:
        if point_in_polygon(x, y, c.points):
            return True
    return False


def _identify_fallback_coastline_name(ref_start: int, ref_end: int,
                                      bearing_rad: float,
                                      scene_bin_size: float,
                                      coastlines: List[CoastlineSegment],
                                      intervals_by_name: dict[str, list[tuple[int, int]]]) -> Optional[str]:
    """
    Best-effort mapping from a missed reference spoke segment back to a coastline
    region name so strict Mode 4 can synthesize the fallback with that region's
    statistics instead of replaying captured bin values directly.
    """
    if ref_end < ref_start:
        return None

    mid_bin = 0.5 * (float(ref_start) + float(ref_end))
    mid_range_m = max(0.0, (mid_bin + 0.5) * float(scene_bin_size))
    mid_x = mid_range_m * math.sin(bearing_rad)
    mid_y = mid_range_m * math.cos(bearing_rad)

    for coastline in coastlines:
        if point_in_polygon(mid_x, mid_y, coastline.points):
            return coastline.name

    best_name = None
    best_gap = None
    for coastline in coastlines:
        intervals = intervals_by_name.get(coastline.name, [])
        if not intervals:
            continue
        for entry_bin, exit_bin in intervals:
            if exit_bin < ref_start:
                gap = ref_start - exit_bin
            elif entry_bin > ref_end:
                gap = entry_bin - ref_end
            else:
                gap = 0
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_name = coastline.name

    return best_name


# ============================================================
# Terrain Occlusion (Radar Shadow)
# ============================================================
def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort and merge overlapping (entry_bin, exit_bin) pairs into non-overlapping list."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_iv[0])]
    for entry, exit_ in sorted_iv[1:]:
        if entry <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], exit_)
        else:
            merged.append([entry, exit_])
    return [(m[0], m[1]) for m in merged]


def compute_shadow_mask(coastlines: List[CoastlineSegment],
                        dx: float, dy: float,
                        bin_size: float, dead_zone: int,
                        num_bins: int) -> Optional[int]:
    """Compute the minimum entry bin where radar shadow begins on this spoke.

    Fires ray along (dx, dy) from origin, finds all coastline entry/exit pairs
    via ray_polygon_intersections(), merges them, and returns the minimum
    entry_bin. Everything at or beyond this bin is shadowed (signal absorbed
    by terrain). Returns None if no coastline hit.
    """
    land_intervals = []
    for coastline in coastlines:
        hits = ray_polygon_intersections(0, 0, dx, dy, coastline.points)
        if len(hits) >= 2:
            for pair_idx in range(0, len(hits) - 1, 2):
                entry_m = hits[pair_idx]
                exit_m = hits[pair_idx + 1] if pair_idx + 1 < len(hits) else entry_m + 5
                entry_bin = max(dead_zone, int(entry_m / bin_size))
                exit_bin = min(num_bins - 1, int(exit_m / bin_size))
                if exit_bin >= entry_bin:
                    land_intervals.append((entry_bin, exit_bin))
    if not land_intervals:
        return None
    return min(entry for entry, _ in land_intervals)


def is_in_radar_shadow(obj_center_bin: int, shadow_entry_bin: int) -> bool:
    """Returns True if obj_center_bin is at or beyond shadow_entry_bin."""
    return obj_center_bin >= shadow_entry_bin


# ============================================================
# Ray-Geometry Intersection Engine
# ============================================================
def ray_polygon_intersections(ox: float, oy: float, dx: float, dy: float,
                              polygon: List[Tuple[float, float]]) -> List[float]:
    """Find all intersection distances of ray (ox,oy)+t*(dx,dy) with polygon edges."""
    hits = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:
            continue
        t = ((x1 - ox) * ey - (y1 - oy) * ex) / denom
        s = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
        if t > 0 and 0 <= s <= 1:
            hits.append(t)
    return sorted(hits)


def compute_object_echo(obj: RadarObject, bearing_rad: float,
                        bin_distances: np.ndarray,
                        scene_bins: int = NUM_BINS,
                        scene_bin_size: float = BIN_SIZE_M,
                        scene_dead_zone: int = DEAD_ZONE_BINS) -> np.ndarray:
    """Render an object with the shared dB-domain object renderer."""
    del bin_distances
    return _render_object_echo_shared(
        obj,
        bearing_rad,
        scene_bins=scene_bins,
        scene_bin_size=scene_bin_size,
        scene_dead_zone=scene_dead_zone,
        sim_mode=SIM_MODE_M1,
    ).echoes

def _build_filled_coastline_template(
        coastlines: List[CoastlineSegment], angle_ticks: List[int],
        scene_bins: int, scene_bin_size: float, scene_dead_zone: int,
        scene_range_m: float, core_echo: float) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
    """Rasterize static filled coastlines once for a full output rotation.

    The old path repeated vectorized edge intersections for every spoke and
    every beam sub-ray.  Editable Mode 5 coastlines are static within a
    rotation, so build a polar land mask once, broaden it across the antenna
    beam, and reuse the result for all rotations in the export.
    """
    ticks = np.asarray(angle_ticks, dtype=np.float64)
    n_spokes = len(ticks)
    land_mask = np.zeros((n_spokes, scene_bins), dtype=bool)
    if n_spokes == 0 or not coastlines:
        return np.zeros((n_spokes, scene_bins), dtype=np.float64), [[] for _ in range(n_spokes)]

    bearings = ticks * (2.0 * math.pi / TICKS_PER_REV)
    rdx = np.sin(bearings)[:, None]
    rdy = np.cos(bearings)[:, None]

    for coastline in coastlines:
        points = np.asarray(coastline.points, dtype=np.float64)
        if len(points) < 3:
            continue
        if np.linalg.norm(points[0] - points[-1]) > 1e-6:
            points = np.vstack([points, points[0]])

        p1 = points[:-1]
        p2 = points[1:]
        ex = p2[:, 0] - p1[:, 0]
        ey = p2[:, 1] - p1[:, 1]
        denom = rdx * ey[None, :] - rdy * ex[None, :]
        valid = np.abs(denom) > 1e-10
        distances = (p1[:, 0][None, :] * ey[None, :] -
                     p1[:, 1][None, :] * ex[None, :]) / np.where(valid, denom, 1.0)
        edge_fraction = (p1[:, 0][None, :] * rdy -
                         p1[:, 1][None, :] * rdx) / np.where(valid, denom, 1.0)
        hit = valid & (distances > 0.0) & (edge_fraction >= 0.0) & \
            (edge_fraction <= 1.0) & (distances <= scene_range_m)
        distances[~hit] = np.inf
        distances.sort(axis=1)

        for pair_idx in range(0, distances.shape[1] - 1, 2):
            start_m = distances[:, pair_idx]
            end_m = distances[:, pair_idx + 1]
            finite = np.isfinite(start_m) & np.isfinite(end_m)
            if not np.any(finite):
                continue
            row_indices = np.flatnonzero(finite)
            start_bins = np.maximum(scene_dead_zone, (start_m[finite] / scene_bin_size).astype(int))
            end_bins = np.minimum(scene_bins - 1, (end_m[finite] / scene_bin_size).astype(int))
            for row, start_bin, end_bin in zip(row_indices, start_bins, end_bins):
                if end_bin >= start_bin:
                    land_mask[row, start_bin:end_bin + 1] = True

    # Approximate the five sub-rays used by the former renderer as shifted
    # polar rows. The maximum preserves the prior strongest-sub-ray behavior.
    beam_strength = np.zeros_like(land_mask, dtype=np.float64)
    half_beam = BEAMWIDTH_DEG / 2.0
    spoke_step_deg = 360.0 / max(1, n_spokes)
    for offset_deg in np.linspace(-half_beam, half_beam, 5):
        offset_rows = int(round(offset_deg / spoke_step_deg))
        weight = 1.0 - 0.3 * abs(offset_deg) / (half_beam + 0.01)
        shifted = np.roll(land_mask, -offset_rows, axis=0)
        beam_strength = np.maximum(beam_strength, shifted * weight)

    bin_ranges = np.maximum(1.0, np.arange(scene_bins, dtype=np.float64) * scene_bin_size)
    reference_range_m = scene_range_m * 0.8
    range_attenuation = np.minimum(1.0, (reference_range_m / bin_ranges) ** 2)
    echoes = beam_strength * (0.75 * float(core_echo)) * range_attenuation[None, :]

    active = beam_strength > 0.0
    edge_bins = active & (
        np.logical_not(np.roll(active, 1, axis=1)) |
        np.logical_not(np.roll(active, -1, axis=1))
    )
    echoes[edge_bins] *= 0.75

    intervals: List[List[Tuple[int, int]]] = []
    for row in active:
        active_bins = np.flatnonzero(row)
        if len(active_bins) == 0:
            intervals.append([])
            continue
        splits = np.flatnonzero(np.diff(active_bins) > 1) + 1
        groups = np.split(active_bins, splits)
        intervals.append([(int(group[0]), int(group[-1])) for group in groups])

    return echoes, intervals





def _compute_object_echo_calibrated(obj, bearing_rad: float, cal: dict,
                                     scene_bins: int = NUM_BINS,
                                     scene_bin_size: float = BIN_SIZE_M,
                                     scene_dead_zone: int = DEAD_ZONE_BINS) -> np.ndarray:
    """Compute calibrated echo contribution of an object at a given bearing.

    Like compute_object_echo() but uses per-object calibration parameters
    from reference data. No jitter — deterministic output.

    Args:
        cal: dict with calibrated_ref_code, calibrated_compression,
             use_reference_spread, spread_pattern, spread_bins, center_bin
    """
    echoes = np.zeros(scene_bins, dtype=np.float64)

    obj_range = math.sqrt(obj.x ** 2 + obj.y ** 2)
    obj_bearing = math.atan2(obj.x, obj.y)

    d_angle = bearing_rad - obj_bearing
    d_angle = (d_angle + math.pi) % (2 * math.pi) - math.pi
    d_angle_deg = math.degrees(d_angle)

    half_beam = BEAMWIDTH_DEG / 2
    if abs(d_angle_deg) > half_beam + 1.0:
        return echoes

    beam_gain = math.exp(-2.77 * (d_angle_deg / half_beam) ** 2)

    ref_range = 50.0
    ref_rcs = 350.0
    ref_code = cal.get('calibrated_ref_code', 190.0)
    compression = cal.get('calibrated_compression', 55.0)

    if obj_range < scene_dead_zone * scene_bin_size:
        return echoes
    if obj.rcs_m2 <= 0 or obj_range <= 0:
        return echoes

    intensity_dB = (10 * math.log10(obj.rcs_m2 / ref_rcs) -
                    40 * math.log10(obj_range / ref_range))
    peak_code = ref_code * (10 ** (intensity_dB / compression))
    peak_db = (
        _display_db_from_code_value(peak_code) +
        _shape_fraction_drop_db(beam_gain) +
        _object_peak_correction_db(obj.obj_type, obj_range)
    )
    peak_code = _code_value_from_display_db(peak_db)
    peak_code = min(252, max(0, peak_code))

    if peak_code < 4:
        return echoes

    center_bin = int(obj_range / scene_bin_size)

    if cal.get('use_reference_spread') and cal.get('spread_pattern'):
        # Paint observed spread pattern aligned to center_bin, scaled by beam_gain
        pattern = cal['spread_pattern']
        pattern_peak = max(float(max(pattern)), 1.0)
        ref_center = cal.get('center_bin', center_bin)
        offset = center_bin - ref_center  # shift pattern to current center

        for pi, pval in enumerate(pattern):
            b = ref_center + pi + offset - len(pattern) // 2
            profile = float(pval) / pattern_peak
            val = _code_from_peak_db_shape(peak_db, profile)
            if 0 <= b < scene_bins and b >= scene_dead_zone and val >= 6:
                echoes[b] = max(echoes[b], val)
    else:
        # Gaussian spread with calibrated spread_bins, no jitter
        spread_bins = cal.get('spread_bins', 2)
        sigma = max(spread_bins * 0.5, 0.5)

        for b in range(max(scene_dead_zone, center_bin - spread_bins * 2),
                       min(scene_bins, center_bin + spread_bins * 2 + 1)):
            dist_bins = abs(b - center_bin)
            profile = math.exp(-0.5 * (dist_bins / sigma) ** 2)
            val = _code_from_peak_db_shape(peak_db, profile)
            if val >= 6:
                echoes[b] = max(echoes[b], val)

    return echoes


def _compute_object_echo_calibrated(obj, bearing_rad: float, cal: dict,
                                     scene_bins: int = NUM_BINS,
                                     scene_bin_size: float = BIN_SIZE_M,
                                     scene_dead_zone: int = DEAD_ZONE_BINS) -> np.ndarray:
    """Render an object with strict Mode 4 priors but shared synthesis."""
    return _render_object_echo_shared(
        obj,
        bearing_rad,
        scene_bins=scene_bins,
        scene_bin_size=scene_bin_size,
        scene_dead_zone=scene_dead_zone,
        sim_mode=SIM_MODE_CURRENT_MODE4,
        cal=cal,
    ).echoes


# ============================================================
# Sweep Generator
# ============================================================
@dataclass
class SpokeDetection:
    """Objects detected on a single spoke (for camera sync output)."""
    obj_index: int
    obj_type: str
    range_m: float
    bearing_deg: float
    peak_echo: int
    x: float
    y: float
    speed_kts: float
    course_deg: float
    motion: str


def generate_sweep(angle_tick: int, objects: List[RadarObject],
                   coastlines: List[CoastlineSegment],
                   gain: int = GAIN) -> Tuple[List[int], List[SpokeDetection]]:
    """Generate one spoke of radar data for a given angle tick.
    Returns (echo_values, detections)."""
    bearing_deg = angle_tick * 360.0 / TICKS_PER_REV
    bearing_rad = math.radians(bearing_deg)
    dx = math.sin(bearing_rad)
    dy = math.cos(bearing_rad)

    echoes = np.zeros(NUM_BINS, dtype=np.float64)

    # --- Coastline returns (saturated at 252) ---
    land_intervals = []
    for coastline in coastlines:
        hits = ray_polygon_intersections(0, 0, dx, dy, coastline.points)
        if len(hits) >= 2:
            for pair_idx in range(0, len(hits) - 1, 2):
                entry_m = hits[pair_idx]
                exit_m = hits[pair_idx + 1] if pair_idx + 1 < len(hits) else entry_m + 5
                entry_bin = max(DEAD_ZONE_BINS, int(entry_m / BIN_SIZE_M))
                exit_bin = min(NUM_BINS - 1, int(exit_m / BIN_SIZE_M))

                if exit_bin >= entry_bin:
                    land_intervals.append((entry_bin, exit_bin))

                # Ramp-up: 6 graduated bins approaching coastline entry
                # Calibrated from real patterns 11-20: ~120→140→164→188→188→220
                ramp_up_codes = [120, 140, 164, 188, 188, 220]
                for ri, rc in enumerate(ramp_up_codes):
                    b = entry_bin - len(ramp_up_codes) + ri
                    if DEAD_ZONE_BINS <= b < NUM_BINS:
                        jitter = random.randint(-12, 12)
                        echoes[b] = max(echoes[b], min(252, max(0, rc + jitter)))

                for b in range(entry_bin, exit_bin + 1):
                    echoes[b] = 252

                # Ramp-down: 6 graduated bins after coastline exit
                # Calibrated from real patterns 11-20: ~164→140→140→120→120→120
                ramp_down_codes = [164, 140, 140, 120, 120, 120]
                for ri, rc in enumerate(ramp_down_codes):
                    b = exit_bin + 1 + ri
                    if DEAD_ZONE_BINS <= b < NUM_BINS:
                        jitter = random.randint(-12, 12)
                        echoes[b] = max(echoes[b], min(252, max(0, rc + jitter)))
        elif len(hits) == 1:
            entry_bin = max(DEAD_ZONE_BINS, int(hits[0] / BIN_SIZE_M))
            for b in range(entry_bin, min(NUM_BINS, entry_bin + 4)):
                echoes[b] = max(echoes[b], random.randint(120, 252))

    # --- Compute terrain shadow ---
    shadow_entry = None
    if land_intervals:
        shadow_entry = min(entry for entry, _ in land_intervals)

    # --- Object returns + detection tracking ---
    detections = []
    for idx, obj in enumerate(objects):
        # Skip objects in radar shadow (behind land)
        if shadow_entry is not None:
            obj_range_m = math.sqrt(obj.x**2 + obj.y**2)
            if int(obj_range_m / BIN_SIZE_M) >= shadow_entry:
                continue
        obj_echo = compute_object_echo(obj, bearing_rad, None)
        peak = np.max(obj_echo)
        if peak >= 4:
            obj_range = math.sqrt(obj.x**2 + obj.y**2)
            obj_bearing = math.degrees(math.atan2(obj.x, obj.y)) % 360
            detections.append(SpokeDetection(
                obj_index=idx,
                obj_type=obj.obj_type,
                range_m=obj_range,
                bearing_deg=obj_bearing,
                peak_echo=int(min(252, peak)),
                x=obj.x,
                y=obj.y,
                speed_kts=obj.speed_kts,
                course_deg=obj.course_deg,
                motion=obj.motion_label,
            ))
        echoes = np.maximum(echoes, obj_echo)

    # --- Noise floor ---
    noise_prob = gain * 0.0001
    if noise_prob > 0:
        noise_mask = np.random.random(NUM_BINS) < noise_prob
        noise_mask[:DEAD_ZONE_BINS] = False
        noise_mask[echoes > 5] = False
        noise_values = np.random.choice([8, 20, 28], size=NUM_BINS)
        echoes[noise_mask] = noise_values[noise_mask]

    # Quantize to exact Furuno DRS4DNXT gain code levels
    result = np.zeros(NUM_BINS, dtype=int)
    for i in range(DEAD_ZONE_BINS, NUM_BINS):
        val = echoes[i]
        if val <= 0:
            result[i] = 0
        else:
            idx = np.argmin(np.abs(FURUNO_GAIN_CODES - val))
            result[i] = FURUNO_GAIN_CODES[idx]

    return result.tolist(), detections


def generate_angle_ticks() -> List[int]:
    """Generate realistic angle tick sequence matching DRS4DNXT spoke pattern."""
    ticks = []
    step = TICKS_PER_REV / SPOKES_PER_REV
    angle = random.randint(0, 10)

    while angle < TICKS_PER_REV:
        tick = int(round(angle))
        tick = max(0, min(TICKS_PER_REV - 1, tick))
        ticks.append(tick)
        jitter = random.choice([0, 0, 0, 0, 0, 0, 0, 2, -2, 0])
        angle += step + jitter

    if random.random() < 0.3:
        ticks.append(ticks[0])

    return ticks


# ============================================================
# CSV Output
# ============================================================
def write_csv(filepath: str, rows: List[dict]) -> None:
    """Write radar data to CSV in exact Furuno DRS4DNXT format."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Status', 'Scale', 'Range', 'Gain', 'Angle', 'EchoValues'])
        for row in rows:
            csv_row = [row['status'], row['scale'], row['range'],
                       row['gain'], row['angle']] + row['echoes']
            writer.writerow(csv_row)


def write_metadata_json(filepath: str, rows: List[dict],
                        scene_metadata: Optional[dict] = None) -> None:
    """Write companion metadata JSON for a radar CSV."""
    spoke_timestamps = {}
    for row in rows:
        spoke_timestamps[str(row['angle'])] = row['timestamp']

    scene_cfg = {}
    if scene_metadata:
        scene_cfg = scene_metadata.get("radar_config", {}) or {}
    rotation_period = float(scene_cfg.get("rotation_period_sec", ROTATION_DT))

    metadata = {
        "created": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
        "simulator": "Furuno DRS4DNXT Geometric Radar Simulator",
        "radar_config": {
            "range_nm": float(scene_cfg.get("range_nm", RANGE_NM)),
            "range_m": round(float(scene_cfg.get("range_m", RANGE_M)), 1),
            "bins": int(scene_cfg.get("bins", NUM_BINS)),
            "bin_size_m": round(float(scene_cfg.get("bin_size_m", BIN_SIZE_M)), 4),
            "range_index": int(scene_cfg.get("range_index", rows[0]['range'] if rows else RANGE_INDEX)),
            "gain": rows[0]['gain'] if rows else scene_cfg.get("gain", GAIN),
            "spokes": len(rows),
            "ticks_per_rev": int(scene_cfg.get("ticks_per_rev", TICKS_PER_REV)),
            "beamwidth_deg": float(scene_cfg.get("beamwidth_deg", BEAMWIDTH_DEG)),
            "rotation_period_sec": rotation_period,
        },
        "timestamp_info": {
            "format": "YYYY-MM-DDTHH:MM:SS.mmm (ISO 8601, millisecond precision)",
            "computation": f"spoke_time = rotation_start + (angle_tick / 8192) * {rotation_period:g}s",
            "camera_sync": "Match spoke timestamps to FLIR camera frame timestamps",
        },
        "spoke_timestamps": spoke_timestamps,
    }

    if scene_metadata:
        metadata["scene"] = scene_metadata

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def write_detections_csv(filepath: str, detection_rows: List[dict],
                         scene_metadata: Optional[dict] = None) -> None:
    """Write per-spoke object detection file for camera sync."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Timestamp', 'Angle', 'Bearing_deg',
            'ObjID', 'ObjType', 'Range_m', 'ObjBearing_deg', 'Peak_echo',
            'X_m', 'Y_m', 'Speed_kts', 'Course_deg', 'Motion'
        ])
        for row in detection_rows:
            writer.writerow([
                row['timestamp'],
                row['angle'],
                f"{row['bearing_deg']:.2f}",
                row['obj_id'],
                row['obj_type'],
                f"{row['range_m']:.2f}",
                f"{row['obj_bearing_deg']:.2f}",
                row['peak_echo'],
                f"{row['x']:.2f}",
                f"{row['y']:.2f}",
                f"{row['speed_kts']:.1f}",
                f"{row['course_deg']:.1f}",
                row['motion'],
            ])


def generate_filename(base_time: datetime) -> str:
    """Generate Furuno-style filename: YYYYMMDD_HHMMSS_mmm.csv"""
    ms = base_time.microsecond // 1000
    return base_time.strftime('%Y%m%d_%H%M%S') + f'_{ms:03d}.csv'


# ============================================================
# Main Simulator
# ============================================================
def simulate_scene(output_dir: str, config: SceneConfig = None,
                   num_rotations: int = 1, base_time: datetime = None,
                   verbose: bool = True, seed: Optional[int] = None,
                   no_implicit_capture: bool = False) -> List[str]:
    """Generate a complete radar scene and write CSV files.

    Builds a scene dict from random generation, then delegates to
    simulate_from_scene() so all modes share the same rendering engine.
    """
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        # Seed before coastline and object generation. simulate_from_scene also
        # receives this seed so the forward renderer starts from a known stream.
        random.seed(seed)
        np.random.seed(seed)

    if config is None:
        config = SceneConfig()
    if base_time is None:
        base_time = datetime.now()

    os.makedirs(output_dir, exist_ok=True)

    # 1. Discover scene templates only when their implicit use is permitted.
    # The opt-out is intended for experiments where scene geometry must come
    # only from the caller. It also prevents auto-extraction from local CSVs.
    scene_template_files = (
        [] if no_implicit_capture else _discover_scene_template_files()
    )
    library = None

    # 2. Generate coastlines
    #    Priority: 70% scene-template (best fidelity), 30% library
    #    Retry up to 3 times if too few coastlines generated (baseline: ≥3)
    coastlines = []
    echo_stats_map = {}
    if not no_implicit_capture and random.random() < config.coastline_probability:
        for _coast_attempt in range(3):
            roll = random.random()
            if scene_template_files and roll < 0.70:
                coastlines, echo_stats_map = generate_coastlines_from_scene_template(
                    scene_template_files, RANGE_M)
                src = "scene-template"
            else:
                library = build_coastline_library(verbose=verbose)
                if library and library.primitives:
                    coastlines, echo_stats_map = generate_random_coastlines_from_library(library, RANGE_M)
                    src = "library"
                else:
                    # Fallback: scene-template if library unavailable
                    if scene_template_files:
                        coastlines, echo_stats_map = generate_coastlines_from_scene_template(
                            scene_template_files, RANGE_M)
                        src = "scene-template-fallback"
                    else:
                        src = "none (no templates or library available)"
            if len(coastlines) >= 3:
                break
            # Too few coastlines — retry with alternate source
        if verbose:
            print(f"  Generated {len(coastlines)} coastline features ({src})")

    # 3. Generate objects
    objects = generate_random_objects(config, RANGE_M, coastlines)
    type_counts = {}
    for obj in objects:
        type_counts[obj.obj_type] = type_counts.get(obj.obj_type, 0) + 1
    if verbose:
        print(f"  Generated {len(objects)} objects: {type_counts}")

    # 4. Build scene dict (no profile data → forces geometric rendering path)
    scene = {
        "radar_config": {
            "status": STATUS, "scale": SCALE, "range_index": RANGE_INDEX,
            "gain": config.gain, "bins": NUM_BINS, "bin_size_m": BIN_SIZE_M,
            "dead_zone_bins": DEAD_ZONE_BINS, "range_m": RANGE_M,
            "range_nm": RANGE_NM, "ticks_per_rev": TICKS_PER_REV,
        },
        "clutter": {
            "noise_probability": config.gain * 0.0001,
            "noise_codes": [8, 20, 28],
        },
        "decay": {
            "ramp_up_range": [80, 200],
            "ramp_down_range": [40, 160],
        },
        "coastlines": [
            {"name": c.name, "points": [[p[0], p[1]] for p in c.points]}
            for c in coastlines
        ],
        "objects": [
            {"type": obj.obj_type, "x": obj.x, "y": obj.y,
             "rcs_m2": obj.rcs_m2, "length_m": obj.length_m, "width_m": obj.width_m,
             "heading_deg": obj.heading_deg, "speed_kts": obj.speed_kts,
             "course_deg": obj.course_deg, "turn_rate_dps": obj.turn_rate_dps}
            for obj in objects
        ],
    }

    # Attach inherited echo stats from library primitives
    if echo_stats_map:
        scene["coastline_stats"] = echo_stats_map

    scene["_sim_mode"] = SIM_MODE_M1
    if seed is not None:
        scene["_sim_seed"] = seed

    # 5. Delegate to unified rendering engine
    return simulate_from_scene(scene, coastlines, objects, output_dir,
                               scene_path="random_scene",
                               num_rotations=num_rotations, verbose=verbose,
                               base_time=base_time)


def generate_batch(output_base: str, num_scenes: int = 5,
                   rotations_per_scene: int = 10,
                   verbose: bool = True, seed: Optional[int] = None,
                   no_implicit_capture: bool = False) -> None:
    """Generate multiple random scenes, each in its own subfolder."""
    print(f"Generating {num_scenes} scenes, {rotations_per_scene} rotations each")
    print(f"Output: {output_base}")
    print(f"DRS4DNXT @ {RANGE_NM} NM | {NUM_BINS} bins | {BIN_SIZE_M:.3f} m/bin")
    print("=" * 60)

    all_files = []
    for scene_idx in range(num_scenes):
        scene_seed = seed + scene_idx if seed is not None else None
        if scene_seed is not None:
            random.seed(scene_seed)
            np.random.seed(scene_seed)
        scene_name = f"scene_{scene_idx + 1:03d}"
        scene_dir = os.path.join(output_base, scene_name)

        config = SceneConfig(
            num_buoys=(0, random.randint(1, 6)),
            num_small_boats=(0, random.randint(1, 5)),
            num_medium_boats=(0, random.randint(0, 3)),
            num_large_boats=(0, random.randint(0, 2)),
            num_barges=(0, random.randint(0, 1)),
            coastline_probability=random.uniform(0.75, 0.98),
            gain=random.choice([60, 65, 70, 75, 80]),
        )

        base_time = datetime.now() - timedelta(days=random.randint(0, 30),
                                                hours=random.randint(0, 23),
                                                minutes=random.randint(0, 59))

        print(f"\nScene {scene_idx + 1}/{num_scenes}: {scene_name}")
        files = simulate_scene(
            scene_dir, config, rotations_per_scene, base_time, verbose,
            seed=scene_seed, no_implicit_capture=no_implicit_capture)
        all_files.extend(files)

    print(f"\n{'=' * 60}")
    print(f"Done. Generated {len(all_files)} CSV files across {num_scenes} scenes.")


# ============================================================
# Mode 12: Real-Coastline Random Simulation
# ============================================================
def simulate_mode12(output_dir, location=None, num_scenes=1,
                    num_rotations=10, range_nm=None, verbose=True,
                    radarloc_file=None, interactive=False,
                    prefer_source=None, local_radarloc_production=False,
                    modec_options=None):
    """Mode C uses the bundled RustCore scene loader and CSV renderer.

    Existing local files are never re-ranked or re-queried. Source resolution
    is used only when no .radarloc was provided. Old numerical pipelines remain
    isolated below for legacy API callers; other simulator modes are unchanged.
    """
    from coastline_defined.scene_pipeline import run_modec
    files = run_modec(sys.modules[__name__], output_dir, location, num_scenes,
        num_rotations, range_nm, radarloc_file, prefer_source, verbose, modec_options)
    if interactive and files:
        print("Mode C export complete. Open the generated CSV in the Studio or web PPI viewer.")
    return files


def _simulate_mode12_legacy(output_dir, location=None, num_scenes=1,
                    num_rotations=10, range_nm=None, verbose=True,
                    radarloc_file=None, interactive=False,
                    prefer_source=None):
    """Mode 12: Real-coastline random simulation using authoritative shoreline data.

    Resolves real-world shoreline geometry for a location, preferring local
    NOAA/USGS/GeoJSON files, then repo-local authoritative cache, then official
    remote services, and finally OSM as a last fallback. The geometry is matched
    to library echo profiles for realistic rendering before random objects are
    generated and passed to the geometric simulation engine.

    Pipeline per scene:
      1. Resolve location (catalog / coordinates / GNIS / Nominatim / .radarloc)
      2. Load shoreline from local or cached authoritative sources
      3. Optionally query official remote sources, then OSM as last fallback
      4. Fallback: if no real geometry is available → fractal coastlines (Mode 1)
      5. Convert to CoastlineSegment objects (with smoothing)
      6. Match to library echo profiles
      7. Generate random objects avoiding coastlines
      8. Build scene dict + simulate_from_scene() → CSV
      9. (Optional) Launch interactive PPI viewer

    Args:
        output_dir: Output directory for CSV files
        location: None (random), 'lat,lon', or place name
        num_scenes: Number of scenes to generate
        num_rotations: Rotations (CSV files) per scene
        range_nm: Override radar range in NM (default: per location or 0.125)
        verbose: Print progress information
        radarloc_file: Path to .radarloc file (bypasses OSM query)
        interactive: Launch interactive PPI viewer after generation
    """
    print("=" * 60)
    print("  Furuno DRS4DNXT Mode 12: Real-Coastline Simulation")
    print("=" * 60)
    print(f"  Scenes: {num_scenes}, Rotations: {num_rotations}")
    if radarloc_file:
        print(f"  Source: {radarloc_file}")
    print(f"  Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    auto_group_output = (
        os.path.normcase(os.path.abspath(output_dir)) ==
        os.path.normcase(os.path.abspath(MODE12_OUTPUT_BASE))
    )

    # Build coastline library once (for echo profile matching)
    library = build_coastline_library(verbose=verbose)

    all_files = []
    last_scene_data = None  # For interactive viewer

    for scene_idx in range(num_scenes):
        print(f"\n--- Scene {scene_idx + 1}/{num_scenes} ---")

        prep = _prepare_mode12_coastlines(
            location=location, range_nm=range_nm, radarloc_file=radarloc_file,
            library=library, verbose=verbose, prefer_source=prefer_source)
        if prep is None:
            return []

        loc = prep["loc"]
        osm_features = prep["osm_features"]
        coastlines = prep["coastlines"]
        echo_stats_map = prep["echo_stats_map"]
        mode12_metadata = prep["metadata"]
        sim_range_m = prep["sim_range_m"]
        sim_range_nm = prep["sim_range_nm"]
        loc_range_m = prep["loc_range_m"]
        loc_range_nm = prep["loc_range_nm"]

        if verbose:
            print(f"  Location: {loc['name']}")
            print(f"  Coordinates: ({loc['lat']:.4f}, {loc['lon']:.4f})")
            print(f"  Category: {loc.get('category', 'unknown')}")
            print(f"  Resolution: {loc.get('resolution_method', 'unknown')}")
            print(f"  Source type: {loc.get('source_type', 'unknown')}")
            if loc.get("source_path"):
                print(f"  Source path: {loc['source_path']}")
            if not radarloc_file:
                print(f"  Query radius: {loc_range_m:.0f} m ({loc_range_nm:.1f} NM)")
            if prefer_source:
                print(f"  Preferred source override: {prefer_source}")
            _print_modec_validation_summary(mode12_metadata.get("coastline_validation"))

        # 4. Generate random objects avoiding coastlines (min 13 objects)
        #    Minimums: 4+3+3+2+1 = 13 guaranteed
        config = SceneConfig(
            num_buoys=(4, random.randint(4, 6)),
            num_small_boats=(3, random.randint(4, 6)),
            num_medium_boats=(3, random.randint(3, 4)),
            num_large_boats=(2, random.randint(2, 3)),
            num_barges=(1, random.randint(1, 2)),
            gain=random.choice([60, 65, 70, 75, 80]),
        )
        objects = generate_random_objects(config, sim_range_m, coastlines)

        if verbose:
            type_counts = {}
            for obj in objects:
                type_counts[obj.obj_type] = type_counts.get(obj.obj_type, 0) + 1
            print(f"  Objects: {len(objects)} — {type_counts}")

        # 5. Build scene dict (geometric rendering path)
        scene = _build_mode12_scene(
            config, coastlines, objects, sim_range_m, sim_range_nm,
            metadata=mode12_metadata, echo_stats_map=echo_stats_map)

        # 6. Determine output directory for this scene
        run_created_at = datetime.now()
        if num_scenes > 1:
            scene_dir = _modec_scene_output_dir(
                output_dir,
                loc["name"],
                created_at=run_created_at,
                scene_idx=scene_idx + 1,
                auto_group=auto_group_output,
            )
        else:
            scene_dir = _modec_scene_output_dir(
                output_dir,
                loc["name"],
                created_at=run_created_at,
                auto_group=auto_group_output,
            )
        os.makedirs(scene_dir, exist_ok=True)

        base_time = datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59))

        # 7. Simulate and write CSV
        files = simulate_from_scene(scene, coastlines, objects, scene_dir,
                                     scene_path="mode12_scene",
                                     num_rotations=num_rotations,
                                     verbose=verbose, base_time=base_time)
        all_files.extend(files)

        # 8. Save scene JSON with location metadata
        scene_json_path = os.path.join(scene_dir, "mode12_scene.json")
        with open(scene_json_path, 'w') as f:
            json.dump(scene, f, indent=2)
        validation_json_path = os.path.join(scene_dir, "coastline_validation.json")
        with open(validation_json_path, 'w') as f:
            json.dump(mode12_metadata.get("coastline_validation", {}), f, indent=2)
        if verbose:
            print(f"  Scene JSON: {scene_json_path}")
            print(f"  Coastline validation: {validation_json_path}")

        last_scene_data = {
            "scene": scene, "coastlines": coastlines, "objects": objects,
            "scene_dir": scene_dir, "scene_json_path": scene_json_path,
            "loc": loc, "config": config, "echo_stats_map": echo_stats_map,
        }

    print(f"\n{'=' * 60}")
    print(f"Mode 12 complete. Generated {len(all_files)} CSV files across {num_scenes} scenes.")

    # Store last scene data as module attribute (allows GUI to retrieve it
    # for launching the interactive viewer on the main thread)
    import sys
    sys.modules[__name__]._mode12_last_scene_data = last_scene_data

    # 9. Launch interactive viewer if requested (CLI path — runs on main thread)
    if interactive and last_scene_data:
        _launch_mode12_interactive_viewer(last_scene_data, output_dir,
                                           num_rotations, library, verbose)

    return all_files


def _launch_mode12_interactive_viewer(scene_data, output_dir, num_rotations,
                                       library, verbose):
    """Launch an animated interactive PPI viewer for Mode 12 scene tuning.

    Features:
      - Animated object motion with heading indicators and trail histories
      - Play/Pause/Step controls with adjustable speed multiplier
      - Real-time parameter sliders (gain, noise, object count, rotations)
      - Zoom/pan via matplotlib toolbar + scroll wheel
      - Regenerate objects and re-run simulation buttons
      - Elapsed-time and rotation counter display
      - Object legend with type-specific colors and sizes

    Args:
        scene_data: Dict with scene, coastlines, objects, loc, config, etc.
        output_dir: Base output directory for re-runs
        num_rotations: Default rotations for re-runs
        library: Coastline library for echo profile matching
        verbose: Print progress
    """
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Slider, Button
    except ImportError:
        print("WARNING: matplotlib not available — skipping interactive viewer.")
        print("  Install with: pip install matplotlib")
        return

    import copy as _copy

    scene = scene_data["scene"]
    coastlines = scene_data["coastlines"]
    loc = scene_data["loc"]
    config = scene_data["config"]
    echo_stats_map = scene_data.get("echo_stats_map", {})

    # Deep-copy objects so animation doesn't corrupt the originals for re-run
    objects = [_copy.deepcopy(obj) for obj in scene_data["objects"]]
    # Keep originals for reset
    original_objects = scene_data["objects"]

    # Animation / UI state
    DT_PER_FRAME = ROTATION_DT  # Each frame = one 2.5s rotation
    TRAIL_LENGTH = 40            # How many past positions to keep per object
    state = {
        "playing": False,
        "speed": 1.0,
        "gain": scene["radar_config"]["gain"],
        "noise_prob": scene["clutter"]["noise_probability"],
        "num_objects": len(objects),
        "rotations": num_rotations,
        "elapsed": 0.0,
        "frame": 0,
    }

    # Trail history: list of deque per object
    from collections import deque
    trails = [deque(maxlen=TRAIL_LENGTH) for _ in objects]

    # Color / size maps
    COLOR_MAP = {
        "buoy": "#ffff00", "small_boat": "#ff8800",
        "medium_boat": "#ff4400", "large_boat": "#ff0000",
        "barge": "#ff00ff",
    }
    SIZE_MAP = {
        "buoy": 20, "small_boat": 35, "medium_boat": 55,
        "large_boat": 75, "barge": 90,
    }
    HEADING_LEN = {  # length of heading indicator line in meters
        "buoy": 0, "small_boat": 8, "medium_boat": 15,
        "large_boat": 25, "barge": 20,
    }

    # ── Figure layout ──────────────────────────────────────────
    fig = plt.figure(figsize=(11, 10), facecolor='#0a0a0a')
    fig.canvas.manager.set_window_title(
        f"Mode 12 Interactive — {loc['name']}")

    ax_ppi = fig.add_axes([0.08, 0.28, 0.84, 0.65], polar=True,
                          facecolor='#001a00')
    ax_ppi.set_theta_zero_location('N')
    ax_ppi.set_theta_direction(-1)
    ax_ppi.set_ylim(0, RANGE_M)
    ax_ppi.tick_params(colors='#00ff00', labelsize=7)
    ax_ppi.grid(True, color='#003300', linewidth=0.4)

    title_text = ax_ppi.set_title(
        f"{loc['name']}  ({loc['lat']:.4f}, {loc['lon']:.4f})",
        color='#00ff00', fontsize=10, pad=12)

    # Range rings with labels
    ring_fracs = [0.25, 0.5, 0.75, 1.0]
    for frac in ring_fracs:
        r = RANGE_M * frac
        theta_ring = np.linspace(0, 2 * np.pi, 120)
        ax_ppi.plot(theta_ring, [r] * 120, color='#003300', linewidth=0.4)
        ax_ppi.text(0.02, r, f"{r:.0f}m", fontsize=6, color='#005500',
                    ha='left', va='bottom')

    # ── Draw static coastlines ──────────────────────────────────
    n_coastline_lines = 0
    for cs in coastlines:
        pts = cs.points
        if len(pts) < 3:
            continue
        bearings, ranges = [], []
        for px, py in pts:
            ranges.append(math.sqrt(px * px + py * py))
            bearings.append(math.atan2(px, py))
        ax_ppi.fill(bearings, ranges, color='#00aa00', alpha=0.45)
        ax_ppi.plot(bearings + [bearings[0]], ranges + [ranges[0]],
                    color='#00ff00', linewidth=0.8, alpha=0.6)
        n_coastline_lines += 2
    n_static_lines = len(ax_ppi.lines)  # remember so we don't delete coastlines

    # ── Animated artists ────────────────────────────────────────
    # Object scatter (will be redrawn each frame)
    obj_scatter = ax_ppi.scatter([], [], s=[], c=[], zorder=5,
                                 edgecolors='white', linewidths=0.3)
    # Heading lines and trail lines are managed manually
    heading_lines = []
    trail_lines = []

    # Status text
    status_text = fig.text(0.50, 0.94,
                           "PAUSED | Frame 0 | 0.0 s | 0 rotations",
                           fontsize=9, color='#00cc00', ha='center',
                           fontfamily='monospace')

    def _xy_to_polar(x, y):
        return math.atan2(x, y), math.sqrt(x * x + y * y)

    def _update_frame(_frame_num):
        """Animation callback — advance objects and redraw."""
        # Advance physics if playing
        if state["playing"]:
            dt = DT_PER_FRAME * state["speed"]
            for i, obj in enumerate(objects):
                # Record trail before moving
                trails[i].append((obj.x, obj.y))
                obj.update_position(dt, coastlines, RANGE_M)
            state["elapsed"] += dt
            state["frame"] += 1

        # ── Clear dynamic artists ──
        # Remove old heading lines and trail lines
        while len(ax_ppi.lines) > n_static_lines:
            ax_ppi.lines[-1].remove()

        # ── Draw trails ──
        for i, obj in enumerate(objects):
            if not trails[i]:
                continue
            trail_pts = list(trails[i])
            n_trail = len(trail_pts)
            if n_trail < 2:
                continue
            c = COLOR_MAP.get(obj.obj_type, "#ffffff")
            # Draw trail with fading alpha (older = more transparent)
            for j in range(1, n_trail):
                alpha = 0.08 + 0.35 * (j / n_trail)
                b0, r0 = _xy_to_polar(*trail_pts[j - 1])
                b1, r1 = _xy_to_polar(*trail_pts[j])
                ax_ppi.plot([b0, b1], [r0, r1], color=c, alpha=alpha,
                            linewidth=0.8)
            # Trail from last trail point to current position
            b0, r0 = _xy_to_polar(*trail_pts[-1])
            b1, r1 = _xy_to_polar(obj.x, obj.y)
            ax_ppi.plot([b0, b1], [r0, r1], color=c, alpha=0.5, linewidth=0.8)

        # ── Draw objects + heading indicators ──
        bearings, ranges, sizes, colors = [], [], [], []
        for obj in objects:
            b, r = _xy_to_polar(obj.x, obj.y)
            bearings.append(b)
            ranges.append(r)
            colors.append(COLOR_MAP.get(obj.obj_type, "#ffffff"))
            sizes.append(SIZE_MAP.get(obj.obj_type, 25))

            # Heading indicator line
            hl = HEADING_LEN.get(obj.obj_type, 0)
            if hl > 0 and obj.is_moving:
                h_rad = math.radians(obj.heading_deg)
                tip_x = obj.x + hl * math.sin(h_rad)
                tip_y = obj.y + hl * math.cos(h_rad)
                b_tip, r_tip = _xy_to_polar(tip_x, tip_y)
                ax_ppi.plot([b, b_tip], [r, r_tip],
                            color=colors[-1], linewidth=1.5, alpha=0.9)

        obj_scatter.set_offsets(np.column_stack([bearings, ranges])
                                if bearings else np.empty((0, 2)))
        obj_scatter.set_sizes(sizes)
        obj_scatter.set_facecolors(colors)

        # ── Status text ──
        play_str = "PLAYING" if state["playing"] else "PAUSED"
        rots_done = state["elapsed"] / ROTATION_DT
        status_text.set_text(
            f"{play_str} | Frame {state['frame']} | "
            f"{state['elapsed']:.1f}s | {rots_done:.1f} rotations | "
            f"x{state['speed']:.1f}")

        return [obj_scatter, status_text]

    # ── Sliders ─────────────────────────────────────────────────
    sl_color = '#224422'
    ax_gain = fig.add_axes([0.15, 0.19, 0.55, 0.02])
    ax_noise = fig.add_axes([0.15, 0.15, 0.55, 0.02])
    ax_objs = fig.add_axes([0.15, 0.11, 0.55, 0.02])
    ax_rots = fig.add_axes([0.15, 0.07, 0.55, 0.02])
    ax_speed = fig.add_axes([0.15, 0.03, 0.55, 0.02])

    sl_gain = Slider(ax_gain, 'Gain', 30, 100,
                     valinit=state["gain"], valstep=5, color=sl_color)
    sl_noise = Slider(ax_noise, 'Noise', 0.0, 0.05,
                      valinit=state["noise_prob"], color=sl_color)
    sl_objs = Slider(ax_objs, 'Objects', 0, 30,
                     valinit=state["num_objects"], valstep=1, color=sl_color)
    sl_rots = Slider(ax_rots, 'Rotations', 1, 50,
                     valinit=state["rotations"], valstep=1, color=sl_color)
    sl_speed = Slider(ax_speed, 'Speed', 0.25, 10.0,
                      valinit=1.0, valstep=0.25, color=sl_color)

    for sl in (sl_gain, sl_noise, sl_objs, sl_rots, sl_speed):
        sl.label.set_color('#00cc00')
        sl.label.set_fontsize(8)
        sl.valtext.set_color('#00cc00')
        sl.valtext.set_fontsize(8)

    def _on_gain(v):  state["gain"] = int(v)
    def _on_noise(v): state["noise_prob"] = v
    def _on_objs(v):  state["num_objects"] = int(v)
    def _on_rots(v):  state["rotations"] = int(v)
    def _on_speed(v): state["speed"] = v

    sl_gain.on_changed(_on_gain)
    sl_noise.on_changed(_on_noise)
    sl_objs.on_changed(_on_objs)
    sl_rots.on_changed(_on_rots)
    sl_speed.on_changed(_on_speed)

    # ── Buttons ─────────────────────────────────────────────────
    btn_kw = dict(color='#1a331a', hovercolor='#2a4a2a')

    ax_play = fig.add_axes([0.78, 0.17, 0.18, 0.04])
    btn_play = Button(ax_play, 'Play / Pause', **btn_kw)
    btn_play.label.set_color('#00ff00')

    def on_play(event):
        state["playing"] = not state["playing"]
    btn_play.on_clicked(on_play)

    ax_step = fig.add_axes([0.78, 0.12, 0.18, 0.04])
    btn_step = Button(ax_step, 'Step (1 rot)', **btn_kw)
    btn_step.label.set_color('#00ff00')

    def on_step(event):
        state["playing"] = False
        dt = DT_PER_FRAME * state["speed"]
        for i, obj in enumerate(objects):
            trails[i].append((obj.x, obj.y))
            obj.update_position(dt, coastlines, RANGE_M)
        state["elapsed"] += dt
        state["frame"] += 1
    btn_step.on_clicked(on_step)

    ax_reset = fig.add_axes([0.78, 0.07, 0.18, 0.04])
    btn_reset = Button(ax_reset, 'Reset Positions', **btn_kw)
    btn_reset.label.set_color('#00ff00')

    def on_reset(event):
        state["playing"] = False
        state["elapsed"] = 0.0
        state["frame"] = 0
        for i, obj in enumerate(objects):
            src = original_objects[i]
            obj.x, obj.y = src.x, src.y
            obj.heading_deg = src.heading_deg
            obj.course_deg = src.course_deg
            obj.speed_kts = src.speed_kts
            trails[i].clear()
    btn_reset.on_clicked(on_reset)

    ax_regen = fig.add_axes([0.78, 0.02, 0.18, 0.04])
    btn_regen = Button(ax_regen, 'Regen Objects', **btn_kw)
    btn_regen.label.set_color('#00ff00')

    def on_regen(event):
        nonlocal objects, original_objects, trails
        state["playing"] = False
        state["elapsed"] = 0.0
        state["frame"] = 0
        n = state["num_objects"]
        n_buoy = max(0, n // 4)
        n_small = max(0, n // 4)
        n_med = max(0, n // 5)
        n_large = max(0, max(1, n - n_buoy - n_small - n_med - max(0, n // 8)))
        n_barge = max(0, n - n_buoy - n_small - n_med - n_large)
        regen_config = SceneConfig(
            num_buoys=(n_buoy, n_buoy),
            num_small_boats=(n_small, n_small),
            num_medium_boats=(n_med, n_med),
            num_large_boats=(n_large, n_large),
            num_barges=(n_barge, n_barge),
            gain=state["gain"],
        )
        new_objects = generate_random_objects(regen_config, RANGE_M, coastlines)
        objects = new_objects
        original_objects = [_copy.deepcopy(o) for o in new_objects]
        scene_data["objects"] = new_objects
        trails = [deque(maxlen=TRAIL_LENGTH) for _ in objects]
        print(f"  Regenerated {len(new_objects)} objects")
    btn_regen.on_clicked(on_regen)

    # ── Re-run Simulation button (bottom-left) ──
    ax_rerun = fig.add_axes([0.02, 0.005, 0.22, 0.02])
    btn_rerun = Button(ax_rerun, 'Re-run Simulation  >>  CSV', **btn_kw)
    btn_rerun.label.set_color('#ffcc00')
    btn_rerun.label.set_fontsize(8)

    def on_rerun(event):
        scene["radar_config"]["gain"] = state["gain"]
        scene["clutter"]["noise_probability"] = state["noise_prob"]
        cur = scene_data["objects"]
        scene["objects"] = [
            {"type": o.obj_type, "x": o.x, "y": o.y,
             "rcs_m2": o.rcs_m2, "length_m": o.length_m, "width_m": o.width_m,
             "heading_deg": o.heading_deg, "speed_kts": o.speed_kts,
             "course_deg": o.course_deg, "turn_rate_dps": o.turn_rate_dps}
            for o in cur
        ]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rerun_dir = os.path.join(output_dir, f"rerun_{timestamp}")
        base_time = datetime.now() - timedelta(
            days=random.randint(0, 30), hours=random.randint(0, 23),
            minutes=random.randint(0, 59))
        print(f"\n--- Re-running simulation -> {rerun_dir} ---")
        files = simulate_from_scene(scene, coastlines, cur, rerun_dir,
                                     scene_path="mode12_rerun",
                                     num_rotations=state["rotations"],
                                     verbose=verbose, base_time=base_time)
        scene_json_path = os.path.join(rerun_dir, "mode12_scene.json")
        with open(scene_json_path, 'w') as f:
            json.dump(scene, f, indent=2)
        print(f"  Re-run complete: {len(files)} CSV files -> {rerun_dir}")
        # Auto-open the output folder
        _open_folder(rerun_dir)
    btn_rerun.on_clicked(on_rerun)

    # ── Legend ───────────────────────────────────────────────────
    legend_y = 0.265
    for obj_type, color in COLOR_MAP.items():
        fig.text(0.02, legend_y, "\u25cf", color=color, fontsize=10,
                 fontfamily='monospace')
        fig.text(0.04, legend_y, obj_type.replace("_", " "),
                 color='#888888', fontsize=7)
        legend_y -= 0.015

    fig.text(0.20, 0.265, "Space=Play/Pause  R=Reset  S=Step  +/-=Speed",
             fontsize=7, color='#555555')

    # ── Keyboard shortcuts ──────────────────────────────────────
    def on_key(event):
        if event.key == ' ':
            on_play(None)
        elif event.key == 'r':
            on_reset(None)
        elif event.key == 's':
            on_step(None)
        elif event.key == '+' or event.key == '=':
            new_spd = min(10.0, state["speed"] + 0.5)
            sl_speed.set_val(new_spd)
        elif event.key == '-':
            new_spd = max(0.25, state["speed"] - 0.5)
            sl_speed.set_val(new_spd)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # ── Start animation ─────────────────────────────────────────
    anim = FuncAnimation(fig, _update_frame, interval=80, blit=False,
                         cache_frame_data=False)

    plt.show()


def _open_folder(path):
    """Open a folder in the system file browser."""
    import subprocess, platform
    path = os.path.abspath(path)
    if platform.system() == 'Windows':
        os.startfile(path)
    elif platform.system() == 'Darwin':
        subprocess.Popen(['open', path])
    else:
        # WSL: try Windows explorer, fall back to xdg-open
        try:
            # Convert WSL path to Windows path
            win_path = subprocess.check_output(
                ['wslpath', '-w', path], text=True).strip()
            subprocess.Popen(['explorer.exe', win_path])
        except Exception:
            try:
                subprocess.Popen(['xdg-open', path])
            except Exception:
                pass


# ============================================================
# YOLO Training Data Pipeline
# ============================================================

# Class mapping — matches terrain-occluded simulator for compatibility,
# plus Object Creation's native types mapped into the same scheme
YOLO_CLASS_MAP = {
    'buoy': 0,
    'small_boat': 1,
    'medium_boat': 2,
    'large_boat': 3,
    'barge': 4,
    # Aliases for terrain-occluded simulator compatibility
    'fishing': 1,
    'sailing': 1,
    'pilot': 2,
    'tug': 2,
    'cargo': 3,
    'tanker': 3,
    'passenger': 3,
    'unknown': 2,
}

YOLO_CLASS_NAMES = ['buoy', 'small_boat', 'medium_boat', 'large_boat', 'barge']


def render_ppi_image(csv_path: str, image_size: int = 512) -> np.ndarray:
    """Render a Furuno CSV file to a square PPI image (numpy array).

    Converts the polar radar data to a Cartesian green-on-black image
    suitable for YOLO training.

    Args:
        csv_path: Path to Furuno-format CSV file
        image_size: Output image size in pixels (square)

    Returns:
        numpy array of shape (image_size, image_size, 3) uint8 (BGR for OpenCV compat)
    """
    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    cx = cy = image_size // 2
    r_max = image_size // 2 - 2

    with open(csv_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line[0].isalpha():
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            angle_tick = int(parts[4])
            echoes = parts[5:]
            n_bins = len(echoes)
            bearing_rad = angle_tick * 2.0 * math.pi / TICKS_PER_REV
            sin_b = math.sin(bearing_rad)
            cos_b = -math.cos(bearing_rad)  # screen Y is inverted

            for b_idx, echo_str in enumerate(echoes):
                val = int(echo_str)
                if val <= 0:
                    continue
                r = (b_idx / n_bins) * r_max
                px = int(cx + sin_b * r)
                py = int(cy + cos_b * r)
                if 0 <= px < image_size and 0 <= py < image_size:
                    intensity = min(255, val)
                    # Green channel primary, slight in red for high echoes
                    g = intensity
                    r_val = max(0, intensity - 180)
                    img[py, px] = [0, g, r_val]
    return img


def render_ppi_from_grid(grid: np.ndarray, image_size: int = 512) -> np.ndarray:
    """Render a polar grid (n_ticks x n_bins) to a square PPI image."""
    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    cx = cy = image_size // 2
    r_max = image_size // 2 - 2
    n_ticks, n_bins = grid.shape

    for tick in range(n_ticks):
        bearing_rad = tick * 2.0 * math.pi / n_ticks
        sin_b = math.sin(bearing_rad)
        cos_b = -math.cos(bearing_rad)
        for b in range(n_bins):
            val = grid[tick, b]
            if val > 0:
                r = (b / n_bins) * r_max
                px = int(cx + sin_b * r)
                py = int(cy + cos_b * r)
                if 0 <= px < image_size and 0 <= py < image_size:
                    intensity = min(255, int(val))
                    img[py, px] = [0, intensity, max(0, intensity - 180)]
    return img


def generate_yolo_labels(objects: List[RadarObject], image_size: int = 512,
                          range_m: float = RANGE_M) -> List[str]:
    """Generate YOLO-format label lines from actual radar echo physics.

    Bounding boxes are derived from how the DRS4DNXT actually renders each
    object, not from its physical size:

      Angular width  = beamwidth (3.9 deg) at the object's range on the PPI.
                       The beam illuminates the target over this angular span,
                       so the echo paints across that many pixels tangentially.

      Range depth    = spread_cap * 2 bins, converted to PPI pixels.
                       The radar echo spreads across this many range bins via
                       the Gaussian profile. spread_cap is per-type:
                       buoy=2, small_boat=5, medium_boat=8, large_boat=12, barge=15.

      Padding        = +2px each side to avoid clipping the echo edge.

    This produces tight boxes that closely wrap the actual green pixels on
    the PPI, not oversized rectangles based on hull dimensions.

    Args:
        objects: List of RadarObject instances with current positions
        image_size: PPI image size in pixels
        range_m: Radar range in meters

    Returns:
        List of YOLO label strings
    """
    half = image_size / 2.0
    r_max = half - 2
    pixels_per_meter = r_max / range_m
    pixels_per_bin = r_max / NUM_BINS

    # Per-type spread caps (from the rendering engine, line ~3224)
    SPREAD_CAP = {"buoy": 2, "small_boat": 5, "medium_boat": 8,
                  "large_boat": 12, "barge": 15}

    # Minimum bbox size in pixels (so tiny objects are still clickable)
    MIN_BOX_PX = 8
    # Padding around the echo footprint
    PAD_PX = 3

    lines = []

    for obj in objects:
        obj_range = math.sqrt(obj.x ** 2 + obj.y ** 2)
        if obj_range > range_m or obj_range < 1.0:
            continue

        class_id = YOLO_CLASS_MAP.get(obj.obj_type, 2)

        # ── Object center on PPI ──
        bearing_rad = math.atan2(obj.x, obj.y)
        range_ratio = obj_range / range_m
        px = half + math.sin(bearing_rad) * range_ratio * r_max
        py = half - math.cos(bearing_rad) * range_ratio * r_max

        # ── Angular width (tangential on PPI) ──
        # The beam illuminates ±(BEAMWIDTH/2) degrees around the object.
        # At the object's range on the PPI, this angular span maps to pixels:
        #   arc_length_px = beamwidth_rad * range_on_ppi_px
        range_px = range_ratio * r_max
        beam_arc_px = math.radians(BEAMWIDTH_DEG) * range_px
        # Very close objects have tiny arc — ensure minimum visibility
        bbox_w = max(MIN_BOX_PX, beam_arc_px + PAD_PX * 2)

        # ── Range depth (radial on PPI) ──
        # The echo spreads ± spread_cap bins from center, rendered as
        # a Gaussian. Total echo footprint = spread_cap * 4 bins
        # (the loop paints from center_bin - 2*spread to center_bin + 2*spread)
        spread_cap = SPREAD_CAP.get(obj.obj_type, 5)
        radial_extent = obj.length_m * abs(math.cos(bearing_rad - math.radians(obj.heading_deg))) + \
                        obj.width_m * abs(math.sin(bearing_rad - math.radians(obj.heading_deg)))
        actual_spread = max(2, min(spread_cap, int(radial_extent / BIN_SIZE_M / 6)))
        spread_depth_px = actual_spread * 4 * pixels_per_bin
        bbox_h = max(MIN_BOX_PX, spread_depth_px + PAD_PX * 2)

        # ── Normalize to 0-1 ──
        ncx = max(0.0, min(1.0, px / image_size))
        ncy = max(0.0, min(1.0, py / image_size))
        nw = max(0.001, min(1.0, bbox_w / image_size))
        nh = max(0.001, min(1.0, bbox_h / image_size))

        lines.append(f"{class_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")

    return lines


def write_yolo_labels(filepath: str, label_lines: List[str]) -> str:
    """Write YOLO label lines to a file."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        f.write('\n'.join(label_lines))
        if label_lines:
            f.write('\n')
    return filepath


def write_dataset_yaml(dataset_dir: str) -> str:
    """Write a YOLOv8-compatible dataset.yaml file.

    Args:
        dataset_dir: Root of the dataset directory (contains images/, labels/)

    Returns:
        Path to the written yaml file
    """
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    content = f"""# Furuno DRS4DNXT Radar Object Detection Dataset
# Generated by Object Creation Simulator Mode 12
# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val

nc: {len(YOLO_CLASS_NAMES)}
names: {YOLO_CLASS_NAMES}
"""
    with open(yaml_path, 'w') as f:
        f.write(content)
    return yaml_path


def generate_training_data(output_dir: str, location: str = None,
                           radarloc_file: str = None, range_nm: float = None,
                           num_scenes: int = 10, rotations_per_scene: int = 10,
                           image_size: int = 512, val_split: float = 0.2,
                           waypoints: List[Tuple[float, float]] = None,
                           waypoint_obj_type: str = "medium_boat",
                           waypoint_speed_kts: float = 8.0,
                           verbose: bool = True) -> str:
    """Generate a complete YOLO training dataset from Mode 12 scenarios.

    Creates the standard YOLOv8 directory structure:
        output_dir/
            dataset.yaml
            images/
                train/  scene_001_rot_001.png ...
                val/    scene_008_rot_001.png ...
            labels/
                train/  scene_001_rot_001.txt ...
                val/    scene_008_rot_001.txt ...

    Each scene uses the same real-world coastline but randomized objects.
    If waypoints are provided, one "own-ship" object follows the waypoint
    path through each scene (simulating harbor approach etc.).

    Args:
        output_dir: Root output directory for the dataset
        location: Location string for Mode 12 ('lat,lon' or place name)
        radarloc_file: Optional .radarloc file path
        range_nm: Override radar range in NM
        num_scenes: Number of different object configurations
        rotations_per_scene: Rotations (frames) per scene
        image_size: PPI image size in pixels
        val_split: Fraction of scenes for validation (default 0.2)
        waypoints: Optional list of (x, y) waypoints for a scripted vessel
        waypoint_obj_type: Type of the waypoint-following vessel
        waypoint_speed_kts: Speed of the waypoint-following vessel
        verbose: Print progress

    Returns:
        Path to the generated dataset.yaml
    """
    print("=" * 60)
    print("  YOLO Training Data Generator — Mode 12")
    print("=" * 60)

    # Create directory structure
    for split in ('train', 'val'):
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    library = build_coastline_library(verbose=False)
    prep = _prepare_mode12_coastlines(
        location=location, range_nm=range_nm, radarloc_file=radarloc_file,
        library=library, verbose=verbose)
    if prep is None:
        return None

    coastlines = prep["coastlines"]
    loc = prep["loc"]
    echo_stats_map = prep["echo_stats_map"]
    sim_range_m = prep["sim_range_m"]

    if verbose:
        print(f"  Location: {loc['name']}")
        print(f"  Coastline segments: {len(coastlines)}")
        print(f"  Scenes: {num_scenes}, Rotations/scene: {rotations_per_scene}")
        print(f"  Image size: {image_size}x{image_size}")
        if waypoints:
            print(f"  Waypoint vessel: {waypoint_obj_type} at {waypoint_speed_kts} kts, "
                  f"{len(waypoints)} waypoints")

    n_val = max(1, int(num_scenes * val_split))
    n_train = num_scenes - n_val
    total_images = 0

    try:
        from PIL import Image as _PILImage
        _has_pil = True
    except ImportError:
        _has_pil = False
        print("  WARNING: PIL not available. Saving PPI as raw numpy (.npy)")
        print("  Install with: pip install Pillow")

    for scene_idx in range(num_scenes):
        split = 'train' if scene_idx < n_train else 'val'
        if verbose:
            print(f"\n  Scene {scene_idx + 1}/{num_scenes} [{split}]", end="")

        # Generate random objects
        config = SceneConfig(
            num_buoys=(2, random.randint(3, 6)),
            num_small_boats=(2, random.randint(3, 5)),
            num_medium_boats=(1, random.randint(2, 4)),
            num_large_boats=(1, random.randint(1, 3)),
            num_barges=(0, random.randint(0, 2)),
            gain=random.choice([60, 65, 70, 75, 80]),
        )
        objects = generate_random_objects(config, sim_range_m, coastlines)

        # Add waypoint-following vessel if requested
        if waypoints:
            wp_obj = RadarObject(
                x=waypoints[0][0], y=waypoints[0][1],
                obj_type=waypoint_obj_type,
                rcs_m2=random.uniform(200, 2000),
                length_m=random.uniform(15, 40),
                width_m=random.uniform(5, 12),
                heading_deg=0.0,
                speed_kts=waypoint_speed_kts,
                course_deg=0.0,
                label="waypoint_vessel",
                waypoints=list(waypoints[1:]),
                waypoint_radius_m=5.0,
            )
            objects.append(wp_obj)

        if verbose:
            print(f" — {len(objects)} objects")

        scene = _build_mode12_scene(
            config, coastlines, objects, sim_range_m, RANGE_NM,
            echo_stats_map=echo_stats_map)

        # Simulate — generate CSV files
        scene_tmp_dir = os.path.join(output_dir, '_tmp_csv',
                                      f'scene_{scene_idx + 1:03d}')
        base_time = datetime.now() - timedelta(
            days=random.randint(0, 30), hours=random.randint(0, 23),
            minutes=random.randint(0, 59))

        all_output_files = simulate_from_scene(
            scene, coastlines, objects, scene_tmp_dir,
            scene_path="training_scene",
            num_rotations=rotations_per_scene,
            verbose=False, base_time=base_time)
        # Filter to only main radar CSVs (not detections or metadata)
        csv_files = [f for f in all_output_files
                     if f.endswith('.csv') and 'detection' not in f
                     and 'metadata' not in f]

        # Render each rotation to PPI image + YOLO labels
        for rot_idx, csv_path in enumerate(csv_files):
            stem = f"scene_{scene_idx + 1:03d}_rot_{rot_idx + 1:03d}"

            # Render PPI image
            ppi = render_ppi_image(csv_path, image_size)
            img_dir = os.path.join(output_dir, 'images', split)
            if _has_pil:
                pil_img = _PILImage.fromarray(ppi)
                img_path = os.path.join(img_dir, f"{stem}.png")
                pil_img.save(img_path)
            else:
                img_path = os.path.join(img_dir, f"{stem}.npy")
                np.save(img_path, ppi)

            # Generate and write YOLO labels
            labels = generate_yolo_labels(objects, image_size, sim_range_m)
            lbl_path = os.path.join(output_dir, 'labels', split, f"{stem}.txt")
            write_yolo_labels(lbl_path, labels)

            total_images += 1

            # Advance object positions for next rotation
            for obj in objects:
                obj.update_position(ROTATION_DT, coastlines, sim_range_m)

    # Write dataset.yaml
    yaml_path = write_dataset_yaml(output_dir)

    # Clean up temp CSVs (optional — keep them for debugging)
    # shutil.rmtree(os.path.join(output_dir, '_tmp_csv'), ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"  Training data generation complete")
    print(f"  Total images: {total_images}")
    print(f"  Train: {n_train * rotations_per_scene} images")
    print(f"  Val:   {n_val * rotations_per_scene} images")
    print(f"  Classes: {YOLO_CLASS_NAMES}")
    print(f"  Dataset YAML: {yaml_path}")
    print(f"{'=' * 60}")
    print(f"\n  To train YOLOv8:")
    print(f"    pip install ultralytics")
    print(f"    yolo detect train data={yaml_path} model=yolov8n.pt epochs=100 imgsz={image_size}")

    return yaml_path


# ============================================================
# Inline CSV Loader (for validation — avoids circular import)
# ============================================================
def _load_csv_for_validation(filepath):
    """Minimal CSV loader for validation. Returns list of spoke dicts."""
    spokes = []
    header_skipped = False
    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not header_skipped:
                if line[0].isalpha():
                    header_skipped = True
                    continue
                header_skipped = True
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                angle_tick = int(parts[4])
                echoes = [int(v) for v in parts[5:] if v.strip()]
            except ValueError:
                continue
            spokes.append({"angle_tick": angle_tick, "echoes": echoes})
    return spokes


def _build_validation_bin_sets_by_spoke(payload):
    """Convert extract_scene-style segment payloads into per-spoke bin sets."""
    bins_by_idx = {}
    for spoke_idx_str, segments in (payload or {}).items():
        bins_set = set()
        for seg in segments:
            start_bin = seg[0]
            vals = seg[1]
            for vi in range(len(vals)):
                bins_set.add(start_bin + vi)
        bins_by_idx[int(spoke_idx_str)] = bins_set
    return bins_by_idx


def _build_validation_region_masks_from_bin_sets(real_grid, scene_dead_zone,
                                                 coast_bins_by_idx, obj_bins_by_idx):
    """Build coastline/object/noise/empty/dead masks over an aligned grid."""
    n_spokes, n_bins = real_grid.shape
    coast = np.zeros((n_spokes, n_bins), dtype=bool)
    obj = np.zeros((n_spokes, n_bins), dtype=bool)
    dead = np.zeros((n_spokes, n_bins), dtype=bool)

    if scene_dead_zone > 0:
        dead[:, :min(scene_dead_zone, n_bins)] = True

    for si in range(n_spokes):
        coast_set = coast_bins_by_idx.get(si, set())
        obj_set = obj_bins_by_idx.get(si, set())
        for b in coast_set:
            if 0 <= b < n_bins:
                coast[si, b] = True
        for b in obj_set:
            if 0 <= b < n_bins:
                obj[si, b] = True

    noise = (~dead) & (~coast) & (~obj) & (real_grid > 0)
    empty = (~dead) & (~coast) & (~obj) & (~noise)
    return {
        "coastline": coast,
        "objects": obj,
        "noise": noise,
        "empty": empty,
        "dead_zone": dead,
    }


def _display_db_from_codes(code_grid):
    """Map Furuno echo codes into display-space dB values."""
    return 20.0 * np.log10((code_grid.astype(np.float64) + 1.0) / 252.0)


def _compute_masked_display_db_mae(real_grid, sim_grid, region_mask):
    """Mean absolute display-space dB error on bins where either grid is active."""
    valid = region_mask & ((real_grid > 0) | (sim_grid > 0))
    if not np.any(valid):
        return 0.0
    real_db = _display_db_from_codes(real_grid)
    sim_db = _display_db_from_codes(sim_grid)
    return float(np.mean(np.abs(real_db[valid] - sim_db[valid])))


def _global_ssim_numpy(real_norm, sim_norm):
    """Dependency-light whole-image SSIM over normalized 0-1 grids."""
    x = real_norm.astype(np.float64).ravel()
    y = sim_norm.astype(np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return None
    luma = 1.0
    c1 = (0.01 * luma) ** 2
    c2 = (0.03 * luma) ** 2
    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
    denom = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    if denom == 0:
        return 1.0 if np.allclose(x, y) else 0.0
    ssim_val = ((2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)) / denom
    return float(max(-1.0, min(1.0, ssim_val)))


def _stride_limit_points(points, limit):
    """Deterministically subsample point arrays to keep KD-tree work bounded."""
    if len(points) <= limit:
        return points
    stride = max(1, int(math.ceil(len(points) / float(limit))))
    return points[::stride][:limit]


def _points_from_active_bins(active_mask, angle_ticks, bin_size_m):
    """Convert active spoke/bin cells into Cartesian point coordinates in meters."""
    idx = np.argwhere(active_mask)
    if idx.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    spoke_idx = idx[:, 0]
    bin_idx = idx[:, 1]
    angles = (np.asarray(angle_ticks, dtype=np.float64)[spoke_idx] / TICKS_PER_REV) * (2.0 * math.pi)
    ranges_m = bin_idx.astype(np.float64) * float(bin_size_m)
    xs = ranges_m * np.sin(angles)
    ys = ranges_m * np.cos(angles)
    return np.column_stack((xs, ys))


def _compute_active_point_mae(real_grid, sim_grid, not_dead_mask, angle_ticks,
                              bin_size_m, max_points=150000):
    """Symmetric nearest-neighbor MAE over all active non-dead bins."""
    threshold = 8
    real_points = _points_from_active_bins(not_dead_mask & (real_grid >= threshold), angle_ticks, bin_size_m)
    sim_points = _points_from_active_bins(not_dead_mask & (sim_grid >= threshold), angle_ticks, bin_size_m)

    if len(real_points) == 0 and len(sim_points) == 0:
        return 0.0, 0.0

    max_distance_m = float(real_grid.shape[1]) * float(bin_size_m)
    if len(real_points) == 0 or len(sim_points) == 0:
        return max_distance_m, (max_distance_m / float(bin_size_m)) if bin_size_m > 0 else None

    if not _HAS_SCIPY_VALIDATION:
        return None, None

    real_points = _stride_limit_points(real_points, max_points)
    sim_points = _stride_limit_points(sim_points, max_points)

    real_tree = _cKDTree(real_points)
    sim_tree = _cKDTree(sim_points)
    sim_to_real = real_tree.query(sim_points, k=1)[0]
    real_to_sim = sim_tree.query(real_points, k=1)[0]
    mae_m = float((np.mean(sim_to_real) + np.mean(real_to_sim)) / 2.0)
    mae_bins = (mae_m / float(bin_size_m)) if bin_size_m > 0 else None
    return mae_m, mae_bins


def _compute_noise_overlap_metrics(real_grid, sim_grid, coast_mask, obj_mask, dead_mask):
    """Compute overlap and distribution metrics for noise-only bins."""
    base_noise_region = (~dead_mask) & (~coast_mask) & (~obj_mask)
    active_threshold = 8
    real_noise_active = base_noise_region & (real_grid >= active_threshold)
    sim_noise_active = base_noise_region & (sim_grid >= active_threshold)

    union = real_noise_active | sim_noise_active
    inter = real_noise_active & sim_noise_active
    if not np.any(union):
        noise_iou = 1.0
        noise_jaccard = 1.0
    else:
        ratio = float(np.count_nonzero(inter)) / float(np.count_nonzero(union))
        noise_iou = ratio
        noise_jaccard = ratio

    real_noise_codes = real_grid[real_noise_active]
    sim_noise_codes = sim_grid[sim_noise_active]
    if real_noise_codes.size == 0 and sim_noise_codes.size == 0:
        ks_d = 0.0
        ks_p = 1.0
    elif real_noise_codes.size == 0 or sim_noise_codes.size == 0:
        ks_d = 1.0
        ks_p = 0.0
    elif _HAS_SCIPY_VALIDATION:
        ks = _ks_2samp(real_noise_codes.astype(np.float64), sim_noise_codes.astype(np.float64))
        ks_d = float(ks.statistic)
        ks_p = float(ks.pvalue)
    else:
        ks_d = None
        ks_p = None

    return {
        "noise_iou": noise_iou,
        "noise_jaccard": noise_jaccard,
        "noise_ks_d": ks_d,
        "noise_ks_p": ks_p,
    }


def _compute_advanced_validation_metrics(real_grid, sim_grid, masks, angle_ticks, bin_size_m):
    """Compute richer validation metrics without changing legacy exact/tolerant counts."""
    coast_mask = np.asarray(masks.get("coastline"), dtype=bool)
    obj_mask = np.asarray(masks.get("objects", masks.get("object")), dtype=bool)
    dead_mask = np.asarray(masks.get("dead_zone"), dtype=bool)
    noise_mask = np.asarray(masks.get("noise"), dtype=bool)
    not_dead = ~dead_mask

    real_grid = np.asarray(real_grid, dtype=np.int32)
    sim_grid = np.asarray(sim_grid, dtype=np.int32)
    real_norm = np.clip(real_grid.astype(np.float64) / 252.0, 0.0, 1.0)
    sim_norm = np.clip(sim_grid.astype(np.float64) / 252.0, 0.0, 1.0)

    active_point_mae_m, active_point_mae_bins = _compute_active_point_mae(
        real_grid, sim_grid, not_dead, angle_ticks, bin_size_m)

    metrics = {
        "dynamic_range_mae_db_overall": _compute_masked_display_db_mae(real_grid, sim_grid, not_dead),
        "dynamic_range_mae_db_coastline": _compute_masked_display_db_mae(real_grid, sim_grid, coast_mask),
        "dynamic_range_mae_db_objects": _compute_masked_display_db_mae(real_grid, sim_grid, obj_mask),
        "dynamic_range_mae_db_noise": _compute_masked_display_db_mae(real_grid, sim_grid, noise_mask),
        "active_point_mae_m": active_point_mae_m,
        "active_point_mae_bins": active_point_mae_bins,
        "ppi_ssim": _global_ssim_numpy(real_norm, sim_norm),
    }
    metrics.update(_compute_noise_overlap_metrics(real_grid, sim_grid, coast_mask, obj_mask, dead_mask))
    return metrics


# ============================================================
# Built-in Validation
# ============================================================
def validate_against_source(scene, sim_rows, verbose=True):
    """Validate simulated output against the original Furuno CSV capture.

    Uses source_file_path from the scene to load the original data.
    Matches by spoke INDEX (position), not tick value, to handle duplicate ticks.
    Classifies each bin as coastline/object/noise/empty/dead_zone and
    reports per-component accuracy.
    """
    source_path = scene.get("source_file_path", "")
    if not source_path or not os.path.isfile(source_path):
        if verbose:
            print(f"  Validation skipped: source file not found ({source_path})")
        return None

    source_file = os.path.basename(source_path)
    scene_bins = scene.get("radar_config", {}).get("bins", NUM_BINS)
    scene_dead_zone = scene.get("radar_config", {}).get("dead_zone_bins", DEAD_ZONE_BINS)
    furuno_codes = [0, 8, 20, 28, 36, 44, 56, 64, 76, 88,
                    104, 120, 140, 164, 188, 220, 252]

    # Load original spokes
    orig_spokes = _load_csv_for_validation(source_path)
    if not orig_spokes:
        if verbose:
            print("  Validation skipped: could not load original CSV")
        return None

    # Split into rotations and use the first
    rotations = []
    cur = [orig_spokes[0]]
    for i in range(1, len(orig_spokes)):
        if orig_spokes[i]["angle_tick"] < orig_spokes[i - 1]["angle_tick"] - 100:
            rotations.append(cur)
            cur = []
        cur.append(orig_spokes[i])
    if cur:
        rotations.append(cur)
    orig_rot = rotations[0]

    coast_bins_by_idx = _build_validation_bin_sets_by_spoke(scene.get("coastline_echo_data", {}))
    obj_bins_by_idx = _build_validation_bin_sets_by_spoke(scene.get("object_echo_data", {}))

    # Compare by spoke index (position), not tick value
    num_spokes = min(len(orig_rot), len(sim_rows))
    stats = {
        "overall": {"exact": 0, "tolerant": 0, "total": 0},
        "coastline": {"exact": 0, "tolerant": 0, "total": 0},
        "objects": {"exact": 0, "tolerant": 0, "total": 0},
        "noise": {"exact": 0, "tolerant": 0, "total": 0},
        "empty": {"exact": 0, "tolerant": 0, "total": 0},
        "dead_zone": {"exact": 0, "tolerant": 0, "total": 0},
    }
    aligned_real = np.zeros((num_spokes, scene_bins), dtype=np.int32)
    aligned_sim = np.zeros((num_spokes, scene_bins), dtype=np.int32)
    aligned_angle_ticks = []

    for si in range(num_spokes):
        orig_e = orig_rot[si]["echoes"]
        sim_e = sim_rows[si]["echoes"]
        aligned_angle_ticks.append(int(orig_rot[si]["angle_tick"]))
        coast_set = coast_bins_by_idx.get(si, set())
        obj_set = obj_bins_by_idx.get(si, set())

        limit = min(scene_bins, len(orig_e), len(sim_e))
        if limit > 0:
            aligned_real[si, :limit] = np.asarray(orig_e[:limit], dtype=np.int32)
            aligned_sim[si, :limit] = np.asarray(sim_e[:limit], dtype=np.int32)

        for b in range(limit):
            ov = orig_e[b]
            sv = sim_e[b]

            # Classify bin
            if b < scene_dead_zone:
                cat = "dead_zone"
            elif b in coast_set:
                cat = "coastline"
            elif b in obj_set:
                cat = "objects"
            elif ov > 0:
                cat = "noise"
            else:
                cat = "empty"

            exact = (ov == sv)
            tolerant = exact
            if not exact and ov in furuno_codes and sv in furuno_codes:
                oi = furuno_codes.index(ov)
                svi = furuno_codes.index(sv)
                if abs(oi - svi) <= 1:
                    tolerant = True

            stats["overall"]["total"] += 1
            stats[cat]["total"] += 1
            if exact:
                stats["overall"]["exact"] += 1
                stats[cat]["exact"] += 1
            if tolerant:
                stats["overall"]["tolerant"] += 1
                stats[cat]["tolerant"] += 1

    masks = _build_validation_region_masks_from_bin_sets(
        aligned_real, scene_dead_zone, coast_bins_by_idx, obj_bins_by_idx)
    radar_cfg = scene.get("radar_config", {}) or {}
    bin_size_m = float(radar_cfg.get("bin_size_m") or 0.0)
    if bin_size_m <= 0:
        range_m = float(radar_cfg.get("range_m") or 0.0)
        bin_size_m = (range_m / float(scene_bins)) if range_m > 0 and scene_bins > 0 else BIN_SIZE_M
    stats["advanced_metrics"] = _compute_advanced_validation_metrics(
        aligned_real, aligned_sim, masks, aligned_angle_ticks, bin_size_m)
    if scene.get("_strict_support_report"):
        stats["support_report"] = scene.get("_strict_support_report")

    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATION: Simulated vs Original ({source_file})")
        print(f"{'='*60}")
        print(f"  {'Component':<14} {'Exact':>8} {'Tolerant':>10} {'Bins':>12}")
        print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*12}")
        for cat in ["overall", "coastline", "objects", "noise", "empty", "dead_zone"]:
            s = stats[cat]
            t = max(s["total"], 1)
            ex_pct = 100.0 * s["exact"] / t
            tol_pct = 100.0 * s["tolerant"] / t
            cat_name = cat.capitalize() if cat != "dead_zone" else "Dead zone"
            print(f"  {cat_name:<14} {ex_pct:>7.2f}% {tol_pct:>9.2f}% {s['total']:>11,}")
        adv = stats.get("advanced_metrics", {})
        print("  Advanced metrics")
        print(f"    Display-space dynamic range MAE (overall): {adv.get('dynamic_range_mae_db_overall', 0.0):.4f} dB")
        print(f"    Noise IoU/Jaccard: {adv.get('noise_iou', 0.0):.4f} / {adv.get('noise_jaccard', 0.0):.4f}")
        print(f"    Noise KS (D, p): {adv.get('noise_ks_d', 0.0):.4f}, {adv.get('noise_ks_p', 0.0):.4g}")
        print(f"    Active-point MAE: {adv.get('active_point_mae_m', 0.0):.3f} m ({adv.get('active_point_mae_bins', 0.0):.3f} bins)")
        print(f"    PPI SSIM: {adv.get('ppi_ssim', 0.0):.5f}")
        support_report = stats.get("support_report", {})
        if support_report:
            coast = support_report.get("coastline", {})
            objs = support_report.get("objects", {})
            noise = support_report.get("noise", {})
            dz = support_report.get("dead_zone", {})
            print("  Strict support report")
            print(
                "    Coastline: "
                f"ref={coast.get('reference_segments', 0)}, "
                f"geom-matched={coast.get('geometry_matched_segments', 0)}, "
                f"geom-unmatched={coast.get('geometry_unmatched_segments', 0)}, "
                f"fallback={coast.get('geometry_fallback_segments', 0)}, "
                f"clipped={coast.get('clipped_segments', 0)}"
            )
            print(
                "    Objects: "
                f"support-objects={objs.get('preserved_support_objects', 0)}, "
                f"support-segments={objs.get('preserved_support_segments', 0)}, "
                f"legacy-fallback={objs.get('legacy_geometry_fallback_objects', 0)}"
            )
            print(
                "    Noise: "
                f"support-spokes={noise.get('preserved_support_spokes', 0)}, "
                f"support-segments={noise.get('preserved_support_segments', 0)}, "
                f"legacy-structured={noise.get('legacy_structured_spokes', 0)}"
            )
            print(
                "    Dead zone: "
                f"support-spokes={dz.get('preserved_support_spokes', 0)}, "
                f"support-segments={dz.get('preserved_support_segments', 0)}, "
                f"legacy-fallback={dz.get('legacy_probability_fallback_spokes', 0)}"
            )
        print(f"{'='*60}")

    return stats


# ============================================================
# Scene File Loading — load extracted scene JSON from extract_scene.py
# ============================================================
def load_scene_from_file(filepath: str):
    """Load scene definition from a JSON file produced by extract_scene.py.

    Returns (coastlines, objects, scene_dict).
    """
    with open(filepath, 'r') as f:
        scene = json.load(f)

    coastlines = []
    for c_data in scene.get('coastlines', []):
        points = [(p[0], p[1]) for p in c_data['points']]
        coastlines.append(CoastlineSegment(points=points,
                                           name=c_data.get('name', 'coastline')))

    objects = []
    for o_data in scene.get('objects', []):
        obj = RadarObject(
            x=o_data['x'],
            y=o_data['y'],
            obj_type=o_data['type'],
            rcs_m2=o_data['rcs_m2'],
            length_m=o_data['length_m'],
            width_m=o_data['width_m'],
            heading_deg=o_data['heading_deg'],
            speed_kts=o_data.get('speed_kts', 0.0),
            course_deg=o_data.get('course_deg', 0.0),
        )
        objects.append(obj)

    return coastlines, objects, scene


def _quantize_to_furuno(val):
    """Quantize a value to the nearest Furuno gain code.
    Skip re-quantization if already a valid code."""
    if val <= 0:
        return 0
    ival = int(round(val))
    if ival in FURUNO_GAIN_SET:
        return ival
    idx = np.argmin(np.abs(FURUNO_GAIN_CODES - val))
    return int(FURUNO_GAIN_CODES[idx])


# ============================================================
# Mode 4: Reference-Calibrated Simulation — Calibration Extraction
# ============================================================

def _extract_coastline_calibration(scene: dict) -> dict:
    """Extract per-region coastline calibration from scene reference data.

    Returns dict keyed by region name with:
      mean_core_echo, median_ramp_up, median_ramp_down
    Plus spoke-level lookups from coastline_profiles so strict Mode 4 can
    keep the real segment placement while using local per-segment shaping
    instead of one region-wide median profile.
    """
    coastline_stats = scene.get('coastline_stats', {})
    coastline_profiles = scene.get('coastline_profiles', {})

    regions = {}
    for region_name, stats in coastline_stats.items():
        regions[region_name] = {
            'mean_core_echo': stats.get('mean_core_echo', 252),
            'median_ramp_up': stats.get('median_ramp_up', [104, 188]),
            'median_ramp_down': stats.get('median_ramp_down', [140, 56]),
            'typical_ramp_up_bins': stats.get('typical_ramp_up_bins', 2),
            'typical_ramp_down_bins': stats.get('typical_ramp_down_bins', 2),
            'median_segment_depth_bins': stats.get('median_segment_depth_bins'),
            'p75_segment_depth_bins': stats.get('p75_segment_depth_bins'),
            'segment_count': stats.get('segment_count', 0),
        }

    # Build per-spoke segment lookup from coastline_profiles for validation
    # and strict per-segment coastline shaping.
    spoke_segments = {}
    spoke_segment_stats = {}
    for spoke_idx_str, profile in coastline_profiles.items():
        segs = []
        seg_stats = []
        for seg in profile.get('segments', []):
            if len(seg) >= 2:
                segs.append((seg[0], seg[1]))  # (start_bin, end_bin)
                seg_vals = seg[2] if len(seg) > 2 else []
                seg_stats.append({
                    'start_bin': int(seg[0]),
                    'end_bin': int(seg[1]),
                    'stats': _derive_local_coastline_stats(seg_vals),
                })
        if segs:
            spoke_segments[int(spoke_idx_str)] = segs
        if seg_stats:
            spoke_segment_stats[int(spoke_idx_str)] = seg_stats

    # Build per-spoke reference echo values from coastline_echo_data
    # Used for fallback painting when ray-tracing misses a segment
    coastline_echo_data = scene.get('coastline_echo_data', {})
    spoke_ref_values = {}
    for spoke_idx_str, segments in coastline_echo_data.items():
        bin_vals = {}
        for seg in segments:
            start_bin = seg[0]
            vals = seg[1]
            for vi, val in enumerate(vals):
                bin_vals[start_bin + vi] = val
        if bin_vals:
            spoke_ref_values[int(spoke_idx_str)] = bin_vals

    return {
        'regions': regions,
        'spoke_segments': spoke_segments,
        'spoke_segment_stats': spoke_segment_stats,
        'spoke_ref_values': spoke_ref_values,
    }


def _derive_local_coastline_stats(segment_values: List[int]) -> dict:
    """
    Build a local coastline stat bundle from a single captured coastline
    segment. This keeps strict Mode 4 forward-rendered, but lets each
    segment carry its own ramp lengths, ramp values, and core level instead
    of inheriting only a region-wide median profile.
    """
    defaults = {
        'mean_core_echo': 252.0,
        'median_ramp_up': [104, 188],
        'median_ramp_down': [140, 56],
        'typical_ramp_up_bins': 2,
        'typical_ramp_down_bins': 2,
        'median_segment_depth_bins': 12.0,
        'p75_segment_depth_bins': 18.0,
        'segment_count': 1,
    }
    vals = [int(v) for v in segment_values if v is not None]
    if not vals:
        return dict(defaults)

    peak_code = max(vals)
    peak_idx = int(np.argmax(vals))
    if peak_code <= 0:
        return dict(defaults)

    plateau_threshold = max(int(round(0.85 * peak_code)), peak_code - 16)
    first_core = peak_idx
    while first_core > 0 and vals[first_core - 1] >= plateau_threshold:
        first_core -= 1
    last_core = peak_idx
    while last_core + 1 < len(vals) and vals[last_core + 1] >= plateau_threshold:
        last_core += 1

    core_vals = vals[first_core:last_core + 1] if first_core <= last_core else [peak_code]
    ramp_up_full = vals[:first_core]
    ramp_down_full = vals[last_core + 1:]
    ramp_cap = 6

    return {
        'mean_core_echo': round(float(np.mean(core_vals)), 1),
        'median_ramp_up': [int(v) for v in ramp_up_full[:ramp_cap]],
        'median_ramp_down': [int(v) for v in ramp_down_full[:ramp_cap]],
        'typical_ramp_up_bins': int(len(ramp_up_full)),
        'typical_ramp_down_bins': int(len(ramp_down_full)),
        'median_segment_depth_bins': float(len(vals)),
        'p75_segment_depth_bins': float(len(vals)),
        'segment_count': 1,
    }


def _select_local_coastline_stats(entry_bin: int, exit_bin: int,
                                  base_stats: dict,
                                  spoke_segment_stats: Optional[List[dict]]) -> dict:
    """
    Pick the best matching captured coastline segment on the current spoke and
    overlay its local stat bundle onto the broader region stats.
    """
    if not spoke_segment_stats:
        return base_stats

    best_match = None
    best_overlap = -1
    best_center_gap = None
    best_edge_gap = None
    target_center = 0.5 * (float(entry_bin) + float(exit_bin))

    for seg in spoke_segment_stats:
        seg_start = int(seg.get('start_bin', entry_bin))
        seg_end = int(seg.get('end_bin', exit_bin))
        overlap = max(0, min(exit_bin, seg_end) - max(entry_bin, seg_start) + 1)
        center_gap = abs(target_center - 0.5 * (float(seg_start) + float(seg_end)))
        edge_gap = abs(entry_bin - seg_start) + abs(exit_bin - seg_end)

        if (
            overlap > best_overlap or
            (overlap == best_overlap and (best_center_gap is None or center_gap < best_center_gap)) or
            (
                overlap == best_overlap and best_center_gap is not None and
                center_gap == best_center_gap and (best_edge_gap is None or edge_gap < best_edge_gap)
            )
        ):
            best_match = seg
            best_overlap = overlap
            best_center_gap = center_gap
            best_edge_gap = edge_gap

    if best_match is None:
        return base_stats

    merged = dict(base_stats) if isinstance(base_stats, dict) else {}
    merged.update(best_match.get('stats', {}) or {})
    return merged


def _extract_object_calibration(scene: dict, objects, bin_size: float,
                                dead_zone: int) -> list:
    """Back-calculate calibrated radar equation parameters for each object.

    For each object:
    1. Compute expected (range, bearing) → (center_bin, bearing_tick)
    2. Scan object_echo_data for spokes near that bearing (±beamwidth)
    3. Find segments overlapping expected center_bin → extract observed peak, spread
    4. Solve for calibrated_ref_code or calibrated_compression

    Returns list of per-object calibration dicts.
    """
    object_echo_data = scene.get('object_echo_data', {})
    ref_range = 50.0
    ref_rcs = 350.0
    generic_ref_code = 190.0
    generic_compression = 55.0

    calibrations = []

    for obj_idx, obj in enumerate(objects):
        obj_range = math.sqrt(obj.x ** 2 + obj.y ** 2)
        obj_bearing_rad = math.atan2(obj.x, obj.y)
        obj_bearing_deg = math.degrees(obj_bearing_rad) % 360
        center_bin = int(obj_range / bin_size)
        bearing_tick = int(obj_bearing_deg * TICKS_PER_REV / 360.0)

        # Compute generic intensity_dB for this object
        if obj_range < dead_zone * bin_size or obj.rcs_m2 <= 0 or obj_range <= 0:
            calibrations.append({
                'peak_echo': 0, 'center_bin': center_bin,
                'spread_pattern': None, 'spread_bins': 2,
                'calibrated_ref_code': generic_ref_code,
                'calibrated_compression': generic_compression,
                'use_reference_spread': False,
            })
            continue

        intensity_dB = (10 * math.log10(obj.rcs_m2 / ref_rcs) -
                        40 * math.log10(obj_range / ref_range))

        # Search for observed echo data near this object's bearing
        half_beam_ticks = int((BEAMWIDTH_DEG / 2 + 1.0) * TICKS_PER_REV / 360.0)
        observed_peak = 0
        best_spread_pattern = None
        best_spread_bins = 2

        # Check spokes within ±beamwidth of object bearing
        num_ticks = len(scene.get('angle_ticks', [])) or TICKS_PER_REV
        for spoke_idx_str, segments in object_echo_data.items():
            spoke_idx = int(spoke_idx_str)
            # Estimate tick for this spoke (approximate — use angle_ticks if available)
            angle_ticks = scene.get('angle_ticks', None)
            if angle_ticks and spoke_idx < len(angle_ticks):
                spoke_tick = angle_ticks[spoke_idx]
            else:
                spoke_tick = int(spoke_idx * TICKS_PER_REV / num_ticks)

            tick_diff = abs(spoke_tick - bearing_tick)
            if tick_diff > TICKS_PER_REV / 2:
                tick_diff = TICKS_PER_REV - tick_diff
            if tick_diff > half_beam_ticks:
                continue

            # Check segments for overlap with expected center_bin
            for seg in segments:
                start_bin = seg[0]
                vals = seg[1]
                end_bin = start_bin + len(vals) - 1

                # Check if this segment overlaps the expected object position
                if end_bin < center_bin - 10 or start_bin > center_bin + 10:
                    continue

                seg_peak = max(vals) if vals else 0
                if seg_peak > observed_peak:
                    observed_peak = seg_peak
                    best_spread_pattern = list(vals)
                    best_spread_bins = len(vals)

        # Back-calculate calibrated parameters
        calibrated_ref_code = generic_ref_code
        calibrated_compression = generic_compression
        use_reference_spread = False

        if observed_peak > 0 and intensity_dB != 0:
            # Try to solve for calibrated_ref_code:
            # peak_code = ref_code * 10^(intensity_dB / compression)
            # At beam center (gain=1): observed_peak = ref_code * 10^(intensity_dB/55)
            power_term = 10 ** (intensity_dB / generic_compression)
            if power_term > 0:
                candidate_ref = observed_peak / power_term
                if 50 <= candidate_ref <= 400:
                    calibrated_ref_code = candidate_ref
                else:
                    # Solve for compression instead:
                    # observed_peak = 190 * 10^(intensity_dB / compression)
                    # intensity_dB / compression = log10(observed_peak / 190)
                    ratio = observed_peak / generic_ref_code
                    if ratio > 0 and ratio != 1.0:
                        log_ratio = math.log10(ratio)
                        if log_ratio != 0:
                            candidate_comp = intensity_dB / log_ratio
                            calibrated_compression = max(20, min(120, candidate_comp))

            if best_spread_pattern and len(best_spread_pattern) >= 2:
                use_reference_spread = True

        calibrations.append({
            'peak_echo': observed_peak,
            'center_bin': center_bin,
            'spread_pattern': best_spread_pattern,
            'spread_bins': max(2, best_spread_bins),
            'calibrated_ref_code': calibrated_ref_code,
            'calibrated_compression': calibrated_compression,
            'use_reference_spread': use_reference_spread,
        })

    return calibrations


def _extract_dead_zone_calibration(scene: dict) -> dict:
    """Analyze dead_zone_echo_data to extract per-bin probability and code distribution.

    Returns dict with:
      bin_probabilities: list of float (bins 0-30) — probability of echo at each bin
      code_distribution: list of (code, weight) pairs
      bin_code_distributions: per-bin code distributions for more local strict rendering
      spoke_probability: overall probability a spoke has any dead zone echo
    """
    dead_zone_echo_data = scene.get('dead_zone_echo_data', {})
    dead_zone_bins = scene.get('radar_config', {}).get('dead_zone_bins', DEAD_ZONE_BINS)

    bin_counts = np.zeros(dead_zone_bins, dtype=int)
    code_histogram = {}
    bin_code_histograms = [dict() for _ in range(dead_zone_bins)]
    spokes_with_dz = 0
    total_spokes = max(len(dead_zone_echo_data), 1)

    # Count across all spokes in the scene's angle_ticks
    num_spokes = len(scene.get('angle_ticks', [])) or TICKS_PER_REV
    total_spokes = num_spokes

    for spoke_idx_str, segments in dead_zone_echo_data.items():
        has_echo = False
        for seg in segments:
            start_bin = seg[0]
            vals = seg[1]
            for vi, val in enumerate(vals):
                b = start_bin + vi
                if 0 <= b < dead_zone_bins and val > 0:
                    bin_counts[b] += 1
                    code_histogram[int(val)] = code_histogram.get(int(val), 0) + 1
                    bin_hist = bin_code_histograms[b]
                    ival = int(val)
                    bin_hist[ival] = bin_hist.get(ival, 0) + 1
                    has_echo = True
        if has_echo:
            spokes_with_dz += 1

    bin_probabilities = (bin_counts / total_spokes).tolist()
    spoke_probability = spokes_with_dz / total_spokes

    # Normalize code distribution to weights
    code_distribution = []
    total_codes = sum(code_histogram.values()) if code_histogram else 1
    for code, count in sorted(code_histogram.items()):
        code_distribution.append((code, count / total_codes))

    bin_code_distributions = {}
    for b, hist in enumerate(bin_code_histograms):
        if not hist:
            continue
        total_bin_codes = sum(hist.values())
        if total_bin_codes <= 0:
            continue
        bin_code_distributions[b] = [
            (code, count / total_bin_codes)
            for code, count in sorted(hist.items())
        ]

    return {
        'bin_probabilities': bin_probabilities,
        'code_distribution': code_distribution,
        'bin_code_distributions': bin_code_distributions,
        'spoke_probability': spoke_probability,
    }


def _extract_noise_calibration(scene: dict) -> dict:
    """Extract noise calibration from scene clutter config and noise_echo_data.

    Returns dict with:
      noise_prob, noise_codes, noise_weights, avg_noise_bins_per_spoke
    """
    clutter = scene.get('clutter', {})
    noise_echo_data = scene.get('noise_echo_data', {})

    noise_prob = clutter.get('noise_probability', 0.001)
    noise_codes = clutter.get('noise_codes', [8, 20, 28])
    noise_weights = clutter.get('noise_weights', None)

    # Compute average noise bins per spoke from reference data
    total_noise_bins = 0
    num_spokes = len(scene.get('angle_ticks', [])) or TICKS_PER_REV
    for spoke_idx_str, segments in noise_echo_data.items():
        for seg in segments:
            total_noise_bins += len(seg[1]) if len(seg) > 1 else 0

    avg_noise_bins = total_noise_bins / num_spokes if num_spokes > 0 else 0

    return {
        'noise_prob': noise_prob,
        'noise_codes': noise_codes,
        'noise_weights': noise_weights,
        'avg_noise_bins_per_spoke': avg_noise_bins,
        'noise_field_stats': clutter.get('noise_field_stats'),
    }


def _resolve_scene_input(input_path, verbose=True):
    """Resolve a scene input that may be a JSON, CSV file, or folder of CSVs.

    Accepts:
      - Scene JSON file (*_scene.json or *.json) → loaded directly
      - Single Furuno CSV file (*.csv) → auto-extracts scene via extract_scene
      - Folder containing CSV files → uses the first CSV, auto-extracts

    Returns:
        (coastlines, objects, scene, scene_path) tuple
    """
    if os.path.isdir(input_path):
        # Folder — find first CSV
        csvs = sorted([f for f in os.listdir(input_path)
                        if f.endswith('.csv') and 'detection' not in f.lower()])
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {input_path}")
        csv_path = os.path.join(input_path, csvs[0])
        if verbose:
            print(f"  Folder input: using {csvs[0]} ({len(csvs)} CSVs in folder)")
        return _extract_and_load(csv_path, verbose)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if input_path.lower().endswith('.csv'):
        return _extract_and_load(input_path, verbose)

    # Assume JSON scene file
    coastlines, objects, scene = load_scene_from_file(input_path)
    return coastlines, objects, scene, input_path


def _extract_and_load(csv_path, verbose=True):
    """Extract a scene from a Furuno CSV and return it ready for simulation.

    Auto-runs extract_scene.py's extraction function on the CSV, saves
    the scene JSON alongside it, and returns the loaded scene.
    """
    try:
        from extract_scene import extract_scene_from_csv
    except ImportError:
        # Try relative import from same directory
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "extract_scene",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_scene.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        extract_scene_from_csv = mod.extract_scene_from_csv

    if verbose:
        print(f"  Auto-extracting scene from: {os.path.basename(csv_path)}")

    scene = extract_scene_from_csv(csv_path, verbose=verbose)

    # Save scene JSON next to the CSV (or in CWD if CSV is read-only)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    scene_json_path = os.path.join(os.path.dirname(csv_path), f"{base}_scene.json")
    try:
        import json as _json
        with open(scene_json_path, 'w') as f:
            _json.dump(scene, f, indent=2)
        if verbose:
            print(f"  Scene JSON saved: {scene_json_path}")
    except OSError:
        # Read-only location — save to CWD
        scene_json_path = os.path.join(os.getcwd(), f"{base}_scene.json")
        with open(scene_json_path, 'w') as f:
            _json.dump(scene, f, indent=2)
        if verbose:
            print(f"  Scene JSON saved: {scene_json_path}")

    coastlines, objects, scene = load_scene_from_file(scene_json_path)
    return coastlines, objects, scene, scene_json_path


def _load_mode4_population_calibration(calibration_path: Optional[str] = None) -> dict:
    """Load the corpus-level Mode 4 predictive calibration file."""
    cal_path = calibration_path or MODE4_CALIBRATION_PATH
    if not os.path.isfile(cal_path):
        raise FileNotFoundError(f"Mode 4 calibration file not found: {cal_path}")
    with open(cal_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_mode4_object_learned_model(model_path: Optional[str] = None) -> Optional[dict]:
    if not MODE4_USE_OBJECT_LEARNED_MODEL:
        return None
    if not _HAS_PANDAS:
        return None
    path = os.path.abspath(model_path or MODE4_OBJECT_LEARNED_MODEL_PATH)
    if not os.path.isfile(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached_path = _MODE4_OBJECT_LEARNED_MODEL_CACHE.get("path")
    cached_mtime = _MODE4_OBJECT_LEARNED_MODEL_CACHE.get("mtime")
    cached_artifact = _MODE4_OBJECT_LEARNED_MODEL_CACHE.get("artifact")
    if cached_artifact is not None and cached_path == path and cached_mtime == mtime:
        return cached_artifact  # type: ignore[return-value]
    try:
        with open(path, "rb") as f:
            artifact = pickle.load(f)
    except Exception:
        return None
    if not isinstance(artifact, dict):
        return None
    _MODE4_OBJECT_LEARNED_MODEL_CACHE["path"] = path
    _MODE4_OBJECT_LEARNED_MODEL_CACHE["mtime"] = mtime
    _MODE4_OBJECT_LEARNED_MODEL_CACHE["artifact"] = artifact
    return artifact


def _derive_ref_code_from_peak_observation(peak_echo: Optional[float],
                                           range_m: Optional[float],
                                           rcs_m2: Optional[float],
                                           compression: float = 55.0,
                                           ref_range_m: float = 50.0,
                                           ref_rcs_m2: float = 350.0) -> Optional[float]:
    if peak_echo is None or range_m is None or rcs_m2 is None:
        return None
    try:
        peak_echo = float(peak_echo)
        range_m = float(range_m)
        rcs_m2 = float(rcs_m2)
        compression = float(compression)
    except (TypeError, ValueError):
        return None
    if peak_echo <= 0.0 or range_m <= 0.0 or rcs_m2 <= 0.0 or compression <= 0.0:
        return None
    intensity_db = (
        10.0 * math.log10(rcs_m2 / ref_rcs_m2) -
        40.0 * math.log10(range_m / ref_range_m)
    )
    power_term = 10.0 ** (intensity_db / compression)
    if power_term <= 0.0:
        return None
    candidate = peak_echo / power_term
    if not math.isfinite(candidate):
        return None
    return float(_clamp(candidate, 50.0, 400.0))


def _object_corpus_bucket_for_scene(scene: Optional[dict], raw_obj: Optional[dict],
                                    obj: RadarObject) -> dict:
    calibration = (scene or {}).get("_mode4_population_calibration") or {}
    obj_cfg = calibration.get("objects", {}) if isinstance(calibration, dict) else {}
    obj_class = None
    if isinstance(raw_obj, dict):
        obj_class = raw_obj.get("type") or raw_obj.get("geom_type")
    if not obj_class:
        obj_class = getattr(obj, "obj_type", "medium_boat")
    bucket = obj_cfg.get(obj_class, {}) if isinstance(obj_cfg, dict) else {}
    return bucket if isinstance(bucket, dict) else {}


def _predictive_object_scene_weight(scene: Optional[dict]) -> float:
    calibration = (scene or {}).get("_mode4_population_calibration") or {}
    tuning = calibration.get("predictive_tuning", {}) if isinstance(calibration, dict) else {}
    try:
        return float(_clamp(tuning.get("object_rcs_scene_weight", 0.0), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _resolve_object_render_priors(obj: RadarObject,
                                  raw_obj: Optional[dict],
                                  sim_mode: Optional[str],
                                  scene_bin_size: float,
                                  cal: Optional[dict] = None,
                                  scene: Optional[dict] = None) -> ObjectRenderPriors:
    priors = ObjectRenderPriors()
    cal = cal or {}
    raw_obj = raw_obj if isinstance(raw_obj, dict) else {}
    reference_family = sim_mode in {
        SIM_MODE_CURRENT_MODE4,
        SIM_MODE_M4G,
        SIM_MODE_M4P,
    }
    if cal:
        try:
            if cal.get("calibrated_ref_code") is not None:
                priors.ref_code = float(cal.get("calibrated_ref_code"))
        except (TypeError, ValueError):
            pass
        try:
            if cal.get("calibrated_compression") is not None:
                priors.compression = float(cal.get("calibrated_compression"))
        except (TypeError, ValueError):
            pass
        try:
            if cal.get("spread_bins") is not None:
                priors.spread_bins = float(cal.get("spread_bins"))
        except (TypeError, ValueError):
            pass
        pattern = cal.get("spread_pattern")
        if cal.get("use_reference_spread") and isinstance(pattern, list) and pattern:
            priors.reference_pattern = [float(v) for v in pattern]
            try:
                priors.reference_center_bin = int(cal.get("center_bin"))
            except (TypeError, ValueError):
                priors.reference_center_bin = None
            priors.reference_profile_blend = 0.35
            priors.support_length_bins = len(pattern)

    local_range_m = raw_obj.get("range_m")
    if local_range_m is None:
        local_range_m = math.hypot(obj.x, obj.y)
    local_ref_code = None
    if reference_family:
        local_ref_code = _derive_ref_code_from_peak_observation(
            raw_obj.get("peak_echo"),
            local_range_m,
            raw_obj.get("rcs_m2", obj.rcs_m2),
            compression=priors.compression,
        )
        radial_bins = raw_obj.get("radial_bins")
        if radial_bins is not None:
            try:
                radial_bins = float(radial_bins)
            except (TypeError, ValueError):
                radial_bins = None
            if radial_bins and radial_bins > 0.0:
                priors.support_length_bins = int(max(radial_bins, priors.support_length_bins or 0))
                if scene_bin_size > 0:
                    priors.spread_bins = max(float(priors.spread_bins or 0.0), float(radial_bins) / 4.0)

    if sim_mode == SIM_MODE_CURRENT_MODE4:
        if float(cal.get("peak_echo", 0) or 0) <= 0.0 and local_ref_code is not None:
            priors.ref_code = local_ref_code
    elif sim_mode == SIM_MODE_M4G:
        if local_ref_code is not None:
            priors.ref_code = local_ref_code
        priors.reference_pattern = None
        priors.reference_profile_blend = 0.0
    elif sim_mode == SIM_MODE_M4P:
        bucket = _object_corpus_bucket_for_scene(scene, raw_obj, obj)
        corpus_ref_code = _derive_ref_code_from_peak_observation(
            bucket.get("peak_echo_p50"),
            bucket.get("range_m_p50"),
            bucket.get("rcs_m2_p50"),
            compression=priors.compression,
        )
        scene_weight = _predictive_object_scene_weight(scene)
        if local_ref_code is not None and corpus_ref_code is not None:
            priors.ref_code = _blend_scalar(local_ref_code, corpus_ref_code, scene_weight)
        elif local_ref_code is not None:
            priors.ref_code = local_ref_code
        elif corpus_ref_code is not None:
            priors.ref_code = corpus_ref_code
        priors.reference_pattern = None
        priors.reference_profile_blend = 0.0
    else:
        priors.reference_pattern = None
        priors.reference_profile_blend = 0.0
        if sim_mode not in {SIM_MODE_CURRENT_MODE4, SIM_MODE_M4G, SIM_MODE_M4P}:
            priors.support_length_bins = None

    return priors


def _render_object_echo_shared(obj: RadarObject, bearing_rad: float,
                               scene_bins: int, scene_bin_size: float,
                               scene_dead_zone: int,
                               sim_mode: Optional[str],
                               raw_obj: Optional[dict] = None,
                               cal: Optional[dict] = None,
                               support_segments: Optional[list] = None,
                               learned_object_model: Optional[dict] = None,
                               scene: Optional[dict] = None) -> Any:
    priors = _resolve_object_render_priors(
        obj,
        raw_obj,
        sim_mode,
        scene_bin_size,
        cal=cal,
        scene=scene,
    )
    return _render_object_echo_core(
        obj,
        bearing_rad,
        priors,
        scene_bins=scene_bins,
        scene_bin_size=scene_bin_size,
        scene_dead_zone=scene_dead_zone,
        beamwidth_deg=BEAMWIDTH_DEG,
        support_segments=support_segments,
        learned_artifact=learned_object_model,
        raw_obj=raw_obj,
    )


def _blend_scalar(scene_value: Optional[float], corpus_value: float, scene_weight: float) -> float:
    """Blend a scene-derived scalar with a corpus-derived scalar."""
    if scene_value is None:
        return float(corpus_value)
    w = max(0.0, min(1.0, float(scene_weight)))
    return float(w * float(scene_value) + (1.0 - w) * float(corpus_value))


def _blend_code_sequence(scene_codes: Optional[List[int]], corpus_codes: List[int],
                         scene_weight: float, scale: float = 1.0) -> List[int]:
    """Blend two code sequences elementwise, padding with endpoint values when needed."""
    if not corpus_codes and not scene_codes:
        return []
    if not scene_codes:
        return [int(round(min(252, max(0, c * scale)))) for c in corpus_codes]
    w = max(0.0, min(1.0, float(scene_weight)))
    length = max(len(scene_codes), len(corpus_codes))
    out = []
    for i in range(length):
        sc = scene_codes[i] if i < len(scene_codes) else scene_codes[-1]
        cc = corpus_codes[i] if i < len(corpus_codes) else corpus_codes[-1]
        val = (w * float(sc) + (1.0 - w) * float(cc)) * scale
        out.append(int(round(min(252, max(0, val)))))
    return out


def _blend_histograms(scene_codes: Optional[List[int]], scene_weights: Optional[List[float]],
                      corpus_hist: dict, scene_weight: float) -> tuple[list[int], list[float]] | tuple[None, None]:
    """Blend a scene noise palette with a corpus histogram over the union of code bins."""
    if not corpus_hist and not scene_codes:
        return None, None
    w = max(0.0, min(1.0, float(scene_weight)))
    corpus_map = {int(code): float(weight) for code, weight in corpus_hist.items()}
    scene_map = {}
    if scene_codes and scene_weights and len(scene_codes) == len(scene_weights):
        scene_map = {int(code): float(weight) for code, weight in zip(scene_codes, scene_weights)}
    codes = sorted(set(corpus_map.keys()) | set(scene_map.keys()))
    if not codes:
        return None, None
    blended = []
    for code in codes:
        sw = scene_map.get(code, 0.0)
        cw = corpus_map.get(code, 0.0)
        blended.append(w * sw + (1.0 - w) * cw)
    total = sum(blended)
    if total <= 0:
        return None, None
    weights = [b / total for b in blended]
    return codes, weights


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _round_clamped_code(value: float) -> int:
    return int(round(min(252.0, max(0.0, value))))


def _safe_ratio_float(numer: float, denom: float, default: float = 0.0) -> float:
    denom_val = float(denom)
    if abs(denom_val) < 1e-9:
        return float(default)
    return float(numer) / denom_val


def _display_db_from_code_value(value: float) -> float:
    safe = max(0.0, float(value))
    return 20.0 * math.log10((safe + 1.0) / 252.0)


def _code_value_from_display_db(display_db: float) -> float:
    return _clamp(252.0 * (10.0 ** (float(display_db) / 20.0)) - 1.0, 0.0, 252.0)


def _shape_fraction_drop_db(fraction: float) -> float:
    """Convert an amplitude-style shape fraction into a display-dB drop."""
    return 20.0 * math.log10(max(1e-6, float(fraction)))


def _code_from_peak_db_shape(peak_db: float, shape_fraction: float,
                             *delta_db_terms: float,
                             floor_code: Optional[float] = None) -> float:
    bin_db = float(peak_db) + _shape_fraction_drop_db(shape_fraction)
    for delta in delta_db_terms:
        bin_db += float(delta or 0.0)
    if floor_code is not None:
        bin_db = max(bin_db, _display_db_from_code_value(float(floor_code)))
    return _code_value_from_display_db(bin_db)


def _apply_display_db_correction(value: float, delta_db: float) -> float:
    if value <= 0.0 or abs(delta_db) < 1e-9:
        return float(value)
    corrected_db = _display_db_from_code_value(value) + float(delta_db)
    return _code_value_from_display_db(corrected_db)


def _normalized_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    diff = (float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0
    return float(diff)


def _object_support_phase_label(offset: int, seg_len: int) -> str:
    if seg_len <= 2:
        return "short"
    if offset <= 1:
        return "entry"
    if offset >= seg_len - 2:
        return "exit"
    return "interior"


def _projected_object_extents(length_m: float, width_m: float, aspect_deg: float) -> tuple[float, float]:
    aspect_rad = math.radians(float(aspect_deg))
    radial_extent = abs(float(length_m) * math.cos(aspect_rad)) + abs(float(width_m) * math.sin(aspect_rad))
    cross_extent = abs(float(length_m) * math.sin(aspect_rad)) + abs(float(width_m) * math.cos(aspect_rad))
    return float(radial_extent), float(cross_extent)


def _gaussian_bump(x: float, center: float, sigma: float) -> float:
    sigma = max(1e-6, float(sigma))
    return math.exp(-0.5 * ((float(x) - float(center)) / sigma) ** 2)


def _coastline_range_correction_db(range_m: float) -> float:
    if range_m < 25.0:
        return 0.5
    if range_m < 60.0:
        return 1.25
    if range_m < 120.0:
        return 2.25
    if range_m < 180.0:
        return 1.5
    return 0.75


def _coastline_interior_correction_db(range_m: float, depth_from_entry: int,
                                      support_len_bins: int, core_depth_bins: int,
                                      rel_pos: float = 0.5) -> float:
    if support_len_bins <= 3:
        return 0.0
    rel = _clamp(rel_pos, 0.0, 1.0)
    support_scale = _clamp((support_len_bins - 3) / 8.0, 0.0, 1.0)
    interior_peak = _gaussian_bump(rel, 0.58, 0.24)
    exit_peak = _gaussian_bump(rel, 0.92, 0.10)
    edge_penalty = _gaussian_bump(min(rel, 1.0 - rel), 0.0, 0.10)
    if depth_from_entry >= max(0, support_len_bins - 2):
        phase_db = 0.75 + 0.85 * exit_peak
    elif depth_from_entry >= core_depth_bins:
        phase_db = 1.6 + 1.25 * interior_peak
    else:
        return 0.0
    phase_db -= 0.35 * edge_penalty
    return support_scale * (phase_db + _coastline_range_correction_db(range_m))


def _object_peak_correction_db(obj_type: str, range_m: float) -> float:
    type_db = OBJECT_TYPE_DB_CORRECTIONS.get(obj_type, OBJECT_TYPE_DB_CORRECTIONS["medium_boat"])
    if range_m < 35.0:
        range_db = 2.0
    elif range_m < 80.0:
        range_db = 0.75
    elif range_m < 140.0:
        range_db = -0.5
    elif range_m < 175.0:
        range_db = 0.5
    else:
        range_db = 2.0
    return type_db + range_db


def _object_support_correction_db(obj_type: str, support_len_bins: int, rel_pos: float) -> float:
    if support_len_bins <= 2:
        return -2.0
    edge_db = 1.6 if (rel_pos <= 0.2 or rel_pos >= 0.8) else 0.0
    if obj_type == "buoy" and edge_db > 0.0:
        edge_db += 0.5
    return edge_db


def _noise_probability_scale_for_family(family: str, range_m: float) -> float:
    scale = 1.3 if family == "open_water" else 1.1
    if family == "open_water" and range_m >= 100.0:
        scale += 0.1
    return scale


def _noise_code_correction_db(family: str, range_m: float) -> float:
    base = 2.5 if family == "open_water" else 0.9
    if 50.0 <= range_m < 200.0:
        base += 0.5
    return base


def _noise_support_correction_db(family: str, support_len_bins: int,
                                 rel_pos: float, range_m: float) -> float:
    rel = _clamp(rel_pos, 0.0, 1.0)
    interior_peak = _gaussian_bump(rel, 0.50, 0.22)
    edge_peak = _gaussian_bump(rel, 0.08, 0.10) + _gaussian_bump(rel, 0.92, 0.10)
    if support_len_bins <= 1:
        return -0.45 if family == "open_water" else -0.15
    support_scale = _clamp((support_len_bins - 1) / 3.0, 0.0, 1.0)
    range_boost = 0.25 if 60.0 <= range_m < 180.0 else 0.0
    if family == "open_water":
        return support_scale * (0.55 + range_boost) * interior_peak - 0.45 * edge_peak
    return support_scale * (0.85 + 0.5 * range_boost) * interior_peak - 0.20 * edge_peak


def _dead_zone_correction_db(bin_idx: int, scene_dead_zone: int) -> float:
    if scene_dead_zone <= 1:
        return 0.0
    rel = float(bin_idx) / float(max(1, scene_dead_zone - 1))
    if rel <= 0.15 or rel >= 0.85:
        return -0.75
    if 0.3 <= rel <= 0.7:
        return 0.5
    return 0.0


def _dead_zone_support_correction_db(bin_idx: int, scene_dead_zone: int,
                                     support_len_bins: int, rel_pos: float) -> float:
    base = _dead_zone_correction_db(bin_idx, scene_dead_zone)
    rel = _clamp(rel_pos, 0.0, 1.0)
    if support_len_bins <= 1:
        return base - 0.25
    support_scale = _clamp((support_len_bins - 1) / 2.0, 0.0, 1.0)
    interior_peak = _gaussian_bump(rel, 0.50, 0.24)
    edge_peak = _gaussian_bump(min(rel, 1.0 - rel), 0.0, 0.12)
    return base + support_scale * 0.60 * interior_peak - 0.25 * edge_peak


def _coastline_stat_or_default(stats: dict, key: str, default):
    value = stats.get(key)
    return default if value is None else value


def _normalize_code_histogram(hist: Optional[dict]) -> dict[str, float]:
    if not isinstance(hist, dict):
        return {}
    clean = {}
    for code, weight in hist.items():
        try:
            icode = int(code)
            fweight = float(weight)
        except (TypeError, ValueError):
            continue
        if fweight > 0:
            clean[icode] = clean.get(icode, 0.0) + fweight
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {str(code): clean[code] / total for code in sorted(clean)}


def _histogram_to_arrays(hist: dict[str, float], family: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    clean = _normalize_code_histogram(hist)
    codes = []
    weights = []
    for code_str, weight in clean.items():
        code = int(code_str)
        if family == "open_water" and code > MODE4_NOISE_OPEN_WATER_MAX_CODE:
            continue
        if code in FURUNO_GAIN_SET and weight > 0:
            codes.append(code)
            weights.append(float(weight))
    if not codes:
        return None, None
    w = np.array(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0:
        return None, None
    return np.array(codes, dtype=int), (w / total)


def _blend_hist_maps(scene_hist: Optional[dict], corpus_hist: Optional[dict], scene_weight: float) -> dict[str, float]:
    scene_map = _normalize_code_histogram(scene_hist or {})
    corpus_map = _normalize_code_histogram(corpus_hist or {})
    if not scene_map and not corpus_map:
        return {}
    if not corpus_map:
        return scene_map
    if not scene_map:
        return corpus_map
    w = max(0.0, min(1.0, float(scene_weight)))
    codes = sorted(set(scene_map.keys()) | set(corpus_map.keys()), key=lambda x: int(x))
    blended = {}
    for code in codes:
        blended[code] = w * float(scene_map.get(code, 0.0)) + (1.0 - w) * float(corpus_map.get(code, 0.0))
    return _normalize_code_histogram(blended)


def _noise_field_cell_key(sector_idx: int, band_idx: int) -> str:
    return f"{int(sector_idx)}_{int(band_idx)}"


def _noise_field_layout(stats: Optional[dict], scene_bins: int, scene_dead_zone: int) -> tuple[int, int, list[int]]:
    n_sectors = int((stats or {}).get("bearing_sectors", MODE4_NOISE_FIELD_SECTORS) or MODE4_NOISE_FIELD_SECTORS)
    n_bands = int((stats or {}).get("range_bands", MODE4_NOISE_FIELD_BANDS) or MODE4_NOISE_FIELD_BANDS)
    n_sectors = max(1, n_sectors)
    n_bands = max(1, n_bands)
    band_edges = np.linspace(scene_dead_zone, scene_bins, n_bands + 1)
    band_edges = [int(round(v)) for v in band_edges]
    band_edges[0] = scene_dead_zone
    band_edges[-1] = scene_bins
    return n_sectors, n_bands, band_edges


def _build_noise_field_family_defaults(stats: Optional[dict]) -> dict:
    defaults = {}
    if not isinstance(stats, dict):
        return defaults
    family_defaults = stats.get("family_defaults") or {}
    if isinstance(family_defaults, dict):
        for family, cell in family_defaults.items():
            if isinstance(cell, dict):
                defaults[family] = {
                    "occupancy_probability": float(cell.get("occupancy_probability", 0.0) or 0.0),
                    "median_run_len": float(cell.get("median_run_len", 1.0) or 1.0),
                    "code_histogram": _normalize_code_histogram(cell.get("code_histogram") or {}),
                }
    families = stats.get("families") or {}
    for family, family_cells in families.items():
        if family in defaults or not isinstance(family_cells, dict):
            continue
        occs = []
        runs = []
        hists = []
        for cell in family_cells.values():
            if not isinstance(cell, dict):
                continue
            occ = cell.get("occupancy_probability")
            if occ is not None:
                occs.append(float(occ))
            run_len = cell.get("median_run_len")
            if run_len is not None:
                runs.append(float(run_len))
            hist = _normalize_code_histogram(cell.get("code_histogram") or {})
            if hist:
                hists.append(hist)
        if occs or runs or hists:
            mean_hist = {}
            if hists:
                codes = sorted({code for hist in hists for code in hist.keys()}, key=lambda x: int(x))
                for code in codes:
                    mean_hist[code] = sum(float(hist.get(code, 0.0)) for hist in hists) / float(len(hists))
                mean_hist = _normalize_code_histogram(mean_hist)
            defaults[family] = {
                "occupancy_probability": float(np.median(occs)) if occs else 0.0,
                "median_run_len": float(np.median(runs)) if runs else 1.0,
                "code_histogram": mean_hist,
            }
    return defaults


def _blend_noise_field_stats(scene_stats: Optional[dict], corpus_stats: Optional[dict],
                             scene_weight: float, probability_scale: float) -> Optional[dict]:
    if not scene_stats and not corpus_stats:
        return None
    scene_stats = scene_stats or {}
    corpus_stats = corpus_stats or {}
    scene_defaults = _build_noise_field_family_defaults(scene_stats)
    corpus_defaults = _build_noise_field_family_defaults(corpus_stats)
    families = {}
    scene_families = scene_stats.get("families") or {}
    corpus_families = corpus_stats.get("families") or {}
    for family in ("open_water", "near_land"):
        result_cells = {}
        scene_cells = scene_families.get(family) or {}
        corpus_cells = corpus_families.get(family) or {}
        cell_keys = sorted(set(scene_cells.keys()) | set(corpus_cells.keys()))
        for cell_key in cell_keys:
            scene_cell = scene_cells.get(cell_key) if isinstance(scene_cells.get(cell_key), dict) else None
            corpus_cell = corpus_cells.get(cell_key) if isinstance(corpus_cells.get(cell_key), dict) else None
            if scene_cell and corpus_cell:
                occ = _blend_scalar(
                    scene_cell.get("occupancy_probability"),
                    float(corpus_cell.get("occupancy_probability", 0.0) or 0.0),
                    scene_weight,
                )
                run_len = _blend_scalar(
                    scene_cell.get("median_run_len"),
                    float(corpus_cell.get("median_run_len", 1.0) or 1.0),
                    scene_weight,
                )
                hist = _blend_hist_maps(
                    scene_cell.get("code_histogram"),
                    corpus_cell.get("code_histogram"),
                    scene_weight,
                )
            elif scene_cell:
                occ = float(scene_cell.get("occupancy_probability", 0.0) or 0.0)
                run_len = float(scene_cell.get("median_run_len", 1.0) or 1.0)
                hist = _normalize_code_histogram(scene_cell.get("code_histogram") or {})
            elif corpus_cell:
                occ = float(corpus_cell.get("occupancy_probability", 0.0) or 0.0)
                run_len = float(corpus_cell.get("median_run_len", 1.0) or 1.0)
                hist = _normalize_code_histogram(corpus_cell.get("code_histogram") or {})
            else:
                continue
            result_cells[cell_key] = {
                "occupancy_probability": float(max(0.0, occ) * probability_scale),
                "median_run_len": float(max(1.0, run_len)),
                "code_histogram": hist,
            }
        families[family] = result_cells

    family_defaults = {}
    for family in ("open_water", "near_land"):
        scene_default = scene_defaults.get(family)
        corpus_default = corpus_defaults.get(family)
        if scene_default and corpus_default:
            occ = _blend_scalar(scene_default.get("occupancy_probability"),
                                float(corpus_default.get("occupancy_probability", 0.0) or 0.0),
                                scene_weight)
            run_len = _blend_scalar(scene_default.get("median_run_len"),
                                    float(corpus_default.get("median_run_len", 1.0) or 1.0),
                                    scene_weight)
            hist = _blend_hist_maps(scene_default.get("code_histogram"),
                                    corpus_default.get("code_histogram"),
                                    scene_weight)
        elif corpus_default:
            occ = float(corpus_default.get("occupancy_probability", 0.0) or 0.0)
            run_len = float(corpus_default.get("median_run_len", 1.0) or 1.0)
            hist = _normalize_code_histogram(corpus_default.get("code_histogram") or {})
        elif scene_default:
            occ = float(scene_default.get("occupancy_probability", 0.0) or 0.0)
            run_len = float(scene_default.get("median_run_len", 1.0) or 1.0)
            hist = _normalize_code_histogram(scene_default.get("code_histogram") or {})
        else:
            continue
        family_defaults[family] = {
            "occupancy_probability": float(max(0.0, occ) * probability_scale),
            "median_run_len": float(max(1.0, run_len)),
            "code_histogram": hist,
        }

    base_stats = corpus_stats if corpus_stats else scene_stats
    n_sectors = int(base_stats.get("bearing_sectors", MODE4_NOISE_FIELD_SECTORS) or MODE4_NOISE_FIELD_SECTORS)
    n_bands = int(base_stats.get("range_bands", MODE4_NOISE_FIELD_BANDS) or MODE4_NOISE_FIELD_BANDS)
    return {
        "bearing_sectors": n_sectors,
        "range_bands": n_bands,
        "families": families,
        "family_defaults": family_defaults,
    }


def _build_object_support_lookup(scene: dict, scene_bins: int) -> dict[int, np.ndarray]:
    lookup = {}
    for spoke_idx_str, segments in (scene.get("object_echo_data") or {}).items():
        try:
            spoke_idx = int(spoke_idx_str)
        except (TypeError, ValueError):
            continue
        mask = np.zeros(scene_bins, dtype=bool)
        for seg in segments:
            if not isinstance(seg, (list, tuple)) or len(seg) < 2:
                continue
            start_bin = int(seg[0])
            vals = seg[1] if isinstance(seg[1], (list, tuple)) else []
            end_bin = min(scene_bins, start_bin + len(vals))
            if end_bin > start_bin:
                mask[max(0, start_bin):end_bin] = True
        if np.any(mask):
            lookup[spoke_idx] = mask
    return lookup


def _build_object_support_by_object(scene: dict) -> dict[int, dict[int, list[dict]]]:
    """
    Build a per-object support lookup from object_echo_data when segment
    ownership is available. Older scene JSON files omit the owner index and
    simply return an empty lookup so strict Mode 4 can fall back cleanly.
    """
    by_object = {}
    for spoke_idx_str, segments in (scene.get("object_echo_data") or {}).items():
        try:
            spoke_idx = int(spoke_idx_str)
        except (TypeError, ValueError):
            continue
        for seg in segments:
            if not isinstance(seg, (list, tuple)) or len(seg) < 3:
                continue
            try:
                start_bin = int(seg[0])
                owner_idx = int(seg[2])
            except (TypeError, ValueError):
                continue
            vals = seg[1] if isinstance(seg[1], (list, tuple)) else []
            if owner_idx < 0 or not vals:
                continue
            by_object.setdefault(owner_idx, {}).setdefault(spoke_idx, []).append({
                "start_bin": start_bin,
                "values": [int(v) for v in vals],
                "end_bin": start_bin + len(vals) - 1,
            })
    return by_object


def _build_near_land_mask(scene_bins: int, scene_dead_zone: int,
                          land_intervals: List[Tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(scene_bins, dtype=bool)
    for entry_bin, exit_bin in land_intervals:
        start = max(scene_dead_zone, int(entry_bin) - MODE4_NOISE_NEAR_LAND_BINS)
        end = min(scene_bins, int(exit_bin) + MODE4_NOISE_NEAR_LAND_BINS + 1)
        if end > start:
            mask[start:end] = True
    return mask


def _select_noise_cell(stats: Optional[dict], family: str, sector_idx: int, band_idx: int) -> Optional[dict]:
    if not isinstance(stats, dict):
        return None
    families = stats.get("families") or {}
    family_cells = families.get(family) or {}
    cell = family_cells.get(_noise_field_cell_key(sector_idx, band_idx))
    if isinstance(cell, dict):
        return cell
    family_defaults = stats.get("family_defaults") or {}
    default_cell = family_defaults.get(family)
    if isinstance(default_cell, dict):
        return default_cell
    return None


def _band_index_for_bin(bin_idx: int, band_edges: List[int]) -> int:
    for band_idx in range(max(0, len(band_edges) - 1)):
        if band_edges[band_idx] <= bin_idx < band_edges[band_idx + 1]:
            return band_idx
    return max(0, len(band_edges) - 2)


def _choose_noise_histogram_codes(spoke_rng: np.random.RandomState,
                                  family: str, band_idx: int, bin_idx: int,
                                  noise_stats: Optional[dict],
                                  legacy_noise_codes: np.ndarray,
                                  legacy_noise_weights: Optional[List[float]],
                                  sector_idx: int) -> tuple[np.ndarray, Optional[np.ndarray], int]:
    hist_codes = None
    hist_weights = None
    base_code = 8

    cell = _select_noise_cell(noise_stats, family, sector_idx, band_idx)
    if cell is not None:
        hist_codes, hist_weights = _histogram_to_arrays(cell.get("code_histogram") or {}, family)

    if hist_codes is None or len(hist_codes) == 0:
        hist_codes = legacy_noise_codes
        if legacy_noise_weights and len(legacy_noise_weights) == len(legacy_noise_codes):
            w = np.array(legacy_noise_weights, dtype=float)
            hist_weights = w / np.sum(w) if np.sum(w) > 0 else None
        else:
            hist_weights = None

    if hist_codes is None or len(hist_codes) == 0:
        hist_codes = np.array([8], dtype=int)
        hist_weights = None

    if hist_weights is not None:
        base_code = int(spoke_rng.choice(hist_codes, p=hist_weights))
    else:
        base_code = int(spoke_rng.choice(hist_codes))
    return hist_codes, hist_weights, base_code


def _paint_noise_support_segments(echoes: np.ndarray, tick: int, scene_bins: int,
                                  scene_dead_zone: int, scene_bin_size: float,
                                  noise_stats: Optional[dict],
                                  legacy_noise_codes: np.ndarray,
                                  legacy_noise_weights: Optional[List[float]],
                                  object_support_mask: Optional[np.ndarray],
                                  land_intervals: List[Tuple[int, int]],
                                  noise_segments: list) -> bool:
    if not noise_segments:
        return False

    if noise_stats:
        n_sectors, _, band_edges = _noise_field_layout(noise_stats, scene_bins, scene_dead_zone)
    else:
        n_sectors = MODE4_NOISE_FIELD_SECTORS
        band_edges = [int(round(v)) for v in np.linspace(scene_dead_zone, scene_bins, MODE4_NOISE_FIELD_BANDS + 1)]
        band_edges[0] = scene_dead_zone
        band_edges[-1] = scene_bins

    sector_idx = min(n_sectors - 1, int((tick % TICKS_PER_REV) * n_sectors / TICKS_PER_REV))
    near_land_mask = _build_near_land_mask(scene_bins, scene_dead_zone, land_intervals)
    spoke_rng = np.random.RandomState(int(tick) * 1000 + 42)
    painted_any = False

    for seg in noise_segments:
        if not isinstance(seg, (list, tuple)) or len(seg) < 2:
            continue
        start_bin = int(seg[0])
        vals = seg[1] if isinstance(seg[1], (list, tuple)) else []
        seg_end = min(scene_bins, start_bin + len(vals))
        if seg_end <= start_bin:
            continue

        run_start = max(scene_dead_zone, start_bin)
        while run_start < seg_end:
            family = "near_land" if near_land_mask[run_start] else "open_water"
            band_idx = _band_index_for_bin(run_start, band_edges)
            run_end = run_start + 1
            while run_end < seg_end:
                if near_land_mask[run_end] != near_land_mask[run_start]:
                    break
                if _band_index_for_bin(run_end, band_edges) != band_idx:
                    break
                run_end += 1

            _, _, base_code = _choose_noise_histogram_codes(
                spoke_rng,
                family,
                band_idx,
                run_start,
                noise_stats,
                legacy_noise_codes,
                legacy_noise_weights,
                sector_idx,
            )
            base_code = _round_clamped_code(_apply_display_db_correction(
                float(base_code),
                _noise_code_correction_db(
                    family,
                    float(run_start) * float(scene_bin_size),
                ),
            ))
            code_idx = int(np.where(FURUNO_GAIN_CODES == base_code)[0][0]) if base_code in FURUNO_GAIN_SET else None

            run_len = max(1, run_end - run_start)
            for offset, bb in enumerate(range(run_start, run_end)):
                if bb < scene_dead_zone:
                    continue
                if echoes[bb] > MODE4_NOISE_MAX_EXISTING_ECHO or (
                    object_support_mask is not None and object_support_mask[bb]
                ):
                    continue
                code = base_code
                if code_idx is not None and offset > 0:
                    jitter = int(spoke_rng.choice([-1, 0, 1]))
                    alt_idx = code_idx + jitter
                    if 0 <= alt_idx < len(FURUNO_GAIN_CODES):
                        code = int(FURUNO_GAIN_CODES[alt_idx])
                rel_pos = float(offset) / float(max(1, run_len - 1)) if run_len > 1 else 0.5
                code = _round_clamped_code(_apply_display_db_correction(
                    float(code),
                    _noise_support_correction_db(
                        family,
                        run_len,
                        rel_pos,
                        float(bb) * float(scene_bin_size),
                    ),
                ))
                echoes[bb] = max(echoes[bb], code)
                painted_any = True

            run_start = run_end

    return painted_any


def _paint_dead_zone_support_segments(echoes: np.ndarray, tick: int,
                                      scene_dead_zone: int, dz_segments: list,
                                      dz_code_dist: list[tuple[int, float]],
                                      dz_bin_code_dists: Optional[dict[int, list[tuple[int, float]]]] = None) -> bool:
    if not dz_segments:
        return False
    dz_rng = np.random.RandomState(int(tick) * 1000 + 7)
    if dz_code_dist:
        dz_codes = np.array([int(c) for c, _ in dz_code_dist], dtype=int)
        dz_weights = np.array([float(w) for _, w in dz_code_dist], dtype=float)
        dz_weights = dz_weights / dz_weights.sum() if dz_weights.sum() > 0 else None
    else:
        dz_codes = np.array([8], dtype=int)
        dz_weights = None

    painted_any = False
    for seg in dz_segments:
        if not isinstance(seg, (list, tuple)) or len(seg) < 2:
            continue
        start_bin = int(seg[0])
        vals = seg[1] if isinstance(seg[1], (list, tuple)) else []
        seg_len = len(vals)
        for vi, _ in enumerate(vals):
            b = start_bin + vi
            if 0 <= b < scene_dead_zone:
                local_dist = (dz_bin_code_dists or {}).get(b)
                if local_dist:
                    local_codes = np.array([int(c) for c, _ in local_dist], dtype=int)
                    local_weights = np.array([float(w) for _, w in local_dist], dtype=float)
                    local_weights = local_weights / local_weights.sum() if local_weights.sum() > 0 else None
                    if local_weights is not None:
                        code = int(dz_rng.choice(local_codes, p=local_weights))
                    else:
                        code = int(dz_rng.choice(local_codes))
                elif dz_weights is not None:
                    code = int(dz_rng.choice(dz_codes, p=dz_weights))
                else:
                    code = int(dz_rng.choice(dz_codes))
                code = _round_clamped_code(_apply_display_db_correction(
                    float(code),
                    _dead_zone_support_correction_db(
                        b,
                        scene_dead_zone,
                        seg_len,
                        float(vi) / float(max(1, seg_len - 1)) if seg_len > 1 else 0.5,
                    ),
                ))
                echoes[b] = max(echoes[b], code)
                painted_any = True
    return painted_any


def _build_simulated_object_support_profile(obj, cal: dict, length_bins: int) -> np.ndarray:
    """
    Build a synthetic within-support radial profile from object class and
    calibrated spread, without defaulting to replayed captured amplitudes.
    """
    n = max(1, int(length_bins))
    if n == 1:
        return np.ones(1, dtype=np.float64)

    spread_bins = max(1.0, float(cal.get('spread_bins', 2) or 2))
    type_cfg = {
        "buoy": {"plateau_frac": 0.00, "gamma": 2.8, "sigma_scale": 0.45},
        "small_boat": {"plateau_frac": 0.10, "gamma": 2.2, "sigma_scale": 0.55},
        "medium_boat": {"plateau_frac": 0.18, "gamma": 1.8, "sigma_scale": 0.70},
        "large_boat": {"plateau_frac": 0.28, "gamma": 1.5, "sigma_scale": 0.85},
        "barge": {"plateau_frac": 0.38, "gamma": 1.3, "sigma_scale": 1.00},
    }
    cfg = type_cfg.get(obj.obj_type, type_cfg["medium_boat"])

    x = np.linspace(0.0, 1.0, n)
    plateau_frac = float(_clamp(cfg["plateau_frac"] + 0.02 * min(spread_bins, 10.0), 0.0, 0.55))
    left_end = max(0.05, 0.5 - plateau_frac / 2.0)
    right_start = min(0.95, 0.5 + plateau_frac / 2.0)
    gamma = max(1.1, float(cfg["gamma"]))

    profile = np.ones(n, dtype=np.float64)
    left_mask = x < left_end
    right_mask = x > right_start
    if np.any(left_mask):
        profile[left_mask] = np.power(np.clip(x[left_mask] / left_end, 0.0, 1.0), gamma)
    if np.any(right_mask):
        profile[right_mask] = np.power(
            np.clip((1.0 - x[right_mask]) / max(1e-6, 1.0 - right_start), 0.0, 1.0),
            gamma,
        )

    sigma = max(0.18, min(0.95, cfg["sigma_scale"] * spread_bins / max(2.0, n)))
    gaussian = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
    profile = 0.55 * profile + 0.45 * gaussian
    profile = np.clip(profile, 0.0, None)
    peak = float(np.max(profile))
    return profile / peak if peak > 0 else np.ones(n, dtype=np.float64)


def _build_object_learned_segment_features(
    obj,
    raw_obj: Optional[dict],
    bearing_deg: float,
    obj_bearing_deg: float,
    obj_range_m: float,
    beam_offset_deg: float,
    seg_len: int,
    start_bin: int,
    end_bin: int,
    scene_bin_size: float,
    baseline_segment_peak_code: float,
    baseline_segment_peak_db: float,
) -> dict:
    raw = raw_obj or {}
    obj_type = raw.get("type") or raw.get("geom_type") or getattr(obj, "obj_type", "unknown")
    geom_type = raw.get("geom_type", obj_type)
    material_proxy = raw.get("material", geom_type)
    length_m = float(getattr(obj, "length_m", raw.get("length_m", 0.0)) or 0.0)
    width_m = float(getattr(obj, "width_m", raw.get("width_m", 0.0)) or 0.0)
    radial_bins = float(raw.get("radial_bins", 0.0) or 0.0)
    angular_spokes = float(raw.get("angular_spokes", 0.0) or 0.0)
    obj_heading_deg = float(getattr(obj, "heading_deg", raw.get("heading_deg", 0.0)) or 0.0)
    aspect_to_spoke_deg = _normalized_angle_diff_deg(bearing_deg, obj_heading_deg)
    aspect_to_radar_deg = _normalized_angle_diff_deg(obj_bearing_deg, obj_heading_deg)
    projected_radial_extent_m, projected_cross_range_extent_m = _projected_object_extents(
        length_m,
        width_m,
        aspect_to_spoke_deg,
    )
    support_len_m = float(seg_len) * float(scene_bin_size)
    support_density = _safe_ratio_float(
        float(seg_len),
        max(1.0, radial_bins * max(1.0, angular_spokes)),
        default=0.0,
    )
    return {
        "object_type": obj_type,
        "geom_type": geom_type,
        "material_proxy": material_proxy,
        "object_range_m": float(obj_range_m),
        "object_bearing_deg": float(obj_bearing_deg),
        "beam_offset_deg": float(beam_offset_deg),
        "abs_beam_offset_deg": abs(float(beam_offset_deg)),
        "object_heading_deg": obj_heading_deg,
        "aspect_to_spoke_deg": aspect_to_spoke_deg,
        "aspect_to_radar_deg": aspect_to_radar_deg,
        "rcs_m2": float(getattr(obj, "rcs_m2", raw.get("rcs_m2", 0.0)) or 0.0),
        "length_m": length_m,
        "width_m": width_m,
        "length_width_ratio": _safe_ratio_float(length_m, max(width_m, 0.1), default=0.0),
        "radial_bins": radial_bins,
        "angular_spokes": angular_spokes,
        "support_len_bins": float(seg_len),
        "support_len_m": support_len_m,
        "support_density": support_density,
        "projected_radial_extent_m": projected_radial_extent_m,
        "projected_cross_range_extent_m": projected_cross_range_extent_m,
        "projected_radial_bins": _safe_ratio_float(projected_radial_extent_m, scene_bin_size, default=0.0),
        "projected_cross_range_bins": _safe_ratio_float(projected_cross_range_extent_m, scene_bin_size, default=0.0),
        "segment_start_bin": float(start_bin),
        "segment_end_bin": float(max(start_bin, end_bin - 1)),
        "segment_center_bin": 0.5 * float(start_bin + max(start_bin, end_bin - 1)),
        "baseline_segment_peak_code": float(baseline_segment_peak_code),
        "baseline_segment_peak_db": float(baseline_segment_peak_db),
    }


def _predict_object_learned_peak_delta_db(learned_object_model: Optional[dict], feature_row: dict) -> float:
    if not learned_object_model or not _HAS_PANDAS:
        return 0.0
    model = learned_object_model.get("peak_model")
    if model is None:
        return 0.0
    limit = float(learned_object_model.get("peak_correction_db_limit", MODE4_OBJECT_PEAK_CORRECTION_LIMIT_DB))
    try:
        frame = _pd.DataFrame([feature_row])
        pred = float(model.predict(frame)[0])
    except Exception:
        return 0.0
    return float(_clamp(pred, -limit, limit))


def _predict_object_learned_shape_delta_db(learned_object_model: Optional[dict],
                                           feature_rows: list[dict]) -> np.ndarray:
    if not feature_rows or not learned_object_model or not _HAS_PANDAS:
        return np.zeros(len(feature_rows), dtype=np.float64)
    model = learned_object_model.get("shape_model")
    if model is None:
        return np.zeros(len(feature_rows), dtype=np.float64)
    limit = float(learned_object_model.get("shape_correction_db_limit", MODE4_OBJECT_SHAPE_CORRECTION_LIMIT_DB))
    try:
        frame = _pd.DataFrame(feature_rows)
        pred = np.asarray(model.predict(frame), dtype=np.float64)
    except Exception:
        return np.zeros(len(feature_rows), dtype=np.float64)
    return np.clip(pred, -limit, limit)


def _paint_object_echo_on_support(obj, bearing_rad: float, cal: dict,
                                  support_segments: list, scene_bins: int = NUM_BINS,
                                  scene_bin_size: float = BIN_SIZE_M,
                                  scene_dead_zone: int = DEAD_ZONE_BINS,
                                  raw_obj: Optional[dict] = None,
                                  learned_object_model: Optional[dict] = None) -> np.ndarray:
    """
    Strict Mode 4 object rendering: preserve the reference spoke/bin support,
    but simulate the amplitudes inside that support from the calibrated object
    model and reference-informed shape templates.
    """
    echoes = np.zeros(scene_bins, dtype=np.float64)
    if not support_segments:
        return echoes

    obj_range = math.sqrt(obj.x ** 2 + obj.y ** 2)
    obj_bearing = math.atan2(obj.x, obj.y)
    obj_bearing_deg = math.degrees(obj_bearing) % 360.0
    bearing_deg = math.degrees(bearing_rad) % 360.0
    d_angle = bearing_rad - obj_bearing
    d_angle = (d_angle + math.pi) % (2 * math.pi) - math.pi
    d_angle_deg = math.degrees(d_angle)
    half_beam = BEAMWIDTH_DEG / 2
    if abs(d_angle_deg) > half_beam + 1.0:
        return echoes

    beam_gain = math.exp(-2.77 * (d_angle_deg / half_beam) ** 2)
    ref_range = 50.0
    ref_rcs = 350.0
    ref_code = float(cal.get('calibrated_ref_code', 190.0))
    compression = float(cal.get('calibrated_compression', 55.0))

    if obj_range < scene_dead_zone * scene_bin_size or obj.rcs_m2 <= 0 or obj_range <= 0:
        return echoes

    intensity_dB = (10 * math.log10(obj.rcs_m2 / ref_rcs) -
                    40 * math.log10(obj_range / ref_range))
    peak_code = ref_code * (10 ** (intensity_dB / compression))
    peak_db = (
        _display_db_from_code_value(peak_code) +
        _shape_fraction_drop_db(beam_gain) +
        _object_peak_correction_db(obj.obj_type, obj_range)
    )
    peak_code = _code_value_from_display_db(peak_db)
    peak_code = min(252.0, max(0.0, peak_code))
    if peak_code < 4:
        return echoes

    for seg in support_segments:
        start_bin = int(seg.get("start_bin", 0))
        vals = seg.get("values", []) if isinstance(seg, dict) else []
        seg_len = len(vals)
        if seg_len <= 0:
            continue
        end_bin = min(scene_bins, start_bin + seg_len)
        if end_bin <= start_bin:
            continue
        n = end_bin - start_bin

        local_profile = _build_simulated_object_support_profile(obj, cal, n)
        active_floor = 8.0 if peak_code > 0 else None
        segment_bins: list[int] = []
        baseline_values: list[float] = []
        baseline_dbs: list[float] = []
        shape_feature_rows: list[dict] = []
        for offset, b in enumerate(range(start_bin, end_bin)):
            if b < scene_dead_zone:
                continue
            rel_pos = float(offset) / float(max(1, n - 1)) if n > 1 else 0.5
            baseline_val = _code_from_peak_db_shape(
                peak_db,
                float(local_profile[offset]),
                _object_support_correction_db(obj.obj_type, n, rel_pos),
                floor_code=active_floor,
            )
            baseline_code = float(_round_clamped_code(baseline_val))
            baseline_db = _display_db_from_code_value(baseline_code)
            segment_bins.append(b)
            baseline_values.append(float(baseline_val))
            baseline_dbs.append(float(baseline_db))
            if learned_object_model:
                phase = _object_support_phase_label(offset, n)
                feature_row = {
                    "baseline_db": baseline_db,
                    "baseline_code": baseline_code,
                    "range_m": float(b) * float(scene_bin_size),
                    "log_range_m": math.log(max(float(b) * float(scene_bin_size), 1.0)),
                    "bin_idx": float(b),
                    "support_len_bins": float(n),
                    "support_len_m": float(n) * float(scene_bin_size),
                    "depth_from_entry_bins": float(offset),
                    "depth_to_exit_bins": float(max(0, n - 1 - offset)),
                    "edge_distance_bins": float(min(offset, max(0, n - 1 - offset))),
                    "rel_pos": rel_pos,
                    "phase": phase,
                    "normalized_center_distance": abs(rel_pos - 0.5) * 2.0,
                    "is_short_support": 1.0 if n <= 2 else 0.0,
                    "is_edge_bin": 1.0 if offset == 0 or offset == n - 1 else 0.0,
                }
                shape_feature_rows.append(feature_row)

        if not segment_bins:
            continue

        baseline_segment_peak_code = max(float(_round_clamped_code(v)) for v in baseline_values)
        baseline_segment_peak_db = _display_db_from_code_value(baseline_segment_peak_code)
        segment_context = _build_object_learned_segment_features(
            obj,
            raw_obj,
            bearing_deg,
            obj_bearing_deg,
            obj_range,
            d_angle_deg,
            n,
            start_bin,
            end_bin,
            scene_bin_size,
            baseline_segment_peak_code,
            baseline_segment_peak_db,
        )
        peak_delta_db = _predict_object_learned_peak_delta_db(learned_object_model, segment_context)

        if learned_object_model and shape_feature_rows:
            for row in shape_feature_rows:
                row.update(segment_context)
                row["baseline_drop_from_segment_peak_db"] = baseline_segment_peak_db - float(row["baseline_db"])
            shape_delta_db = _predict_object_learned_shape_delta_db(learned_object_model, shape_feature_rows)
        else:
            shape_delta_db = np.zeros(len(segment_bins), dtype=np.float64)

        for idx, b in enumerate(segment_bins):
            val = _code_value_from_display_db(
                baseline_dbs[idx] + peak_delta_db + float(shape_delta_db[idx])
            )
            if val >= 6:
                echoes[b] = max(echoes[b], val)

    return echoes


def _paint_object_echo_on_support(obj, bearing_rad: float, cal: dict,
                                  support_segments: list, scene_bins: int = NUM_BINS,
                                  scene_bin_size: float = BIN_SIZE_M,
                                  scene_dead_zone: int = DEAD_ZONE_BINS,
                                  raw_obj: Optional[dict] = None,
                                  learned_object_model: Optional[dict] = None) -> np.ndarray:
    """Render strict Mode 4 support segments with the shared object renderer."""
    return _render_object_echo_shared(
        obj,
        bearing_rad,
        scene_bins=scene_bins,
        scene_bin_size=scene_bin_size,
        scene_dead_zone=scene_dead_zone,
        sim_mode=SIM_MODE_CURRENT_MODE4,
        raw_obj=raw_obj,
        cal=cal,
        support_segments=support_segments,
        learned_object_model=learned_object_model,
    ).echoes


def _paint_structured_noise(echoes: np.ndarray, tick: int, scene_bins: int,
                            scene_dead_zone: int, scene_bin_size: float,
                            noise_stats: Optional[dict],
                            legacy_noise_prob: float, legacy_noise_codes: np.ndarray,
                            legacy_noise_weights: Optional[List[float]],
                            object_support_mask: Optional[np.ndarray],
                            land_intervals: List[Tuple[int, int]]) -> bool:
    if not noise_stats:
        return False
    n_sectors, n_bands, band_edges = _noise_field_layout(noise_stats, scene_bins, scene_dead_zone)
    sector_idx = min(n_sectors - 1, int((tick % TICKS_PER_REV) * n_sectors / TICKS_PER_REV))
    near_land_mask = _build_near_land_mask(scene_bins, scene_dead_zone, land_intervals)
    spoke_rng = np.random.RandomState(int(tick) * 1000 + 42)
    painted_any = False

    for band_idx in range(n_bands):
        band_start = max(scene_dead_zone, band_edges[band_idx])
        band_end = min(scene_bins, band_edges[band_idx + 1])
        b = band_start
        while b < band_end:
            if b < scene_dead_zone or echoes[b] > MODE4_NOISE_MAX_EXISTING_ECHO or (
                object_support_mask is not None and object_support_mask[b]
            ):
                b += 1
                continue
            family = "near_land" if near_land_mask[b] else "open_water"
            cell = _select_noise_cell(noise_stats, family, sector_idx, band_idx)
            if cell is None:
                occ = float(max(0.0, legacy_noise_prob))
                median_run_len = 1.0
                hist_codes = legacy_noise_codes if legacy_noise_codes.size > 0 else None
                if legacy_noise_weights and len(legacy_noise_weights) == len(legacy_noise_codes):
                    w = np.array(legacy_noise_weights, dtype=float)
                    hist_weights = w / np.sum(w) if np.sum(w) > 0 else None
                else:
                    hist_weights = None
            else:
                occ = float(cell.get("occupancy_probability", 0.0) or 0.0)
                median_run_len = float(cell.get("median_run_len", 1.0) or 1.0)
                hist_codes, hist_weights = _histogram_to_arrays(cell.get("code_histogram") or {}, family)
            range_m = float(b) * float(scene_bin_size)
            p_start = occ / max(1.0, median_run_len)
            p_start *= _noise_probability_scale_for_family(family, range_m)
            p_start = float(_clamp(p_start, 0.0, 1.0))
            if hist_codes is None or hist_weights is None:
                if legacy_noise_codes.size == 0:
                    b += 1
                    continue
                hist_codes = legacy_noise_codes
                if legacy_noise_weights and len(legacy_noise_weights) == len(legacy_noise_codes):
                    w = np.array(legacy_noise_weights, dtype=float)
                    hist_weights = w / np.sum(w) if np.sum(w) > 0 else None
                else:
                    hist_weights = None
            if hist_codes is None or len(hist_codes) == 0:
                b += 1
                continue
            if spoke_rng.random() >= p_start:
                b += 1
                continue
            run_len = int(_clamp(round(median_run_len), 1, 4))
            if hist_weights is not None:
                base_code = int(spoke_rng.choice(hist_codes, p=hist_weights))
            else:
                base_code = int(spoke_rng.choice(hist_codes))
            base_code = _round_clamped_code(_apply_display_db_correction(
                float(base_code),
                _noise_code_correction_db(family, range_m),
            ))
            code_idx = int(np.where(FURUNO_GAIN_CODES == base_code)[0][0]) if base_code in FURUNO_GAIN_SET else None
            run_painted = 0
            for offset in range(run_len):
                bb = b + offset
                if bb >= band_end or bb >= scene_bins:
                    break
                if near_land_mask[bb] != near_land_mask[b]:
                    break
                if echoes[bb] > MODE4_NOISE_MAX_EXISTING_ECHO or (
                    object_support_mask is not None and object_support_mask[bb]
                ):
                    break
                code = base_code
                if code_idx is not None and offset > 0:
                    jitter = int(spoke_rng.choice([-1, 0, 1]))
                    alt_idx = code_idx + jitter
                    if 0 <= alt_idx < len(FURUNO_GAIN_CODES):
                        code = int(FURUNO_GAIN_CODES[alt_idx])
                rel_pos = float(offset) / float(max(1, run_len - 1)) if run_len > 1 else 0.5
                code = _round_clamped_code(_apply_display_db_correction(
                    float(code),
                    _noise_support_correction_db(
                        family,
                        run_len,
                        rel_pos,
                        float(bb) * float(scene_bin_size),
                    ),
                ))
                echoes[bb] = max(echoes[bb], code)
                run_painted += 1
            if run_painted > 0:
                painted_any = True
                b += run_painted
            else:
                b += 1

    return painted_any


def _paint_legacy_noise(echoes: np.ndarray, scene_bins: int, scene_dead_zone: int,
                        noise_prob: float, noise_codes_arr: np.ndarray,
                        noise_weights: Optional[List[float]],
                        rng: np.random.RandomState,
                        object_support_mask: Optional[np.ndarray] = None) -> None:
    noise_mask = rng.random(scene_bins) < noise_prob
    noise_mask[:scene_dead_zone] = False
    noise_mask[echoes > MODE4_NOISE_MAX_EXISTING_ECHO] = False
    if object_support_mask is not None:
        noise_mask[object_support_mask] = False
    if noise_weights and len(noise_weights) == len(noise_codes_arr):
        w = np.array(noise_weights, dtype=float)
        w = w / w.sum()
        noise_values = rng.choice(noise_codes_arr, size=scene_bins, p=w)
    else:
        noise_values = rng.choice(noise_codes_arr, size=scene_bins)
    echoes[noise_mask] = noise_values[noise_mask]


def _paint_mode4_coastline_interval(echoes: np.ndarray, entry_bin: int, exit_bin: int,
                                    scene_bin_size: float, scene_dead_zone: int,
                                    scene_bins: int, region_stats: dict,
                                    clip_to_support: bool = False) -> tuple[int, int] | tuple[None, None]:
    if exit_bin < entry_bin:
        return None, None
    paint_start = max(scene_dead_zone, int(entry_bin))
    paint_end = min(scene_bins - 1, int(exit_bin))
    if paint_end < paint_start:
        return None, None

    ramp_up_seq = [int(v) for v in _coastline_stat_or_default(region_stats, "median_ramp_up", [104, 188])]
    ramp_down_seq = [int(v) for v in _coastline_stat_or_default(region_stats, "median_ramp_down", [140, 56])]
    mean_core_echo = float(_coastline_stat_or_default(region_stats, "mean_core_echo", 252.0))
    typical_ramp_up_bins = int(_coastline_stat_or_default(region_stats, "typical_ramp_up_bins", max(2, len(ramp_up_seq))))
    segment_depth_default = max(1, paint_end - paint_start + 1)
    median_segment_depth = float(
        _coastline_stat_or_default(region_stats, "median_segment_depth_bins", segment_depth_default)
    )
    entry_peak = min(252.0, max(0.0, mean_core_echo))
    core_depth_bins = int(_clamp(
        max(typical_ramp_up_bins + 2, round(median_segment_depth * 0.15)),
        MODE4_COASTLINE_CORE_DEPTH_MIN_BINS,
        MODE4_COASTLINE_CORE_DEPTH_MAX_BINS,
    ))

    if clip_to_support:
        num_bins = paint_end - paint_start + 1
        if num_bins <= 0:
            return None, None

        up_len = min(len(ramp_up_seq), num_bins)
        down_len = min(len(ramp_down_seq), max(0, num_bins - up_len))
        while up_len + down_len > num_bins:
            if down_len >= up_len and down_len > 0:
                down_len -= 1
            elif up_len > 0:
                up_len -= 1
            else:
                break

        values = np.zeros(num_bins, dtype=np.float64)
        entry_range = max(paint_start, scene_dead_zone + 1) * scene_bin_size
        for bi in range(num_bins):
            depth_from_entry = bi
            rel_pos = float(bi) / float(max(1, num_bins - 1)) if num_bins > 1 else 0.5
            if depth_from_entry < core_depth_bins:
                val = entry_peak
            else:
                b_range = max((paint_start + bi + 1) * scene_bin_size, entry_range)
                decay_ratio = b_range / max(entry_range, 1.0)
                decay_db = 40.0 * math.log10(max(1.001, decay_ratio))
                log_amp_decay = 10 ** (-decay_db / MODE4_COASTLINE_DECAY_DB_DIVISOR)
                terrain_atten = math.exp(
                    -MODE4_COASTLINE_TERRAIN_ABSORB_PER_BIN * (depth_from_entry - core_depth_bins + 1)
                )
                val = entry_peak * log_amp_decay * terrain_atten
                val = _apply_display_db_correction(
                    val,
                    _coastline_interior_correction_db(
                        b_range,
                        depth_from_entry,
                        num_bins,
                        core_depth_bins,
                        rel_pos,
                    ),
                )
            values[bi] = val

        for bi in range(up_len):
            values[bi] = float(ramp_up_seq[bi])
        for bi in range(down_len):
            values[num_bins - down_len + bi] = min(
                values[num_bins - down_len + bi],
                float(ramp_down_seq[bi]),
            )

        for bi, b in enumerate(range(paint_start, paint_end + 1)):
            ival = _round_clamped_code(values[bi])
            if ival >= 6:
                echoes[b] = max(echoes[b], ival)
        return paint_start, paint_end

    for ri, rv in enumerate(ramp_up_seq):
        b = paint_start - len(ramp_up_seq) + ri
        if scene_dead_zone <= b < scene_bins:
            val = float(rv)
            ival = _round_clamped_code(val)
            if ival >= 6:
                echoes[b] = max(echoes[b], ival)

    entry_range = max(paint_start, scene_dead_zone + 1) * scene_bin_size
    support_len_bins = paint_end - paint_start + 1
    for bi, b in enumerate(range(paint_start, paint_end + 1)):
        depth_from_entry = bi
        rel_pos = float(bi) / float(max(1, support_len_bins - 1)) if support_len_bins > 1 else 0.5
        if depth_from_entry < core_depth_bins:
            val = entry_peak
        else:
            b_range = max((b + 1) * scene_bin_size, entry_range)
            decay_ratio = b_range / max(entry_range, 1.0)
            decay_db = 40.0 * math.log10(max(1.001, decay_ratio))
            log_amp_decay = 10 ** (-decay_db / MODE4_COASTLINE_DECAY_DB_DIVISOR)
            terrain_atten = math.exp(
                -MODE4_COASTLINE_TERRAIN_ABSORB_PER_BIN * (depth_from_entry - core_depth_bins + 1)
            )
            val = entry_peak * log_amp_decay * terrain_atten
            val = _apply_display_db_correction(
                val,
                _coastline_interior_correction_db(
                    b_range,
                    depth_from_entry,
                    support_len_bins,
                    core_depth_bins,
                    rel_pos,
                ),
            )
        ival = _round_clamped_code(val)
        if ival >= 6:
            echoes[b] = max(echoes[b], ival)

    for ri, rv in enumerate(ramp_down_seq):
        b = paint_end + 1 + ri
        if scene_dead_zone <= b < scene_bins:
            val = float(rv)
            ival = _round_clamped_code(val)
            if ival >= 6:
                echoes[b] = max(echoes[b], ival)

    return paint_start, paint_end


def _apply_mode4_population_defaults(scene: dict, objects: List[RadarObject],
                                     calibration: dict) -> None:
    """Replace scene-specific intensity defaults with corpus-level predictive defaults."""
    coast_cfg = calibration.get("coastline", {})
    coast_up = coast_cfg.get("ramp_up_codes_p50", [220, 252, 252, 252, 252, 252])
    coast_down = coast_cfg.get("ramp_down_codes_p50", [252, 252, 252, 252, 252, 164])
    core_echo = coast_cfg.get("core_echo_p50", 252.0)
    median_segment_depth = coast_cfg.get("median_segment_depth_bins_p50")
    p75_segment_depth = coast_cfg.get("p75_segment_depth_bins_p50")
    tuning = calibration.get("predictive_tuning", {}) or {}
    coast_scene_weight = float(tuning.get("coastline_scene_weight", 0.0))
    coast_core_scale = float(tuning.get("coastline_core_scale", 1.0))
    coast_ramp_scale = float(tuning.get("coastline_ramp_scale", 1.0))
    noise_hist_scene_weight = float(tuning.get("noise_hist_scene_weight", 0.0))
    noise_probability_scale = float(tuning.get("noise_probability_scale", 1.0))
    object_rcs_scene_weight = float(tuning.get("object_rcs_scene_weight", 0.0))
    object_rcs_scale = float(tuning.get("object_rcs_scale", 1.0))

    existing_stats = scene.get("coastline_stats", {}) or {}
    coastline_names = [c.get("name", f"coastline_{i}") for i, c in enumerate(scene.get("coastlines", []))]
    if not coastline_names and existing_stats:
        coastline_names = list(existing_stats.keys())
    scene["coastline_stats"] = {}
    for name in coastline_names:
        scene_stat = existing_stats.get(name, {}) if isinstance(existing_stats.get(name, {}), dict) else {}
        blended_up = _blend_code_sequence(
            scene_stat.get("median_ramp_up"), list(coast_up),
            coast_scene_weight, scale=coast_ramp_scale)
        blended_down = _blend_code_sequence(
            scene_stat.get("median_ramp_down"), list(coast_down),
            coast_scene_weight, scale=coast_ramp_scale)
        scene_core = scene_stat.get("mean_core_echo")
        blended_core = _blend_scalar(scene_core, float(core_echo), coast_scene_weight) * coast_core_scale
        scene_median_depth = scene_stat.get("median_segment_depth_bins")
        scene_p75_depth = scene_stat.get("p75_segment_depth_bins")
        scene["coastline_stats"][name] = {
            "median_ramp_up": blended_up,
            "median_ramp_down": blended_down,
            "mean_core_echo": float(min(252.0, max(0.0, blended_core))),
            "typical_ramp_up_bins": int(scene_stat.get("typical_ramp_up_bins", 2) or 2),
            "typical_ramp_down_bins": int(scene_stat.get("typical_ramp_down_bins", 2) or 2),
            "median_segment_depth_bins": _blend_scalar(scene_median_depth, float(median_segment_depth), coast_scene_weight)
            if median_segment_depth is not None else float(scene_median_depth or 12.0),
            "p75_segment_depth_bins": _blend_scalar(scene_p75_depth, float(p75_segment_depth), coast_scene_weight)
            if p75_segment_depth is not None else float(scene_p75_depth or scene_median_depth or 18.0),
            "segment_count": int(scene_stat.get("segment_count", 0) or 0),
        }

    noise_cfg = calibration.get("noise", {})
    corpus_noise_field_stats = calibration.get("noise_field_stats") or {}
    clutter = scene.setdefault("clutter", {})
    corpus_noise_prob = float(
        noise_cfg.get("noise_probability_p50",
                      clutter.get("noise_probability", GAIN * 0.0001))
    )
    scene_noise_prob = clutter.get("noise_probability", None)
    if scene_noise_prob is not None:
        # Predictive mode should stay corpus-informed, but the training corpus
        # can overstate clutter on sparse scenes. Cap the corpus median by the
        # extracted scene's aggregate clutter rate to avoid flooding empty bins
        # while still using the corpus code distribution and nonzero prior.
        corpus_noise_prob = min(corpus_noise_prob, float(scene_noise_prob))
    clutter["noise_probability"] = max(0.0, corpus_noise_prob * noise_probability_scale)
    noise_hist = noise_cfg.get("noise_codes_histogram", {})
    blended_codes, blended_weights = _blend_histograms(
        clutter.get("noise_codes"), clutter.get("noise_weights"),
        noise_hist, noise_hist_scene_weight)
    if blended_codes and blended_weights:
        clutter["noise_codes"] = blended_codes
        clutter["noise_weights"] = blended_weights
    elif noise_hist:
        codes = sorted(int(k) for k in noise_hist.keys())
        weights = [float(noise_hist[str(code)]) for code in codes]
        clutter["noise_codes"] = codes
        clutter["noise_weights"] = weights
    scene_noise_field_stats = clutter.get("noise_field_stats")
    blended_noise_field_stats = _blend_noise_field_stats(
        scene_noise_field_stats,
        corpus_noise_field_stats,
        noise_hist_scene_weight,
        noise_probability_scale,
    )
    if blended_noise_field_stats:
        clutter["noise_field_stats"] = blended_noise_field_stats

    scene_specs = scene.get("objects", [])
    obj_cfg = calibration.get("objects", {})
    for idx, obj in enumerate(objects):
        raw = scene_specs[idx] if idx < len(scene_specs) else {}
        # Use the extractor's final class label first. The corpus is built from
        # post-Phase-1 scene JSONs whose object buckets reflect the final
        # geometry-plus-RCS classification, not the uncapped geometric envelope.
        obj_class = raw.get("type") or raw.get("geom_type") or obj.obj_type
        cls_cfg = obj_cfg.get(obj_class, {})
        p50_rcs = cls_cfg.get("rcs_m2_p50")
        if p50_rcs is not None:
            scene_rcs = raw.get("rcs_m2", obj.rcs_m2)
            blended_rcs = _blend_scalar(scene_rcs, float(p50_rcs), object_rcs_scene_weight)
            obj.rcs_m2 = max(1.0, float(blended_rcs) * object_rcs_scale)


def _prepare_scene_for_sim_mode(scene: dict, objects: List[RadarObject],
                                sim_mode: str,
                                calibration_path: Optional[str] = None,
                                verbose: bool = True) -> None:
    """Mutate a loaded scene in-place for the requested simulator mode."""
    scene["_sim_mode"] = sim_mode
    scene.pop("_mode4_population_calibration", None)
    if sim_mode == SIM_MODE_CURRENT_MODE4:
        scene["_mode42"] = True
        return

    scene.pop("_mode42", None)

    if sim_mode == SIM_MODE_M1:
        scene["coastline_stats"] = {}
        scene["decay"] = {
            "ramp_up_range": [80, 200],
            "ramp_down_range": [40, 160],
        }
        clutter = scene.setdefault("clutter", {})
        default_gain = scene.get("radar_config", {}).get("gain", GAIN)
        clutter["noise_probability"] = default_gain * 0.0001
        clutter["noise_codes"] = [8, 20, 28]
        clutter.pop("noise_weights", None)
        return

    if sim_mode == SIM_MODE_M4G:
        # Geometry-conditioned preview: keep extracted polygon geometry and
        # per-scene aggregate stats, but do not consume per-spoke replay data.
        return

    if sim_mode == SIM_MODE_M4P:
        calibration = _load_mode4_population_calibration(calibration_path)
        scene["predictive_mode"] = True
        scene["_mode4_population_calibration"] = calibration
        _apply_mode4_population_defaults(scene, objects, calibration)
        if verbose:
            print(f"  Loaded predictive calibration from: {calibration_path or MODE4_CALIBRATION_PATH}")
        return

    if sim_mode == SIM_MODE_M42:
        # Replay ceiling: per-spoke support arrays drive the painter.
        return

    raise ValueError(f"Unknown sim mode: {sim_mode}")


def simulate_from_scene_file(input_path: str, output_dir: str,
                              num_rotations: int = 1,
                              verbose: bool = True) -> List[str]:
    """Mode 42: Simulate from a scene JSON, CSV file, or folder of CSVs.

    Accepts scene JSON, a single Furuno CSV (auto-extracts scene), or a
    folder containing CSVs (uses first CSV, auto-extracts).
    """
    coastlines, objects, scene, scene_path = _resolve_scene_input(input_path, verbose)
    _prepare_scene_for_sim_mode(scene, objects, SIM_MODE_M42, verbose=verbose)
    return simulate_from_scene(scene, coastlines, objects, output_dir,
                               scene_path=scene_path,
                               num_rotations=num_rotations, verbose=verbose)


def simulate_mode42(input_path: str, output_dir: str,
                    num_rotations: int = 1,
                    verbose: bool = True) -> List[str]:
    """Current Mode 4 baseline: reference-calibrated simulation from scene input.

    Accepts scene JSON, a single Furuno CSV (auto-extracts scene), or a
    folder containing CSVs (uses first CSV, auto-extracts).

    Runs the full geometric simulation pipeline (ray-tracing, radar equation,
    Furuno quantization) but calibrates every parameter from the reference scene
    so the output approximates the original capture.
    """
    coastlines, objects, scene, scene_path = _resolve_scene_input(input_path, verbose)
    _prepare_scene_for_sim_mode(scene, objects, SIM_MODE_CURRENT_MODE4, verbose=verbose)
    if verbose:
        print("=" * 60)
        print(f"  {SIM_MODE_LABELS[SIM_MODE_CURRENT_MODE4]}")
        print("=" * 60)
        print(f"  Scene: {scene_path}")
        print(f"  Output: {output_dir}")
        print(f"  Rotations: {num_rotations}")
    return simulate_from_scene(scene, coastlines, objects, output_dir,
                               scene_path=scene_path,
                               num_rotations=num_rotations, verbose=verbose)


def simulate_scene_mode(input_path: str, output_dir: str,
                        sim_mode: str,
                        num_rotations: int = 1,
                        verbose: bool = True,
                        calibration_path: Optional[str] = None) -> List[str]:
    """Explicit sim-mode entry point for M1 / M4G / M4P / M42 / current baseline."""
    coastlines, objects, scene, scene_path = _resolve_scene_input(input_path, verbose)
    _prepare_scene_for_sim_mode(scene, objects, sim_mode,
                                calibration_path=calibration_path,
                                verbose=verbose)
    if verbose:
        print("=" * 60)
        print(f"  {SIM_MODE_LABELS.get(sim_mode, sim_mode)}")
        print("=" * 60)
        print(f"  Scene: {scene_path}")
        print(f"  Output: {output_dir}")
        print(f"  Rotations: {num_rotations}")
    return simulate_from_scene(scene, coastlines, objects, output_dir,
                               scene_path=scene_path,
                               num_rotations=num_rotations, verbose=verbose)


def simulate_from_scene(scene: dict, coastlines: List[CoastlineSegment],
                        objects: List[RadarObject], output_dir: str,
                        scene_path: str = "custom_scene",
                        num_rotations: int = 1,
                        verbose: bool = True,
                        base_time: datetime = None) -> List[str]:
    """Core simulation engine that renders radar data from a scene dict.

    Used by:
    - simulate_scene() for Mode 1 (random generation)
    - simulate_mode42() for Mode 4 (reference-calibrated simulation)
    - simulate_from_scene_file() for Mode 42 (profile-guided scene JSON)
    - simulate_custom() for Mode 5 (custom simulation builder)

    When scene['_mode42'] is True → reference-calibrated simulation (Mode 4).
    When profile data exists and is valid → profile-guided rendering (Mode 42).
    When profile data is absent → geometric rendering (ray-tracing + radar equation + noise).
    """

    sim_seed = scene.get("_sim_seed", None)
    if sim_seed is None:
        raw_seed = os.environ.get("SIM_RANDOM_SEED")
        if raw_seed not in (None, ""):
            try:
                sim_seed = int(raw_seed)
            except ValueError:
                sim_seed = None
    if sim_seed is not None:
        random.seed(int(sim_seed))
        np.random.seed(int(sim_seed))

    cfg = scene.get('radar_config', {})
    scene_gain = cfg.get('gain', GAIN)
    scene_status = cfg.get('status', STATUS)
    scene_scale = cfg.get('scale', SCALE)
    scene_range_idx = cfg.get('range_index', RANGE_INDEX)
    scene_bins = cfg.get('bins', NUM_BINS)
    scene_bin_size = cfg.get('bin_size_m', BIN_SIZE_M)
    scene_angle_ticks = scene.get('angle_ticks', None)
    scene_range_m = cfg.get('range_m', RANGE_M)

    # Dead zone: use scene-specific value if set (for custom range), else default
    scene_dead_zone = cfg.get('dead_zone_bins', DEAD_ZONE_BINS)

    clutter = scene.get('clutter', {})
    noise_prob = clutter.get('noise_probability', scene_gain * 0.0001)
    noise_codes = clutter.get('noise_codes', [8, 20, 28])
    noise_weights = clutter.get('noise_weights', None)
    noise_field_stats = clutter.get('noise_field_stats')
    if noise_codes:
        noise_codes_arr = np.array([c for c in noise_codes if c in FURUNO_GAIN_CODES.tolist()])
        if len(noise_codes_arr) == 0:
            noise_codes_arr = np.array([8, 20, 28])
    else:
        noise_codes_arr = np.array([8, 20, 28])

    decay = scene.get('decay', {})
    ramp_up_range = decay.get('ramp_up_range', [80, 200])
    ramp_down_range = decay.get('ramp_down_range', [40, 160])

    # Coastline stats for realistic echo profiles
    coastline_stats = scene.get('coastline_stats', {})

    # Coastline profiles: per-spoke bin ranges (WHERE to paint)
    coastline_profiles = scene.get('coastline_profiles', {})

    # Object echo data: per-spoke bin ranges for detected objects (WHERE to paint)
    object_echo_data = scene.get('object_echo_data', {})

    # Noise echo data: per-spoke residual non-zero bins (WHERE to paint noise)
    noise_echo_data = scene.get('noise_echo_data', {})

    # Dead zone echo data: non-zero bins in the dead zone (bins 0-30)
    dead_zone_echo_data = scene.get('dead_zone_echo_data', {})

    sim_mode = scene.get("_sim_mode", None)
    reference_calibrated = sim_mode == SIM_MODE_CURRENT_MODE4 or (
        sim_mode is None and scene.get('_mode42', False)
    )
    profile_guided = bool(coastline_profiles) and sim_mode == SIM_MODE_M42
    predictive_mode = sim_mode == SIM_MODE_M4P
    geometry_preview = sim_mode == SIM_MODE_M4G
    synthetic_mode = sim_mode == SIM_MODE_M1
    shared_mode4_forward = reference_calibrated or predictive_mode or geometry_preview

    if reference_calibrated:
        coastline_cal = _extract_coastline_calibration(scene)
        object_calibrations = _extract_object_calibration(scene, objects, scene_bin_size, scene_dead_zone)
        dead_zone_cal = _extract_dead_zone_calibration(scene)
        noise_cal = _extract_noise_calibration(scene)
        mode42_validation_log = []
        active_noise_field_stats = noise_cal.get('noise_field_stats')
        strict_support_report = {
            "coastline": {
                "reference_segments": 0,
                "geometry_matched_segments": 0,
                "geometry_unmatched_segments": 0,
                "geometry_fallback_segments": 0,
                "clipped_segments": 0,
            },
            "objects": {
                "preserved_support_objects": 0,
                "preserved_support_segments": 0,
                "legacy_geometry_fallback_objects": 0,
            },
            "noise": {
                "preserved_support_spokes": 0,
                "preserved_support_segments": 0,
                "legacy_structured_spokes": 0,
            },
            "dead_zone": {
                "preserved_support_spokes": 0,
                "preserved_support_segments": 0,
                "legacy_probability_fallback_spokes": 0,
            },
        }
    else:
        active_noise_field_stats = noise_field_stats
        strict_support_report = None

    object_support_lookup = _build_object_support_lookup(scene, scene_bins)
    object_support_by_object = _build_object_support_by_object(scene) if reference_calibrated else {}
    use_learned_object_model = reference_calibrated or predictive_mode or geometry_preview
    learned_object_model = _load_mode4_object_learned_model() if use_learned_object_model else None
    scene_object_specs = scene.get("objects", []) or []

    # Filled polygons are used by location scenes and Mode 5's editable land
    # shapes. Keep the legacy source-file fallback for older saved Mode 5 scenes.
    coastline_render_mode = str(scene.get("coastline_render_mode", "")).lower()
    mode12_filled_coastline = (
        bool(scene.get('mode12_metadata'))
        or coastline_render_mode == "filled_polygon"
        or scene.get("source_file") == "gui_scene_builder"
    )

    if verbose:
        print(f"  Scene: {len(coastlines)} coastlines, {len(objects)} objects")
        print(f"  Gain: {scene_gain}, Bins: {scene_bins}")
        if reference_calibrated:
            print(f"  Simulation mode: {SIM_MODE_LABELS[SIM_MODE_CURRENT_MODE4]}")
            n_cal = sum(1 for c in object_calibrations if c['peak_echo'] > 0) if reference_calibrated else 0
            print(f"  Objects calibrated: {n_cal}/{len(objects)}")
            print(f"  Learned object residual model: {'enabled' if learned_object_model else 'disabled'}")
            print(f"  Coastline regions: {len(coastline_cal['regions'])}")
            print(f"  Dead zone spoke probability: {dead_zone_cal['spoke_probability']:.4f}")
            print(f"  Noise avg bins/spoke: {noise_cal['avg_noise_bins_per_spoke']:.1f}")
        elif profile_guided:
            print(f"  Simulation mode: {SIM_MODE_LABELS[SIM_MODE_M42]}")
            print(f"  Coastline profiles: {len(coastline_profiles)} spokes")
            print(f"  Object profiles: {len(object_echo_data)} spokes")
            print(f"  Noise profiles: {len(noise_echo_data)} spokes")
        elif predictive_mode:
            print(f"  Simulation mode: {SIM_MODE_LABELS[SIM_MODE_M4P]}")
        elif geometry_preview:
            print(f"  Simulation mode: {SIM_MODE_LABELS[SIM_MODE_M4G]}")
        elif synthetic_mode:
            print(f"  Simulation mode: {SIM_MODE_LABELS[SIM_MODE_M1]}")
        else:
            if mode12_filled_coastline:
                print(f"  Simulation mode: filled-polygon coastlines + ray-traced objects")
            else:
                print(f"  Simulation mode: ray-tracing + radar equation + statistical noise")
        if coastline_stats:
            print(f"  Coastline stats available for {len(coastline_stats)} regions")
        if scene_angle_ticks:
            print(f"  Using {len(scene_angle_ticks)} angle ticks from scene file")

    os.makedirs(output_dir, exist_ok=True)
    if base_time is None:
        base_time = datetime.now()
    rotation_dt = 2.5

    generated_files = []
    all_sim_rows = []  # Collect for validation

    # Static editable coastline geometry does not change during an export.
    # Rasterize it once and reuse its per-spoke result for every rotation.
    agg_stats = {}
    if coastline_stats:
        agg_stats = coastline_stats[next(iter(coastline_stats))]
    default_core_echo = agg_stats.get('mean_core_echo', 252)
    output_angle_ticks = scene_angle_ticks if scene_angle_ticks else generate_angle_ticks()
    if mode12_filled_coastline:
        filled_coastline_echoes, filled_coastline_intervals = _build_filled_coastline_template(
            coastlines, output_angle_ticks, scene_bins, scene_bin_size,
            scene_dead_zone, scene_range_m, default_core_echo,
        )
    else:
        filled_coastline_echoes, filled_coastline_intervals = None, None

    for rot in range(num_rotations):
        rotation_time = base_time + timedelta(seconds=rot * rotation_dt)
        filename = generate_filename(rotation_time)
        filepath = os.path.join(output_dir, filename)
        scene_stem = os.path.splitext(filename)[0]
        meta_subdir = os.path.join(output_dir, 'scene_metadata', scene_stem)
        os.makedirs(meta_subdir, exist_ok=True)
        det_filename = scene_stem + '_detections.csv'
        det_filepath = os.path.join(meta_subdir, det_filename)

        angle_ticks = output_angle_ticks

        rows = []
        detection_rows = []

        # Get aggregate coastline stats for echo value generation
        agg_stats = {}
        if coastline_stats:
            first_key = next(iter(coastline_stats))
            agg_stats = coastline_stats[first_key]
        default_ramp_up = agg_stats.get('median_ramp_up', [104, 188])
        default_ramp_down = agg_stats.get('median_ramp_down', [140, 56])
        default_core_echo = agg_stats.get('mean_core_echo', 252)

        # Per-polygon extension fractions for Mode 1 land mass fill (azimuthally coherent)
        # Fraction of polygon thickness — e.g. a 40-bin polygon extends 12-32 extra bins
        coastline_cont_fracs = {id(c): random.uniform(0.3, 0.8) for c in coastlines}

        for spoke_idx, tick in enumerate(angle_ticks):
            spoke_offset_sec = (tick / TICKS_PER_REV) * rotation_dt
            spoke_time = rotation_time + timedelta(seconds=spoke_offset_sec)
            spoke_ts = spoke_time.strftime('%Y-%m-%dT%H:%M:%S.') + \
                       f"{spoke_time.microsecond // 1000:03d}"

            bearing_deg = tick * 360.0 / TICKS_PER_REV
            bearing_rad = math.radians(bearing_deg)
            dx = math.sin(bearing_rad)
            dy = math.cos(bearing_rad)

            echoes = np.zeros(scene_bins, dtype=np.float64)
            detections = []
            spoke_idx_str = str(spoke_idx)

            # --- Coastline returns ---
            # Primary: use coastline_profiles for WHERE to paint (per-spoke bin ranges)
            # + coastline_stats for WHAT values (ramp-up → core → ramp-down)
            # Fallback: ray-tracing for spokes without profile data
            shadow_entry = None
            profile = coastline_profiles.get(spoke_idx_str)
            if profile_guided and profile and not reference_calibrated:
                # Profile-guided rendering: paint each segment using reference values
                # from the scene profile, with statistical fallback
                land_intervals = []
                for seg in profile.get('segments', []):
                    start_bin = seg[0]
                    end_bin = seg[1]
                    ref_values = seg[2] if len(seg) > 2 else None
                    num_seg_bins = end_bin - start_bin + 1

                    if num_seg_bins < 1:
                        continue

                    paint_start = max(scene_dead_zone, start_bin)
                    paint_end = min(scene_bins - 1, end_bin)
                    if paint_end >= paint_start:
                        land_intervals.append((paint_start, paint_end))

                    if ref_values and len(ref_values) == num_seg_bins:
                        # Use per-bin reference values from the scene profile
                        for vi, val in enumerate(ref_values):
                            b = start_bin + vi
                            if 0 <= b < scene_bins:
                                echoes[b] = max(echoes[b], val)
                    else:
                        # Fallback: use statistical ramp/core/ramp pattern
                        ramp_up_len = min(len(default_ramp_up), num_seg_bins // 3)
                        ramp_down_len = min(len(default_ramp_down), num_seg_bins // 3)
                        core_len = num_seg_bins - ramp_up_len - ramp_down_len

                        if core_len < 1:
                            core_val = min(252, default_core_echo)
                            for b in range(start_bin, end_bin + 1):
                                if 0 <= b < scene_bins:
                                    echoes[b] = max(echoes[b], core_val)
                        else:
                            for ri in range(ramp_up_len):
                                b = start_bin + ri
                                rv = default_ramp_up[ri] if ri < len(default_ramp_up) else default_ramp_up[-1]
                                if 0 <= b < scene_bins:
                                    echoes[b] = max(echoes[b], rv)
                            core_val = min(252, default_core_echo)
                            core_start = start_bin + ramp_up_len
                            core_end = end_bin - ramp_down_len
                            for b in range(core_start, core_end + 1):
                                if 0 <= b < scene_bins:
                                    echoes[b] = max(echoes[b], core_val)
                            for ri in range(ramp_down_len):
                                b = end_bin - ramp_down_len + 1 + ri
                                rv = default_ramp_down[ri] if ri < len(default_ramp_down) else default_ramp_down[-1]
                                if 0 <= b < scene_bins:
                                    echoes[b] = max(echoes[b], rv)
                if land_intervals:
                    shadow_entry = min(entry for entry, _ in land_intervals)
            elif shared_mode4_forward:
                # Shared forward-rendered coastline path for current Mode 4, M4G,
                # and M4P. Strict Mode 4 preserves the reference coastline
                # support by spoke/bin, but still uses the shared forward
                # coastline renderer for the actual echo synthesis.
                land_intervals = []
                geometric_intervals = []
                geometric_intervals_by_name = {}
                cal_regions = coastline_cal.get('regions', {}) if reference_calibrated else coastline_stats
                ref_spoke_segs = coastline_cal.get('spoke_segments', {}).get(spoke_idx, []) if reference_calibrated else []
                ref_spoke_segment_stats = coastline_cal.get('spoke_segment_stats', {}).get(spoke_idx, []) if reference_calibrated else []
                strict_reference_support = bool(reference_calibrated and ref_spoke_segs)

                for coastline in coastlines:
                    region_cal = cal_regions.get(coastline.name, {}) if isinstance(cal_regions, dict) else {}
                    coastline_intervals = geometric_intervals_by_name.setdefault(coastline.name, [])
                    hits = ray_polygon_intersections(0, 0, dx, dy, coastline.points)
                    if len(hits) >= 2:

                        for pair_idx in range(0, len(hits) - 1, 2):
                            entry_m = hits[pair_idx]
                            exit_m = hits[pair_idx + 1] if pair_idx + 1 < len(hits) else entry_m + 5
                            entry_bin = int(entry_m / scene_bin_size)
                            exit_bin = min(scene_bins - 1, int(exit_m / scene_bin_size))

                            if exit_bin < entry_bin:
                                continue

                            geometric_intervals.append((entry_bin, exit_bin))
                            coastline_intervals.append((entry_bin, exit_bin))

                            if reference_calibrated and not strict_reference_support:
                                for ref_start, ref_end in ref_spoke_segs:
                                    if abs(entry_bin - ref_start) > 3 or abs(exit_bin - ref_end) > 3:
                                        if (ref_start <= exit_bin and ref_end >= entry_bin):
                                            mode42_validation_log.append(
                                                f"spoke {spoke_idx}: drift entry={entry_bin} vs ref={ref_start}, "
                                                f"exit={exit_bin} vs ref={ref_end}")

                            if strict_reference_support:
                                continue

                            local_region_cal = (
                                _select_local_coastline_stats(
                                    entry_bin,
                                    exit_bin,
                                    region_cal,
                                    ref_spoke_segment_stats,
                                )
                                if reference_calibrated else region_cal
                            )

                            # Paint this interval with the best matching local
                            # segment stats from the current spoke when available.
                            painted_interval = _paint_mode4_coastline_interval(
                                echoes,
                                entry_bin,
                                exit_bin,
                                scene_bin_size,
                                scene_dead_zone,
                                scene_bins,
                                local_region_cal,
                            )
                            if painted_interval[0] is not None:
                                land_intervals.append(painted_interval)
                            continue

                    elif len(hits) == 1 and not strict_reference_support:
                        entry_bin = max(scene_dead_zone, int(hits[0] / scene_bin_size))
                        local_region_cal = (
                            _select_local_coastline_stats(
                                entry_bin,
                                min(scene_bins - 1, entry_bin + 3),
                                region_cal,
                                ref_spoke_segment_stats,
                            )
                            if reference_calibrated else region_cal
                        )
                        painted_interval = _paint_mode4_coastline_interval(
                            echoes,
                            entry_bin,
                            min(scene_bins - 1, entry_bin + 3),
                            scene_bin_size,
                            scene_dead_zone,
                            scene_bins,
                            local_region_cal,
                        )
                        if painted_interval[0] is not None:
                            land_intervals.append(painted_interval)

                if strict_reference_support:
                    # Strict Mode 4 preserves coastline support from the
                    # original extraction, but still uses the shared coastline
                    # interval renderer to generate echo values.
                    for ref_start, ref_end in ref_spoke_segs:
                        region_name = _identify_fallback_coastline_name(
                            ref_start,
                            ref_end,
                            bearing_rad,
                            scene_bin_size,
                            coastlines,
                            geometric_intervals_by_name,
                        )
                        strict_support_report["coastline"]["reference_segments"] += 1
                        base_region_cal = (
                            cal_regions.get(region_name, {})
                            if region_name and isinstance(cal_regions, dict)
                            else agg_stats
                        )
                        if region_name:
                            strict_support_report["coastline"]["geometry_matched_segments"] += 1
                        else:
                            strict_support_report["coastline"]["geometry_unmatched_segments"] += 1
                        local_region_cal = _select_local_coastline_stats(
                            ref_start,
                            ref_end,
                            base_region_cal,
                            ref_spoke_segment_stats,
                        )
                        painted_interval = _paint_mode4_coastline_interval(
                            echoes,
                            ref_start,
                            ref_end,
                            scene_bin_size,
                            scene_dead_zone,
                            scene_bins,
                            local_region_cal,
                            clip_to_support=True,
                        )
                        if painted_interval[0] is not None:
                            land_intervals.append(painted_interval)
                            strict_support_report["coastline"]["clipped_segments"] += 1
                elif reference_calibrated:
                    # Fallback: if a real coastline segment exists on this spoke but
                    # the ray cast missed it, use the real segment bounds to place a
                    # synthesized coastline interval after the main pass. The real
                    # capture provides WHERE the missed segment lives; the Mode 4
                    # equations still determine the echo values.
                    for ref_start, ref_end in ref_spoke_segs:
                        covered = False
                        for rt_entry, rt_exit in geometric_intervals:
                            if rt_entry <= ref_end and rt_exit >= ref_start:
                                covered = True
                                break
                        if not covered:
                            fallback_name = _identify_fallback_coastline_name(
                                ref_start,
                                ref_end,
                                bearing_rad,
                                scene_bin_size,
                                coastlines,
                                geometric_intervals_by_name,
                            )
                            fallback_region_cal = (
                                cal_regions.get(fallback_name, {})
                                if fallback_name and isinstance(cal_regions, dict)
                                else agg_stats
                            )
                            local_fallback_cal = _select_local_coastline_stats(
                                ref_start,
                                ref_end,
                                fallback_region_cal,
                                ref_spoke_segment_stats,
                            )
                            painted_interval = _paint_mode4_coastline_interval(
                                echoes,
                                ref_start,
                                ref_end,
                                scene_bin_size,
                                scene_dead_zone,
                                scene_bins,
                                local_fallback_cal,
                            )
                            if painted_interval[0] is not None:
                                land_intervals.append(painted_interval)
                                strict_support_report["coastline"]["geometry_fallback_segments"] += 1
                            mode42_validation_log.append(
                                f"spoke {spoke_idx}: synthesized fallback segment "
                                f"{ref_start}-{ref_end} from profiles"
                                + (f" using {fallback_name}" if fallback_name else ""))

                # Compute terrain shadow for this spoke
                shadow_entry = None
                if land_intervals:
                    shadow_entry = min(entry for entry, _ in land_intervals)
            elif mode12_filled_coastline and filled_coastline_echoes is not None:
                # Reuse the static, beam-broadened polar coastline template.
                echoes = np.maximum(echoes, filled_coastline_echoes[spoke_idx])
                land_intervals = filled_coastline_intervals[spoke_idx]
                shadow_entry = min((entry for entry, _ in land_intervals), default=None)
            elif mode12_filled_coastline:
                # Mode 12: filled-polygon coastline rendering
                # Casts multiple sub-rays across the antenna beamwidth,
                # uses even-odd fill (all bins between intersection pairs
                # are "inside land"), applies R^-3 range attenuation and
                # Furuno gain code quantization. Produces solid filled
                # land masses with smooth beam-broadened edges and
                # natural range-dependent intensity fade.
                land_intervals = []
                half_beam = BEAMWIDTH_DEG / 2.0
                num_sub_rays = 5
                # DRS4DNXT at 0.125 NM uses high STC gain at short range.
                # The log amplifier compresses the R^-3 distributed target
                # law. Use ref_range equal to the radar range so attenuation
                # is gentle across the 231.5m field of view.
                ref_range_m = scene_range_m * 0.8

                for coastline in coastlines:
                    pts_array = np.array(coastline.points)
                    if len(pts_array) < 3:
                        continue
                    # Edge start/end points for vectorized intersection
                    p1 = pts_array[:-1]
                    p2 = pts_array[1:]
                    ex = p2[:, 0] - p1[:, 0]
                    ey = p2[:, 1] - p1[:, 1]

                    for ray_i in range(num_sub_rays):
                        # Sub-ray offset across beamwidth
                        offset_deg = -half_beam + (2.0 * half_beam * ray_i / (num_sub_rays - 1)) if num_sub_rays > 1 else 0.0
                        ray_bearing_rad = bearing_rad + math.radians(offset_deg)
                        rdx = math.sin(ray_bearing_rad)
                        rdy = math.cos(ray_bearing_rad)
                        # Cosine taper weight: center ray = 1.0, edge rays ≈ 0.7
                        ray_weight = 1.0 - 0.3 * abs(offset_deg) / (half_beam + 0.01)

                        # Vectorized ray-polygon intersection
                        denom = rdx * ey - rdy * ex
                        valid = np.abs(denom) > 1e-10
                        if not np.any(valid):
                            continue
                        p1v = p1[valid]
                        exv, eyv = ex[valid], ey[valid]
                        denomv = denom[valid]
                        t = ((p1v[:, 0]) * eyv - (p1v[:, 1]) * exv) / denomv
                        u = ((p1v[:, 0]) * rdy - (p1v[:, 1]) * rdx) / denomv
                        hit_mask = (t > 0) & (u >= 0) & (u <= 1) & (t <= scene_range_m)
                        hits = np.sort(t[hit_mask])

                        if len(hits) < 2:
                            continue

                        # Even-odd fill: between each pair of hits, ray is inside land
                        for hi in range(0, len(hits) - 1, 2):
                            enter_m = hits[hi]
                            exit_m = hits[hi + 1] if hi + 1 < len(hits) else scene_range_m
                            start_bin = max(scene_dead_zone, int(enter_m / scene_bin_size))
                            end_bin = min(scene_bins - 1, int(exit_m / scene_bin_size))

                            if end_bin < start_bin:
                                continue
                            land_intervals.append((start_bin, end_bin))

                            # Fill bins with reflectivity * range attenuation * ray weight
                            for b in range(start_bin, end_bin + 1):
                                b_range_m = max(1.0, b * scene_bin_size)
                                # R^-2 range law (compressed from R^-3 by
                                # Furuno log amplifier at short range)
                                range_atten = min(1.0, (ref_range_m / b_range_m) ** 2)
                                reflectivity = 0.75
                                intensity = reflectivity * ray_weight * range_atten

                                # Edge bin reduction (smoother polygon edges)
                                if b == start_bin or b == end_bin:
                                    intensity *= 0.75
                                    # Edge jitter for irregular coastline texture
                                    intensity *= random.uniform(0.6, 1.0)

                                # Convert 0-1 intensity to Furuno echo code
                                echo_val = intensity * default_core_echo
                                echo_val = min(252, max(0, int(echo_val)))
                                echoes[b] = max(echoes[b], echo_val)

                # Compute terrain shadow for this spoke
                shadow_entry = None
                if land_intervals:
                    shadow_entry = min(entry for entry, _ in land_intervals)

            else:
                # Fallback: ray-trace coastline polygons (Mode 1 — generic geometric)
                land_intervals = []
                for coastline in coastlines:
                    hits = ray_polygon_intersections(0, 0, dx, dy, coastline.points)
                    if len(hits) >= 2:
                        stats = coastline_stats.get(coastline.name, {})
                        if stats:
                            # Use per-region statistics from extracted scene
                            ramp_up_seq = stats.get('median_ramp_up', default_ramp_up)
                            ramp_down_seq = stats.get('median_ramp_down', default_ramp_down)
                            core_echo = stats.get('mean_core_echo', default_core_echo)
                            use_random_ramps = False
                        else:
                            # No stats: randomize ramps per bin from decay config
                            # (matches generate_sweep() behavior)
                            core_echo = random.randint(244, 252)
                            use_random_ramps = True

                        for pair_idx in range(0, len(hits) - 1, 2):
                            entry_m = hits[pair_idx]
                            exit_m = hits[pair_idx + 1] if pair_idx + 1 < len(hits) else entry_m + 5

                            # Edge irregularity: jitter the entry point by ±1-3 bins
                            # to simulate rough terrain, vegetation, and surface
                            # texture that real coastlines exhibit on radar
                            edge_jitter_bins = random.randint(-2, 2)
                            entry_bin = max(scene_dead_zone,
                                            int(entry_m / scene_bin_size) + edge_jitter_bins)
                            exit_bin = min(scene_bins - 1, int(exit_m / scene_bin_size))

                            if exit_bin < entry_bin:
                                continue

                            land_intervals.append((entry_bin, exit_bin))

                            num_core_bins = exit_bin - entry_bin + 1
                            emit_ramps = num_core_bins >= 3

                            # Extend paint region for solid land mass (per-polygon coherent)
                            # Extension proportional to polygon thickness with log-decay fade
                            if use_random_ramps and num_core_bins >= 5:
                                cont_frac = coastline_cont_fracs[id(coastline)]
                                ext_bins = int(num_core_bins * cont_frac)
                                paint_exit_bin = min(scene_bins - 1, exit_bin + ext_bins)
                            else:
                                paint_exit_bin = exit_bin

                            # Ramp-up bins before entry
                            if use_random_ramps and emit_ramps:
                                rup_len = random.randint(2, 5)
                                ramp_up_codes = np.linspace(random.randint(36, 76), random.randint(188, 220), rup_len).astype(int)
                                for ri, rc in enumerate(ramp_up_codes):
                                    b = entry_bin - rup_len + ri
                                    if scene_dead_zone <= b < scene_bins:
                                        jitter = random.randint(-12, 12)
                                        echoes[b] = max(echoes[b], min(252, max(0, rc + jitter)))
                            elif not use_random_ramps:
                                n_ramp_up = len(ramp_up_seq)
                                for ri, rv in enumerate(ramp_up_seq):
                                    b = entry_bin - n_ramp_up + ri
                                    if scene_dead_zone <= b < scene_bins:
                                        echoes[b] = max(echoes[b], rv)

                            # Core coastline bins with per-bin scintillation + depth decay
                            #
                            # Real radar coastline echo profile:
                            #   [ramp-up] → [strong leading edge, 2-6 bins] → [decay into land]
                            #
                            # The leading edge (first few bins after entry) reflects strongly.
                            # Beyond that, echo attenuates as the radar pulse penetrates
                            # terrain/vegetation. This creates the characteristic bright
                            # leading edge with fading interior seen on real PPI displays.
                            #
                            # Core depth = 3-8 bins of full intensity (varies with terrain)
                            # Then R^4 decay from the entry range, not the exit range.
                            num_paint_bins = paint_exit_bin - entry_bin + 1
                            if use_random_ramps and num_paint_bins > 6:
                                scint = fractal_noise_1d(num_paint_bins, amplitude=random.uniform(3, 8), beta=1.2)
                            else:
                                scint = np.zeros(num_paint_bins)

                            # Leading edge: strong for first few bins (terrain reflectivity)
                            core_depth = random.randint(3, 8)  # bins of strong return
                            entry_range = max(entry_bin, scene_dead_zone + 1) * scene_bin_size

                            for bi, b in enumerate(range(entry_bin, paint_exit_bin + 1)):
                                if scene_dead_zone <= b < scene_bins:
                                    depth_from_entry = bi  # bins past the leading edge
                                    if depth_from_entry < core_depth:
                                        # Strong leading edge — full core echo
                                        val = min(252, max(200, core_echo + scint[bi]))
                                    else:
                                        # Depth decay — echo attenuates into land mass
                                        # Using R^4 decay from entry point, compressed
                                        # through the Furuno log amplifier (/55 factor)
                                        b_range = b * scene_bin_size
                                        decay_ratio = b_range / max(entry_range, 1.0)
                                        decay_dB = 40.0 * math.log10(max(1.001, decay_ratio))
                                        log_amp_decay = 10 ** (-decay_dB / 35.0)
                                        # Additional exponential terrain absorption
                                        terrain_atten = math.exp(-0.12 * depth_from_entry)
                                        val = int(core_echo * log_amp_decay * terrain_atten + scint[bi])
                                        val = min(252, max(0, val))
                                    if val >= 6:
                                        echoes[b] = max(echoes[b], val)

                            # Ramp-down bins after exit
                            if use_random_ramps and emit_ramps:
                                rdn_len = random.randint(2, 5)
                                ramp_down_codes = np.linspace(random.randint(164, 220), random.randint(28, 76), rdn_len).astype(int)
                                for ri, rc in enumerate(ramp_down_codes):
                                    b = paint_exit_bin + 1 + ri
                                    if scene_dead_zone <= b < scene_bins:
                                        jitter = random.randint(-12, 12)
                                        echoes[b] = max(echoes[b], min(252, max(0, rc + jitter)))
                            elif not use_random_ramps:
                                for ri, rv in enumerate(ramp_down_seq):
                                    b = exit_bin + 1 + ri
                                    if scene_dead_zone <= b < scene_bins:
                                        echoes[b] = max(echoes[b], rv)

                    elif len(hits) == 1:
                        entry_bin = max(scene_dead_zone, int(hits[0] / scene_bin_size))
                        for b in range(entry_bin, min(scene_bins, entry_bin + 4)):
                            echoes[b] = max(echoes[b], random.randint(120, 252))

                # Compute terrain shadow for this spoke
                shadow_entry = None
                if land_intervals:
                    shadow_entry = min(entry for entry, _ in land_intervals)

            # --- Object returns ---
            obj_profile = object_echo_data.get(spoke_idx_str)
            if profile_guided and not reference_calibrated and obj_profile:
                # Profile-guided: paint object bins using reference values
                for seg in obj_profile:
                    start_bin = seg[0]
                    vals = seg[1]
                    for vi, val in enumerate(vals):
                        b = start_bin + vi
                        if 0 <= b < scene_bins:
                            echoes[b] = max(echoes[b], val)
                # Log detections for objects visible on this spoke
                for idx, obj in enumerate(objects):
                    obj_range = math.sqrt(obj.x**2 + obj.y**2)
                    obj_bearing_rad = math.atan2(obj.x, obj.y)
                    d_angle = bearing_rad - obj_bearing_rad
                    d_angle = (d_angle + math.pi) % (2 * math.pi) - math.pi
                    if abs(math.degrees(d_angle)) <= BEAMWIDTH_DEG / 2:
                        detections.append(SpokeDetection(
                            obj_index=idx, obj_type=obj.obj_type,
                            range_m=obj_range,
                            bearing_deg=math.degrees(obj_bearing_rad) % 360,
                            peak_echo=int(min(252, max(echoes))),
                            x=obj.x, y=obj.y,
                            speed_kts=obj.speed_kts, course_deg=obj.course_deg,
                            motion=obj.motion_label,
                        ))
            elif not profile_guided:
                # Shared object renderer for strict Mode 4, M4G, M4P, and
                # native synthetic object generation.
                for idx, obj in enumerate(objects):
                    cal = object_calibrations[idx] if reference_calibrated and idx < len(object_calibrations) else {}
                    raw_obj = scene_object_specs[idx] if idx < len(scene_object_specs) and isinstance(scene_object_specs[idx], dict) else {}
                    obj_support_segments = object_support_by_object.get(idx, {}).get(spoke_idx, []) if reference_calibrated else []
                    used_preserved_support = bool(obj_support_segments)
                    if reference_calibrated and used_preserved_support:
                        strict_support_report["objects"]["preserved_support_objects"] += 1
                        strict_support_report["objects"]["preserved_support_segments"] += len(obj_support_segments)
                    if shadow_entry is not None and not used_preserved_support:
                        obj_range_m = math.sqrt(obj.x**2 + obj.y**2)
                        if int(obj_range_m / scene_bin_size) >= shadow_entry:
                            continue
                    render_result = _render_object_echo_shared(
                        obj,
                        bearing_rad,
                        scene_bins=scene_bins,
                        scene_bin_size=scene_bin_size,
                        scene_dead_zone=scene_dead_zone,
                        sim_mode=sim_mode,
                        raw_obj=raw_obj,
                        cal=cal,
                        support_segments=obj_support_segments if used_preserved_support else None,
                        learned_object_model=learned_object_model,
                        scene=scene,
                    )
                    obj_echo = render_result.echoes
                    peak = np.max(obj_echo)
                    if peak >= 4:
                        if reference_calibrated and not used_preserved_support:
                            strict_support_report["objects"]["legacy_geometry_fallback_objects"] += 1
                        obj_range = math.sqrt(obj.x**2 + obj.y**2)
                        obj_bearing = math.degrees(math.atan2(obj.x, obj.y)) % 360
                        detections.append(SpokeDetection(
                            obj_index=idx, obj_type=obj.obj_type,
                            range_m=obj_range, bearing_deg=obj_bearing,
                            peak_echo=int(min(252, peak)),
                            x=obj.x, y=obj.y,
                            speed_kts=obj.speed_kts, course_deg=obj.course_deg,
                            motion=obj.motion_label,
                        ))
                    echoes = np.maximum(echoes, obj_echo)

            object_support_mask = object_support_lookup.get(spoke_idx)

            # --- Noise injection ---
            noise_profile = noise_echo_data.get(spoke_idx_str)
            if profile_guided and not reference_calibrated and noise_profile:
                # Profile-guided: paint noise at observed positions using reference values
                for seg in noise_profile:
                    start_bin = seg[0]
                    vals = seg[1]
                    for vi, val in enumerate(vals):
                        b = start_bin + vi
                        if 0 <= b < scene_bins and echoes[b] <= 0:
                            echoes[b] = val
            elif shared_mode4_forward:
                # Shared structured-noise path for current Mode 4, M4G, and M4P.
                spoke_rng = np.random.RandomState(tick * 1000 + 42)
                m42_noise_prob = noise_cal.get('noise_prob', noise_prob) if reference_calibrated else noise_prob
                m42_noise_codes = noise_cal.get('noise_codes', [8, 20, 28]) if reference_calibrated else noise_codes
                m42_noise_weights = noise_cal.get('noise_weights', None) if reference_calibrated else noise_weights
                m42_codes_arr = np.array([c for c in m42_noise_codes
                                          if c in FURUNO_GAIN_SET], dtype=int)
                if len(m42_codes_arr) == 0:
                    m42_codes_arr = np.array([8, 20, 28], dtype=int)
                if reference_calibrated and noise_profile:
                    strict_support_report["noise"]["preserved_support_spokes"] += 1
                    strict_support_report["noise"]["preserved_support_segments"] += len(noise_profile)
                    used_structured_noise = _paint_noise_support_segments(
                        echoes,
                        tick,
                        scene_bins,
                        scene_dead_zone,
                        scene_bin_size,
                        active_noise_field_stats,
                        m42_codes_arr,
                        m42_noise_weights,
                        object_support_mask,
                        land_intervals,
                        noise_profile,
                    )
                else:
                    if reference_calibrated:
                        strict_support_report["noise"]["legacy_structured_spokes"] += 1
                    used_structured_noise = _paint_structured_noise(
                        echoes,
                        tick,
                        scene_bins,
                        scene_dead_zone,
                        scene_bin_size,
                        active_noise_field_stats,
                        m42_noise_prob,
                        m42_codes_arr,
                        m42_noise_weights,
                        object_support_mask,
                        land_intervals,
                    )
                if not used_structured_noise and m42_noise_prob > 0:
                    _paint_legacy_noise(
                        echoes,
                        scene_bins,
                        scene_dead_zone,
                        m42_noise_prob,
                        m42_codes_arr,
                        m42_noise_weights,
                        spoke_rng,
                        object_support_mask,
                    )
            elif not profile_guided and noise_prob > 0:
                # Fallback: random noise injection (only when no profiles)
                _paint_legacy_noise(
                    echoes,
                    scene_bins,
                    scene_dead_zone,
                    noise_prob,
                    noise_codes_arr,
                    noise_weights,
                    np.random.RandomState(tick * 1000 + 42),
                    object_support_mask,
                )

            # --- Dead zone echo data ---
            if profile_guided and not reference_calibrated:
                dz_profile = dead_zone_echo_data.get(spoke_idx_str)
                if dz_profile:
                    for seg in dz_profile:
                        start_bin = seg[0]
                        vals = seg[1]
                        for vi, val in enumerate(vals):
                            b = start_bin + vi
                            if 0 <= b < scene_dead_zone:
                                echoes[b] = max(echoes[b], val)
            elif reference_calibrated:
                dz_profile = dead_zone_echo_data.get(spoke_idx_str)
                dz_code_dist = dead_zone_cal.get('code_distribution', [])
                dz_bin_code_dists = dead_zone_cal.get('bin_code_distributions', {})
                if dz_profile:
                    strict_support_report["dead_zone"]["preserved_support_spokes"] += 1
                    strict_support_report["dead_zone"]["preserved_support_segments"] += len(dz_profile)
                    _paint_dead_zone_support_segments(
                        echoes,
                        tick,
                        scene_dead_zone,
                        dz_profile,
                        dz_code_dist,
                        dz_bin_code_dists,
                    )
                else:
                    # Legacy fallback for older scene JSON files.
                    strict_support_report["dead_zone"]["legacy_probability_fallback_spokes"] += 1
                    dz_rng = np.random.RandomState(tick * 1000 + 7)
                    dz_bin_probs = dead_zone_cal.get('bin_probabilities', [])
                    if dz_code_dist:
                        dz_codes = np.array([c for c, _ in dz_code_dist])
                        dz_weights = np.array([w for _, w in dz_code_dist])
                        dz_weights = dz_weights / dz_weights.sum()
                        for b in range(min(len(dz_bin_probs), scene_dead_zone)):
                            if dz_rng.random() < dz_bin_probs[b]:
                                code = dz_rng.choice(dz_codes, p=dz_weights)
                                code = _round_clamped_code(_apply_display_db_correction(
                                    float(code),
                                    _dead_zone_correction_db(b, scene_dead_zone),
                                ))
                                echoes[b] = max(echoes[b], code)

            # Quantize to Furuno gain codes
            if profile_guided and not reference_calibrated:
                # Profile-guided: reference values are already exact, no quantization
                result = np.round(echoes).astype(int)
                result = np.clip(result, 0, 252)
            elif reference_calibrated:
                # Mode 4: Furuno quantization (simulating, not replaying)
                result = np.zeros(scene_bins, dtype=int)
                for i in range(scene_bins):
                    if echoes[i] > 0:
                        result[i] = _quantize_to_furuno(echoes[i])
                    else:
                        result[i] = 0
            else:
                result = np.zeros(scene_bins, dtype=int)
                for i in range(scene_dead_zone, scene_bins):
                    val = echoes[i]
                    if val <= 0:
                        result[i] = 0
                    else:
                        result[i] = _quantize_to_furuno(val)

            row = {
                'timestamp': spoke_ts,
                'status': scene_status,
                'scale': scene_scale,
                'range': scene_range_idx,
                'gain': scene_gain,
                'angle': tick,
                'echoes': result.tolist(),
            }
            rows.append(row)

            spoke_bearing_deg = tick * 360.0 / TICKS_PER_REV
            for det in detections:
                detection_rows.append({
                    'timestamp': spoke_ts, 'angle': tick,
                    'bearing_deg': spoke_bearing_deg,
                    'obj_id': det.obj_index, 'obj_type': det.obj_type,
                    'range_m': det.range_m, 'obj_bearing_deg': det.bearing_deg,
                    'peak_echo': det.peak_echo,
                    'x': det.x, 'y': det.y,
                    'speed_kts': det.speed_kts, 'course_deg': det.course_deg,
                    'motion': det.motion,
                })

        # Write CSV
        write_csv(filepath, rows)
        meta_filename = scene_stem + '_metadata.json'
        meta_filepath = os.path.join(meta_subdir, meta_filename)

        rot_metadata = {
            "rotation": f"{rot + 1}/{num_rotations}",
            "scene_file": os.path.basename(scene_path),
            "simulated_time": rotation_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
            "elapsed_sec": round(rot * rotation_dt, 1),
            "rotation_start": rows[0]['timestamp'] if rows else "",
            "rotation_end": rows[-1]['timestamp'] if rows else "",
            "radar_config": {
                "status": scene_status,
                "scale": scene_scale,
                "range_index": scene_range_idx,
                "range_nm": round(scene_range_m / 1852.0, 6),
                "range_m": round(scene_range_m, 1),
                "bins": scene_bins,
                "bin_size_m": round(scene_bin_size, 4),
                "dead_zone_bins": scene_dead_zone,
                "gain": scene_gain,
                "ticks_per_rev": TICKS_PER_REV,
                "beamwidth_deg": BEAMWIDTH_DEG,
                "rotation_period_sec": rotation_dt,
            },
            "simulation_mode": (
                SIM_MODE_LABELS[SIM_MODE_CURRENT_MODE4] if reference_calibrated else
                SIM_MODE_LABELS[SIM_MODE_M42] if profile_guided else
                SIM_MODE_LABELS[SIM_MODE_M4P] if predictive_mode else
                SIM_MODE_LABELS[SIM_MODE_M4G] if geometry_preview else
                SIM_MODE_LABELS[SIM_MODE_M1] if synthetic_mode else
                "ray-tracing + radar equation + statistical noise"
            ),
            "simulation_seed": sim_seed,
            "detections_in_rotation": len(detection_rows),
            "detections_file": det_filename,
            "coastlines": {
                "count": len(coastlines),
                "types": [c.name for c in coastlines] if coastlines else [],
            },
            "objects": [],
        }
        for i, obj in enumerate(objects):
            rng = math.sqrt(obj.x**2 + obj.y**2)
            brg = math.degrees(math.atan2(obj.x, obj.y)) % 360
            rot_metadata["objects"].append({
                "id": i, "type": obj.obj_type,
                "rcs_m2": round(obj.rcs_m2, 1),
                "size_m": f"{obj.length_m:.1f}x{obj.width_m:.1f}",
                "x_m": round(obj.x, 2), "y_m": round(obj.y, 2),
                "range_m": round(rng, 1), "bearing_deg": round(brg, 1),
                "heading_deg": round(obj.heading_deg, 1),
                "speed_kts": round(obj.speed_kts, 1),
                "course_deg": round(obj.course_deg, 1),
                "motion": obj.motion_label,
            })
        write_metadata_json(meta_filepath, rows, rot_metadata)
        write_detections_csv(det_filepath, detection_rows, rot_metadata)
        generated_files.append(filepath)
        generated_files.append(det_filepath)
        generated_files.append(meta_filepath)

        if verbose:
            moving = sum(1 for o in objects if o.is_moving)
            static = len(objects) - moving
            print(f"  Wrote {filename} ({len(rows)} spokes, "
                  f"{len(detection_rows)} detections, "
                  f"{moving} moving / {static} stationary)")

        # Collect rows for validation (first rotation only)
        if rot == 0:
            all_sim_rows = rows

        for obj in objects:
            obj.update_position(rotation_dt, coastlines, scene_range_m)

    # --- Auto-validate against source ---
    if reference_calibrated and strict_support_report is not None:
        scene["_strict_support_report"] = strict_support_report
    if all_sim_rows:
        validate_against_source(scene, all_sim_rows, verbose=verbose)

    # --- Mode 4 calibration report ---
    if reference_calibrated and verbose:
        print(f"\n{'='*60}")
        print(f"MODE 4 CALIBRATION REPORT")
        print(f"{'='*60}")
        n_calibrated = sum(1 for c in object_calibrations if c['peak_echo'] > 0)
        print(f"  Objects calibrated: {n_calibrated}/{len(objects)}")
        for i, cal in enumerate(object_calibrations):
            if i < len(objects):
                obj = objects[i]
                status = "calibrated" if cal['peak_echo'] > 0 else "generic"
                spread = "reference" if cal.get('use_reference_spread') else "gaussian"
                print(f"    [{i}] {obj.obj_type}: {status}, "
                      f"ref_code={cal['calibrated_ref_code']:.1f}, "
                      f"compression={cal['calibrated_compression']:.1f}, "
                      f"spread={spread}")
        drift_events = [e for e in mode42_validation_log if 'drift' in e]
        fallback_events = [e for e in mode42_validation_log if 'fallback' in e]
        print(f"  Coastline drift events: {len(drift_events)}")
        if drift_events and len(drift_events) <= 10:
            for e in drift_events:
                print(f"    {e}")
        elif drift_events:
            print(f"    (showing first 10 of {len(drift_events)})")
            for e in drift_events[:10]:
                print(f"    {e}")
        print(f"  Coastline fallback segments: {len(fallback_events)}")
        if fallback_events and len(fallback_events) <= 10:
            for e in fallback_events:
                print(f"    {e}")
        elif fallback_events:
            print(f"    (showing first 10 of {len(fallback_events)})")
            for e in fallback_events[:10]:
                print(f"    {e}")
        report = scene.get("_strict_support_report", {})
        if report:
            coast = report.get("coastline", {})
            objs = report.get("objects", {})
            noise = report.get("noise", {})
            dz = report.get("dead_zone", {})
            print("  Strict support report")
            print(
                "    Coastline: "
                f"ref={coast.get('reference_segments', 0)}, "
                f"geom-matched={coast.get('geometry_matched_segments', 0)}, "
                f"geom-unmatched={coast.get('geometry_unmatched_segments', 0)}, "
                f"fallback={coast.get('geometry_fallback_segments', 0)}, "
                f"clipped={coast.get('clipped_segments', 0)}"
            )
            print(
                "    Objects: "
                f"support-objects={objs.get('preserved_support_objects', 0)}, "
                f"support-segments={objs.get('preserved_support_segments', 0)}, "
                f"legacy-fallback={objs.get('legacy_geometry_fallback_objects', 0)}"
            )
            print(
                "    Noise: "
                f"support-spokes={noise.get('preserved_support_spokes', 0)}, "
                f"support-segments={noise.get('preserved_support_segments', 0)}, "
                f"legacy-structured={noise.get('legacy_structured_spokes', 0)}"
            )
            print(
                "    Dead zone: "
                f"support-spokes={dz.get('preserved_support_spokes', 0)}, "
                f"support-segments={dz.get('preserved_support_segments', 0)}, "
                f"legacy-fallback={dz.get('legacy_probability_fallback_spokes', 0)}"
            )
        print(f"{'='*60}")

    return generated_files


# ============================================================
# Custom Simulation Builder (Mode 5)
# ============================================================

# Default dead zone physical distance (meters) — hardware constant
DEAD_ZONE_M = DEAD_ZONE_BINS * BIN_SIZE_M  # ~8.3 m


def _build_object_from_spec(spec: dict, range_m: float) -> RadarObject:
    """Create a RadarObject from a minimal user specification.

    Required fields: type, x, y (in meters from radar).
    Optional fields use sensible defaults based on object type.
    """
    obj_type = spec.get("type", "medium_boat")
    params = OBJECT_PARAMS.get(obj_type, OBJECT_PARAMS["medium_boat"])

    x = float(spec["x"])
    y = float(spec["y"])
    rcs = float(spec.get("rcs_m2", sum(params["rcs_range"]) / 2))
    length = float(spec.get("length_m", sum(params["length_range"]) / 2))
    width = float(spec.get("width_m", sum(params["width_range"]) / 2))
    heading = float(spec.get("heading_deg", random.uniform(0, 360)))
    speed = float(spec.get("speed_kts", 0.0))
    course = float(spec.get("course_deg", 0.0))
    turn_rate = float(spec.get("turn_rate_dps", 0.0))

    return RadarObject(
        x=x, y=y, obj_type=obj_type, rcs_m2=rcs,
        length_m=length, width_m=width, heading_deg=heading,
        speed_kts=speed, course_deg=course, turn_rate_dps=turn_rate,
    )


def simulate_custom(output_dir: str,
                    source_csv: str = None,
                    gain: int = None,
                    range_nm: float = None,
                    noise_prob: float = None,
                    noise_codes: list = None,
                    custom_objects: list = None,
                    keep_csv_objects: bool = True,
                    clear_coastlines: bool = False,
                    generate_coastlines: bool = False,
                    coastline_probability: float = 0.7,
                    geojson_path: str = None,
                    radar_lat: float = None,
                    radar_lon: float = None,
                    num_rotations: int = 1,
                    verbose: bool = True) -> List[str]:
    """Custom Simulation Builder — Mode 5.

    Build a radar simulation with full control over parameters:
    - Load coastline environment from a real Furuno CSV capture (optional)
    - Generate random fractal coastlines from scratch (optional)
    - Override radar settings (gain, range, noise/clutter)
    - Define custom objects (boats, buoys) with positions and characteristics
    - Generate Furuno-format CSV output with detections and metadata

    RENDERING GUARANTEE:
    When a source CSV is loaded and NO parameters are changed, the output
    uses profile-guided rendering and matches Mode 42 (Extract+Simulate+Validate)
    exactly. When parameters are modified, geometric simulation is used
    (ray-tracing for coastlines, radar equation for objects, statistical noise).
    Both paths use the same underlying rendering engine.

    Args:
        output_dir: Output directory for generated CSV files
        source_csv: Path to Furuno CSV to load environment from (optional)
        gain: Override radar gain (0-100). None = use source/default (75)
        range_nm: Override radar range in NM. None = use source/default (0.125)
        noise_prob: Override noise probability (0-1). None = auto from gain
        noise_codes: Override noise quantization codes. None = use source/default
        custom_objects: List of object dicts [{type, x, y, ...}] to add
        keep_csv_objects: If True (default), keep objects extracted from CSV
        clear_coastlines: If True, remove all coastlines from scene
        generate_coastlines: If True, generate random fractal coastlines
            (used when building from scratch without source CSV)
        coastline_probability: Probability of generating coastlines when
            generate_coastlines is True (0-1, default 0.7)
        num_rotations: Number of radar rotations to generate
        verbose: Print progress information

    Returns:
        List of generated file paths
    """
    # ---- Step 1: Build the base scene ----
    if source_csv:
        if verbose:
            print("=" * 60)
            print("CUSTOM SIMULATION: Loading environment from CSV...")
            print("=" * 60)

        # Import extract_scene and extract
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        try:
            from extract_scene import extract_scene_from_csv
        except ImportError:
            print("ERROR: extract_scene.py not found in same directory")
            return []

        scene = extract_scene_from_csv(source_csv, verbose=verbose)
    else:
        if verbose:
            print("=" * 60)
            print("CUSTOM SIMULATION: Building from scratch...")
            print("=" * 60)

        # Build an empty scene with defaults
        default_range_nm = range_nm if range_nm else RANGE_NM
        default_range_m = default_range_nm * 1852.0
        default_bins = NUM_BINS
        default_bin_size = default_range_m / default_bins
        default_gain = gain if gain else GAIN

        scene = {
            "source_file": "custom",
            "source_file_path": "",
            "extracted": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "radar_config": {
                "status": STATUS, "scale": SCALE,
                "range_index": furuno_range_index_for_nm(default_range_nm),
                "gain": default_gain,
                "range_nm": default_range_nm,
                "range_m": round(default_range_m, 1),
                "bins": default_bins,
                "bin_size_m": round(default_bin_size, 4),
                "ticks_per_rev": TICKS_PER_REV,
                "spokes_in_rotation": SPOKES_PER_REV,
            },
            "clutter": {
                "noise_probability": default_gain * 0.0001,
                "noise_codes": [8, 20, 28],
                "noise_weights": [0.45, 0.35, 0.20],
                "total_noise_bins": 0,
                "total_empty_bins": 0,
            },
            "decay": {
                "ramp_up_range": [80, 200],
                "ramp_down_range": [40, 160],
            },
            "coastline_stats": {},
            "coastline_profiles": {},
            "angle_ticks": None,
            "coastline_echo_data": {},
            "object_echo_data": {},
            "noise_echo_data": {},
            "dead_zone_echo_data": {},
            "coastlines": [],
            "objects": [],
        }

        # Generate random coastlines when building from scratch (library-based, fractal suppressed)
        if generate_coastlines and random.random() < coastline_probability:
            cur_range_m = default_range_m
            gen_library = build_coastline_library(verbose=False)
            if gen_library and gen_library.primitives:
                generated_coasts, _ = generate_random_coastlines_from_library(gen_library, cur_range_m)
            else:
                scene_template_files = _discover_scene_template_files()
                if scene_template_files:
                    generated_coasts, _ = generate_coastlines_from_scene_template(
                        scene_template_files, cur_range_m)
                else:
                    generated_coasts = []
            scene["coastlines"] = [
                {"name": c.name, "points": [[p[0], p[1]] for p in c.points]}
                for c in generated_coasts
            ]
            if verbose:
                print(f"  Generated {len(generated_coasts)} coastline features (library)")

    # ---- Step 1b: Load coastlines from GeoJSON (if provided) ----
    if geojson_path:
        if radar_lat is None or radar_lon is None:
            print("ERROR: --geojson requires --radar-lat and --radar-lon")
            return []
        cur_range_m = scene.get("radar_config", {}).get("range_m", RANGE_M)
        geojson_coasts = load_coastlines_from_geojson(
            geojson_path, radar_lat, radar_lon, cur_range_m)
        geojson_coast_dicts = [
            {"name": c.name, "points": [[p[0], p[1]] for p in c.points]}
            for c in geojson_coasts
        ]
        scene["coastlines"] = scene.get("coastlines", []) + geojson_coast_dicts
        if verbose:
            print(f"  Loaded {len(geojson_coasts)} coastline features from GeoJSON")
            print(f"  Radar position: ({radar_lat:.6f}, {radar_lon:.6f})")

        # Match GeoJSON coastlines to library primitives for echo profiles
        geojson_library = build_coastline_library(verbose=False)
        if geojson_library and geojson_library.primitives:
            geojson_echo_stats = assign_echo_profiles_to_geojson_coastlines(
                geojson_coasts, geojson_library)
            if geojson_echo_stats:
                existing_stats = scene.get("coastline_stats", {})
                existing_stats.update(geojson_echo_stats)
                scene["coastline_stats"] = existing_stats
                if verbose:
                    print(f"  Matched {len(geojson_echo_stats)} coastlines "
                          f"to library echo profiles")

    # ---- Step 2: Track what changed (affects rendering mode) ----
    orig_cfg = scene.get("radar_config", {})
    params_changed = False

    # ---- Step 3: Apply overrides ----

    # Range override
    if range_nm is not None and range_nm != orig_cfg.get("range_nm", RANGE_NM):
        params_changed = True
        new_range_m = range_nm * 1852.0
        new_bin_size = new_range_m / orig_cfg.get("bins", NUM_BINS)
        new_dead_zone_bins = max(1, int(DEAD_ZONE_M / new_bin_size))

        scene["radar_config"]["range_nm"] = range_nm
        scene["radar_config"]["range_m"] = round(new_range_m, 1)
        scene["radar_config"]["range_index"] = furuno_range_index_for_nm(range_nm)
        scene["radar_config"]["bin_size_m"] = round(new_bin_size, 4)
        scene["radar_config"]["dead_zone_bins"] = new_dead_zone_bins

        if verbose:
            print(f"\n  Range override: {range_nm} NM "
                  f"({new_range_m:.1f} m, {new_bin_size:.4f} m/bin, "
                  f"dead zone {new_dead_zone_bins} bins)")

    # Gain override
    if gain is not None and gain != orig_cfg.get("gain", GAIN):
        params_changed = True
        scene["radar_config"]["gain"] = gain
        # Auto-adjust noise probability based on gain if not explicitly set
        if noise_prob is None:
            scene["clutter"]["noise_probability"] = gain * 0.0001
        if verbose:
            print(f"  Gain override: {gain}")

    # Noise probability override
    if noise_prob is not None:
        orig_np = scene.get("clutter", {}).get("noise_probability", 0)
        if abs(noise_prob - orig_np) > 1e-8:
            params_changed = True
        scene["clutter"]["noise_probability"] = noise_prob
        if verbose:
            print(f"  Noise probability override: {noise_prob:.6f}")

    # Noise codes override
    if noise_codes is not None:
        params_changed = True
        scene["clutter"]["noise_codes"] = noise_codes
        # Reset weights to uniform
        scene["clutter"]["noise_weights"] = [1.0 / len(noise_codes)] * len(noise_codes)
        if verbose:
            print(f"  Noise codes override: {noise_codes}")

    # ---- Step 4: Handle coastlines ----
    if clear_coastlines:
        params_changed = True
        scene["coastlines"] = []
        scene["coastline_profiles"] = {}
        scene["coastline_echo_data"] = {}
        scene["coastline_stats"] = {}
        if verbose:
            print("  Coastlines: CLEARED")

    # ---- Step 5: Handle objects ----
    if not keep_csv_objects:
        params_changed = True
        scene["objects"] = []
        scene["object_echo_data"] = {}
        if verbose:
            print("  CSV objects: REMOVED")

    if custom_objects:
        params_changed = True
        cur_range_nm = scene["radar_config"].get("range_nm", RANGE_NM)
        cur_range_m = cur_range_nm * 1852.0

        for obj_spec in custom_objects:
            obj = _build_object_from_spec(obj_spec, cur_range_m)
            obj_dict = {
                "type": obj.obj_type, "x": obj.x, "y": obj.y,
                "rcs_m2": obj.rcs_m2, "length_m": obj.length_m,
                "width_m": obj.width_m, "heading_deg": obj.heading_deg,
                "speed_kts": obj.speed_kts, "course_deg": obj.course_deg,
            }
            scene["objects"].append(obj_dict)
            if verbose:
                obj_range = math.sqrt(obj.x**2 + obj.y**2)
                obj_bearing = math.degrees(math.atan2(obj.x, obj.y)) % 360
                print(f"  + {obj.obj_type}: ({obj.x:.1f}, {obj.y:.1f}) "
                      f"range={obj_range:.1f}m brg={obj_bearing:.1f}° "
                      f"rcs={obj.rcs_m2:.1f}m²")

    # ---- Step 6: Force geometric rendering if parameters changed ----
    if params_changed:
        # Clear all profile/echo data → forces geometric rendering path
        scene["coastline_profiles"] = {}
        scene["object_echo_data"] = {}
        scene["noise_echo_data"] = {}
        scene["dead_zone_echo_data"] = {}
        # Also clear source_file_path to skip validation against original
        scene["source_file_path"] = ""

        if verbose:
            print(f"\n  Parameters modified → using geometric simulation")
            print(f"    Coastline rendering: ray-tracing + coastline statistics")
            print(f"    Object rendering: radar equation")
            print(f"    Noise rendering: statistical injection")
    else:
        if verbose:
            print(f"\n  No parameter changes → using profile-guided simulation")
            print(f"    Output will match Mode 42 (Extract+Simulate+Validate) exactly")

    # ---- Step 7: Save scene to temp file and simulate ----
    os.makedirs(output_dir, exist_ok=True)
    scene_filename = f"custom_scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    scene_path = os.path.join(output_dir, scene_filename)
    with open(scene_path, "w") as f:
        json.dump(scene, f, indent=2)

    if verbose:
        num_coast = len(scene.get("coastlines", []))
        num_obj = len(scene.get("objects", []))
        r_nm = scene["radar_config"].get("range_nm", RANGE_NM)
        g = scene["radar_config"].get("gain", GAIN)
        np_val = scene.get("clutter", {}).get("noise_probability", 0)
        print(f"\n  Custom scene: {num_coast} coastlines, {num_obj} objects")
        print(f"  Range: {r_nm} NM, Gain: {g}, Noise prob: {np_val:.6f}")
        print(f"  Scene file: {scene_path}")
        print()

    generated_files = simulate_from_scene_file(scene_path, output_dir,
                                                num_rotations=num_rotations,
                                                verbose=verbose)

    if verbose and generated_files:
        abs_output = os.path.abspath(output_dir)
        csv_files = [f for f in generated_files if f.endswith('.csv')
                     and '_detections' not in f]
        det_files = [f for f in generated_files if '_detections.csv' in f]
        meta_files = [f for f in generated_files if f.endswith('_metadata.json')]
        print(f"\n{'=' * 60}")
        print(f"  CUSTOM SIMULATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Output directory : {abs_output}")
        print(f"  Radar CSV files  : {len(csv_files)}")
        print(f"  Detection files  : {len(det_files)}")
        print(f"  Metadata files   : {len(meta_files)}")
        print(f"  Scene definition : {os.path.basename(scene_path)}")
        if csv_files:
            print(f"  First radar CSV  : {os.path.basename(csv_files[0])}")
        print(f"{'=' * 60}")

    return generated_files


# ============================================================
# Random Output Directory Naming
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")
RANDOM_OUTPUT_BASE = os.path.join(OUTPUTS_ROOT, "random")
MODE4_OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "mode4")
MODE42_OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "mode42")
M1_SCENE_OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "m1")
M4G_OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "m4g")
M4P_OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "m4p")
CUSTOM_OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "custom")
TRAINING_OUTPUT_DIR = os.path.join(OUTPUTS_ROOT, "training_data")
MODE12_OUTPUT_BASE = os.path.join(OUTPUTS_ROOT, "mode_c")
RANDOM_OUTPUT_PREFIX = "Random Output"


def _next_random_output_dir() -> str:
    """Create the next sequentially-numbered random output directory.

    Scans existing 'Random Output *' folders in the Object Creation directory,
    finds the highest number, increments by 1, and appends a timestamp.
    Format: 'Random Output <N> <YYYYMMDD_HHMMSS>'
    Number range: 1-100000.
    """
    os.makedirs(RANDOM_OUTPUT_BASE, exist_ok=True)
    existing_numbers = []
    if os.path.isdir(RANDOM_OUTPUT_BASE):
        for name in os.listdir(RANDOM_OUTPUT_BASE):
            if name.startswith(RANDOM_OUTPUT_PREFIX) and os.path.isdir(
                    os.path.join(RANDOM_OUTPUT_BASE, name)):
                # Extract number: "Random Output 42 20260224_153012" → 42
                rest = name[len(RANDOM_OUTPUT_PREFIX):].strip()
                parts = rest.split(None, 1)
                if parts:
                    try:
                        existing_numbers.append(int(parts[0]))
                    except ValueError:
                        pass

    next_num = max(existing_numbers, default=0) + 1
    next_num = min(next_num, 100000)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{RANDOM_OUTPUT_PREFIX} {next_num} {timestamp}"
    return os.path.join(RANDOM_OUTPUT_BASE, folder_name)


MODE12_OUTPUT_PREFIX = "Mode 12 Output"


def _next_mode12_output_dir() -> str:
    """Create the next sequentially-numbered Mode 12 output directory.

    Scans existing 'Mode 12 Output *' folders, finds the highest number,
    increments by 1, and appends a timestamp.
    Format: 'Mode 12 Output <N> <YYYYMMDD_HHMMSS>'
    """
    os.makedirs(MODE12_OUTPUT_BASE, exist_ok=True)
    existing_numbers = []
    if os.path.isdir(MODE12_OUTPUT_BASE):
        for name in os.listdir(MODE12_OUTPUT_BASE):
            if name.startswith(MODE12_OUTPUT_PREFIX) and os.path.isdir(
                    os.path.join(MODE12_OUTPUT_BASE, name)):
                rest = name[len(MODE12_OUTPUT_PREFIX):].strip()
                parts = rest.split(None, 1)
                if parts:
                    try:
                        existing_numbers.append(int(parts[0]))
                    except ValueError:
                        pass

    next_num = max(existing_numbers, default=0) + 1
    next_num = min(next_num, 100000)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{MODE12_OUTPUT_PREFIX} {next_num} {timestamp}"
    return os.path.join(MODE12_OUTPUT_BASE, folder_name)


def _default_output_dir_for_sim_mode(sim_mode: str) -> str:
    """Return the default output folder for an explicit scene simulation mode."""
    mapping = {
        SIM_MODE_CURRENT_MODE4: MODE4_OUTPUT_DIR,
        SIM_MODE_M42: MODE42_OUTPUT_DIR,
        SIM_MODE_M4G: M4G_OUTPUT_DIR,
        SIM_MODE_M4P: M4P_OUTPUT_DIR,
        SIM_MODE_M1: M1_SCENE_OUTPUT_DIR,
    }
    return mapping.get(sim_mode, MODE4_OUTPUT_DIR)


def _modec_run_folder_name(location_name: str, created_at: Optional[datetime] = None,
                           scene_idx: Optional[int] = None) -> str:
    """Build a Mode C run folder name from creation time and resolved location."""
    created_at = created_at or datetime.now()
    timestamp = created_at.strftime("%Y%m%d_%H%M%S")
    slug = _normalize_location_key(location_name).replace(" ", "_")[:60] or "random_preset"
    folder_name = f"{timestamp}_{slug}"
    if scene_idx is not None:
        folder_name += f"_scene_{scene_idx:03d}"
    return folder_name


def _modec_scene_output_dir(base_output_dir: str, location_name: str,
                            created_at: Optional[datetime] = None,
                            scene_idx: Optional[int] = None,
                            auto_group: bool = False) -> str:
    """Resolve the final scene output folder for Mode C runs."""
    if not auto_group:
        if scene_idx is not None:
            return os.path.join(base_output_dir, f"scene_{scene_idx:03d}")
        return base_output_dir
    return os.path.join(
        base_output_dir,
        _modec_run_folder_name(location_name, created_at=created_at,
                               scene_idx=scene_idx),
    )


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Furuno DRS4DNXT Geometric Radar Simulator (0.125 NM)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory for CSV files (random mode auto-generates if omitted)")
    parser.add_argument("-s", "--scenes", type=int, default=3,
                        help="Number of random scenes to generate")
    parser.add_argument("-r", "--rotations", type=int, default=10,
                        help="Rotations (CSV files) per scene")
    parser.add_argument("--single", action="store_true",
                        help="Generate a single scene in the output directory (no subfolder)")
    parser.add_argument("--scene-file", type=str, default=None,
                        help="Mode 42: scene JSON, CSV file, or folder of CSVs")
    parser.add_argument("--sim-mode", type=str, default=None,
                        choices=SIM_MODE_CHOICES,
                        help="Explicit scene simulation mode: current_mode4, m1, m4g, m4p, or m42")
    parser.add_argument("--calibration-file", type=str, default=MODE4_CALIBRATION_PATH,
                        help="[--sim-mode m4p] Predictive calibration JSON to load")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress verbose output")

    # ---- Mode 4: Reference-calibrated simulation ----
    parser.add_argument("--reference-sim", "--mode42", "--mode4", type=str, default=None,
                        metavar="INPUT",
                        help="Mode 4: reference-calibrated from scene JSON, CSV, or folder")

    # ---- Custom simulation builder (Mode 5) ----
    parser.add_argument("--custom", action="store_true",
                        help="Custom simulation builder mode")
    parser.add_argument("--source-csv", type=str, default=None,
                        help="[Custom] Load coastlines/environment from Furuno CSV")
    parser.add_argument("--gain", type=int, default=None,
                        help="[Custom] Override radar gain (0-100)")
    parser.add_argument("--range-nm", type=float, default=None,
                        help="[Custom] Override radar range in nautical miles")
    parser.add_argument("--noise-prob", type=float, default=None,
                        help="[Custom] Override noise probability (0-1)")
    parser.add_argument("--noise-codes", type=str, default=None,
                        help="[Custom] Override noise codes (comma-separated, e.g. '8,20,28')")
    parser.add_argument("--add-object", action="append", default=[],
                        metavar="SPEC",
                        help="[Custom] Add object: 'type,x,y[,rcs,length,width,heading,speed,course]' "
                             "(repeatable)")
    parser.add_argument("--objects-file", type=str, default=None,
                        help="[Custom] JSON file with object definitions "
                             "[{type,x,y,...}, ...]")
    parser.add_argument("--clear-objects", action="store_true",
                        help="[Custom] Remove objects extracted from source CSV")
    parser.add_argument("--clear-coastlines", action="store_true",
                        help="[Custom] Remove all coastlines from scene")
    parser.add_argument("--generate-coastlines", action="store_true",
                        help="[Custom] Generate random fractal coastlines "
                             "(for building from scratch without --source-csv)")
    parser.add_argument("--coastline-prob", type=float, default=0.7,
                        help="[Custom] Probability of coastline generation (0-1, default 0.7)")
    parser.add_argument("--geojson", type=str, default=None,
                        help="[Custom] Load coastlines from GeoJSON file")
    parser.add_argument("--radar-lat", type=float, default=None,
                        help="[Custom] Radar latitude for GeoJSON coordinate transform")
    parser.add_argument("--radar-lon", type=float, default=None,
                        help="[Custom] Radar longitude for GeoJSON coordinate transform")

    # ---- Mode 12: Real-coastline random simulation ----
    parser.add_argument("--mode12", action="store_true",
                        help="Mode 12: Real-coastline random simulation using OSM data")
    parser.add_argument("--location", type=str, default=None,
                        help="[Mode 12] Location: 'lat,lon' or place name (default: random)")
    parser.add_argument("--radarloc", type=str, default=None,
                        help="[Mode 12] Load coastline data from .radarloc file")
    parser.add_argument("--local-radarloc-production", action="store_true",
                        help="[Mode 12] Use the selected .radarloc directly without source ranking or network rebuilds")
    parser.add_argument("--coastline-gain", type=float, default=None,
                        help="[Mode C] Gain 0-1; legacy --gain 0-100 remains accepted")
    parser.add_argument("--sea-clutter", type=float, default=None)
    parser.add_argument("--rain-clutter", type=float, default=None)
    parser.add_argument("--sea-state", type=int, default=None)
    parser.add_argument("--rain-rate-mmh", type=float, default=None)
    parser.add_argument("--receiver-noise", action="store_true")
    parser.add_argument("--receiver-noise-level", type=float, default=None)
    parser.add_argument("--origin-xy", default=None)
    parser.add_argument("--attenuation-model", choices=["current", "entry_r4_depth"], default=None)
    parser.add_argument("--object-profile", choices=["mode1", "fixed", "none"], default=None)
    parser.add_argument("--random-targets", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-implicit-capture", action="store_true",
        help="Do not discover scene templates, build capture-derived coastline libraries, or auto-extract local CSVs in random mode")
    parser.add_argument("--spokes", type=int, default=None)
    parser.add_argument("--raytrace-spokes", type=int, default=None)
    parser.add_argument("--beam-width-deg", type=float, default=None)
    parser.add_argument("--strict-tick-raytrace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--interactive", action="store_true",
                        help="[Mode 12] Launch interactive PPI viewer after generation")
    parser.add_argument("--prefetch-modec", action="store_true",
                        help="[Mode 12] Resolve and cache the best available real shoreline source without simulating")
    parser.add_argument("--resolve-modec-location", action="store_true",
                        help="[Mode 12] Resolve a location to coordinates only and exit")
    parser.add_argument("--search-modec-location", action="store_true",
                        help="[Mode 12] Search for likely location matches and exit")
    parser.add_argument("--search-output", type=str, default=None,
                        help="[Mode 12] Optional TSV output path for --search-modec-location")
    parser.add_argument("--prefer-source", type=str, default=None,
                        choices=["auto", "local_geojson", "usgs", "noaa", "osm", "radarloc"],
                        help="[Mode 12] Prefer a specific coastline source for testing/validation")

    # ---- Training data generation ----
    parser.add_argument("--training-data", action="store_true",
                        help="Generate YOLO training dataset from Mode 12 scenarios")
    parser.add_argument("--image-size", type=int, default=512,
                        help="[Training] PPI image size in pixels (default: 512)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="[Training] Fraction of scenes for validation (default: 0.2)")
    parser.add_argument("--waypoints", type=str, default=None,
                        help="[Training] Waypoints for scripted vessel: 'x1,y1;x2,y2;x3,y3'")
    parser.add_argument("--wp-type", type=str, default="medium_boat",
                        help="[Training] Waypoint vessel type (default: medium_boat)")
    parser.add_argument("--wp-speed", type=float, default=8.0,
                        help="[Training] Waypoint vessel speed in knots (default: 8)")

    args = parser.parse_args()

    if args.custom:
        # ---- Parse custom objects ----
        custom_objects = []

        # Objects from --add-object flags
        for spec_str in args.add_object:
            parts = [p.strip() for p in spec_str.split(",")]
            if len(parts) < 3:
                print(f"ERROR: --add-object needs at least type,x,y — got: {spec_str}")
                sys.exit(1)
            obj_spec = {"type": parts[0], "x": float(parts[1]), "y": float(parts[2])}
            field_names = ["rcs_m2", "length_m", "width_m",
                           "heading_deg", "speed_kts", "course_deg"]
            for i, name in enumerate(field_names):
                if i + 3 < len(parts) and parts[i + 3]:
                    obj_spec[name] = float(parts[i + 3])
            custom_objects.append(obj_spec)

        # Objects from --objects-file (CSV or JSON)
        if args.objects_file:
            ext = os.path.splitext(args.objects_file)[1].lower()
            if ext == ".csv":
                with open(args.objects_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        obj_spec = {"type": row["type"]}
                        for key in ["x", "y", "rcs_m2", "length_m", "width_m",
                                     "heading_deg", "speed_kts", "course_deg"]:
                            if key in row and row[key]:
                                obj_spec[key] = float(row[key])
                        custom_objects.append(obj_spec)
            else:
                with open(args.objects_file, "r") as f:
                    file_objects = json.load(f)
                if isinstance(file_objects, list):
                    custom_objects.extend(file_objects)
                else:
                    print(f"ERROR: --objects-file JSON must contain an array")
                    sys.exit(1)

        # Parse noise codes
        noise_codes_list = None
        if args.noise_codes:
            noise_codes_list = [int(c.strip()) for c in args.noise_codes.split(",")]

        custom_output = args.output if args.output else CUSTOM_OUTPUT_DIR
        print("=" * 60)
        print("  Furuno DRS4DNXT Custom Simulation Builder")
        print("=" * 60)
        simulate_custom(
            output_dir=custom_output,
            source_csv=args.source_csv,
            gain=args.gain,
            range_nm=args.range_nm,
            noise_prob=args.noise_prob,
            noise_codes=noise_codes_list,
            custom_objects=custom_objects if custom_objects else None,
            keep_csv_objects=not args.clear_objects,
            clear_coastlines=args.clear_coastlines,
            generate_coastlines=args.generate_coastlines,
            coastline_probability=args.coastline_prob,
            geojson_path=args.geojson,
            radar_lat=args.radar_lat,
            radar_lon=args.radar_lon,
            num_rotations=args.rotations,
            verbose=not args.quiet,
        )
    elif args.sim_mode and (args.scene_file or args.reference_sim):
        sim_input = args.scene_file if args.scene_file else args.reference_sim
        sim_output = args.output if args.output else _default_output_dir_for_sim_mode(args.sim_mode)
        simulate_scene_mode(sim_input, sim_output, args.sim_mode,
                            num_rotations=args.rotations,
                            verbose=not args.quiet,
                            calibration_path=args.calibration_file)
    elif args.reference_sim:
        mode42_output = args.output if args.output else MODE4_OUTPUT_DIR
        simulate_mode42(args.reference_sim, mode42_output,
                        num_rotations=args.rotations,
                        verbose=not args.quiet)
    elif args.training_data:
        td_output = args.output if args.output else TRAINING_OUTPUT_DIR
        # Parse waypoints if provided: "x1,y1;x2,y2;x3,y3"
        wp_list = None
        if args.waypoints:
            wp_list = []
            for pt in args.waypoints.split(';'):
                coords = pt.strip().split(',')
                if len(coords) >= 2:
                    wp_list.append((float(coords[0]), float(coords[1])))
        from coastline_defined.scene_pipeline import run_modec
        run_modec(sys.modules[__name__], td_output, args.location, args.scenes,
            args.rotations, args.range_nm, args.radarloc, args.prefer_source,
            not args.quiet, options={"training_data": True, "image_size": args.image_size,
                "val_split": args.val_split, "waypoints": args.waypoints,
                "wp_type": args.wp_type, "wp_speed": args.wp_speed,
                "gain": args.coastline_gain if args.coastline_gain is not None else (args.gain / 100 if args.gain is not None else None),
                "sea_clutter": args.sea_clutter, "rain_clutter": args.rain_clutter,
                "sea_state": args.sea_state, "rain_rate_mmh": args.rain_rate_mmh,
                "receiver_noise": args.receiver_noise, "receiver_noise_level": args.receiver_noise_level,
                "origin_xy": args.origin_xy, "radar_lat": args.radar_lat, "radar_lon": args.radar_lon,
                "attenuation_model": args.attenuation_model,
                "object_profile": args.object_profile, "random_targets": args.random_targets,
                "seed": args.seed, "spokes": args.spokes, "raytrace_spokes": args.raytrace_spokes,
                "beam_width_deg": args.beam_width_deg, "strict_tick_raytrace": args.strict_tick_raytrace,
                "manual_object": args.add_object})
    elif args.prefetch_modec:
        result = prefetch_modec_source(location=args.location, range_nm=args.range_nm,
                                       verbose=not args.quiet,
                                       prefer_source=args.prefer_source)
        if result is None:
            sys.exit(1)
    elif args.resolve_modec_location:
        result = resolve_modec_location_only(location=args.location, range_nm=args.range_nm,
                                             verbose=not args.quiet)
        if result is None:
            sys.exit(1)
    elif args.search_modec_location:
        results = search_modec_location_candidates(location=args.location, limit=8,
                                                  verbose=not args.quiet, output_path=args.search_output)
        if not results:
            sys.exit(1)
    elif args.mode12:
        mode12_output = args.output if args.output else _next_mode12_output_dir()
        simulate_mode12(output_dir=mode12_output, location=args.location,
                        num_scenes=args.scenes, num_rotations=args.rotations,
                        range_nm=args.range_nm, verbose=not args.quiet,
                        radarloc_file=args.radarloc, interactive=args.interactive,
                        prefer_source=args.prefer_source,
                        local_radarloc_production=args.local_radarloc_production,
                        modec_options={"gain": args.coastline_gain if args.coastline_gain is not None else (args.gain / 100 if args.gain is not None else None),
                            "sea_clutter": args.sea_clutter, "rain_clutter": args.rain_clutter,
                            "sea_state": args.sea_state, "rain_rate_mmh": args.rain_rate_mmh,
                            "receiver_noise": args.receiver_noise, "receiver_noise_level": args.receiver_noise_level,
                            "origin_xy": args.origin_xy, "radar_lat": args.radar_lat, "radar_lon": args.radar_lon,
                            "attenuation_model": args.attenuation_model,
                            "object_profile": args.object_profile, "random_targets": args.random_targets,
                            "seed": args.seed, "spokes": args.spokes, "raytrace_spokes": args.raytrace_spokes,
                            "beam_width_deg": args.beam_width_deg, "strict_tick_raytrace": args.strict_tick_raytrace,
                            "manual_object": args.add_object})
    elif args.scene_file:
        scene_output = args.output if args.output else MODE42_OUTPUT_DIR
        print(f"Simulating from scene file: {args.scene_file}")
        print(f"Output: {scene_output}")
        simulate_from_scene_file(args.scene_file, scene_output,
                                  num_rotations=args.rotations,
                                  verbose=not args.quiet)
    elif args.single:
        output_dir = args.output if args.output else _next_random_output_dir()
        print(f"Generating single scene: {args.rotations} rotations")
        print(f"Output: {output_dir}")
        simulate_scene(output_dir, num_rotations=args.rotations,
                      verbose=not args.quiet, seed=args.seed,
                      no_implicit_capture=args.no_implicit_capture)
    else:
        output_dir = args.output if args.output else _next_random_output_dir()
        generate_batch(output_dir, args.scenes, args.rotations,
                      verbose=not args.quiet, seed=args.seed,
                      no_implicit_capture=args.no_implicit_capture)
