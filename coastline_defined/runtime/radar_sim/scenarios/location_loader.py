"""Load .radarloc location files into the radar simulator."""
import json
import os
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from collections import deque
from radar_core import rust_bridge
from ..environment.coastline import Coastline, CoastlinePoint
from ..environment.attenuation import ATTENUATION_ENTRY_R4_DEPTH, env_attenuation_model
from ..environment.terrain import HeightMap, TerrainConfig
from ..objects.vessel import Vessel, VesselType


# Default renderer for .radarloc coastlines when the file does not specify one.
# Set RADARLOC_DEFAULT_RENDER_MODE=furuno_hybrid to revert loaded locations to
# the previous terrain-owned shoreline behavior without changing the launcher.
DEFAULT_RADARLOC_RENDER_MODE = (
    os.environ.get("RADARLOC_DEFAULT_RENDER_MODE", "polar_intersection").strip()
    or "polar_intersection"
)
# Keep location exports and embedded callers on the same entry-range R^4
# default; callers can still explicitly request a legacy attenuation model.
DEFAULT_RADARLOC_ATTENUATION_MODEL = env_attenuation_model(ATTENUATION_ENTRY_R4_DEPTH)
DEFAULT_RADARLOC_SIMPLIFY_MIN_POINTS = max(
    0, int(os.environ.get("RADARLOC_SIMPLIFY_MIN_POINTS", "64"))
)
DEFAULT_RADARLOC_SIMPLIFY_MIN_TOLERANCE_M = max(
    0.0, float(os.environ.get("RADARLOC_SIMPLIFY_MIN_TOLERANCE_M", "2.0"))
)
DEFAULT_RADARLOC_SIMPLIFY_TOLERANCE_SCALE = max(
    0.0, float(os.environ.get("RADARLOC_SIMPLIFY_TOLERANCE_SCALE", "0.1"))
)


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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _open_shore_land_bias_cells(cell_size: float) -> int:
    """Bias ambiguous broad-scene cells toward land by a small shoreline margin.

    In harbor and marsh exports, pure distance-to-origin-water can overfill
    marsh platforms between many narrow channels. A modest land preference for
    cells reached by both the water and land floods keeps more of those
    inter-channel areas on the land side without removing the actual water
    polygons.

    Default is 0. Measured against the OSM ground truth for Charleston Harbor a
    non-zero bias is a clear net loss -- it widens the tie band between the two
    floods into visible land bars across open water:

        bias 0 -> 95.3% agreement, 10.5% of true water called land
        bias 1 -> 94.4% agreement, 13.2% of true water called land
        bias 2 -> 93.0% agreement, 17.9% of true water called land

    Set RADARLOC_OPEN_SHORE_LAND_BIAS_CELLS to re-enable it for a scene that
    genuinely needs the marsh-platform behaviour.
    """
    return max(0, _env_int("RADARLOC_OPEN_SHORE_LAND_BIAS_CELLS", 0))


def _open_shore_grid_size(base_grid_size: int, coastline_count: int) -> int:
    """Use a denser raster for broad open-shore scenes with many shoreline cuts.

    Charleston-style harbor/marsh exports contain many thin land fingers and
    channels. At the legacy 768 grid those narrow separators can collapse,
    making water spread through areas that should stay land. A denser raster is
    a straightforward way to preserve more of that shoreline separation.
    """
    if coastline_count >= 350:
        default_size = 3072
    elif coastline_count >= 200:
        default_size = 2048
    else:
        default_size = 1536
    return max(base_grid_size, _env_int("RADARLOC_OPEN_SHORE_GRID_SIZE", default_size))


DETAIL_FEATURE_CLASSES = {
    "pier",
    "breakwater",
    "groyne",
    "quay",
    "jetty",
    "small_land_feature",
}

WATER_FILL_FEATURE_CLASSES = {
    "water",
    "river",
    "pond",
    "lake",
    "basin",
    "reservoir",
    "shoreline",
    "canal",
    "stream",
    "harbour",
    "harbor",
    "lagoon",
    "bay",
    "fairway",
    "dock",
    "strait",
}

MAJOR_SHORELINE_FEATURE_CLASSES = {
    "coastline",
    "shoreline",
}

# Wetland flavors exported by the location generator. Radar-wise these are
# low-lying land: weak-but-real returns, never open water to carve away.
# Sand/beach surfaces: strong rough-surface scatterers -- more interior
# detail than vegetated marsh, far more than paved/urban ground.
BEACH_FEATURE_CLASSES = {
    "beach",
    "sand",
    "dune",
}

WETLAND_FEATURE_CLASSES = {
    "wetland",
    "marsh",
    "saltmarsh",
    "tidalflat",
    "reedbed",
    "swamp",
    "wet_meadow",
    "mangrove",
    "bog",
    "fen",
}


def _feature_prefers_water_fill(feature_class: str) -> bool:
    """Return True when a closed OSM-derived feature should carve blue water.

    Broad harbor exports may contain many closed water features alongside a few
    closed coastline rings. The water features should become filled water in the
    loaded mask, while coastline rings are safer to keep on the stricter
    connected-water path so islands are not accidentally inverted.
    """
    return str(feature_class or "").strip().lower() in WATER_FILL_FEATURE_CLASSES


def _feature_supports_major_bridge(feature_class: str) -> bool:
    """Return True for shoreline classes that can use a broader closure pass."""
    return str(feature_class or "").strip().lower() in MAJOR_SHORELINE_FEATURE_CLASSES


def _location_name_suggests_open_shore(location_name: str) -> bool:
    """Heuristic hint for legacy harbor/coastal exports without topology tags."""
    name = str(location_name or "").strip().lower()
    if not name:
        return False
    tokens = (
        "maritime",
        "harbor",
        "harbour",
        "bay",
        "ocean",
        "coast",
        "coastal",
        "inlet",
        "estuary",
        "sound",
        "gulf",
        "atlantic",
        "pacific",
    )
    return any(token in name for token in tokens)


def _infer_legacy_closed_feature_class(name: str,
                                       default_class: str = "small_land_feature") -> str:
    """Best-effort class for closed legacy polygons without OSM feature tags.

    `default_class` is what an unrecognised name falls back to. Open-shore
    exports are mostly land islands, but enclosed-water exports (lakes,
    reservoirs) are mostly water rings, so the caller picks the right default.
    """
    text = str(name or "").strip().lower()
    if not text:
        return default_class

    water_tokens = (
        "harbor",
        "harbour",
        "bay",
        "lagoon",
        "basin",
        "reservoir",
        "lake",
        "pond",
        "river",
        "stream",
        "canal",
        "water",
    )
    if any(token in text for token in water_tokens):
        return "water"

    land_tokens = (
        "island",
        "fort",
        "point",
        "bank",
        "shoal",
        "sanctuary",
        "park",
    )
    if any(token in text for token in land_tokens):
        return "small_land_feature"

    return default_class


def _infer_legacy_radarloc_context(meta: dict, coastline_docs: list,
                                   range_nm: float) -> dict:
    """Infer basic shoreline context for old geometry-only .radarloc files.

    Some legacy or third-party `.radarloc` exports contain only shoreline
    points plus `closed` flags. The echo renderer can still use that linework,
    but the scene-view terrain raster needs a little more context to avoid
    treating a harbor as a generic lake/polygon fill problem.
    """
    explicit_scene_topology = str(meta.get("scene_topology", "") or "").strip().lower()
    explicit_origin_surface = str(meta.get("origin_surface", "") or "").strip().lower()
    explicit_origin_source = str(meta.get("origin_surface_source", "") or "").strip()
    has_feature_tags = any("feature_class" in coast for coast in coastline_docs)

    if explicit_scene_topology or explicit_origin_surface or has_feature_tags:
        return {
            "scene_topology": explicit_scene_topology,
            "origin_surface": explicit_origin_surface,
            "origin_surface_source": explicit_origin_source,
            "feature_classes": {},
            "assume_topology_managed_open": False,
        }

    range_m = max(0.0, float(range_nm) * 1852.0)
    open_count = 0
    closed_polygons = []
    boundary_touching_open = False
    total_closed_area = 0.0

    for coast_data in coastline_docs:
        raw_points = coast_data.get("points", [])
        if len(raw_points) < 2:
            continue
        pts = [
            CoastlinePoint(float(point["x"]), float(point["y"]))
            for point in raw_points
            if "x" in point and "y" in point
        ]
        if len(pts) < 2:
            continue

        if bool(coast_data.get("closed", False)) and len(pts) >= 3:
            closed_polygons.append(pts)
            total_closed_area += _polygon_area(pts)
        else:
            open_count += 1
            if range_m > 0.0:
                for endpoint in (pts[0], pts[-1]):
                    if max(abs(endpoint.x), abs(endpoint.y)) >= range_m * 0.90:
                        boundary_touching_open = True

    center_in_closed_polygon = any(
        _point_in_polygon(0.0, 0.0, pts)
        for pts in closed_polygons
    )
    grid_area = (2.0 * range_m) ** 2 if range_m > 0.0 else 0.0
    location_hint_open_shore = _location_name_suggests_open_shore(meta.get("location_name", ""))

    inferred_scene_topology = ""
    if open_count > 0 and (
        boundary_touching_open
        or location_hint_open_shore
        or not center_in_closed_polygon
        or (grid_area > 0.0 and total_closed_area < grid_area * 0.20)
    ):
        inferred_scene_topology = "open_shore"
    elif closed_polygons and open_count == 0 and center_in_closed_polygon:
        inferred_scene_topology = "enclosed_water"

    inferred_origin_surface = ""
    if inferred_scene_topology in {"open_shore", "enclosed_water"}:
        inferred_origin_surface = "water"
    elif center_in_closed_polygon:
        inferred_origin_surface = "water"

    feature_classes = {}
    if inferred_scene_topology in {"open_shore", "enclosed_water"}:
        # Enclosed-water scenes (lakes, reservoirs) need name-based classes just
        # as much as open shores. Without them every ring stays
        # "legacy_unclassified", which downstream display code fills as land.
        for idx, coast_data in enumerate(coastline_docs):
            if "feature_class" in coast_data:
                continue
            if bool(coast_data.get("closed", False)):
                feature_classes[idx] = _infer_legacy_closed_feature_class(
                    coast_data.get("name", ""),
                    default_class=(
                        "water" if inferred_scene_topology == "enclosed_water"
                        else "small_land_feature"
                    ),
                )
            else:
                feature_classes[idx] = "coastline"

    return {
        "scene_topology": inferred_scene_topology,
        "origin_surface": inferred_origin_surface,
        "origin_surface_source": (
            explicit_origin_source
            or ("legacy_geometry_inference" if (inferred_scene_topology or inferred_origin_surface) else "")
        ),
        "feature_classes": feature_classes,
        "assume_topology_managed_open": inferred_scene_topology == "open_shore" and open_count > 0,
    }


def _default_reflectivity_for_feature(feature_class: str) -> float:
    """Coarse radar-surface reflectivity for generated OSM feature classes."""
    if feature_class in {"pier", "quay", "jetty"}:
        return 1.00
    if feature_class in {"breakwater", "groyne"}:
        return 0.94
    if feature_class == "small_land_feature":
        return 0.88
    if feature_class == "dock":
        # Marina/dock basins are water surfaces: no intensity echo at all --
        # only land features paint returns.
        return 0.0
    if feature_class in BEACH_FEATURE_CLASSES:
        # Dry sand: rough, bright surface with deep visible texture.
        return 0.66
    if feature_class in WETLAND_FEATURE_CLASSES:
        # Marsh/tidalflat: soft, wet, low-relief surface -- weak but real echo.
        return 0.34
    if feature_class == "water_or_shoreline":
        return 0.72
    return 0.45


def _default_echo_profile_for_feature(feature_class: str) -> dict:
    """Material-aware coastline defaults for generated coastal .radarloc files."""
    if feature_class in {"pier", "breakwater", "groyne", "quay", "jetty"}:
        return {
            "num_rays": 9,
            "edge_factor": 0.86,
            "leading_edge_boost": 1.20,
            "far_range_floor": 0.28,
            "range_decay_strength": 0.92,
            "continuous_return_depth_m": 65.0,
            "continuous_texture_strength": 0.24,
            "continuous_texture_floor": 0.50,
            "ray_core_depth_m": 1.0,
            "ray_penetration_max_m": 8.0,
        }
    if feature_class == "small_land_feature":
        return {
            "num_rays": 9,
            "edge_factor": 0.78,
            "leading_edge_boost": 1.12,
            "far_range_floor": 0.24,
            "range_decay_strength": 1.00,
            "continuous_return_depth_m": 90.0,
            "continuous_texture_strength": 0.36,
            "continuous_texture_floor": 0.42,
            "ray_core_depth_m": 1.2,
            "ray_penetration_max_m": 10.0,
        }
    if feature_class == "water_or_shoreline":
        return {
            "num_rays": 9,
            "edge_factor": 0.70,
            "leading_edge_boost": 1.05,
            "far_range_floor": 0.18,
            "range_decay_strength": 1.06,
            "continuous_return_depth_m": 520.0,
            "continuous_texture_strength": 0.46,
            "continuous_texture_floor": 0.34,
            "ray_core_depth_m": 1.2,
            "ray_penetration_max_m": 16.0,
        }
    if feature_class == "dock":
        return {
            "num_rays": 5,
            "edge_factor": 0.30,
            "leading_edge_boost": 1.0,
            "far_range_floor": 0.02,
            "range_decay_strength": 1.2,
            "continuous_return_depth_m": 20.0,
            "continuous_texture_strength": 0.15,
            "continuous_texture_floor": 0.10,
            "ray_core_depth_m": 0.5,
            "ray_penetration_max_m": 4.0,
        }
    if feature_class in WETLAND_FEATURE_CLASSES:
        # Marsh platforms: diffuse, textured, no crisp leading edge, and the
        # beam penetrates well into the vegetation before extinction.
        return {
            "num_rays": 7,
            "edge_factor": 0.52,
            "leading_edge_boost": 1.00,
            "far_range_floor": 0.12,
            "range_decay_strength": 1.10,
            "continuous_return_depth_m": 500.0,
            "continuous_texture_strength": 0.55,
            "continuous_texture_floor": 0.24,
            "ray_core_depth_m": 2.0,
            "ray_penetration_max_m": 40.0,
        }
    if feature_class in BEACH_FEATURE_CLASSES:
        # Sand: bright leading edge and deep, well-textured interior fill --
        # beaches paint far more detail than roads or bare developed ground.
        return {
            "num_rays": 9,
            "edge_factor": 0.80,
            "leading_edge_boost": 1.15,
            "far_range_floor": 0.22,
            "range_decay_strength": 0.90,
            "continuous_return_depth_m": 1600.0,
            "continuous_texture_strength": 0.42,
            "continuous_texture_floor": 0.40,
            "ray_core_depth_m": 2.5,
            "ray_penetration_max_m": 60.0,
        }
    return {}


def _set_profile_default(profile: dict, explicit_profile: dict,
                         key: str, value) -> None:
    """Apply launcher/material defaults without overriding explicit .radarloc fields."""
    if key not in explicit_profile:
        profile[key] = value


@dataclass
class RadarLocationData:
    """Parsed contents of a .radarloc file."""
    location_name: str = ""
    center_lat: float = 0.0
    center_lon: float = 0.0
    range_nm: float = 6.0
    coastlines: list = field(default_factory=list)
    terrain: Optional[HeightMap] = None
    vessels: list = field(default_factory=list)
    weather_config: dict = field(default_factory=dict)
    origin_surface: str = "unknown"
    origin_surface_source: str = ""
    scene_topology: str = ""
    _closed_indices: set = field(default_factory=set)  # Indices of closed polygons


def load_radarloc_file(filepath: str) -> RadarLocationData:
    """Load and parse a .radarloc JSON file.

    Args:
        filepath: Path to the .radarloc file.

    Returns:
        RadarLocationData with parsed coastlines, terrain, and vessels.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is invalid.
        json.JSONDecodeError: If JSON is malformed.
    """
    if rust_bridge.rust_available():
        try:
            # Native preflight catches malformed/oversized coastline files before
            # the Python loader allocates terrain grids or coastline objects.
            rust_bridge.radarloc_summary(filepath)
        except Exception as exc:
            raise ValueError(f"Rust .radarloc preflight failed: {exc}") from exc

    with open(filepath, "r") as f:
        doc = json.load(f)

    version = doc.get("version", "")
    if not version.startswith("1."):
        raise ValueError(f"Unsupported .radarloc version: {version}")

    result = RadarLocationData()

    # Metadata
    meta = doc.get("metadata", {})
    coastline_docs = doc.get("coastlines", [])
    result.location_name = meta.get("location_name", "Unknown Location")
    result.center_lat = meta.get("center_lat", 0.0)
    result.center_lon = meta.get("center_lon", 0.0)
    result.range_nm = meta.get("range_nm", 6.0)
    legacy_context = _infer_legacy_radarloc_context(meta, coastline_docs, result.range_nm)
    result.origin_surface = (
        str(meta.get("origin_surface", "") or "").strip().lower()
        or str(legacy_context.get("origin_surface", "") or "").strip().lower()
        or "unknown"
    )
    result.origin_surface_source = (
        str(meta.get("origin_surface_source", "") or "").strip()
        or str(legacy_context.get("origin_surface_source", "") or "").strip()
    )
    result.scene_topology = (
        str(meta.get("scene_topology", "") or "").strip().lower()
        or str(legacy_context.get("scene_topology", "") or "").strip().lower()
    )
    weather_doc = doc.get("weather", {}) or meta.get("weather", {}) or {}
    noise_doc = doc.get("noise", {}) or meta.get("noise", {}) or {}
    result.weather_config = {
        "sea_state": int(weather_doc.get("sea_state", 0) or 0),
        "wind_speed_knots": float(weather_doc.get("wind_speed_knots", 0.0) or 0.0),
        "wind_direction": float(weather_doc.get("wind_direction", 0.0) or 0.0),
        "rain_rate_mmh": float(weather_doc.get("rain_rate_mmh", 0.0) or 0.0),
        "visibility_nm": float(weather_doc.get("visibility_nm", 10.0) or 10.0),
        "receiver_noise_enabled": bool(noise_doc.get("receiver_noise_enabled", False)),
        "receiver_noise_level": float(noise_doc.get("receiver_noise_level", 0.0) or 0.0),
    }
    simplify_tolerance_m = max(
        DEFAULT_RADARLOC_SIMPLIFY_MIN_TOLERANCE_M,
        (2.0 * result.range_nm * 1852.0 / 768.0) * DEFAULT_RADARLOC_SIMPLIFY_TOLERANCE_SCALE,
    )
    if result.scene_topology == "open_shore":
        # Harbor/tidal .radarloc exports now preserve the exact queried
        # shoreline linework so tributaries, jetties, and narrow water cuts
        # keep their original detail when reloaded.
        simplify_tolerance_m = 0.0
    inferred_feature_classes = dict(legacy_context.get("feature_classes") or {})
    assume_topology_managed_open = bool(
        legacy_context.get("assume_topology_managed_open", False)
    )

    # Coastlines - track which ones are closed for terrain generation
    closed_indices = set()
    for idx, coast_data in enumerate(coastline_docs):
        points = coast_data.get("points", [])
        if len(points) < 3:
            continue
        is_closed = coast_data.get("closed", False)
        inferred_feature_class = inferred_feature_classes.get(idx, "")
        has_feature_class = (
            ("feature_class" in coast_data or bool(inferred_feature_class))
            and _env_bool("RADARLOC_USE_FEATURE_MATERIALS", True)
        )
        feature_class = str(
            coast_data.get("feature_class", inferred_feature_class or "legacy_unclassified")
            or "legacy_unclassified"
        )
        preserve_detail = (
            bool(coast_data.get("preserve_detail", False))
            or feature_class in DETAIL_FEATURE_CLASSES
        )
        simplified_points = _simplify_loaded_points(
            points,
            tolerance_m=0.0 if (preserve_detail or result.scene_topology == "open_shore")
            else simplify_tolerance_m,
            closed=is_closed,
            min_points=0 if preserve_detail else DEFAULT_RADARLOC_SIMPLIFY_MIN_POINTS,
        )
        if len(simplified_points) < 3:
            continue
        material_profile = (
            dict(_default_echo_profile_for_feature(feature_class))
            if has_feature_class
            else {}
        )
        explicit_echo_profile = dict(coast_data.get("echo_profile") or {})
        echo_profile = dict(material_profile)
        echo_profile.update(explicit_echo_profile)

        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "attenuation_model",
            coast_data.get("attenuation_model", DEFAULT_RADARLOC_ATTENUATION_MODEL),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "num_rays",
            _env_int("RADARLOC_NUM_RAYS", material_profile.get("num_rays", 9)),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "edge_factor",
            _env_float("RADARLOC_EDGE_FACTOR", material_profile.get("edge_factor", 0.78)),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "leading_edge_boost",
            _env_float(
                "RADARLOC_LEADING_EDGE_BOOST",
                material_profile.get("leading_edge_boost", 1.12),
            ),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "far_range_floor",
            _env_float("RADARLOC_FAR_RANGE_FLOOR", material_profile.get("far_range_floor", 0.35)),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "range_decay_strength",
            _env_float(
                "RADARLOC_RANGE_DECAY_STRENGTH",
                material_profile.get("range_decay_strength", 0.85),
            ),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "continuous_return_threshold",
            _env_float("RADARLOC_CONTINUOUS_RETURN_THRESHOLD", 0.06),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "continuous_return_depth_m",
            _env_float(
                "RADARLOC_CONTINUOUS_RETURN_DEPTH_M",
                material_profile.get("continuous_return_depth_m", 800.0),
            ),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "continuous_return_decay_m",
            _env_float("RADARLOC_CONTINUOUS_RETURN_DECAY_M", 1500.0),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "continuous_return_floor",
            _env_float("RADARLOC_CONTINUOUS_RETURN_FLOOR", 0.22),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "continuous_texture_strength",
            _env_float(
                "RADARLOC_CONTINUOUS_TEXTURE_STRENGTH",
                material_profile.get("continuous_texture_strength", 0.30),
            ),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "continuous_texture_floor",
            _env_float(
                "RADARLOC_CONTINUOUS_TEXTURE_FLOOR",
                material_profile.get("continuous_texture_floor", 0.48),
            ),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "origin_range_attenuation",
            _env_bool("RADARLOC_ORIGIN_RANGE_ATTENUATION", True),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "origin_range_ref_m",
            _env_float("RADARLOC_ORIGIN_RANGE_REF_M", 6000.0),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "origin_range_power",
            _env_float("RADARLOC_ORIGIN_RANGE_POWER", 4.0),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "origin_range_floor",
            _env_float("RADARLOC_ORIGIN_RANGE_FLOOR", 0.04),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "occlude_after_first_land",
            _env_bool("RADARLOC_OCCLUDE_AFTER_FIRST_LAND", True),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "secondary_return_scale",
            _env_float("RADARLOC_SECONDARY_RETURN_SCALE", 0.08),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "secondary_return_max_code",
            _env_float("RADARLOC_SECONDARY_RETURN_MAX_CODE", 36.0),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "ray_core_depth_m",
            _env_float("RADARLOC_RAY_CORE_DEPTH_M", material_profile.get("ray_core_depth_m", 1.5)),
        )
        _set_profile_default(
            echo_profile,
            explicit_echo_profile,
            "ray_penetration_max_m",
            _env_float(
                "RADARLOC_RAY_PENETRATION_MAX_M",
                material_profile.get("ray_penetration_max_m", 20.0),
            ),
        )
        topology_managed = any(
            key in coast_data
            for key in (
                "topology_range_clipped",
                "topology_extensions",
                "topology_unresolved_endpoints",
            )
        )
        if (
            not topology_managed
            and assume_topology_managed_open
            and not is_closed
            and feature_class in MAJOR_SHORELINE_FEATURE_CLASSES
        ):
            topology_managed = True

        coastline = Coastline()
        if "reflectivity" in coast_data:
            coastline.reflectivity = float(coast_data.get("reflectivity", coastline.reflectivity))
        elif has_feature_class:
            coastline.reflectivity = _default_reflectivity_for_feature(feature_class)
        # Tiny islets and marsh specks (dredge-spoil dots, tidal hummocks) sit
        # at water level; real radar returns little or nothing from them. Damp
        # both brightness and interior fill by ring area.
        if (
            is_closed
            and len(simplified_points) >= 3
            and feature_class in (
                {"small_land_feature"} | WETLAND_FEATURE_CLASSES
            )
        ):
            tiny_area_m2 = _env_float("RADARLOC_TINY_FEATURE_AREA_M2", 40_000.0)
            ring_area = _polygon_area(
                [CoastlinePoint(px_, py_) for px_, py_ in simplified_points]
            )
            if ring_area < tiny_area_m2:
                coastline.reflectivity *= 0.30
                echo_profile = dict(echo_profile or {})
                echo_profile["continuous_return_depth_m"] = 40.0
                echo_profile["far_range_floor"] = 0.04
                echo_profile["leading_edge_boost"] = 1.0
        coastline.configure_rendering(
            mode=coast_data.get("render_mode", DEFAULT_RADARLOC_RENDER_MODE),
            echo_profile=echo_profile,
        )
        coastline.source_feature_id = str(coast_data.get("id", "") or "")
        coastline.source_feature_class = str(feature_class or "")
        coastline.source_name = str(coast_data.get("name", "") or "")
        coastline.legacy_geometry_mode = (result.origin_surface_source == "legacy_geometry_inference")
        coastline.is_closed_polygon = bool(is_closed)
        coastline.topology_managed = topology_managed
        coastline.topology_unresolved_endpoints = tuple(
            str(value) for value in coast_data.get("topology_unresolved_endpoints", [])
        )
        coastline.set_points(simplified_points)
        # Close polygon if flagged
        if is_closed and len(simplified_points) >= 3:
            first = simplified_points[0]
            last = simplified_points[-1]
            if abs(first[0] - last[0]) > 0.1 or abs(first[1] - last[1]) > 0.1:
                coastline.add_point(first[0], first[1])
            closed_indices.add(len(result.coastlines))
        result.coastlines.append(coastline)

    # Marina pontoons: piers whose geometry lies inside a dock/marina basin
    # are floating structures at water level -- a whisper on real radar,
    # unlike fixed harbor piers. Dampen them after all features are loaded.
    dock_polys = []
    for coastline in result.coastlines:
        if (
            getattr(coastline, "is_closed_polygon", False)
            and str(getattr(coastline, "source_feature_class", "")).lower() == "dock"
            and len(coastline.points) >= 3
        ):
            dock_polys.append(coastline.points)
    if dock_polys:
        for coastline in result.coastlines:
            cls = str(getattr(coastline, "source_feature_class", "")).lower()
            if cls not in {"pier", "jetty", "quay"} or not coastline.points:
                continue
            mid = coastline.points[len(coastline.points) // 2]
            if any(_point_in_polygon(mid.x, mid.y, poly) for poly in dock_polys):
                coastline.reflectivity = 0.0
                profile = getattr(coastline, "echo_profile", None)
                if profile is not None:
                    profile.continuous_return_depth_m = 15.0
                    profile.far_range_floor = 0.02
                    profile.leading_edge_boost = 1.0

    # Store closed indices for terrain generation
    result._closed_indices = closed_indices

    # Terrain
    terrain_data = doc.get("terrain", {})
    if terrain_data.get("enabled", False) and "grid" in terrain_data:
        grid_info = terrain_data["grid"]
        elevations = terrain_data.get("elevations", [])
        if elevations:
            grid_array = np.array(elevations, dtype=np.float32)
            config = TerrainConfig(
                origin_x=grid_info.get("origin_x", 0.0),
                origin_y=grid_info.get("origin_y", 0.0),
                cell_size=grid_info.get("cell_size", 50.0),
            )
            result.terrain = HeightMap(config, grid_array)

    # Vessels
    vessel_type_map = {
        "cargo": VesselType.CARGO,
        "tanker": VesselType.TANKER,
        "fishing": VesselType.FISHING,
        "sailing": VesselType.SAILING,
        "passenger": VesselType.PASSENGER,
        "tug": VesselType.TUG,
        "pilot": VesselType.PILOT,
        "buoy": VesselType.BUOY,
    }
    for v_data in doc.get("vessels", []):
        vtype = vessel_type_map.get(v_data.get("type", ""), VesselType.UNKNOWN)
        creation_class = v_data.get("object_creation_type", v_data.get("type"))
        if creation_class in ("small_boat", "medium_boat"):
            vtype = VesselType.FISHING
        elif creation_class == "large_boat":
            vtype = VesselType.CARGO
        elif creation_class == "barge":
            vtype = VesselType.TANKER
        elif creation_class != "buoy":
            creation_class = None
        vessel = Vessel(
            id=v_data.get("id", "vessel"),
            name=v_data.get("name", ""),
            vessel_type=vtype,
            x=v_data.get("x", 0.0),
            y=v_data.get("y", 0.0),
            course=v_data.get("course", 0.0),
            speed=v_data.get("speed", 0.0),
            length=float(v_data.get("length", v_data.get("length_m", 50.0))),
            beam=float(v_data.get("beam", v_data.get("width_m", 10.0))),
            height=float(v_data.get("height", 15.0)),
            rcs=v_data.get("rcs", v_data.get("rcs_m2")),
            heading_deg=v_data.get("heading_deg"),
            object_creation_class=creation_class,
        )
        result.vessels.append(vessel)

    # Ensure coastlines create occlusion terrain.
    # - No explicit terrain: generate 13m land from coastline polygons
    # - Explicit terrain: merge 13m minimum for land areas (use max of real vs 13m)
    # Only use CLOSED polygons for terrain fill (unclosed are shoreline segments)
    terrain_authoritative = bool(
        meta.get("terrain_authoritative", False)
        or (doc.get("terrain", {}) or {}).get("authoritative", False)
    )
    if terrain_authoritative and result.terrain is not None:
        # Retain the generator raster, then repair coarse cells from closed OSM
        # water rings before it drives primary/secondary classification.
        _repair_authoritative_water_raster(
            result.terrain, result.coastlines, result._closed_indices
        )
        # The terrain is scene view/occlusion support only; ray-hit coastline
        # polygons still create the display echo.
        setattr(result.terrain, "generated_from_coastline", True)
    elif result.coastlines:
        coastline_terrain = _terrain_from_coastlines(
            result.coastlines, result.range_nm,
            closed_indices=result._closed_indices,
            origin_surface=result.origin_surface,
            scene_topology=result.scene_topology)
        if coastline_terrain is not None:
            setattr(coastline_terrain, "generated_from_coastline", True)
        if result.terrain is None:
            result.terrain = coastline_terrain
        elif coastline_terrain is not None:
            # Merge: land cells get max(explicit_elevation, 13m)
            result.terrain = _merge_terrain(result.terrain, coastline_terrain)
            setattr(result.terrain, "generated_from_coastline", False)

    # Over-water structures (marina pontoons, finger piers, floating
    # breakwaters) sit at water level and return essentially nothing on a
    # real display. Any pier/quay/jetty/breakwater/groyne whose geometry is
    # mostly over mapped water is silenced; structures on land keep echoing.
    if result.terrain is not None and result.coastlines:
        _grid = result.terrain.grid
        _cfg = result.terrain.config
        _rows, _cols = _grid.shape
        for _c in result.coastlines:
            _cls = str(getattr(_c, "source_feature_class", "")).strip().lower()
            if _cls not in DETAIL_FEATURE_CLASSES or not _c.points:
                continue
            _wet = _tot = 0
            _stride = max(1, len(_c.points) // 40)
            for _p in _c.points[::_stride]:
                _cc = int((_p.x - _cfg.origin_x) / _cfg.cell_size)
                _rr = int((_p.y - _cfg.origin_y) / _cfg.cell_size)
                if 0 <= _rr < _rows and 0 <= _cc < _cols:
                    _tot += 1
                    if _grid[_rr, _cc] <= 0.5:
                        _wet += 1
            if _tot and (_wet / _tot) >= 0.6:
                _c.reflectivity = 0.0

    return result


def _merge_terrain(explicit: HeightMap, coastline: HeightMap) -> HeightMap:
    """Merge explicit elevation data with coastline-derived terrain.

    Rules:
    - Water areas (coastline=0): Always use 0m (water overrides explicit terrain)
    - Land areas (coastline>0): Use max(explicit, coastline) for occlusion

    The merge happens on whichever grid is finer. Explicit DEM grids in lake
    exports are often coarse (e.g. 128x128, ~300 m cells at 10+ nm ranges);
    downsampling the 768x768 coastline mask onto them used to reduce the whole
    shoreline to blocky pixels. Elevation interpolates gracefully to a finer
    grid, but a land/water mask does not survive being decimated.
    """
    ex_grid = explicit.grid
    co_grid = coastline.grid
    ex_cells = ex_grid.shape[0] * ex_grid.shape[1]
    co_cells = co_grid.shape[0] * co_grid.shape[1]

    if co_cells >= ex_cells:
        target_shape, target_config = co_grid.shape, coastline.config
    else:
        target_shape, target_config = ex_grid.shape, explicit.config

    def _resample(grid, shape, order):
        if grid.shape == shape:
            return grid
        scale_r = shape[0] / grid.shape[0]
        scale_c = shape[1] / grid.shape[1]
        try:
            from scipy.ndimage import zoom
            out = zoom(grid, (scale_r, scale_c), order=order)
        except ImportError:
            # No scipy - nearest neighbor via index maps
            rows = np.minimum((np.arange(shape[0]) / scale_r).astype(int), grid.shape[0] - 1)
            cols = np.minimum((np.arange(shape[1]) / scale_c).astype(int), grid.shape[1] - 1)
            out = grid[rows[:, None], cols[None, :]]
        # zoom can come out a cell short of the target on non-integer scales
        if out.shape != shape:
            fixed = np.zeros(shape, dtype=out.dtype)
            fixed[:out.shape[0], :out.shape[1]] = out[:shape[0], :shape[1]]
            if out.shape[0] < shape[0]:
                fixed[out.shape[0]:, :] = fixed[out.shape[0] - 1, :]
            if out.shape[1] < shape[1]:
                fixed[:, out.shape[1]:] = fixed[:, out.shape[1] - 1][:, None]
            out = fixed
        return out

    # Elevation interpolates (order=1); the land/water mask must stay nearest
    # neighbour (order=0) so no half-land cells appear along the shoreline.
    ex_grid = _resample(ex_grid, target_shape, order=1)
    co_grid = _resample(co_grid, target_shape, order=0)

    # Merge: water (coastline=0) stays 0, land uses max elevation
    merged = np.where(co_grid == 0, 0.0, np.maximum(ex_grid, co_grid))

    return HeightMap(target_config, merged.astype(np.float32))


def _rdp_keep_mask(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Return a boolean keep-mask using iterative Ramer-Douglas-Peucker."""
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = True
    keep[-1] = True
    if len(points) <= 2 or epsilon <= 0.0:
        keep[:] = True
        return keep

    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end - start <= 1:
            continue
        seg = points[end] - points[start]
        seg_len_sq = float(seg[0] * seg[0] + seg[1] * seg[1])
        interior = points[start + 1:end]
        if len(interior) == 0:
            continue

        rel = interior - points[start]
        if seg_len_sq <= 1e-12:
            dists = np.hypot(rel[:, 0], rel[:, 1])
        else:
            t = np.clip((rel @ seg) / seg_len_sq, 0.0, 1.0)
            proj = points[start] + np.outer(t, seg)
            diff = interior - proj
            dists = np.hypot(diff[:, 0], diff[:, 1])

        max_idx = int(np.argmax(dists))
        max_dist = float(dists[max_idx])
        if max_dist > epsilon:
            split_idx = start + 1 + max_idx
            keep[split_idx] = True
            stack.append((start, split_idx))
            stack.append((split_idx, end))

    return keep


def _simplify_loaded_points(points: list, tolerance_m: float,
                            closed: bool, min_points: int) -> list:
    """Simplify loaded coastline points conservatively for faster rendering."""
    if len(points) < max(3, min_points) or tolerance_m <= 0.0:
        return [(float(pt["x"]), float(pt["y"])) for pt in points]

    arr = np.array([(float(pt["x"]), float(pt["y"])) for pt in points], dtype=np.float32)
    if closed and len(arr) >= 2:
        first = arr[0]
        last = arr[-1]
        if np.hypot(*(last - first)) <= 0.1:
            arr = arr[:-1]
    if len(arr) < max(3, min_points):
        return [(float(x), float(y)) for x, y in arr]

    # Drop near-duplicate consecutive vertices first.
    deltas = np.diff(arr, axis=0)
    keep_dense = np.ones(len(arr), dtype=bool)
    keep_dense[1:] = np.hypot(deltas[:, 0], deltas[:, 1]) > max(0.05, tolerance_m * 0.2)
    arr = arr[keep_dense]
    if len(arr) < 3:
        arr = np.array([(float(pt["x"]), float(pt["y"])) for pt in points], dtype=np.float32)
        if closed and len(arr) >= 2 and np.hypot(*(arr[-1] - arr[0])) <= 0.1:
            arr = arr[:-1]

    if len(arr) >= max(3, min_points):
        keep_mask = _rdp_keep_mask(arr, tolerance_m)
        arr = arr[keep_mask]

    if closed and len(arr) >= 3:
        arr = np.vstack([arr, arr[0]])
    return [(float(x), float(y)) for x, y in arr]


def _polygon_area(points) -> float:
    """Calculate signed area of a polygon using the shoelace formula."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y
        area -= points[j].x * points[i].y
    return abs(area) / 2.0


def _point_in_polygon(px: float, py: float, points: list) -> bool:
    """Ray casting algorithm to test if point is inside polygon."""
    n = len(points)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = points[i].x, points[i].y
        xj, yj = points[j].x, points[j].y
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _coastline_endpoint_gap(points: list) -> float:
    if len(points) < 2:
        return float("inf")
    return float(np.hypot(points[-1].x - points[0].x, points[-1].y - points[0].y))


def _endpoint_radius(points: list) -> float:
    if len(points) < 2:
        return float("inf")
    return max(
        float(np.hypot(points[0].x, points[0].y)),
        float(np.hypot(points[-1].x, points[-1].y)),
    )


def _point_to_segment_distance_and_projection(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
):
    """Return point-to-segment distance plus the projected world coordinate."""
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-12:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len_sq))
    qx = ax + t * vx
    qy = ay + t * vy
    return float(np.hypot(px - qx, py - qy)), qx, qy, t


def _close_binary_mask(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Close tiny raster seams without broadly reshaping the shoreline mask."""
    iterations = max(0, int(radius_cells))
    if iterations <= 0:
        return mask
    try:
        from scipy.ndimage import binary_closing

        size = 2 * iterations + 1
        structure = np.ones((size, size), dtype=bool)
        return binary_closing(mask, structure=structure, iterations=1)
    except Exception:
        closed = mask.copy()
        for _ in range(iterations):
            padded = np.pad(closed, 1, mode="constant", constant_values=False)
            dilated = np.zeros_like(closed, dtype=bool)
            for dr in range(3):
                for dc in range(3):
                    dilated |= padded[dr:dr + closed.shape[0], dc:dc + closed.shape[1]]
            padded = np.pad(dilated, 1, mode="constant", constant_values=False)
            eroded = np.ones_like(closed, dtype=bool)
            for dr in range(3):
                for dc in range(3):
                    eroded &= padded[dr:dr + closed.shape[0], dc:dc + closed.shape[1]]
            closed = eroded
        return closed


def _dilate_binary_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilate a boolean mask with a compact 3x3 neighborhood."""
    steps = max(0, int(iterations))
    if steps <= 0:
        return mask
    try:
        from scipy.ndimage import binary_dilation

        return binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=steps)
    except Exception:
        grown = mask.copy()
        for _ in range(steps):
            padded = np.pad(grown, 1, mode="constant", constant_values=False)
            dilated = np.zeros_like(grown, dtype=bool)
            for dr in range(3):
                for dc in range(3):
                    dilated |= padded[dr:dr + grown.shape[0], dc:dc + grown.shape[1]]
            grown = dilated
        return grown


def _scanline_fill_polygon(grid: np.ndarray, points: list,
                           origin_x: float, origin_y: float,
                           cell_size: float, fill_value: float) -> None:
    """Fill a single polygon into the grid using scanline algorithm."""
    grid_size = grid.shape[0]
    n = len(points)
    if n < 3:
        return

    pts = np.array([(p.x, p.y) for p in points], dtype=np.float32)
    p1 = pts
    p2 = np.roll(pts, -1, axis=0)
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    x2 = p2[:, 0]
    y2 = p2[:, 1]

    min_y = float(np.min(y1))
    max_y = float(np.max(y1))
    row_start = max(0, int((min_y - origin_y) / cell_size) - 1)
    row_end = min(grid_size - 1, int((max_y - origin_y) / cell_size) + 1)

    for r in range(row_start, row_end + 1):
        wy = origin_y + (r + 0.5) * cell_size
        mask = ((y1 <= wy) & (wy < y2)) | ((y2 <= wy) & (wy < y1))
        if not np.any(mask):
            continue
        denom = y2[mask] - y1[mask]
        if len(denom) == 0:
            continue
        t = (wy - y1[mask]) / denom
        crossings = x1[mask] + t * (x2[mask] - x1[mask])
        if len(crossings) < 2:
            continue

        crossings.sort()
        starts = crossings[0::2]
        ends = crossings[1::2]
        for x_start, x_end in zip(starts, ends):
            c_start = max(0, int((float(x_start) - origin_x) / cell_size))
            c_end = min(grid_size, int((float(x_end) - origin_x) / cell_size) + 1)
            if c_start < c_end:
                grid[r, c_start:c_end] = fill_value


def _repair_authoritative_water_raster(
    terrain: HeightMap, coastlines: list, closed_indices: set
) -> int:
    """Repair coarse authoritative cells from closed OSM water polygons.

    The supplied grid remains the base terrain, while closed harbor and channel
    rings correct water cells that a coarse raster marked as land. Wetlands,
    beaches, and genuine small land rings are reasserted afterwards. Marina
    detail is intentionally not reasserted: water-level pontoons must not form
    terrain barriers or false shoreline echoes.
    """
    if terrain is None or not coastlines or not closed_indices:
        return 0

    grid = terrain.grid
    cfg = terrain.config
    water_mask = np.zeros_like(grid, dtype=np.uint8)
    land_mask = np.zeros_like(grid, dtype=np.uint8)
    for index in closed_indices:
        if not (0 <= index < len(coastlines)):
            continue
        coastline = coastlines[index]
        points = getattr(coastline, "points", None) or []
        if len(points) < 3:
            continue
        feature_class = str(getattr(coastline, "source_feature_class", "")).strip().lower()
        if _feature_prefers_water_fill(feature_class):
            _scanline_fill_polygon(
                water_mask, points, cfg.origin_x, cfg.origin_y, cfg.cell_size, 1.0
            )
        elif (
            feature_class == "small_land_feature"
            or feature_class in WETLAND_FEATURE_CLASSES
            or feature_class in BEACH_FEATURE_CLASSES
        ):
            _scanline_fill_polygon(
                land_mask, points, cfg.origin_x, cfg.origin_y, cfg.cell_size, 1.0
            )

    repair = (water_mask > 0) & ~(land_mask > 0)
    repaired_cells = int(np.count_nonzero(repair & (grid > 0.5)))
    if np.any(repair):
        grid[repair] = 0.0
    if np.any(land_mask):
        grid[land_mask > 0] = np.maximum(grid[land_mask > 0], 1.0)
    setattr(terrain, "authoritative_water_cells_repaired", repaired_cells)
    return repaired_cells


def _terrain_from_coastlines(coastlines: list, range_nm: float,
                             land_height: float = 13.0,
                             grid_size: int = 768,  # Finer loaded-coastline land masks
                             closed_indices: set = None,
                             origin_surface: str = "unknown",
                             scene_topology: str = "") -> HeightMap:
    """Generate a terrain height map by rasterizing coastline polygons.

    Handles two cases:
    1. Lakes/reservoirs: Many closed polygons that ENCLOSE water
       → Start with land, fill polygons as water
    2. Ocean coastlines: Mostly open ways marking shore boundary
       → Start with water at center, flood-fill land from coastlines

    Detection:
    - Closed polygons that enclose the own-ship origin strongly indicate a
      lake/reservoir-style scene where polygons define water bodies.
    - Open shoreline segments strongly indicate a coastal/estuary scene where
      the shoreline should act as a barrier and outside flood-fill should mark
      land.
    - Broad closed coverage can still trigger lake mode when no open shoreline
      segments exist, but the threshold is intentionally conservative.

    Args:
        coastlines: List of Coastline objects.
        range_nm: Radar range in nautical miles (defines grid extent).
        land_height: Default land elevation in meters.
        grid_size: Grid resolution (rows and cols).
        closed_indices: Set of indices indicating which coastlines are closed.

    Returns:
        HeightMap with land cells elevated, water cells at 0.
    """
    from collections import deque

    if str(scene_topology or "").strip().lower() == "open_shore":
        grid_size = _open_shore_grid_size(grid_size, len(coastlines))

    range_m = range_nm * 1852.0
    cell_size = (2 * range_m) / grid_size
    origin_x = -range_m
    origin_y = -range_m

    if closed_indices is None:
        closed_indices = set(range(len(coastlines)))

    auto_close_gap_m = max(12.0, min(220.0, cell_size * 14.0))
    endpoint_edge_guard_m = max(40.0, min(range_m * 0.08, 500.0))
    origin_surface = str(origin_surface or "unknown").strip().lower()
    scene_topology = str(scene_topology or "").strip().lower()

    # Collect closed polygons with bounding boxes. Also promote near-closed
    # open chains into fill-only polygons so older .radarloc files with small
    # endpoint gaps still produce a sane harbor mask.
    closed_polygons = []
    total_closed_area = 0.0
    for idx, coastline in enumerate(coastlines):
        pts = coastline.points
        if len(pts) < 3:
            continue

        is_closed = idx in closed_indices
        if not is_closed:
            gap_m = _coastline_endpoint_gap(pts)
            endpoint_radius = _endpoint_radius(pts)
            if gap_m <= auto_close_gap_m and endpoint_radius <= range_m + endpoint_edge_guard_m:
                pts = list(pts)
                first = pts[0]
                last = pts[-1]
                if abs(first.x - last.x) > 0.1 or abs(first.y - last.y) > 0.1:
                    pts.append(CoastlinePoint(first.x, first.y))
                is_closed = True

        if is_closed:
            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            bbox = (min(xs), max(xs), min(ys), max(ys))
            area = _polygon_area(pts)
            total_closed_area += area
            closed_polygons.append((
                pts,
                bbox,
                area,
                {
                    "feature_id": getattr(coastline, "source_feature_id", ""),
                    "feature_class": getattr(coastline, "source_feature_class", ""),
                },
            ))

    # Calculate total grid area
    grid_area = (2 * range_m) ** 2

    open_segment_count = max(0, len(coastlines) - len(closed_indices))
    center_in_closed_polygon = any(
        _point_in_polygon(0.0, 0.0, pts)
        for pts, _, _, _ in closed_polygons
    )

    # Decide mode:
    # - Any open shoreline segments push the scene toward coastal/ocean mode.
    # - Enclosed-water scenes still use lake mode when the own-ship origin sits
    #   inside a closed polygon or when closed-water coverage is very dominant.
    use_lake_mode = bool(closed_polygons) and open_segment_count == 0 and (
        center_in_closed_polygon or total_closed_area > grid_area * 0.20
    )
    if scene_topology == "open_shore":
        use_lake_mode = False
    elif scene_topology == "enclosed_water" and closed_polygons and open_segment_count == 0:
        use_lake_mode = True

    if use_lake_mode:
        # LAKE MODE: Closed polygons enclose water
        grid = np.full((grid_size, grid_size), land_height, dtype=np.float32)

        # Sort by area descending
        closed_polygons.sort(key=lambda x: x[2], reverse=True)

        # Fill closed polygons as water using scanlines rather than
        # per-cell point-in-polygon tests.
        for pts, _, _, _ in closed_polygons:
            _scanline_fill_polygon(grid, pts, origin_x, origin_y, cell_size, 0.0)
    else:
        # OCEAN / HARBOR MODE:
        # Treat the own-ship origin as known water and the map edges as known
        # land. Shoreline segments act as barriers, and each open cell is
        # classified by whichever side can reach it more naturally.
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Draw all coastline segments as barriers
        def world_to_grid(wx, wy):
            c = int((wx - origin_x) / cell_size)
            r = int((wy - origin_y) / cell_size)
            return max(0, min(grid_size-1, r)), max(0, min(grid_size-1, c))

        barrier = np.zeros((grid_size, grid_size), dtype=np.uint8)
        major_feature_cells: dict[int, set[tuple[int, int]]] = {}
        water_edge_seeds = set()
        water_side_seeds = set()
        land_side_seeds = set()
        topology_managed_open = any(
            idx not in closed_indices
            and bool(getattr(coastline, "topology_managed", False))
            for idx, coastline in enumerate(coastlines)
        )
        range_ring_mask = np.zeros((grid_size, grid_size), dtype=bool)
        edge_seed_margin = 1
        edge_seed_span = 2

        def _mark_water_edge_seed(r: int, c: int) -> None:
            if r <= edge_seed_margin:
                for cc in range(max(0, c - edge_seed_span), min(grid_size, c + edge_seed_span + 1)):
                    water_edge_seeds.add((0, cc))
            if r >= grid_size - 1 - edge_seed_margin:
                for cc in range(max(0, c - edge_seed_span), min(grid_size, c + edge_seed_span + 1)):
                    water_edge_seeds.add((grid_size - 1, cc))
            if c <= edge_seed_margin:
                for rr in range(max(0, r - edge_seed_span), min(grid_size, r + edge_seed_span + 1)):
                    water_edge_seeds.add((rr, 0))
            if c >= grid_size - 1 - edge_seed_margin:
                for rr in range(max(0, r - edge_seed_span), min(grid_size, r + edge_seed_span + 1)):
                    water_edge_seeds.add((rr, grid_size - 1))

        def _draw_barrier_line(
            x1: float,
            y1: float,
            x2: float,
            y2: float,
            *,
            mark_coast_opening: bool = False,
            feature_cell_collector: set[tuple[int, int]] | None = None,
        ) -> None:
            r1, c1 = world_to_grid(x1, y1)
            r2, c2 = world_to_grid(x2, y2)
            dr, dc = abs(r2 - r1), abs(c2 - c1)
            sr, sc = (1 if r1 < r2 else -1), (1 if c1 < c2 else -1)
            err = dr - dc
            while True:
                barrier[r1, c1] = 1
                grid[r1, c1] = land_height
                if feature_cell_collector is not None:
                    feature_cell_collector.add((r1, c1))
                if mark_coast_opening:
                    _mark_water_edge_seed(r1, c1)
                if r1 == r2 and c1 == c2:
                    break
                prev_r, prev_c = r1, c1
                e2 = 2 * err
                if e2 > -dc:
                    err -= dc
                    r1 += sr
                if e2 < dr:
                    err += dr
                    c1 += sc
                if r1 != prev_r and c1 != prev_c:
                    # A bare Bresenham line is only 8-connected, so a diagonal
                    # step leaves a gap an 8-connected flood can squeeze through.
                    # Mark the orthogonal partner cell to keep barriers sealed.
                    barrier[prev_r, c1] = 1
                    grid[prev_r, c1] = land_height
                    if feature_cell_collector is not None:
                        feature_cell_collector.add((prev_r, c1))

        def _mark_shore_sides(x1: float, y1: float, x2: float, y2: float,
                              feature_class: str) -> None:
            """Seed land/water sides without turning the range ring into terrain."""
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            if length <= 1e-6:
                return
            nx, ny = -dy / length, dx / length
            offset_m = max(2.25 * cell_size, 2.0)
            steps = max(1, min(128, int(np.ceil(length / max(cell_size * 8.0, 1.0)))))
            normalized_class = str(feature_class or "").strip().lower()
            for step in range(steps + 1):
                t = step / steps
                mx = x1 + dx * t
                my = y1 + dy * t
                left = (mx + nx * offset_m, my + ny * offset_m)
                right = (mx - nx * offset_m, my - ny * offset_m)
                if normalized_class == "coastline":
                    # OSM natural=coastline ways are directed with land on the
                    # left and water on the right.
                    land_point, water_point = left, right
                elif np.hypot(*left) >= np.hypot(*right):
                    land_point, water_point = left, right
                else:
                    land_point, water_point = right, left
                if np.hypot(*land_point) < range_m - cell_size:
                    land_side_seeds.add(world_to_grid(*land_point))
                if np.hypot(*water_point) < range_m - cell_size:
                    water_side_seeds.add(world_to_grid(*water_point))

        for idx, coastline in enumerate(coastlines):
            pts = coastline.points
            if len(pts) < 2:
                continue
            feature_class = str(getattr(coastline, "source_feature_class", "") or "").strip().lower()
            skip_barrier = (
                scene_topology == "open_shore"
                and idx in closed_indices
                and _feature_prefers_water_fill(feature_class)
            )
            # Open pier/breakwater/quay lines are radar echo features standing
            # over water, not land/water dividers. Drawing them as barriers
            # fences the harbor into pockets the water flood cannot reach and
            # large water areas end up classified as land.
            if idx not in closed_indices and feature_class in DETAIL_FEATURE_CLASSES:
                skip_barrier = True
            # Closed OSM water polygons already carry their own interior fill.
            # Treating their perimeters as hard barriers can fracture one real
            # harbor or river into disconnected raster islands.
            if skip_barrier:
                continue
            collect_open_feature = (
                scene_topology == "open_shore"
                and idx not in closed_indices
            )
            feature_cell_collector = major_feature_cells.setdefault(idx, set()) if collect_open_feature else None
            mark_coast_opening = (
                scene_topology == "open_shore"
                and idx not in closed_indices
                and feature_class in {"coastline", "shoreline"}
            )
            # Broad harbour exports concatenate several disjoint OSM ways into a
            # single `points` array. The joins between them are not shoreline:
            # they are straight jumps across open water, and drawing them lays
            # spurious land lines across the harbour (and seeds land/water on the
            # wrong sides). Detect them as segments far longer than this way's
            # own typical spacing and leave them out of the raster entirely.
            seam_threshold_m = float("inf")
            if idx not in closed_indices and len(pts) > 8:
                spans = [
                    float(np.hypot(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y))
                    for i in range(len(pts) - 1)
                ]
                median_span = float(np.median(spans))
                if median_span > 0.0:
                    seam_threshold_m = max(150.0, median_span * 12.0)

            for i in range(len(pts) - 1):
                span_m = float(np.hypot(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y))
                if span_m > seam_threshold_m:
                    continue
                _draw_barrier_line(
                    pts[i].x,
                    pts[i].y,
                    pts[i + 1].x,
                    pts[i + 1].y,
                    mark_coast_opening=mark_coast_opening,
                    feature_cell_collector=feature_cell_collector,
                )
                if (
                    scene_topology == "open_shore"
                    and idx not in closed_indices
                    and bool(getattr(coastline, "topology_managed", False))
                ):
                    _mark_shore_sides(
                        pts[i].x,
                        pts[i].y,
                        pts[i + 1].x,
                        pts[i + 1].y,
                        feature_class,
                    )

        if scene_topology == "open_shore" and topology_managed_open:
            rows, cols = np.indices((grid_size, grid_size), dtype=np.float32)
            world_x = origin_x + (cols + 0.5) * cell_size
            world_y = origin_y + (rows + 0.5) * cell_size
            radial_distance = np.hypot(world_x, world_y)
            range_ring_mask = np.abs(radial_distance - range_m) <= cell_size * 0.72
            barrier[range_ring_mask] = 1

        # Seal tiny endpoint gaps between open shoreline fragments so the flood
        # fill does not leak through geometry that should form one harbor edge.
        open_endpoints = []
        for idx, coastline in enumerate(coastlines):
            if idx in closed_indices or len(coastline.points) < 2:
                continue
            # New generator output has already been clipped, intersected, and
            # audited in vector space. Re-bridging it here can close a harbor
            # mouth that the vector pass intentionally left open. Legacy files
            # keep the raster fallback because they lack topology metadata.
            if bool(getattr(coastline, "topology_managed", False)):
                continue
            feature_class = str(getattr(coastline, "source_feature_class", "") or "").strip().lower()
            # Pier/breakwater lines are standalone structures; welding their
            # ends to nearby shoreline invents barriers across open water.
            if feature_class in DETAIL_FEATURE_CLASSES:
                continue
            start = coastline.points[0]
            end = coastline.points[-1]
            open_endpoints.append((idx, "start", start.x, start.y, feature_class))
            open_endpoints.append((idx, "end", end.x, end.y, feature_class))

        bridge_gap_m = max(10.0, min(220.0, cell_size * 12.0))
        used_endpoints = set()
        for i in range(len(open_endpoints)):
            idx_a, end_a, ax, ay, _ = open_endpoints[i]
            key_a = (idx_a, end_a)
            if key_a in used_endpoints:
                continue
            best = None
            for j in range(i + 1, len(open_endpoints)):
                idx_b, end_b, bx, by, _ = open_endpoints[j]
                key_b = (idx_b, end_b)
                if idx_a == idx_b or key_b in used_endpoints:
                    continue
                gap_m = float(np.hypot(ax - bx, ay - by))
                if gap_m > bridge_gap_m:
                    continue
                if best is None or gap_m < best[0]:
                    best = (gap_m, idx_b, end_b, bx, by)
            if best is None:
                continue
            _, idx_b, end_b, bx, by = best
            _draw_barrier_line(ax, ay, bx, by)
            used_endpoints.add(key_a)
            used_endpoints.add((idx_b, end_b))

        # Broad harbor scenes can still contain long but visually obvious
        # shoreline gaps that sit far beyond the local raster bridge threshold.
        # Close only mutual-nearest major shoreline endpoints so we reconnect
        # severed harbor edges like the Charleston / Mount Pleasant corridor
        # without broadly wiring every open water feature together.
        if scene_topology == "open_shore":
            major_bridge_gap_m = max(bridge_gap_m, min(range_m * 0.22, 2600.0))
            remaining = []
            for idx, end_name, x, y, feature_class in open_endpoints:
                key = (idx, end_name)
                if key in used_endpoints or not _feature_supports_major_bridge(feature_class):
                    continue
                remaining.append({
                    "idx": idx,
                    "end_name": end_name,
                    "x": x,
                    "y": y,
                    "feature_class": feature_class,
                })

            nearest_map = {}
            for endpoint in remaining:
                best = None
                for candidate in remaining:
                    if endpoint["idx"] == candidate["idx"]:
                        continue
                    dist_m = float(np.hypot(endpoint["x"] - candidate["x"], endpoint["y"] - candidate["y"]))
                    if dist_m > major_bridge_gap_m:
                        continue
                    if best is None or dist_m < best[0]:
                        best = (dist_m, candidate)
                if best is not None:
                    nearest_map[(endpoint["idx"], endpoint["end_name"])] = best

            for endpoint in remaining:
                key_a = (endpoint["idx"], endpoint["end_name"])
                best = nearest_map.get(key_a)
                if best is None:
                    continue
                _, candidate = best
                key_b = (candidate["idx"], candidate["end_name"])
                reverse = nearest_map.get(key_b)
                if reverse is None or (reverse[1]["idx"], reverse[1]["end_name"]) != key_a:
                    continue
                if key_a in used_endpoints or key_b in used_endpoints or key_a >= key_b:
                    continue
                _draw_barrier_line(endpoint["x"], endpoint["y"], candidate["x"], candidate["y"])
                used_endpoints.add(key_a)
                used_endpoints.add(key_b)

            # Some harbor exports stop a shoreline a few cells shy of another
            # shoreline segment rather than another endpoint. Snapping those
            # dangling ends into the nearest major shoreline segment closes the
            # polygon cleanly and prevents land from spilling across waterways.
            major_snap_gap_m = max(cell_size * 8.0, min(range_m * 0.08, 900.0))
            for endpoint in remaining:
                key_a = (endpoint["idx"], endpoint["end_name"])
                if key_a in used_endpoints:
                    continue
                best_segment = None
                for candidate_idx, candidate_coastline in enumerate(coastlines):
                    if candidate_idx == endpoint["idx"]:
                        continue
                    candidate_class = getattr(candidate_coastline, "source_feature_class", "")
                    if len(candidate_coastline.points) < 2 or not _feature_supports_major_bridge(candidate_class):
                        continue
                    pts = candidate_coastline.points
                    for seg_idx in range(len(pts) - 1):
                        dist_m, qx, qy, _ = _point_to_segment_distance_and_projection(
                            endpoint["x"],
                            endpoint["y"],
                            pts[seg_idx].x,
                            pts[seg_idx].y,
                            pts[seg_idx + 1].x,
                            pts[seg_idx + 1].y,
                        )
                        if dist_m > major_snap_gap_m:
                            continue
                        if best_segment is None or dist_m < best_segment[0]:
                            best_segment = (dist_m, qx, qy)
                if best_segment is None:
                    continue
                _, qx, qy = best_segment
                _draw_barrier_line(endpoint["x"], endpoint["y"], qx, qy)
                used_endpoints.add(key_a)

            unresolved_major_indices = sorted({
                idx
                for idx, end_name, _, _, _ in open_endpoints
                if (idx, end_name) not in used_endpoints
            })
            layered_closure_max_m = max(cell_size * 4.0, min(range_m * 0.07, 900.0))
            layered_closure_max_layers = max(1, int(round(layered_closure_max_m / max(cell_size, 1.0))))
            for feature_idx in unresolved_major_indices:
                cells = major_feature_cells.get(feature_idx)
                if not cells:
                    continue
                rows = [rc[0] for rc in cells]
                cols = [rc[1] for rc in cells]
                pad = layered_closure_max_layers + 2
                r0 = max(0, min(rows) - pad)
                r1 = min(grid_size, max(rows) + pad + 1)
                c0 = max(0, min(cols) - pad)
                c1 = min(grid_size, max(cols) + pad + 1)
                component_mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
                for r, c in cells:
                    component_mask[r - r0, c - c0] = True
                grown_mask = component_mask.copy()
                local_barrier = barrier[r0:r1, c0:c1]
                other_barrier = (local_barrier == 1) & (~component_mask)
                for _ in range(layered_closure_max_layers):
                    dilated = _dilate_binary_mask(grown_mask, 1)
                    ring = dilated & (~grown_mask)
                    if not np.any(ring):
                        break
                    ring_open = ring & (local_barrier == 0)
                    if np.any(ring_open):
                        local_barrier[ring_open] = 1
                        grid[r0:r1, c0:c1][ring_open] = land_height
                    grown_mask |= ring_open
                    ring_contact = ring & other_barrier
                    touches_frame = (
                        (r0 == 0 and np.any(ring[0, :]))
                        or (r1 == grid_size and np.any(ring[-1, :]))
                        or (c0 == 0 and np.any(ring[:, 0]))
                        or (c1 == grid_size and np.any(ring[:, -1]))
                    )
                    if np.any(ring_contact) or touches_frame:
                        break

        water_dist = np.full((grid_size, grid_size), -1, dtype=np.int32)
        land_dist = np.full((grid_size, grid_size), -1, dtype=np.int32)

        # Chamfer (3,4) weights approximate Euclidean distance to within ~2%.
        # A plain 4-connected BFS measures Manhattan distance, whose iso-lines
        # are diamonds; where the water and land floods meet, the tie locus is a
        # broad axis-aligned band rather than a thin seam. Committing those bands
        # to land (the open-shore bias below) is what painted the floating
        # rectangles across Charleston Harbor in the scene view.
        _CHAMFER_ORTH = 3
        _CHAMFER_DIAG = 4

        def _bfs(seed_points, dist_grid):
            # Hot loop: work on flat Python lists rather than indexing numpy
            # element-by-element, which is roughly an order of magnitude slower
            # per access. Results are written back into `dist_grid` at the end.
            open_flat = (barrier == 0).ravel().tolist()
            dist = dist_grid.ravel().tolist()
            width = grid_size
            steps = (
                (-width, _CHAMFER_ORTH), (width, _CHAMFER_ORTH),
                (-1, _CHAMFER_ORTH), (1, _CHAMFER_ORTH),
                (-width - 1, _CHAMFER_DIAG), (-width + 1, _CHAMFER_DIAG),
                (width - 1, _CHAMFER_DIAG), (width + 1, _CHAMFER_DIAG),
            )
            total = width * width
            buckets: dict[int, list] = {}
            for r, c in seed_points:
                if 0 <= r < width and 0 <= c < width:
                    i = r * width + c
                    if open_flat[i] and dist[i] < 0:
                        dist[i] = 0
                        buckets.setdefault(0, []).append(i)
            if not buckets:
                return
            current = 0
            highest = 0
            while current <= highest:
                queue = buckets.pop(current, None)
                if queue is None:
                    current += 1
                    continue
                for i in queue:
                    if dist[i] != current:
                        continue  # superseded by a shorter path already settled
                    col = i % width
                    for offset, step in steps:
                        j = i + offset
                        if j < 0 or j >= total:
                            continue
                        # Reject wrap-around across the left/right edges.
                        if abs((j % width) - col) > 1:
                            continue
                        if not open_flat[j]:
                            continue
                        nd = current + step
                        existing = dist[j]
                        if existing < 0 or nd < existing:
                            dist[j] = nd
                            buckets.setdefault(nd, []).append(j)
                            if nd > highest:
                                highest = nd
                current += 1
            dist_grid[:, :] = np.array(dist, dtype=dist_grid.dtype).reshape(width, width)

        # Seed the flood fill from the radar origin, which should sit on the
        # navigable water body represented by the loaded shoreline geometry.
        center_r, center_c = world_to_grid(0.0, 0.0)
        water_seeds = []
        if origin_surface != "land" and barrier[center_r, center_c] == 0:
            water_seeds.append((center_r, center_c))
        elif origin_surface != "land":
            # If simplification or rasterization places the origin on a barrier
            # cell, search a small neighborhood for the nearest open-water seed.
            for radius in range(1, 7):
                found_seed = False
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = center_r + dr, center_c + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size and barrier[nr, nc] == 0:
                            water_seeds.append((nr, nc))
                            found_seed = True
                            break
                    if found_seed:
                        break
                if found_seed:
                    break
        edge_seeds = []
        for i in range(grid_size):
            for candidate in ((0, i), (grid_size - 1, i), (i, 0), (i, grid_size - 1)):
                if candidate not in water_edge_seeds:
                    edge_seeds.append(candidate)

        if water_edge_seeds:
            water_seeds.extend(sorted(water_edge_seeds))
        if water_side_seeds:
            water_seeds.extend(sorted(water_side_seeds))
        if land_side_seeds:
            edge_seeds.extend(sorted(land_side_seeds))

        _bfs(water_seeds, water_dist)
        _bfs(edge_seeds, land_dist)

        if origin_surface == "land":
            _bfs([(center_r, center_c)], land_dist)

        non_barrier = barrier == 0
        water_only = non_barrier & (water_dist >= 0) & (land_dist < 0)
        land_only = non_barrier & (land_dist >= 0) & (water_dist < 0)
        both = non_barrier & (water_dist >= 0) & (land_dist >= 0)

        grid[land_only] = land_height
        grid[water_only] = 0.0
        if scene_topology == "open_shore":
            # Distances are chamfer units now, so a bias expressed in cells has
            # to be scaled by the orthogonal step weight.
            land_bias_cells = _open_shore_land_bias_cells(cell_size) * _CHAMFER_ORTH
            grid[both] = np.where(
                water_dist[both] + land_bias_cells < land_dist[both],
                0.0,
                land_height,
            )
        else:
            grid[both] = np.where(
                water_dist[both] <= land_dist[both],
                0.0,
                land_height,
            )
        # Even with a near-Euclidean metric the exact tie locus between the two
        # floods can leave one-cell-wide land filaments strung across open
        # water. They are not geography -- on the PPI they read as spurious
        # straight-line targets -- so peel off any non-barrier land cell that
        # has water directly opposite it on either axis. Real shoreline is
        # either thicker than this or backed by a barrier cell, both of which
        # survive; barrier cells are re-asserted immediately below regardless.
        for _ in range(3):
            land_now = grid > 0
            peelable = land_now & (barrier == 0)
            if not np.any(peelable):
                break
            water_now = ~land_now
            horiz = np.zeros_like(peelable)
            horiz[:, 1:-1] = water_now[:, :-2] & water_now[:, 2:]
            vert = np.zeros_like(peelable)
            vert[1:-1, :] = water_now[:-2, :] & water_now[2:, :]
            filament = peelable & (horiz | vert)
            if not np.any(filament):
                break
            grid[filament] = 0.0

        physical_barrier = (barrier == 1) & (~range_ring_mask)
        grid[physical_barrier] = land_height
        # The ring exists only to close the flood-fill domain. Leaving it as
        # elevated terrain would create an artificial echo at maximum range.
        grid[range_ring_mask] = 0.0

        # Also carve closed water polygons that touch the connected water
        # network back into water. This keeps tributaries and side basins that
        # genuinely connect to the harbor, without letting one large detached
        # polygon flood the whole scene.
        mapped_water = np.zeros_like(grid, dtype=bool)
        strong_land_reassert = np.zeros_like(grid, dtype=bool)  # islands, piers
        weak_land_reassert = np.zeros_like(grid, dtype=bool)    # marsh rings
        for pts, _, _, meta in closed_polygons:
            poly_mask = np.zeros_like(grid, dtype=np.uint8)
            _scanline_fill_polygon(poly_mask, pts, origin_x, origin_y, cell_size, 1.0)
            mask = poly_mask > 0
            feature_class = str(meta.get("feature_class", "") or "").strip().lower()
            if _feature_prefers_water_fill(feature_class):
                grid[mask] = 0.0
                # Only unambiguous navigation channels are exempt from the
                # final connectivity pass. Broad "water" polygons around
                # marshes stay gated, else tidal ponds flood areas that are
                # effectively land at radar scale.
                if feature_class in {"river", "canal", "fairway", "strait", "dock"}:
                    mapped_water |= mask
            elif (
                feature_class not in {"legacy_unclassified", "small_land_feature"}
                and feature_class not in DETAIL_FEATURE_CLASSES
                and feature_class not in WETLAND_FEATURE_CLASSES
                and feature_class not in BEACH_FEATURE_CLASSES
                and np.any(water_dist[mask] >= 0)
            ):
                # Piers/breakwaters and marsh rings are land detail; carving
                # them to water because they touch the water network would
                # erase exactly the features the generator now exports.
                grid[mask] = 0.0
            elif (
                feature_class == "small_land_feature"
                or feature_class in DETAIL_FEATURE_CLASSES
            ):
                strong_land_reassert |= mask
            elif feature_class in BEACH_FEATURE_CLASSES:
                strong_land_reassert |= mask
            elif feature_class in WETLAND_FEATURE_CLASSES:
                weak_land_reassert |= mask

        # OSM riverbank polygons frequently span bank-to-bank across marsh
        # platforms and islands; the islands/marsh arrive as their own closed
        # rings. Re-assert them AFTER the water carves so a wide river polygon
        # cannot flatten the land features mapped inside it.
        # Priority: islands/piers > mapped channels > marsh rings. An island
        # mapped as an inner ring of a river polygon is land; a channel dredged
        # through marsh stays water.
        weak_land_reassert &= ~mapped_water
        land_reassert = strong_land_reassert | weak_land_reassert
        if np.any(land_reassert):
            grid[land_reassert] = land_height
            mapped_water &= ~strong_land_reassert

        if scene_topology == "open_shore":
            # Reconnect water bodies split only by tiny raster seams, but keep
            # the harbor mask anchored to actual OSM water polygons. That keeps
            # the fill close to mapped rivers, basins, and channels instead of
            # letting the generic flood stage invent large extra water areas.
            candidate_water = (grid == 0.0) & (barrier == 0)
            seam_close_cells = max(1, int(round(12.0 / max(cell_size, 1.0))))
            candidate_water = _close_binary_mask(candidate_water, seam_close_cells) & (barrier == 0)

            connected_water = np.zeros_like(candidate_water, dtype=bool)
            queue = deque()
            for r, c in water_seeds:
                if 0 <= r < grid_size and 0 <= c < grid_size and candidate_water[r, c] and not connected_water[r, c]:
                    connected_water[r, c] = True
                    queue.append((r, c))
            while queue:
                r, c = queue.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size and candidate_water[nr, nc] and not connected_water[nr, nc]:
                        connected_water[nr, nc] = True
                        queue.append((nr, nc))

            # Mapped OSM water polygons are ground truth even when their
            # connection to the harbor lies outside the range ring (e.g. the
            # Intracoastal Waterway behind a barrier island). Never re-land
            # them just because the flood could not reach them.
            connected_water |= mapped_water & (barrier == 0)
            grid[(barrier == 0) & connected_water] = 0.0
            grid[(barrier == 0) & (~connected_water)] = land_height
            if np.any(land_reassert):
                grid[land_reassert & (barrier == 0)] = land_height

            # Cells beyond the range ring are unreachable by the water flood
            # (the ring is a barrier), so the frame-land seeding paints open
            # ocean corners as terrain. Classify them from the nearest cell
            # just inside the ring instead: ocean edges stay ocean, mainland
            # edges stay mainland.
            if np.any(range_ring_mask):
                try:
                    from scipy.ndimage import distance_transform_edt
                    rows_i, cols_i = np.indices((grid_size, grid_size), dtype=np.float32)
                    wx = origin_x + (cols_i + 0.5) * cell_size
                    wy = origin_y + (rows_i + 0.5) * cell_size
                    outside = np.hypot(wx, wy) > range_m
                    del rows_i, cols_i, wx, wy
                    if np.any(outside):
                        inside = ~outside & (~range_ring_mask)
                        _, (near_r, near_c) = distance_transform_edt(
                            ~inside, return_indices=True)
                        grid[outside] = grid[near_r[outside], near_c[outside]]
                except ImportError:
                    pass

    # Return terrain if there's both land and water
    has_land = np.any(grid > 0)
    has_water = np.any(grid == 0)
    if has_land and has_water:
        config = TerrainConfig(
            origin_x=origin_x,
            origin_y=origin_y,
            cell_size=cell_size,
        )
        return HeightMap(config, grid)
    return None
