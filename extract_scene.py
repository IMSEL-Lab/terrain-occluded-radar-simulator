"""
Radar Scene Extractor - Extract coastlines, objects, clutter, and decay from
real Furuno DRS4DNXT CSV captures. Outputs a scene JSON that can be loaded
by radar_simulator.py via --scene-file to reproduce the capture.

Usage:
    python extract_scene.py input.csv [--validate] [-v]
"""

import os
import sys
import csv
import json
import math
import argparse
import numpy as np
from collections import deque
from datetime import datetime
from typing import List, Tuple, Dict, Optional

_PACKAGE_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _PACKAGE_SOURCE not in sys.path:
    sys.path.insert(0, _PACKAGE_SOURCE)

from ddmr.protocol import get_range_profile

# DRS4DNXT constants (must match radar_simulator.py)
TICKS_PER_REV = 8192
DEAD_ZONE_BINS = 31
FURUNO_GAIN_CODES = [0, 8, 20, 28, 36, 44, 56, 64, 76, 88,
                     104, 120, 140, 164, 188, 220, 252]
COASTLINE_SATURATION_GATE = 220
NOISE_FIELD_SECTORS = 24
NOISE_FIELD_BANDS = 8
NOISE_NEAR_LAND_BINS = 12


# ============================================================
# CSV Loading
# ============================================================

def load_csv(filepath, range_profile=None, radial_mapping=None):
    """Load a Furuno radar CSV. Returns metadata + list of spoke dicts."""
    spokes = []
    status = scale = range_idx = gain = 0
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
                status = int(parts[0])
                scale = int(parts[1])
                range_idx = int(parts[2])
                gain = int(parts[3])
                angle_tick = int(parts[4])
                echoes = [int(v) for v in parts[5:] if v.strip()]
            except ValueError:
                continue
            spokes.append({"angle_tick": angle_tick, "echoes": echoes})

    num_bins = max(len(s["echoes"]) for s in spokes) if spokes else 0
    for s in spokes:
        if len(s["echoes"]) < num_bins:
            s["echoes"].extend([0] * (num_bins - len(s["echoes"])))

    range_max_m = None
    bin_size_m = None
    profile_name = None
    if range_profile is not None:
        profile = get_range_profile(range_profile)
        profile_name = profile.name
        range_max_m = profile.range_max_m(range_idx)
        if radial_mapping is not None:
            bin_size_m = profile.bin_size_m(
                range_idx, num_bins, radial_mapping=radial_mapping
            )

    return {
        "status": status, "scale": scale, "range_index": range_idx,
        "gain": gain, "num_bins": num_bins,
        "range_max_m": range_max_m,
        "bin_size_m": bin_size_m,
        "normalized_bin_size": 1.0 / num_bins if num_bins else None,
        "range_profile": profile_name,
        "radial_mapping": radial_mapping,
        "spokes": spokes,
    }


def split_rotations(spokes):
    """Split spoke list into rotations at angle wraparound."""
    if not spokes:
        return []
    rotations = []
    cur = [spokes[0]]
    for i in range(1, len(spokes)):
        if spokes[i]["angle_tick"] < spokes[i - 1]["angle_tick"] - 100:
            rotations.append(cur)
            cur = []
        cur.append(spokes[i])
    if cur:
        rotations.append(cur)
    return rotations


# ============================================================
# Coastline Detection
# ============================================================

def detect_coastlines(spokes, num_bins, bin_size_m,
                      land_threshold=104, min_run=6, *, peak_gate=None):
    """Detect coastline regions from echo data.
    Returns (land_mask, polygons, echo_map, coastline_echo_data, labels).

    coastline_echo_data: dict keyed by angle_tick string ->
        list of [start_bin, [echo_values]] segments for land bins only.
    """
    ns = len(spokes)
    echo_map = np.zeros((ns, num_bins), dtype=np.int32)
    for i, sp in enumerate(spokes):
        for j in range(min(len(sp["echoes"]), num_bins)):
            echo_map[i, j] = sp["echoes"][j]

    # Phase 1: find land runs per spoke
    land_mask = np.zeros((ns, num_bins), dtype=bool)
    for i in range(ns):
        j = 0
        while j < num_bins:
            if echo_map[i, j] >= land_threshold:
                start = j
                has_sat = False
                while j < num_bins and echo_map[i, j] >= land_threshold:
                    if echo_map[i, j] >= (COASTLINE_SATURATION_GATE if peak_gate is None else peak_gate):
                        has_sat = True
                    j += 1
                if (j - start) >= min_run and has_sat:
                    land_mask[i, start:j] = True
            else:
                j += 1

    # Phase 2: connected-component labeling (8-connected, spoke-wrapped)
    labels = np.zeros((ns, num_bins), dtype=np.int32)
    next_lbl = 1
    for i in range(ns):
        for j in range(num_bins):
            if land_mask[i, j] and labels[i, j] == 0:
                lbl = next_lbl
                next_lbl += 1
                q = deque([(i, j)])
                labels[i, j] = lbl
                while q:
                    ci, cj = q.popleft()
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ni = (ci + di) % ns
                            nj = cj + dj
                            if 0 <= nj < num_bins and land_mask[ni, nj] and labels[ni, nj] == 0:
                                labels[ni, nj] = lbl
                                q.append((ni, nj))

    # Phase 3: build polygons from full land_mask boundaries
    polygons = []
    for lbl in range(1, next_lbl):
        region = (labels == lbl)
        size = int(np.sum(region))
        if size < 30:
            land_mask[region] = False
            labels[region] = 0
            continue

        spoke_edges = {}
        for i in range(ns):
            cols = np.where(region[i])[0]
            if len(cols) == 0:
                continue
            spoke_edges[i] = (int(cols[0]), int(cols[-1]))

        if not spoke_edges:
            continue

        sorted_si = sorted(spoke_edges.keys())

        # Compute arc span to decide if splitting is needed
        bearings_deg = []
        for si in sorted_si:
            brg = spokes[si]["angle_tick"] * 360.0 / TICKS_PER_REV
            bearings_deg.append(brg)

        arc_span = _compute_arc_span(bearings_deg)

        if arc_span > 270.0:
            # Split large ring polygon into ~90 deg sub-polygons
            sub_polys = _split_ring_polygon(sorted_si, spoke_edges, spokes,
                                            bin_size_m, arc_span)
            for sp_idx, sp_poly in enumerate(sub_polys):
                polygons.append({
                    "name": f"coastline_{len(polygons)}",
                    "points": sp_poly["points"],
                    "num_spokes": sp_poly["num_spokes"],
                    "size_bins": sp_poly["size_bins"],
                    "parent_label": lbl,
                })
        else:
            # Standard polygon construction
            inner_pts = []
            outer_pts = []
            for si in sorted_si:
                brg = spokes[si]["angle_tick"] * 360.0 / TICKS_PER_REV
                rad = math.radians(brg)
                first_bin, last_bin = spoke_edges[si]
                inner_r = first_bin * bin_size_m
                outer_r = (last_bin + 1) * bin_size_m
                inner_pts.append((inner_r * math.sin(rad), inner_r * math.cos(rad)))
                outer_pts.append((outer_r * math.sin(rad), outer_r * math.cos(rad)))

            polygon = inner_pts + list(reversed(outer_pts))
            polygons.append({
                "name": f"coastline_{len(polygons)}",
                "points": polygon,
                "num_spokes": len(sorted_si),
                "size_bins": size,
                "parent_label": lbl,
            })

    # Phase 4: per-spoke coastline echo data for validation reference
    # Keyed by spoke index (not tick) to avoid duplicate-tick data loss
    coastline_echo_data = {}
    for i in range(ns):
        segments = []
        j = 0
        while j < num_bins:
            if land_mask[i, j]:
                seg_start = j
                seg_vals = []
                while j < num_bins and land_mask[i, j]:
                    seg_vals.append(int(echo_map[i, j]))
                    j += 1
                segments.append([seg_start, seg_vals])
            else:
                j += 1
        if segments:
            coastline_echo_data[str(i)] = segments

    return land_mask, polygons, echo_map, coastline_echo_data, labels


def _compute_arc_span(bearings_deg):
    """Compute the angular span of a set of bearings in degrees."""
    if len(bearings_deg) < 2:
        return 0.0
    sorted_b = sorted(bearings_deg)
    # Find the largest gap between consecutive bearings
    max_gap = 0.0
    for i in range(1, len(sorted_b)):
        gap = sorted_b[i] - sorted_b[i - 1]
        max_gap = max(max_gap, gap)
    # Also check wraparound gap
    wrap_gap = (360.0 - sorted_b[-1]) + sorted_b[0]
    max_gap = max(max_gap, wrap_gap)
    # Arc span = 360 - largest gap
    return 360.0 - max_gap


def _split_ring_polygon(sorted_si, spoke_edges, spokes, bin_size_m, arc_span):
    """Split a ring polygon into ~90Â° sub-polygons for stable ray-tracing."""
    num_chunks = max(4, int(math.ceil(arc_span / 90.0)))
    chunk_size = max(1, len(sorted_si) // num_chunks)

    sub_polys = []
    for c in range(num_chunks):
        start_idx = c * chunk_size
        end_idx = min((c + 1) * chunk_size, len(sorted_si))
        if c == num_chunks - 1:
            end_idx = len(sorted_si)
        chunk_si = sorted_si[start_idx:end_idx]
        if not chunk_si:
            continue

        inner_pts = []
        outer_pts = []
        chunk_bins = 0
        for si in chunk_si:
            brg = spokes[si]["angle_tick"] * 360.0 / TICKS_PER_REV
            rad = math.radians(brg)
            first_bin, last_bin = spoke_edges[si]
            inner_r = first_bin * bin_size_m
            outer_r = (last_bin + 1) * bin_size_m
            inner_pts.append((inner_r * math.sin(rad), inner_r * math.cos(rad)))
            outer_pts.append((outer_r * math.sin(rad), outer_r * math.cos(rad)))
            chunk_bins += (last_bin - first_bin + 1)

        polygon_pts = inner_pts + list(reversed(outer_pts))
        sub_polys.append({
            "points": polygon_pts,
            "num_spokes": len(chunk_si),
            "size_bins": chunk_bins,
        })

    return sub_polys


# ============================================================
# Coastline Profiles & Statistics
# ============================================================

def build_coastline_profiles(spokes, land_mask, echo_map, num_bins):
    """Build per-spoke coastline bin ranges with echo values.

    Returns dict keyed by spoke index â†’ {tick, segments: [[start, end, [echoes]]]}
    """
    ns = len(spokes)
    profiles = {}
    for i in range(ns):
        segments = []
        j = 0
        while j < num_bins:
            if land_mask[i, j]:
                seg_start = j
                seg_vals = []
                while j < num_bins and land_mask[i, j]:
                    seg_vals.append(int(echo_map[i, j]))
                    j += 1
                segments.append([seg_start, j - 1, seg_vals])
            else:
                j += 1
        if segments:
            profiles[str(i)] = {
                "tick": spokes[i]["angle_tick"],
                "segments": segments,
            }
    return profiles


def _percentile(values, pct):
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    arr = np.array(sorted(float(v) for v in values), dtype=float)
    return float(np.percentile(arr, pct))


def _median_sequence(seqs, max_len=6):
    if not seqs:
        return []
    max_actual = min(max_len, max(len(s) for s in seqs))
    result = []
    for pos in range(max_actual):
        vals_at_pos = [s[pos] for s in seqs if pos < len(s)]
        if vals_at_pos:
            result.append(int(np.median(vals_at_pos)))
    return result


def _default_coastline_stats():
    return {
        "mean_core_echo": 252.0,
        "median_ramp_up": [104, 188],
        "median_ramp_down": [140, 56],
        "typical_ramp_up_bins": 2,
        "typical_ramp_down_bins": 2,
        "median_segment_depth_bins": 12.0,
        "p75_segment_depth_bins": 18.0,
        "segment_count": 0,
    }


def build_coastline_stats(land_mask, echo_map, labels, spokes,
                          num_bins, polygons):
    """Compute per-coastline intensity statistics for realistic echo generation.

    Returns dict keyed by coastline name â†’ stats dict.
    """
    ns = len(spokes)
    stats = {}
    label_stats = {}
    if labels is None or labels.shape != land_mask.shape:
        labels = np.zeros_like(echo_map, dtype=np.int32)
        next_lbl = 1
        for i in range(ns):
            j = 0
            while j < num_bins:
                if land_mask[i, j]:
                    while j < num_bins and land_mask[i, j]:
                        labels[i, j] = next_lbl
                        j += 1
                    next_lbl += 1
                else:
                    j += 1

    accum = {}
    for lbl in np.unique(labels):
        if lbl <= 0:
            continue
        accum[int(lbl)] = {
            "core_vals": [],
            "ramp_up_seqs": [],
            "ramp_down_seqs": [],
            "segment_depths": [],
        }

    for i in range(ns):
        j = 0
        while j < num_bins:
            lbl = int(labels[i, j]) if j < labels.shape[1] else 0
            if lbl > 0 and land_mask[i, j]:
                seg_vals = []
                while j < num_bins and land_mask[i, j] and int(labels[i, j]) == lbl:
                    seg_vals.append(int(echo_map[i, j]))
                    j += 1

                bucket = accum.get(lbl)
                if bucket is None:
                    continue
                bucket["segment_depths"].append(len(seg_vals))
                if len(seg_vals) < 3:
                    bucket["core_vals"].extend(seg_vals)
                    continue

                first_sat = None
                last_sat = None
                for k, v in enumerate(seg_vals):
                    if v >= 252:
                        if first_sat is None:
                            first_sat = k
                        last_sat = k

                if first_sat is not None:
                    ramp_up = seg_vals[:first_sat]
                    core = seg_vals[first_sat:last_sat + 1]
                    ramp_down = seg_vals[last_sat + 1:]
                    bucket["core_vals"].extend(core)
                    if ramp_up:
                        bucket["ramp_up_seqs"].append(ramp_up)
                    if ramp_down:
                        bucket["ramp_down_seqs"].append(ramp_down)
                else:
                    bucket["core_vals"].extend(seg_vals)
            else:
                j += 1

    fallback = _default_coastline_stats()
    for lbl, bucket in accum.items():
        median_ramp_up = _median_sequence(bucket["ramp_up_seqs"])
        median_ramp_down = _median_sequence(bucket["ramp_down_seqs"])
        ramp_up_bins = [len(s) for s in bucket["ramp_up_seqs"]]
        ramp_down_bins = [len(s) for s in bucket["ramp_down_seqs"]]
        mean_core = float(np.mean(bucket["core_vals"])) if bucket["core_vals"] else fallback["mean_core_echo"]
        median_depth = _percentile(bucket["segment_depths"], 50) or fallback["median_segment_depth_bins"]
        p75_depth = _percentile(bucket["segment_depths"], 75) or fallback["p75_segment_depth_bins"]
        label_stats[int(lbl)] = {
            "mean_core_echo": round(mean_core, 1),
            "median_ramp_up": median_ramp_up if median_ramp_up else list(fallback["median_ramp_up"]),
            "median_ramp_down": median_ramp_down if median_ramp_down else list(fallback["median_ramp_down"]),
            "typical_ramp_up_bins": int(np.median(ramp_up_bins)) if ramp_up_bins else fallback["typical_ramp_up_bins"],
            "typical_ramp_down_bins": int(np.median(ramp_down_bins)) if ramp_down_bins else fallback["typical_ramp_down_bins"],
            "median_segment_depth_bins": round(float(median_depth), 2),
            "p75_segment_depth_bins": round(float(p75_depth), 2),
            "segment_count": int(len(bucket["segment_depths"])),
        }

    for poly in polygons:
        parent_label = int(poly.get("parent_label", 0) or 0)
        stats[poly["name"]] = dict(label_stats.get(parent_label, fallback))

    return stats


def build_noise_field_stats(spokes, echo_map, land_mask, obj_mask, num_bins):
    """Build a coarse, forward-renderable structured noise field model."""
    ns = len(spokes)
    if ns == 0 or num_bins <= DEAD_ZONE_BINS:
        return {
            "bearing_sectors": NOISE_FIELD_SECTORS,
            "range_bands": NOISE_FIELD_BANDS,
            "families": {"open_water": {}, "near_land": {}},
        }

    near_land_mask = np.zeros((ns, num_bins), dtype=bool)
    for i in range(ns):
        j = DEAD_ZONE_BINS
        while j < num_bins:
            if land_mask[i, j]:
                start = j
                while j < num_bins and land_mask[i, j]:
                    j += 1
                end = j - 1
                near_start = max(DEAD_ZONE_BINS, start - NOISE_NEAR_LAND_BINS)
                near_end = min(num_bins, end + NOISE_NEAR_LAND_BINS + 1)
                near_land_mask[i, near_start:near_end] = True
            else:
                j += 1

    band_edges = np.linspace(DEAD_ZONE_BINS, num_bins, NOISE_FIELD_BANDS + 1)
    band_edges = [int(round(v)) for v in band_edges]
    band_edges[0] = DEAD_ZONE_BINS
    band_edges[-1] = num_bins

    families = {
        "open_water": {},
        "near_land": {},
    }
    accum = {
        "open_water": {},
        "near_land": {},
    }

    def _cell_accum(family, sector_idx, band_idx):
        key = f"{sector_idx}_{band_idx}"
        family_accum = accum[family]
        if key not in family_accum:
            family_accum[key] = {
                "eligible": 0,
                "active": 0,
                "hist": {},
                "run_lengths": [],
            }
        return family_accum[key]

    for i, spoke in enumerate(spokes):
        sector_idx = min(
            NOISE_FIELD_SECTORS - 1,
            int((spoke["angle_tick"] % TICKS_PER_REV) * NOISE_FIELD_SECTORS / TICKS_PER_REV),
        )
        for band_idx in range(NOISE_FIELD_BANDS):
            band_start = max(DEAD_ZONE_BINS, band_edges[band_idx])
            band_end = min(num_bins, band_edges[band_idx + 1])
            if band_end <= band_start:
                continue
            band_slice = slice(band_start, band_end)
            land_band = land_mask[i, band_slice]
            obj_band = obj_mask[i, band_slice]
            echo_band = echo_map[i, band_slice]
            near_band = near_land_mask[i, band_slice]

            for family, family_mask in (
                ("near_land", near_band),
                ("open_water", ~near_band),
            ):
                eligible_mask = family_mask & (~land_band) & (~obj_band)
                cell = _cell_accum(family, sector_idx, band_idx)
                eligible_count = int(np.count_nonzero(eligible_mask))
                cell["eligible"] += eligible_count
                if eligible_count == 0:
                    continue
                active_mask = eligible_mask & (echo_band > 0)
                active_codes = echo_band[active_mask]
                cell["active"] += int(active_codes.size)
                for code in active_codes.tolist():
                    icode = int(code)
                    cell["hist"][icode] = cell["hist"].get(icode, 0) + 1

                run_len = 0
                for idx in range(len(active_mask)):
                    if eligible_mask[idx] and active_mask[idx]:
                        run_len += 1
                    else:
                        if run_len > 0:
                            cell["run_lengths"].append(run_len)
                            run_len = 0
                if run_len > 0:
                    cell["run_lengths"].append(run_len)

    for family, family_cells in accum.items():
        for key, cell in family_cells.items():
            if cell["eligible"] <= 0:
                continue
            hist_total = sum(cell["hist"].values())
            hist = {}
            if hist_total > 0:
                hist = {
                    str(code): round(count / hist_total, 6)
                    for code, count in sorted(cell["hist"].items())
                }
            median_run_len = float(np.median(cell["run_lengths"])) if cell["run_lengths"] else 1.0
            families[family][key] = {
                "occupancy_probability": round(cell["active"] / max(cell["eligible"], 1), 6),
                "code_histogram": hist,
                "median_run_len": round(max(1.0, median_run_len), 3),
            }

    return {
        "bearing_sectors": NOISE_FIELD_SECTORS,
        "range_bands": NOISE_FIELD_BANDS,
        "families": families,
    }


# ============================================================
# Object Detection
# ============================================================

def _iter_component_pixels_by_label(labels):
    """Yield positive component labels with row-major pixel coordinates.

    One nonzero scan plus a stable group-by avoids rescanning the full label
    image once per component. Stable sorting preserves np.argwhere row order.
    """
    ncols = labels.shape[1]
    rows, cols = np.nonzero(labels > 0)
    if rows.size == 0:
        return

    flat_indices = rows * ncols + cols
    component_labels = labels[rows, cols]
    order = np.argsort(component_labels, kind="stable")
    ordered_labels = component_labels[order]
    ordered_flat_indices = flat_indices[order]
    starts = np.r_[0, np.flatnonzero(ordered_labels[1:] != ordered_labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(ordered_labels)]
    for start, end in zip(starts, ends):
        pixels = np.column_stack(
            np.divmod(ordered_flat_indices[start:end], ncols)
        )
        yield int(ordered_labels[start]), pixels


def detect_objects(spokes, num_bins, bin_size_m, land_mask, echo_map):
    """Detect discrete objects (boats, buoys) in non-coastline areas.
    Returns (objects_list, obj_mask, object_echo_data).

    Tightened from original: 3x3 BFS, min 8 pixels, min peak 56, RCS floor 2.0.
    """
    ns = len(spokes)
    obj_mask = np.zeros((ns, num_bins), dtype=bool)
    for i in range(ns):
        for j in range(DEAD_ZONE_BINS, num_bins):
            if echo_map[i, j] > 0 and not land_mask[i, j]:
                obj_mask[i, j] = True

    # Connected-component labeling with tighter 3x3 neighborhood
    labels = np.zeros((ns, num_bins), dtype=np.int32)
    next_lbl = 1
    for i in range(ns):
        for j in range(num_bins):
            if obj_mask[i, j] and labels[i, j] == 0:
                lbl = next_lbl
                next_lbl += 1
                q = deque([(i, j)])
                labels[i, j] = lbl
                while q:
                    ci, cj = q.popleft()
                    for di in range(-1, 2):       # 3x3 neighborhood
                        for dj in range(-1, 2):   # was 3x5, now 3x3
                            if di == 0 and dj == 0:
                                continue
                            ni = (ci + di) % ns
                            nj = cj + dj
                            if 0 <= nj < num_bins and obj_mask[ni, nj] and labels[ni, nj] == 0:
                                labels[ni, nj] = lbl
                                q.append((ni, nj))

    cluster_records = []
    for lbl, pixels in _iter_component_pixels_by_label(labels):
        rec = _build_object_cluster_record(
            lbl, pixels, spokes, num_bins, bin_size_m, land_mask, obj_mask, echo_map
        )
        if rec is not None:
            cluster_records.append(rec)

    cluster_records = _merge_aligned_clusters(
        cluster_records,
        spokes=spokes,
        bin_size_m=bin_size_m,
        echo_map=echo_map,
    )

    classified_records = []
    for rec in cluster_records:
        geom_type, otype, rcs = _classify_object(
            rec["radial_bins"], rec["unique_spokes"], rec["peak_echo"], rec["obj_range"]
        )
        if rcs < 2.0:
            continue
        enriched = dict(rec)
        enriched["geom_type"] = geom_type
        enriched["type"] = otype
        enriched["rcs_m2"] = float(rcs)
        classified_records.append(enriched)

    classified_records = _merge_same_bearing_major_pairs(
        classified_records,
        spokes=spokes,
        bin_size_m=bin_size_m,
        echo_map=echo_map,
    )

    valid_labels = set()
    label_to_object_idx = {}
    objects = []
    for rec in classified_records:
        for src_lbl in rec["labels"]:
            valid_labels.add(src_lbl)

        radial_bins = rec["radial_bins"]
        otype = rec["type"]
        if otype == "buoy":
            length = max(1.0, radial_bins * bin_size_m)
            width = length
        elif otype == "small_boat":
            length = max(4.0, radial_bins * bin_size_m)
            width = max(2.0, length * 0.4)
        elif otype == "medium_boat":
            length = max(10.0, radial_bins * bin_size_m)
            width = max(4.0, length * 0.35)
        elif otype == "large_boat":
            length = max(25.0, radial_bins * bin_size_m)
            width = max(8.0, length * 0.25)
        else:
            length = max(35.0, radial_bins * bin_size_m)
            width = max(10.0, length * 0.3)

        obj_idx = len(objects)
        for src_lbl in rec["labels"]:
            label_to_object_idx[int(src_lbl)] = obj_idx

        objects.append({
            "type": otype, "geom_type": rec["geom_type"],
            "x": round(rec["cx"], 2), "y": round(rec["cy"], 2),
            "range_m": round(rec["obj_range"], 1), "bearing_deg": round(rec["obj_bearing"], 1),
            "rcs_m2": round(rec["rcs_m2"], 1), "length_m": round(length, 1),
            "width_m": round(width, 1), "heading_deg": round(rec["obj_bearing"], 1),
            "speed_kts": 0.0, "course_deg": 0.0,
            "peak_echo": int(rec["peak_echo"]), "num_pixels": int(rec["n_pixels"]),
            "radial_bins": radial_bins, "angular_spokes": rec["unique_spokes"],
        })

    # Build a clean obj_mask with only valid (filtered) objects
    filtered_obj_mask = np.zeros((ns, num_bins), dtype=bool)
    owner_map = np.full((ns, num_bins), -1, dtype=np.int32)
    for lbl in valid_labels:
        filtered_obj_mask[labels == lbl] = True
        owner_idx = label_to_object_idx.get(int(lbl))
        if owner_idx is not None:
            owner_map[labels == lbl] = int(owner_idx)

    # Build per-spoke object echo data (keyed by spoke index).
    # The optional third element stores the owning object index so strict
    # Mode 4 can preserve support by object while still simulating values.
    object_echo_data = {}
    for i in range(ns):
        segments = []
        j = DEAD_ZONE_BINS
        while j < num_bins:
            if filtered_obj_mask[i, j]:
                seg_start = j
                seg_vals = []
                owner_idx = int(owner_map[i, j]) if owner_map[i, j] >= 0 else None
                while (
                    j < num_bins and filtered_obj_mask[i, j] and
                    (owner_idx is None or int(owner_map[i, j]) == owner_idx)
                ):
                    seg_vals.append(int(echo_map[i, j]))
                    j += 1
                if owner_idx is None:
                    segments.append([seg_start, seg_vals])
                else:
                    segments.append([seg_start, seg_vals, int(owner_idx)])
            else:
                j += 1
        if segments:
            object_echo_data[str(i)] = segments

    return objects, filtered_obj_mask, object_echo_data


def _estimate_rcs(peak_echo, range_m):
    """Invert the radar equation from radar_simulator.py to estimate RCS."""
    if peak_echo <= 0 or range_m <= 0:
        return 10.0
    try:
        ref_range = 50.0
        ref_rcs = 350.0
        ref_code = 190.0
        compression = 55.0
        intensity_dB = compression * math.log10(max(peak_echo, 1) / ref_code)
        range_term = 40.0 * math.log10(max(range_m, 1) / ref_range)
        rcs = ref_rcs * (10.0 ** ((intensity_dB + range_term) / 10.0))
        return max(1.0, min(50000.0, rcs))
    except (ValueError, OverflowError):
        return 10.0


def _geom_classify_object(radial_bins, unique_spokes):
    """Return the purely geometric class before any RCS sanity adjustment."""
    if radial_bins <= 3 and unique_spokes <= 8:
        return "buoy"
    elif radial_bins <= 12 and unique_spokes <= 20:
        return "small_boat"
    elif radial_bins <= 30 and unique_spokes <= 50:
        return "medium_boat"
    return "large_boat"


def _classify_object(radial_bins, unique_spokes, peak_echo, range_m):
    """Classify an extracted object using geometry with an RCS sanity cap."""
    rcs = _estimate_rcs(peak_echo, range_m)
    geo = _geom_classify_object(radial_bins, unique_spokes)

    rcs_class = (
        "buoy" if rcs < 10 else
        "small_boat" if rcs < 100 else
        "medium_boat" if rcs < 800 else
        "large_boat" if rcs < 5000 else
        "barge"
    )

    order = ["buoy", "small_boat", "medium_boat", "large_boat", "barge"]
    return geo, order[min(order.index(geo), order.index(rcs_class))], rcs


def _build_object_cluster_record(lbl, pixels, spokes, num_bins, bin_size_m, land_mask, obj_mask, echo_map):
    """Build a filtered cluster record suitable for later merging/classification."""
    peak_echo = max(echo_map[int(p[0]), int(p[1])] for p in pixels)

    if len(pixels) < 3 or peak_echo < 36:
        return None

    unique_spokes = len(set(int(p[0]) for p in pixels))
    min_bin = int(min(p[1] for p in pixels))
    max_bin = int(max(p[1] for p in pixels))
    radial_bins = max_bin - min_bin + 1
    n_pixels = len(pixels)

    if n_pixels > 50:
        border_near_land = 0
        border_total = 0
        for pi, pj in pixels:
            pi, pj = int(pi), int(pj)
            is_border = False
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ni = (pi + di) % len(spokes)
                    nj = pj + dj
                    if 0 <= nj < num_bins and not obj_mask[ni, nj]:
                        is_border = True
                        break
                if is_border:
                    break
            if is_border:
                border_total += 1
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni = (pi + di) % len(spokes)
                        nj = pj + dj
                        if 0 <= nj < num_bins and land_mask[ni, nj]:
                            border_near_land += 1
                            break
                    else:
                        continue
                    break
        if border_total > 0 and border_near_land / border_total > 0.4:
            return None

    if unique_spokes > 80:
        return None
    if n_pixels > 100 and unique_spokes > 30 and radial_bins < 6:
        return None

    sx = sy = tw = 0.0
    for pi, pj in pixels:
        v = echo_map[int(pi), int(pj)]
        brg = spokes[int(pi)]["angle_tick"] * 360.0 / TICKS_PER_REV
        rad = math.radians(brg)
        rm = (int(pj) + 0.5) * bin_size_m
        sx += rm * math.sin(rad) * v
        sy += rm * math.cos(rad) * v
        tw += v
    if tw == 0:
        return None

    cx, cy = sx / tw, sy / tw
    obj_range = math.sqrt(cx ** 2 + cy ** 2)
    obj_bearing = math.degrees(math.atan2(cx, cy)) % 360

    return {
        "labels": [lbl],
        "pixels": [(int(p[0]), int(p[1])) for p in pixels],
        "n_pixels": n_pixels,
        "unique_spokes": unique_spokes,
        "min_bin": min_bin,
        "max_bin": max_bin,
        "radial_bins": radial_bins,
        "peak_echo": int(peak_echo),
        "cx": cx,
        "cy": cy,
        "obj_range": obj_range,
        "obj_bearing": obj_bearing,
    }


def _merge_cluster_records(a, b, spokes, bin_size_m, echo_map):
    pixels = a["pixels"] + b["pixels"]
    sx = sy = tw = 0.0
    spoke_ids = set()
    min_bin = float("inf")
    max_bin = -1
    peak_echo = 0
    for pi, pj in pixels:
        v = int(echo_map[pi, pj])
        brg = spokes[pi]["angle_tick"] * 360.0 / TICKS_PER_REV
        rad = math.radians(brg)
        rm = (pj + 0.5) * bin_size_m
        sx += rm * math.sin(rad) * v
        sy += rm * math.cos(rad) * v
        tw += v
        spoke_ids.add(pi)
        min_bin = min(min_bin, pj)
        max_bin = max(max_bin, pj)
        peak_echo = max(peak_echo, v)

    cx, cy = (sx / tw, sy / tw) if tw > 0 else (0.0, 0.0)
    obj_range = math.sqrt(cx ** 2 + cy ** 2)
    obj_bearing = math.degrees(math.atan2(cx, cy)) % 360

    return {
        "labels": a["labels"] + b["labels"],
        "pixels": pixels,
        "n_pixels": len(pixels),
        "unique_spokes": len(spoke_ids),
        "min_bin": int(min_bin),
        "max_bin": int(max_bin),
        "radial_bins": int(max_bin - min_bin + 1),
        "peak_echo": int(peak_echo),
        "cx": cx,
        "cy": cy,
        "obj_range": obj_range,
        "obj_bearing": obj_bearing,
    }


def _merge_aligned_clusters(clusters, spokes=None, bin_size_m=None, echo_map=None,
                            bearing_tol_deg=2.0, range_overlap_m=10.0):
    """Merge clusters that are near-identical in bearing and overlapping in range."""
    if spokes is None or bin_size_m is None or echo_map is None:
        return clusters

    from heapq import heappop, heappush

    records = list(clusters)
    count = len(records)
    if count < 2:
        return records

    bearings = [record["obj_bearing"] for record in records]
    normalized_finite = all(
        math.isfinite(bearing) and 0.0 <= bearing <= 360.0
        for bearing in bearings
    )
    if bearing_tol_deg < 0 and normalized_finite:
        return records

    all_pairs = (
        not normalized_finite
        or not math.isfinite(bearing_tol_deg)
        or bearing_tol_deg >= 180.0
    )
    exact_bearing = bearing_tol_deg == 0.0 and normalized_finite
    use_buckets = (
        not all_pairs and not exact_bearing and bearing_tol_deg > 0.0
    )
    if use_buckets:
        try:
            bucket_ratio = 360.0 / float(bearing_tol_deg)
        except OverflowError:
            bucket_ratio = math.inf
        if not math.isfinite(bucket_ratio):
            all_pairs = True
            use_buckets = False

    if use_buckets:
        bucket_count = math.ceil(bucket_ratio)
        bucket_width = 360.0 / bucket_count
        neighbor_radius = math.ceil(bearing_tol_deg / bucket_width)
        bearing_buckets: dict[int, set[int]] = {}
        index_buckets = [0] * count
        for index, bearing in enumerate(bearings):
            bucket = math.floor((bearing % 360.0) / bucket_width) % bucket_count
            index_buckets[index] = bucket
            bearing_buckets.setdefault(bucket, set()).add(index)
    elif exact_bearing:
        bearing_buckets = {}
        index_buckets = [0] * count
        for index, bearing in enumerate(bearings):
            bucket = bearing % 360.0
            bearing_buckets.setdefault(bucket, set()).add(index)
            index_buckets[index] = bucket
        neighbor_radius = 0
        bucket_count = 0
    else:
        bearing_buckets = {}
        index_buckets = [0] * count
        neighbor_radius = 0
        bucket_count = 0

    alive = [True] * count
    versions = [0] * count
    candidates = []

    # Stable ids preserve the active-list order after pop(j), so lexicographic
    # heap pairs reproduce the legacy nested scan; versions stale touched pairs.

    def nearby_indices(index):
        if all_pairs:
            return range(count)
        bucket = index_buckets[index]
        if exact_bearing:
            return bearing_buckets.get(bucket, ())
        neighbors = set()
        for offset in range(-neighbor_radius, neighbor_radius + 1):
            neighbor_bucket = (bucket + offset) % bucket_count
            neighbors.update(bearing_buckets.get(neighbor_bucket, ()))
        return neighbors

    def enqueue_pair(first, second):
        if first == second:
            return
        i, j = (first, second) if first < second else (second, first)
        if not alive[i] or not alive[j]:
            return
        a = records[i]
        b = records[j]
        d_brg = abs(a["obj_bearing"] - b["obj_bearing"])
        d_brg = min(d_brg, 360.0 - d_brg)
        if d_brg > bearing_tol_deg:
            return
        a_min = a["min_bin"] * bin_size_m
        a_max = (a["max_bin"] + 1) * bin_size_m
        b_min = b["min_bin"] * bin_size_m
        b_max = (b["max_bin"] + 1) * bin_size_m
        range_gap = max(a_min, b_min) - min(a_max, b_max)
        if range_gap > range_overlap_m:
            return
        heappush(candidates, (i, j, versions[i], versions[j]))

    for i in range(count):
        for j in nearby_indices(i):
            if j > i:
                enqueue_pair(i, j)

    while candidates:
        i, j, version_i, version_j = heappop(candidates)
        if not alive[i] or not alive[j]:
            continue
        if versions[i] != version_i or versions[j] != version_j:
            continue

        a = records[i]
        b = records[j]

        if use_buckets or exact_bearing:
            old_i_bucket = index_buckets[i]
            bucket_members = bearing_buckets[old_i_bucket]
            bucket_members.remove(i)
            if not bucket_members:
                del bearing_buckets[old_i_bucket]
            bearing_buckets.setdefault(index_buckets[j], set()).discard(j)

        records[i] = _merge_cluster_records(a, b, spokes, bin_size_m, echo_map)
        alive[j] = False
        versions[i] += 1
        versions[j] += 1

        if use_buckets or exact_bearing:
            new_bearing = records[i]["obj_bearing"]
            if exact_bearing:
                new_bucket = new_bearing % 360.0
            else:
                new_bucket = math.floor(new_bearing / bucket_width) % bucket_count
            index_buckets[i] = new_bucket
            bearing_buckets.setdefault(new_bucket, set()).add(i)

        for other in nearby_indices(i):
            if alive[other]:
                enqueue_pair(i, other)

    return [record for index, record in enumerate(records) if alive[index]]


def _merge_same_bearing_major_pairs(records, spokes=None, bin_size_m=None, echo_map=None,
                                    bearing_bucket_deg=0.5, min_spokes=40,
                                    max_range_gap_m=80.0):
    """Collapse exactly-two major vessel fragments that survive the first merge pass.

    This targets broadside same-bearing splits like Pattern 13 without broadly
    merging crowded multi-fragment bearing stacks such as Pattern 15.
    """
    if spokes is None or bin_size_m is None or echo_map is None:
        return records

    major_types = {"medium_boat", "large_boat", "barge"}
    buckets = {}
    for idx, rec in enumerate(records):
        key = int(round(rec["obj_bearing"] / bearing_bucket_deg)) % int(round(360.0 / bearing_bucket_deg))
        buckets.setdefault(key, []).append(idx)

    merged_indices = set()
    merged_records = []
    for bucket_indices in buckets.values():
        if len(bucket_indices) != 2:
            continue
        a = records[bucket_indices[0]]
        b = records[bucket_indices[1]]
        if a["type"] not in major_types or b["type"] not in major_types:
            continue
        if a["unique_spokes"] < min_spokes or b["unique_spokes"] < min_spokes:
            continue

        d_brg = abs(a["obj_bearing"] - b["obj_bearing"])
        d_brg = min(d_brg, 360.0 - d_brg)
        if d_brg > bearing_bucket_deg:
            continue

        a_min = a["min_bin"] * bin_size_m
        a_max = (a["max_bin"] + 1) * bin_size_m
        b_min = b["min_bin"] * bin_size_m
        b_max = (b["max_bin"] + 1) * bin_size_m
        range_gap = max(a_min, b_min) - min(a_max, b_max)
        if range_gap < 0 or range_gap > max_range_gap_m:
            continue

        merged = _merge_cluster_records(a, b, spokes, bin_size_m, echo_map)
        geom_type, otype, rcs = _classify_object(
            merged["radial_bins"], merged["unique_spokes"], merged["peak_echo"], merged["obj_range"]
        )
        merged["geom_type"] = geom_type
        merged["type"] = otype
        merged["rcs_m2"] = float(rcs)
        merged_indices.update(bucket_indices)
        merged_records.append(merged)

    if not merged_records:
        return records

    survivors = [rec for idx, rec in enumerate(records) if idx not in merged_indices]
    survivors.extend(merged_records)
    survivors.sort(key=lambda rec: (rec["obj_bearing"], rec["obj_range"]))
    return survivors


# ============================================================
# Noise / Clutter / Decay Analysis
# ============================================================

def analyze_noise(echo_map, land_mask, obj_mask):
    """Measure noise characteristics in empty areas (excluding land, objects,
    and a 2-bin radial buffer around both)."""
    ns, nb = echo_map.shape

    # Build buffered exclusion mask: 2-bin radial buffer around land and objects
    exclusion = np.copy(land_mask | obj_mask)
    buffered = np.copy(exclusion)
    for i in range(ns):
        for j in range(nb):
            if exclusion[i, j]:
                for dj in range(-2, 3):
                    nj = j + dj
                    if 0 <= nj < nb:
                        buffered[i, nj] = True

    noise_count = 0
    total_empty = 0
    noise_vals = []
    for i in range(ns):
        for j in range(DEAD_ZONE_BINS, nb):
            if not buffered[i, j]:
                if echo_map[i, j] > 0:
                    noise_count += 1
                    noise_vals.append(int(echo_map[i, j]))
                total_empty += 1
    noise_prob = noise_count / max(total_empty, 1)
    dist = {}
    for v in noise_vals:
        dist[v] = dist.get(v, 0) + 1
    codes = sorted(dist.keys())
    weights = [dist[c] / max(len(noise_vals), 1) for c in codes]
    return {
        "noise_probability": round(noise_prob, 6),
        "noise_codes": codes if codes else [8, 20, 28],
        "noise_weights": [round(w, 4) for w in weights] if weights else [0.4, 0.35, 0.25],
        "total_noise_bins": noise_count,
        "total_empty_bins": total_empty,
    }


def analyze_decay(echo_map, land_mask, num_bins):
    """Measure leading/trailing edge decay profile of coastline returns."""
    ns = echo_map.shape[0]
    ramp_up_vals = []
    ramp_down_vals = []
    for i in range(ns):
        j = 0
        while j < num_bins:
            if land_mask[i, j]:
                k = j - 1
                while k >= DEAD_ZONE_BINS and echo_map[i, k] > 0 and not land_mask[i, k]:
                    ramp_up_vals.append(int(echo_map[i, k]))
                    k -= 1
                while j < num_bins and land_mask[i, j]:
                    j += 1
                while j < num_bins and echo_map[i, j] > 0 and not land_mask[i, j]:
                    ramp_down_vals.append(int(echo_map[i, j]))
                    j += 1
            else:
                j += 1
    return {
        "ramp_up_range": [int(min(ramp_up_vals)), int(max(ramp_up_vals))] if ramp_up_vals else [80, 200],
        "ramp_down_range": [int(min(ramp_down_vals)), int(max(ramp_down_vals))] if ramp_down_vals else [40, 160],
    }


# ============================================================
# Validation
# ============================================================

def validate(original_spokes, simulated_path, num_bins):
    """Compare original spoke data against a simulated CSV.
    Matches by spoke index (position), not tick value, to handle duplicate ticks.
    """
    sim = load_csv(simulated_path)
    sim_rotations = split_rotations(sim["spokes"])
    if not sim_rotations:
        return {"error": "simulated CSV has no data"}
    sim_spokes = sim_rotations[0]

    num_spokes = min(len(original_spokes), len(sim_spokes))

    total = match = nonzero_total = nonzero_match = zero_total = zero_match = 0
    tolerant_match = 0
    bin_errors = []

    for si in range(num_spokes):
        orig_e = original_spokes[si]["echoes"][:num_bins]
        sim_e = sim_spokes[si]["echoes"][:num_bins]

        for b in range(num_bins):
            ov = orig_e[b] if b < len(orig_e) else 0
            sv = sim_e[b] if b < len(sim_e) else 0
            total += 1
            if ov == sv:
                match += 1
                tolerant_match += 1
            else:
                bin_errors.append(abs(ov - sv))
                if ov in FURUNO_GAIN_CODES and sv in FURUNO_GAIN_CODES:
                    oi = FURUNO_GAIN_CODES.index(ov)
                    si_idx = FURUNO_GAIN_CODES.index(sv)
                    if abs(oi - si_idx) <= 1:
                        tolerant_match += 1
            if ov > 0:
                nonzero_total += 1
                if sv == ov:
                    nonzero_match += 1
            else:
                zero_total += 1
                if sv == 0:
                    zero_match += 1

    return {
        "total_bins": total,
        "exact_match": match,
        "exact_accuracy": round(100.0 * match / max(total, 1), 2),
        "tolerant_match": tolerant_match,
        "tolerant_accuracy": round(100.0 * tolerant_match / max(total, 1), 2),
        "nonzero_total": nonzero_total,
        "nonzero_exact": nonzero_match,
        "nonzero_accuracy": round(100.0 * nonzero_match / max(nonzero_total, 1), 2),
        "zero_total": zero_total,
        "zero_exact": zero_match,
        "zero_accuracy": round(100.0 * zero_match / max(zero_total, 1), 2),
        "mean_error": round(float(np.mean(bin_errors)), 2) if bin_errors else 0.0,
    }


# ============================================================
# Reusable extraction function
# ============================================================

def extract_scene_from_csv(filepath, rotation=0, land_threshold=104,
                           min_land_run=6, verbose=True, range_profile=None,
                           radial_mapping=None):
    """Extract a complete radar scene from a Furuno CSV capture.

    Returns a scene dict with all data needed for simulation:
    coastlines, objects, profiles, stats, noise, angle_ticks, etc.

    This is the same extraction used by the CLI, exposed as a callable function
    so radar_simulator.py can import and use it for Mode 5 (custom simulation).
    """
    if verbose:
        print(f"Loading: {filepath}")
    if range_profile is None or radial_mapping != "linear_full_range":
        raise ValueError(
            "scene extraction creates metric geometry; pass an explicit "
            "range_profile and radial_mapping='linear_full_range'"
        )
    data = load_csv(filepath, range_profile, radial_mapping)
    nb = data["num_bins"]
    bsm = data["bin_size_m"]
    if verbose:
        print(f"  {len(data['spokes'])} spokes, {nb} bins, "
              f"bin_size={bsm:.4f} m, gain={data['gain']}")

    rotations = split_rotations(data["spokes"])
    if verbose:
        print(f"  {len(rotations)} rotation(s)")
    if rotation >= len(rotations):
        raise ValueError(f"Rotation {rotation} not found (have {len(rotations)})")

    rot = rotations[rotation]
    ticks = [s["angle_tick"] for s in rot]
    if verbose:
        print(f"  Using rotation {rotation}: {len(rot)} spokes, "
              f"ticks {ticks[0]}-{ticks[-1]}")

    # --- Coastlines ---
    if verbose:
        print("\nDetecting coastlines...")
    land_mask, coastlines, echo_map, coastline_echo_data, coastline_labels = detect_coastlines(
        rot, nb, bsm, land_threshold, min_land_run)
    if verbose:
        print(f"  {len(coastlines)} coastline region(s), "
              f"{int(np.sum(land_mask)):,} land bins")
        for c in coastlines:
            print(f"    {c['name']}: {c['num_spokes']} spokes, "
                  f"{c['size_bins']:,} bins, {len(c['points'])} vertices")

    # --- Coastline profiles & stats ---
    if verbose:
        print("\nBuilding coastline profiles & statistics...")
    coastline_profiles = build_coastline_profiles(rot, land_mask, echo_map, nb)
    if verbose:
        print(f"  Coastline profiles: {len(coastline_profiles)} spokes")

    coastline_stats = build_coastline_stats(
        land_mask, echo_map, coastline_labels, rot, nb, coastlines)
    if verbose and coastline_stats:
        sample_name = list(coastline_stats.keys())[0]
        s = coastline_stats[sample_name]
        print(f"  Stats (sample {sample_name}):")
        print(f"    mean_core_echo: {s['mean_core_echo']}")
        print(f"    median_ramp_up: {s['median_ramp_up']}")
        print(f"    median_ramp_down: {s['median_ramp_down']}")
        print(f"    typical_ramp_up_bins: {s['typical_ramp_up_bins']}")
        print(f"    typical_ramp_down_bins: {s['typical_ramp_down_bins']}")
        print(f"    median_segment_depth_bins: {s['median_segment_depth_bins']}")
        print(f"    p75_segment_depth_bins: {s['p75_segment_depth_bins']}")
        print(f"    segment_count: {s['segment_count']}")

    # --- Objects ---
    if verbose:
        print("\nDetecting objects...")
    objects, obj_mask, object_echo_data = detect_objects(
        rot, nb, bsm, land_mask, echo_map)
    if verbose:
        print(f"  {len(objects)} object(s)")
        for o in objects:
            print(f"    {o['type']}: range={o['range_m']:.1f}m "
                  f"brg={o['bearing_deg']:.1f} peak={o['peak_echo']} "
                  f"rcs={o['rcs_m2']:.1f}")

    # --- Remaining non-zero bins (noise_echo_data) ---
    if verbose:
        print("\nExtracting remaining non-zero bins (noise_echo_data)...")
    noise_echo_data = {}
    noise_bin_count = 0
    for i in range(len(rot)):
        segments = []
        j = DEAD_ZONE_BINS
        while j < nb:
            if echo_map[i, j] > 0 and not land_mask[i, j] and not obj_mask[i, j]:
                seg_start = j
                seg_vals = []
                while j < nb and echo_map[i, j] > 0 and not land_mask[i, j] and not obj_mask[i, j]:
                    seg_vals.append(int(echo_map[i, j]))
                    j += 1
                segments.append([seg_start, seg_vals])
                noise_bin_count += len(seg_vals)
            else:
                j += 1
        if segments:
            noise_echo_data[str(i)] = segments
    if verbose:
        print(f"  {len(noise_echo_data)} spokes with residual data, "
              f"{noise_bin_count:,} bins captured")

    # Extract dead zone non-zero bins (bins 0 to DEAD_ZONE_BINS-1)
    dead_zone_echo_data = {}
    dz_bin_count = 0
    for i in range(len(rot)):
        segments = []
        j = 0
        while j < DEAD_ZONE_BINS:
            if echo_map[i, j] > 0:
                seg_start = j
                seg_vals = []
                while j < DEAD_ZONE_BINS and echo_map[i, j] > 0:
                    seg_vals.append(int(echo_map[i, j]))
                    j += 1
                segments.append([seg_start, seg_vals])
                dz_bin_count += len(seg_vals)
            else:
                j += 1
        if segments:
            dead_zone_echo_data[str(i)] = segments
    if verbose and dz_bin_count > 0:
        print(f"  Dead zone non-zero bins: {len(dead_zone_echo_data)} spokes, "
              f"{dz_bin_count:,} bins")

    # --- Noise (excluding land, objects, and buffer) ---
    if verbose:
        print("\nAnalyzing clutter/noise...")
    noise = analyze_noise(echo_map, land_mask, obj_mask)
    noise_field_stats = build_noise_field_stats(rot, echo_map, land_mask, obj_mask, nb)
    noise["noise_field_stats"] = noise_field_stats
    if verbose:
        print(f"  Noise probability: {noise['noise_probability']:.6f}")
        print(f"  Noise codes: {noise['noise_codes'][:10]}")
        print(f"  Structured noise field: "
              f"{noise_field_stats['bearing_sectors']} sectors x "
              f"{noise_field_stats['range_bands']} bands")

    # --- Decay ---
    if verbose:
        print("\nAnalyzing decay profile...")
    decay = analyze_decay(echo_map, land_mask, nb)
    if verbose:
        print(f"  Ramp-up range: {decay['ramp_up_range']}")
        print(f"  Ramp-down range: {decay['ramp_down_range']}")

    # --- Build scene dict ---
    scene = {
        "source_file": os.path.basename(filepath),
        "source_file_path": os.path.abspath(filepath),
        "extracted": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "radar_config": {
            "status": data["status"], "scale": data["scale"],
            "range_index": data["range_index"], "gain": data["gain"],
            "range_nm": round(data["range_max_m"] / 1852.0, 6),
            "range_m": round(data["range_max_m"], 1),
            "bins": nb, "bin_size_m": round(bsm, 4),
            "range_profile": data["range_profile"],
            "radial_mapping": data["radial_mapping"],
            "csv_range_contract_verified": False,
            "ticks_per_rev": TICKS_PER_REV,
            "spokes_in_rotation": len(rot),
        },
        "clutter": noise,
        "decay": decay,
        "coastline_stats": coastline_stats,
        "coastline_profiles": coastline_profiles,
        "angle_ticks": ticks,
        "coastline_echo_data": coastline_echo_data,
        "object_echo_data": object_echo_data,
        "noise_echo_data": noise_echo_data,
        "dead_zone_echo_data": dead_zone_echo_data,
        "coastlines": [
            {"name": c["name"],
             "points": [[round(x, 3), round(y, 3)] for x, y in c["points"]],
             "parent_label": int(c.get("parent_label", 0) or 0)}
            for c in coastlines
        ],
        "objects": [
            {"type": o["type"], "geom_type": o.get("geom_type", o["type"]),
             "x": o["x"], "y": o["y"],
             "range_m": o.get("range_m"), "bearing_deg": o.get("bearing_deg"),
             "rcs_m2": o["rcs_m2"], "length_m": o["length_m"],
             "width_m": o["width_m"], "heading_deg": o["heading_deg"],
             "speed_kts": o["speed_kts"], "course_deg": o["course_deg"],
             "peak_echo": o.get("peak_echo"), "num_pixels": o.get("num_pixels"),
             "radial_bins": o.get("radial_bins"),
             "angular_spokes": o.get("angular_spokes")}
            for o in objects
        ],
    }

    if verbose:
        print(f"\n  Extracted: {len(coastlines)} coastlines, "
              f"{len(objects)} objects, {len(ticks)} angle ticks")

    return scene


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract radar scene from Furuno CSV capture")
    parser.add_argument("input", help="Input Furuno CSV file path")
    parser.add_argument("-o", "--output",
                        help="Output scene JSON (default: <input>_scene.json)")
    parser.add_argument("--rotation", type=int, default=0,
                        help="Which rotation to extract (0=first)")
    parser.add_argument("--land-threshold", type=int, default=104,
                        help="Min echo for land detection (default 104)")
    parser.add_argument("--min-land-run", type=int, default=6,
                        help="Min contiguous land bins (default 6)")
    parser.add_argument(
        "--range-profile",
        choices=["observed-sdk-range-index"],
        help="Explicit raw Range-code profile; audited CSV exporter provenance is unverified",
    )
    parser.add_argument(
        "--assume-linear-full-range",
        action="store_true",
        help="Assume echo bins span linearly from zero to the selected display range",
    )
    parser.add_argument("--validate", action="store_true",
                        help="Auto-simulate from extracted scene and validate accuracy")
    parser.add_argument("--sim-output",
                        help="Output dir for simulated CSV (default: outputs/validation/)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.output:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   base + "_scene.json")

    # --- Extract ---
    scene = extract_scene_from_csv(
        args.input, rotation=args.rotation,
        land_threshold=args.land_threshold,
        min_land_run=args.min_land_run, verbose=args.verbose,
        range_profile=args.range_profile,
        radial_mapping=("linear_full_range"
                        if args.assume_linear_full_range else None))

    # --- Write scene JSON ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"\nScene written: {args.output}")
    print(f"  {len(scene['coastlines'])} coastlines, "
          f"{len(scene['objects'])} objects, "
          f"{len(scene['angle_ticks'])} angle ticks")
    print(f"  coastline_echo_data: {len(scene.get('coastline_echo_data', {}))} spokes")
    print(f"  object_echo_data: {len(scene.get('object_echo_data', {}))} spokes")
    print(f"  noise_echo_data: {len(scene.get('noise_echo_data', {}))} spokes")
    print(f"  coastline_profiles: {len(scene.get('coastline_profiles', {}))} spokes")
    print(f"  coastline_stats: {len(scene.get('coastline_stats', {}))} regions")

    # --- Auto-validate ---
    if args.validate:
        print(f"\n{'='*60}")
        print("VALIDATION: simulating from extracted scene...")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        try:
            import radar_simulator as rsim
        except ImportError:
            print("ERROR: radar_simulator.py not found in same directory")
            sys.exit(1)

        sim_dir = args.sim_output
        if not sim_dir:
            sim_dir = os.path.join(script_dir, "outputs", "validation")
        print(f"  Output: {sim_dir}")

        rsim.simulate_from_scene_file(
            args.output, sim_dir, num_rotations=1, verbose=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
