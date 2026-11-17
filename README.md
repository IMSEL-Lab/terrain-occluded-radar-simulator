# Terrain-Occluded Radar Simulator

The Terrain-Occluded Radar Simulator (TORS) generates maritime radar displays from synthetic scenes or geographic shoreline geometry. Capture-conditioned reconstruction extracts return support from a measured sweep and generates display intensities using the Furuno or Canadian Aquatic Navigation for Observation of the Environment (CANOE) configuration.

The software accompanies *Terrain-Occluded Radar Simulation for Maritime Autonomous Navigation* by Noah Young, J.C. Vaught, and Yi Wang. Evaluation inputs, fitted profiles, numerical results, figure data, and reproduction receipts are in the [Zenodo evaluation archive](https://doi.org/10.5281/zenodo.24316336). The software release is maintained at [IMSEL-Lab/terrain-occluded-radar-simulator](https://github.com/IMSEL-Lab/terrain-occluded-radar-simulator).

## Install and generate a sweep

Install Python 3.10 or later and uv, then install the locked environment from the repository root. The supplied examples contain synthetic geometry and run without a dataset download.

```sh
uv sync --locked
uv run tors synthetic --output /tmp/tors-synthetic
uv run tors map --geojson examples/shoreline.geojson --lat 0 --lon 0 \
  --output /tmp/tors-map
```

Each command creates a new output directory containing a scene, radar CSV, detection sidecar, metadata, and a run manifest. Existing output directories are refused. Use `uv run tors --help` to list the workflows and `uv run tors map --help` to inspect a workflow's options. The package can also be installed with `pip install .` and used as `tors`.

## Scene-driven generation

Manual scene generation reads explicitly arranged coastlines and vessels from a scene JSON. The supplied example runs with either the ray--polygon shoreline renderer or the filled-polygon renderer.

```sh
uv run tors scene --input examples/manual-scene.json \
  --coastline-renderer filled-polygon --output /tmp/tors-manual-scene
```

Synthetic generation transforms an explicit coastline template, places vessels, and renders a sweep. The template is JSON with a `coastlines` array. Each coastline supplies `name` and Cartesian `points` in meters, with x right and y forward. An optional `coastline_stats` mapping supplies aggregate echo statistics. Template discovery and coastline-library construction are restricted to the supplied input. The default seed is 42, with two vessels in each of four classes.

```sh
uv run tors synthetic --template examples/synthetic-template.json \
  --seed 42 --vessels-per-class 2 --output /tmp/tors-custom-scene
```

Map generation accepts GeoJSON longitude/latitude coordinates and the radar's latitude and longitude. Line features become shoreline strips through the native 3 m buffer. Polygon features supply land polygons. The default scan contains 400 spokes and 300 bins spanning 150 m. Supply `--spokes`, `--bins`, and `--range-m` to change this grid. A scene JSON passed with `--grid-scene` supplies the exact radar configuration and bearing ticks for an archived scan. Coordinates in the archive's `native_renderer_line_input.geojson` are local meter offsets encoded as degrees around the origin, so its command uses `--lat 0 --lon 0`.

```sh
uv run tors map --geojson "$EVIDENCE/data/map-to-radar/native_renderer_line_input.geojson" \
  --lat 0 --lon 0 \
  --grid-scene "$EVIDENCE/data/map-to-radar/native_renderer_scene.json" \
  --time 2025-08-28T14:52:34.126 --output /tmp/tors-paper-map
```

The mapped-shoreline evaluation uses the corrected full radar-sensor orientation and explicit rectangular shoreline strips. Set `EVIDENCE` to the absolute path of the extracted companion archive. Its download script retrieves the hash-verified CANOE scans and navigation files into a dataset directory.

Location-file generation reads a `.radarloc` map and exposes the location-specific occlusion, attenuation, clutter, receiver-noise, and manually placed vessel controls. Its bundled Python runtime includes the cumulative occlusion and entry-depth attenuation equations used in the paper. The supplied location example is synthetic and needs no map download.

```sh
uv run tors location --radarloc examples/synthetic-location.radarloc \
  --spokes 32 --object-profile none --output-dir /tmp/tors-location
```

Use `uv run tors location --help` for full controls. Evaluation of the mapped-shoreline configuration uses the frozen CANOE inputs.

```sh
python3 "$EVIDENCE/scripts/download_canoe.py" /tmp/tors-canoe-dataset
uv run tors evaluate-map --data-root /tmp/tors-canoe-dataset \
  --source-root "$EVIDENCE/data/canoe-forward-driver" \
  --manifests "$EVIDENCE/manifests" --output-dir /tmp/tors-map-evaluation --score
```

The bundled evaluation configuration freezes 30 calibration scans, 60 validation scans, 60 test scans, map alignment, display normalization, and scoring. `--smoke-train` renders one calibration scan. The source map and route records are checked against their frozen hashes. `--manifests` must contain `canoe_selected_scans.json` and `canoe_source_manifest.json`. The 100-configuration geometry audit uses the bundled synthetic primitive.

```sh
uv run tors audit-geometry --output /tmp/tors-geometry-audit
```

## Capture-conditioned reconstruction

The Furuno workflow reads the selected rotation from each measured CSV, preserves the observed bearing sequence, extracts return support, and produces a reconstructed sweep. Its comparison also includes an all-zero return and source-profile replay. The bundled configuration identifies the three scored Folly Beach capture filenames and hashes. Place permitted source files under the relative directories specified in that configuration.

```sh
uv run tors furuno --source-root /absolute/path/to/furuno-captures \
  --output-dir /tmp/tors-furuno-smoke --smoke-only
uv run tors furuno --source-root /absolute/path/to/furuno-captures \
  --output-dir /tmp/tors-furuno-evaluation
```

Furuno scoring uses observed display codes divided by 252, preserves missing-bearing masks, and rejects missing rendered source bearings. Its radial mapping is the explicitly recorded linear full-range assumption. The raw capture files require access from the data holder.

The CANOE workflow smooths the observed grid, applies the sensor-specific run/component thresholds, and samples fresh intensities from a fitted range-dependent profile. The input directory supplies `canoe_display_norm_0_1_over_255.csv`, `common_valid_mask.csv`, `spokes.csv`, and `source_canoe_radar_png.png`. The profile records its training scans and extraction thresholds. Outputs include seeds 0, 1, and 2 and a control that shifts support by a quarter revolution.

```sh
uv run tors canoe --input "$EVIDENCE/data/canoe-conditioned" \
  --profile "$EVIDENCE/data/canoe-conditioned/sensor_profile.json" \
  --output /tmp/tors-canoe-reconstruction
```

Refit the CANOE profile using the frozen 30 training scans, then verify the detector masks and source hashes.

```sh
uv run tors calibrate-canoe --manifests "$EVIDENCE/manifests" \
  --dataset /tmp/tors-canoe-dataset --output-dir /tmp/tors-canoe-fit
uv run tors verify-canoe-profile --dataset /tmp/tors-canoe-dataset \
  --output-dir /tmp/tors-canoe-fit
```

## Formats and validation

The Furuno-style output CSV uses the header `Status,Scale,Range,Gain,Angle,EchoValues` followed by one row per spoke. Each row contains the five metadata fields and the radial display codes. Bearings use 8,192 ticks per revolution. Furuno outputs use integer codes from 0 through 252. CANOE reconstruction exports measured microsecond timestamps, encoder counts, renderer ticks, and newly generated 8-bit display codes from 0 through 255. The companion archive retains the normalization and masks used for each comparison.

Run the implementation checks from the repository root.

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check .
uv run pytest -q
```

The tests cover capture parsing, angular and radial masks, full sensor-frame coordinates, shoreline construction, scalar fitting, deterministic sampling, detector thresholds, and geometry accounting. The numerical renderer and extractor retain their evaluated formatting. Ruff and type checking cover the package, runners, and tests. The companion reproduction receipts record output-level comparisons for the paper inputs.

## Authorship and source terms

Noah Young developed the scene renderer and extractor. J.C. Vaught developed the reconstruction adaptation, evaluation, reproducibility tooling, and consolidated package. Yi Wang supervised the research. The software is released under the [BSD Zero Clause License (0BSD)](LICENSE). Data licensing is recorded by the companion archive and remains attached to each source dataset. See [NOTICE.md](NOTICE.md).
