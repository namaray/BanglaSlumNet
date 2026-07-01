# BanglaSlumNet Project Recovery Plan

Last updated: 2026-07-01

This document explains what must be done to make BanglaSlumNet work as a defensible research project. It answers the practical questions:

- Do we need manual annotation?
- What exactly should we annotate?
- Where should the imagery and reference data come from?
- Where should files be stored?
- How should the codebase change?
- What experiment sequence should we run next?
- What result is good enough to claim?

Short answer: yes, we need manual annotation. The current weak labels are enough for pipeline development, but they are not enough for strong pixel-level slum segmentation claims.

---

## 1. Current Situation

BanglaSlumNet currently has:

- a working Colab pipeline;
- Sentinel-2 exports and tiling;
- 12 Dhaka study regions;
- region-type weak labels;
- LocateAnything/MoonViT feature caching;
- SAS-Net clean-tile caching;
- socioeconomic fusion architecture;
- corrected collapse diagnostics;
- report and presentation artifacts.

The main scientific problem is supervision quality.

The current labels say:

```text
built pixel inside known informal region -> slum
built pixel inside known formal control region -> formal
non-built pixel -> ignored
```

This fixed the zero-slum bug, but it is still coarse. It assumes that every built pixel inside an informal-region box is slum and every built pixel inside a formal-control box is formal. That is not true at pixel level.

Because of this, the model can learn region identity and class prevalence instead of real informal-settlement structure.

---

## 2. What Must Change

To make the project work, we need four upgrades:

| Priority | Upgrade | Why it matters |
|---:|---|---|
| 1 | Manual evaluation labels | We need a trustworthy test set before claiming performance |
| 2 | Manual or semi-manual training labels | Region-level labels are too noisy for segmentation |
| 3 | Better task framing | Tile classification may fit the current data better than pixel segmentation |
| 4 | Better context layers/high-res grounding | Roads, poverty, and fine settlement cues need stronger sources |

The minimum viable recovery is:

```text
manual evaluation subset + tile-level baseline + corrected Exp2 rerun
```

The stronger paper version is:

```text
manual train/val/test polygons + tile-level and segmentation baselines + high-res grounding + verified socioeconomic layers
```

---

## 3. Do We Have To Manually Annotate Data?

Yes.

We do not necessarily need to manually annotate all 720 tiles immediately. But we need at least a manually verified evaluation subset. Without that, we cannot know whether the model is detecting slums or just reproducing weak region labels.

### 3.1 Minimum Annotation Requirement

Create a manual test set:

| Item | Recommended amount |
|---|---:|
| Regions | all 12 current regions |
| Tiles per region | 8-12 |
| Total tiles | 96-144 |
| Classes | slum, formal dense, other/unknown |
| Use | final evaluation only |

This is the minimum for a defensible diagnostic paper.

### 3.2 Better Annotation Requirement

Create manual train/val/test labels:

| Split | Recommended amount | Use |
|---|---:|---|
| Train | 250-350 tiles | model learning |
| Val | 60-100 tiles | tuning/early stopping |
| Test | 100-150 tiles | final reporting |

If time is limited, prioritize test labels first. A trustworthy evaluation set is more valuable than more weak training labels.

### 3.3 Best Version

Create polygon annotations over full region boxes:

- annotate true informal-settlement boundaries;
- annotate formal dense built-up areas;
- mark uncertain areas as unknown;
- rasterize polygons to Sentinel-2 tile grids.

This lets the project become a real dataset contribution.

---

## 4. What Exactly Should Be Annotated?

We should annotate settlement areas, not individual roofs.

At 10 m Sentinel-2 resolution, individual roofs and lanes are often below the reliable visual scale. The right annotation target is a coarse polygon or mask representing settlement type.

### 4.1 Classes

Use this class scheme:

| Class ID | Name | Meaning | Used in loss? |
|---:|---|---|---|
| 0 | unknown_ignore | Water, vegetation, unclear areas, mixed/untrusted areas | No |
| 1 | informal_slum | Known or visually verified informal settlement area | Yes |
| 2 | formal_dense | Dense formal residential/commercial built area | Yes |
| 3 | other_built | Industrial, airport, large institutions, sparse formal, ambiguous built-up | Optional/no |

For the current binary segmentation model:

```text
positive = class 1
negative = class 2
ignore = class 0 and usually class 3
```

Class 3 is useful for annotation honesty. It prevents forcing uncertain built-up places into slum/formal classes.

### 4.2 Annotation Geometry

Preferred geometry:

- polygon boundaries in GeoJSON or GeoPackage;
- coordinate reference system: EPSG:4326 for storage;
- rasterized to tile CRS/grid during preprocessing.

Avoid:

- freehand pixel painting on screenshots without georeferencing;
- labels that cannot be traced back to map coordinates;
- mixing visual labels and model predictions without marking their source.

### 4.3 What Counts As Informal/Slum?

Annotate as informal/slum when most of the area shows:

- dense small structures;
- irregular layout;
- narrow internal lanes or no clear road grid;
- organic growth pattern;
- known slum/informal-settlement name from external references;
- proximity to water edges, rail/road embankments, industrial edges, or public land when locally relevant.

Do not label only from poverty assumptions. The annotation must be based on a combination of visible morphology and external settlement knowledge.

### 4.4 What Counts As Formal Dense?

Annotate as formal dense when most of the area shows:

- dense built-up texture;
- planned or semi-planned road grid;
- larger building footprints;
- institutional/commercial/residential formal blocks;
- known formal areas such as Dhanmondi, Uttara, Gulshan-Baridhara, or Old Dhaka.

Old Dhaka is especially important: it is dense and irregular, but not the same target class as slum. It is the stress test.

### 4.5 What Should Be Unknown?

Mark as unknown/ignore:

- water;
- parks/vegetation;
- construction sites;
- industrial-only compounds;
- airports/rail yards;
- areas where annotators disagree;
- mixed boundaries where a coarse polygon would be dishonest;
- places where imagery date and reference date clearly conflict.

Unknown is not a failure. Unknown prevents bad labels from poisoning training.

---

## 5. Where Should Annotation Data Come From?

Use a source hierarchy. Do not rely on one source.

### 5.1 Primary Visual Sources

| Source | Use | Notes |
|---|---|---|
| Sentinel-2 exported tiles | Model-aligned reference | Low resolution, but directly matches model input |
| Google Earth / Google Maps / Bing / Esri World Imagery | Visual interpretation reference | High resolution, useful for annotation, but check licensing before redistributing imagery |
| OpenStreetMap | Roads, waterways, landmarks, named places | Good for context, incomplete in informal areas |

High-resolution web imagery is very useful for drawing polygons, but we should not redistribute downloaded high-res imagery unless the license clearly allows it. For a publishable dataset, the safest release is:

```text
manual polygons + scripts + open Sentinel-2/GHSL/Dynamic World layers
```

not:

```text
bulk Google/Esri imagery chips
```

### 5.2 Existing Informal-Settlement Boundary Sources

Use these as starting references, not as unquestioned ground truth:

| Source | What it gives | Link |
|---|---|---|
| World Bank Data Catalog / ESA EO4SD-Urban Dhaka informal settlements | Probable informal settlement locations for Dhaka, including 2006/2010 sources and 2017 VHR interpretation | https://datacatalog.worldbank.org/search/dataset/0041703/dhaka-bangladesh-informal-settlements-esa-eo4sd-urban |
| EnergyData mirror of ESA EO4SD-Urban Dhaka informal settlements | Same/similar downloadable dataset listing, updated catalog page | https://energydata.info/dataset/dhaka-bangladesh-informal-settlements-esa-eo4sd-urban |
| HOT / OSM Dhaka informal-settlement mapping context | POIs and mapping activity around Dhaka slums | https://www.hotosm.org/en/news/mapping-dhakas-informal-settlements-for-climate-resilience-and-urban-development/ |
| World Bank Urban Informal Settlements Survey 2016 | Household survey/context, not clean pixel polygons | https://microdata.worldbank.org/index.php/catalog/2864 |
| Bangladesh slum census/mapping literature | Historical slum definitions and census context | https://pmc.ncbi.nlm.nih.gov/articles/PMC2701942/ |

Recommended use:

1. Load ESA EO4SD polygons into QGIS.
2. Overlay current high-resolution imagery and OSM.
3. Correct boundaries manually for the target year/imagery.
4. Add formal-control polygons manually.
5. Store corrected polygons as our own annotation layer with source notes.

### 5.3 Model Data Sources Already Used

These remain useful for features and weak labels:

| Source | Use |
|---|---|
| Sentinel-2 L2A | model imagery |
| Dynamic World | built-up evidence |
| GHSL built-up | built-up evidence/context |
| VIIRS nighttime light | economic/electricity proxy |
| WorldPop / GHS-POP | population proxy |
| OSM roads | road/access context |

---

## 6. Annotation Tool Recommendation

Use QGIS as the main annotation tool.

Why QGIS:

- handles georeferenced rasters and polygons correctly;
- exports GeoJSON/GeoPackage;
- can load Sentinel-2 tiles, OSM, ESA polygons, and high-res XYZ basemaps;
- avoids the georeferencing problems of screenshot-based annotation.

Alternative tools:

| Tool | When to use |
|---|---|
| QGIS | Best default for geospatial polygon labels |
| CVAT | Good for image-chip annotation, but geospatial handling needs extra care |
| Label Studio | Good if using image chips and simple masks |
| geojson.io | Quick polygon editing only, not ideal for full workflow |
| Google Earth Pro | Useful for visual inspection, not best as final label store |

Recommendation:

```text
QGIS for final geospatial labels.
CVAT/Label Studio only if we build a chip export/import conversion script.
```

---

## 7. Storage Plan

Large data should stay out of Git. Store it on Google Drive and optionally local disk. Only small metadata, schemas, and scripts should go into the repo.

### 7.1 Google Drive Storage

Use this structure:

```text
/gdrive/MyDrive/BanglaSlumNet/
├── data/
│   ├── raw/
│   │   ├── s2/
│   │   ├── socioeconomic/
│   │   ├── external_boundaries/
│   │   │   ├── esa_eo4sd_urban/
│   │   │   ├── osm/
│   │   │   └── survey_reference/
│   │   └── highres_reference/
│   │       └── README_LICENSES.md
│   ├── annotations/
│   │   ├── manual_v1/
│   │   │   ├── polygons/
│   │   │   │   ├── banglaslumnet_manual_v1.gpkg
│   │   │   │   └── banglaslumnet_manual_v1.geojson
│   │   │   ├── rasterized/
│   │   │   │   ├── labels/
│   │   │   │   └── masks/
│   │   │   ├── qa/
│   │   │   │   ├── disagreements.geojson
│   │   │   │   └── audit_notes.csv
│   │   │   └── README.md
│   │   └── manual_v2/
│   ├── tiles/
│   ├── labels/
│   ├── socioeconomic/
│   └── features_cache/
├── results/
└── model_cache/
```

### 7.2 Repo Storage

Store these in Git:

```text
docs/ANNOTATION_GUIDELINES.md
docs/PROJECT_RECOVERY_PLAN.md
config/annotation_schema.yaml
scripts/rasterize_manual_annotations.py
scripts/audit_manual_annotations.py
```

Do not store large GeoTIFFs or high-res imagery in Git.

### 7.3 Local Storage

For local work on Windows:

```text
D:\papers\BanglaSlumPaper\BanglaSlumNet\data\
```

This may contain copied data for inspection, but Drive should be the canonical data store.

---

## 8. File Naming Rules

### 8.1 Polygon Annotation Files

Use:

```text
banglaslumnet_manual_v1.gpkg
banglaslumnet_manual_v1.geojson
```

Layer name:

```text
settlement_polygons
```

### 8.2 Rasterized Label Files

Use existing tile IDs:

```text
{tile_id}_manual_label.tif
{tile_id}_manual_mask.tif
```

Example:

```text
korail_2021_dry_000_002_manual_label.tif
korail_2021_dry_000_002_manual_mask.tif
```

### 8.3 Annotation Metadata

Use:

```text
manual_v1_manifest.csv
manual_v1_audit.csv
manual_v1_split.json
```

---

## 9. Polygon Attribute Schema

Every annotation polygon should have these fields:

| Field | Type | Example | Required |
|---|---|---|---|
| `poly_id` | string | `korail_p001` | yes |
| `region_id` | string | `korail` | yes |
| `class_id` | integer | `1` | yes |
| `class_name` | string | `informal_slum` | yes |
| `confidence` | integer | `3` | yes |
| `source_primary` | string | `manual_highres` | yes |
| `source_secondary` | string | `esa_eo4sd;osm` | no |
| `imagery_date` | string | `2021-dry` or `unknown` | yes |
| `annotator` | string | `nafiz` | yes |
| `reviewer` | string | `zayan` | no |
| `review_status` | string | `pending/approved/disputed` | yes |
| `notes` | string | free text | no |

Confidence scale:

| Value | Meaning |
|---:|---|
| 1 | uncertain, do not use for training |
| 2 | likely, use only for weak training |
| 3 | confident, use for train/val/test |

Only confidence 3 should be used for final test metrics.

---

## 10. Annotation Workflow

### Step 1 - Prepare Reference Layers

In QGIS, load:

1. current region boxes from `config/regions_dhaka.yaml`;
2. Sentinel-2 tile grid or region composites;
3. ESA EO4SD Dhaka informal-settlement polygons;
4. OSM roads/water/buildings if available;
5. high-resolution XYZ basemap for visual interpretation.

### Step 2 - Select Tiles

Select tiles deliberately. Do not randomly annotate only easy cases.

For each region, choose:

- 3-4 obvious target tiles;
- 3-4 boundary/mixed tiles;
- 2-4 hard-negative or confusing tiles.

For formal controls, choose dense areas that look visually similar to slums.

### Step 3 - Draw Polygons

Draw broad settlement polygons:

- informal/slum;
- formal dense;
- unknown/ignore.

Do not draw tiny roof-level polygons. The model resolution does not justify that.

### Step 4 - Add Metadata

For every polygon:

- fill class;
- fill confidence;
- fill annotator;
- fill source notes;
- mark disputed areas.

### Step 5 - Review

At least two team members should review each test tile.

Review rules:

- if both agree, mark `approved`;
- if disagreement is small, adjust polygon and mark approved;
- if disagreement is large, mark unknown or disputed;
- do not force a label just to increase sample count.

### Step 6 - Rasterize

Rasterize polygons to match tile grids exactly:

- same CRS;
- same transform;
- same width/height;
- same tile ID;
- class IDs preserved.

### Step 7 - Audit

Run an audit script before training:

- count pixels per class;
- count tiles per region;
- check no empty masks;
- check no missing labels;
- check train/val/test separation;
- check formal-control pixels exist in test;
- render quick overlays for human inspection.

---

## 11. Quality-Control Rules

### 11.1 Annotator Agreement

For the final test set:

- every tile should be reviewed by at least two people;
- disagreements should be resolved before use;
- unresolved areas should become unknown/ignore.

### 11.2 Leakage Prevention

Do not let the same location appear in both train and test across years.

If a tile from Korail 2021 is in test, the same coordinate area from Korail 2020/2022/2023 should not be in train.

Use location-grouped splits:

```text
group_id = region_id + tile_row + tile_col
```

Split by `group_id`, not only by file.

### 11.3 Class Balance

The manual test set should include:

- informal pixels;
- formal dense pixels;
- mixed/unknown pixels;
- hard formal controls.

Avoid a test set where one class dominates completely.

### 11.4 Date Mismatch

If high-res reference imagery is from a very different year than Sentinel-2:

- mark `imagery_date`;
- lower confidence if the settlement changed;
- use recent Sentinel-2/Google Earth timeline if possible.

---

## 12. Code Changes Needed

### 12.1 Add Annotation Config

Create:

```text
config/annotation_schema.yaml
```

It should define:

```yaml
classes:
  0: unknown_ignore
  1: informal_slum
  2: formal_dense
  3: other_built

use_for_training: [1, 2]
ignore_classes: [0, 3]
min_confidence_train: 2
min_confidence_test: 3
```

### 12.2 Add Rasterization Script

Create:

```text
scripts/rasterize_manual_annotations.py
```

Inputs:

- polygon GeoJSON/GeoPackage;
- tile manifest;
- tile GeoTIFF paths.

Outputs:

- per-tile manual label rasters;
- per-tile manual mask rasters;
- audit CSV.

### 12.3 Add Dataset Loader Support

Update the dataset class so it can choose label source:

```yaml
labels:
  source: weak_region_type | manual | hybrid
```

Modes:

| Mode | Meaning |
|---|---|
| `weak_region_type` | current weak labels |
| `manual` | use only manual labels |
| `hybrid` | train on weak labels but validate/test on manual labels |

Recommended first mode:

```text
hybrid
```

Train on weak labels or weak+manual, but evaluate only on manual test labels.

### 12.4 Add Tile-Level Baseline

Add a simple tile classifier:

```text
tile -> informal vs formal_dense
```

Inputs:

- RGB/S2;
- VLM feature vector pooled over tile;
- socioeconomic summary statistics.

Why:

- current labels are region-level;
- tile-level classification may be a better match;
- it gives a sanity baseline before spending more CU on segmentation.

### 12.5 Add Manual Metrics

Evaluation should report metrics separately for:

- weak-label validation;
- manual validation;
- manual test.

Never mix them without labeling the table clearly.

---

## 13. Experiment Plan After Annotation

### Phase A - Manual Evaluation Only

Goal:

```text
Can current models beat trivial baselines on trusted labels?
```

Run:

1. baseline CNN;
2. VLM visual;
3. VLM language;
4. full model with only verified socioeconomic channels.

Evaluate on manual test set only.

Decision:

- if balanced accuracy remains near 0.50, the current feature/supervision setup is not working;
- if dense-formal FPR drops with language/socioeconomic features, the hypothesis has support.

### Phase B - Manual Training Subset

Goal:

```text
Does cleaner supervision improve segmentation?
```

Run:

- train on manual train set;
- validate on manual val set;
- test on manual test set;
- compare to weak-label training.

Decision:

- if manual training improves results, weak labels were the bottleneck;
- if not, image resolution/features are likely the bottleneck.

### Phase C - Tile-Level Classification

Goal:

```text
Is the problem solvable at tile/region level?
```

Run:

- RGB tile classifier;
- VLM pooled feature classifier;
- VLM + socioeconomic classifier.

Decision:

- if tile classification works better than segmentation, reframe the paper around screening/detection rather than pixel masks.

### Phase D - High-Resolution Grounding

Goal:

```text
Does LocateAnything need high-res imagery to be useful?
```

Run LocateAnything on high-resolution reference chips, then aggregate features to Sentinel-2 tile space.

Decision:

- if high-res grounding improves language ablation, the original 10 m VLM grounding was too coarse;
- if it does not, language grounding may not be the right signal.

---

## 14. Success Criteria

### 14.1 Minimum Success

The project is working at a diagnostic research level if:

- manual test labels exist;
- models are evaluated on manual labels;
- predicted-positive rate is not collapsed;
- balanced accuracy is clearly above random;
- formal-control FPR is measured and discussed.

### 14.2 Strong Paper Success

The project is strong if:

- language model beats neutral VLM visual baseline;
- full socioeconomic model reduces formal-control FPR;
- recall on known informal settlements stays acceptable;
- manual test results agree with qualitative overlays;
- tile-level and segmentation baselines are both reported honestly.

### 14.3 Stop Conditions

Stop spending CU if:

- `pred_pos_rate` stays near 0 or 1;
- balanced accuracy stays near 0.50 after multiple corrected runs;
- manual-label audit shows too few formal or slum pixels;
- train/val/test leakage is detected;
- socioeconomic placeholders are accidentally included in headline claims.

---

## 15. Recommended Two-Week Work Plan

### Day 1-2: Annotation Setup

- create QGIS project;
- load regions, Sentinel-2, ESA polygons, OSM, high-res basemap;
- create `manual_v1` GeoPackage with schema;
- choose 100-150 test tiles.

### Day 3-5: Manual Test Annotation

- annotate all selected test tiles;
- review each tile by a second person;
- mark uncertainty as unknown;
- export GeoJSON/GeoPackage.

### Day 6: Rasterization and Audit

- implement rasterization script;
- rasterize manual labels to tile grids;
- run class-count audit;
- inspect overlays.

### Day 7-8: Manual Evaluation

- adapt dataset loader for manual labels;
- run current best models on manual test;
- compute balanced accuracy, FPR, IoU, precision/recall/F1.

### Day 9-10: Tile-Level Baseline

- implement simple tile classifier;
- compare RGB, VLM, and VLM+socioeconomic features.

### Day 11-12: Manual Training Labels

- annotate additional train/val tiles if manual test shows promise;
- train segmentation on manual/hybrid labels.

### Day 13-14: Final Analysis

- decide final framing:
  - segmentation paper;
  - tile-screening paper;
  - dataset/pipeline paper;
  - negative-results/diagnostic paper.
- update report, figures, and presentation.

---

## 16. Team Task Split

Suggested division for five team members:

| Person | Main responsibility |
|---|---|
| Member 1 | QGIS project setup, region/tile selection |
| Member 2 | Informal-region annotation |
| Member 3 | Formal-control annotation |
| Member 4 | Review/QC and disagreement resolution |
| Member 5 | Rasterization scripts, audit, model reruns |

Everyone should annotate a small shared calibration set first. Compare labels, agree on class definitions, then divide the remaining tiles.

---

## 17. Final Recommendation

The project can work, but not by simply rerunning Phase 4.

The next real move is:

```text
build manual_v1 labels -> evaluate honestly -> decide segmentation vs tile classification -> rerun only the useful experiments
```

Manual annotation is not a side task. It is the missing scientific foundation. Once we have trusted labels, the existing BanglaSlumNet codebase becomes useful: it can test whether language grounding and socioeconomic context actually reduce Dhaka's dense-formal false positives.

