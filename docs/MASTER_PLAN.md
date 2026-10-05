# BanglaSlumNet — Master Plan

Last updated: 2026-10-05

This is the single source of truth for what BanglaSlumNet is, what we found, what the paper claims, and what we do next. It **supersedes the direction** set in `PROJECT_RECOVERY_PLAN.md`, `SPEC.md`, `SUPERVISOR_BRIEF.md`, and `NEXT_SESSION.md`. Those files stay in the repo as history (see [§13](#13-status-of-older-documents)).

---

## 0. TL;DR

- **The old plan** was a new model: LocateAnything vision-language features + socioeconomic fusion on Sentinel-2. It has produced no trustworthy result. The best corrected run sits at **balanced accuracy ≈ 0.50**.
- **Why it failed:**
  1. The labels are mostly wrong. Only 4–26% of each "slum" region box is actually slum, and the formal control boxes contain real slums, including Korail itself.
  2. Region boxes overlap, so the same ground is labeled both slum and formal.
  3. The vision-language model was trained on high-resolution photos and is being fed 10 m imagery it was not built for.
- **The new plan** drops model novelty and builds the paper on data and findings:
  1. **A year-matched, open-data Dhaka slum benchmark.** It uses third-party reference slum outlines (ESA EO4SD-Urban 2017), Sentinel-2, and Google Open Buildings 2.5D Temporal, with dense-formal look-alikes as hard negatives.
  2. **A failure analysis of GRAM**, the AAAI-26 Outstanding Paper slum detector. Preliminary scoring shows it ranks Korail's slum pixels *below* the surrounding formal areas.
  3. **A measurement of how much region-box weak labels cost.**
  4. **A 2017–2023 slum change map of Dhaka.**
- **No manual annotation** is needed. A light spot-check of test areas for changes since 2017 is recommended.
- **Everything uses free data** in Google Earth Engine + Colab.

---

## 1. What the paper is about

### 1.1 Working titles

- *Where Slum Detectors Fail: An Open-Data Benchmark for Informal-Settlement Mapping in Dhaka*
- *Dense but Formal: Benchmarking and Mapping Dhaka's Informal Settlements with Year-Consistent Open Data*

### 1.2 One-sentence pitch

Dhaka is missing from every recent slum-mapping benchmark, and the state-of-the-art detector fails there. We build a year-consistent open-data benchmark for Dhaka, show why existing models and cheap weak labels break, and produce the first open-data slum change map of the city for 2017–2023 *(the "first" claim needs a final literature check)*.

### 1.3 Research questions

| # | Question |
|---|---|
| RQ1 | How well does a state-of-the-art generalizable slum detector (GRAM) transfer to Dhaka, and where does it fail? |
| RQ2 | Can free, year-matched open data map Dhaka's slums at neighbourhood scale, and separate them from dense *formal* areas? |
| RQ3 | How much accuracy do coarse region-box weak labels cost compared with reference-polygon labels? |
| RQ4 | How did Dhaka's slum extent change between 2017 and 2023? |

### 1.4 Contributions

| # | Contribution | Type | Status |
|---|---|---|---|
| C1 | Dhaka benchmark: year-matched labels and imagery, dense-formal hard negatives, spatially blocked splits, released as scripts and derived labels | Dataset / benchmark | Planned |
| C2 | Transfer-failure analysis of GRAM on Dhaka | Empirical finding | Preliminary result exists ([§3.3](#33-gram-fails-on-dhaka-preliminary)) |
| C3 | Measured cost of region-box weak supervision | Empirical finding | Label audit done ([§3.1](#31-region-box-labels-are-mostly-wrong)); training comparison planned |
| C4 | Annual open-data slum map of Dhaka, 2017–2023 | Applied science | Planned |

### 1.5 What the paper does **not** claim

- No new architecture. Models are deliberately simple and standard.
- No pixel-accurate slum boundaries. Sentinel-2 at 10 m supports neighbourhood-scale mapping, not building-level outlines.
- No Bangladesh-wide map. ESA reference data covers Dhaka's core city only.

### 1.6 Why this is novel enough

The model side of the field is crowded:
- [SLUM-i](https://arxiv.org/html/2602.04525): a 2026 multi-city benchmark using few labels.
- [GRAM](https://arxiv.org/abs/2511.10300): a mixture-of-experts model that adapts to new cities; AAAI-26 Outstanding Paper.
- [A 2026 study](https://www.nature.com/articles/s44458-026-00054-6) mapping slum expansion during COVID with minimal labels on sub-metre imagery.
- [SatBLIP](https://arxiv.org/pdf/2604.14373): vision-language learning on satellite imagery.

A slightly better model on 10 m imagery won't beat these. What these works lack:
- **Dhaka.** SLUM-i's seven cities include none in Bangladesh, and Dhaka is not among GRAM's 12 training cities.
- **The dense-formal regime.** Dhaka's formal city is itself extremely dense, which breaks the "sprawling slum vs low-density formal" contrast most training cities show.

A benchmark where the state of the art measurably fails, plus findings on label cost and change over time, is a defensible contribution even if our own model is simple.

---

## 2. How we got here

| When | What happened |
|---|---|
| 2026-04 | Blueprints v1/v2 (Google Docs). |
| 2026-05-15 | **v3: GRAM zero-shot on Dhaka** (`gram-zero-shot` branch, `gram_baseline/FINDINGS.md`). 27 Esri tiles over Korail, Mirpur and Old Dhaka. GRAM appeared to flag dense areas as slum everywhere. There was no ground truth, so only mean probabilities were reported. |
| 2026-06 | **v4 pipeline built on `main`.** 12 Dhaka region boxes, Sentinel-2 tiling (720 tiles, 128×128 px), SAS-Net image normalization, LocateAnything/MoonViT feature caching, cross-attention socioeconomic fusion, experiment matrix, preflight and collapse diagnostics. |
| 2026-06-16 | Weak-label rule switched to region type (informal box → slum, formal box → formal) after the VIIRS-threshold rule collapsed to all-formal. |
| 2026-06-17 | First results looked good (HC-IoU ≈ 0.66) but were an **all-slum collapse** (recall 1.0). After the fix: HC-IoU ≈ 0.46, **validation balanced accuracy ≈ 0.50**, i.e. no real signal. |
| 2026-06-23 | Overleaf report written (`report/main.tex`). |
| 2026-07-01 | `PROJECT_RECOVERY_PLAN.md`: diagnosis is supervision quality; proposed a two-week manual annotation effort. Never started. |
| 2026-07 → 10 | No commits on any branch. |
| 2026-10-05 | **Re-think.** Checked the ESA reference polygons, audited the region boxes, scored GRAM against ESA, and reviewed data options. Produced this plan. |

---

## 3. Key evidence so far

### 3.1 Region-box labels are mostly wrong

Share of each region box covered by ESA EO4SD-Urban 2017 slum polygons. Under the current rule, every built pixel in an informal box is labeled slum and every built pixel in a formal box is labeled formal.

| Region | Our label | ESA polygons | Slum km² | **Slum % of box** |
|---|---|---:|---:|---:|
| kamrangirchar | informal | 34 | 2.33 | 25.5% |
| bhashantek | informal | 82 | 1.19 | 13.0% |
| karail_extension | informal | 49 | 1.05 | 11.5% |
| hazaribagh | informal | 65 | 1.02 | 11.2% |
| kallyanpur | informal | 112 | 1.01 | 11.0% |
| mirpur_beribadh | informal | 39 | 0.75 | 8.1% |
| korail | informal | 20 | 0.62 | 6.8% |
| tongi | informal | 22 | 0.37 | 4.1% |
| **gulshan_baridhara** | **formal** | 67 | 1.89 | **20.6%** |
| **uttara** | **formal** | 53 | 1.30 | **14.2%** |
| dhanmondi | formal | 40 | 0.45 | 4.9% |
| old_dhaka | formal | 51 | 0.26 | 2.9% |

- Boxes are ~9.1 km² each. Shares are of total box area; the share among *built* pixels will be somewhat higher, because water and parks are unlabeled.
- **The Korail slum polygon (0.414 km²) lies 90% inside the `gulshan_baridhara` formal box.** Dhaka's best-known slum is labeled slum in one region and formal in another.
- `uttara` contains Dakshin Khan, a 0.62 km² slum, labeled formal.
- The top ~25% of the `tongi` box is beyond the ESA mapped area. Missing polygons there do not mean non-slum.

### 3.2 Region boxes overlap and contradict each other

Labels are exported per region (`gee/export_weak_labels.py:76-78`), so overlapping ground gets a different label in each region's file.

| Box A | Box B | Overlap (% of A) | Effect |
|---|---|---:|---|
| karail_extension (informal) | gulshan_baridhara (formal) | 54% | Same pixels labeled slum *and* formal |
| hazaribagh (informal) | dhanmondi (formal) | 42% | Same pixels labeled slum *and* formal |
| korail (informal) | gulshan_baridhara (formal) | 32% | Same pixels labeled slum *and* formal |
| tongi (informal) | uttara (formal) | 24% | Same pixels labeled slum *and* formal |
| korail | karail_extension | 69% | Same ground can sit in train *and* test |
| kamrangirchar | hazaribagh | 26% | Same ground can sit in train *and* test |
| kallyanpur | mirpur_beribadh | 19% | Same ground can sit in train *and* test |

All 12 boxes are also still marked `ESTIMATE` / `TODO_VERIFY` in `config/regions_dhaka.yaml`. Together, §3.1 and §3.2 are a sufficient explanation for the ≈ 0.50 balanced accuracy.

### 3.3 GRAM fails on Dhaka (preliminary)

GRAM's 27 saved probability maps from the May run (`gram-zero-shot` branch), scored against ESA polygons. A pixel counts as predicted slum at probability > 0.5.

| Location | ESA slum % | GRAM predicted slum % | Precision | Recall | IoU | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Korail | 16.6% | 51.8% | 11.3% | 35.5% | 9.4% | **0.42** |
| Mirpur | 4.2% | 59.0% | 2.2% | 30.4% | 2.1% | **0.32** |
| Old Dhaka | 18.9% | 45.9% | 31.0% | 75.3% | 28.1% | 0.74 |
| **All** | 13.2% | 52.2% | 13.6% | 53.9% | 12.2% | **0.54** |

- **Korail and Mirpur score worse than chance.** GRAM assigns slum pixels *lower* probability than the surrounding formal areas. The Korail tile that is 41% slum gets the lowest mean probability of the nine (0.27).
- **Correction to `FINDINGS.md`:** it claimed GRAM blanket-flags *formal* Old Dhaka as slum. Those tiles are about 19% ESA slum (riverside settlements), and GRAM ranks them reasonably (AUC 0.74). Do not use Old Dhaka as the GRAM failure example.
- **Why this is preliminary:**
  1. Only GRAM's first phase was run. Its main idea is a second, test-time adaptation phase that adapts to a new city using its own pseudo-labels (`main_moe_pl_v3.py`); that was never run.
  2. The tiles were zoom-16 Esri imagery at **2.19 m/px**. `FINDINGS.md` wrongly says ~1.2 m/px. GRAM was trained on finer imagery.
  3. Only 4 of 12 domain indices were tried, chosen by activation on a Korail test tile.
  4. Only 27 tiles at 3 sites.
  5. The imagery is recent, but the labels are from 2017.

### 3.4 Imagery and model mismatch

- At Sentinel-2's 10 m, the median ESA slum polygon is about **55 pixels**. Polygons under 100 px make up 71% of the count but only 20% of slum area. Neighbourhood-scale mapping is feasible; building-level outlines are not.
- LocateAnything was trained on high-resolution photos. A 128 px tile at 10 m covers 1.28 km, far outside its training distribution. The v4 code already disabled LocateAnything label validation "until high-res grounding."

---

## 4. Data

### 4.1 What we use

| Role | Source | Resolution / years | Access | Notes |
|---|---|---|---|---|
| **Reference labels** | [ESA EO4SD-Urban Dhaka informal settlements 2017](https://datacatalog.worldbank.org/search/dataset/0041703/dhaka-bangladesh-informal-settlements-esa-eo4sd-urban) | 1,624 polygons, 23.9 km², interpreted from very-high-resolution imagery; 2017 | Free, CC-BY 4.0, 4.3 MB shapefile | Covers Dhaka "Core City" (~306 km²). Slum-type attribute `NB_LOCTYP`: High density 903 / Low density 438 / Pocket 248 / Linear along rivers 27 / Linear along railway 8. Ward name in `AL3_NAMEF` (99 wards). |
| **Imagery** | Sentinel-2 | 10 m; 2017–2023 | Free, GEE | **2017 L2A verified in GEE (2026-10-05).** `COPERNICUS/S2_SR_HARMONIZED` gives 100% coverage of the core city from tiles 45QZG + 46QBM. "2017 dry" (Dec 2016–Feb 2017) has 5 dates; "2018 dry" (Dec 2017–Feb 2018) has 4 dates; all scenes are under 20% cloud. The 2017 L2A archive is sparse (18 scenes vs 68 L1C), but enough for median composites. |
| **Building structure** | [Google Open Buildings 2.5D Temporal](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1) | ~4 m effective; annual 2016–2023 | Free, GEE | Building presence, height, and fractional count. Small, low, densely packed buildings are the most direct slum cue available for free. It is itself an AI product estimated from Sentinel-2, so its errors become ours. |
| Context | VIIRS nighttime lights, WorldPop / GHS-POP, GHSL built-up, Dynamic World | 10–500 m | Free, GEE | Already wired into the pipeline. |
| **Do not interpret** | `osm_roads` (an accessibility proxy) and `wb_poverty` (a constant 0) | — | — | Known placeholders. Keep them out of headline claims. |
| GRAM baseline input | Esri World Imagery | Zoom 16 = 2.19 m/px here; zoom 18 ≈ 0.55 m/px | Free to view; terms restrict bulk download and ML use | Use only for the GRAM comparison. Licensing must be checked ([§11](#11-open-decisions)). |

### 4.2 Which year: 2017 benchmark + 2023 map

ESA labels are from 2017. On 2026-10-05 we measured how much of that slum area changed by 2023, using Open Buildings Temporal inside each ESA polygon. A polygon counts as "changed" if it got ≥ 3 m taller (redevelopment) or lost ≥ 50% of its building presence (clearance). The 2017→2018 comparison serves as the noise floor, since little really changes in one year.

| Thresholds | 2017→2018 (noise) | 2017→2023 |
|---|---:|---:|
| +2 m / ×0.6 presence | 0.7% | 6.0% |
| **+3 m / ×0.5 presence** | **0.4%** | **2.6%** |
| +4 m / ×0.4 presence | 0.2% | 1.6% |

- **About 94–98% of 2017 slum area is still structurally the same in 2023.** The median slum polygon gained +0.1 m in height, with a presence ratio of 1.03.
- Change concentrates in small and linear settlements. At +3 m / ×0.5: pockets 6.6%, linear along rivers 10.2%, linear along railway 37% (tiny area). High-density slums changed only 2.5%.
- **What this cannot see:** *new* slums that appeared after 2017 (they would be wrongly labeled non-slum), and in-place upgrading with no height change.

**Design:**
- **2017 = the benchmark.** Labels and inputs come from the same year. The paper's headline numbers come from here.
- **2023 = the current map and a second training set.** Use 2023 Sentinel-2 + Open Buildings with ESA labels after two masks:
  1. ignore ESA polygons flagged as changed;
  2. ignore cells whose building presence rose sharply from 2017 to 2023 (possible new slums).

  Report 2017-trained vs 2023-trained models on the same test blocks.
- **2024 onward:** Open Buildings Temporal ends at 2023. Later years would need a Sentinel-2-only model. The ablation (E3) shows how much is lost without it.
- Spot-checks for 2023 are easier than for 2017, because current Esri / Google imagery shows roughly the same period.

Caveat: Open Buildings Temporal is a model output, not observed truth. Its year-to-year noise is the 2017→2018 column above.

### 4.3 What we considered and rejected

| Option | Why not |
|---|---|
| Planet NICFI basemaps (~4.8 m) | Free access ended. The program closed in January 2025, and Earth Engine availability ran only to 2026-08-01. The paid replacement costs about $180/month. |
| Commercial sub-metre imagery (Maxar, Pléiades) | Cost. Revisit only if a research grant appears. |
| Esri / Google basemaps as training data | Licence restricts bulk download and ML use. Fine for viewing and spot-checks. |
| Full manual annotation (the recovery plan) | ESA already did the hard slum delineation from better imagery. |
| Bangladesh-wide mapping | There is no national reference polygon set. Dhaka first. |

### 4.4 Is Google Earth Engine enough?

Yes, for this plan. It hosts every layer in §4.1 except the Esri tiles. It has no sub-metre imagery for Bangladesh, which only matters for the GRAM comparison. Model training stays in Colab.

### 4.5 Storage

- **Google Drive is the canonical data store:** `/gdrive/MyDrive/BanglaSlumNet/`.
- ESA polygons go in `data/raw/external_boundaries/esa_eo4sd_urban/`.
- Derived labels go in `data/labels/esa_v1/`.
- **Large data never goes in Git.** The repo holds only code, configs, small manifests, and docs.

---

## 5. Method

### 5.1 Task framing

- **Primary task: grid-cell classification.** Each 100 m cell is classified slum vs non-slum, using a context window of Sentinel-2 + Open Buildings layers around it. 100 m cells are a common unit in deprived-area mapping, such as the IDEAMAPS grid. The core city gives about 30,000 cells.
- **Secondary task: segmentation**, reported for comparability with GRAM and older work.

### 5.2 Labels (no annotation)

| Label | Rule |
|---|---|
| slum | Cell's ESA slum fraction ≥ 50% *(threshold to tune; report sensitivity)* |
| non-slum | Cell is built-up, its ESA slum fraction is 0, and it lies inside the ESA mapped area |
| ignore | Mixed cells (0 < fraction < 50%), non-built cells, cells outside the ESA mapped area, and a boundary buffer of about 20 m around polygon edges. **For years after 2017, also:** ESA polygons flagged as changed, and cells with a sharp rise in building presence since 2017 (§4.2). |

For segmentation, the pixel version is: slum = built ∩ inside polygon; non-slum = built ∩ inside mapped area ∩ outside every polygon; ignore = the rest.

**ESA mapped-area boundary.** We need the Dhaka North + South City Corporation boundary, or ESA's "Core City" outline, to decide where *absence* of a polygon means non-slum. To obtain.

### 5.3 Spatial splits (anti-leakage)

- Use **spatial block cross-validation**: about 2 × 2 km blocks, with 5 folds grouped by block. Ward-grouped splits are an alternative.
- Never random pixel or cell splits, and never the old region-box splits.
- The same location must never appear in train and test across different years.

### 5.4 Hard-negative strata

Report metrics separately for:
- **Dense-formal areas.** Define them from Open Buildings density/height plus known formal districts (Old Dhaka core, Gulshan, Banani, Dhanmondi, Uttara sectors). The key number is the **false-positive rate on dense-formal areas**.
- ESA slum types (High density / Low density / Pocket / Linear).
- Polygon size classes (< 100 px, 100–1,000 px, > 1,000 px at 10 m).

### 5.5 Models (deliberately simple)

| Model | Inputs | Purpose |
|---|---|---|
| Gradient boosting / random forest | Per-cell statistics: Sentinel-2 bands and indices, Open Buildings height and count, context | Strong, interpretable baseline |
| Small CNN or U-Net | Sentinel-2 + Open Buildings rasters | Main learned model |
| GRAM, full | Esri tiles, with GRAM's own test-time adaptation | State-of-the-art comparison (RQ1) |
| *(optional)* LocateAnything features | Cached MoonViT features | One ablation row only, if cheap; not the headline ([§11](#11-open-decisions)) |

### 5.6 Metrics

AUC, balanced accuracy, precision / recall / F1, IoU, **dense-formal false-positive rate**, and predicted-positive rate (to catch collapse). Report by stratum (§5.4) and as mean ± standard deviation across spatial folds.

---

## 6. Experiments

| ID | Experiment | Answers | Output |
|---|---|---|---|
| **E0** | Label audit: ESA vs region boxes, overlaps, polygon sizes | RQ3, C3 | Done (§3.1–3.2); to be turned into a figure |
| **E1** | GRAM done properly: zoom-18 tiles, test-time adaptation, all 12 domain indices, scored on ESA-labelled test blocks | RQ1, C2 | GRAM table + failure maps |
| **E2** | Main benchmark: 2017, spatial CV, models from §5.5 | RQ2, C1 | Main results table |
| **E3** | Ablation: Sentinel-2 only → + Open Buildings → + context (→ + LocateAnything, optional) | RQ2 | Ablation table |
| **E4** | Weak-label cost: same model trained on (a) old region-box labels vs (b) ESA labels, both tested on ESA | RQ3, C3 | One comparison table |
| **E5** | Change map: apply the best E2 model to 2018–2023; spot-check known events such as Korail fires and evictions | RQ4, C4 | Annual maps + change statistics |

### 6.1 Decision gates

- **After E1:** if GRAM with adaptation does well on Dhaka (say AUC ≥ 0.8), C2 becomes "GRAM transfers if adapted," which is still useful. The paper then leans on C1, C3 and C4.
- **After E2:** if our spatial-CV AUC is near 0.5, it becomes a benchmark plus negative-results paper (C1–C3). Skip E5.

### 6.2 Stop conditions (do not spend more GPU if…)

- predicted-positive rate stays near 0 or 1;
- spatial-CV balanced accuracy stays near 0.50 after corrected runs;
- leakage is found between train and test blocks;
- a placeholder layer (`osm_roads`, `wb_poverty`) has slipped into a headline result.

---

## 7. Work plan

The week numbers are relative; adjust them to the team's timetable. Owners are suggestions.

| Week | Tasks | Suggested owner |
|---|---|---|
| 1 | Put the ESA shapefile on Drive. Get the city-corporation / mapped-area boundary. Pull Open Buildings Temporal for 2017–2023. Decide the open items in §11. | Angkon (lead), Fardeen |
| 1–2 | Build the 100 m grid and ESA-derived labels (§5.2). Build spatial blocks (§5.3). Add audit checks (class counts per fold, no leakage). Turn E0 into a figure. | Angkon, Zayan |
| 2–3 | **E1:** fetch zoom-18 tiles for test blocks (if licence allows), run GRAM adaptation and the domain sweep, score on ESA. | Angkon (ran the original GRAM baseline), Nafiz |
| 2–3 | **E2/E3:** export per-cell features, train the boosting baseline and CNN, run spatial CV. | Zayan, Masum |
| 4 | **E4:** weak-label vs ESA-label comparison. Define dense-formal strata. Write up the hard-negative analysis. | Fardeen |
| 5 | **E5:** 2018–2023 maps and change statistics. Spot-check 20–40 cells per year against Esri / Google imagery by eye. | Masum, all |
| 6 | Paper writing, figures, release scripts and derived labels. | All |

---

## 8. Code changes

### 8.1 New

| File | Purpose |
|---|---|
| `gee/export_esa_labels.py` (or local) | Rasterize ESA polygons to the 10 m grid and 100 m cells; write label, ignore mask, and slum fraction |
| `gee/export_open_buildings.py` | Export Open Buildings Temporal (presence, height, count) per year |
| `gee/export_s2_annual.py` | Year-matched Sentinel-2 L2A dry-season composites, 2017–2023 (2017 coverage verified) |
| `src/data/grid.py` | 100 m grid, per-cell features, spatial blocks and folds |
| `src/models/cell_classifier.py` | Boosting baseline + small CNN |
| `scripts/score_against_esa.py` | Generalization of the 2026-10-05 GRAM scoring script |
| `config/benchmark.yaml` | Thresholds, buffer, block size, folds, years |

### 8.2 Reuse

`src/eval/metrics.py` (balanced accuracy, collapse checks), `src/tracking/` (registry), `src/data/preflight.py`, `src/data/socioeconomic.py`, `gee/ee_export_utils.py`, `src/viz/` (palette, plots), `src/eval/gram_baseline.py` (wrapper), notebook structure (P0 setup cells).

### 8.3 Retire or sideline

| Item | New status |
|---|---|
| `config/regions_dhaka.yaml` region boxes | Keep only as named places for reporting. No longer a label source. |
| `gee/export_weak_labels.py` region-type rule | Kept only to reproduce E4's weak-label arm |
| LocateAnything feature pipeline | Optional ablation |
| Cross-attention fusion model | Optional; the simple models come first |
| SAS-Net | Optional (2017 L2A exists, so it isn't needed to correct L1C imagery) |

---

## 9. Paper outline

1. **Introduction:** dense-megacity problem; Dhaka missing from benchmarks; GRAM fails; contributions C1–C4.
2. **Related work:** Sentinel-2 slum mapping (Kuffer et al., Gram-Hansen et al., Matarira et al.); generalization and few-label work (GRAM, SLUM-i, the minimal-supervision study); vision-language models in remote sensing (SatBLIP); label noise and uncertainty (Fisher et al.).
3. **Study area and data:** Dhaka core city; ESA reference; Sentinel-2; Open Buildings Temporal; year matching.
4. **Label audit:** region boxes vs reference (E0); the cost of weak labels (E4).
5. **Benchmark:** grid, labels, spatial splits, hard-negative strata, metrics.
6. **Experiments:** GRAM transfer (E1); baselines and ablations (E2/E3).
7. **Dhaka slum change 2017–2023** (E5).
8. **Discussion:** what 10 m + building layers can and cannot see; limitations (2017 reference, Open Buildings errors, core city only).
9. **Conclusion and release.**

### 9.1 Draft abstract (placeholders in brackets — do not finalize before E1/E2)

> Recent slum-detection models report strong cross-city generalization, yet Dhaka — one of the world's densest megacities — is absent from recent slum-mapping benchmarks. We show that this gap matters: applied to Dhaka, GRAM, a state-of-the-art region-adaptive slum detector, [ranks informal-settlement pixels below the surrounding formal neighbourhoods (AUC [x])]. We introduce BanglaSlumNet, a year-consistent open-data benchmark for Dhaka. It pairs 1,624 reference informal-settlement polygons interpreted from very-high-resolution imagery (ESA EO4SD-Urban, 2017) with Sentinel-2 imagery and Open Buildings 2.5D Temporal building height and density layers from the same year, evaluated under spatially blocked splits with explicit dense-formal hard negatives. We also find that the coarse region-level weak labels common in low-resource settings mislabel most pixels: only 4–26% of the area inside known-informal regions is slum, while formal control regions contain up to 21% slum, including Korail itself. Models trained on reference labels [reach AUC [x] and reduce the dense-formal false-positive rate from [x] to [x]]. Applying the best model to 2018–2023 yields [an annual open-data slum map of Dhaka showing [trend]].

`report/main.tex` holds a *progress-report* abstract (2026-10-05). It states only established findings (label audit, preliminary GRAM scoring, 2017→2023 stability) and describes the benchmark as the new direction. Replace it with a version of the draft above once E1/E2 results exist. The report body still describes the v4 pipeline.

### 9.2 Target venue

Decide after E1/E2. A strong E1 + E2 fits a remote-sensing journal or an AI-for-social-good / Earth-observation track. If E2 is weak, aim for a benchmark or workshop venue.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| ESA 2017 labels contain errors, or slums changed after 2017 | Year matching for core results. Measured change by 2023 is only ~2–6% of slum area (§4.2); mask changed polygons and new construction. Spot-checks for later years. Report ESA as a *reference*, not ground truth. |
| New slums after 2017 are labeled non-slum in later years | Ignore cells with a sharp rise in building presence since 2017; spot-check them |
| Missing polygons outside the ESA mapped area are mistaken for non-slum | Obtain the mapped-area boundary; ignore everything outside it |
| Open Buildings errors carry into our model | Ablate with and without it; discuss as a limitation |
| GRAM authors object to how we ran it | Run their full adaptation, at their resolution, with the full domain sweep, using their released code |
| Esri licence blocks the zoom-18 download | Ask; fall back to fewer tiles, or report the zoom-16 result with the caveat |
| Novelty challenged ("someone already did Dhaka") | Final literature check before submission; frame the main novelty as the benchmark and failure analysis rather than "first" claims |
| Small pocket slums are invisible at 10 m | Stratify by polygon size and state the detection floor openly |

---

## 11. Open decisions

| # | Decision | Recommendation |
|---|---|---|
| D1 | Adopt this plan over `PROJECT_RECOVERY_PLAN.md`? | Yes |
| D2 | LocateAnything: optional ablation or drop? | Optional ablation only if the cached features can be reused cheaply; otherwise drop |
| D3 | Esri zoom-18 tiles for the GRAM rerun: licence OK for research use? | Check Esri terms before downloading; keep the tile count minimal |
| D4 | Cell size: 100 m, or also 50 m? | 100 m primary; 50 m as a sensitivity check |
| D5 | Slum-fraction threshold for a "slum" cell | 50%; report sensitivity at 30% and 70% |
| D6 | Spot-check effort for 2018–2023 | 20–40 cells per year, one person, viewing imagery only |

---

## 12. How we work

- **Compute:** Google Colab. The repo is cloned fresh each session to `/content/BanglaSlumNet`, never onto Drive, because Git on Drive breaks (dubious ownership, detached HEAD).
- **Data:** Google Drive at `/gdrive/MyDrive/BanglaSlumNet/`.
- **Earth Engine project:** `banglaslumnet-research` (used by every `init_ee(...)` call in `gee/`).
- **Loop:**
  1. Edit locally.
  2. Run `python -m py_compile`.
  3. Commit and push to `main`.
  4. In Colab, restart the runtime and **reopen the notebook from the GitHub tab** (uploading gives a stale copy).
  5. Always run notebook cells P0.1–P0.5 first (they set `cwd` and `sys.path`).
- **Local clone:** the repo is about 150 MB and a full clone times out on slow connections. Use `git clone --depth 50`.

### 12.1 Where things are

| Path | What |
|---|---|
| `docs/MASTER_PLAN.md` | **This file** |
| `report/main.tex` | Overleaf report (v4 framing; to be rewritten) |
| `notebooks/BanglaSlumNet_Colab.ipynb` | v4 pipeline notebook |
| `config/` | `default.yaml`, `experiments.yaml`, `regions_dhaka.yaml` |
| `gee/` | Earth Engine export scripts |
| `src/` | Data, models, training, evaluation, tracking, visualization |
| Branch `gram-zero-shot` | GRAM code, checkpoint, 27 Dhaka probability maps, `FINDINGS.md` |

---

## 13. Status of older documents

| Document | Status |
|---|---|
| `PROJECT_RECOVERY_PLAN.md` | Superseded. Its diagnosis (supervision quality) still holds; its fix (two weeks of manual annotation) is replaced by ESA labels plus spot-checks. |
| `SPEC.md` | Superseded as the paper narrative. Useful as the v4 pipeline description. |
| `SUPERVISOR_BRIEF.md` | Superseded. Its "GRAM fails" claim must be updated with §3.3, including the Old Dhaka correction. |
| `NEXT_SESSION.md` | Superseded. |
| `BanglaSlumNetV4.md`, `AGENT_HANDOFF.md` | v4 reference only. |
| `DATA_CARD.md` | Outdated (lists 5 regions). Rewrite for the ESA-based benchmark. |
| `RESULTS.md` | Empty. No trustworthy results exist yet. |
| `gram-zero-shot:gram_baseline/FINDINGS.md` | Partially wrong: the Old Dhaka claim and the "~1.2 m/px" resolution. See §3.3. |

---

## 14. References and links

- ESA EO4SD-Urban Dhaka informal settlements 2017: <https://datacatalog.worldbank.org/search/dataset/0041703/dhaka-bangladesh-informal-settlements-esa-eo4sd-urban> — direct zip: <https://datacatalogfiles.worldbank.org/ddh-published/0041703/1/DR0052090/eo4sd_dhaka_informal_2017.zip>
- Open Buildings 2.5D Temporal (GEE): <https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1> — overview: <https://research.google/blog/open-buildings-25d-temporal-dataset-tracks-building-changes-across-the-global-south/>
- GRAM (AAAI-26): <https://arxiv.org/abs/2511.10300> — code: <https://github.com/DS4H-GIS/GRAM>
- SLUM-i: <https://arxiv.org/html/2602.04525>
- Minimally supervised sub-metre slum mapping (2026): <https://www.nature.com/articles/s44458-026-00054-6>
- SatBLIP: <https://arxiv.org/pdf/2604.14373>
- NICFI end of free access: <https://developers.google.com/earth-engine/datasets/catalog/projects_planet-nicfi_assets_basemaps_asia>, <https://nimbo.earth/stories/end-nicfi-satellite-tropical-forest-monitoring-alternative/>
- Existing report bibliography: `report/references.bib`
