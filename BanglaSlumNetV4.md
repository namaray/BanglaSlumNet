# BanglaSlumNet V4 - Complete Project Reference

Last updated: 2026-06-27

This file is the master project reference for BanglaSlumNet. It explains the research idea, dataset, codebase, notebook phases, experiments, results so far, bugs found, fixes made, report/presentation artifacts, and the next scientific steps.

For engineering handoff, also read `AGENT_HANDOFF.md`. For the Colab pipeline, use `notebooks/BanglaSlumNet_Colab.ipynb`. For the written report, use `report/main.tex`.

---

## 1. One-Sentence Project Summary

BanglaSlumNet is a Dhaka-focused informal-settlement detection pipeline that tests whether language-grounded visual features from LocateAnything, fused with socioeconomic context, can reduce the common remote-sensing failure where dense formal neighborhoods are mistaken for slums.

---

## 2. Core Research Problem

The central problem is not simply detecting dense built-up areas. In Dhaka, many places are dense:

- informal settlements such as Korail, Bhashantek, Kamrangirchar, Kallyanpur, Hazaribagh, Tongi, Mirpur Beribadh, and Karail extension;
- dense formal or higher-income controls such as Old Dhaka, Gulshan-Baridhara, Dhanmondi, and Uttara.

At Sentinel-2 resolution, these areas can share similar visible cues: roof density, texture, irregularity, and built-up intensity. An optical-only model can therefore learn a shortcut:

```text
dense urban texture -> slum
```

That shortcut creates false positives in dense formal controls. BanglaSlumNet asks whether adding semantic and socioeconomic evidence can help separate:

```text
dense informal settlement
```

from:

```text
dense formal settlement
```

---

## 3. Research Questions

1. How can informal settlements be distinguished from visually similar dense formal neighborhoods in Dhaka?
2. Does language grounding provide useful signal beyond neutral visual features?
3. Can socioeconomic layers reduce false positives in dense formal control regions?

---

## 4. Hypothesis

Optical imagery alone is underdetermined for dense-megacity slum detection. A better model should combine three sources of evidence:

| Signal | What it contributes |
|---|---|
| Optical imagery | Roofs, built-up texture, settlement layout |
| Language grounding | Conceptual distinction between informal and formal dense settlement |
| Socioeconomic layers | Nighttime light, population, access/connectivity, deprivation proxy, built-up extent |

In short:

```text
RGB density alone -> ambiguous
RGB + language + socioeconomic context -> better informal/formal discrimination
```

---

## 5. Contribution, Stated Honestly

The project currently contributes:

1. A Dhaka region benchmark designed around dense-formal confusion, not random negatives.
2. A weak-label creation pipeline using built-up masks and known region type.
3. A cache-aware Colab workflow for expensive VLM feature extraction and SAS-Net clean-tile caching.
4. A two-stage architecture combining Sentinel-2, LocateAnything grounding maps, and socioeconomic cross-attention.
5. A debugging/evaluation framework that detects trivial all-slum and all-background collapse.
6. A report/presentation package documenting both the intended method and what the current experiments actually showed.

The project should not yet be claimed as a validated high-performing slum segmentation model. The current evidence is diagnostic: the pipeline works, major data/metric bugs were found and fixed, and corrected visual-only performance suggests that current weak labels and coarse features are not enough for strong pixel-level claims.

---

## 6. Dataset

### 6.1 Study Area

City: Dhaka, Bangladesh

The benchmark contains 12 regions:

| Informal target regions | Formal dense control regions |
|---|---|
| Korail | Old Dhaka |
| Bhashantek | Gulshan-Baridhara |
| Karail extension | Dhanmondi |
| Kamrangirchar | Uttara |
| Kallyanpur |  |
| Hazaribagh |  |
| Tongi |  |
| Mirpur Beribadh |  |

The formal controls are essential. They are deliberately difficult negative cases: dense areas that should not be classified as slums.

### 6.2 Imagery

- Source: Sentinel-2 Level-2A
- Main bands: B2, B3, B4, B8
- Nominal resolution: 10 m
- Season: dry-season composites
- Years used in configuration: 2020-2023
- Wet-season exports were attempted but are often unreliable because monsoon cloud cover leaves empty or poor composites.

### 6.3 Tiling

- Tile size: 128 px
- Approximate physical size: 1.28 km x 1.28 km
- Total tiles: 720
- Split: stratified by region
- Typical split intent: train/validation/test by manifest, keeping regional coverage balanced.

### 6.4 Label Classes

The label convention is:

| Class | Meaning |
|---|---|
| 0 | Unknown / ignored |
| 1 | Slum / informal built pixel |
| 2 | Formal dense built pixel |

### 6.5 Current Weak-Label Rule

The original design attempted to use a VIIRS nighttime-light threshold to separate slum and formal pixels. That failed in practice: it produced zero slum pixels. This would make training meaningless.

The current rule is region-type weak supervision:

```text
if pixel is built-up and region is informal:
    label = 1
elif pixel is built-up and region is formal control:
    label = 2
else:
    label = 0
```

Built-up evidence comes from GHSL built-up and Dynamic World built class. The high-confidence mask is the built-up portion of known regions.

This solved the zero-slum bug, but it is still weak supervision. It is not manual pixel-level ground truth.

### 6.6 Verified Label Counts

Latest successful verification:

```text
total tiles       : 720
total HC pixels   : 11308540
label px U/slum/F : {0: 490390, 1: 7382955, 2: 3923135}
```

By region:

| Region | Unknown | Slum | Formal |
|---|---:|---:|---:|
| bhashantek | 38,740 | 944,300 | 0 |
| dhanmondi | 2,000 | 0 | 981,040 |
| gulshan_baridhara | 1,000 | 0 | 982,040 |
| hazaribagh | 61,375 | 921,665 | 0 |
| kallyanpur | 4,300 | 978,740 | 0 |
| kamrangirchar | 79,640 | 903,400 | 0 |
| karail_extension | 13,435 | 969,605 | 0 |
| korail | 15,940 | 967,100 | 0 |
| mirpur_beribadh | 228,595 | 754,445 | 0 |
| old_dhaka | 4,685 | 0 | 978,355 |
| tongi | 39,340 | 943,700 | 0 |
| uttara | 1,340 | 0 | 981,700 |

Status:

```text
OK - labels contain slum/formal classes and HC pixels. Safe to run Phase 4.
```

---

## 7. Supporting Geospatial Layers

| Layer | Role | Current status |
|---|---|---|
| Sentinel-2 | Optical imagery | Operational |
| Dynamic World | Built-class evidence | Operational |
| GHSL built-up | Built-up mask/context | Operational |
| WorldPop | Population-density proxy | Operational |
| GHS-POP | Population proxy | Operational |
| VIIRS nighttime light | Electricity/economic-activity proxy | Operational |
| Road/access channel | Connectivity/accessibility proxy | Placeholder/proxy |
| Poverty channel | Deprivation proxy | Zero placeholder |

Important caveat: roads and poverty ablation results should not be scientifically interpreted until the placeholder/proxy layers are replaced with real, validated assets.

---

## 8. Architecture

### 8.1 High-Level Pipeline

```text
Sentinel-2 tile
    -> SAS-Net appearance normalization
    -> clean tile
    -> LocateAnything / MoonViT prompt-grounded feature extraction
    -> cached grounding-map features
    -> socioeconomic tensor alignment
    -> cross-attention fusion
    -> lightweight decoder
    -> slum probability mask
```

### 8.2 Stage 1: SAS-Net

File: `src/models/sasnet.py`

Purpose:

- normalize atmospheric/seasonal appearance differences;
- separate scene structure from appearance statistics;
- cache clean tiles for later experiments.

Observed status:

- SAS-Net training completed.
- Best validation loss from a run reached approximately `0.0071`.
- Clean-tile caching mostly completed.
- One corrupted socioeconomic tile caused a RasterIO failure later, not a SAS-Net model failure.

Known warning:

```text
FutureWarning: torch.cuda.amp.GradScaler(...) is deprecated
```

This is not fatal. It should eventually be changed to `torch.amp.GradScaler('cuda', ...)`.

### 8.3 Stage 2: LocateAnything / MoonViT Features

Files:

- `src/locate_anything/worker.py`
- `src/locate_anything/feature_extractor.py`
- `src/locate_anything/prompts.py`
- `src/locate_anything/_compat.py`

Model:

- `nvidia/LocateAnything-3B`
- Frozen VLM backbone
- MoonViT visual encoder
- Features cached to avoid repeated expensive inference.

Feature mode used:

- grounding-map features
- prompt-specific box coverage maps
- coarse spatial map, typically represented as 32 x 32 coverage.

Prompt roles:

| Prompt type | Purpose |
|---|---|
| Neutral dense-built prompt | Visual-only baseline |
| Slum/informal prompt | Ask for informal-settlement concept |
| Formal dense prompt | Ask for dense but formal urban concept |

Important caveat: LocateAnything is designed for visual grounding, but Sentinel-2 at 10 m is coarse. Many informal-settlement cues are below that resolution.

### 8.4 Fusion Module

File: `src/models/fusion.py`

The fusion design:

```text
F = V + CrossAttention(Q=V, K=E, V=E)
```

where:

- `V` = visual/VLM feature map;
- `E` = socioeconomic tensor;
- `F` = fused representation.

The design lets visual features query socioeconomic context. Socioeconomic channels can be masked for ablations.

### 8.5 Decoder

File: `src/models/decoder.py`

Purpose:

- upsample fused features;
- output a per-pixel slum probability mask.

### 8.6 Model Configurations

File: `src/models/banglaslumnet.py`

| Config | Visual input | Language | Socioeconomic fusion | Purpose |
|---|---|---|---|---|
| `baseline_cnn` | RGB/S2 features | No | No | Optical-only baseline |
| `vlm_visual` | LocateAnything neutral features | Neutral prompt | No | VLM visual baseline |
| `vlm_lang` | LocateAnything prompt features | Slum/formal prompts | No | Language ablation |
| `full` | LocateAnything prompt features | Slum/formal prompts | Yes | Full BanglaSlumNet |

---

## 9. Codebase Structure

```text
BanglaSlumNet/
├── AGENT_HANDOFF.md
├── BanglaSlumNetV4.md
├── config/
│   ├── default.yaml
│   ├── regions_dhaka.yaml
│   └── experiments.yaml
├── gee/
│   ├── export_s2_composites.py
│   ├── export_weak_labels.py
│   ├── export_socioeconomic.py
│   └── ee_export_utils.py
├── notebooks/
│   └── BanglaSlumNet_Colab.ipynb
├── src/
│   ├── data/
│   ├── eval/
│   ├── locate_anything/
│   ├── models/
│   ├── tracking/
│   ├── train/
│   └── viz/
├── scripts/
│   ├── download_models.py
│   └── make_paper_figures.py
├── report/
│   ├── main.tex
│   ├── references.bib
│   ├── diagram_instructions.md
│   └── figures/
├── slide_content.md
└── presentation_script.md
```

Drive layout in Colab:

```text
/gdrive/MyDrive/BanglaSlumNet/
├── data/
│   ├── tiles/
│   ├── labels/
│   ├── socioeconomic/
│   └── features_cache/
├── results/
│   ├── runs/
│   ├── figures/
│   └── tables/
└── model_cache/
```

Important workflow rule:

- Code should be cloned fresh to `/content/BanglaSlumNet`.
- Data/results/model cache should live on Google Drive.
- Do not clone the repo onto Drive.

---

## 10. Notebook Phases

Main notebook: `notebooks/BanglaSlumNet_Colab.ipynb`

Every code cell now has a preceding markdown/text cell explaining:

- what the next cell is supposed to do;
- what result was found;
- what errors were faced;
- what changes were made to fix those errors.

### P0 - Setup

Purpose:

- mount Drive;
- clone code to `/content/BanglaSlumNet`;
- install dependencies;
- set working directory and `sys.path`;
- load config.

Important: always run P0.1-P0.5 in a fresh Colab session.

### P1 - Export and Tiling

Purpose:

- optionally reset labels/results;
- export Sentinel-2, weak labels, and socioeconomic rasters;
- tile all aligned rasters into 128 px samples;
- build manifest.

Important: if labels are stale, run P1.0 with `RESET_LABELS=True` once, then set it back to `False`.

### P2 - Weak Labels and Confidence

Purpose:

- ingest weak labels;
- optionally run LocateAnything validation;
- generate confidence/data-audit artifacts.

Current setting:

```yaml
use_locate_anything_validation: false
```

Reason: LA validation on 10 m Sentinel-2 zeroed useful high-confidence labels. LA is still used for features, not for label validation.

### P3.1 - VLM Feature Caching

Purpose:

- extract LocateAnything/MoonViT features;
- save them to Drive;
- skip on rerun if cached.

Status:

- Cached.

### P3.2 - SAS-Net Training and Clean-Tile Caching

Purpose:

- train SAS-Net;
- generate clean tiles;
- cache outputs.

Important Colab change:

- use `num_workers=0` to avoid multiprocessing worker cleanup spam/errors in Colab.

Observed warnings:

- `numpy RuntimeWarning: invalid value encountered in subtract`
- repeated warning came from percentile/stat operations, not necessarily fatal.

Corrupted tile found:

```text
tongi_2021_dry_000_002_socioec.tif
MissingRequired: TIFF directory is missing required "StripOffsets" field
```

Action taken:

```text
bad socioec tile files: 1
deleted corrupted socioec tile files: 1
```

### P4.1 - Registry

Purpose:

- load experiment matrix;
- initialize run registry;
- skip completed runs and resume interrupted runs.

### P4.1b - Verify Labels Gate

Purpose:

- verify total tiles;
- verify HC pixels;
- verify slum/formal/unknown pixel counts;
- fail before GPU training if labels are empty or degenerate.

Latest status:

```text
OK - labels contain slum/formal classes and HC pixels. Safe to run Phase 4.
```

### P4.2 - Experiment Matrix

Purpose:

- train segmentation heads over cached features;
- evaluate each run;
- write run folders and result CSVs.

Current safe overnight run was restricted to Exp2 central ablation rows to avoid wasting CU on unfinished orchestration.

### P4.3 - Optional GRAM / Baseline Comparison

Purpose:

- compare against external or optical baseline where available.

Status:

- optional, not finalized as a trusted comparison.

### P5 - Figures and Tables

Purpose:

- regenerate paper figures;
- write result tables;
- summarize headline metrics.

Status:

- works structurally, but final scientific figures depend on trustworthy Phase 4 outputs.

---

## 11. Experiments

Experiment matrix file: `config/experiments.yaml`

### Exp 1 - Atmospheric / SAS-Net Ablation

Question:

```text
Does SAS-Net appearance normalization improve performance over raw or classical correction?
```

Rows:

- raw S2
- classical atmospheric correction / histogram or DOS-style correction
- SAS-Net normalized tile

Current caveat:

- The generic loop was not yet fully validated as a true input switch for raw/classical/SAS-Net comparisons.
- Do not overclaim Exp1 until the orchestration is checked.

### Exp 2 - Central Fusion Ablation

Question:

```text
Do language and socioeconomic channels improve discrimination beyond visual features?
```

Rows:

| Row | Meaning |
|---|---|
| `exp2_visual_only` | VLM visual features only, no language, no socioeconomics |
| `exp2_vlm_lang` | Add slum/formal language concept prompts |
| `exp2_viirs_only` | Add nighttime light |
| `exp2_viirs_pop` | Add VIIRS + population |
| `exp2_viirs_pop_roads` | Add road/access proxy |
| `exp2_viirs_pop_roads_poverty` | Add poverty proxy |
| `exp2_full` | Full language + socioeconomic model |
| `exp2_baseline_cnn` | Optical-only CNN baseline |

This is the central experiment for the paper.

### Exp 3 - Leave-One-Region-Out

Question:

```text
Does the model generalize to held-out Dhaka regions?
```

Current caveat:

- True LORO requires dedicated split handling.
- Do not report LORO until held-out regions are truly excluded from training.

---

## 12. Metrics

Important metrics:

| Metric | Why it matters |
|---|---|
| HC-IoU | Segmentation overlap on high-confidence pixels |
| Precision | How many predicted slum pixels are actually slum-labeled |
| Recall | How many slum-labeled pixels were found |
| F1 | Balance between precision and recall |
| Specificity | Ability to reject formal/negative pixels |
| Balanced accuracy | Average of positive and negative class performance |
| Predicted-positive rate | Detects all-slum or all-background collapse |
| Target-positive rate | Shows label class balance |
| FPR on formal controls | Core failure metric for dense formal false positives |

Key lesson:

HC-IoU alone was not enough. It made an all-slum predictor look deceptively decent because slum pixels dominate the weak labels.

---

## 13. Major Bugs and Fixes

### 13.1 Zero-Slum Label Bug

Symptom:

- labels had formal pixels and HC pixels but zero slum pixels.

Root cause:

- VIIRS threshold weak-label rule did not separate classes as expected.

Fix:

- switched to region-type weak labels.

Verification:

- slum and formal labels now both present.

### 13.2 Misleading All-Slum HC-IoU

Symptom:

- early visual-only runs produced HC-IoU around 0.66.
- recall was 1.0.
- precision approximately matched slum prevalence.

Interpretation:

- model predicted slum almost everywhere.
- apparent score was a class-prior artifact.

Fixes:

- added `pred_pos_rate`;
- added `target_pos_rate`;
- added specificity and balanced accuracy;
- changed early stopping metric to balanced accuracy;
- used balanced BCE.

### 13.3 All-Background Collapse

Symptom:

- several configs produced all-zero metrics.

Interpretation:

- model predicted no slum or training/eval was misconfigured.

Fixes:

- balanced loss;
- HC/known-pixel masking;
- better diagnostics.

### 13.4 Missing Formal-Control FPR

Symptom:

- FPR fields were `nan`.

Root cause:

- evaluation did not construct correct formal-control masks.

Fix:

- added formal-control evaluation masks for Old Dhaka, Gulshan-Baridhara, Dhanmondi, and Uttara.

### 13.5 Socioeconomic Rows Misconfigured

Symptom:

- some socioeconomic ablation rows were configured as `vlm_lang`, so fusion was not actually active.

Fix:

- corrected `exp2_viirs_*` rows to use `model.config: full`.

### 13.6 Colab Multiprocessing Assertion

Symptom:

```text
AssertionError: can only test a child process
```

Context:

- appeared around DataLoader worker cleanup.

Fix:

- set `num_workers=0` for Colab training/evaluation loops.

### 13.7 Corrupted Socioeconomic Tile

Symptom:

```text
RasterioIOError: tongi_2021_dry_000_002_socioec.tif:
MissingRequired: TIFF directory is missing required "StripOffsets" field
```

Fix:

- detected and deleted one corrupted socioec tile file.
- rerun/rebuild allowed caching to continue.

---

## 14. Training and Result Status

### 14.1 SAS-Net

Observed successful training:

```text
[SASNet] Epoch 50/50 | train=0.0125 val=0.0077
SAS-Net training done. Best val loss: 0.0071 -> results/runs/sasnet_best.pt
```

Interpretation:

- SAS-Net training completed.
- Warnings did not stop training.
- The corrupted socioeconomic tile affected clean-tile caching, not the SAS-Net model itself.

### 14.2 Early Phase 4, Before Fixes

Observed pattern:

```text
HC-IoU around 0.6657
recall = 1.0
precision around slum prevalence
```

Interpretation:

- all-slum collapse.
- not valid final performance.

Other rows:

- some produced 0.0 metrics.
- likely all-background collapse or misconfigured rows.

### 14.3 Corrected Phase 4 Diagnostic

Latest corrected visual-only screenshot/run:

```text
exp2_visual_only
val_balanced_accuracy around 0.50
HC-IoU around 0.4647
pred_pos no longer stuck at exactly 0 or 1
```

Interpretation:

- collapse diagnostics are now working.
- visual-only signal is weak under current labels/features.
- the lower score is more honest than the earlier fake 0.66.

Important statement:

```text
The codebase is now in a better debugging state, but the current data and supervision may be too weak for strong per-pixel segmentation claims.
```

---

## 15. What the Results Mean So Far

The project has not failed. It has clarified the actual bottleneck.

What we learned:

1. The pipeline can run end-to-end.
2. The dataset now contains both slum and formal labels.
3. Cached VLM features and SAS-Net outputs allow feasible experimentation.
4. Early metrics were misleading because trivial predictors were not detected.
5. Corrected diagnostics show that current visual-only segmentation is weak.
6. Region-type weak labels may be too coarse for pixel-level claims.
7. LocateAnything grounding on 10 m Sentinel-2 may be too coarse for fine settlement morphology.

The scientific direction is still valid, but the next experiment should be designed carefully rather than simply spending more CU on the same weak setup.

---

## 16. Recommended Next Scientific Changes

### 16.1 Manual Evaluation Subset

Create a small manually checked evaluation set:

- select representative tiles from informal and formal regions;
- annotate coarse slum/formal polygons or built-pixel masks;
- include difficult dense formal controls;
- keep this set held out from all training.

This is the most important next step for defensible claims.

### 16.2 Tile-Level Classification Baseline

Because labels are region-level, a tile-level formulation may be more honest:

```text
tile -> informal region / formal dense control
```

If tile classification works while pixel segmentation remains weak, the paper can be reframed as screening or region-level detection instead of pixel-level mapping.

### 16.3 High-Resolution VLM Grounding

Use higher-resolution imagery for LocateAnything grounding if possible:

- ESRI z16 or similar;
- then align/collapse features to Sentinel-2 tiles;
- use Sentinel-2 for scalable context, but high-res imagery for fine visual concepts.

### 16.4 Replace Placeholder Socioeconomic Layers

Before claiming roads/poverty effects:

- replace road/access proxy with validated roads/accessibility data;
- replace zero poverty placeholder with real poverty/deprivation proxy.

### 16.5 Dedicated Exp1 and Exp3 Orchestration

Implement explicitly:

- raw vs classical vs SAS-Net input switching for Exp1;
- true held-out region exclusion for LORO Exp3.

---

## 17. Report and Presentation Artifacts

### 17.1 Report

Folder:

```text
report/
```

Important files:

| File | Purpose |
|---|---|
| `report/main.tex` | Overleaf-ready LaTeX report |
| `report/references.bib` | Bibliography |
| `report/diagram_instructions.md` | Designer instructions for diagrams |
| `report/README.md` | Upload/compile notes |
| `BanglaSlumNet_report_overleaf.zip` | Direct Overleaf upload package |

Figures currently included:

| Figure | Status |
|---|---|
| `dataset_samples.png` | Representative Sentinel-2 sample crops |
| `dhaka_composite_overview.png` | Dhaka composite overview |
| `confidence_strata.png` | Replaced with weak-label composition chart |

Designer placeholder diagrams still expected:

- `figures/system_architecture.pdf`
- `figures/compute_pipeline.pdf`
- optionally `figures/weak_label_process.pdf`
- optionally `figures/experiment_debugging.pdf`

### 17.2 Presentation

Files:

| File | Purpose |
|---|---|
| `slide_content.md` | Expanded 20-slide content and flow |
| `presentation_script.md` | Script for 21 slides, 5 lines per slide, technical explanations |

Speaker assignment:

| Slides | Speaker |
|---|---|
| 1-4 | Nafiz |
| 5-8 | Zayan |
| 9-12 | Mizi |
| 13-17 | Fardeen |
| 18-21 | Namare |

Slides 14 onward explain experiment phases, current findings, why we stopped, how the codebase works, and what changes are needed.

---

## 18. Literature Covered in the Report

The report includes a focused literature set covering:

- slum/informal-settlement mapping reviews;
- object-based and deep-learning methods;
- Sentinel-2 and Google Earth Engine workflows;
- uncertainty-aware informal-settlement mapping;
- U-Net and SegFormer segmentation architectures;
- LocateAnything / visual grounding;
- nighttime lights and poverty estimation;
- Dynamic World;
- attention and AdaIN-style appearance normalization.

Important cited themes:

| Theme | Why it matters |
|---|---|
| Informal settlement morphology | Shows why density/irregularity are useful but not universal |
| Deep segmentation | Provides baseline architecture context |
| Foundation/VLM grounding | Motivates language-conditioned features |
| Socioeconomic proxies | Motivates VIIRS/population/access layers |
| Uncertainty and interpretability | Motivates collapse diagnostics and cautious claims |

---

## 19. How to Run Safely in Colab

Recommended Colab sequence:

1. Runtime restart.
2. Open notebook from GitHub, not from a stale upload.
3. Run P0.1-P0.5.
4. If labels need refresh:
   - run P1.0 with `RESET_LABELS=True`;
   - set `RESET_LABELS=False`;
   - run P1.4 and P1.5.
5. Run P2.1-P2.3.
6. Run P4.1b before any GPU-heavy Phase 4 training.
7. Only run Phase 4 if P4.1b says labels contain slum/formal classes and HC pixels.
8. Prefer the restricted Exp2 central matrix until Exp1/LORO are separately validated.
9. Watch:
   - `val_balanced_accuracy`;
   - `hc_iou`;
   - `pred_pos`;
   - formal-control FPR.
10. Stop if `pred_pos` is stuck near 0 or 1 for many epochs.

---

## 20. Current Git / Artifact Notes

Recent important commits:

| Commit | Meaning |
|---|---|
| `79c0769` | Strengthened label verification gate |
| `6b515df` | Resume SAS-Net clean-tile caching |
| `2e2f0ae` | Prevent class-prior collapse in Phase 4 |
| `f171210` | Limit overnight run to central ablation |
| `a3e49fc` | Annotate Colab experiment phases |
| `dbe46ad` | Add Overleaf-ready report |
| `0ece736` | Replace empty audit figure |

Local note:

- `BanglaSlumNet_report_overleaf/` may exist as an extracted copy of the zip and can remain untracked.

---

## 21. What Not To Claim Yet

Do not claim:

- final state-of-the-art performance;
- validated pixel-level slum segmentation;
- proven socioeconomic improvement;
- proven roads/poverty contribution;
- trustworthy LORO generalization;
- trustworthy SAS-Net ablation improvement;
- LocateAnything works well on 10 m imagery.

Safe claims:

- the pipeline is implemented end-to-end;
- Dhaka dense-formal confusion is a real and important evaluation target;
- the weak labels now contain both slum and formal classes;
- early metrics exposed trivial collapse;
- corrected diagnostics made the results more honest;
- current evidence suggests stronger supervision/high-res grounding or tile-level framing is needed.

---

## 22. Best Current Paper Framing

The most defensible framing is:

```text
BanglaSlumNet is a reproducible prototype and dataset-construction effort for testing language-grounded socioeconomic fusion in Dhaka informal-settlement detection.
```

Avoid framing it as:

```text
BanglaSlumNet solves slum segmentation in Dhaka.
```

Recommended conclusion:

```text
The work built an end-to-end, compute-aware pipeline and exposed the main scientific bottleneck: region-level weak labels and medium-resolution imagery are not yet enough for strong pixel-level claims. The next step is a manually verified evaluation subset, tile-level baseline, validated socioeconomic layers, and high-resolution VLM grounding.
```

---

## 23. Immediate Next Steps

Highest priority:

1. Build a manual evaluation subset.
2. Add a tile-level classification baseline.
3. Replace road and poverty placeholders.
4. Validate true Exp1 and Exp3 orchestration.
5. Try high-resolution VLM grounding.
6. Then rerun the central Exp2 matrix and report only corrected metrics.

For tomorrow's presentation/report, the honest message is:

```text
We built the full pipeline, found and fixed serious label/metric bugs, and learned that the current setup is diagnostic rather than final. The strongest next step is improving supervision and evaluation.
```

