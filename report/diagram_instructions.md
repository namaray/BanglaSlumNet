# Diagram Instructions for the BanglaSlumNet Report

Place final diagram files in `report/figures/` using the exact filenames below. Export each diagram as vector PDF for the report and optionally PNG for slides.

## 1. `system_architecture.pdf`

**Purpose:** Main methodology figure.

**Canvas:** Landscape, approximately 16:9 or a wide journal figure.

**Flow:**

1. Sentinel-2 tile, labeled `B2 / B3 / B4 / B8`.
2. SAS-Net block:
   - Scene encoder.
   - Appearance encoder.
   - Reference-appearance renderer.
   - Output labeled `clean tile`.
3. LocateAnything branch:
   - Clean tile plus two prompt boxes.
   - Prompt A: `dense informal settlement`.
   - Prompt B: `dense formal housing`.
   - Frozen MoonViT / LocateAnything block.
   - Two small grounding heatmaps.
   - Concatenate into visual feature map `V`.
4. Socioeconomic branch:
   - VIIRS nightlight.
   - WorldPop / GHS-POP.
   - Roads/access.
   - Poverty.
   - GHSL built-up.
   - Stack into tensor `E`.
5. Fusion block:
   - `CrossAttention(Q=V, K=E, V=E) + V`.
6. Lightweight decoder.
7. Output slum-probability mask.

**Visual conventions:**

- Optical imagery: natural-color thumbnail.
- Language features: teal.
- Socioeconomic layers: use several distinct colors.
- Trainable modules: solid outline.
- Frozen/cached modules: dashed outline with snowflake/cache symbol.
- Do not imply that the current output has been scientifically validated.

## 2. `compute_pipeline.pdf`

**Purpose:** Explain reproducibility and compute saving.

**Flow:**

`GitHub code -> /content/BanglaSlumNet`

Parallel persistent branch:

`Google Drive -> data / model_cache / features_cache / results`

Then:

`P0 Setup -> P1 GEE exports -> P2 labels -> P3 VLM + SAS-Net cache -> P4 registry experiments -> P5 figures/tables`

**Callouts:**

- `Run VLM once`.
- `Cache expensive artifacts`.
- `Registry resumes incomplete runs`.
- `CPU preflight before GPU spend`.

**Visual conventions:** Quiet technical pipeline, not a marketing graphic.

## 3. Optional `weak_label_process.pdf`

**Purpose:** Explain the current annotation rule and its limitation.

**Inputs:**

- Region boundary marked informal or formal control.
- GHSL built-up mask.
- Dynamic World built class.

**Rule:**

- Informal region AND built -> Class 1.
- Formal region AND built -> Class 2.
- Non-built -> Class 0 / ignored.

**Required warning box:**

`Weak region-type supervision: not manual pixel-level ground truth.`

## 4. Optional `experiment_debugging.pdf`

**Purpose:** Show why the initial result was rejected.

**Three panels:**

1. `Initial labels`: VIIRS rule -> zero slum pixels.
2. `Initial Phase 4`: apparent HC-IoU around 0.66 -> recall 1.0 -> all-slum collapse.
3. `Corrected diagnostics`: balanced BCE + balanced accuracy + predicted-positive rate -> honest visual-only result near random balance.

**Tone:** Scientific debugging timeline. Avoid presenting the corrected diagnostic result as final model performance.

## 5. Dataset Sample Replacement

The included `dataset_samples.png` contains six crops from the only Dhaka GeoTIFF available locally. Before final submission, replace it with six verified samples:

- Three informal-region tiles.
- Three formal-dense control tiles.
- Use the same rendering stretch for all samples.
- Add region name, year, and class beneath each image.
- Prefer examples from the held-out test set.

