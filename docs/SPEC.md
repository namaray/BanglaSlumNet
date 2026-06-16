# BanglaSlumNet × LocateAnything — Implementation Specification

**Version:** 1.0 (engineering spec for Claude Code) **Repo:** [https://github.com/namaray/BanglaSlumNet.git](https://github.com/namaray/BanglaSlumNet.git) **Target compute:** Google Colab Pro+ (A100 40GB / L4 / T4 fallback) **Scope:** Dhaka only (national mapping deferred; see §12) **Status:** This document is the build contract. Claude Code consumes this to scaffold the repo, write the Colab notebook, and produce all training/eval/ablation/plotting code.

---

## 0\. How to read this document

This spec is written for two audiences in sequence:

1. **Claude Code**, which will scaffold the repository, create the Colab notebook(s), and write every code file referenced here. Each module below has a contract: inputs, outputs, file path, and acceptance criteria.  
2. **The research team** (you), who will run the notebook in Colab, inspect the tracked results, regenerate the paper figures, and write the manuscript from the recorded numbers.

Wherever this spec says **\[CODE\]**, Claude Code must produce a runnable artifact at the stated path. Wherever it says **\[RUN\]**, the team executes a notebook cell. Wherever it says **\[DECISION\]**, a human chooses a branch (and the notebook should expose that as a config flag, not a code edit).

The guiding constraint throughout: **be frugal with compute units (CUs).** Every design choice below is biased toward the cheapest configuration that still produces a defensible result. Expensive options are flagged and gated behind config switches that default to OFF.

---

## 1\. Executive summary of the technical plan

### 1.1 What the original proposal established

BanglaSlumNet v3 already proved, with a real measured experiment, that **GRAM (AAAI'26) cannot separate Dhaka's dense formal core from real slums** — zero-shot mean slum probability is statistically indistinguishable across Korail (0.48, real slum), Mirpur (0.51, mixed), and Old Dhaka (0.44, formal-dense). The mechanistic attribution is a dataset bias: GRAM's 12 training cities all exhibit a sprawling-slum-vs-low-density-formal contrast that Dhaka violates, so GRAM collapses into a dense-built-up detector.

The proposed fix is two-stage: **Stage 1** atmospheric disentanglement (SAS-Net) and **Stage 2** socioeconomic cross-attention fusion that injects a modality orthogonal to satellite texture (nighttime lights, population, roads, poverty).

### 1.2 What LocateAnything changes

We adopt **directions A \+ B** from the design discussion:

- **Direction B (weak-label enhancement):** Use LocateAnything-3B's zero-shot vision-language grounding to *visually validate* the OSM+GHSL+VIIRS weak labels. A tile labeled "slum" by the geospatial fusion is only promoted to high-confidence if LocateAnything's visual grounding also flags an informal-settlement signature. This produces a cleaner training set and a fourth, independent signal in the confidence stratification.  
    
- **Direction A (VLM backbone for the main model):** Replace the generic CNN/UNet optical encoder in Stage 2 with **LocateAnything's frozen MoonViT vision encoder \+ language-conditioned grounding features**. We prompt the VLM with the *concept* the model needs ("dense informal settlement with narrow unpaved lanes" vs. "dense formal masonry housing with road access"), extract its visual-grounding feature maps, and fuse those with the socioeconomic tensor via cross-attention. The language prompt explicitly encodes the discrimination that GRAM lacks.

### 1.3 The one-sentence contribution

*We show that a frontier visual-grounding VLM (LocateAnything), prompted with the conceptual distinction between informal and dense-formal settlement and fused with socioeconomic priors, corrects the dense-megacity failure mode that defeats optical-only foundation slum detectors on Dhaka.*

### 1.4 The narrative spine of the paper

1. **Failure mode** (already measured): GRAM sees slums everywhere in dense Dhaka.  
2. **Why optical-only cannot fix it**: visual texture is genuinely ambiguous; the disambiguating signal must come from outside the RGB channel.  
3. **Two orthogonal external signals**: (a) *language* — the VLM is told what a slum *is*, conceptually; (b) *socioeconomics* — nighttime darkness, population, roads, poverty.  
4. **Result**: precision on formal-dense control regions (Old Dhaka, Gulshan) rises sharply without sacrificing recall on real slums (Korail).  
5. **Ablation** isolates the contribution of each signal.

---

## 2\. Critical constraints and decisions locked in

### 2.1 Licensing (must appear in the paper and the repo README)

LocateAnything-3B is released under the **NVIDIA License: non-commercial, academic/non-profit research use only.** Commercial use is prohibited. Its components carry the **Qwen Research License** (Qwen2.5-3B-Instruct) and **MIT** (MoonViT-SO-400M). Implications:

- This is fine for a CVPR/ICCV/TGRS academic submission.  
- Any released weights derived from LocateAnything, and any released Dhaka map produced with it, must be tagged **research-only** and must retain NVIDIA's license and attribution notices.  
- The repo README must carry a `THIRD_PARTY_LICENSES.md` listing NVIDIA, Qwen, and MoonViT terms.  
- **\[DECISION\]** If the team wants a redistributable map free of NVIDIA license entailment, produce the *final national map* (deferred, §12) with the BanglaSlumNet-trained segmentation head on Sentinel-2 only, treating LocateAnything as a frozen feature extractor used during research — and document this carefully. For the Dhaka paper, research-only is acceptable.

### 2.2 Model facts (verified from the HF model card, 26 May 2026\)

| Property | Value |
| :---- | :---- |
| Model ID | `nvidia/LocateAnything-3B` |
| Params | \~4B (3B LM \+ ViT \+ projector) |
| Vision encoder | MoonViT-SO-400M (native resolution) |
| Language model | Qwen2.5-3B-Instruct |
| Precision | BF16 |
| Load | `AutoModel.from_pretrained(..., trust_remote_code=True)` |
| Coordinate space | normalized integers `[0,1000]`, output as `<box><x1><y1><x2><y2></box>` |
| Generation modes | `fast` (MTP), `slow` (NTP/AR), `hybrid` (default) |
| Supported GPUs | Ampere (A100), Hopper, Lovelace (L40/4090), Blackwell |
| Deps | `transformers==4.57.1`, `opencv-python-headless==4.11.0.86`, `numpy==1.25.0`, `Pillow==11.1.0`, `peft`, `torchvision`, `decord==0.6.0`, `lmdb==1.7.5` |
| MagiAttention | Optional, Hopper/Blackwell only — **NOT available on Colab A100 (Ampere)**, falls back to PyTorch SDPA automatically. Do not attempt to install it on Colab. |

**Colab reality:** A100 is Ampere → MagiAttention unavailable → MTP runs via SDPA fallback (functional, slower). For our use (feature extraction \+ LoRA fine-tune), this is fine. We never need LocateAnything's fast parallel *decoding* speed because we are not generating long box sequences in production — we extract features and/or run a handful of grounding prompts per tile.

### 2.3 Why we do NOT fully fine-tune the 4B VLM

Full fine-tuning of a 4B VLM on a 40GB A100 is infeasible without sharding/offload, and it would burn CUs catastrophically. Decisions:

- **Vision encoder (MoonViT): frozen.** We extract features once and cache them.  
- **Adaptation: LoRA** on the projector \+ a small number of LM layers only when we run the language-conditioning path, and even then **default OFF** (zero-shot prompting is the cheap baseline; LoRA is the upgrade gated behind a flag).  
- **Trainable parameters live in our own heads**: the cross-attention fusion module and the lightweight segmentation decoder. These are small (a few M params) and train fast.

This keeps every training run inside a single A100 session.

### 2.4 Scope lock

- **In scope:** Dhaka five-region benchmark (3 informal: Korail, Bhashantek, Karail-extension; 2 formal-dense control: Old Dhaka brick core, Gulshan-2/Baridhara), weak-label pipeline, LocateAnything weak-label validation, SAS-Net Stage 1, LocateAnything+socioeconomic fusion Stage 2, Experiments 1–3, full ablation.  
- **Deferred (§12):** national 64-district 10-year map, Experiments 4 (temporal) and 5 (national). These remain in the paper as "downstream application / future work" unless the team explicitly re-scopes.

---

## 3\. System architecture

### 3.1 End-to-end pipeline diagram (textual)

                          ┌─────────────────────────────────────────┐

                          │   DATA LAYER  (Google Earth Engine)       │

                          │   • Sentinel-2 seasonal composites        │

                          │   • ESRI z16 tiles (GRAM-baseline only)   │

                          │   • OSM / GHSL / VIIRS / WorldPop / WB     │

                          └───────────────┬───────────────────────────┘

                                          │  export GeoTIFF \+ PNG tiles

                                          ▼

        ┌──────────────────────────────────────────────────────────────────┐

        │  WEAK-LABEL PIPELINE  (§5)                                          │

        │  OSM∩GHSL∩VIIRS  →  slum / formal-dense / unknown                  │

        │           \+                                                         │

        │  LocateAnything-3B zero-shot grounding validation  (Direction B)   │

        │           ↓                                                         │

        │  4-signal confidence score → High-Confidence (HC) eval subset      │

        │                            → Noisy training labels                  │

        └───────────────┬────────────────────────────────────────────────────┘

                        │

       ┌────────────────┴───────────────┐

       ▼                                ▼

┌──────────────┐                ┌────────────────────────────────────────┐

│  STAGE 1      │                │  SOCIOECONOMIC TENSOR  E                 │

│  SAS-Net      │                │  \[VIIRS, WorldPop, GHS-POP, OSM-roads,   │

│  atmospheric  │                │   WB-poverty, GHSL-builtup\]  (C\_e chans) │

│  disentangle  │                │  resampled to tile grid, normalized      │

│  → clean S2   │                └───────────────┬──────────────────────────┘

└──────┬───────┘                                │

       │ clean RGB tile                          │

       ▼                                         │

┌─────────────────────────────────────────┐     │

│  STAGE 2  VLM BACKBONE  (Direction A)     │     │

│  LocateAnything MoonViT (frozen)          │     │

│  \+ language prompt conditioning           │     │

│  → visual-grounding feature map  V        │     │

└──────────────────┬────────────────────────┘     │

                   ▼                               ▼

        ┌──────────────────────────────────────────────────┐

        │  CROSS-ATTENTION FUSION                            │

        │  F \= CrossAttn(Q=V, K=E, V=E) \+ V   (residual)     │

        └───────────────────┬────────────────────────────────┘

                            ▼

        ┌──────────────────────────────────────────────────┐

        │  LIGHTWEIGHT SEGMENTATION DECODER (UNet-style)     │

        │  → binary slum mask (per-pixel)                    │

        └───────────────────┬────────────────────────────────┘

                            ▼

            HC-IoU / All-IoU / Precision / Recall / F1 / FPR-on-control

### 3.2 The three model configurations the code must support (config-flagged)

| Config | Stage 1 | Stage 2 backbone | Language prompt | Socioeconomic fusion | Purpose |
| :---- | :---- | :---- | :---- | :---- | :---- |
| `baseline_cnn` | off | ResNet/SegFormer encoder | n/a | off | Reproduce optical-only failure (our GRAM analogue, fully controlled) |
| `vlm_visual` | on | LocateAnything MoonViT (frozen) | generic ("locate dense built-up") | off | VLM features, no concept, no socioeconomics |
| `vlm_lang` | on | LocateAnything MoonViT (frozen) | discriminative ("informal vs formal-dense") | off | \+ language concept |
| `full` | on | LocateAnything MoonViT (frozen) | discriminative | **on (all channels)** | Full BanglaSlumNet |

Plus per-channel fusion ablations under `full` (see §9).

### 3.3 Why this resolves the failure mode (paper argument)

The failure is that RGB texture is insufficient. We add **two orthogonal signals**:

1. **Language conditioning** turns the VLM from a "what is here" describer into a "is this the informal concept" discriminator. The prompt encodes domain knowledge (unpaved narrow lanes, irregular roofs, no setbacks) that the purely visual GRAM never had.  
2. **Socioeconomic cross-attention** injects nighttime darkness (the single strongest proxy per the proposal), population density, road structure, and poverty — none of which appear in RGB.

When visual texture collapses (Old Dhaka vs Korail), the fused representation still separates them because the *non-visual* signals differ: Gulshan is bright at night and road-connected; Korail is dark and road-poor.

---

## 4\. Repository structure

Claude Code creates exactly this layout. **\[CODE\]** for every file marked.

BanglaSlumNet/

├── README.md                         \[CODE\] project overview, license notices, quickstart

├── THIRD\_PARTY\_LICENSES.md           \[CODE\] NVIDIA \+ Qwen \+ MoonViT terms

├── requirements.txt                  \[CODE\] pinned deps (matches §2.2 \+ ours)

├── requirements\_colab.txt            \[CODE\] Colab-specific install order

├── config/

│   ├── default.yaml                  \[CODE\] master config (paths, hyperparams, flags)

│   ├── regions\_dhaka.yaml            \[CODE\] 5 region bounding boxes \+ metadata

│   └── experiments.yaml              \[CODE\] experiment matrix (Exp 1/2/3, ablations)

├── gee/

│   ├── 01\_export\_s2\_composites.js    \[CODE\] GEE: Sentinel-2 seasonal composites

│   ├── 02\_export\_esri\_tiles.py       \[CODE\] ESRI z16 tiles for GRAM baseline only

│   ├── 03\_weak\_labels.js             \[CODE\] GEE: OSM∩GHSL∩VIIRS fusion \+ HC mask export

│   └── 04\_export\_socioeconomic.js    \[CODE\] GEE: VIIRS/WorldPop/GHS-POP/WB/GHSL layers

├── src/

│   ├── \_\_init\_\_.py

│   ├── data/

│   │   ├── tiles.py                  \[CODE\] tile dataset, alignment, normalization

│   │   ├── weak\_labels.py            \[CODE\] label fusion \+ confidence stratification

│   │   ├── socioeconomic.py          \[CODE\] load/resample/normalize E tensor

│   │   └── augment.py                \[CODE\] geometric/photometric augmentation

│   ├── locate\_anything/

│   │   ├── worker.py                 \[CODE\] LocateAnythingWorker (from HF card, adapted)

│   │   ├── feature\_extractor.py      \[CODE\] frozen MoonViT feature extraction \+ cache

│   │   ├── label\_validator.py        \[CODE\] Direction B: zero-shot grounding validation

│   │   └── prompts.py                \[CODE\] prompt templates for slum discrimination

│   ├── models/

│   │   ├── sasnet.py                 \[CODE\] Stage 1 atmospheric disentanglement

│   │   ├── fusion.py                 \[CODE\] cross-attention fusion module

│   │   ├── decoder.py                \[CODE\] lightweight UNet segmentation head

│   │   ├── baseline\_cnn.py           \[CODE\] SegFormer/ResNet optical-only baseline

│   │   └── banglaslumnet.py          \[CODE\] assembles configs from §3.2

│   ├── train/

│   │   ├── train\_sasnet.py           \[CODE\] Stage 1 trainer

│   │   ├── train\_segmenter.py        \[CODE\] Stage 2 \+ fusion \+ decoder trainer

│   │   └── losses.py                 \[CODE\] seg losses \+ SAS-Net consistency loss

│   ├── eval/

│   │   ├── metrics.py                \[CODE\] HC-IoU, All-IoU, P/R/F1, FPR-control, SSIM/PSNR

│   │   ├── evaluate.py               \[CODE\] run a config on a split, dump results JSON

│   │   └── gram\_baseline.py          \[CODE\] wrap existing GRAM run for head-to-head

│   ├── tracking/

│   │   ├── recorder.py               \[CODE\] structured results logging (JSON+CSV)

│   │   └── registry.py               \[CODE\] experiment run registry (resume-safe)

│   └── viz/

│       ├── plots.py                  \[CODE\] all matplotlib paper figures

│       ├── palette.py                \[CODE\] consistent color palette (navy/teal/steel/slate)

│       └── qualitative.py            \[CODE\] side-by-side prediction overlays

├── notebooks/

│   ├── BanglaSlumNet\_Colab.ipynb     \[CODE\] THE master notebook (end-to-end, CU-aware)

│   └── 00\_smoke\_test.ipynb           \[CODE\] 5-min sanity check before burning CUs

├── scripts/

│   ├── download\_models.py            \[CODE\] cache LocateAnything \+ checkpoints to Drive

│   └── make\_paper\_figures.py         \[CODE\] regenerate every figure from results JSON

├── results/                          (gitignored except .gitkeep) run outputs

│   ├── runs/                         per-run JSON \+ checkpoints

│   ├── figures/                      generated PNG/PDF figures

│   └── tables/                       generated LaTeX/CSV tables

├── data/                             (gitignored) tiles, labels, features

│   ├── tiles/                        S2 \+ ESRI tiles

│   ├── labels/                       weak labels \+ HC masks

│   ├── socioeconomic/                E-tensor rasters

│   └── features\_cache/               cached MoonViT features

└── docs/

    ├── SPEC.md                       this document

    ├── DATA\_CARD.md                  \[CODE\] dataset documentation for release

    └── RESULTS.md                    \[CODE\] auto-updated results summary

---

## 5\. Data pipeline

### 5.1 Regions (config/regions\_dhaka.yaml)

Five regions, retained from the v3 proposal. Each needs a WGS84 bounding box and a class hint.

| Region | Type | Role | Notes |
| :---- | :---- | :---- | :---- |
| Korail | informal | train \+ eval | Largest Dhaka slum, beside Gulshan lake |
| Bhashantek | informal | train \+ eval | Government-resettlement \+ informal mix |
| Karail-extension | informal | train \+ eval | Periphery informal growth |
| Old Dhaka brick core | formal-dense control | train \+ eval | The discriminator-killer: dense pucca masonry |
| Gulshan-2 / Baridhara | formal-dense control | train \+ eval | Affluent, dense, road-connected, bright at night |

**\[DECISION\]** Claude Code writes placeholder bounding boxes with clearly-marked `TODO_VERIFY` comments. The team must verify/adjust coordinates in GEE before the first real export. The notebook has a "preview region on map" cell so this is a 2-minute visual check, not guesswork.

### 5.2 Imagery

- **BanglaSlumNet input:** Sentinel-2 L2A, seasonal best-pixel composites, 10 m, bands B2/B3/B4/B8 (+ B11/B12 optional for SWIR haze signal). One dry-season and one wet-season composite per region per year for the training years. For the Dhaka-only paper, use **2020–2023** composites (4 years × 2 seasons) — enough temporal variety for SAS-Net consistency without exploding tile count.  
- **GRAM-baseline input only:** ESRI World Imagery z16 tiles (\~1.2 m/px), to match GRAM's native resolution. This path reuses the team's existing `gram_baseline/` bundle and is *only* for the head-to-head failure-mode reproduction, not for training BanglaSlumNet.

### 5.3 Tiling

- Tile size: **256×256** at 10 m (≈ 2.56 km × 2.56 km per tile). This matches GRAM preprocessing and keeps MoonViT inputs reasonable.  
- Stride: 128 px (50% overlap) for training tiles to multiply data; 256 (no overlap) for eval tiles to avoid leakage.  
- Target counts: 500–1000 weakly-labeled training tiles; \~100 HC eval tiles (matches v3 target scale).

### 5.4 Acceptance criteria for the data layer

- [ ] Every tile has aligned: RGB(+SWIR) array, weak-label mask, HC mask, and the resampled socioeconomic tensor — all on the identical 256×256 grid.  
- [ ] `src/data/tiles.py` raises a clear error if any layer is misaligned (assert grid shape \+ geotransform).  
- [ ] A `dataset_manifest.json` records every tile, its region, split, and per-pixel label-class counts.

---

## 6\. Weak-label pipeline (§5 of proposal, extended with Direction B)

### 6.1 Geospatial fusion (unchanged core, in GEE)

`gee/03_weak_labels.js` implements the v3 rules:

- **slum pixel** \= OSM residential ∩ GHSL built-up ∩ VIIRS dark (≤ city median)  
- **formal-dense pixel** \= OSM residential ∩ GHSL built-up ∩ VIIRS bright (\> city median)  
- **unknown** \= anything else  
- Per-pixel **3-signal agreement score** ∈ {0,1,2,3}.

Export: per-tile noisy label mask \+ per-tile 3-signal HC mask.

### 6.2 Direction B — LocateAnything zero-shot validation (NEW)

`src/locate_anything/label_validator.py` adds a **fourth signal**.

For each candidate tile (use the higher-resolution ESRI z16 view of the same footprint, since LocateAnything is trained on natural-resolution imagery and will ground far better on 1.2 m/px than on 10 m S2):

1. Run two grounding prompts (templates in `prompts.py`):  
   - `P_slum`: *"Locate all areas of dense informal settlement: clusters of small irregular rooftops, narrow unpaved gaps between structures, no organized road grid."*  
   - `P_formal`: *"Locate all areas of formal urban housing: regular building footprints, organized street grid, wide paved roads."*  
2. Use **`generation_mode="slow"`** (most robust) at low `max_new_tokens` (e.g. 512\) — we want quality, not speed, and tile counts are small.  
3. Parse boxes with `LocateAnythingWorker.parse_boxes`. Rasterize each prompt's boxes into a coverage mask.  
4. Define `la_slum_score(pixel) = slum_coverage − formal_coverage` ∈ \[−1, 1\].  
5. **Fusion into confidence:** a pixel reaches the **High-Confidence (HC) eval subset** only if the 3 geospatial signals agree **AND** `la_slum_score` agrees in sign (positive for slum, negative for formal-dense). This is a 4-way agreement, strictly stronger than v3's 3-way HC.

This directly uses the VLM's visual understanding to filter out cases where the geospatial proxies are misled (e.g., a dark-at-night formal warehouse district that OSM tags residential).

**CU cost control:** LocateAnything validation runs **once**, offline, over ≤ \~1000 tiles, results cached to `data/labels/la_validation.json`. Never re-run during training. Budget: \~1–3 s/tile on A100 in slow mode → \~30–50 min one-time. Gate behind `config.weak_labels.use_locate_anything_validation: true`.

### 6.3 Outputs

- `data/labels/<tile_id>_noisy.png` — 3-class noisy training mask  
- `data/labels/<tile_id>_hc.png` — binary HC eval mask (4-signal agreement)  
- `data/labels/confidence.json` — per-tile signal breakdown for the data card

### 6.4 Acceptance criteria

- [ ] HC subset is non-empty for every region and contains both slum and formal-dense pixels.  
- [ ] `confidence.json` reports, per region, the fraction of pixels at each agreement level (this becomes a paper table).  
- [ ] Validation is fully cached and re-runnable without GPU once cached.

---

## 7\. LocateAnything integration details

### 7.1 Worker (`src/locate_anything/worker.py`)

Port the `LocateAnythingWorker` class verbatim from the HF model card (it is the canonical, tested interface), with these adaptations:

- Add a `load_in_4bit` option (bitsandbytes) **default False**; enable only if the team hits OOM on a smaller GPU. On A100 40GB, BF16 is fine.  
- Add `extract_visual_features(image)` that returns the MoonViT patch features (see §7.2) instead of generating text.  
- Wrap model loading so the heavy weights are cached to Google Drive (`scripts/download_models.py`) and reused across sessions — **never re-download per run** (CU \+ wall-clock saver).

### 7.2 Feature extraction (`src/locate_anything/feature_extractor.py`) — Direction A core

This is the most delicate integration point. LocateAnything is a `trust_remote_code` model; its internal module names are not guaranteed stable. Claude Code must:

1. On first load, **introspect** the model: print `model.config`, the vision tower attribute path, and module tree. The notebook has a dedicated **"inspect model internals"** cell whose output the team pastes back if the attribute path differs from the assumed one.  
2. Assume the vision encoder is reachable (typical EAGLE/Qwen-VL layout) at something like `model.vision_model` / `model.visual` / `model.vision_tower`. Implement a small resolver that tries known candidates and falls back to a regex over named modules for a MoonViT-like block.  
3. Register a **forward hook** on the last vision-encoder block to capture patch embeddings of shape `[B, N_patches, D]`. Reshape to a 2D feature grid `[B, D, H_f, W_f]` using the native-resolution patch layout (the processor exposes `image_grid_hws`).  
4. **Cache** features to `data/features_cache/<tile_id>.npy` keyed by `(tile_id, prompt_id)`. Feature extraction is the single most expensive repeated op; caching it once turns every subsequent training epoch into a cheap operation over precomputed tensors.

**Language conditioning:** for the `vlm_lang` and `full` configs, run the encoder with the discriminative prompt prepended (the VLM's cross-modal attention lets the text query modulate which visual regions are salient). Cache separate feature maps per prompt. For `vlm_visual`, use a neutral prompt.

**\[DECISION / RISK\]** If hooking the internal encoder proves brittle (remote code changes), fall back to **"grounding-map features"**: run the VLM's grounding prompts and rasterize the predicted box/confidence maps into dense channels, then treat *those* as the visual feature input to fusion. This is more robust (uses only the public `generate` API) at some loss of feature richness. The notebook exposes `feature_mode: ["hidden_state", "grounding_map"]`, default `hidden_state` with automatic fallback to `grounding_map` on hook failure.

### 7.3 Prompts (`src/locate_anything/prompts.py`)

Centralize all prompt strings. At minimum:

- `NEUTRAL` (for vlm\_visual): "Locate all dense built-up residential areas."  
- `SLUM_DISCRIMINATIVE`: the informal-settlement description from §6.2.  
- `FORMAL_DISCRIMINATIVE`: the formal-housing description from §6.2.

Keep them versioned (a `PROMPT_VERSION` constant) so results JSON records which prompts produced which numbers — essential for the paper's reproducibility.

---

## 8\. Models

### 8.1 Stage 1 — SAS-Net (`src/models/sasnet.py`)

Scene-Appearance Separation per the proposal: `I = R(s, a) + n`, with `R` an AdaIN-based differentiable renderer. Structure encoder `E_s`, appearance encoder `E_a`, renderer `R`.

- **Losses** (`train/losses.py`): reconstruction `L_rec = ||I − R(E_s(I), E_a(I))||`; **scene-consistency** `L_consist = ||E_s(I_t1) − E_s(I_t2)||` across two capture dates of the same location (the core trick — structure must be date-invariant); plus an appearance-swap re-render loss.  
- **Size:** \~3.35M params per proposal — trains on a single GPU quickly.  
- **Output:** every tile re-rendered at a fixed clean reference appearance `a_ref`. Cache the clean tiles.  
- **Validation:** SSIM/PSNR of normalized output vs. clean reference (Experiment 1).

**CU control:** SAS-Net Stage 1 trains on raw S2 tiles only (no VLM in the loop). Short run: target ≤ 1 A100-hour. Cache clean tiles so Stage 2 never re-runs SAS-Net.

### 8.2 Cross-attention fusion (`src/models/fusion.py`)

`F = CrossAttention(Q=V, K=E, V=E) + V` where `V` is the (projected) VLM visual feature grid and `E` is the socioeconomic tensor projected to the same `[D, H_f, W_f]` grid.

- Multi-head attention over flattened spatial tokens; standard pre-norm transformer block.  
- Residual preserves visual features; socioeconomic channels *modulate* ambiguous regions (the architectural answer to the failure mode).  
- Per-channel ablation hooks: the module accepts a `channel_mask` so we can zero out individual socioeconomic channels without retraining the projector shape (§9).

### 8.3 Segmentation decoder (`src/models/decoder.py`)

Lightweight UNet-style head: 2–3 upsampling blocks from `[D, H_f, W_f]` to `[1, 256, 256]`, sigmoid → binary slum probability. Few M params.

### 8.4 Optical-only baseline (`src/models/baseline_cnn.py`)

SegFormer-B0 or ResNet-UNet on clean RGB only, no language, no socioeconomics. This is our **fully-controlled analogue of the GRAM failure** — same data, same eval, optical-only. It should reproduce high recall on Korail with high false positives on Old Dhaka, mirroring the measured GRAM behavior. This is cleaner for the paper than comparing across resolutions to GRAM directly (we still report the real GRAM head-to-head separately via `eval/gram_baseline.py`).

### 8.5 Assembly (`src/models/banglaslumnet.py`)

A single `build_model(config)` that returns the right composition for `baseline_cnn | vlm_visual | vlm_lang | full`. All four share the decoder and eval code so comparisons are apples-to-apples.

### 8.6 Acceptance criteria

- [ ] A forward pass on one tile succeeds for all four configs (smoke test).  
- [ ] Trainable-parameter count is printed per config and is small for VLM configs (frozen encoder verified).  
- [ ] Cached features are used (a second epoch does not call the VLM).

---

## 9\. Experiments, training, testing, validation

All experiments train only the fusion module \+ decoder (and SAS-Net once). The VLM encoder is frozen and cached. This is what makes the whole study fit in a handful of A100 sessions.

### 9.1 Data splits

- **Train:** noisy weak labels on overlapping tiles across all 5 regions, excluding HC eval tiles.  
- **Validation:** a held-out 15% of HC tiles, used for early stopping / checkpoint selection.  
- **Test (primary):** the remaining HC tiles. Spatially disjoint from train (split by geographic block, **not** random pixel split — prevents leakage).  
- Report **HC-IoU** (primary) and **All-IoU** (secondary, pessimistic) per §6 of proposal.

### 9.2 Experiment 1 — Atmospheric correction ablation (SAS-Net)

Objective: prove Stage 1 helps before any fusion. Hold Stage 2 fixed at `vlm_visual`.

| Condition | Stage-1 input | Metric focus |
| :---- | :---- | :---- |
| Raw | raw S2, no correction | baseline |
| FORCE/CMAC | classical atmospheric correction | comparison |
| SAS-Net (ours) | SAS-Net normalized | expected best |

Report: HC-IoU, All-IoU, plus SSIM/PSNR of the normalized imagery. **\[DECISION\]** FORCE/CMAC is optional and CU-cheap (CPU); if the team lacks a FORCE setup, substitute a simple histogram-matching / dark-object-subtraction classical baseline and label it as such honestly.

### 9.3 Experiment 2 — Socioeconomic fusion ablation (CENTRAL)

Objective: identify which signals fix the formal-vs-informal failure. Stage 1 \= SAS-Net for all rows.

| Variant | Backbone config | Channels added | Expectation |
| :---- | :---- | :---- | :---- |
| Visual only | `vlm_visual` | none | reproduces failure: high recall Korail, high FPR Old Dhaka |
| \+ Language concept | `vlm_lang` | none (prompt only) | first precision gain from concept alone |
| \+ Nighttime lights | `full`, mask=VIIRS | VIIRS | expected **largest single** gain (per proposal) |
| \+ Population | `full`, mask=VIIRS+WorldPop | \+WorldPop | removes non-residential FPs |
| \+ Roads | `full`, mask=+OSM-roads | \+OSM | boundary precision |
| \+ Poverty | `full`, mask=+WB | \+WB | marginal, helps cross-city |
| Full fusion | `full`, all channels | all | best overall |

Primary metric: **HC-IoU** and **FPR on formal-dense control regions** (Old Dhaka \+ Gulshan). The headline result is FPR-on-control dropping sharply as orthogonal signals are added, while Korail recall holds.

### 9.4 Experiment 3 — Cross-region generalization (Dhaka-internal)

Since we scoped to Dhaka, redesign the v3 cross-*city* test as a **leave-one-region-out (LORO)** generalization test within Dhaka:

- Train on 4 regions, test on the held-out 5th. Rotate so every region is held out once.  
- The critical fold: **train without Old Dhaka, test on Old Dhaka** — does the model still suppress false positives on a formal-dense pattern it never saw? This is the strongest evidence that socioeconomic priors transfer, not memorize.  
- Metric: HC-IoU drop and FPR-on-control on the held-out region vs. in-distribution.

**\[DECISION\]** If the team also wants the original cross-*city* claim, the same pipeline runs on Chittagong/Khulna with their weak labels — but that requires extra GEE exports. Default: Dhaka LORO only. Cross-city is a one-flag extension.

### 9.5 GRAM head-to-head (`src/eval/gram_baseline.py`)

Wrap the team's existing GRAM run. Report GRAM zero-shot and (optionally) fine-tuned GRAM on the *same HC eval tiles*, alongside all BanglaSlumNet configs, in one master table. This is the table reviewers will look at first.

### 9.6 Training hyperparameters (config/default.yaml defaults)

- Optimizer: AdamW, lr `3e-4` for fusion+decoder, weight decay `0.01`, cosine schedule.  
- Batch size: 8–16 tiles (features cached → memory light). Auto-reduce on OOM.  
- Epochs: 40–80 with early stopping on val HC-IoU (patience 10). Because we train tiny heads over cached features, an epoch is seconds-to-minutes.  
- Loss: Dice \+ weighted BCE (slum class up-weighted; handle noisy labels with optional label smoothing `0.05`, gated by `config.train.label_smoothing`).  
- Mixed precision: BF16 (A100).  
- Seed: fixed and recorded; `config.seed` default 1337\.

### 9.7 CU budget table (estimates, A100 40GB)

| Task | One-time? | Est. A100 time | Note |
| :---- | :---- | :---- | :---- |
| Model \+ data download to Drive | yes | 15–30 min | cached forever |
| LocateAnything weak-label validation (\~1k tiles, slow mode) | yes | 30–50 min | cached |
| MoonViT feature extraction (all tiles × prompts) | yes | 30–60 min | cached `.npy` |
| SAS-Net Stage 1 training | yes | ≤ 1 hr | cache clean tiles |
| Stage 2 head training (one config) | per config | 5–20 min | over cached features |
| Full ablation (Exp 1+2+3, all configs/folds) | — | 2–4 hr total | dominated by re-extraction only if cache missed |
| **Total to full results** | — | **\~5–7 A100-hours** | fits a couple of Pro+ sessions |

The design intent: **pay the VLM cost once, cache everything, then iterate on cheap heads.** If a session disconnects, the registry (§10) resumes from cache.

### 9.8 Acceptance criteria

- [ ] Every experiment row writes a results JSON with: config hash, prompt version, seed, all metrics, and the path to its checkpoint.  
- [ ] Re-running an already-completed row is a no-op (skips via registry) unless `--force`.  
- [ ] The full ablation completes within the CU budget on cached features.

---

## 10\. Results tracking and documentation

### 10.1 Recorder (`src/tracking/recorder.py`)

Every eval call writes a flat JSON record:

{

  "run\_id": "exp2\_full\_allchan\_seed1337",

  "experiment": "exp2\_fusion\_ablation",

  "config": { "...full resolved config..." },

  "config\_hash": "ab12cd34",

  "prompt\_version": "v1",

  "seed": 1337,

  "git\_commit": "…",

  "timestamp": "…",

  "metrics": {

    "hc\_iou": 0.0, "all\_iou": 0.0, "precision": 0.0, "recall": 0.0,

    "f1": 0.0, "map50": 0.0, "fpr\_control\_old\_dhaka": 0.0,

    "fpr\_control\_gulshan": 0.0, "korail\_recall": 0.0,

    "ssim": null, "psnr": null

  },

  "per\_region": { "korail": {...}, "old\_dhaka": {...}, "...": {} },

  "checkpoint": "results/runs/exp2\_full\_allchan\_seed1337/best.pt"

}

All records are also flattened into `results/tables/all_runs.csv` for one-line pandas loading into the figure code.

### 10.2 Registry (`src/tracking/registry.py`)

- Maps `config_hash → run status (pending|running|done|failed)`.  
- On notebook restart, the orchestration cell queries the registry and only runs missing rows — this is the resume-safety that protects CUs against Colab disconnects.

### 10.3 Auto-documented results (`docs/RESULTS.md`)

`scripts/make_paper_figures.py` regenerates `docs/RESULTS.md` from `all_runs.csv`: the master comparison table, the ablation table, and the LORO table, all in Markdown \+ a LaTeX export to `results/tables/*.tex` ready to paste into the manuscript.

---

## 11\. Visualization / paper figures (`src/viz/`)

### 11.1 Palette (`src/viz/palette.py`)

Reuse the team's established consistent palette: **navy / teal / steel-blue / slate**, with a fixed mapping (e.g. navy \= BanglaSlumNet-full, steel \= ablation variants, slate \= GRAM/baseline, teal \= highlight). Define once; every figure imports it. Margin-box annotations style matches the team's prior matplotlib convention.

### 11.2 Required figures (`src/viz/plots.py`, each a function returning a saved PNG \+ PDF)

1. **fig\_failure\_repro** — bar chart: mean slum probability / FPR-on-control for GRAM vs `baseline_cnn` vs `full`, across Korail / Mirpur / Old Dhaka. The "everyone sees slums everywhere, we don't" figure.  
2. **fig\_exp1\_sasnet** — grouped bars: Raw vs FORCE vs SAS-Net on HC-IoU \+ a SSIM/PSNR inset.  
3. **fig\_exp2\_ablation** — the central figure: incremental HC-IoU (line) and FPR-on-control (line, secondary axis) as channels are added Visual→+Lang→+VIIRS→+Pop→+Roads→+Poverty→Full. This is the paper's money figure; it visually proves each orthogonal signal helps.  
4. **fig\_exp3\_loro** — heatmap or grouped bars: per-held-out-region HC-IoU and FPR-on-control, in vs out of distribution.  
5. **fig\_master\_table** — rendered comparison table (also exported as LaTeX).  
6. **fig\_qualitative** (`src/viz/qualitative.py`) — N×M grid: tile | GRAM pred | baseline pred | BanglaSlumNet pred | HC ground truth, for representative Korail and Old Dhaka tiles. The visual proof.  
7. **fig\_pr\_curves** — precision-recall curves per config on the HC test set.  
8. **fig\_confidence\_strata** — per-region stacked bars of label-agreement levels (from the data card), justifying the HC subset.

### 11.3 Conventions

- All figures: 300 DPI, both `.png` (slides) and `.pdf` (LaTeX), saved to `results/figures/`.  
- Every figure reads numbers **only** from `results/tables/all_runs.csv` — never hardcoded. Re-running the script after new results regenerates everything.  
- Font sizes and margins set for single-column CVPR width by default; a `--wide` flag for double-column.

### 11.4 Acceptance criteria

- [ ] `python scripts/make_paper_figures.py` regenerates all figures \+ tables from results with zero manual editing.  
- [ ] Each figure function has a tiny synthetic-data unit smoke test so the plotting code can be verified before real results exist.

---

## 12\. Deferred work (kept in paper as application / future work)

- **National Bangladesh mapping (Exp 5):** run the trained `full` head over Sentinel-2 for all 64 districts, 2015–2025, producing the 10 m time-series map. Heavy GEE \+ inference; deferred. Note: if released, must respect the licensing in §2.1.  
- **Temporal robustness (Exp 4):** 30 fixed locations, all S2 captures 2015–2025, detection-confidence variance, SAS-Net vs raw. Deferred but cheap to add later since the pipeline already ingests multi-date tiles.  
- Both remain in the manuscript's "Application at scale" and "Future work" sections so the contribution story is intact even though the Dhaka experiments carry the paper.

---

## 13\. The master Colab notebook (`notebooks/BanglaSlumNet_Colab.ipynb`)

Claude Code builds the notebook with these cells, in order, each idempotent and CU-aware. Markdown headers separate phases; every heavy phase checks the cache/registry first and skips if already done.

**Phase 0 — Setup**

1. Mount Google Drive; set `PROJECT_ROOT` to a Drive folder so all caches/checkpoints survive disconnects.  
2. `git clone` the repo (or `git pull` if present).  
3. Install `requirements_colab.txt` in the correct order (torch matching Colab CUDA first, then pinned VLM deps). Do **not** install MagiAttention.  
4. GPU check cell: print `nvidia-smi`, assert ≥ 24 GB or warn.

**Phase 1 — Models & data (one-time, cached)** 5\. `download_models.py` → cache LocateAnything to Drive. 6\. **Inspect model internals** cell (§7.2) — prints module tree for the feature-extractor resolver. 7\. Region preview-on-map cell (verify bounding boxes). 8\. Trigger/confirm GEE exports (instructions \+ the JS to paste into the GEE code editor, since GEE JS runs in GEE, not Colab; the Python `ee` path is offered as an alternative). Tiles land in Drive. 9\. Build `dataset_manifest.json`.

**Phase 2 — Weak labels (one-time, cached)** 10\. Run geospatial fusion ingest. 11\. Run LocateAnything validation (Direction B) → cache. 12\. Stratify HC vs noisy; write `confidence.json`; render `fig_confidence_strata`.

**Phase 3 — Feature & SAS-Net caching (one-time)** 13\. MoonViT feature extraction over all tiles × prompts → `.npy` cache. 14\. Train SAS-Net; cache clean tiles; record SSIM/PSNR.

**Phase 4 — Experiments (cheap, resumable)** 15\. Orchestration cell: read `experiments.yaml`, query registry, run only missing rows. Each row trains the small head over cached features and writes a results JSON. 16\. GRAM head-to-head eval on HC tiles.

**Phase 5 — Figures & tables** 17\. `make_paper_figures.py` → every figure \+ LaTeX table \+ `docs/RESULTS.md`. 18\. Display all figures inline for a final visual check.

**Phase 6 — Smoke test variant (`00_smoke_test.ipynb`)** A 5-minute end-to-end on 4 tiles and 1 config, to verify wiring before any real CU spend. Run this first, always.

### 13.1 Notebook acceptance criteria

- [ ] Fresh runtime → Phase 0–5 completes using only cached artifacts on a second run (no recompute).  
- [ ] A mid-run disconnect, then re-run, resumes without redoing cached phases.  
- [ ] Every figure renders inline at the end.

---

## 14\. Build order for Claude Code (do these in sequence)

1. Repo scaffold \+ `README.md` \+ license files \+ `requirements*.txt` \+ `config/*.yaml` (with `TODO_VERIFY` markers on region boxes).  
2. `src/data/*` \+ `gee/*` scripts \+ `dataset_manifest` builder. Unit-test alignment on synthetic arrays.  
3. `src/locate_anything/*` — worker (port from HF card), feature extractor with introspection \+ fallback, label validator, prompts.  
4. `src/models/*` — SAS-Net, fusion, decoder, baseline, assembly. Smoke-test forward pass per config.  
5. `src/train/*` \+ `src/eval/*` \+ `src/tracking/*`.  
6. `src/viz/*` with synthetic-data smoke tests for each figure.  
7. `notebooks/00_smoke_test.ipynb` then `notebooks/BanglaSlumNet_Colab.ipynb`.  
8. `scripts/*`, `docs/DATA_CARD.md`, `docs/RESULTS.md` template.

### 14.1 Global engineering rules for Claude Code

- **Everything is config-driven.** No hardcoded paths, hyperparameters, or region boxes in source; all in `config/`.  
- **Cache aggressively, recompute never.** Any op that touches the VLM or GEE writes to Drive and checks the cache first.  
- **Fail loud on misalignment.** Assert grid shapes and geotransforms; a silent misaligned label is the worst-case bug here.  
- **Frozen means frozen.** Assert `requires_grad=False` on the VLM encoder and print trainable param counts.  
- **Record provenance.** Every results JSON carries config hash, prompt version, git commit, seed.  
- **Synthetic smoke tests** for data alignment, model forward passes, and every plotting function, so the team can validate wiring before spending CUs.  
- **No MagiAttention on Colab** (Ampere); rely on SDPA fallback.  
- **Respect the license**: research-only banner in README and on any released artifact.

---

## 15\. Open decisions for the team (surface these in the notebook as flags)

| ID | Decision | Default | Where |
| :---- | :---- | :---- | :---- |
| D1 | Feature mode: hidden-state vs grounding-map | `hidden_state` (auto-fallback) | §7.2 |
| D2 | LoRA-adapt the VLM LM layers? | `false` (zero-shot prompting only) | §2.3 |
| D3 | FORCE/CMAC vs simple classical correction for Exp 1 | classical if FORCE absent | §9.2 |
| D4 | Cross-region (LORO) only, or also cross-city (Chittagong/Khulna)? | LORO only | §9.4 |
| D5 | Training years for S2 composites | 2020–2023, 2 seasons | §5.2 |
| D6 | Include real GRAM head-to-head or controlled baseline only? | both | §8.4, §9.5 |
| D7 | SWIR bands (B11/B12) in input? | optional, off | §5.2 |

---

## 16\. What "done" looks like

The team can, in one or two Colab Pro+ sessions:

1. Run the smoke test (5 min) — wiring verified.  
2. Run Phases 1–3 once — VLM, labels, features, SAS-Net all cached to Drive (\~3–4 A100-hr).  
3. Run Phase 4 — the entire experiment matrix over cached features (\~1–2 A100-hr).  
4. Run Phase 5 — every paper figure and table regenerated from recorded numbers.  
5. Open `docs/RESULTS.md`, read the headline table, and start writing the manuscript with real numbers and publication-ready figures already in `results/figures/`.

The paper writes itself from the recorded results: failure-mode reproduction (fig 1), SAS-Net ablation (fig 2), the central socioeconomic+language fusion ablation (fig 3), cross-region generalization (fig 4), the master comparison table, and qualitative overlays (fig 6).  

---

## 17. v4 amendments (2026-06-16 — what actually got built/changed vs the contract above)

These supersede the corresponding parts of §§4–13. See `BanglaSlumNetV4.md` and
`AGENT_HANDOFF.md` for full detail.

- **Regions:** expanded from 5 to **12** (`config/regions_dhaka.yaml`): 8 informal + 4
  formal-dense, to get enough unique slum tiles.
- **Tile size:** 256 → **128 px** (Dhaka neighborhoods are small/adjacent); tile ALL dry-season
  composites; stratified-by-region split → **720 tiles**.
- **Weak labels:** the per-pixel OSM∩GHSL∩VIIRS rule (§6.1) **collapsed to 0 slum / all formal**
  (VIIRS dark/bright threshold never separated the neighborhoods). **Replaced with region-type
  supervision**: built pixels (GHSL∪DynamicWorld) in informal regions → slum, in formal regions
  → formal; HC = built pixels. This is the current labeling in `gee/export_weak_labels.py`.
- **Direction B (LA HC-validation):** **OFF** (`use_locate_anything_validation: false`) — LA on
  10 m S2 is unreliable and zeroed HC. LA *features* still used. Re-enable with high-res ESRI
  grounding (future).
- **Feature mode (D1):** default **`grounding_map`** (prompt-specific 32×32 box-coverage maps;
  needed for the language ablation). `hidden_state` available for prompt-agnostic visual feats.
- **GEE export:** runs in-notebook via the Python `ee` API (`gee/export_*.py`); S2 exported
  full-extent via `unmask`; wet-season composites are empty (monsoon) and fail harmlessly.
- **Placeholders still to replace:** `osm_roads` (accessibility proxy) and `wb_poverty` (zeros).
- **Status:** pipeline complete & cached; region-type labels just applied; pending the P4.1b
  verify gate (slum>0, HC>0) then the Phase-4 run for real numbers.
