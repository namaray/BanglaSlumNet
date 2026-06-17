# BanglaSlumNet Presentation Slide Content

## Presentation Flow

The presentation should move like a research story:

1. **Why this problem matters**: slum mapping is important, but Dhaka exposes a serious weakness in optical-only remote sensing.
2. **What prior work gives us and where it fails**: classic remote sensing, deep segmentation, and foundation models help, but density is not the same as informality.
3. **What we propose**: BanglaSlumNet adds language-grounded visual reasoning and socioeconomic context.
4. **How we built and evaluated it**: Dhaka regions, weak labels, cached feature pipeline, experiment matrix.
5. **What we found / will report**: verified labels now pass; final experimental numbers will be inserted after Colab Phase 4 and Phase 5 complete.

---

# Speaker 1: Problem and Motivation

## Slide 1 - BanglaSlumNet: Research Question

**Slide content**

**BanglaSlumNet: Language-Grounded Socioeconomic Fusion for Dense-Megacity Slum Detection**

This project studies informal-settlement detection in Dhaka, Bangladesh. The main research question is:

**Can a vision-language model, guided by socioeconomic context, distinguish informal settlements from dense formal neighborhoods in a megacity where optical texture alone is ambiguous?**

The key challenge is that many satellite models learn to detect dense built-up areas. In Dhaka, that is not enough. A dense neighborhood may be an informal settlement, but it may also be a formal, planned, road-connected, high-income neighborhood. Our work asks whether language and socioeconomic priors can separate those cases.

**Suggested slide structure**

| Item | Content |
|---|---|
| Title | BanglaSlumNet |
| Subtitle | Language-grounded socioeconomic fusion for Dhaka slum detection |
| Research question | Can we separate informal density from formal density? |
| Case study | Dhaka, Bangladesh |

---

## Slide 2 - Why Slum Detection Matters

**Slide content**

Accurate informal-settlement maps are important for urban planning, infrastructure allocation, public health, climate resilience, disaster response, and poverty monitoring. In fast-growing cities, official maps are often incomplete or outdated. Field surveys provide high-quality information, but they are expensive, slow, and difficult to repeat across many locations.

Remote sensing offers a scalable alternative. Satellite imagery can cover large urban areas regularly and consistently. However, informal settlements are not defined only by how roofs look from above. They are also shaped by access to roads, electricity, public services, population pressure, and poverty. A model that sees only optical imagery can miss this larger context.

**Key message**

The mapping problem is not simply "find dense roofs." It is "identify a social and spatial condition from imperfect visual evidence."

---

## Slide 3 - The Dhaka Challenge: Density Is Not Informality

**Slide content**

Dhaka is a difficult test case because it is dense almost everywhere. Informal settlements such as Korail, Bhashantek, and Kamrangirchar can visually resemble formal dense neighborhoods such as Old Dhaka, Gulshan-Baridhara, Dhanmondi, or Uttara in medium-resolution satellite imagery.

This creates a failure mode for optical-only models:

| What the model sees | What it may predict | Why this is a problem |
|---|---|---|
| Dense rooftops | Slum | Dense formal areas become false positives |
| Irregular texture | Slum | Old urban fabric may be misclassified |
| Bright or dark RGB variation | Settlement type | Lighting and service access are not visible in RGB |
| Built-up area | Informality | Built-up does not mean informal |

The central difficulty is not that the image contains no useful information. The difficulty is that the image is underdetermined: two socially different places can look visually similar at 10 m resolution.

**Designer instructions**

Create a side-by-side comparison diagram:
- Left panel: "Informal settlement region"
- Right panel: "Dense formal control region"
- Use satellite-like patches or abstract tile placeholders.
- Place a central label: "Visually similar, socially different."

---

## Slide 4 - Research Problem and Hypothesis

**Slide content**

The research problem is that optical-only slum detectors often confuse **density** with **informality**. This is especially serious in dense megacities like Dhaka, where formal and informal neighborhoods can share similar roof size, building spacing, and texture.

Our hypothesis is:

**A model can reduce dense-formal false positives if it combines three signals:**

| Signal | What it contributes |
|---|---|
| Optical imagery | Roofs, built-up texture, settlement layout |
| Language grounding | Conceptual distinction between informal and formal dense settlement |
| Socioeconomic layers | Nighttime light, population, road access, poverty proxy, built-up extent |

In other words, BanglaSlumNet does not ask the image to solve the whole problem alone. It adds semantic and socioeconomic evidence that is aligned with how informality actually appears in cities.

**Designer instructions**

Create a simple equation-style graphic:

`RGB density alone -> ambiguous`

`RGB + language + socioeconomic context -> informal/formal discrimination`

---

# Speaker 2: Literature Review and Research Gap

## Slide 5 - Earlier Remote-Sensing Approaches

**Slide content**

Early informal-settlement mapping relied on handcrafted remote-sensing features: spectral bands, texture measures, morphology, object-based image analysis, and rule-based classification. These approaches are interpretable and useful, but they often depend on expert feature engineering and local calibration.

Deep learning improved the field by learning features automatically. CNNs, UNets, SegFormer-like models, and semantic segmentation pipelines can detect complex spatial patterns more effectively than handcrafted methods. However, supervised deep learning still depends on reliable labels. For informal settlements, high-quality pixel-level labels are expensive and often unavailable.

**Summary table**

| Approach family | Strength | Limitation |
|---|---|---|
| Handcrafted features | Interpretable, low compute | City-specific, limited semantic understanding |
| Classical ML | Works with smaller datasets | Depends on feature design |
| CNN/UNet segmentation | Learns spatial patterns | Needs labeled data |
| Foundation models | Strong general features | May inherit dataset bias |

---

## Slide 6 - Foundation Models and Optical-Only Failure

**Slide content**

Recent geospatial foundation models learn broad visual representations from large imagery collections. They are attractive because they can transfer across regions and reduce the need for task-specific labels.

But a foundation model trained mainly on optical patterns can still learn the wrong shortcut. If many training examples associate informal settlements with dense irregular built-up areas, the model may treat density as the core signal. In Dhaka, this shortcut fails because dense formal areas also exist.

This is the reason BanglaSlumNet uses optical-only models as a baseline rather than assuming they solve the problem. We expect optical-only approaches to show high confusion in formal dense controls.

**Key contrast**

| Model behavior | Works when | Fails when |
|---|---|---|
| Detect dense built-up texture | Slums are visually distinct from formal areas | Formal neighborhoods are also dense |
| Detect irregular urban morphology | Informality has unique shape patterns | Old formal urban fabric is irregular too |
| Transfer learned visual features | New city resembles training cities | New city violates learned assumptions |

**Designer instructions**

Create a flow diagram:

`Satellite tile -> Optical foundation model -> "dense built-up" response -> false slum prediction`

Add a warning callout: "Density is a shortcut, not the target."

---

## Slide 7 - Vision-Language Models as a New Signal

**Slide content**

Vision-language models allow us to connect image regions with text descriptions. This is useful because the difference between informal and formal dense settlement is partly conceptual. We can describe the target as more than a visual class:

| Prompt concept | Intended meaning |
|---|---|
| Dense informal settlement | Small irregular rooftops, narrow gaps, organic layout, limited access |
| Dense formal housing | Planned roads, organized blocks, larger formal structures, connected infrastructure |
| Dense built-up area | Neutral visual baseline without the slum concept |

LocateAnything-3B gives us a way to ground these prompts in image tiles. We use it as a frozen feature extractor. The model produces prompt-specific grounding maps, and those maps become input features for the segmentation head.

**Designer instructions**

Create a prompt-to-map diagram:

`Prompt + Sentinel-2 tile -> LocateAnything/MoonViT -> grounding map`

Show two prompt branches:
- Slum concept prompt
- Formal dense prompt

---

## Slide 8 - Research Gap BanglaSlumNet Addresses

**Slide content**

The gap is not simply that previous methods are weak. The gap is that dense-megacity slum detection needs a model that combines visual, semantic, and socioeconomic evidence.

BanglaSlumNet targets four missing pieces at once:

| Need | Why it matters | BanglaSlumNet response |
|---|---|---|
| Formal dense controls | Test whether model confuses density with slum | Old Dhaka, Gulshan-Baridhara, Dhanmondi, Uttara |
| Language concepts | Distinguish informal from merely dense | Slum/formal prompts through LocateAnything |
| Socioeconomic context | Add non-RGB evidence | VIIRS, population, roads/access proxy, poverty proxy, GHSL |
| Compute efficiency | Avoid repeated VLM inference | Cache features and train small heads |

The contribution is therefore not just a new segmentation model. It is a pipeline designed around the specific failure mode of Dhaka.

---

# Speaker 3: What We Are Doing

## Slide 9 - BanglaSlumNet System Overview

**Slide content**

BanglaSlumNet is a two-stage framework.

**Stage 1: Scene-Appearance Separation**

SAS-Net normalizes atmospheric and seasonal appearance differences in Sentinel-2 tiles. The goal is to make the downstream model focus more on structure and less on scene appearance.

**Stage 2: Language-Grounded Socioeconomic Fusion**

LocateAnything/MoonViT produces prompt-specific visual grounding features. These features are fused with socioeconomic layers using cross-attention. A lightweight decoder then predicts a per-pixel slum probability mask.

**Architecture summary**

| Component | Input | Output | Trainable? |
|---|---|---|---|
| SAS-Net | Sentinel-2 tile | Clean/normalized tile | Trained once |
| LocateAnything/MoonViT | Tile + prompt | Cached grounding features | Frozen |
| Socioeconomic encoder/fusion | Context layers | Fused representation | Trainable |
| Decoder | Fused features | Slum probability mask | Trainable |

**Designer instructions**

Create the main architecture diagram:

`Sentinel-2 tile -> SAS-Net -> clean tile`

Then branch into:
- `LocateAnything grounding features`
- `Socioeconomic tensor`

Merge with:
`Cross-attention fusion -> decoder -> slum mask`

---

## Slide 10 - Language Grounding in BanglaSlumNet

**Slide content**

The language part of BanglaSlumNet is designed to tell the model what distinction matters. Instead of asking only "is this dense?", we ask for concepts related to informal settlement structure and formal dense housing.

**Prompt roles**

| Prompt role | Purpose |
|---|---|
| Neutral visual prompt | Baseline VLM feature without slum concept |
| Slum prompt | Captures informal settlement cues |
| Formal prompt | Captures dense but formal urban cues |

These prompts produce grounding maps. The maps are coarse but useful because they are prompt-specific. This makes the language ablation meaningful: we can compare neutral visual features against language-conditioned features.

**Important implementation detail**

The VLM is not fine-tuned. It is frozen. We compute features once, save them to Drive, and train only the fusion and decoder heads.

**Designer instructions**

Create a two-row prompt comparison:

| Prompt | Grounding map |
|---|---|
| Informal/slum concept | heatmap |
| Formal dense concept | heatmap |

Use this as a conceptual visualization, not necessarily real output unless final figures are available.

---

## Slide 11 - Socioeconomic Fusion

**Slide content**

Socioeconomic features are added because RGB imagery cannot directly show service access, road connectivity, wealth, or nighttime activity. These variables help separate settlements that look visually similar.

**Current socioeconomic channels**

| Channel | Intended signal | Interpretation |
|---|---|---|
| VIIRS nighttime lights | Electricity and economic activity | Formal affluent areas often brighter |
| WorldPop | Population density | Helps distinguish residential intensity |
| GHS-POP | Population estimate | Additional population proxy |
| Road/access proxy | Connectivity | Formal areas usually more connected |
| Poverty placeholder | Socioeconomic deprivation | Placeholder; must be replaced for final claims |
| GHSL built-up | Built extent | Identifies built pixels for context |

The fusion module uses cross-attention: visual features query the socioeconomic tensor. This means the model can learn where context should change the interpretation of visual density.

**Designer instructions**

Create a layer-stack diagram:
- VIIRS
- Population
- Roads/access
- Poverty proxy
- Built-up

All layers feed into "Cross-attention fusion" alongside VLM features.

---

## Slide 12 - Model Variants We Compare

**Slide content**

We evaluate BanglaSlumNet through controlled ablations. Each variant removes or adds one major source of information.

| Model variant | Visual signal | Language signal | Socioeconomic fusion | Purpose |
|---|---|---|---|---|
| `baseline_cnn` | RGB only | No | No | Optical-only baseline |
| `vlm_visual` | VLM visual features | Neutral prompt | No | Tests VLM visual representation |
| `vlm_lang` | VLM features | Slum/formal prompts | No | Tests language grounding |
| `full` | VLM features | Slum/formal prompts | Yes | Full BanglaSlumNet |

The expected pattern is that the optical-only baseline should struggle with dense formal controls, language should improve the concept separation, and socioeconomic fusion should reduce false positives further.

---

# Speaker 4: Procedure and Experimental Design

## Slide 13 - Study Area and Dataset

**Slide content**

The dataset covers 12 Dhaka regions: 8 informal regions and 4 formal dense controls. The controls are important because they directly test the failure mode: dense areas that should not be predicted as slums.

**Region groups**

| Informal regions | Formal dense control regions |
|---|---|
| Korail | Old Dhaka |
| Bhashantek | Gulshan-Baridhara |
| Karail extension | Dhanmondi |
| Kamrangirchar | Uttara |
| Kallyanpur |  |
| Hazaribagh |  |
| Tongi |  |
| Mirpur Beribadh |  |

**Dataset facts**

| Property | Value |
|---|---|
| City | Dhaka |
| Regions | 12 |
| Tile size | 128 px |
| Resolution | Sentinel-2, 10 m |
| Total tiles | 720 |
| Split | Stratified by region |

**Designer instructions**

Create a Dhaka map with 12 bounding boxes:
- Informal regions in one color
- Formal controls in another color
- Include a legend and short caption: "Controls are dense formal areas, not negative random samples."

---

## Slide 14 - Phase 1: Data Export and Region Benchmark

**Slide content**

The first experimental phase was to build a Dhaka benchmark that directly tests the failure mode we care about. We selected both known informal settlements and dense formal control regions, because the model must learn not only where slums are, but also where dense urban fabric is **not** a slum.

The data pipeline exports Sentinel-2 composites, weak-label rasters, and socioeconomic layers from Google Earth Engine. These are then tiled into aligned 128 px samples so the RGB image, label mask, high-confidence mask, and socioeconomic tensor all share the same grid.

**What we did**

| Step | Purpose | Status |
|---|---|---|
| Select 12 Dhaka regions | Balance informal and formal dense examples | Done |
| Export Sentinel-2 dry-season composites | Main optical imagery | Done |
| Export socioeconomic tensors | VIIRS, population, roads/proxy, poverty/proxy, built-up | Done with caveats |
| Tile all rasters | Produce aligned model inputs | Done |

**Dataset state**

| Property | Value |
|---|---:|
| Regions | 12 |
| Informal regions | 8 |
| Formal dense controls | 4 |
| Total tiles | 720 |
| Tile size | 128 px |
| Resolution | Sentinel-2, 10 m |

The key design choice is the formal-control set: Old Dhaka, Gulshan-Baridhara, Dhanmondi, and Uttara are not random negatives. They are intentionally difficult dense areas.

---

## Slide 15 - Phase 2: Weak Labels and Label Verification

**Slide content**

The second phase was weak-label construction. Our first attempted weak-label rule used nighttime brightness to separate slum and formal areas. That rule failed: it produced zero slum pixels, which caused all-zero training metrics.

We fixed this by switching to region-type weak supervision:

| Input | Rule | Output |
|---|---|---|
| Informal region + built pixel | Label as slum | Class 1 |
| Formal control region + built pixel | Label as formal | Class 2 |
| Non-built pixel | Unknown | Class 0 |

Built pixels come from GHSL built-up and Dynamic World built class. The high-confidence mask is the built-up portion of each known region. This is not manual ground truth, but it is a defensible weak-supervision strategy for a region-level benchmark.

Before training, we added a label verification gate. This gate checks that both slum and formal pixels exist, and that all per-tile labels are present.

**Verification result**

| Check | Status |
|---|---|
| Total tiles | 720 |
| HC pixels | 11,308,540 |
| Slum pixels | 7,382,955 |
| Formal pixels | 3,923,135 |
| Gate result | Passed |

**Interpretation**

This means the pipeline no longer has the earlier “zero slum label” bug. However, the labels remain weak: every built pixel in an informal region becomes slum, and every built pixel in a formal region becomes formal. This is enough to test the pipeline, but it is not yet equivalent to manual pixel-level ground truth.

---

## Slide 16 - Phase 3: Feature and SAS-Net Caching

**Slide content**

The third phase was feature caching. This is the compute-heavy part of the system, so the codebase is designed to do it once and reuse the results.

Two expensive artifacts are cached:

1. **LocateAnything/MoonViT grounding-map features**
   - Extracted once per tile and prompt.
   - Stored in `data/features_cache/`.
   - Reused by all VLM experiment rows.

2. **SAS-Net clean tiles**
   - SAS-Net was trained to normalize scene appearance.
   - Clean tile outputs are cached beside the image tiles.
   - If clean tiles exist, Phase 4 can use them without retraining SAS-Net.

**How the codebase avoids repeated compute**

| Phase | Purpose | Compute cost | Cache behavior |
|---|---|---|---|
| P0 | Setup, Drive, repo, config | Low | Re-run every session |
| P1 | GEE exports and tiling | CPU/GEE | Skips existing exports |
| P2 | Weak labels and confidence | CPU | Rebuilds after label reset |
| P3.1 | MoonViT feature extraction | High GPU once | Cached features |
| P3.2 | SAS-Net clean tiles | GPU once | Cached clean tiles |
| P4 | Experiment matrix | Moderate GPU | Registry skips completed runs |
| P5 | Figures and tables | CPU | Regenerates from CSV |

This cache-first design matters because the VLM is too expensive to call during every training epoch. In Phase 4, the model trains only small projection, fusion, and decoder heads over cached tensors.

**Designer instructions**

Create a horizontal compute pipeline:

`GEE exports -> tiling -> labels -> VLM feature cache -> SAS-Net cache -> experiment registry -> results`

Add cache icons above VLM features, SAS-Net clean tiles, and registry.

---

## Slide 17 - Phase 4: Experiments We Intended to Run

**Slide content**

The full experimental plan had three groups. The central group is Experiment 2, because it tests the paper’s main claim: whether language and socioeconomic signals help beyond visual features.

| Experiment | Question | Main comparison |
|---|---|---|
| Exp 1: SAS-Net ablation | Does appearance normalization help? | Raw/classical/SAS-Net |
| Exp 2: Fusion ablation | Which signal fixes dense-formal confusion? | Visual -> language -> socioeconomic |
| Exp 3: Leave-one-region-out | Does the model generalize across Dhaka regions? | Hold out one region at a time |

**Experiment 2 matrix**

| Row | Meaning |
|---|---|
| `exp2_visual_only` | VLM visual features with neutral prompt |
| `exp2_vlm_lang` | Add slum/formal language prompts |
| `exp2_viirs_only` | Add nighttime lights |
| `exp2_viirs_pop` | Add VIIRS + population |
| `exp2_viirs_pop_roads` | Add road/access proxy |
| `exp2_viirs_pop_roads_poverty` | Add poverty proxy |
| `exp2_full` | Full language + socioeconomic model |
| `exp2_baseline_cnn` | Optical-only CNN baseline |

**Metrics we track**

| Metric | What it measures |
|---|---|
| HC-IoU | Primary segmentation score on high-confidence pixels |
| Balanced accuracy | Whether both slum and formal classes are handled |
| Predicted-positive rate | Whether the model collapses to all-slum or all-formal |
| FPR on formal controls | Whether dense formal areas are falsely predicted as slum |
| Precision/Recall/F1 | Segmentation classification balance |

We stopped focusing on raw HC-IoU alone because it initially rewarded trivial predictions. A model that predicts slum everywhere can appear to perform well when slum pixels dominate the weak labels. Balanced accuracy and predicted-positive rate are therefore essential diagnostics.

---

# Speaker 5: Findings, Discussion, and Next Steps

## Slide 18 - What Happened in the First Phase 4 Run

**Slide content**

The first Phase 4 result looked promising at first because several visual-only runs produced HC-IoU around 0.66. After analyzing the result files, we found that this number was misleading.

The model had collapsed into simple class-prior behavior:

| Observed result | What it meant |
|---|---|
| `recall = 1.0` and `precision ≈ 0.66` | Model was predicting slum almost everywhere |
| Several rows had all metrics `0.0` | Other configs collapsed to predicting no slum |
| Formal-control FPR was `nan` | Evaluation was not yet constructing control masks |
| Socioeconomic ablation rows were misconfigured | Some rows were not actually using fusion |

This is why we stopped treating the first result table as final. It was not a true model finding; it was a debugging signal. The pipeline was running, but the training objective and evaluation diagnostics were not yet strong enough to prevent trivial solutions.

**Fixes we made**

| Issue | Fix |
|---|---|
| HC-IoU rewarded all-slum masks | Added balanced accuracy monitoring |
| Loss encouraged class-prior behavior | Switched to balanced BCE on HC pixels |
| No collapse diagnostics | Added `pred_pos_rate` and `target_pos_rate` |
| FPR fields were missing | Added formal-control masks in evaluation |
| Socioeconomic rows did not activate fusion | Corrected experiment config to use `full` |

---

## Slide 19 - Where We Stopped and Why

**Slide content**

This is the screenshot we will show as our current stopping point. It shows the corrected Phase 4 run after the debugging fixes.

**What the screenshot shows**

| Evidence in screenshot | Interpretation |
|---|---|
| `exp2_visual_only` completed | Central ablation started successfully |
| `val_balanced_accuracy ≈ 0.50` | Visual-only features are weak, close to random balance |
| `pred_pos` is no longer 0 or 1 | The collapse diagnostic is working |
| `HC-IoU = 0.4647` after visual-only | More realistic than the earlier fake 0.66 |
| `exp2_vlm_lang` began | Next test is whether language helps |

**Why we stopped here**

We stopped because the early visual-only result showed the system was now behaving honestly: it was no longer hiding collapse behind a misleading metric, but it also was not yet producing strong discrimination. At this point, continuing blindly would spend compute without guaranteeing scientifically useful numbers.

The right interpretation is:

**The codebase is now in a better debugging state, but the current data and supervision may be too weak for strong per-pixel segmentation claims.**

**Designer instructions**

Use the provided Colab screenshot as the main visual.

Add three callouts on the screenshot:
- "Balanced accuracy added"
- "Predicted-positive rate detects collapse"
- "Visual-only result is weak but honest"

---

## Slide 20 - How the Codebase Works and What Needs to Change

**Slide content**

The codebase is organized to make the experiment reproducible and compute-efficient. Code is cloned fresh into Colab, while data, model cache, features, and results persist on Google Drive.

**Codebase structure**

| Folder | Role |
|---|---|
| `gee/` | Earth Engine exporters for S2, weak labels, socioeconomic layers |
| `src/data/` | Tiling, labels, dataset loading, preflight validation |
| `src/locate_anything/` | LocateAnything worker, prompts, feature extraction |
| `src/models/` | SAS-Net, fusion module, decoder, baselines |
| `src/train/` | SAS-Net and segmentation training loops |
| `src/eval/` | Metrics and evaluation |
| `src/tracking/` | Registry and result recorder |
| `notebooks/` | Colab orchestration phases |

**What works now**

| Component | Status |
|---|---|
| GEE export and tiling | Working |
| Label verification gate | Working |
| Feature caching | Working |
| SAS-Net training/cache | Working |
| Registry resume logic | Working |
| Collapse diagnostics | Added and working |

**What needs to change next**

| Needed change | Why it matters |
|---|---|
| Create a small manual evaluation subset | Weak region labels are not enough for strong segmentation claims |
| Consider tile-level classification | Current labels are region-level, so tile-level framing may be more scientifically honest |
| Replace roads and poverty placeholders | Current socioeconomic ablations are not fully trustworthy |
| Use high-resolution imagery for VLM grounding | LocateAnything is likely too coarse on 10 m Sentinel-2 |
| Implement true LORO and true SAS-Net ablation orchestration | Current generic loop only supports the central Exp 2 matrix safely |

**Closing statement**

---
Our current conclusion is not "the model is finished." It is more precise: **we built an end-to-end, cache-aware experimental pipeline; identified and fixed label and metric collapse bugs; and learned that stronger supervision or a tile-level framing is needed before making strong segmentation claims.**
