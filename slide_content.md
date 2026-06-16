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

## Slide 14 - Weak Label Construction

**Slide content**

The original weak-label idea used a VIIRS dark/bright rule to separate slum and formal pixels. In practice, that rule collapsed to zero slum pixels, making the training labels unusable.

We changed the labeling rule to region-type weak supervision:

| Input | Rule | Output |
|---|---|---|
| Informal region + built pixel | Label as slum | Class 1 |
| Formal control region + built pixel | Label as formal | Class 2 |
| Non-built pixel | Unknown | Class 0 |

Built pixels come from GHSL built-up and Dynamic World built class. The high-confidence mask is the built-up portion of each known region. This is not manual ground truth, but it is a defensible weak-supervision strategy for a region-level benchmark.

**Current verification result**

| Check | Status |
|---|---|
| Total tiles | 720 |
| HC pixels | 11,308,540 |
| Slum pixels | 7,382,955 |
| Formal pixels | 3,923,135 |
| Gate result | Passed |

**Designer instructions**

Create a label-flow diagram:

`Region type + built-up mask -> noisy label + HC mask`

Show class colors:
- 0 unknown
- 1 slum
- 2 formal

---

## Slide 15 - Compute-Efficient Colab Pipeline

**Slide content**

The pipeline is designed to avoid wasting Colab compute units. The expensive parts are run once and cached on Google Drive. Later experiments train small heads over cached tensors.

**Pipeline phases**

| Phase | Purpose | Compute cost | Cache behavior |
|---|---|---|---|
| P0 | Setup, Drive, repo, config | Low | Re-run every session |
| P1 | GEE exports and tiling | CPU/GEE | Skips existing exports |
| P2 | Weak labels and confidence | CPU | Rebuilds after label reset |
| P3.1 | MoonViT feature extraction | High GPU once | Cached features |
| P3.2 | SAS-Net clean tiles | GPU once | Cached clean tiles |
| P4 | Experiment matrix | Moderate GPU | Registry skips completed runs |
| P5 | Figures and tables | CPU | Regenerates from CSV |

This structure matters because LocateAnything inference is expensive. Feature caching lets us run many ablations without repeatedly calling the VLM.

**Designer instructions**

Create a horizontal pipeline with cache icons above P3.1, P3.2, and P4 registry.

---

## Slide 16 - Experimental Design and Metrics

**Slide content**

We run three main experiment groups.

| Experiment | Question | Main comparison |
|---|---|---|
| Exp 1: SAS-Net ablation | Does appearance normalization help? | Raw/classical/SAS-Net |
| Exp 2: Fusion ablation | Which signal fixes dense-formal confusion? | Visual -> language -> socioeconomic |
| Exp 3: Leave-one-region-out | Does the model generalize across Dhaka regions? | Hold out one region at a time |

**Main metrics**

| Metric | What it measures |
|---|---|
| HC-IoU | Primary segmentation score on high-confidence pixels |
| All-IoU | Broader score over all labeled pixels |
| Precision/Recall/F1 | Classification balance |
| FPR on formal controls | Whether dense formal areas are falsely predicted as slum |
| Korail recall | Whether the model still detects a known major slum |

The central result will come from Experiment 2. The strongest evidence would be a full model that improves HC-IoU while reducing false positives in formal dense controls.

---

# Speaker 5: Findings, Discussion, and Next Steps

## Slide 17 - Verified Data State Before Training

**Slide content**

Before Phase 4 training, we ran a label verification gate to ensure the dataset is not degenerate. This was necessary because an earlier label rule produced all-zero slum metrics.

**Verification output**

| Quantity | Value |
|---|---:|
| Total tiles | 720 |
| Total HC pixels | 11,308,540 |
| Unknown pixels | 490,390 |
| Slum pixels | 7,382,955 |
| Formal pixels | 3,923,135 |

**Interpretation**

The dataset now contains both target classes. Informal regions contain slum-labeled built pixels, and formal control regions contain formal-labeled built pixels. This means Phase 4 training is safe to run, and all-zero metrics should no longer be caused by missing slum labels.

---

## Slide 18 - Main Findings Placeholder

**Slide content**

This slide will be completed after Colab Phase 4 and Phase 5 finish.

**Insert final Experiment 2 results here**

| Model | HC-IoU | All-IoU | Precision | Recall | FPR on formal controls |
|---|---:|---:|---:|---:|---:|
| `baseline_cnn` | TBD | TBD | TBD | TBD | TBD |
| `vlm_visual` | TBD | TBD | TBD | TBD | TBD |
| `vlm_lang` | TBD | TBD | TBD | TBD | TBD |
| `full` | TBD | TBD | TBD | TBD | TBD |

**Expected analysis to write after results**

- Compare optical-only baseline against VLM-based variants.
- Check whether language grounding improves over neutral VLM features.
- Check whether socioeconomic fusion reduces false positives in Old Dhaka, Gulshan-Baridhara, Dhanmondi, and Uttara.
- Identify the best-performing model and report the headline metric.

**Designer instructions**

Reserve space for two charts:
1. Bar chart: model variant vs. HC-IoU.
2. Line or step chart: visual-only -> language -> VIIRS -> population -> roads -> poverty -> full, with HC-IoU and FPR-control.

---

## Slide 19 - Limitations and Scientific Caveats

**Slide content**

The method is promising, but the limitations must be stated clearly.

| Limitation | Why it matters | How we address or disclose it |
|---|---|---|
| Weak labels are region-type labels | They are not manual pixel-level ground truth | Report as weak supervision, not definitive annotation |
| 10 m Sentinel-2 is coarse | Small lanes and roofs are hard to see | Use VLM features cautiously; future high-res grounding |
| Roads and poverty layers include placeholders/proxies | Some ablation rows may not be fully meaningful | Replace assets before final paper claims |
| Some regions are close geographically | Leave-one-region-out may contain leakage | Discuss and consider spatial-block split |
| Wet-season composites failed | Seasonal robustness is limited | Report dry-season scope |

The honest framing is that BanglaSlumNet is a Dhaka weak-supervision benchmark and method prototype, not a finished national mapping system.

---

## Slide 20 - Conclusion and Next Steps

**Slide content**

BanglaSlumNet addresses a specific failure mode in slum detection: optical-only models confuse dense formal neighborhoods with informal settlements in Dhaka.

The proposed solution combines:

| Component | Contribution |
|---|---|
| Sentinel-2 imagery | Visual structure |
| LocateAnything language grounding | Concept-level slum/formal distinction |
| Socioeconomic context | Non-visual evidence for service access and settlement condition |
| Cross-attention fusion | Learns how context should modulate visual interpretation |
| Weak-label benchmark | Tests both informal regions and formal dense controls |

**Immediate next steps**

1. Complete Phase 4 experiment matrix.
2. Run Phase 5 to generate figures, tables, and `RESULTS.md`.
3. Insert real numbers into Slide 18.
4. Replace placeholder roads/poverty layers before strong ablation claims.
5. Write the manuscript results and limitations sections from recorded outputs only.

**Closing message**

The core idea is simple: in dense megacities, slum mapping should not rely on optical density alone. It needs semantic and socioeconomic context.

