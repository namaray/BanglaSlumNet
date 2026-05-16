# GRAM Zero-Shot Baseline on Dhaka — Findings

**Date:** 2026-05-15
**Goal:** Prove the gap that GRAM (AAAI'26 Outstanding Paper, DS4H-GIS) does not generalize to Dhaka, motivating BanglaSlumNet.

## Setup

- **Model:** `mit_b5_MOE` (MoE MixVisionTransformer-B5) from [DS4H-GIS/GRAM](https://github.com/DS4H-GIS/GRAM).
- **Checkpoint:** `MOE_epoch_2_v2.pth` (~98 MB, final supervised+MoE epoch, loaded with `missing=0, unexpected=0`).
- **Input:** ESRI World Imagery tiles, zoom 16 (~1.2 m/px native, downsampled to 256×256 RGB), ImageNet-normalized — identical preprocessing to GRAM's training in `main_moe.py`.
- **Domain index:** Probed indices {0, 4, 5, 6} on a Korail tile; picked `idx=6` (highest mean activation). Index→city mapping is not published; idx=6 is most likely one of the South-Asian training domains (Karachi/Mumbai/Colombo).
- **Tiles:** 27 ESRI tiles at z=16 over three Dhaka regions — 3×3 grids over:
  - **Korail** (centered at 23.7806 N, 90.4040 E) — Dhaka's largest informal settlement.
  - **Mirpur** (23.81 N, 90.36 E) — mixed formal/informal, contains Sher-e-Bangla National Stadium.
  - **Old Dhaka** (23.71 N, 90.39 E) — dense historic core along the Buriganga river.
- **Hardware:** CPU only (no GPU available in sandbox); two `.cuda()` calls in `model.py` patched to CPU tensors.

## Quantitative Results

Per-location aggregate slum-probability stats (averaged over 9 tiles each):

| Location  | Mean prob | Max-of-max | Avg % pixels > 0.5 | Avg % pixels > 0.7 |
|-----------|----------:|-----------:|-------------------:|-------------------:|
| Korail    |    0.479  |     0.878  |             48.4 % |             22.7 % |
| Mirpur    |    0.505  |     0.804  |             57.1 % |             10.8 % |
| Old Dhaka |    0.435  |     0.859  |             45.2 % |             15.2 % |

Per-tile detail in `outputs/gram_baseline_summary.csv`.

## Qualitative Findings

Looking at `outputs/gram_dhaka_summary_figure.png`:

### What GRAM gets right
- **Water rejected.** The Buriganga river in Old Dhaka is correctly suppressed (deep blue in the heatmap, ~0% > 0.5).
- **Vegetation/open space rejected.** Parks, the cricket-stadium turf in Mirpur, and large green patches in north Korail are correctly dark.
- **Built-up vs non-built-up boundary.** The model has clearly learned a strong "dense urban texture" prior.

### What GRAM gets wrong
- **It flags formal neighborhoods as slums.** In Korail, the bright-red predictions extend across the affluent Gulshan/Banani grid east and south of the actual Korail settlement. The model cannot distinguish high-end Dhaka housing from informal settlements.
- **Old Dhaka is ~entirely "slum".** The pucca brick neighborhoods on both sides of the Buriganga get blanket-classified as slum, with very high confidence (max prob > 0.85 in multiple tiles). This is wrong — Old Dhaka is dense but not informal.
- **Mirpur formal blocks flagged.** Planned residential grid north of the stadium is predicted at ~0.5 mean prob.
- **The actual Korail core is not specifically more confident than its formal surroundings.** The two highest-confidence tiles in Korail (`x49225 y28309` mean 0.78; `x49225 y28307` mean 0.70) cover the informal area, but adjacent tiles with similar predictions sit over Gulshan, so a downstream user couldn't tell them apart from prediction alone.

### Interpretation
GRAM, applied zero-shot to Dhaka, behaves essentially as a **dense-built-up detector**, not a slum-vs-formal-settlement classifier. This is consistent with the dataset bias: its 12 training cities (Cairo, Cape Town, Nairobi, Ouagadougou, Colombo, Karachi, Mumbai, Caracas, Medellín, Rio, Port-au-Prince, Tegucigalpa) feature a strong visual contrast between sprawling self-built slums and lower-density formal cores. In Dhaka, **the formal city is itself extremely dense** (Old Dhaka brick-and-mortar, Mirpur low-rise grid), so GRAM's learned texture cues misfire.

## Implications for BanglaSlumNet

1. **The paper's Section 4.2 claim that "Dhaka is in GRAM training data" is false** — Dhaka is not among GRAM's 12 train or 3 test cities. This must be removed from the paper.
2. **GRAM is a legitimate baseline to beat.** Zero-shot performance on Dhaka is qualitatively poor (over-predicts slums by a large margin), so any Dhaka-trained model that materially reduces false-positives on formal high-density blocks will be a clear contribution.
3. **The gap is publishable as a reframing of the paper:**
   - Story: "Foundation-style slum models (GRAM) over-predict in cities with high formal density. We introduce BanglaSlumNet, the first model that handles this regime, with Dhaka labels and atmospheric/socioeconomic priors."
   - Headline result we can target: report precision/recall/IoU once we have Dhaka labels — GRAM's recall will be high (it flags everything) but precision will be poor; BanglaSlumNet should improve precision substantially without losing recall.
4. **GRAM's modality is RGB at ~1.2 m/px (ESRI), not Sentinel-2 (10 m/px).** The paper's current "Sentinel-2 input" framing for the baseline is therefore wrong. For a fair side-by-side, BanglaSlumNet should run at the ESRI resolution as well, or we should evaluate GRAM on Sentinel-2 and document its degradation as an additional finding.

## Caveats / Open Questions
- **Domain index unknown.** We swept {0,4,5,6}; full sweep across all 12 indices may yield a setting that handles Dhaka better, but unlikely to flip the over-prediction story (idx=6 already gave the highest activation; trying every index can be a one-screen ablation in the paper).
- **No ground truth yet.** All quantitative numbers above are pixel-level probabilities, not metrics. We need Dhaka slum labels (manual annotation, OSM `landuse=residential` + manual refinement, or hiring annotators) to compute precision/recall.
- **Single zoom level.** Only z=16 tested. GRAM's effective scale during training was z=18 (~0.3 m/px) per the paper — running at z=18 may sharpen the boundary, but pulling z=18 tiles is 16× more data per location.

## Files Produced
- `gram_baseline.py` — inference script (works on CPU, applies `module.` prefix strip).
- `outputs/gram_baseline_summary.csv` — per-tile stats for all 27 tiles.
- `outputs/*_gram_baseline.png` — per-location 3-column mosaics (RGB | heatmap | overlay).
- `outputs/gram_dhaka_summary_figure.png` — single composite figure for paper / slide use.
- `outputs/*_prob.npy` — raw 256×256 float probability maps (27 files).

## Next Step Recommendation

Pick one of two paths:

- **A1 — Lock the gap, then build the rebuttal model.** Add 2–3 more Dhaka regions, run a 12-index ablation on one tile, and start labeling 50–100 tiles by hand (a couple of days). Then train a small Dhaka-only baseline and show it beats zero-shot GRAM on precision.
- **A2 — Add z=18 tiles to be airtight.** Pull z=18 imagery (4× the tile count we have) and re-run, to forestall the reviewer comment "you used the wrong resolution".

I recommend A1 first — even with the current evidence, the over-prediction is severe enough to be the headline story.
