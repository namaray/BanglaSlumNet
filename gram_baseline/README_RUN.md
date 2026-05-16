# Reproducing the GRAM-on-Dhaka Zero-Shot Baseline

This bundle lets you reproduce the GRAM-fails-on-Dhaka experiment from
`FINDINGS.md` on your own machine in ~5 minutes (CPU is fine).

## What this experiment does (the process)

The pipeline has four stages:

```
   ESRI World           GRAM MoE             slum prob              composite
   Imagery tiles  ─►    ViT-B5 (256×256) ─►  map (0–1)        ─►    figure +
   (z=16, ~1.2m/px)     pretrained ckpt      per pixel              CSV stats
```

1. **Fetch imagery** (`fetch_dhaka_tiles.py`)
   Pulls 27 ESRI World Imagery tiles at zoom 16 over three Dhaka regions
   (Korail, Mirpur, Old Dhaka), 3×3 grid per region. ESRI is the same source
   GRAM was trained on, so we eliminate "input format" as a confound.

2. **Load model + checkpoint** (`gram_baseline.py → build_model`)
   - Constructs `mit_b5_MOE` from `model.py` (the GRAM MoE MixVisionTransformer).
   - Loads `checkpoint/MOE_epoch_2_v2.pth` (released by DS4H-GIS; final
     supervised+MoE epoch).
   - Strips the `module.` prefix because the checkpoint was saved from a
     `DataParallel`-wrapped model. After stripping, the load is clean:
     `missing=0, unexpected=0`.

3. **Run inference per tile** (`gram_baseline.py → infer_one`)
   - Resize each tile to 256×256, ImageNet-normalize (the exact preprocessing
     used in GRAM's `main_moe.py`).
   - Forward pass:
     `seg, dom_logits, mi_loss = model(image, country_idx)`
     - `seg` is `(1, 2, 256, 256)` — channel 1 is the "slum" class.
     - `country_idx` is a domain selector for the MoE.
   - Apply softmax over channels, take channel 1 → 256×256 slum-probability map.

   **Domain index probing.** GRAM's training metadata (which integer maps to
   which of the 12 cities) is not published. The script sweeps a few candidate
   indices on a Korail tile and picks the one with the highest mean activation
   (most "confident" expert) — empirically `idx=6` wins.

4. **Save outputs and aggregate**
   - Per tile: `{loc}_x{x}_y{y}_prob.npy` (raw probability map).
   - Per location: 3-column mosaic PNG: `[input | heatmap | binary overlay]`.
   - Summary: `gram_baseline_summary.csv` with `mean_prob`, `max_prob`,
     `pct_slum_p50`, `pct_slum_p70` per tile.
   - Composite figure for the paper: `gram_dhaka_summary_figure.png`
     (run `python3 make_summary_chart.py` after the main run).

## Setup

Requires Python 3.9+. CPU only is fine; one inference pass on 27 tiles takes
~2–3 minutes on a modern laptop.

```bash
# 1. Create a venv (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install deps
pip install -r requirements.txt

# 3. (Once) download the GRAM checkpoint
#    URL is in the GRAM repo's checkpoint/ folder on GitHub.
#    Place it at: checkpoint/MOE_epoch_2_v2.pth
#    The bundle's checkpoint/ folder already has it — skip this step.

# 4. Fetch Dhaka imagery (needs internet, ~5 MB)
python3 fetch_dhaka_tiles.py

# 5. Run the baseline
python3 gram_baseline.py

# 6. Build the composite figure
python3 make_summary_chart.py
```

Outputs land in `./outputs/`.

## Important code-level notes

### Two CPU patches in `model.py`
Original GRAM hardcodes `.cuda()` in `model.py` lines 189 and 330. If you have
a GPU, no patch needed — the bundle ships the patched version (CPU-safe) so it
runs on either. The fix is just removing the `.cuda()` suffix:

```python
# line 189
self.MI_task_gate = torch.zeros(self.domain_num, self.expert_num)  # was .cuda()

# line 330
MI_loss = ((self.MI_task_gate + 1e-4) * torch.log(self.MI_task_gate / (P_TI * P_EI) + 1e-4)).sum()
```

If you do have a GPU and want to go back to the original behavior, restore the
`.cuda()` calls — they're harmless on CPU as long as `DEVICE = cuda`.

### timm compatibility
GRAM's `requirements.txt` pins `timm==0.3.2`, which imports
`from torch._six` — gone in modern torch. Use `timm>=0.6` and the imports in
`model.py` all still resolve. The requirements.txt in this bundle already does
this.

### Checkpoint loading
```python
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
state = ckpt.get("state_dict", ckpt)
# Strip DataParallel prefix
new_state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
model.load_state_dict(new_state, strict=False)
```
`weights_only=False` is needed because the checkpoint pickles non-tensor
objects (training state). Trust this file only because you downloaded it from
the official DS4H-GIS repo.

### Forward signature
```python
seg, dom_logits, mi_loss = model(image_tensor, country_idx_tensor)
#   seg:        (B, 2, 256, 256) — channel 1 = "slum"
#   dom_logits: (B, num_domains) — auxiliary domain classifier
#   mi_loss:    scalar — mutual-information regularizer (training only)
probs = torch.softmax(seg, dim=1)[0, 1]  # 256×256 slum prob
```

## How to extend

- **Add more cities/regions**: edit `LOCATIONS` in `fetch_dhaka_tiles.py`.
- **Different zoom**: change `ZOOM = 16`. Note GRAM was trained at z=18
  (~0.3 m/px) too, so trying z=18 is a legitimate ablation — just `4×` more
  tiles per region.
- **Full domain sweep**: replace `DOMAIN_CANDIDATES = [0,4,5,6]` in
  `gram_baseline.py` with `list(range(12))` to evaluate all 12 training cities
  as domain priors. This is the cleanest "we tried everything" ablation for
  the paper.
- **Add metrics**: once you have ground-truth Dhaka masks, drop a
  `compute_iou(prob, mask)` call inside the per-tile loop.

## File map

```
gram_baseline/
├── README_RUN.md            ← this file
├── FINDINGS.md              ← writeup of results + paper implications
├── requirements.txt
├── fetch_dhaka_tiles.py     ← stage 1: imagery
├── gram_baseline.py         ← stages 2–4: inference + outputs
├── make_summary_chart.py    ← composite figure for paper
├── model.py                 ← GRAM model (CPU-patched)
├── main_moe.py              ← original training script (reference only)
├── main_moe_pl_v3.py        ← (reference only)
├── dataloader.py            ← (reference only)
├── augmentation.py          ← (reference only)
├── utils.py                 ← (reference only)
├── checkpoint/
│   └── MOE_epoch_2_v2.pth   ← 98 MB, official DS4H-GIS release
├── dhaka_tiles/             ← populated by fetch_dhaka_tiles.py
└── outputs/                 ← populated by gram_baseline.py
```

## Attribution

- Model + code adapted from [DS4H-GIS/GRAM](https://github.com/DS4H-GIS/GRAM),
  AAAI 2026 Outstanding Paper Award. The repo declares no license, so treat
  the model as research-only; do not redistribute the checkpoint.
- ESRI World Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User
  Community — used here under the standard public tile-service terms for
  research and review.
