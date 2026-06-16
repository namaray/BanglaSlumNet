# BanglaSlumNet v4 — Paper Reference & Current State

> Supersedes `BanglaSlumNet_v3.md` (now stale). This is the single reference for
> writing the manuscript: the contribution, method, data, experiments, exactly what
> was built, what works, what's still weak, and what numbers to report. Engineering
> hand-off (how we run things) is in `AGENT_HANDOFF.md`. The original build contract
> is `docs/SPEC.md` (+ the "v4 amendments" block appended to it).

---

## 1. Contribution (one sentence)

We show that a frontier visual-grounding VLM (**NVIDIA LocateAnything-3B**), prompted
with the conceptual distinction between informal and dense-formal settlement and fused
via cross-attention with socioeconomic priors (nighttime lights, population, roads,
poverty), corrects the dense-megacity failure mode that defeats optical-only foundation
slum detectors (e.g. GRAM) on Dhaka.

## 2. The problem / failure mode

Optical-only foundation models cannot separate Dhaka's dense **formal** core (Old Dhaka
brick housing, Gulshan) from real **informal** settlements (Korail, etc.) — RGB texture
is genuinely ambiguous at scale. The disambiguating signal must come from **outside the
RGB channel**. We add two orthogonal external signals:

1. **Language** — prompt the VLM with the *concept* of a slum ("dense informal
   settlement: small irregular rooftops, narrow unpaved gaps, no road grid") vs. formal
   housing, so the model is told what a slum *is*, not just "what is built-up."
2. **Socioeconomics** — VIIRS nighttime lights, WorldPop / GHS-POP population, roads,
   poverty. Slums are dark-at-night and road-poor; the affluent formal core is bright and
   road-connected, even when they look alike in RGB.

## 3. Method / architecture

```
Sentinel-2 tile ─► SAS-Net (Stage 1: scene/appearance disentangle) ─► clean tile
                                                                         │
LocateAnything-3B (frozen MoonViT) ── grounding prompts (slum / formal) ─┤
   → grounding-map features V  [prompt-specific box-coverage]            │
                                                                         ▼
Socioeconomic tensor E (VIIRS, WorldPop, GHS-POP, roads, poverty) ─► CrossAttention(Q=V,K=E,V=E)+V
                                                                         ▼
                                            lightweight UNet decoder ─► per-pixel slum mask
```

- **Stage 1 — SAS-Net** (`src/models/sasnet.py`): AdaIN-based scene/appearance separation;
  renders every tile at a fixed "clean" reference appearance. Trained once, clean tiles cached.
- **Stage 2 backbone — LocateAnything-3B**, frozen (`src/locate_anything/`). MoonViT vision
  encoder (`model.vision_model`, hidden 1152). We use **grounding-map** features: run the
  slum-prompt and formal-prompt groundings per tile, rasterize predicted boxes into dense
  32×32 coverage maps (prompt-specific → carries the language signal). Cached to `.npy`.
  (Alternative `hidden_state` mode = raw 1152-d MoonViT patch features, but prompt-agnostic.)
- **Fusion** (`src/models/fusion.py`): multi-head cross-attention, `F = CrossAttn(Q=V,K=E,V=E)+V`,
  with a per-channel `channel_mask` for socioeconomic ablations.
- **Decoder** (`src/models/decoder.py`): UNet-style upsampling → `[1, tile, tile]` sigmoid.
- **Only the fusion + decoder train** (~3.5–10 M params). VLM is frozen; features cached →
  training is seconds/epoch over cached tensors.

### Four model configs (`src/models/banglaslumnet.py::build_model`)
| config | backbone | prompt | socioeconomic fusion |
|---|---|---|---|
| `baseline_cnn` | SegFormer-B0/ResNet-UNet on RGB | – | off |
| `vlm_visual` | LocateAnything features | neutral | off |
| `vlm_lang` | LocateAnything features | slum vs formal | off |
| `full` | LocateAnything features | slum vs formal | on (all channels) |

## 4. Data

- **Region benchmark:** **12 Dhaka regions** in `config/regions_dhaka.yaml` —
  **8 informal** (korail, bhashantek, karail_extension, kamrangirchar, kallyanpur,
  hazaribagh, tongi, mirpur_beribadh) and **4 formal-dense control**
  (old_dhaka, gulshan_baridhara, dhanmondi, uttara). Each ~3 km box.
- **Imagery:** Sentinel-2 L2A, dry-season best-pixel composites, bands B2/B3/B4/B8, 10 m.
  Years 2020–2023 (wet season fails — monsoon clouds; dry only). Exported full-extent
  (unmask) so tiles aren't cropped to valid-data bbox.
- **Tiling:** **128 px** tiles (1.28 km), train stride 64 / eval stride 128. Labels and the
  socioeconomic tensor are **reprojected onto the RGB grid** per tile → alignment guaranteed.
  → **720 tiles** (60 per region), stratified-by-region split (≈504 train / 108 val / 108 test).
- **Weak labels (CURRENT = region-type supervision):** built-up mask = GHSL built OR
  Dynamic-World built (class 6); every built pixel in a known-**informal** region → **slum (1)**,
  in a known-**formal** region → **formal-dense (2)**; off-built → unknown (0). HC mask = built
  pixels (high-confidence for the region's known type).
  - **Why region-type and not the spec's VIIRS rule:** the per-pixel OSM∩GHSL∩VIIRS rule
    collapsed to **0 slum / all formal** (VIIRS dark/bright threshold never separated the
    neighborhoods; see §7). Region-type labeling is a standard, reliable weak-supervision
    scheme here and guarantees both classes. This is the single most important methodological
    change from v3/spec.
- **Socioeconomic tensor** (`gee/export_socioeconomic.py`): VIIRS, WorldPop, GHS-POP,
  osm_roads (PLACEHOLDER: accessibility proxy), wb_poverty (PLACEHOLDER: zeros), GHSL built.
  Re-export with real roads/poverty assets before trusting those two ablation rows.
- **Direction B (LocateAnything label validation):** built but **OFF** for now
  (`use_locate_anything_validation: false`). LA grounding on 10 m S2 is unreliable and zeroed
  the 4-signal HC. The spec intends LA validation on high-res ESRI z16 (~1.2 m) tiles; that's
  future work. LA *features* (grounding maps) are still used and are ~89% non-zero.

## 5. Experiments (what each shows)

- **Exp 1 — atmospheric ablation** (`exp1_*`): raw vs classical vs SAS-Net (backbone fixed
  `vlm_visual`). Does Stage-1 normalization help?
- **Exp 2 — socioeconomic+language fusion (CENTRAL)** (`exp2_*`): incremental rows
  visual-only → +language → +VIIRS → +population → +roads → +poverty → full, plus a
  controlled `baseline_cnn`. Headline: **FPR on the formal-dense control falls while slum
  recall holds** as orthogonal signals are added.
- **Exp 3 — leave-one-region-out** (`exp3_loro_*`): train on 11 regions, test on the held-out
  one. Critical fold: train without Old Dhaka, test on it — do socioeconomic priors transfer?

Matrix is in `config/experiments.yaml`; the orchestration loop (notebook P4.2) trains each
row over cached features, writes `results/runs/<id>.json` + appends `results/tables/all_runs.csv`.

### Metrics (`src/eval/metrics.py`)
HC-IoU (primary), All-IoU, precision/recall/F1, **FPR-on-control** (Old Dhaka, Gulshan),
Korail recall, SSIM/PSNR (SAS-Net). Figures auto-generate from the CSV via
`scripts/make_paper_figures.py` (fig1 failure-repro, fig2 SAS-Net, fig3 the central fusion
ablation, fig4 LORO, fig5 master table, fig6 qualitative, fig7 PR, fig8 confidence strata).

## 6. Implementation facts (verified live)

- LocateAnything-3B: `AutoModel.from_pretrained(trust_remote_code=True)`, BF16; needs a
  `decord` stub (no py3.12 wheel; image-only). Real API = chat messages +
  `processor.py_apply_chat_template` + `process_vision_info` + custom `model.generate(...)`;
  box format `<box><x1><y1><x2><y2></box>`, coords [0,1000]. Runs on A100/L4 (MagiAttention →
  SDPA fallback). Vision encoder returns a **list** of patch tensors (256-px input → 100
  merged tokens → 10×10 grid, 1152-d).
- Frozen-encoder asserted; ~3.5–10 M trainable head params per config.
- Everything caches to Drive; a run registry makes the matrix resumable.

## 7. Status, known limitations & honest caveats (for the paper)

**Status (2026-06-16):** full pipeline runs end-to-end on Colab. Data = 720 balanced tiles.
First two full runs gave all-zero metrics; root cause found = **degenerate labels (0 slum)**
from the VIIRS rule. Fixed by switching to **region-type labeling**. Awaiting the verify gate
(notebook P4.1b: must show slum>0 and HC>0) and the subsequent Phase-4 run for real numbers.

**Limitations to disclose:**
1. **Weak labels are region-level**, not per-pixel ground truth — every built pixel in a slum
   region is called slum. Coarse; defensible as weak supervision but state it plainly. (A
   manually-annotated HC test subset would strengthen the eval.)
2. **LA grounding on 10 m S2 is weak** — features are coarse 32×32 box-coverage maps. The
   intended high-res (ESRI z16) grounding is future work.
3. **`osm_roads` / `wb_poverty` are placeholders** → the roads/poverty ablation rows aren't
   meaningful until real assets are swapped in.
4. **Adjacent regions** (Korail/Karail-extension/Gulshan ~2 km apart, ~3 km boxes overlap) →
   the LORO test has some leakage; spatial-block splitting would tighten it.
5. **Dry-season only** (monsoon composites empty).

**What to report once P4 produces non-zero numbers:** the master comparison table
(baseline_cnn vs vlm_visual vs vlm_lang vs full, + GRAM if wired), the Exp-2 incremental
fusion curve (HC-IoU ↑, FPR-on-control ↓), Exp-1 SAS-Net deltas, Exp-3 LORO, and qualitative
overlays — all regenerated from `results/tables/all_runs.csv`. Do **not** report the earlier
all-zero runs.
