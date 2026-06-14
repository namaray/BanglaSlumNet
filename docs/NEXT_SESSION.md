# Next Session Plan — BanglaSlumNet

> Hand-off doc to start a fresh Claude Code session. Self-contained: it explains the
> project, the setup, how we work, the exact current state, and a prioritized plan.
> Read this top-to-bottom before touching anything.

---

## 0. One-paragraph context

BanglaSlumNet detects informal settlements (slums) in Dhaka from Sentinel-2 imagery using
**NVIDIA LocateAnything-3B** (a frozen vision-language grounding model) prompted with the
*concept* of a slum, fused via cross-attention with **socioeconomic** layers (nighttime
lights, population, roads, poverty). The thesis: optical-only models can't separate slums
from the dense formal core in Dhaka; language + socioeconomic signals fix that. The full
pipeline is **built and working end-to-end**; the open problem is **dataset scale** (only 75
tiles, slum class badly under-represented), so current metrics are ≈0. The next session's job
is to **scale/rebalance the data and get real numbers** — not to rebuild the system.

The authoritative spec is `docs/SPEC.md`. A supervisor-facing summary is
`docs/SUPERVISOR_BRIEF.md`.

---

## 1. Repo & environment

- **GitHub:** `https://github.com/namaray/BanglaSlumNet`, work on **`main`** (the user wants
  every change pushed to `main` directly; old pre-project history is preserved in branch
  `legacy`). End commits with the Claude co-author trailer.
- **Local machine:** Windows + PowerShell. Only `pyyaml` is installed locally (no torch/
  numpy/rasterio) — so locally you can only `python -m py_compile` to syntax-check and
  validate YAML/notebook JSON. **All real execution happens in Colab.**
- **Colab runtime:** Python 3.12, A100 (Ampere). MagiAttention unavailable → SDPA fallback
  (fine). Use background execution for long runs.

### How code vs. data are split (important)
- **Code** is cloned fresh each Colab session to **local disk**: `/content/BanglaSlumNet`.
  Do NOT clone the repo onto Drive — git-on-Drive fails (dubious-ownership, detached HEAD).
- **Data / results / model cache** live on **Drive**: `/gdrive/MyDrive/BanglaSlumNet/`
  (`data/`, `results/`, `model_cache/`). These persist across sessions and are what the user
  pulls locally for the paper.

### How we work (the loop)
1. Make code/notebook changes locally, `py_compile` to check, commit + push to `main`.
2. In Colab: **Restart session → reopen the notebook from the GitHub tab → Run all.**
   (Uploading the notebook gives a stale copy — always open from the GitHub tab.)
3. Heavy phases are cached on Drive and skip on re-run; the experiment **registry** makes the
   matrix resumable. So re-running after a fix is cheap.
4. For notebook *cell* changes the user must reopen from GitHub; for `.py` changes a restart
   picks them up. (This staleness caused much friction — flag it.)

---

## 2. Exact current state (what's done / cached on Drive)

Already computed and cached under `/gdrive/MyDrive/BanglaSlumNet/` — do **not** recompute
unless the data changes:
- ✅ LocateAnything-3B downloaded to `model_cache/` (~8 GB).
- ✅ GEE exports for 5 regions (Sentinel-2 **dry-season only** — wet-season failed: monsoon
  clouds, expected), weak labels, socioeconomic tensor.
- ✅ Tiled to **75 tiles @ 128 px** (`data/dataset_manifest.json`), 4-signal HC masks
  (`la_validation.json`, ~152k HC pixels).
- ✅ MoonViT features cached (`data/features_cache/*.npy`) for `full` (slum+formal prompts)
  and `vlm_visual` (neutral prompt), in `grounding_map` mode (32×32, 1–2 channels).
- ✅ SAS-Net trained, clean tiles cached.
- ✅ Experiment matrix runs to completion; results CSV + figures generate.

### Verified facts about LocateAnything (don't re-derive)
- Vision encoder: `model.vision_model` (MoonViT, hidden 1152; returns a **list** of patch
  tensors; 256-px input → 400 patches pre-merge → 100 post-merge → 10×10 grid).
- Loads with `trust_remote_code=True`, BF16; requires a `decord` stub (image-only) — handled
  in `src/locate_anything/_compat.py`.
- Real API = chat messages + `processor.py_apply_chat_template` + `process_vision_info` +
  custom `model.generate(...)`; box format `<box><x1><y1><x2><y2></box>`, coords in [0,1000].
  All ported in `src/locate_anything/worker.py`.
- `generate` works on A100 (MagiAttention → SDPA).

---

## 3. THE problem to solve next: dataset scale & balance

**Symptom:** training loss flat ~0.62, HC-IoU ≈ 0 for every config. **Cause:** only 75 tiles,
and the slum (informal) regions contribute ~5 near-duplicate tiles each vs. ~30 for the formal
controls. The head has essentially no unique slum signal.

**Why so few:** the informal neighbourhoods are small (~1–3 km) and three of them (Korail,
Karail-extension, Gulshan) are adjacent, so large non-overlapping boxes aren't possible, and
only dry-season composites exist (×4 years = near-duplicates of the same location).

### Options to fix (discuss with the user, pick one or combine)
1. **Add more slum areas across Dhaka** (e.g. additional well-known settlements) as new
   regions in `config/regions_dhaka.yaml`, re-export via GEE, re-tile. More *unique*
   locations is the goal — this is the cleanest scientific fix.
2. **Enlarge informal boxes** to capture surrounding informal fabric (accepting some overlap),
   and/or **reduce tile size** further (e.g. 96 px) for more samples per region.
3. **Recover wet-season** composites by relaxing the cloud threshold or widening date windows
   (more temporal samples; lower quality).
4. **Reconsider the unit of analysis**: instead of 5 tiny fixed neighbourhoods, define 2–3
   larger Dhaka study areas tiled densely, with per-pixel weak labels doing the slum/formal
   supervision (most data-efficient; changes the benchmark framing — confirm with user).

**Target:** tens–hundreds of *unique* slum tiles, with both classes present in train/val/test.
The split is already stratified by region (`build_manifest_from_records` in
`src/data/tiles.py`).

---

## 4. Secondary improvements (after data is fixed)

- **Feature mode:** the central language ablation needs `grounding_map` (prompt-specific), but
  it's coarse (32×32, 1–2 ch). For the *visual* configs (`vlm_visual`, and as a richer option
  generally) try `hidden_state` (1152-dim MoonViT features). Decision flag `D1` in P0.5 /
  `config/default.yaml`. Consider: hidden_state for visual configs, grounding_map for the
  language configs (the extractor already auto-selects prompts per config).
- **Replace placeholder data layers** before trusting roads/poverty ablation rows:
  `gee/export_socioeconomic.py` → `osm_roads` (currently Oxford accessibility proxy, also
  deprecated asset) and `wb_poverty` (currently constant 0). Find real GEE assets (GRIP4
  roads; a gridded poverty / relative-wealth-index layer).
- **Training:** loss was flat — once data is real, sanity-check that train loss *decreases*;
  if not, revisit LR (`config.train.lr`), class weighting, or feature richness.

---

## 5. Concrete first actions for the next session

1. Read `docs/SPEC.md` §5 (data) and §9 (experiments), and this file.
2. Ask the user which data-scaling option (§3) they want. Default recommendation: **option 1**
   (add more unique Dhaka slum regions) — it's the most defensible for the paper.
3. Edit `config/regions_dhaka.yaml` (and the GEE exporters if adding regions), push, and have
   the user re-run notebook **Phase 1 (GEE export) → P1.5 (tiling) → P2 (labels) → P3
   (features) → P4 (experiments) → P5 (figures)**. Only the *new* tiles incur VLM cost
   (per-tile caching).
4. Sanity gate: after tiling, check `Total tiles` and per-region counts (want a balanced,
   larger set); after training, confirm **train loss decreases** and **HC-IoU > 0** on at
   least the `full` config before running the whole matrix.
5. Regenerate figures (`scripts/make_paper_figures.py`) and read `docs/RESULTS.md`.

---

## 6. Gotchas / lessons (so the next session doesn't relive them)

- Open notebooks from the **GitHub tab**, not by upload (stale-cell trap).
- Clone code to `/content`, never Drive. Add `safe.directory` if you ever must git on Drive.
- Don't hard-pin numpy/Pillow/decord on Colab (Python 3.12 wheels break); `requirements_colab.txt`
  is already cleaned.
- BCE must run in float32 under autocast (already fixed in `losses.py`).
- `num_workers=0` in Colab DataLoaders avoids worker-cleanup spam; warnings are filtered in P4.2.
- Eval needs HC tiles in the test split; the split is stratified and eval falls back to all
  test tiles if the HC-only set is empty (already handled).
- The user is non-expert in the tooling and prefers: clear step-by-step instructions, you
  making the engineering decisions with a stated recommendation, and pushing fixes to `main`
  for them to pull. They are CU-sensitive — never burn Colab compute on avoidable recompute;
  lean on the caches.

---

## 7. Useful paths & commands

- Spec: `docs/SPEC.md` · Brief: `docs/SUPERVISOR_BRIEF.md`
- Config: `config/default.yaml`, `config/regions_dhaka.yaml`, `config/experiments.yaml`
- Drive root (Colab): `/gdrive/MyDrive/BanglaSlumNet/` (data, results, model_cache)
- Local syntax check (Windows PowerShell):
  `python -m py_compile (Get-ChildItem -Recurse -Filter *.py -Path src,scripts,gee | % { $_.FullName })`
- Regenerate figures: `python scripts/make_paper_figures.py --results_dir <drive>/results`
- GEE project id: `banglaslumnet` (user's own; EE API enabled + project registered).
