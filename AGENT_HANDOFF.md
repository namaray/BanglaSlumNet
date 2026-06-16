# Agent Hand-off — BanglaSlumNet

> Read this to continue the project in a fresh agent session (Claude Code, Codex, etc.).
> Pair with `BanglaSlumNetV4.md` (paper/method reference), `docs/NEXT_SESSION.md`, and
> `docs/SPEC.md` (build contract + v4 amendments). A bootstrap prompt for a new chat is at
> the very bottom (§9).

---

## 1. How we work (the setup)

- **GitHub:** https://github.com/namaray/BanglaSlumNet — work on **`main`**, push every change
  directly to `main` (user's instruction). Old pre-project history is preserved in branch
  `legacy`. A stale `feature/locate-anything-integration` branch also exists; ignore it.
  End commits with the Claude co-author trailer.
- **Compute:** Google **Colab** (Pro+), GPU = A100 or **L4** (L4 is fine; 23.7 GB ≥ enough,
  no 4-bit needed). MagiAttention unavailable on Ampere → SDPA fallback (fine).
- **Code vs data split (important):**
  - **Code** is cloned **fresh each Colab session to LOCAL disk: `/content/BanglaSlumNet`.**
    Do NOT clone onto Drive — git-on-Drive fails (dubious-ownership / detached-HEAD).
  - **Data, results, model cache** live on **Drive: `/gdrive/MyDrive/BanglaSlumNet/`**
    (`data/`, `results/`, `model_cache/`). These persist across sessions; the user pulls them
    locally for the paper.
- **Local dev machine:** Windows + PowerShell, repo at `D:\papers\BanglaSlumPaper\BanglaSlumNet`.
  Only `pyyaml` installed (NO torch/numpy/rasterio) → locally you can only
  `python -m py_compile` to syntax-check and validate YAML/notebook JSON. **All real
  execution is in Colab.**
- **GEE:** project id **`banglaslumnet`** (the user's own; EE API enabled + project registered
  for noncommercial use). GEE exports run in-notebook via the Python `ee` API (`gee/export_*.py`).
- **HuggingFace:** `nvidia/LocateAnything-3B` is public (no gate / token needed).

### The iteration loop
1. Edit code/notebook locally → `py_compile` → commit + push to `main`.
2. In Colab: **Runtime → Restart session → reopen the notebook from the GitHub tab → run.**
   (Uploading the notebook gives a stale copy — ALWAYS open from the GitHub tab.)
3. Heavy phases cache to Drive and skip on re-run; the experiment registry makes Phase 4
   resumable. After a `.py` change a restart picks it up; after a notebook *cell* change the
   user must reopen from GitHub.
4. **Always run P0.1–P0.5 first** in any fresh/restarted session (sets cwd + `sys.path` +
   `cfg`); skipping P0.2 is the cause of recurring `ModuleNotFoundError: No module named 'src'`.

## 2. Repo structure

```
BanglaSlumNet/
├── BanglaSlumNetV4.md          # paper/method reference (read this)
├── AGENT_HANDOFF.md            # this file
├── config/                     # default.yaml, regions_dhaka.yaml (12 regions), experiments.yaml
├── gee/                        # Python `ee` exporters: export_s2_composites / export_weak_labels
│                               #   / export_socioeconomic / ee_export_utils (+ legacy .js)
├── src/
│   ├── data/                   # tiles.py (dataset+manifest+split), tiling.py, weak_labels.py,
│   │                           #   socioeconomic.py, augment.py, preflight.py
│   ├── locate_anything/        # worker.py (real LA API), feature_extractor.py, label_validator.py,
│   │                           #   prompts.py, _compat.py (decord stub)
│   ├── models/                 # sasnet.py, fusion.py, decoder.py, baseline_cnn.py, banglaslumnet.py
│   ├── train/                  # train_sasnet.py, train_segmenter.py, losses.py
│   ├── eval/                   # metrics.py, evaluate.py, gram_baseline.py
│   ├── tracking/               # recorder.py, registry.py
│   └── viz/                    # plots.py, palette.py, qualitative.py
├── notebooks/                  # BanglaSlumNet_Colab.ipynb (master), 00_smoke_test.ipynb
├── scripts/                    # download_models.py, make_paper_figures.py
└── docs/                       # SPEC.md, SUPERVISOR_BRIEF.md, NEXT_SESSION.md, DATA_CARD.md, RESULTS.md
```
On Drive: `/gdrive/MyDrive/BanglaSlumNet/{data/{tiles,labels,socioeconomic,features_cache},results/{runs,figures,tables},model_cache}`.

## 3. Notebook phases (BanglaSlumNet_Colab.ipynb)

- **P0.1–P0.5** Setup: mount Drive, clone code to /content, pip install, GPU check, load `cfg`.
- **P1.0** (maintenance) label reset — `RESET_LABELS=True` once to wipe stale labels/results.
- **P1.1/P1.2** download + load LocateAnything (SKIP these if you only need re-labeling /
  Phase 4 over cached features — saves memory; P1.5 OOM'd with the model loaded).
- **P1.3** region preview (all 12 boxes on a folium map).
- **P1.4** GEE exports (S2 / weak labels / socioeconomic) via Python `ee`. `GEE_PROJECT='banglaslumnet'`.
- **P1.5** tiling → per-tile stacks + manifest (has a labels-only fast path for re-label runs).
- **P2.1–P2.3** weak-label ingest, LA validation (currently disabled), confidence figure.
- **P3.1** MoonViT feature extraction (cached). **P3.2** SAS-Net train + cache clean tiles.
- **P4.1** registry. **P4.1b** VERIFY-LABELS gate (asserts slum>0 and HC>0 — run before Phase 4!).
- **P4.2** experiment matrix (16 runs over cached features). **P4.3** optional GRAM head-to-head.
- **P5** figures + tables + headline results.

## 4. Current state (2026-06-16)

- ✅ 720 balanced tiles @128 px (12 regions × 60); stratified split.
- ✅ MoonViT grounding-map features cached (`data/features_cache/`, ~89% non-zero).
- ✅ SAS-Net trained + clean tiles cached. Model downloaded to Drive `model_cache/`.
- ✅ Full pipeline runs end-to-end; figures/CSV/registry all generate.
- ⚠️ **Labels:** first two Phase-4 runs gave ALL-ZERO metrics → root cause = degenerate labels
  (**0 slum pixels**, all formal) from the VIIRS dark/bright rule. **LAST FIX (this session):
  switched `gee/export_weak_labels.py` to REGION-TYPE labeling** (informal regions→slum,
  formal→formal; HC = built pixels). Not yet verified on real data.

## 5. Obstacles faced & fixed (so you don't relive them)

- `ModuleNotFoundError: src` → must run P0.2 (chdir /content + sys.path) every session.
- git-on-Drive failures (dubious ownership / detached HEAD) → clone code to `/content`, not Drive.
- Stale uploaded notebook → always open from the GitHub tab.
- py3.12 wheels: dropped hard pins (numpy/Pillow/decord) in `requirements_colab.txt`.
- `decord` required by LA remote code, no 3.12 wheel → stub in `src/locate_anything/_compat.py`
  (with valid `__spec__`).
- Real LA API differs from a generic VLM → ported chat-template/`generate`/box-regex into `worker.py`.
- MoonViT returns a list → hook unwraps it.
- BCE unsafe under autocast → run loss in float32.
- IoU returned NaN on empty union → return 0; always save a checkpoint; guard empty eval loader.
- Split dumped all HC tiles into eval (0 train) → stratified-by-region split.
- 256-px tiles gave only 5 tiles → 128 px + tile all years + 12 regions → 720 tiles.
- GEE S2 exports cropped to valid-data bbox → `unmask` for full-extent (wet-season composites
  are empty → `[FAILED]`, expected/harmless).
- P1.5 OOM/crash → labels-only fast path in `tiling.py` + per-region `gc`; and skip P1.1/P1.2
  (don't hold the 4 B model in memory during tiling).
- **0-slum labels** → region-type labeling (the last fix).

## 6. Learnings

- Verify the *data* (labels/HC/features non-zero) on CPU BEFORE spending GPU — the P4.1b gate
  exists for this. All-zero metrics almost always = empty labels/HC, not a model bug.
- Keep the VLM out of memory for non-feature phases.
- The user is non-expert in the tooling and CU-sensitive: give clear step-by-step instructions,
  make the engineering call with a stated recommendation, lean on caches, push fixes to `main`.

## 7. The last fix & how to confirm it

`gee/export_weak_labels.py` now does region-type labeling. To apply it:
1. Restart Colab, reopen notebook from GitHub, run **P0.1–P0.5**.
2. **P1.0**: `RESET_LABELS=True`, run, set back to `False` (wipes old all-formal labels).
3. **Skip P1.1/P1.2.** Run **P1.4** (regenerates weak labels — prints `region: type=… -> class N`),
   then **P1.5** (re-tile, fast path).
4. **P2.1 → P2.2 (skips, LA off) → P2.3.**
5. **P4.1b** must now print **slum (1) > 0** and **HC > 0** → `OK … Safe to run Phase 4.`

## 8. What to do next

1. Confirm P4.1b is green (slum>0, HC>0). If slum is still 0, inspect `weeklabels_<region>.tif`
   band 1 values and the GHSL/DW built mask (built mask may be empty → no labeled pixels).
2. Run **P3.1/P3.2** (skip, cached) → **P4.2** (16 runs). Watch P4.2: training loss should now
   **decrease** and **HC-IoU > 0** (unlike before).
3. Run **P5** → figures + `all_runs.csv` + `docs/RESULTS.md`. User pulls Drive `results/` locally.
4. Read the numbers; write the manuscript from `BanglaSlumNetV4.md` §5/§7 (IEEE Access template
   to be supplied by the user). Report real numbers only.
5. Improvements if results are weak: per-pixel/manual HC test subset; `hidden_state` features
   for visual configs; real `osm_roads`/`wb_poverty` GEE assets; high-res ESRI grounding for LA.

## 9. Bootstrap prompt for a fresh Codex/Claude chat

```
You are continuing the BanglaSlumNet project (informal-settlement detection in Dhaka with
NVIDIA LocateAnything-3B + socioeconomic fusion). Repo: https://github.com/namaray/BanglaSlumNet
(work on main, push to main). Read these in the repo first: AGENT_HANDOFF.md, BanglaSlumNetV4.md,
docs/NEXT_SESSION.md, docs/SPEC.md.

Setup: code runs in Google Colab cloned to /content/BanglaSlumNet; data/results/model_cache live
on Google Drive /gdrive/MyDrive/BanglaSlumNet; GEE project id is "banglaslumnet". Local machine
is Windows/PowerShell with only pyyaml (use `python -m py_compile` to check; real runs are Colab).
The user is non-expert in the tooling and CU-sensitive — give clear step-by-step Colab
instructions, make engineering decisions with a recommendation, and push fixes to main for them
to pull (they restart Colab and reopen the notebook from the GitHub tab).

Current state: 720 tiles, MoonViT grounding-map features + SAS-Net cached. The last fix switched
weak labels to REGION-TYPE supervision (gee/export_weak_labels.py) because the VIIRS rule produced
0 slum pixels. NEXT: have the user reset labels (notebook P1.0 RESET_LABELS=True once), re-run
P1.4 → P1.5 → P2, then the P4.1b verify gate (must show slum>0 and HC>0). If green, run Phase 4
(P4.2) and Phase 5 (P5) to get real numbers, then help write the paper. If P4.1b still shows 0
slum, debug the weak-label export (check GHSL/DW built mask and weeklabels band-1 values).
Start by asking the user to paste the P4.1b output.
```
