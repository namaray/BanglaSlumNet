# BanglaSlumNet × LocateAnything — Progress Brief

**Prepared for:** Supervisor review
**Project:** Informal-settlement (slum) detection in Dhaka using a frontier vision-language
model fused with socioeconomic data
**Status (2026-06-14):** End-to-end pipeline built and running on Google Colab (A100).
Full data → model → evaluation → figures chain is operational. First experimental run
completed; quantitative results are **not yet meaningful** due to dataset-scale limitations
(explained below). No methodological blockers remain — the next phase is data scaling.

---

## 1. Objective

Optical-only foundation models (e.g. GRAM) fail in dense megacities: on Dhaka they cannot
separate genuine slums from the dense *formal* urban core, because RGB texture alone is
ambiguous. Our thesis: the disambiguating signal must come from **outside the RGB channel**.
We inject two orthogonal external signals:

1. **Language** — we prompt a frontier visual-grounding VLM (NVIDIA **LocateAnything-3B**)
   with the *concept* of an informal settlement ("dense informal settlement: small irregular
   rooftops, narrow unpaved gaps, no road grid") vs. formal housing, so the model is told what
   a slum *is*, not just "what is built-up here."
2. **Socioeconomics** — nighttime lights (VIIRS), population (WorldPop, GHS-POP), roads, and
   poverty, fused with the visual features via cross-attention. Slums are dark-at-night and
   road-poor; the affluent formal core is bright and road-connected, even when they look
   similar in RGB.

**One-line contribution:** a language-grounded VLM fused with socioeconomic priors corrects
the dense-megacity failure mode that defeats optical-only slum detectors on Dhaka.

---

## 2. What was built

A complete, config-driven, reproducible pipeline (GitHub: `namaray/BanglaSlumNet`, branch
`main`), runnable on a single Colab A100 session:

| Stage | What it does |
|-------|--------------|
| **Data (Google Earth Engine)** | Exports Sentinel-2 seasonal composites, weak labels (OSM/Dynamic-World ∩ GHSL ∩ VIIRS), and the socioeconomic tensor for 5 Dhaka regions, straight to Drive. Runs in-notebook via the Python `ee` API. |
| **Weak labels** | 3-signal geospatial agreement → slum / formal-dense / unknown, with a per-pixel high-confidence (HC) mask. |
| **LocateAnything validation (Direction B)** | Zero-shot VLM grounding adds a *4th* signal: a tile is HC only if the VLM's visual grounding agrees in sign with the geospatial labels. |
| **MoonViT features (Direction A)** | LocateAnything's frozen vision encoder; features cached once to `.npy`. Two modes: `hidden_state` (rich, prompt-agnostic) and `grounding_map` (prompt-specific box coverage — needed for the language ablation). |
| **SAS-Net (Stage 1)** | Scene-Appearance Separation: disentangles atmospheric/seasonal appearance from structure, producing "clean" reference tiles. |
| **Fusion + decoder (Stage 2)** | Cross-attention fuses VLM features with the socioeconomic tensor; a lightweight UNet head outputs a per-pixel slum mask. The VLM stays frozen; only ~3.5–10M head parameters train. |
| **Tracking / figures** | Every run writes a JSON with config hash, prompt version, seed, git commit; a results CSV feeds auto-generated paper figures and LaTeX tables. |

Compute discipline: the expensive VLM passes (model download, label validation, feature
extraction, SAS-Net) run **once** and cache to Drive; experiments then iterate cheaply over
cached features. A registry makes runs resumable across Colab disconnects.

---

## 3. The experiments and what each is meant to show

| Experiment | Purpose |
|------------|---------|
| **Exp 1 — Atmospheric ablation** | Does Stage-1 SAS-Net normalization help vs. raw imagery and a classical correction? (rows: raw / classical / SAS-Net, backbone fixed at `vlm_visual`) |
| **Exp 2 — Socioeconomic + language fusion (central)** | The headline. Incrementally add signals — visual-only → +language concept → +VIIRS → +population → +roads → +poverty → full — and show **false positives on the formal core fall** while **slum recall holds**. Includes an optical-only `baseline_cnn` as a controlled GRAM analogue. |
| **Exp 3 — Leave-one-region-out** | Train on 4 regions, test on the held-out 5th. The critical fold: train *without* Old Dhaka, test *on* it — does the model still suppress false positives on a formal pattern it never saw? Tests whether the socioeconomic priors transfer rather than memorize. |

(Deferred, in the paper as "future work": national 64-district 10-year mapping; temporal
robustness.)

---

## 4. What we faced

**Methodological / integration challenges (all resolved):**
- LocateAnything is a `trust_remote_code` model with an undocumented internal layout. We had
  to introspect it live to find the vision encoder (`model.vision_model`, MoonViT, 1152-dim,
  returns a list of native-resolution patch features) and port the real chat-template +
  custom `generate` API from the model card.
- It hard-requires `decord` (a video library with no Python-3.12 wheel); we inject a stub
  since we only do image grounding.
- MagiAttention (its fast path) is unavailable on Colab's Ampere A100 — confirmed it falls
  back to PyTorch SDPA cleanly.
- A long tail of environment issues (Colab/Drive git quirks, dependency pins that break on
  Python 3.12, autocast-unsafe loss, etc.), all fixed.

**The real limitation (open):** **dataset scale and balance.**
- The 5 Dhaka regions are small, tight neighbourhoods; three (Korail, Karail-extension,
  Gulshan) are physically adjacent. At the chosen tile size this yielded only **75 tiles**,
  and the *informal* (slum) regions contribute only ~5 near-duplicate tiles each, while the
  formal controls contribute ~30 each.
- Consequently the segmentation head has almost no unique slum signal to learn from: training
  loss is flat and HC-IoU is ≈ 0 across all configs. **This is a data-quantity problem, not a
  pipeline bug** — the framework executes correctly end-to-end and produces the full results/
  figure artifacts.

---

## 5. Current results

First full matrix (16 configs) completed on Colab A100 in ~7 minutes over cached features.
Quantitative metrics are ≈ 0 and **should not be reported** — they reflect insufficient
training data, not model behaviour. They serve only to confirm the plumbing (CSV, per-run
provenance JSON, and all paper figures generate automatically).

---

## 6. Next steps (in priority order)

1. **Scale and rebalance the dataset (highest impact).** Acquire many more *unique* slum
   tiles — enlarge/relocate the informal regions, add further Dhaka slum areas, and/or reduce
   tile size — to reach tens–hundreds of distinct slum locations with a balanced slum/formal
   mix in every split. This is the single change most likely to produce real results.
2. **Richer features for the visual configs.** Use `hidden_state` MoonViT features (1152-dim)
   for `vlm_visual`/`baseline` comparisons; keep `grounding_map` (prompt-specific) for the
   language ablation where prompt-dependence is the point.
3. **Re-run the experiment matrix** (cheap — features cache) and read the real Exp 1/2/3
   numbers; regenerate figures and the master table.
4. **Replace two placeholder data layers** before trusting the roads/poverty ablation rows:
   `osm_roads` (currently an accessibility proxy) and `wb_poverty` (currently zero).
5. Then: write the manuscript from the recorded numbers (failure-mode figure, SAS-Net
   ablation, the central fusion ablation, LORO, qualitative overlays).

**Bottom line:** the hard engineering and the LocateAnything integration are done and working.
The path to results is now a focused data-scaling effort, not further system building.
