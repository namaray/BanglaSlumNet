# BanglaSlumNet × LocateAnything

**RESEARCH-ONLY — See license notice below.**

BanglaSlumNet is a two-stage framework for informal-settlement detection in dense megacities, targeting the failure mode where optical-only foundation models (e.g., GRAM) cannot distinguish Dhaka's dense formal core from real slums.

## Architecture

- **Stage 1 – SAS-Net:** Scene-Appearance Separation to disentangle atmospheric haze and seasonal variation from structural content.
- **Stage 2 – VLM Backbone + Socioeconomic Fusion:** LocateAnything-3B's frozen MoonViT encoder, language-conditioned with discriminative slum/formal prompts, fused via cross-attention with socioeconomic priors (nighttime lights, population, roads, poverty).

## Quickstart

1. Open `notebooks/00_smoke_test.ipynb` in Colab and run all cells (≤ 5 min, 4 tiles).
2. Verify bounding boxes in `config/regions_dhaka.yaml` (all `TODO_VERIFY` markers).
3. Run GEE scripts in `gee/` from the GEE Code Editor; export tiles to Google Drive.
4. Open `notebooks/BanglaSlumNet_Colab.ipynb` and run Phase 0–5 sequentially.

## Repository Structure

```
BanglaSlumNet/
├── config/          # All hyperparameters, region boxes, experiment matrix
├── gee/             # Google Earth Engine export scripts
├── src/             # All Python source
│   ├── data/        # Tile dataset, weak labels, socioeconomic loader
│   ├── locate_anything/  # LocateAnything integration (worker, extractor, validator)
│   ├── models/      # SAS-Net, fusion, decoder, baseline, assembly
│   ├── train/       # Training loops and losses
│   ├── eval/        # Metrics and evaluation
│   ├── tracking/    # Results recorder and run registry
│   └── viz/         # Paper figures
├── notebooks/       # Colab notebooks
├── scripts/         # Utility scripts
├── data/            # (gitignored) tiles, labels, features
├── results/         # (gitignored except .gitkeep) run outputs
└── docs/            # Spec, data card, results summary
```

## License Notice

**This project is for non-commercial academic research only.**

This codebase integrates **LocateAnything-3B** (NVIDIA License — non-commercial research use only). Any weights derived from LocateAnything, and any maps produced with it, must be tagged **research-only** and must retain NVIDIA's license and attribution notices.

See `THIRD_PARTY_LICENSES.md` for full license text.

Components:
- **LocateAnything-3B** — NVIDIA Non-Commercial Research License
- **MoonViT-SO-400M** — MIT License
- **Qwen2.5-3B-Instruct** — Qwen Research License

## Citation

```bibtex
@article{banglaslumnet2026,
  title={BanglaSlumNet: Language-Grounded Socioeconomic Fusion for Dense-Megacity Slum Detection},
  year={2026}
}
```
