# BanglaSlumNet Data Card

## Overview

**Name:** BanglaSlumNet Dhaka open-data benchmark
**Task:** classify 100 m grid cells as slum / non-slum
**Geography:** Dhaka, Bangladesh. The area ESA mapped: Dhaka North + South City Corporations
(expanded boundaries), 305.8 km².
**Years:** 2017 (reference year) and 2023
**Built by:** the pipeline in `scripts/` and `gee/`, configured by `config/benchmark.yaml`.
Raw and derived data are not stored in git.

## Sources

| Role | Source | Resolution / date | Licence |
|---|---|---|---|
| Reference labels | ESA EO4SD-Urban, *Dhaka informal settlements 2017* (World Bank data catalog): 1,624 polygons, 23.94 km², interpreted from very-high-resolution imagery | vector, 2017 | CC BY 4.0 |
| Mapped-area boundary | geoBoundaries BGD ADM4 (simplified), from Bangladesh Bureau of Statistics / OCHA | vector, 2020 | CC BY 3.0 IGO |
| Imagery | Copernicus Sentinel-2 L2A (`COPERNICUS/S2_SR_HARMONIZED`), dry-season median (Dec–Feb), SCL cloud mask | 10 m | Copernicus open licence |
| Building structure | Google Open Buildings 2.5D Temporal (`GOOGLE/Research/open-buildings-temporal/v1`): presence, height, fractional count | ~4 m effective, annual | see the GEE catalog terms |
| Land cover | Google Dynamic World V1, same window | 10 m | see the GEE catalog terms |
| Night lights | VIIRS DNB monthly (`NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`), median of the year | ~463 m | see the GEE catalog terms |
| Population | WorldPop R2025A (`WorldPop/GP/100m/R2025A/v1/pop`) | 100 m, annual | CC BY 4.0 |

Esri World Imagery is used only for the GRAM baseline, never as benchmark input.

## Construction

Run in order, with the Python environment in `~/.venvs/ee`:

```
python scripts/build_mapped_area.py           # ESA mapped area from ADM4 units
python scripts/build_benchmark_grid.py        # 100 m grid + ESA fractions
python gee/export_cell_features.py --year 2017
python gee/export_cell_features.py --year 2023
python scripts/build_cell_table.py            # labels, folds -> data/derived/cells_<year>.csv
```

**Mapped area.** ESA does not publish its ward boundaries. Each ESA polygon is located in the
geoBoundaries ADM4 units, and a unit is kept only if it is no larger than 1.25× the ESA area of
the wards it holds. The result is 303.0 km² against ESA's 305.8 km², and contains 99.4% of the
ESA polygons.

**Grid.** 100 m cells in UTM 46N (EPSG:32646), 279 × 212 cells, padded 1 km beyond the mapped
area. ESA polygons are burned in on the 10 m sub-grid.

**Labels per cell** (`src/data/grid.py`):

| Label | Rule |
|---|---|
| slum (1) | ESA slum share ≥ 0.5 |
| non-slum (0) | ESA slum share = 0, and no ESA polygon within 20 m |
| ignore (−1) | everything else: mixed cells, cells outside the mapped area shrunk by 100 m, not built-up |

Both classes must be built-up: Open Buildings presence ≥ 0.15 for that year. Dynamic World was
rejected for this mask because it drops 25% of ESA slum cells; Open Buildings keeps 98%.

**2023 labels** reuse the 2017 reference with two masks based on Open Buildings change since 2017:
- slum cells are ignored if they became ≥ 3 m taller, or lost ≥ 50% of their building presence;
- non-slum cells are ignored if their presence rose by ≥ 0.2 (possible new slums).

**Spatial folds.** Cells are grouped into 2 km blocks, and each block goes entirely to one of 5
folds. Blocks are dealt so every fold gets about the same number of slum cells. Folds are drawn
once, from the 2017 labels, and reused for 2023.

## Counts

| Year | Slum cells | Non-slum cells | Slum cells per fold |
|---|---:|---:|---:|
| 2017 | 1,730 | 6,733 | 346 |
| 2023 | 1,657 | 7,449 | 327–337 |

## Hard-negative strata (non-slum cells only)

| Stratum | Definition | 2017 cells |
|---|---|---:|
| dense | Open Buildings presence ≥ 0.339 (the 25th percentile of slum cells) | 3,496 |
| look-alike | dense, and mean building height ≤ 4.91 m (the 75th percentile of slum cells) | 484 |
| formal districts | inside the Old Dhaka, Gulshan/Baridhara, Dhanmondi or Uttara boxes of `config/regions_dhaka.yaml` | 1,248 |

## Known limitations

- **Slum area captured.** Slum cells capture only 57% of ESA's slum area. Small pockets and mixed
  cells fall below the 0.5 share and are ignored. Recall on polygons under 1 ha is reported separately.
- **Reference age.** The reference is ESA's 2017 interpretation, not field truth. For 2023,
  Open Buildings suggests ~2–6% of slum area changed; changed cells are masked, but slums
  that appeared after 2017 can only be masked, not labelled.
- **Strata reuse model features.** The dense and look-alike strata are defined from the same
  Open Buildings bands the model uses. They describe where open data stops separating the
  classes; they are not independent tests.
- **Mapped-area boundary** is reconstructed from administrative units, not ESA's own boundary.
- **Open Buildings Temporal** is itself a model output estimated from Sentinel-2, and ends in 2023.
