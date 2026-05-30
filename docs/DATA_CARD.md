# BanglaSlumNet Data Card

## Dataset Overview

**Name:** BanglaSlumNet Dhaka Benchmark  
**Task:** Binary informal-settlement segmentation  
**Geography:** Dhaka, Bangladesh (5 regions)  
**License:** Research-only (see THIRD_PARTY_LICENSES.md)

## Data Sources

| Source | Layer | Resolution | License |
|--------|-------|-----------|---------|
| Copernicus Sentinel-2 L2A | RGB + NIR imagery | 10 m | Open |
| ESRI World Imagery | High-res tiles (GRAM baseline only) | ~1.2 m | ESRI terms |
| OpenStreetMap | Residential tags | Vector | ODbL |
| GHSL Built-S2 2020 | Built-up fraction | 10 m | Open |
| VIIRS DNB Monthly | Nighttime lights | ~500 m (resampled) | Open |
| WorldPop GP/100m | Population density | 100 m | CC BY 4.0 |
| GHS-POP 2020 | Population count | 100 m | Open |
| WorldBank GriddedPoverty | Poverty index | TODO_VERIFY | TODO_VERIFY |

## Regions

| Region | Type | Tiles (approx.) | HC pixels (approx.) |
|--------|------|-----------------|---------------------|
| Korail | Informal | TODO | TODO |
| Bhashantek | Informal | TODO | TODO |
| Karail extension | Informal | TODO | TODO |
| Old Dhaka brick core | Formal-dense | TODO | TODO |
| Gulshan-2 / Baridhara | Formal-dense | TODO | TODO |

_Fill in after dataset_manifest.json is built._

## Label Generation

**3-signal geospatial fusion (OSM ∩ GHSL ∩ VIIRS):**
- Slum pixel: OSM residential ∩ GHSL built-up ∩ VIIRS dark (≤ city median)
- Formal-dense pixel: OSM residential ∩ GHSL built-up ∩ VIIRS bright (> city median)
- Unknown: anything else

**4-signal HC promotion (Direction B):**
- Geospatial 3-way agreement AND LocateAnything-3B visual grounding agrees in sign
- HC pixels form the primary evaluation subset

## Tile Specification

- Size: 256 × 256 pixels at 10 m/px (≈ 2.56 km × 2.56 km)
- Train stride: 128 px (50% overlap)
- Eval stride: 256 px (no overlap)
- Training years: 2020–2023, dry and wet seasons

## Known Limitations

- Weak labels are programmatically generated and contain noise
- OSM coverage in informal areas may be incomplete
- WorldBank poverty layer requires verification (TODO_VERIFY)
- Bounding boxes are approximate and require visual confirmation in GEE
