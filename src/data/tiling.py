"""
Tiling: split regional GeoTIFFs (exported once by GEE to Drive) into aligned
per-tile 256x256 stacks that SlumTileDataset reads.

Why reproject instead of plain crop: GEE exports S2, labels, and socioeconomic
rasters separately. Even with the same scale/CRS they are not guaranteed to be
pixel-identical. We define the grid from the S2 window and reproject the label
and socioeconomic sources ONTO that exact window grid, so a misaligned tile is
impossible by construction (this is the single worst silent bug in this pipeline).

Outputs per tile (all on the identical 256x256 grid):
    {tile_id}_rgb.tif      float32, C_bands
    {tile_id}_noisy.tif    uint8 (0=unknown,1=slum,2=formal)
    {tile_id}_hc.tif       uint8 (1=high-confidence)
    {tile_id}_socioec.tif  float32, len(channels) bands with descriptions

Everything is cached: a tile that already exists is skipped (0 recompute).
"""

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except ImportError:
    raise ImportError("rasterio required: pip install rasterio")

from .weak_labels import apply_la_validation, LABEL_SLUM, LABEL_FORMAL


def _windows(width: int, height: int, tile: int, stride: int):
    """Yield (row_idx, col_idx, Window) covering the raster with given stride."""
    for r, row_off in enumerate(range(0, max(height - tile + 1, 1), stride)):
        for c, col_off in enumerate(range(0, max(width - tile + 1, 1), stride)):
            yield r, c, Window(col_off, row_off, tile, tile)


def _reproject_onto(src_path: str, bands: List[int], dst_transform, dst_crs,
                    tile: int, resampling: Resampling) -> np.ndarray:
    """Reproject selected bands of a source raster onto the destination window grid."""
    out = np.zeros((len(bands), tile, tile), dtype=np.float32)
    with rasterio.open(src_path) as src:
        for i, b in enumerate(bands):
            reproject(
                source=rasterio.band(src, b),
                destination=out[i],
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
            )
    return out


def tile_region(
    region: str,
    s2_path: str,
    labels_path: Optional[str],
    socioec_path: Optional[str],
    out_tiles_dir: str,
    out_labels_dir: str,
    out_socioec_dir: str,
    s2_band_count: int,
    socioeconomic_channels: List[str],
    tile: int = 256,
    stride: int = 128,
    la_validation: Optional[Dict] = None,
    use_la: bool = False,
    viirs_percentile_threshold: float = 50.0,
    composite_tag: Optional[str] = None,
) -> List[Dict]:
    """
    Tile one region. Returns list of per-tile metadata dicts.
    Missing label/socioec sources are tolerated (tiles still written, zeros filled)
    but reported so preflight can flag them before any GPU spend.
    """
    out_tiles_dir = Path(out_tiles_dir); out_tiles_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir = Path(out_labels_dir); out_labels_dir.mkdir(parents=True, exist_ok=True)
    out_socioec_dir = Path(out_socioec_dir); out_socioec_dir.mkdir(parents=True, exist_ok=True)

    tag = composite_tag or region
    records = []
    with rasterio.open(s2_path) as s2:
        s2_crs = s2.crs
        width, height = s2.width, s2.height
        n_s2_bands = min(s2_band_count, s2.count)

        for r, c, win in _windows(width, height, tile, stride):
            tile_id = f"{tag}_{r:03d}_{c:03d}"
            rgb_out = out_tiles_dir / f"{tile_id}_rgb.tif"
            noisy_out = out_labels_dir / f"{tile_id}_noisy.tif"
            hcgeo_out = out_labels_dir / f"{tile_id}_hcgeo.tif"  # 3-signal geo HC (preserved)
            hc_out = out_labels_dir / f"{tile_id}_hc.tif"        # final HC (geo or 4-signal)
            socioec_out = out_socioec_dir / f"{tile_id}_socioec.tif"

            # Cache check — skip fully written tiles
            if all(p.exists() for p in [rgb_out, noisy_out, hcgeo_out, hc_out, socioec_out]):
                hc_count = _quick_hc_count(hc_out)
                records.append({"tile_id": tile_id, "region": region,
                                "hc_pixel_count": hc_count, "split": None})
                continue

            # Fast path: rgb + socioec already exist but the labels were reset.
            # Regenerate ONLY the 3 label files from the existing tile's grid — this
            # avoids re-reading the big S2 composite and rewriting rgb/socioec, which
            # is what made full re-tiling exhaust memory / Drive I/O.
            if rgb_out.exists() and socioec_out.exists() and labels_path and Path(labels_path).exists():
                with rasterio.open(str(rgb_out)) as rt:
                    wt, wcrs, th, tw = rt.transform, rt.crs, rt.height, rt.width
                lbl = _reproject_onto(labels_path, [1, 2, 3], wt, wcrs, th, Resampling.nearest)
                noisy = lbl[0].astype(np.uint8)
                hc_geo = lbl[1].astype(np.uint8)
                hc_mask = apply_la_validation(noisy, hc_geo, tile_id, la_validation, use_la)
                prof_u = {"driver": "GTiff", "dtype": "uint8", "count": 1,
                          "height": th, "width": tw, "crs": wcrs, "transform": wt}
                with rasterio.open(str(noisy_out), "w", **prof_u) as d:
                    d.write(noisy[np.newaxis])
                with rasterio.open(str(hcgeo_out), "w", **prof_u) as d:
                    d.write(hc_geo[np.newaxis])
                with rasterio.open(str(hc_out), "w", **prof_u) as d:
                    d.write(hc_mask[np.newaxis])
                records.append({"tile_id": tile_id, "region": region,
                                "hc_pixel_count": int(hc_mask.sum()), "split": None})
                del lbl, noisy, hc_geo, hc_mask
                continue

            win_transform = s2.window_transform(win)
            rgb = s2.read(list(range(1, n_s2_bands + 1)), window=win,
                          boundless=True, fill_value=0).astype(np.float32)
            if rgb.shape[-2:] != (tile, tile):
                # pad to tile size (edge window)
                rgb = _pad_to(rgb, tile)

            profile_f = {"driver": "GTiff", "dtype": "float32", "count": rgb.shape[0],
                         "height": tile, "width": tile, "crs": s2_crs, "transform": win_transform}
            with rasterio.open(str(rgb_out), "w", **profile_f) as dst:
                dst.write(np.clip(rgb, 0, 1))

            # ── Labels: reproject 3-band weeklabels onto this grid ──────────────
            if labels_path and Path(labels_path).exists():
                lbl = _reproject_onto(labels_path, [1, 2, 3], win_transform, s2_crs,
                                      tile, Resampling.nearest)
                noisy = lbl[0].astype(np.uint8)
                hc_geo = lbl[1].astype(np.uint8)
            else:
                noisy = np.zeros((tile, tile), dtype=np.uint8)
                hc_geo = np.zeros((tile, tile), dtype=np.uint8)

            hc_mask = apply_la_validation(noisy, hc_geo, tile_id, la_validation, use_la)

            profile_u = {"driver": "GTiff", "dtype": "uint8", "count": 1,
                         "height": tile, "width": tile, "crs": s2_crs, "transform": win_transform}
            with rasterio.open(str(noisy_out), "w", **profile_u) as dst:
                dst.write(noisy[np.newaxis])
            with rasterio.open(str(hcgeo_out), "w", **profile_u) as dst:
                dst.write(hc_geo[np.newaxis])
            with rasterio.open(str(hc_out), "w", **profile_u) as dst:
                dst.write(hc_mask[np.newaxis])

            # ── Socioeconomic: reproject channels onto this grid ────────────────
            if socioec_path and Path(socioec_path).exists():
                with rasterio.open(socioec_path) as se:
                    desc = list(se.descriptions) if se.descriptions else []
                    band_map = {d.lower(): i + 1 for i, d in enumerate(desc) if d}
                    n_src_bands = se.count
                if band_map:
                    bands = [band_map.get(ch.lower(), 0) for ch in socioeconomic_channels]
                elif n_src_bands >= len(socioeconomic_channels):
                    # No band descriptions in the GeoTIFF: fall back to config order
                    # (the ee export writes bands in socioeconomic_channels order).
                    bands = [i + 1 for i in range(len(socioeconomic_channels))]
                else:
                    bands = [0] * len(socioeconomic_channels)
                eco = np.zeros((len(socioeconomic_channels), tile, tile), dtype=np.float32)
                present = [b for b in bands if b > 0]
                if present:
                    with rasterio.open(socioec_path) as se:
                        for i, b in enumerate(bands):
                            if b > 0:
                                reproject(source=rasterio.band(se, b), destination=eco[i],
                                          dst_transform=win_transform, dst_crs=s2_crs,
                                          resampling=Resampling.bilinear)
            else:
                eco = np.zeros((len(socioeconomic_channels), tile, tile), dtype=np.float32)

            profile_e = {"driver": "GTiff", "dtype": "float32", "count": eco.shape[0],
                         "height": tile, "width": tile, "crs": s2_crs, "transform": win_transform}
            with rasterio.open(str(socioec_out), "w", **profile_e) as dst:
                dst.write(eco)
                for i, ch in enumerate(socioeconomic_channels):
                    dst.set_band_description(i + 1, ch)

            records.append({"tile_id": tile_id, "region": region,
                            "hc_pixel_count": int(hc_mask.sum()), "split": None})

    return records


def refresh_hc_with_la(
    config: dict,
    la_validation: Dict,
    manifest_path: Optional[str] = None,
) -> int:
    """
    After LocateAnything validation (Phase 2), promote each tile's HC mask to the
    4-signal version WITHOUT re-tiling. Reads the preserved 3-signal geo-HC
    ({tile_id}_hcgeo.tif) + noisy label, applies the VLM sign-agreement, rewrites
    {tile_id}_hc.tif. Idempotent (always recomputed from hcgeo, never doubled).
    Returns number of tiles updated.
    """
    paths = config["paths"]
    labels_dir = Path(paths["labels_dir"])
    manifest_path = manifest_path or paths["manifest"]
    with open(manifest_path) as f:
        tiles = json.load(f)["tiles"]

    updated = 0
    new_counts = {}
    for t in tiles:
        tid = t["tile_id"]
        noisy_p = labels_dir / f"{tid}_noisy.tif"
        hcgeo_p = labels_dir / f"{tid}_hcgeo.tif"
        hc_p = labels_dir / f"{tid}_hc.tif"
        if not (noisy_p.exists() and hcgeo_p.exists()):
            continue
        with rasterio.open(str(noisy_p)) as s:
            noisy = s.read(1).astype(np.uint8); crs, tr = s.crs, s.transform
        with rasterio.open(str(hcgeo_p)) as s:
            hc_geo = s.read(1).astype(np.uint8)
        hc_mask = apply_la_validation(noisy, hc_geo, tid, la_validation, use_la=True)
        prof = {"driver": "GTiff", "dtype": "uint8", "count": 1,
                "height": noisy.shape[0], "width": noisy.shape[1], "crs": crs, "transform": tr}
        with rasterio.open(str(hc_p), "w", **prof) as d:
            d.write(hc_mask[np.newaxis])
        new_counts[tid] = int(hc_mask.sum())
        updated += 1

    # Refresh hc_pixel_count in the manifest so splits/preflight stay accurate
    for t in tiles:
        if t["tile_id"] in new_counts:
            t["hc_pixel_count"] = new_counts[t["tile_id"]]
    with open(manifest_path, "w") as f:
        json.dump({"version": "1.0", "n_tiles": len(tiles), "tiles": tiles}, f, indent=2)

    print(f"refresh_hc_with_la: updated {updated} tiles with 4-signal HC.")
    return updated


def _pad_to(arr: np.ndarray, tile: int) -> np.ndarray:
    c, h, w = arr.shape
    out = np.zeros((c, tile, tile), dtype=arr.dtype)
    out[:, :h, :w] = arr[:, :tile, :tile]
    return out


def _quick_hc_count(hc_path: Path) -> int:
    with rasterio.open(str(hc_path)) as src:
        return int(src.read(1).sum())


def tile_all_regions(
    config: dict,
    regions_yaml: str,
    reference_year: Optional[int] = None,
    reference_season: Optional[str] = None,
    la_validation: Optional[Dict] = None,
) -> str:
    """
    Tile every region using the reference composite, write/refresh the manifest.
    Returns the manifest path. Idempotent: already-tiled tiles are skipped.
    """
    import yaml
    from .tiles import build_manifest_from_records

    paths = config["paths"]
    data_cfg = config["data"]
    s2_bands = data_cfg.get("s2_bands", ["B2", "B3", "B4", "B8"])
    eco_channels = config["fusion"]["socioeconomic_channels"]
    year = reference_year or data_cfg.get("training_years", [2022])[0]
    season = reference_season or data_cfg.get("seasons", ["dry"])[0]

    with open(regions_yaml) as f:
        regions = yaml.safe_load(f)["regions"]

    import glob as _glob
    all_records = []
    for region in regions:
        labels_path = str(Path(paths["labels_dir"]) / f"weeklabels_{region}.tif")
        socioec_path = str(Path(paths["socioeconomic_dir"]) / f"socioeconomic_{region}.tif")
        # Tile EVERY available S2 composite for this region (all years/seasons that
        # downloaded), multiplying the tile count and adding temporal variety.
        comps = sorted(_glob.glob(str(Path(paths["tiles_dir"]) / f"s2_{region}_*.tif")))
        comps = [c for c in comps if Path(c).name.startswith(f"s2_{region}_")]
        if not comps:
            print(f"  [skip] no S2 composites for {region}")
            continue
        region_total = 0
        for s2_path in comps:
            tag = Path(s2_path).stem[len("s2_"):]   # e.g. "korail_2020_dry"
            recs = tile_region(
                region=region, composite_tag=tag, s2_path=s2_path,
                labels_path=labels_path if Path(labels_path).exists() else None,
                socioec_path=socioec_path if Path(socioec_path).exists() else None,
                out_tiles_dir=paths["tiles_dir"], out_labels_dir=paths["labels_dir"],
                out_socioec_dir=paths["socioeconomic_dir"],
                s2_band_count=len(s2_bands), socioeconomic_channels=eco_channels,
                tile=data_cfg.get("tile_size", 256), stride=data_cfg.get("train_stride", 128),
                la_validation=la_validation,
                use_la=config.get("weak_labels", {}).get("use_locate_anything_validation", False) and la_validation is not None,
                viirs_percentile_threshold=config.get("weak_labels", {}).get("viirs_percentile_threshold", 50.0),
            )
            all_records.extend(recs)
            region_total += len(recs)
        import gc
        gc.collect()
        print(f"  {region}: {region_total} tiles from {len(comps)} composites")

    manifest_path = build_manifest_from_records(
        all_records, output_path=paths["manifest"],
        val_fraction=data_cfg.get("val_fraction", 0.15), seed=config.get("seed", 1337),
    )
    return manifest_path


# ── Smoke test ─────────────────────────────────────────────────────────────────
def _smoke_test():
    import tempfile
    from rasterio.transform import from_bounds
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Synthetic regional S2 (4 bands, 400x400), labels (3 bands), socioec (2 bands)
        tr = from_bounds(0, 0, 1, 1, 400, 400)
        with rasterio.open(str(tmp / "s2_test_2022_dry.tif"), "w", driver="GTiff",
                           dtype="float32", count=4, width=400, height=400,
                           crs="EPSG:4326", transform=tr) as d:
            d.write(np.random.rand(4, 400, 400).astype(np.float32))
        with rasterio.open(str(tmp / "weeklabels_test.tif"), "w", driver="GTiff",
                           dtype="uint8", count=3, width=400, height=400,
                           crs="EPSG:4326", transform=tr) as d:
            arr = np.zeros((3, 400, 400), dtype=np.uint8)
            arr[0, :200] = 1; arr[1, :200] = 1; arr[2] = 3
            d.write(arr)
        with rasterio.open(str(tmp / "socioeconomic_test.tif"), "w", driver="GTiff",
                           dtype="float32", count=2, width=400, height=400,
                           crs="EPSG:4326", transform=tr) as d:
            d.write(np.random.rand(2, 400, 400).astype(np.float32))
            d.set_band_description(1, "viirs"); d.set_band_description(2, "worldpop")

        recs = tile_region(
            region="test", s2_path=str(tmp / "s2_test_2022_dry.tif"),
            labels_path=str(tmp / "weeklabels_test.tif"),
            socioec_path=str(tmp / "socioeconomic_test.tif"),
            out_tiles_dir=str(tmp), out_labels_dir=str(tmp), out_socioec_dir=str(tmp),
            s2_band_count=4, socioeconomic_channels=["viirs", "worldpop"],
            tile=256, stride=128,
        )
        assert len(recs) > 0
        assert (tmp / f"{recs[0]['tile_id']}_rgb.tif").exists()
        assert (tmp / f"{recs[0]['tile_id']}_socioec.tif").exists()
    print("tiling.py smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
