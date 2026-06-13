"""
Preflight dataset validation — RUN ON CPU BEFORE ANY GPU SPEND.

This is the single most important CU-protection in the pipeline. It catches every
"missing file / misaligned grid / uncached feature" error on a CPU runtime in
seconds, instead of crashing a paid A100 session two hours into Phase 4.

Call validate_dataset(config, regions_yaml, model_configs) and only proceed to
training if it returns ok=True.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import rasterio
except ImportError:
    raise ImportError("rasterio required")


def detect_feature_dim(features_cache_dir: str, model_config: str) -> int:
    """
    Infer the concatenated cached-feature channel count for a config by probing one
    cached .npy per prompt. Returns the value to set as config.locate_anything.feature_dim.
    """
    from .tiles import SlumTileDataset
    prompt_ids = SlumTileDataset._resolve_prompt_ids(model_config)
    if not prompt_ids:
        return 0
    cache = Path(features_cache_dir)
    total = 0
    for pid in prompt_ids:
        sample = next(iter(cache.glob(f"*_{pid}.npy")), None)
        if sample is None:
            raise FileNotFoundError(
                f"No cached features for prompt '{pid}' in {cache}. Run Phase 3 first."
            )
        total += int(np.load(str(sample)).shape[0])
    return total


def validate_dataset(
    config: dict,
    regions_yaml: str,
    model_configs: Optional[List[str]] = None,
    check_features: bool = True,
    full_alignment_scan: bool = True,
) -> Dict:
    """
    Validate the entire dataset on CPU. Returns a report dict with ok=bool, errors, warnings.
    Hard failures (missing/misaligned files) go in errors; soft issues in warnings.
    """
    import yaml
    paths = config["paths"]
    eco_channels = config["fusion"]["socioeconomic_channels"]
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Manifest
    manifest_path = paths["manifest"]
    if not Path(manifest_path).exists():
        return {"ok": False, "errors": [f"Manifest missing: {manifest_path}. Run tiling first."],
                "warnings": []}
    with open(manifest_path) as f:
        manifest = json.load(f)
    tiles = manifest.get("tiles", [])
    if not tiles:
        return {"ok": False, "errors": ["Manifest has zero tiles."], "warnings": []}

    splits = {s: [t for t in tiles if t["split"] == s] for s in ("train", "val", "test")}
    for s, ts in splits.items():
        if not ts:
            (errors if s == "train" else warnings).append(f"Split '{s}' is empty.")

    # 2. Per-tile file existence + alignment
    tiles_dir = Path(paths["tiles_dir"]); labels_dir = Path(paths["labels_dir"])
    socioec_dir = Path(paths["socioeconomic_dir"])
    scan = tiles if full_alignment_scan else tiles[:25]
    bad_files = 0
    for t in scan:
        tid = t["tile_id"]
        req = {
            "rgb": tiles_dir / f"{tid}_rgb.tif",
            "noisy": labels_dir / f"{tid}_noisy.tif",
            "hc": labels_dir / f"{tid}_hc.tif",
            "socioec": socioec_dir / f"{tid}_socioec.tif",
        }
        missing = [k for k, p in req.items() if not p.exists()]
        if missing:
            bad_files += 1
            if bad_files <= 10:
                errors.append(f"{tid}: missing {missing}")
            continue
        # Alignment: shapes + transforms must match the rgb grid
        try:
            with rasterio.open(str(req["rgb"])) as r:
                base_shape, base_tr = r.shape, r.transform
            for k in ("noisy", "hc", "socioec"):
                with rasterio.open(str(req[k])) as o:
                    if o.shape != base_shape:
                        errors.append(f"{tid}: {k} shape {o.shape} != rgb {base_shape}")
                    if any(abs(a - b) > 1e-6 for a, b in zip(o.transform, base_tr)):
                        errors.append(f"{tid}: {k} geotransform misaligned vs rgb")
            # socioec channels present
            with rasterio.open(str(req["socioec"])) as o:
                desc = {d.lower() for d in (o.descriptions or []) if d}
                for ch in eco_channels:
                    if ch.lower() not in desc:
                        warnings.append(f"{tid}: socioec missing channel '{ch}' (filled zero)")
        except Exception as e:
            errors.append(f"{tid}: read error {e}")

    if bad_files > 10:
        errors.append(f"...and {bad_files - 10} more tiles with missing files.")

    # 3. HC subset sanity
    hc_total = sum(t.get("hc_pixel_count", 0) for t in tiles)
    if hc_total == 0:
        errors.append("HC subset is EMPTY across all tiles — eval would be meaningless.")
    with open(regions_yaml) as f:
        region_names = list(yaml.safe_load(f)["regions"].keys())
    for rn in region_names:
        rt = [t for t in tiles if t["region"] == rn]
        if rt and sum(t.get("hc_pixel_count", 0) for t in rt) == 0:
            warnings.append(f"Region '{rn}' has no HC pixels.")

    # 4. Cached features for the configs to be run
    if check_features and model_configs:
        from .tiles import SlumTileDataset
        cache = Path(paths["features_cache_dir"])
        for mc in model_configs:
            pids = SlumTileDataset._resolve_prompt_ids(mc)
            if not pids:
                continue
            miss = 0
            for t in tiles:
                for pid in pids:
                    if not (cache / f"{t['tile_id']}_{pid}.npy").exists():
                        miss += 1
            if miss:
                errors.append(f"Config '{mc}': {miss} cached feature files missing. Run Phase 3.")
            else:
                try:
                    fd = detect_feature_dim(str(cache), mc)
                    warnings.append(f"Config '{mc}': feature_dim={fd} (set locate_anything.feature_dim).")
                except FileNotFoundError as e:
                    errors.append(str(e))

    report = {
        "ok": len(errors) == 0,
        "n_tiles": len(tiles),
        "splits": {s: len(ts) for s, ts in splits.items()},
        "hc_total_pixels": hc_total,
        "errors": errors,
        "warnings": warnings,
    }
    return report


def print_report(report: Dict):
    print("=" * 60)
    print(f"PREFLIGHT  ok={report['ok']}  tiles={report.get('n_tiles')}  "
          f"splits={report.get('splits')}  hc_px={report.get('hc_total_pixels')}")
    for w in report.get("warnings", []):
        print(f"  [warn]  {w}")
    for e in report.get("errors", []):
        print(f"  [ERROR] {e}")
    print("PASS — safe to spend GPU." if report["ok"]
          else "FAIL — fix the above on CPU before any GPU spend.")
    print("=" * 60)


def _smoke_test():
    # Validate the validator's logic on a missing-manifest case (no real data).
    cfg = {"paths": {"manifest": "/nonexistent/manifest.json", "tiles_dir": "x",
                     "labels_dir": "x", "socioeconomic_dir": "x", "features_cache_dir": "x"},
           "fusion": {"socioeconomic_channels": ["viirs"]}}
    import tempfile, yaml, os
    with tempfile.TemporaryDirectory() as tmp:
        ry = os.path.join(tmp, "r.yaml")
        with open(ry, "w") as f:
            yaml.safe_dump({"regions": {"korail": {}}}, f)
        rep = validate_dataset(cfg, ry)
        assert rep["ok"] is False and rep["errors"]
    print("preflight.py smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
