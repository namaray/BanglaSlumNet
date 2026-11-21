"""
Benchmark grid, ESA reference fractions, cell labels and spatial folds
(docs/MASTER_PLAN.md §5.2–§5.3; parameters in config/benchmark.yaml).

The benchmark unit is a 100 m cell in UTM 46N. ESA polygons are rasterized on the
10 m sub-grid and averaged up, so every cell carries:
  slum_frac  share of the cell inside an ESA slum polygon
  near_slum  cell lies within nonslum_clearance_m of an ESA polygon
  mapped     cell lies fully inside the ESA mapped area shrunk by mapped_inner_buffer_m
Labels then also need per-year Earth Engine layers (built-up share, Open Buildings),
see assign_labels().
"""

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

LABEL_IGNORE, LABEL_NONSLUM, LABEL_SLUM = -1, 0, 1


def load_benchmark_config(path: str = "config/benchmark.yaml") -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class Grid:
    crs: str
    left: float
    top: float
    cell_m: float
    nrows: int
    ncols: int

    def transform(self, res: Optional[float] = None):
        from rasterio.transform import from_origin
        return from_origin(self.left, self.top, res or self.cell_m, res or self.cell_m)

    def ee_crs_transform(self):
        """crsTransform list for Earth Engine exports aligned to this grid."""
        return [self.cell_m, 0, self.left, 0, -self.cell_m, self.top]

    @property
    def bounds(self):
        return (self.left, self.top - self.nrows * self.cell_m,
                self.left + self.ncols * self.cell_m, self.top)


def make_grid(mapped_area, cfg: Dict) -> Grid:
    """Grid covering the mapped area plus pad_m, snapped outward to whole cells."""
    g = cfg["grid"]
    xmin, ymin, xmax, ymax = mapped_area.to_crs(g["crs"]).total_bounds
    c, pad = g["cell_m"], g["pad_m"]
    left = math.floor((xmin - pad) / c) * c
    right = math.ceil((xmax + pad) / c) * c
    bottom = math.floor((ymin - pad) / c) * c
    top = math.ceil((ymax + pad) / c) * c
    return Grid(g["crs"], left, top, c, int(round((top - bottom) / c)), int(round((right - left) / c)))


def _rasterize(geoms, grid: Grid, res: float) -> np.ndarray:
    from rasterio.features import rasterize
    k = int(round(grid.cell_m / res))
    return rasterize([(gm, 1) for gm in geoms if not gm.is_empty],
                     out_shape=(grid.nrows * k, grid.ncols * k),
                     transform=grid.transform(res), fill=0, dtype="uint8")


def _blocks(a: np.ndarray, k: int) -> np.ndarray:
    """View a (nrows*k, ncols*k) array as (nrows, ncols, k*k)."""
    r, c = a.shape[0] // k, a.shape[1] // k
    return a.reshape(r, k, c, k).swapaxes(1, 2).reshape(r, c, k * k)


def esa_fractions(grid: Grid, esa, mapped_area, cfg: Dict) -> Dict[str, np.ndarray]:
    lab, res = cfg["labels"], cfg["grid"]["pixel_m"]
    k = int(round(grid.cell_m / res))
    esa = esa.to_crs(grid.crs)
    polys = list(esa.geometry.buffer(0))
    near = [p.buffer(lab["nonslum_clearance_m"]) for p in polys]
    inner = mapped_area.to_crs(grid.crs).geometry.buffer(-lab["mapped_inner_buffer_m"])
    return {
        "slum_frac": _blocks(_rasterize(polys, grid, res), k).mean(axis=2).astype("float32"),
        "near_slum": _blocks(_rasterize(near, grid, res), k).any(axis=2),
        "mapped": _blocks(_rasterize(list(inner), grid, res), k).all(axis=2),
    }


def assign_labels(frac: Dict[str, np.ndarray], built: np.ndarray, cfg: Dict,
                  ob: Optional[Dict[str, np.ndarray]] = None,
                  ob_ref: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """
    Cell labels: 1 slum, 0 non-slum, -1 ignore. Both classes must be built-up
    (built >= labels.built_min, where built is the year's labels.built_band) and inside
    the mapped area; non-slum cells must also clear every ESA polygon by
    nonslum_clearance_m. For years after the reference year pass the year's Open Buildings
    means (ob) and the reference year's (ob_ref) to mask change: keys 'building_height'
    and 'building_presence'.
    """
    lab = cfg["labels"]
    usable = frac["mapped"] & (built >= lab["built_min"])
    y = np.full(frac["slum_frac"].shape, LABEL_IGNORE, dtype="int8")
    y[usable & (frac["slum_frac"] >= lab["slum_min_frac"])] = LABEL_SLUM
    y[usable & ~frac["near_slum"] & (frac["slum_frac"] <= lab["nonslum_max_frac"])] = LABEL_NONSLUM
    if ob is not None and ob_ref is not None:
        h, h0 = ob["building_height"], ob_ref["building_height"]
        p, p0 = ob["building_presence"], ob_ref["building_presence"]
        changed = (h - h0 >= lab["change_height_m"]) | (p <= lab["change_presence_ratio"] * p0)
        new_build = (p - p0) >= lab["new_presence_rise"]
        y[(y == LABEL_SLUM) & changed] = LABEL_IGNORE
        y[(y == LABEL_NONSLUM) & new_build] = LABEL_IGNORE
    return y


def assign_folds(grid: Grid, labels: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    Fold id per cell (0..n_folds-1). Cells are grouped into block_m blocks and whole
    blocks go to one fold. Blocks are dealt greedily, largest slum count first, to the fold
    with the fewest slum cells so far (ties broken by a seeded shuffle), so every fold gets
    both classes.
    """
    s = cfg["splits"]
    b = int(round(s["block_m"] / grid.cell_m))
    rows, cols = np.indices(labels.shape)
    block = (rows // b) * math.ceil(grid.ncols / b) + (cols // b)
    ids = np.unique(block)
    rng = np.random.default_rng(s["seed"])
    rng.shuffle(ids)
    n_slum = {i: int(((block == i) & (labels == LABEL_SLUM)).sum()) for i in ids}
    n_lab = {i: int(((block == i) & (labels != LABEL_IGNORE)).sum()) for i in ids}
    load_slum, load_lab = np.zeros(s["n_folds"]), np.zeros(s["n_folds"])
    fold_of = {}
    for i in sorted(ids, key=lambda i: -n_slum[i]):  # stable sort keeps the shuffle for ties
        f = int(np.lexsort((load_lab, load_slum))[0])
        fold_of[i] = f
        load_slum[f] += n_slum[i]
        load_lab[f] += n_lab[i]
    return np.vectorize(fold_of.get)(block).astype("int8")


def save_grid(grid: Grid, path: str, **arrays):
    """Grid metadata (JSON sidecar) + arrays (npz) under the same stem."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p.with_suffix(".npz"), **arrays)
    p.with_suffix(".json").write_text(json.dumps(asdict(grid), indent=2))


def load_grid(path: str):
    p = Path(path)
    grid = Grid(**json.loads(p.with_suffix(".json").read_text()))
    return grid, dict(np.load(p.with_suffix(".npz")))
