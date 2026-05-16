import os
import glob
import torch
import rasterio
import numpy as np

DATA_DIR = os.path.join(os.getcwd(), "dhaka_dataset")

clear_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_clear.tif")))[:5]

print("=== Checking first 5 tiles ===\n")

for cf in clear_files:
    tid = os.path.basename(cf).replace("_clear.tif", "")
    p = lambda s: os.path.join(DATA_DIR, f"{tid}_{s}.tif")

    def load(path):
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)[:, :512, :512]

    clear = load(p("clear"))
    ntl   = load(p("ntl"))
    pop   = load(p("pop"))
    gob   = load(p("gob"))
    label = load(p("label"))
    hc    = load(p("hc_mask"))

    clear_norm = np.clip(clear / 3000.0, 0.0, 1.0)

    print(f"Tile: {tid}")
    print(f"  clear  — shape:{clear.shape}  min:{clear.min():.2f}  max:{clear.max():.2f}  nan:{np.isnan(clear).sum()}")
    print(f"  clear_norm — min:{clear_norm.min():.4f}  max:{clear_norm.max():.4f}  nan:{np.isnan(clear_norm).sum()}")
    print(f"  ntl    — min:{ntl.min():.4f}  max:{ntl.max():.4f}  nan:{np.isnan(ntl).sum()}")
    print(f"  pop    — min:{pop.min():.4f}  max:{pop.max():.4f}  nan:{np.isnan(pop).sum()}")
    print(f"  gob    — min:{gob.min():.4f}  max:{gob.max():.4f}  nan:{np.isnan(gob).sum()}")
    print(f"  label  — unique values: {np.unique(label)}")
    print(f"  hc     — unique values: {np.unique(hc)}")
    print()