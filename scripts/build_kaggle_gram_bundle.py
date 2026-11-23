"""
Assemble the private Kaggle Dataset for the GRAM run (E1): GRAM's model code and released
weights, the Esri Wayback zoom-16 tiles of one release, and scripts/kaggle/gram_tta_kaggle.py.

Inputs (git-ignored):
  data/raw/gram_code/                 model.py + checkpoint/, from the gram-zero-shot branch
  data/raw/esri_wayback_z16/<release> tiles from scripts/fetch_esri_wayback_tiles.py

Upload the resulting zip to Kaggle as a PRIVATE dataset: the tiles are Esri imagery and must
not be redistributed.

CLI:
    python scripts/build_kaggle_gram_bundle.py --release 10768
Writes data/derived/kaggle_gram_bundle/ and data/derived/kaggle_gram_bundle.zip.
"""

import argparse
import glob
import os
import shutil

README = """BanglaSlumNet - GRAM on Dhaka (private; contains Esri imagery, do not redistribute)

Kaggle notebook: Add Data -> this dataset; Settings -> Accelerator: GPU T4 x2 (or P100).
Run one cell:
    !python /kaggle/input/{slug}/gram_tta_kaggle.py
Then download /kaggle/working/gram_outputs.zip from the Output tab.

Esri Wayback release {release}, {n} zoom-16 tiles.
GRAM: Lee et al., AAAI 2026, https://github.com/DS4H-GIS/GRAM
"""


def run(release, out_dir):
    code = os.path.join("data", "raw", "gram_code")
    tiles = os.path.join("data", "raw", "esri_wayback_z16", str(release))
    jpgs = sorted(glob.glob(os.path.join(tiles, "*.jpg")))
    for need in [os.path.join(code, "model.py"), os.path.join(code, "checkpoint", "MOE_epoch_2_v2.pth")]:
        if not os.path.exists(need):
            raise FileNotFoundError(f"{need} missing; extract it from the gram-zero-shot branch")
    if not jpgs:
        raise FileNotFoundError(f"No tiles in {tiles}; run scripts/fetch_esri_wayback_tiles.py first")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "checkpoint"))
    os.makedirs(os.path.join(out_dir, "tiles"))
    shutil.copy(os.path.join(code, "model.py"), out_dir)
    shutil.copy(os.path.join(code, "checkpoint", "MOE_epoch_2_v2.pth"), os.path.join(out_dir, "checkpoint"))
    shutil.copy(os.path.join("scripts", "kaggle", "gram_tta_kaggle.py"), out_dir)
    for j in jpgs:
        shutil.copy(j, os.path.join(out_dir, "tiles"))
    slug = "banglaslumnet-gram-dhaka"
    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(README.format(slug=slug, release=release, n=len(jpgs)))
    archive = shutil.make_archive(out_dir, "zip", out_dir)
    print(f"  bundle: {out_dir}/ ({len(jpgs)} tiles) -> {archive} ({os.path.getsize(archive)/1e6:.0f} MB)")
    print(f"  suggested Kaggle dataset name: {slug} (set it to Private)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--release", type=int, required=True)
    p.add_argument("--out", default=os.path.join("data", "derived", "kaggle_gram_bundle"))
    a = p.parse_args()
    run(a.release, a.out)
