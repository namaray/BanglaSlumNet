import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np

IGNORE_VALUE = 255   # unknown pixels — from generate_weak_labels.py


class BanglaSlumDataset(Dataset):
    """
    Loads one tile per sample. Requires per-tile files:
      {tile_id}_clear.tif      — Sentinel-2 RGBNIR
      {tile_id}_ntl.tif        — Nighttime lights
      {tile_id}_pop.tif        — Population density
      {tile_id}_gob.tif        — Google Open Buildings
      {tile_id}_label.tif      — 1=slum, 0=formal-dense, 255=unknown
      {tile_id}_hc_mask.tif    — 1=high-confidence pixel, 0=uncertain

    Optional:
      {tile_id}_hazy.tif       — used when use_hazy=True

    Returns a batch dict with keys:
      image           [4, 512, 512]  — clear satellite, normalised
      hazy_image      [4, 512, 512]  — only if use_hazy=True
      aux             [3, 512, 512]  — NTL + Pop + GOB stacked
      target          [1, 512, 512]  — 1=slum, 0=formal-dense (255 replaced with 0)
      ignore_mask     [1, 512, 512]  — 1 where label==255 (unknown), 0 elsewhere
      all_labeled_mask[1, 512, 512]  — 1 where label is 0 or 1 (not unknown)
      hc_slum_mask    [1, 512, 512]  — 1 where HC and label==1 (slum)
      hc_formal_mask  [1, 512, 512]  — 1 where HC and label==0 (formal-dense)
      hc_eval_mask    [1, 512, 512]  — 1 for all HC pixels (slum + formal)
      formal_mask     [1, 512, 512]  — 1 where label==0 (all formal pixels, not just HC)
      tile_id         str
      paths           dict           — only if return_metadata=True
    """

    def __init__(self, data_dir, use_hazy=False, return_metadata=True):
        self.data_dir = data_dir
        self.use_hazy = use_hazy
        self.return_metadata = return_metadata

        clear_files = sorted(glob.glob(os.path.join(data_dir, "*_clear.tif")))
        self.tile_ids = []

        for cf in clear_files:
            tid = os.path.basename(cf).replace("_clear.tif", "")
            required = [
                f"{tid}_clear.tif",
                f"{tid}_ntl.tif",
                f"{tid}_pop.tif",
                f"{tid}_gob.tif",
                f"{tid}_label.tif",
                f"{tid}_hc_mask.tif",
            ]
            if all(os.path.exists(os.path.join(data_dir, f)) for f in required):
                self.tile_ids.append(tid)

        if len(self.tile_ids) == 0:
            raise ValueError(
                f"No complete tiles found in {data_dir}.\n"
                "Run download_dhaka_dataset.py then generate_weak_labels.py first."
            )

        print(f"✅ BanglaSlumDataset: {len(self.tile_ids)} complete tiles loaded.")

    def __len__(self):
        return len(self.tile_ids)

    def _path(self, tile_id, suffix):
        return os.path.join(self.data_dir, f"{tile_id}_{suffix}.tif")

    def _read_tif(self, path):
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)[:, :512, :512]   # [C, H, W]

    def _read_mask(self, path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)[:512, :512]
        return arr[None, :, :]   # [1, H, W]

    def _norm_satellite(self, arr):
        return np.clip(arr / 3000.0, 0.0, 1.0)

    def _norm_aux(self, arr):
        vmax = float(arr.max())
        return arr / vmax if vmax > 0 else arr

    def __getitem__(self, idx):
        tid = self.tile_ids[idx]

        # ---- Satellite imagery ----
        clear = self._norm_satellite(self._read_tif(self._path(tid, "clear")))
        sample = {
            "tile_id": tid,
            "image": torch.from_numpy(clear).float(),   # [4, 512, 512]
        }

        if self.use_hazy:
            hazy_path = self._path(tid, "hazy")
            if os.path.exists(hazy_path):
                hazy = self._norm_satellite(self._read_tif(hazy_path))
                sample["hazy_image"] = torch.from_numpy(hazy).float()
            else:
                # Fall back to clear if hazy missing — Stage 1 can still run
                sample["hazy_image"] = sample["image"].clone()

        # ---- Socioeconomic aux stack ----
        ntl = self._norm_aux(self._read_tif(self._path(tid, "ntl")))
        pop = self._norm_aux(self._read_tif(self._path(tid, "pop")))
        gob = self._norm_aux(self._read_tif(self._path(tid, "gob")))
        aux = np.concatenate([ntl, pop, gob], axis=0)              # [3, 512, 512]
        sample["aux"] = torch.from_numpy(aux).float()

        # ---- Labels from generate_weak_labels.py ----
        # label.tif   : 1=slum, 0=formal-dense, 255=unknown
        # hc_mask.tif : 1=high-confidence, 0=uncertain
        raw_label = self._read_mask(self._path(tid, "label"))      # [1, 512, 512]
        hc_mask   = self._read_mask(self._path(tid, "hc_mask"))    # [1, 512, 512]

        # ignore_mask  — 1 where unknown (255), used to zero-out loss
        ignore_mask = (raw_label == IGNORE_VALUE).astype(np.float32)

        # target — clean label with 255 replaced by 0 (masked out via ignore_mask)
        target = raw_label.copy()
        target[raw_label == IGNORE_VALUE] = 0

        # all_labeled_mask — 1 wherever we have a definite label (0 or 1)
        all_labeled_mask = (1.0 - ignore_mask).astype(np.float32)

        # formal_mask — 1 wherever label == 0 (formal-dense), all pixels
        formal_mask = (target == 0).astype(np.float32) * all_labeled_mask

        # HC-split masks (derived from hc_mask + target)
        hc_slum_mask   = hc_mask * (target == 1).astype(np.float32)   # HC & slum
        hc_formal_mask = hc_mask * (target == 0).astype(np.float32)   # HC & formal
        hc_eval_mask   = hc_mask * all_labeled_mask                    # all HC pixels

        sample["target"]            = torch.from_numpy(target).float()
        sample["ignore_mask"]       = torch.from_numpy(ignore_mask).float()
        sample["all_labeled_mask"]  = torch.from_numpy(all_labeled_mask).float()
        sample["formal_mask"]       = torch.from_numpy(formal_mask).float()
        sample["hc_slum_mask"]      = torch.from_numpy(hc_slum_mask).float()
        sample["hc_formal_mask"]    = torch.from_numpy(hc_formal_mask).float()
        sample["hc_eval_mask"]      = torch.from_numpy(hc_eval_mask).float()

        if self.return_metadata:
            sample["paths"] = {
                "clear":   self._path(tid, "clear"),
                "ntl":     self._path(tid, "ntl"),
                "pop":     self._path(tid, "pop"),
                "gob":     self._path(tid, "gob"),
                "label":   self._path(tid, "label"),
                "hc_mask": self._path(tid, "hc_mask"),
            }

        return sample


# ==========================================
# SMOKE TEST
# ==========================================
if __name__ == "__main__":
    dataset = BanglaSlumDataset(data_dir="dhaka_dataset", use_hazy=True)
    loader  = DataLoader(dataset, batch_size=2, shuffle=False)

    print(f"\n✅ Tiles found: {len(dataset)}")

    for batch in loader:
        print(f"image          : {batch['image'].shape}")
        print(f"hazy_image     : {batch['hazy_image'].shape}")
        print(f"aux            : {batch['aux'].shape}")
        print(f"target         : {batch['target'].shape}  "
              f"values={batch['target'].unique().tolist()}")
        print(f"ignore_mask    : {batch['ignore_mask'].shape}  "
              f"ignored%={(batch['ignore_mask'].mean()*100):.1f}")
        print(f"all_labeled    : {batch['all_labeled_mask'].shape}")
        print(f"hc_slum_mask   : {batch['hc_slum_mask'].shape}")
        print(f"hc_formal_mask : {batch['hc_formal_mask'].shape}")
        print(f"hc_eval_mask   : {batch['hc_eval_mask'].shape}")
        print(f"formal_mask    : {batch['formal_mask'].shape}")
        print(f"tile_ids       : {batch['tile_id']}")
        break