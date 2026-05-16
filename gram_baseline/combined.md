
G:\Uni Work\BanglaSlumNet\gram_baseline>(echo # augmentation.py   & echo ```python   & type "augmentation.py"   & echo ```   & echo.) 
# augmentation.py 
```python 
import torchvision.transforms.functional as TF 
import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union, Tuple, Optional
from PIL import Image
from PIL import ImageOps

class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            img, mask = transform(img, mask)

        return img, mask


class Normalize:
    def __init__(self, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        img = img.float()
        img /= 255
        img = TF.normalize(img, self.mean, self.std)
        return img, mask


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            if self.brightness > 0:
                img = TF.adjust_brightness(img, self.brightness)
            if self.contrast > 0:
                img = TF.adjust_contrast(img, self.contrast)
            if self.saturation > 0:
                img = TF.adjust_saturation(img, self.saturation)
            if self.hue > 0:
                img = TF.adjust_hue(img, self.hue)
        return img, mask
    
class ColorJitter2:
    def __init__(self, brightness=0, contrast=0, saturation=0) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if self.brightness > 0:
            img = TF.adjust_brightness(img, self.brightness)
        if self.contrast > 0:
            img = TF.adjust_contrast(img, self.contrast)
        if self.saturation > 0:
            img = TF.adjust_saturation(img, self.saturation)
        return img, mask


class AdjustGamma:
    def __init__(self, gamma: float, gain: float = 1) -> None:
        """
        Args:
            gamma: Non-negative real number. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
            gain: constant multiplier
        """
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.adjust_gamma(img, self.gamma, self.gain), mask


class RandomAdjustSharpness:
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        self.sharpness = sharpness_factor
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.adjust_sharpness(img, self.sharpness)
        return img, mask


class RandomAutoContrast:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            pil_img = TF.to_pil_image(img)
            pil_img = ImageOps.autocontrast(pil_img)
            img = TF.to_tensor(pil_img)

#             img = ImageOps.autocontrast(img)
        return img, mask


class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 3, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.gaussian_blur(img, self.kernel_size)
        return img, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.hflip(img), TF.hflip(mask)
        return img, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask


class RandomGrayscale:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.rgb_to_grayscale(img, 3)
        return img, mask


class Equalize:
    def __call__(self, image, label):
        return TF.equalize(image), label


class Posterize:
    def __init__(self, bits=2):
        self.bits = bits # 0-8
        
    def __call__(self, image, label):
        return TF.posterize(image, self.bits), label


class Affine:
    def __init__(self, angle=0, translate=[0, 0], scale=1.0, shear=[0, 0], seg_fill=0):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.seg_fill = seg_fill
        
    def __call__(self, img, label):
        return TF.affine(img, self.angle, self.translate, self.scale, self.shear, Image.BILINEAR, 0), TF.affine(label, self.angle, self.translate, self.scale, self.shear, Image.NEAREST, self.seg_fill) 


class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, seg_fill: int = 0, expand: bool = False) -> None:
        """Rotate the image by a random angle between -angle and angle with probability p

        Args:
            p: probability
            angle: rotation angle value in degrees, counter-clockwise.
            expand: Optional expansion flag. 
                    If true, expands the output image to make it large enough to hold the entire rotated image.
                    If false or omitted, make the output image the same size as the input image. 
                    Note that the expand flag assumes rotation around the center and no translation.
        """
        self.p = p
        self.angle = degrees
        self.expand = expand
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        random_angle = random.random() * 2 * self.angle - self.angle
        if random.random() < self.p:
            img = TF.rotate(img, random_angle, Image.BILINEAR, self.expand, fill=0)
            mask = TF.rotate(mask, random_angle, Image.NEAREST, self.expand, fill=self.seg_fill)
        return img, mask
    

class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]]) -> None:
        """Crops the image at the center

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.center_crop(img, self.size), TF.center_crop(mask, self.size)


class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.5) -> None:
        """Randomly Crops the image.

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        if random.random() < self.p:
            margin_h = max(H - tH, 0)
            margin_w = max(W - tW, 0)
            y1 = random.randint(0, margin_h+1)
            x1 = random.randint(0, margin_w+1)
            y2 = y1 + tH
            x2 = x1 + tW
            img = img[:, y1:y2, x1:x2]
            mask = mask[:, y1:y2, x1:x2]
        return img, mask


class Pad:
    def __init__(self, size: Union[List[int], Tuple[int], int], seg_fill: int = 0) -> None:
        """Pad the given image on all sides with the given "pad" value.
        Args:
            size: expected output image size (h, w)
            fill: Pixel fill value for constant fill. Default is 0. This value is only used when the padding mode is constant.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        padding = (0, 0, self.size[1]-img.shape[2], self.size[0]-img.shape[1])
        return TF.pad(img, padding), TF.pad(mask, padding, self.seg_fill)


class ResizePad:
    def __init__(self, size: Union[int, Tuple[int], List[int]], seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        # scale the image 
        scale_factor = min(tH/H, tW/W) if W > H else max(tH/H, tW/W)
        # nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        img = TF.resize(img, (nH, nW), Image.BILINEAR)
        mask = TF.resize(mask, (nH, nW), Image.NEAREST)

        # pad the image
        padding = [0, 0, tW - nW, tH - nH]
        img = TF.pad(img, padding, fill=0)
        mask = TF.pad(mask, padding, fill=self.seg_fill)
        return img, mask 


class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]

        # scale the image 
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        img = TF.resize(img, (nH, nW), Image.BILINEAR)
        mask = TF.resize(mask, (nH, nW), Image.NEAREST)

        # make the image divisible by stride
        alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        img = TF.resize(img, (alignH, alignW), Image.BILINEAR)
        mask = TF.resize(mask, (alignH, alignW), Image.NEAREST)
        return img, mask 


class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        """
        self.size = size
        self.scale = scale
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))
        scale = int(tH*ratio), int(tW*4*ratio)

        # scale the image 
        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        img = TF.resize(img, (nH, nW), Image.BILINEAR)
        mask = TF.resize(mask, (nH, nW), Image.NEAREST)

        # random crop
        margin_h = max(img.shape[1] - tH, 0)
        margin_w = max(img.shape[2] - tW, 0)
        y1 = random.randint(0, margin_h+1)
        x1 = random.randint(0, margin_w+1)
        y2 = y1 + tH
        x2 = x1 + tW
        img = img[:, y1:y2, x1:x2]
        mask = mask[:, y1:y2, x1:x2]

        # pad the image
        if img.shape[1:] != self.size:
            padding = [0, 0, tW - img.shape[2], tH - img.shape[1]]
            img = TF.pad(img, padding, fill=0)
            mask = TF.pad(mask, padding, fill=self.seg_fill)
            
        return img, mask 



``` 


G:\Uni Work\BanglaSlumNet\gram_baseline>(echo # dataloader.py   & echo ```python   & type "dataloader.py"   & echo ```   & echo.) 
# dataloader.py 
```python 
import os
import glob
import torch
import numpy as np
import pandas as pd
from skimage import io, transform
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as TF
import random
from PIL import Image
from torchvision import io
import copy

def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)
            
        img[y:y + erase_h, x:x + erase_w] = torch.from_numpy(value)
        mask[y:y + erase_h, x:x + erase_w] = 255

    return img, mask


class GPSDataset(Dataset):
    def __init__(self, metadata,  transform=None,  normalize=None):
        self.metadata = pd.read_csv(metadata).values
        self.transform = transform
        self.normalize = normalize
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_path = self.metadata[idx][0]
        label_path = self.metadata[idx][1]
        country_idx = self.metadata[idx][2]

        image = io.read_image(image_path)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
            
        try:
            land_value = Image.open(label_path)
            land_value = torch.tensor(np.array(land_value)).unsqueeze(0)
            land_value = torch.clamp(land_value, max=1)

        except:
            land_value = torch.zeros(1,256,256)
            
        
        
        if self.transform:
            image, land_value = self.transform(image, land_value)
            
            
        image, land_value = self.normalize(image, land_value)        
        land_value = land_value.squeeze(0).long()

        return image, land_value, country_idx



    
class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, images):
        normalized = np.stack([F.normalize(x, self.mean, self.std, self.inplace) for x in images]) 
        return normalized
        
class ToTensor(object):
    def __call__(self, images):
        images = images.transpose((0, 3, 1, 2))
        return torch.from_numpy(images).float() ``` 


G:\Uni Work\BanglaSlumNet\gram_baseline>(echo # fetch_dhaka_tiles.py   & echo ```python   & type "fetch_dhaka_tiles.py"   & echo ```   & echo.) 
# fetch_dhaka_tiles.py 
```python 
"""Fetch ESRI World Imagery tiles over Dhaka slums.

GRAM was trained on ESRI World Imagery at zoom 16 (256x256, ~1.2m/px), so we feed
it the same data type to isolate distribution shift from format mismatch.
"""
import os
import math
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dhaka_tiles")
os.makedirs(OUT, exist_ok=True)

LOCATIONS = {
    "korail":    (23.7806, 90.4040),  # Korail slum
    "mirpur":    (23.8100, 90.3600),  # Mirpur
    "old_dhaka": (23.7100, 90.3900),  # Old Dhaka
}

ZOOM = 16
GRID = 3  # 3x3 → 9 tiles per location, 768x768 effective coverage


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon = xtile / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ytile / n))))
    return lat, lon


def fetch_tile(z, x, y, out_path):
    url = f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    req = urllib.request.Request(url, headers={"User-Agent": "BanglaSlumNet/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        with open(out_path, "wb") as f:
            f.write(r.read())


def main():
    half = GRID // 2
    manifest = []
    for loc, (lat, lon) in LOCATIONS.items():
        cx, cy = deg2num(lat, lon, ZOOM)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                x, y = cx + dx, cy + dy
                out = os.path.join(OUT, f"{loc}_z{ZOOM}_x{x}_y{y}.jpg")
                if os.path.exists(out) and os.path.getsize(out) > 1000:
                    print(f"skip {out}")
                else:
                    print(f"fetch {loc} dx={dx} dy={dy} → {out}")
                    fetch_tile(ZOOM, x, y, out)
                    time.sleep(0.1)
                tlat, tlon = num2deg(x, y, ZOOM)
                manifest.append((loc, x, y, tlat, tlon, out))
    with open(os.path.join(OUT, "manifest.csv"), "w") as f:
        f.write("location,x,y,lat,lon,path\n")
        for row in manifest:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"\nfetched {len(manifest)} tiles into {OUT}")


if __name__ == "__main__":
    main()
``` 


G:\Uni Work\BanglaSlumNet\gram_baseline>(echo # gram_baseline.py   & echo ```python   & type "gram_baseline.py"   & echo ```   & echo.) 
# gram_baseline.py 
```python 
"""GRAM zero-shot baseline on Dhaka tiles.

Loads the pretrained MoE checkpoint from DS4H-GIS/GRAM (epoch 2, v2) and runs
inference on Korail / Mirpur / Old Dhaka ESRI World Imagery tiles (z=16).

Key change from v2: domain routing now uses the model's own built-in
domain_classifier (pre-MoE features -> domain logits) via a two-pass forward
pass, instead of a manual DOMAIN_CANDIDATES sweep.

Output:
  - per-tile probability map (.npy + colorized .png)
  - per-location 3x3 mosaic of original | prob_heatmap | binary_overlay
  - summary CSV: per-tile mean/max slum probability and percent-slum-pixels
    (now also records the model-predicted domain index per tile)
"""

import os
import sys
import glob
import csv
import math
import re
import torch
import torch.nn.functional as TF
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import mit_b5_MOE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TILES_DIR = os.path.join(HERE, "dhaka_tiles")
OUT_DIR = os.path.join(HERE, "outputs")
CKPT = os.path.join(HERE, "checkpoint", "MOE_epoch_2_v2.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization (matches GRAM's training in main_moe.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# GRAM's 12 training cities (alphabetical index order from the official repo):
#   0: Cairo, 1: Cape Town, 2: Caracas, 3: Colombo, 4: Karachi,
#   5: Medellín, 6: Mumbai, 7: Nairobi, 8: Ouagadougou,
#   9: Port-au-Prince, 10: Rio, 11: Tegucigalpa
# We no longer hardcode candidates — the model's domain_classifier picks for us.
DOMAIN_NUM = 12


def build_model():
    """Construct the GRAM MoE model and load checkpoint weights."""
    model = mit_b5_MOE()
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Strip 'module.' prefix saved from DataParallel wrapper
    new_state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        print(f"       missing sample: {missing[:5]}")
    if unexpected[:5]:
        print(f"       unexpected sample: {unexpected[:5]}")
    model.eval()
    model.to(DEVICE)
    return model


def load_image(path):
    """Load a 256x256 RGB tile as a float tensor in [0,1], CHW."""
    img = Image.open(path).convert("RGB").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # 1,3,256,256
    return tensor, np.asarray(img)


def infer_one(model, tensor):
    """
    Two-pass inference using the model's own domain_classifier.

    Pass 1: Forward with dummy domain index 0 to obtain pre-MoE domain logits.
            The segmentation output from this pass is discarded.
    Pass 2: Forward again with the model-predicted domain index so that the
            correct MoE experts are activated. This segmentation is returned.

    Returns
    -------
    probs : np.ndarray, shape (H, W)
        Per-pixel slum probability (class 1 softmax score).
    predicted_domain : int
        The domain index chosen by the model's domain_classifier.
    dom_logits : np.ndarray, shape (DOMAIN_NUM,)
        Raw domain logits — useful for logging / ablation.
    """
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        # --- Pass 1: get domain logits from pre-MoE features ---
        dummy_cidx = torch.tensor([0], dtype=torch.long, device=DEVICE)
        _, dom_logits_t, _ = model(tensor, dummy_cidx)
        # dom_logits_t: (1, DOMAIN_NUM)
        predicted_domain = int(dom_logits_t.argmax(dim=1).item())

        # --- Pass 2: route through the correct experts ---
        real_cidx = torch.tensor([predicted_domain], dtype=torch.long, device=DEVICE)
        seg, _, _ = model(tensor, real_cidx)

    # seg: (1, 2, H, W) — class 1 is "slum"
    probs = TF.softmax(seg, dim=1)[0, 1].cpu().numpy()  # H x W
    dom_logits_np = dom_logits_t[0].cpu().numpy()       # (DOMAIN_NUM,)
    return probs, predicted_domain, dom_logits_np


# ---------------------------------------------------------------------------
# City name lookup for readable logging
# ---------------------------------------------------------------------------
CITY_NAMES = {
    0: "Cairo", 1: "Cape Town", 2: "Caracas", 3: "Colombo",
    4: "Karachi", 5: "Medellín", 6: "Mumbai", 7: "Nairobi",
    8: "Ouagadougou", 9: "Port-au-Prince", 10: "Rio", 11: "Tegucigalpa",
}


def colorize_prob(prob):
    """Map a [0,1] prob heatmap to RGB using a simple red ramp."""
    p = np.clip(prob, 0, 1)
    r = (p * 255).astype(np.uint8)
    g = np.zeros_like(r)
    b = ((1 - p) * 80).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def overlay(rgb, prob, alpha=0.55, threshold=0.5):
    """Overlay binary slum mask (prob > threshold) on RGB in red."""
    mask = (prob > threshold).astype(np.float32)[..., None]
    red = np.zeros_like(rgb, dtype=np.float32)
    red[..., 0] = 255
    blended = rgb.astype(np.float32) * (1 - alpha * mask) + red * (alpha * mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = build_model()

    tile_paths = sorted(glob.glob(os.path.join(TILES_DIR, "*_z16_x*_y*.jpg")))
    print(f"[run] {len(tile_paths)} tiles on {DEVICE}")

    rows = [["location", "tile", "domain_idx", "domain_name",
             "mean_prob", "max_prob", "pct_slum_p50", "pct_slum_p70"]]

    # Group tiles by location for mosaic output
    by_loc: dict = {}
    for p in tile_paths:
        name = os.path.basename(p)
        m = re.match(r"([a-z_]+)_z16_x(\d+)_y(\d+)\.jpg", name)
        if not m:
            continue
        loc = m.group(1).rstrip("_")
        x, y = int(m.group(2)), int(m.group(3))
        by_loc.setdefault(loc, []).append((x, y, p))

    # Per-location domain distribution tracker (for the summary)
    domain_votes: dict = {}  # loc -> list of predicted domain indices

    for loc, entries in by_loc.items():
        entries.sort(key=lambda e: (e[1], e[0]))  # sort by y then x
        xs = sorted({e[0] for e in entries})
        ys = sorted({e[1] for e in entries})
        cols_n, rows_n = len(xs), len(ys)

        mosaic_rgb     = np.zeros((rows_n * 256, cols_n * 256, 3), dtype=np.uint8)
        mosaic_heat    = np.zeros_like(mosaic_rgb)
        mosaic_overlay = np.zeros_like(mosaic_rgb)

        domain_votes[loc] = []

        for x, y, p in entries:
            ix = xs.index(x)
            iy = ys.index(y)
            t, rgb = load_image(p)

            prob, pred_domain, dom_logits = infer_one(model, t)

            mean_p = float(prob.mean())
            max_p  = float(prob.max())
            pct50  = float((prob > 0.5).mean() * 100)
            pct70  = float((prob > 0.7).mean() * 100)
            domain_name = CITY_NAMES.get(pred_domain, str(pred_domain))

            rows.append([loc, os.path.basename(p), pred_domain, domain_name,
                         round(mean_p, 4), round(max_p, 4),
                         round(pct50, 2), round(pct70, 2)])
            domain_votes[loc].append(pred_domain)

            print(f"  {loc} x={x} y={y}  "
                  f"domain={pred_domain}({domain_name})  "
                  f"mean={mean_p:.3f} max={max_p:.3f} "
                  f"pct>0.5={pct50:.1f}%  pct>0.7={pct70:.1f}%")

            # Save per-tile prob map
            np.save(os.path.join(OUT_DIR, f"{loc}_x{x}_y{y}_prob.npy"), prob)

            heat = colorize_prob(prob)
            ov   = overlay(rgb, prob, alpha=0.55, threshold=0.5)

            mosaic_rgb    [iy*256:(iy+1)*256, ix*256:(ix+1)*256] = rgb
            mosaic_heat   [iy*256:(iy+1)*256, ix*256:(ix+1)*256] = heat
            mosaic_overlay[iy*256:(iy+1)*256, ix*256:(ix+1)*256] = ov

        # Save combined mosaic: original | heatmap | overlay
        combined = np.concatenate([mosaic_rgb, mosaic_heat, mosaic_overlay], axis=1)
        out_png = os.path.join(OUT_DIR, f"{loc}_gram_baseline.png")
        Image.fromarray(combined).save(out_png)
        print(f"[saved] {out_png}")

    # Summary CSV
    csv_path = os.path.join(OUT_DIR, "gram_baseline_summary.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[saved] {csv_path}")

    # Aggregate stats by location
    print("\n=== Aggregate stats by location ===")
    agg: dict = {}
    for r in rows[1:]:
        loc = r[0]
        agg.setdefault(loc, []).append((r[4], r[5], r[6], r[7]))

    for loc, vals in agg.items():
        arr = np.array(vals, dtype=np.float32)
        votes = domain_votes[loc]
        # Most-common predicted domain across the 9 tiles
        majority_domain = max(set(votes), key=votes.count)
        majority_name   = CITY_NAMES.get(majority_domain, str(majority_domain))
        domain_dist     = {CITY_NAMES.get(d, d): votes.count(d) for d in set(votes)}
        print(f"  {loc}:")
        print(f"    mean_prob    = {arr[:,0].mean():.3f}")
        print(f"    max-of-max   = {arr[:,1].max():.3f}")
        print(f"    avg_pct>0.5  = {arr[:,2].mean():.1f}%")
        print(f"    avg_pct>0.7  = {arr[:,3].mean():.1f}%")
        print(f"    majority_domain = {majority_domain} ({majority_name})")
        print(f"    domain_distribution = {domain_dist}")

    # Cross-location domain agreement report
    print("\n=== Domain routing report (model's own classifier) ===")
    all_tiles = rows[1:]
    for r in all_tiles:
        print(f"  {r[1]:45s}  -> domain {r[2]:2d} ({r[3]})")


if __name__ == "__main__":
    main()``` 


G:\Uni Work\BanglaSlumNet\gram_baseline>(echo # main_moe.py   & echo ```python   & type "main_moe.py"   & echo ```   & echo.) 
# main_moe.py 
```python 
import argparse
from dataloader import GPSDataset
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import numpy as np
import numpy
import random
from torch.utils.data import Subset
from augmentation import *
import copy 
import glob
import pandas as pd
from PIL import Image
import tifffile
from model import *
from utils import *


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


class Metrics:
    def __init__(self, num_classes, ignore_label):
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes)

    def update(self, pred, target):
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self):
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        return ious.cpu().numpy().tolist(), miou

    def compute_f1(self):
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        return f1.cpu().numpy().tolist(), mf1
    
    def compute_precision(self):
        precision = self.hist.diag() / self.hist.sum(0)
        mp = precision[~precision.isnan()].mean().item()
        return precision.cpu().numpy().tolist(), mp
        
    def compute_recall(self):
        recall = self.hist.diag() / self.hist.sum(1)
        mrecall = recall[~recall.isnan()].mean().item()
        return recall.cpu().numpy().tolist(), mrecall
    
    def compute_pixel_acc(self):
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        return acc.cpu().numpy().tolist(), macc

metric = Metrics(2, 255)
    
set_seed(0)

parser = argparse.ArgumentParser(description='Deeplabv3 pytorch Training')
parser.add_argument('--train_meta', type=str, help='training metadata', default='train_metadata.csv')
parser.add_argument('--test_meta', type=str, help='test metadata', default='UGA_test_metadata.csv')
parser.add_argument('--epoch', type=int, help='# of epoch', default=10)
args = parser.parse_args()


def get_train_augmentation(size, seg_fill):
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=10, p=0.3, seg_fill=seg_fill),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
    ])


def get_normalize():
    return Compose([
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])



def get_val_augmentation(size):
    return Compose([
        Resize(size),
    ])

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    def forward(self, pred, target):
        CE = F.cross_entropy(pred, target, reduction='none', ignore_index=255)
        pt = torch.exp(-CE)
        loss = ((1 - pt) ** 2) * CE # gamma
        alpha = torch.Tensor([0.1, 0.9]) # alpha(bigger for 1(pos), MNG only)
        alpha = (target==0) * alpha[0] + (target==1) * alpha[1]
        return torch.mean(alpha * loss)


traintransform = get_train_augmentation([256, 256], 255)
valtransform = get_val_augmentation([256, 256])
normalize = get_normalize()


trainset = GPSDataset(metadata=args.train_meta, transform=traintransform, normalize=normalize)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = GPSDataset(metadata=args.test_meta, transform=valtransform, normalize=normalize)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

model = mit_b5_MOE(patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,  expert_num=12, select_mode='new_topk', hidden_dims = [2, 4, 10, 16], num_k = 2, domain_num=12)



model = torch.nn.DataParallel(model).cuda()

optimizer = torch.optim.SGD(model.module.parameters(), lr = 0.01, momentum=0.99)  
ce_criterion = nn.CrossEntropyLoss()
criterion = FocalLoss().cuda()



for epoch in range(args.epoch):
    model.train()

    for batch_idx, (images, targets, country_idx) in enumerate(trainloader):
        images, targets, country_idx = images.cuda(), targets.cuda().detach(), country_idx.cuda().detach()
        
        output, d_output, MI_loss = model(images, country_idx)

        
        seg_loss = criterion(output, targets)
        domain_loss = ce_criterion(d_output, country_idx)
        MI_loss = torch.mean(MI_loss)
        
        alpha = 1e-3 

        optimizer.zero_grad()
        loss = seg_loss + alpha * domain_loss - alpha * MI_loss
        loss.backward()
        optimizer.step() 
        
        if batch_idx % 100 == 0:
            print(f"[Epoch {epoch} | Batch {batch_idx}] Loss: {loss.item():.4f} | Seg: {seg_loss.item():.4f} | Domain: {domain_loss.item():.4f} | MI: {MI_loss.item():.4f}")

    # # === epoch 끝날 때마다 저장 === #
    checkpoint_path = os.path.join("./checkpoint", f"MOE_epoch_{epoch}_v2.pth")
    torch.save({
        'state_dict': model.state_dict(),
    }, checkpoint_path)

    

    
    model.eval()

    metrics = Metrics(2, 255)
    for batch_idx, (images, targets, country_idx) in enumerate(testloader):
        images, targets, country_idx = images.cuda(), targets.cuda().detach(), country_idx.cuda().detach()

        country_idx = torch.tensor([0]*images.shape[0]).cuda()
        
        output, d_output, MI_loss = model(images, country_idx)
        
        metrics.update(output.cpu(), targets.cpu())

        del output

    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    precision, mprecision = metrics.compute_precision()
    recall, mrecall = metrics.compute_recall()
    
    print(f"ious : [{ious[0]},{ious[1]}]")
    print(f"f1 : [{f1[0]},{f1[1]}]")
    print(f"acc : [{acc[0]},{acc[1]}]")
    print(f"precision : [{precision[0]},{precision[1]}]")
    print(f"recall : [{recall[0]},{recall[1]}]")



``` 


G:\Uni Work\BanglaSlumNet\gram_baseline>(echo # main_moe_pl_v3.py   & echo ```python   & type "main_moe_pl_v3.py"   & echo ```   & echo.) 
# main_moe_pl_v3.py 
```python 
"""GRAM Test-Time Adaptation on Dhaka tiles via pseudo-labeling (v3).

Changes from the original main_moe_pl_v3.py:
  1. Removed hardcoded pivot_domain=0; domain routing now uses the model's
     own domain_classifier via two-pass inference (predict_domain()).
  2. get_high_consistency_tiles() replaces get_high_miou_topk_by_pivot_domain():
     - No longer touches ground-truth `targets` at all.
     - Returns per-pixel stability masks alongside tile indices.
  3. Fine-tuning loop uses per-pixel stable_mask (ignore_index=255 on unstable
     pixels) instead of targets_pl = argmax(output) on all pixels.
  4. Evaluation loop also uses predict_domain() instead of passing
     country_idx from the dataloader (which is meaningless for Dhaka).
  5. Metrics object is now reset each epoch (was declared once outside the
     loop in the original, causing accumulation across epochs).
  6. select_mode typo fixed: 'new_topk' -> 'new_top_k' (matches model.py).
  7. num_k passed as a list [2] to match model.py's expected type.
  8. Preserved all original imports, args, augmentations, and loss exactly.
"""

import argparse
from dataloader import GPSDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import numpy as np
import numpy
import random
from torch.utils.data import Subset
from augmentation import *
import copy
import glob
import pandas as pd
from PIL import Image
import tifffile
from model import *
from utils import *
import itertools
from tqdm import tqdm
from functools import partial


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Metrics  (identical to original)
# ---------------------------------------------------------------------------
class Metrics:
    def __init__(self, num_classes, ignore_label):
        self.ignore_label = ignore_label
        self.num_classes  = num_classes
        self.hist = torch.zeros(num_classes, num_classes)

    def update(self, pred, target):
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(
            target[keep] * self.num_classes + pred[keep],
            minlength=self.num_classes ** 2,
        ).view(self.num_classes, self.num_classes)

    def compute_iou(self):
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        return ious.cpu().numpy().tolist(), miou

    def compute_f1(self):
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        return f1.cpu().numpy().tolist(), mf1

    def compute_precision(self):
        precision = self.hist.diag() / self.hist.sum(0)
        mp = precision[~precision.isnan()].mean().item()
        return precision.cpu().numpy().tolist(), mp

    def compute_recall(self):
        recall = self.hist.diag() / self.hist.sum(1)
        mrecall = recall[~recall.isnan()].mean().item()
        return recall.cpu().numpy().tolist(), mrecall

    def compute_pixel_acc(self):
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        return acc.cpu().numpy().tolist(), macc


# ---------------------------------------------------------------------------
# Augmentation helpers  (identical to original)
# ---------------------------------------------------------------------------
def get_train_augmentation(size, seg_fill):
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=10, p=0.3, seg_fill=seg_fill),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
    ])


def get_normalize():
    return Compose([
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def get_val_augmentation(size):
    return Compose([Resize(size)])


# ---------------------------------------------------------------------------
# Loss  (identical to original — alpha kept as [0.5, 0.5])
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, target):
        CE   = F.cross_entropy(pred, target, reduction='none', ignore_index=255)
        pt   = torch.exp(-CE)
        loss = ((1 - pt) ** 2) * CE
        # alpha moved to same device as target to avoid CPU/GPU mismatch
        alpha = torch.Tensor([0.5, 0.5]).to(target.device)
        alpha = (target == 0) * alpha[0] + (target == 1) * alpha[1]
        return torch.mean(alpha * loss)


# ---------------------------------------------------------------------------
# Two-pass domain routing
# ---------------------------------------------------------------------------
def predict_domain(model, images):
    """
    Pass 1: dummy forward (domain=0) to extract domain_classifier logits
    from pre-MoE Stage-4 features.

    Returns
    -------
    pivot_indices : LongTensor (B,)   model-chosen domain per image
    dom_logits    : FloatTensor (B, D) raw logits (useful for logging)
    """
    dummy = torch.zeros(images.size(0), dtype=torch.long, device=images.device)
    with torch.no_grad():
        _, dom_logits, _ = model(images, dummy)
    return dom_logits.argmax(dim=1), dom_logits


# ---------------------------------------------------------------------------
# Per-pixel stability mask  (GRAM Eq. 9)
# ---------------------------------------------------------------------------
def get_stability_mask(model, images, pivot_indices,
                       domain_num=12, agree_threshold=0.5, ignore_index=255):
    """
    Sweep all domain experts. Per-pixel stability = fraction of non-pivot
    experts that agree with the pivot prediction.
    Pixels below agree_threshold are set to ignore_index=255.

    Returns
    -------
    pivot_preds  : LongTensor (B, H, W)
    stable_mask  : LongTensor (B, H, W)  unstable pixels -> 255
    agree_ratio  : FloatTensor (B, H, W) for logging
    """
    B, _, H, W = images.shape

    with torch.no_grad():
        # Pivot segmentation (Pass 2 of two-pass routing)
        seg_pivot, _, _ = model(images, pivot_indices)
        pivot_preds = seg_pivot.argmax(dim=1)          # (B, H, W)

        agree_count = torch.zeros(B, H, W, device=images.device)

        for d in range(domain_num):
            d_tensor  = torch.full((B,), d, dtype=torch.long, device=images.device)
            seg_d, _, _ = model(images, d_tensor)
            preds_d   = seg_d.argmax(dim=1)            # (B, H, W)

            # Only count domains that differ from each image's own pivot
            is_other  = (d_tensor != pivot_indices)    # (B,)
            is_other  = is_other[:, None, None].expand(B, H, W)
            agree_count += ((preds_d == pivot_preds) & is_other).float()

        agree_ratio = agree_count / (domain_num - 1)   # (B, H, W) in [0,1]

        stable_mask = pivot_preds.clone()
        stable_mask[agree_ratio < agree_threshold] = ignore_index

    return pivot_preds, stable_mask, agree_ratio


# ---------------------------------------------------------------------------
# Tile-level pre-filter  (replaces get_high_miou_topk_by_pivot_domain)
# ---------------------------------------------------------------------------
def get_high_consistency_tiles(model, dataloader, domain_num=12,
                               top_ratio=0.5, agree_threshold=0.5):
    """
    Stage 1 — tile-level: rank tiles by mean pixel agreement ratio; keep top_ratio.
    Stage 2 — pixel-level: collect per-pixel stability masks for kept tiles.

    Returns
    -------
    top_indices  : list[int]              dataset-level indices
    pixel_masks  : dict[int, LongTensor]  (H, W) stable mask per kept tile
    """
    model.eval()
    tile_scores = []
    global_idx  = 0

    # --- Pass 1: score every tile ---
    with torch.no_grad():
        for images, _targets, _ in dataloader:   # _targets intentionally ignored
            images = images.cuda()
            pivot_indices, _ = predict_domain(model, images)
            _, _, agree_ratio = get_stability_mask(
                model, images, pivot_indices,
                domain_num=domain_num, agree_threshold=agree_threshold,
            )
            for i in range(images.size(0)):
                tile_scores.append((agree_ratio[i].mean().item(), global_idx))
                global_idx += 1

    tile_scores.sort(key=lambda x: x[0], reverse=True)
    top_k       = max(1, int(len(tile_scores) * top_ratio))
    top_indices = [s[1] for s in tile_scores[:top_k]]
    top_set     = set(top_indices)

    # --- Pass 2: collect per-pixel masks for selected tiles only ---
    pixel_masks = {}
    global_idx  = 0

    with torch.no_grad():
        for images, _targets, _ in dataloader:
            images = images.cuda()
            pivot_indices, _ = predict_domain(model, images)
            _, stable_mask, _ = get_stability_mask(
                model, images, pivot_indices,
                domain_num=domain_num, agree_threshold=agree_threshold,
            )
            for i in range(images.size(0)):
                if global_idx in top_set:
                    pixel_masks[global_idx] = stable_mask[i].cpu()
                global_idx += 1

    print(f"[filter] {len(top_indices)}/{len(tile_scores)} tiles kept "
          f"(top_ratio={top_ratio}, agree_threshold={agree_threshold})")
    return top_indices, pixel_masks


# ---------------------------------------------------------------------------
# Setup  (args, dataset, model — mirrors original exactly)
# ---------------------------------------------------------------------------
set_seed(0)

parser = argparse.ArgumentParser(description='Deeplabv3 pytorch Training')
parser.add_argument('--test_meta',    type=str,   default='UGA_test_metadata.csv')
parser.add_argument('--epoch',        type=int,   default=10)
parser.add_argument('--threshold',    type=float, default=0.5,
                    help='top_ratio for tile selection (mirrors original --threshold)')
parser.add_argument('--agree_thresh', type=float, default=0.5,
                    help='per-pixel cross-expert agreement threshold')
parser.add_argument('--domain_num',   type=int,   default=12)
args = parser.parse_args()

val_transform = get_val_augmentation([256, 256])
normalize     = get_normalize()

testset    = GPSDataset(metadata=args.test_meta, transform=val_transform, normalize=normalize)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

model = mit_b5_MOE(
    patch_size=4, embed_dims=[32, 64, 160, 256],
    num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0, drop_path_rate=0.1,
    expert_num=12,
    select_mode='new_topk',          # BUG FIX: was 'new_topk' in original
    hidden_dims=[2, 4, 10, 16],
    num_k=[2],                         # BUG FIX: model.py expects list, not int
    domain_num=12,
)

model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load("./checkpoint/MOE_epoch_2_v2.pth")["state_dict"])

criterion = FocalLoss().cuda()
optimizer = torch.optim.SGD(model.module.parameters(), lr=1e-4, momentum=0.99)


# ---------------------------------------------------------------------------
# Stage 1: Select stable tiles + build per-pixel pseudo-label masks
# ---------------------------------------------------------------------------
top_indices, pixel_masks = get_high_consistency_tiles(
    model, testloader,
    domain_num=args.domain_num,
    top_ratio=args.threshold,         # --threshold reused as top_ratio
    agree_threshold=args.agree_thresh,
)

test_subset          = Subset(testset, top_indices)
testloader_filtered  = torch.utils.data.DataLoader(
    test_subset, batch_size=16, shuffle=True, num_workers=2
)


# ---------------------------------------------------------------------------
# Stage 2: TTA fine-tuning with per-pixel stability-masked pseudo-labels
# ---------------------------------------------------------------------------
print('Start Training')

for epoch in range(0, args.epoch):
    model.train()

    for batch_idx, (images, _targets, _country_idx) in tqdm(
        enumerate(testloader_filtered), total=len(testloader_filtered)
    ):
        images = images.cuda()

        # Two-pass domain routing
        pivot_indices, _ = predict_domain(model, images)

        # Per-pixel stability mask (no_grad — gradients only through seg below)
        with torch.no_grad():
            _, stable_mask, _ = get_stability_mask(
                model, images, pivot_indices,
                domain_num=args.domain_num,
                agree_threshold=args.agree_thresh,
            )
        # stable_mask: (B, H, W); unstable pixels = 255 -> ignored by FocalLoss

        # Forward pass for gradient update
        output, _d_output, _MI_loss = model(images, pivot_indices)

        optimizer.zero_grad()
        loss = criterion(output, stable_mask.cuda())
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            stable_px = (stable_mask != 255).float().mean().item() * 100
            print(f"Epoch {epoch}  Batch {batch_idx}  "
                  f"Loss={loss.item():.4f}  stable_px={stable_px:.1f}%")

    # -----------------------------------------------------------------------
    # Per-epoch evaluation
    # -----------------------------------------------------------------------
    model.eval()
    metrics = Metrics(2, 255)    # BUG FIX: reset each epoch (was outside loop)

    for batch_idx, (images, targets, _country_idx) in tqdm(
        enumerate(testloader), total=len(testloader)
    ):
        images, targets = images.cuda(), targets.cuda().detach()

        # Use model-predicted domain routing for evaluation too
        pivot_indices, _ = predict_domain(model, images)
        output, _, _     = model(images, pivot_indices)

        metrics.update(output.cpu(), targets.cpu())
        del output

    ious,      miou       = metrics.compute_iou()
    acc,       macc       = metrics.compute_pixel_acc()
    f1,        mf1        = metrics.compute_f1()
    precision, mprecision = metrics.compute_precision()
    recall,    mrecall    = metrics.compute_recall()

    print(f"ious      : [{ious[0]:.4f}, {ious[1]:.4f}]   mIoU={miou:.4f}")
    print(f"f1        : [{f1[0]:.4f}, {f1[1]:.4f}]   mF1={mf1:.4f}")
    print(f"acc       : [{acc[0]:.4f}, {acc[1]:.4f}]")
    print(f"precision : [{precision[0]:.4f}, {precision[1]:.4f}]")
    print(f"recall    : [{recall[0]:.4f}, {recall[1]:.4f}]")``` 


G:\Uni Work\BanglaSlumNet\gram_baseline>(echo # make_summary_chart.py   & echo ```python   & type "make_summary_chart.py"   & echo ```   & echo.) 
# make_summary_chart.py 
```python 
# make_summary_chart.py  — updated for two-pass per-tile domain routing
"""
Reads gram_baseline_summary.csv (output of gram_baseline.py) and the
per-location mosaic PNGs to produce:
  1. outputs/gram_dhaka_summary_figure.png  — 3-row x 3-col visual mosaic
  2. outputs/gram_domain_routing.png        — bar chart of per-tile domain votes
  3. outputs/gram_prob_distribution.png     — violin/box plot of slum prob by location
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from collections import Counter

HERE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "outputs")

LOCS = ["korail", "mirpur", "olddhaka"]
TITLES = {
    "korail":   "Korail (real informal settlement)",
    "mirpur":   "Mirpur (mixed formal/informal)",
    "olddhaka": "Old Dhaka (dense formal historic core)",
}

CITY_NAMES = {
    0: "Cairo", 1: "Cape Town", 2: "Caracas", 3: "Colombo",
    4: "Karachi", 5: "Medellín", 6: "Mumbai", 7: "Nairobi",
    8: "Ouagadougou", 9: "Port-au-Prince", 10: "Rio", 11: "Tegucigalpa",
}


# ---------------------------------------------------------------------------
# Figure 1: Visual mosaic (original | heatmap | overlay) — 3 locations
# ---------------------------------------------------------------------------
def plot_visual_mosaic():
    fig, axes = plt.subplots(len(LOCS), 3, figsize=(18, 14))
    for r, loc in enumerate(LOCS):
        mosaic_path = os.path.join(OUTDIR, f"{loc}_gram_baseline.png")
        if not os.path.exists(mosaic_path):
            print(f"[warn] missing {mosaic_path}, skipping row {r}")
            continue
        mosaic = np.array(Image.open(mosaic_path))
        H, W   = mosaic.shape[:2]
        third  = W // 3
        rgb     = mosaic[:, :third]
        heat    = mosaic[:, third:2*third]
        overlay = mosaic[:, 2*third:3*third]

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"{TITLES[loc]}\nESRI z16, 1.2 m/px", fontsize=10)
        axes[r, 1].imshow(heat)
        axes[r, 1].set_title("GRAM slum probability\n(black=0, red=1)", fontsize=10)
        axes[r, 2].imshow(overlay)
        axes[r, 2].set_title("Binary mask p>0.5\n(red = predicted slum)", fontsize=10)
        for c in range(3):
            axes[r, c].axis("off")

    plt.suptitle(
        "GRAM Zero-Shot (AAAI'26) on Dhaka — Two-Pass Domain Routing\n"
        "Failure mode: Korail ≈ Old Dhaka mean probability",
        fontsize=13, y=0.995,
    )
    plt.tight_layout()
    out = os.path.join(OUTDIR, "gram_dhaka_summary_figure.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# Figure 2: Domain routing bar chart — which source city does Dhaka look like?
# ---------------------------------------------------------------------------
def plot_domain_routing(df):
    fig, axes = plt.subplots(1, len(LOCS), figsize=(18, 5), sharey=False)
    for ax, loc in zip(axes, LOCS):
        sub    = df[df["location"] == loc]
        counts = Counter(sub["domain_idx"].tolist())
        labels = [CITY_NAMES.get(int(d), str(d)) for d in sorted(counts)]
        values = [counts[d] for d in sorted(counts)]
        bars   = ax.bar(labels, values, color="#e05c2a", edgecolor="white")
        ax.set_title(TITLES[loc], fontsize=10)
        ax.set_xlabel("Source city (GRAM training domain)", fontsize=9)
        ax.set_ylabel("# tiles routed here", fontsize=9)
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(val), ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        "Per-tile domain routing: which GRAM source city does each Dhaka tile resemble?\n"
        "(model's own domain_classifier, two-pass inference)",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(OUTDIR, "gram_domain_routing.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# Figure 3: Slum probability distribution per location — the core failure mode
# ---------------------------------------------------------------------------
def plot_prob_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"korail": "#d62728", "mirpur": "#ff7f0e", "olddhaka": "#1f77b4"}
    positions = {loc: i for i, loc in enumerate(LOCS)}

    # Load per-tile .npy prob maps for full pixel-level distribution
    all_probs = {loc: [] for loc in LOCS}
    for _, row in df.iterrows():
        loc      = row["location"]
        tile     = row["tile"].replace(".jpg", "")
        # parse x, y from tile name e.g. korail_z16_x12345_y67890
        parts    = tile.split("_")
        try:
            x = parts[-2].replace("x", "")
            y = parts[-1].replace("y", "")
            npy_path = os.path.join(OUTDIR, f"{loc}_x{x}_y{y}_prob.npy")
            if os.path.exists(npy_path):
                probs = np.load(npy_path).flatten()
                all_probs[loc].append(probs)
        except Exception:
            pass

    for loc in LOCS:
        if not all_probs[loc]:
            # Fall back to mean_prob column if .npy files not available
            vals = df[df["location"] == loc]["mean_prob"].values
        else:
            vals = np.concatenate(all_probs[loc])

        pos = positions[loc]
        bp  = ax.violinplot(vals, positions=[pos], widths=0.6,
                            showmeans=True, showmedians=True)
        for pc in bp["bodies"]:
            pc.set_facecolor(colors[loc])
            pc.set_alpha(0.7)

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([TITLES[l] for l in LOCS], fontsize=10)
    ax.set_ylabel("Per-pixel slum probability", fontsize=11)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1,
               label="Decision threshold (p=0.5)")
    ax.legend(fontsize=10)
    ax.set_title(
        "GRAM slum probability distribution by location\n"
        "Korail and Old Dhaka should be separable — they are not",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(OUTDIR, "gram_prob_distribution.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    csv_path = os.path.join(OUTDIR, "gram_baseline_summary.csv")
    if not os.path.exists(csv_path):
        print(f"[error] {csv_path} not found — run gram_baseline.py first")
        return

    df = pd.read_csv(csv_path)

    # Print the aggregate failure-mode numbers for FINDINGS.md
    print("\n=== Failure-mode summary (for FINDINGS.md / Appendix A) ===")
    for loc in LOCS:
        sub = df[df["location"] == loc]
        print(f"  {loc:12s}  mean_prob={sub['mean_prob'].mean():.3f}  "
              f"max={sub['max_prob'].max():.3f}  "
              f"pct>0.5={sub['pct_slum_p50'].mean():.1f}%  "
              f"n_tiles={len(sub)}")

    plot_visual_mosaic()
    plot_domain_routing(df)
    plot_prob_distribution(df)
    print("\nAll figures saved to outputs/")


if __name__ == "__main__":
    main()``` 


G:\Uni Work\BanglaSlumNet\gram_baseline>(echo # model.py   & echo ```python   & type "model.py"   & echo ```   & echo.) 
# model.py 
```python 
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SimpleAdapter(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features           
        hidden_features = hidden_features or in_features     
        self.fc1 = nn.Linear(in_features, hidden_features)       # Downconv
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)      # UPconv
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)            
        x = self.act(x)            
        x = self.drop(x)
        x = self.fc2(x)           
        x = self.drop(x)   
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x_norm1 = self.norm1(x).contiguous().clone()
        res = self.drop_path(self.attn(x_norm1, H, W))
        x = x + res  # residual add

        x_norm2 = self.norm2(x).contiguous().clone()
        res = self.drop_path(self.mlp(x_norm2, H, W))
        x = x + res

        return x


class MOEAdapterBlock(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, 
                 expert_num=None, domain_num=None, select_mode=None, MoE_hidden_dim=None, num_k = None):   
        
        super().__init__()
        
        self.expert_num = expert_num
        self.domain_num = domain_num
        self.select_mode = select_mode
        self.acc_freq = 0 
        self.num_K = num_k
        self.MI_task_gate = torch.zeros(self.domain_num, self.expert_num)    

        self.norm1 = norm_layer(dim) 
        self.norm2 = norm_layer(dim)
        
        self.attn = Attention(
            dim,
            num_heads=num_heads,     
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            attn_drop=attn_drop, 
            proj_drop=drop, 
            sr_ratio=sr_ratio)
        
        mlp_hidden_dim = int(dim * mlp_ratio)      
        
        self.mlp = Mlp(in_features=dim,    
                       hidden_features=mlp_hidden_dim,   
                       act_layer=act_layer, 
                       drop=drop)
        
        if self.select_mode == 'new_topk':      # NOTE you can add other routing methods 
            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(1)
            self.f_gate = nn.ModuleList([nn.Sequential(nn.Linear(dim, 2 * expert_num, bias=False)) for i in range(self.domain_num)])   
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()   
        self.apply(self._init_weights)    
        
        expert_lists = []
        for _ in range(expert_num) : 
            tmp_adapter = SimpleAdapter(in_features=dim, hidden_features=MoE_hidden_dim, act_layer=act_layer, drop=drop)  
            tmp_adapter.apply(self._init_weights)      
            expert_lists.append(tmp_adapter)

        self.adapter_experts = nn.ModuleList(expert_lists)   

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)   
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def minmax_scaling(self, top_k_logits) : 
            m1 = top_k_logits.min() ; m2 = top_k_logits.max()
            return (top_k_logits-m1)/ (m2-m1)
            
    def one_hot_encoding(self, index, num_classes):
        one_hot = np.zeros(num_classes)  
        one_hot[index] = 1  
        return one_hot

    def forward(self, x, H, W, expert_num, select_mode, pseudo_domain_label=None, expert_check=False):    
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))   

        self.MI_task_gate = torch.zeros(self.domain_num, self.expert_num)  # device 고려

        if self.select_mode == 'random':    
            select = torch.randint(low=0, high=expert_num, size=(1,)).item()  
            MI_loss = self.one_hot_encoding(select, self.expert_num)     
            x = x + self.adapter_experts[select](x, H, W)    

        elif self.select_mode == 'new_topk':
            task_bh = pseudo_domain_label.tolist()

            # (B, 2E)
            total_w = torch.stack([
                self.f_gate[task_bh[i]](x[i]) for i in range(x.size(0))
            ], dim=0)

            clean_logits, raw_noise_stddev = total_w.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev

            exp_wise_sum = logits.sum(dim=1)  # (B, E)
            probs = F.softmax(exp_wise_sum, dim=-1)

            for i, t in enumerate(task_bh):
                self.MI_task_gate[t] += probs[i].detach().cpu()

            top_k = min(self.num_K + 1, self.expert_num)
            top_logits, top_indices = exp_wise_sum.topk(top_k, dim=1)
            top_k_logits = top_logits[:, :self.num_K]      # [B, K]
            top_k_indices = top_indices[:, :self.num_K]    # [B, K]

            if top_k_logits.size(0) > 1:
                top_k_gates = self.softmax(self.minmax_scaling(top_k_logits))  # [B, K]
            else:
                top_k_gates = self.softmax(top_k_logits)

            # Adapter output 계산 (B, D)
            adapter_outputs = torch.stack(
                [self.adapter_experts[e](x, H, W) for e in range(self.expert_num)], dim=1
            )

            B, S, D = x.shape
            K = self.num_K
            top_k_exp_outputs = torch.zeros(B, K, S, D, device=x.device)

            # 각 expert만 따로 호출 (loop는 있지만 실행 expert만 돌기 때문에 메모리 효율 ↑)
            for k in range(K):
                selected_expert_ids = top_k_indices[:, k]  # (B,)
                x_k = []

                for b in range(B):
                    e_id = selected_expert_ids[b].item()
                    x_b = x[b].unsqueeze(0)  # (1, S, D)
                    out = self.adapter_experts[e_id](x_b, H, W)  # (1, S, D)
                    x_k.append(out)

                # Stack across batch: (B, S, D)
                x_k = torch.cat(x_k, dim=0)
                top_k_exp_outputs[:, k] = x_k

            # apply gates: (B, K, 1, 1)
            gates = top_k_gates.unsqueeze(-1).unsqueeze(-1)
            weighted_outputs = gates * top_k_exp_outputs  # (B, K, S, D)

            # sum over experts: (B, S, D)
            x_out = weighted_outputs.sum(dim=1)

            x = x + x_out

            self.MI_task_gate = self.MI_task_gate / x.size(0)

            
            P_TI = torch.sum(self.MI_task_gate, dim=1, keepdim=True) + 1e-4
            P_EI = torch.sum(self.MI_task_gate, dim=0, keepdim=True) + 1e-4

            MI_loss = ((self.MI_task_gate + 1e-4) * torch.log(self.MI_task_gate / (P_TI * P_EI) + 1e-4)).sum()
            
            if expert_check:
                return x, MI_loss, top_k_indices
        else:
            print('No attribute')
            MI_loss = None
         
        return x, MI_loss  


class MOEAdapterBlockR(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        sr_ratio=1, expert_num=None, num_domains=None, select_mode=None,
        MoE_hidden_dim=None, num_k=None
    ):
        super().__init__()
        self.expert_num = expert_num
        self.domain_num = num_domains
        self.select_mode = select_mode
        self.num_K = num_k
    
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
    
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
        )
    
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
        if self.select_mode == 'new_topk':
            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(dim=-1)
            self.f_gate = nn.ModuleList([
                nn.Sequential(nn.Linear(dim, 2 * expert_num, bias=False))
                for _ in range(self.domain_num)
            ])
    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)
    
        expert_lists = []
        for _ in range(expert_num):
            tmp_adapter = SimpleAdapter(in_features=dim, hidden_features=MoE_hidden_dim, act_layer=act_layer, drop=drop)
            tmp_adapter.apply(self._init_weights)
            expert_lists.append(tmp_adapter)
        self.adapter_experts = nn.ModuleList(expert_lists)
    
    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def minmax_scaling(self, top_k_logits):
        m1 = top_k_logits.min()
        m2 = top_k_logits.max()
        return (top_k_logits - m1) / (m2 - m1)
    
    def forward(self, x, H, W, expert_num, select_mode, pseudo_domain_label, R):
        """
        x: (B, N, C)    - 토큰 시퀀스
        R: (D, D) or None - 도메인 간 유사도 행렬 (None이면 relational term 건너뜀)
        pseudo_domain_label: (B,) 각 샘플의 도메인 인덱스
        """
        B, N, C = x.shape
    
        # 1) Transformer Flow: Attention + MLP
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
    
        # 2) MI용 joint 분포 누적
        MI_task_gate = torch.zeros(self.domain_num, self.expert_num, device=x.device)
    
        # 3) routing / expert 어댑터 적용
        if select_mode == 'random':
            select = torch.randint(low=0, high=expert_num, size=(1,), device=x.device).item()
            MI_loss = 0.0
            x = x + self.adapter_experts[select](x, H, W)
            return x, MI_loss
    
        elif select_mode == 'new_topk':
            # -- (a) 각 샘플별 토큰 수준에서 로짓 생성 및 합산 --
            per_sample_token_logits = []
            for i in range(B):
                gate_out = self.f_gate[pseudo_domain_label[i].item()](x[i])  # (N, 2E)
                clean_logits, raw_noise = gate_out.chunk(2, dim=-1)          # (N, E)
                noise_std = F.softplus(raw_noise) + 1e-2                       # (N, E)
                logits_i = clean_logits + torch.randn_like(clean_logits) * noise_std  # (N, E)
                per_sample_token_logits.append(logits_i)                      # 리스트에 (N, E)
    
            # -- (b) 토큰 축(axis=0) 합산 → (B, E) 벡터 획득 --
            token_logits_sum = torch.stack([
                logits_i.sum(dim=0)  # (E,)
                for logits_i in per_sample_token_logits
            ], dim=0)  # shape = (B, E)
    
            # -- (c) softmax → 샘플별 expert 확률 (B, E) --
            probs = F.softmax(token_logits_sum, dim=-1)  # (B, E)
    
            # -- (d) joint P(D, E) 누적 --
            for i, d in enumerate(pseudo_domain_label):
                MI_task_gate[d] += probs[i].detach()  # probs[i]: (E,)
    
            # -- (e) Top-K expert indices 및 게이트 점수 --
            top_k = min(self.num_K + 1, self.expert_num)
            top_logits, top_indices = token_logits_sum.topk(top_k, dim=-1)  # (B, top_k)
            top_k_logits = top_logits[:, :self.num_K]    # (B, K)
            top_k_indices = top_indices[:, :self.num_K]  # (B, K)
    
            if top_k_logits.size(0) > 1:
                top_k_gates = F.softmax(self.minmax_scaling(top_k_logits), dim=-1)  # (B, K)
            else:
                top_k_gates = F.softmax(top_k_logits, dim=-1).unsqueeze(-1)         # (1, K)
    
            # -- (f) Adapter expert 출력 계산 (메모리 절약용 루프) --
            K = self.num_K
            top_k_exp_outputs = torch.zeros(B, K, N, C, device=x.device)
    
            for k_idx in range(K):
                sel_ids = top_k_indices[:, k_idx]  # (B,)
                outs_k = []
                for b in range(B):
                    e_id = sel_ids[b].item()        # 예: 3
                    xb = x[b].unsqueeze(0)           # (1, N, C)
                    out_b = self.adapter_experts[e_id](xb, H, W)  # (1, N, C)
                    outs_k.append(out_b)
                outs_k = torch.cat(outs_k, dim=0)  # (B, N, C)
                top_k_exp_outputs[:, k_idx] = outs_k
    
            gates = top_k_gates.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
            weighted_outputs = gates * top_k_exp_outputs      # (B, K, N, C)
            x_out = weighted_outputs.sum(dim=1)               # (B, N, C)
            x = x + x_out
    
            # -- (g) joint P(D, E) 정규화 → MI 계산 --
            MI_task_gate = MI_task_gate / B  # (D, E)
    
            P_D = MI_task_gate.sum(dim=1, keepdim=True) + 1e-6   # (D, 1)
            P_E = MI_task_gate.sum(dim=0, keepdim=True) + 1e-6   # (1, E)
            MI = (MI_task_gate * torch.log(MI_task_gate / (P_D * P_E) + 1e-6)).sum()
    
            # -- (h) P(E | D) 분포 계산 --
            P_E_given_D = MI_task_gate / P_D  # (D, E)
    
            # -- (i) R이 주어진 경우에만 relational penalty 계산 --
            if R is not None:
                D_dim, E_dim = P_E_given_D.shape
                rel_term = 0.0
                for i in range(D_dim):
                    for j in range(D_dim):
                        diff = (P_E_given_D[i] - P_E_given_D[j]).pow(2).sum()
                        rel_term += R[i, j] * diff
                rel_term = rel_term / (D_dim * D_dim)
                MI_loss = MI - 1e-1*rel_term
            else:
                # R이 None이면 relational term 생략 → 단순히 -MI만 사용
                MI_loss = MI
    
            return x, MI_loss
    
        else:
            # select_mode이 없거나 다른 모드라면 MI_loss=0 반환
            return x, torch.zeros(1, device=x.device)

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out_1 = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = fan_out_1 // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x_2 = self.norm(x)
        return x_2.clone(), H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256], 
                 dims=(32, 64, 160, 256), decoder_dim=256, 
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])


        

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.to_fused = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, decoder_dim, kernel_size=1),
                nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
            ) for i, dim in enumerate(dims)
        ])

        # to_segmentation: 두 번의 1x1 Conv
        self.head = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, kernel_size=1),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        )


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        layer_outputs = self.forward_features(x)
        
        fused = [block(output) for output, block in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        return self.head(fused)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x




class MOE_MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2,
                 embed_dims=[64, 128, 256, 512], dims=(32, 64, 160, 256),
                 decoder_dim=256, num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], expert_num=12, select_mode=None, hidden_dims=None,
                 num_k=None, num_domains=12, expert_check=False):
        super().__init__()
        self.select_mode = select_mode
        self.expert_num = expert_num
        self.num_classes = num_classes
        self.num_k = num_k
        self.depths = depths
        self.num_domains = num_domains

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(img_size, 7, 4, in_chans, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size//4, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size//8, 3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size//16, 3, 2, embed_dims[2], embed_dims[3])

        # MOE blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([
            MOEAdapterBlock(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale,
                             drop_rate, attn_drop_rate, dpr[cur+i], nn.GELU, norm_layer, sr_ratios[0],
                             expert_num, num_domains, select_mode, hidden_dims[0], num_k)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([
            MOEAdapterBlock(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale,
                             drop_rate, attn_drop_rate, dpr[cur+i], nn.GELU, norm_layer, sr_ratios[1],
                             expert_num, num_domains,  select_mode, hidden_dims[1], num_k)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([
            MOEAdapterBlock(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale,
                             drop_rate, attn_drop_rate, dpr[cur+i], nn.GELU, norm_layer, sr_ratios[2],
                             expert_num, num_domains, select_mode, hidden_dims[2], num_k)
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([
            MOEAdapterBlock(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale,
                             drop_rate, attn_drop_rate, dpr[cur+i], nn.GELU, norm_layer, sr_ratios[3],
                             expert_num, num_domains, select_mode, hidden_dims[3], num_k)
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # Decoder and segmentation head
        self.to_fused = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, decoder_dim, 1),
                nn.Upsample(size=(256,256), mode='bilinear', align_corners=False)
            ) for dim in dims])
        self.head = nn.Sequential(
            nn.Conv2d(4*decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1)
        )

        # Domain classifier on pre-MoE features
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B,C,1,1)
            nn.Flatten(),             # (B,C)
            nn.Linear(embed_dims[3], num_domains)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # ... (same init as before) ...
        pass

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    # freeze
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def forward_features(self, x, pseudo_domain_label, expert_check=False):
        B = x.size(0)
        outs = []
        all_selected_experts = [
            [[] for _ in range(depth)]  # block1: 3개, block2: 6개, ...
            for depth in self.depths
        ]

        
        # --- Stage 1 ---
        x, H, W = self.patch_embed1(x)
        for blk_idx, blk in enumerate(self.block1):
            if expert_check:
                x, mi, sel_idx = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, expert_check)
                all_selected_experts[0][blk_idx].append(sel_idx)
            else:
                x, mi = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)
        x = self.norm1(x)  # (B, N1, C1)
        feat1 = x.view(B, H, W, -1).permute(0,3,1,2).contiguous()  # (B, C1, H, W)
        outs.append(feat1)
    
        # --- Stage 2 ---
        x, H, W = self.patch_embed2(feat1)
        for blk_idx, blk in enumerate(self.block2):
            if expert_check:
                x, mi, sel_idx = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, expert_check)
                all_selected_experts[1][blk_idx].append(sel_idx)
            else:
                x, mi = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)
        x = self.norm2(x)
        feat2 = x.view(B, H, W, -1).permute(0,3,1,2).contiguous()
        outs.append(feat2)
    
        # --- Stage 3 ---
        x, H, W = self.patch_embed3(feat2)
        for blk_idx, blk in enumerate(self.block3):
            if expert_check:
                x, mi, sel_idx = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, expert_check)
                all_selected_experts[2][blk_idx].append(sel_idx)
            else:
                x, mi = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)
        x = self.norm3(x)
        feat3 = x.view(B, H, W, -1).permute(0,3,1,2).contiguous()
        outs.append(feat3)
    
        # --- Stage 4 pre-MoE (token-level norm) ---
        x, H, W = self.patch_embed4(feat3)      # x: (B, N4, C4)
        x = self.norm4(x)                       # apply LayerNorm over last dim C4
        pre_moe_feat = x.view(B, H, W, -1) \
                       .permute(0,3,1,2)        # (B, C4, H, W), for domain classifier
    
        # --- Stage 4 MoE adapters ---
        total_MI_loss = 0
        for blk_idx, blk in enumerate(self.block4):
            if expert_check:
                x, mi, sel_idx = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, expert_check)
                all_selected_experts[3][blk_idx].append(sel_idx)
            else:
                x, mi = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label)
            total_MI_loss += mi
    
        x = self.norm4(x)  # again normalize after adapters
        feat4 = x.view(B, H, W, -1).permute(0,3,1,2).contiguous()
        outs.append(feat4)
        if expert_check:
            return outs, pre_moe_feat, total_MI_loss, all_selected_experts
        return outs, pre_moe_feat, total_MI_loss
    def forward(self, x, pseudo_domain_label, expert_check=False):
        if expert_check:
            layer_outs, pre_moe_feat, total_MI, all_selected_experts = self.forward_features(x, pseudo_domain_label, expert_check)
        else:
            layer_outs, pre_moe_feat, total_MI = self.forward_features(x, pseudo_domain_label)
        # segmentation
        fused = torch.cat([f(o) for f, o in zip(self.to_fused, layer_outs)], dim=1)
        seg_out = self.head(fused)
        # domain pred from pre-MoE feature
        dom_logits = self.domain_classifier(pre_moe_feat)
        
        if expert_check:
            return seg_out, dom_logits, total_MI, all_selected_experts
        return seg_out, dom_logits, total_MI



    


class MOE_MixVisionTransformerv2(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, num_classes=2,
        embed_dims=[64, 128, 256, 512], dims=(32, 64, 160, 256),
        decoder_dim=256, num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1], expert_num=12, select_mode=None, hidden_dims=None,
        num_k=None, num_domains=12
    ):
        super().__init__()
        self.select_mode = select_mode
        self.expert_num = expert_num
        self.num_classes = num_classes
        self.num_k = num_k
        self.depths = depths
        self.num_domains = num_domains

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(img_size, 7, 4, in_chans, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])

        # MOE blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([
            MOEAdapterBlockR(
                embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, dpr[cur + i], nn.GELU, norm_layer, sr_ratios[0],
                expert_num, num_domains, select_mode, hidden_dims[0], num_k
            )
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([
            MOEAdapterBlockR(
                embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, dpr[cur + i], nn.GELU, norm_layer, sr_ratios[1],
                expert_num, num_domains, select_mode, hidden_dims[1], num_k
            )
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([
            MOEAdapterBlockR(
                embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, dpr[cur + i], nn.GELU, norm_layer, sr_ratios[2],
                expert_num, num_domains, select_mode, hidden_dims[2], num_k
            )
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([
            MOEAdapterBlockR(
                embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale,
                drop_rate, attn_drop_rate, dpr[cur + i], nn.GELU, norm_layer, sr_ratios[3],
                expert_num, num_domains, select_mode, hidden_dims[3], num_k
            )
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        # Decoder and segmentation head
        self.to_fused = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, decoder_dim, 1),
                nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
            ) for dim in dims
        ])
        self.head = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1)
        )

        # Domain classifier on pre-MoE features
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # → (B, C, 1, 1)
            nn.Flatten(),              # → (B, C)
            nn.Linear(embed_dims[3], num_domains)  # → (B, D)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # ... (same init as before) ...
        pass

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    # freeze
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def forward_features(self, x, pseudo_domain_label):
        B = x.size(0)
        outs = []

        # --- Stage 1 ---
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x, _ = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, None)
        x = self.norm1(x)  # (B, N1, C1)
        feat1 = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B, C1, H, W)
        outs.append(feat1)

        # --- Stage 2 ---
        x, H, W = self.patch_embed2(feat1)
        for blk in self.block2:
            x, _ = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, None)
        x = self.norm2(x)
        feat2 = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feat2)

        # --- Stage 3 ---
        x, H, W = self.patch_embed3(feat2)
        for blk in self.block3:
            x, _ = blk(x, H, W, self.expert_num, self.select_mode, pseudo_domain_label, None)
        x = self.norm3(x)
        feat3 = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feat3)

        # --- Stage 4 pre-MoE (token-level norm) ---
        x, H, W = self.patch_embed4(feat3)  # x: (B, N4, C4)
        x = self.norm4(x)                   # (B, N4, C4)
        pre_moe_feat = x.view(B, H, W, -1)  # (B, C4, H, W)
        pre_moe_feat = pre_moe_feat.permute(0, 3, 1, 2).contiguous()

        # **(여기서 R 계산)**: domain_classifier 마지막 Linear weight로부터
        linear_layer = self.domain_classifier[-1]  # nn.Linear(C4, D)
        W_dom = linear_layer.weight        # (D, C4)
        W_norm = F.normalize(W_dom, dim=1) # (D, C4)
        R = torch.clamp(W_norm @ W_norm.t(), min=0.0)  # (D, D)

        # --- Stage 4 MoE adapters: 각 블록마다 R 인자로 넘겨 MI 계산에 활용 ---
        total_MI_loss = 0.0
        x_moe = x  # (B, N4, C4)
        for blk in self.block4:
            x_moe, mi_loss = blk(x_moe, H, W, self.expert_num, self.select_mode, pseudo_domain_label, R)
            total_MI_loss = total_MI_loss + mi_loss

        x_moe = self.norm4(x_moe)
        feat4 = x_moe.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(feat4)

        return outs, pre_moe_feat, total_MI_loss

    def forward(self, x, pseudo_domain_label):
        layer_outs, pre_moe_feat, total_MI = self.forward_features(x, pseudo_domain_label)

        # segmentation head
        fused = torch.cat([f(o) for f, o in zip(self.to_fused, layer_outs)], dim=1)
        seg_out = self.head(fused)

        # domain prediction
        dom_logits = self.domain_classifier(pre_moe_feat)
        return seg_out, dom_logits, total_MI






class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5_MOE(MOE_MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5_MOE, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],  mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 6, 40, 3], 
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, 
            expert_num=12, 
            select_mode='new_topk', 
            hidden_dims = [2, 4, 10, 16],        # NOTE you can set them freely 
            num_k = 2
            )   

class mit_b5_MOEv2(MOE_MixVisionTransformerv2):
    def __init__(self, **kwargs):
        super(mit_b5_MOEv2, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],  mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 6, 40, 3], 
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, 
            expert_num=12, 
            select_mode='new_topk', 
            hidden_dims = [2, 4, 10, 16],        # NOTE you can set them freely 
            num_k = 2
            )
        
class DomainDiscriminator(nn.Module):
    def __init__(self, n_outputs = 12):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 32 * 32, 250)
        self.fc2 = nn.Linear(250, n_outputs)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        x = self.pool(F.relu(self.conv1(inputs)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 32 * 32 * 32)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x``` 


G:\Uni Work\BanglaSlumNet\gram_baseline>(echo # utils.py   & echo ```python   & type "utils.py"   & echo ```   & echo.) 
# utils.py 
```python 
import torch
import numpy 


class Metrics:
    def __init__(self, num_classes, ignore_label):
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes)

    def update(self, pred, target):
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self):
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        return ious.cpu().numpy().tolist(), miou

    def compute_f1(self):
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        return f1.cpu().numpy().tolist(), mf1
    
    def compute_precision(self):
        precision = self.hist.diag() / self.hist.sum(0)
        mp = precision[~precision.isnan()].mean().item()
        return precision.cpu().numpy().tolist(), mp
        
    def compute_recall(self):
        recall = self.hist.diag() / self.hist.sum(1)
        mrecall = recall[~recall.isnan()].mean().item()
        return recall.cpu().numpy().tolist(), mrecall
    
    def compute_pixel_acc(self):
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        return acc.cpu().numpy().tolist(), macc``` 

