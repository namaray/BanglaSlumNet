"""
Geometric and photometric augmentation for training tiles.
Augmentations apply identically to RGB, label, hc_mask, and socioeconomic tensor
to maintain pixel alignment.
"""

import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            sample["rgb"] = torch.flip(sample["rgb"], dims=[-1])
            sample["label"] = torch.flip(sample["label"], dims=[-1])
            sample["hc_mask"] = torch.flip(sample["hc_mask"], dims=[-1])
            sample["socioec"] = torch.flip(sample["socioec"], dims=[-1])
        return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            sample["rgb"] = torch.flip(sample["rgb"], dims=[-2])
            sample["label"] = torch.flip(sample["label"], dims=[-2])
            sample["hc_mask"] = torch.flip(sample["hc_mask"], dims=[-2])
            sample["socioec"] = torch.flip(sample["socioec"], dims=[-2])
        return sample


class RandomRotate90:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            sample["rgb"] = torch.rot90(sample["rgb"], k=k, dims=[-2, -1])
            sample["label"] = torch.rot90(sample["label"].unsqueeze(0), k=k, dims=[-2, -1]).squeeze(0)
            sample["hc_mask"] = torch.rot90(sample["hc_mask"].unsqueeze(0), k=k, dims=[-2, -1]).squeeze(0)
            sample["socioec"] = torch.rot90(sample["socioec"], k=k, dims=[-2, -1])
        return sample


class ColorJitter:
    """Photometric jitter applied only to RGB channels."""
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.p:
            rgb = sample["rgb"]
            # Brightness
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            rgb = rgb * factor
            # Contrast
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = rgb.mean(dim=[-2, -1], keepdim=True)
            rgb = (rgb - mean) * factor + mean
            sample["rgb"] = rgb.clamp(0, 1)
        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def get_train_transform() -> Compose:
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
    ])


def get_eval_transform() -> Compose:
    return Compose([])  # identity


# ── Smoke test ─────────────────────────────────────────────────────────────────
def _smoke_test():
    sample = {
        "rgb": torch.rand(4, 256, 256),
        "label": torch.randint(0, 3, (256, 256)),
        "hc_mask": torch.randint(0, 2, (256, 256)).bool(),
        "socioec": torch.rand(6, 256, 256),
        "tile_id": "test_0001",
        "region": "korail",
        "split": "train",
    }
    t = get_train_transform()
    out = t(sample)
    assert out["rgb"].shape == (4, 256, 256)
    assert out["label"].shape == (256, 256)
    assert out["hc_mask"].shape == (256, 256)
    assert out["socioec"].shape == (6, 256, 256)
    print("augment.py smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
