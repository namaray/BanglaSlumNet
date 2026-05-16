"""Create a single composite figure summarizing the GRAM-on-Dhaka baseline."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
LOCS = ["korail", "mirpur", "old_dhaka"]
TITLES = {
    "korail":    "Korail (largest slum in Dhaka)",
    "mirpur":    "Mirpur (mixed formal/informal)",
    "old_dhaka": "Old Dhaka (dense historic core + river)",
}

fig, axes = plt.subplots(len(LOCS), 3, figsize=(18, 14))
for r, loc in enumerate(LOCS):
    mosaic = np.array(Image.open(os.path.join(OUT_DIR, f"{loc}_gram_baseline.png")))
    H, W = mosaic.shape[:2]
    third = W // 3
    rgb     = mosaic[:, 0:third]
    heat    = mosaic[:, third:2*third]
    overlay = mosaic[:, 2*third:3*third]
    axes[r, 0].imshow(rgb);     axes[r, 0].set_title(f"{TITLES[loc]}\nESRI imagery (z=16, ~1.2 m/px)", fontsize=11)
    axes[r, 1].imshow(heat);    axes[r, 1].set_title("GRAM slum probability (class 1)\nblue = 0, red = 1", fontsize=11)
    axes[r, 2].imshow(overlay); axes[r, 2].set_title("Binary slum mask (p > 0.5)\noverlaid in red on RGB", fontsize=11)
    for c in range(3):
        axes[r, c].axis("off")

plt.suptitle("GRAM (zero-shot, AAAI'26) on Dhaka — pretrained MoE checkpoint, domain idx=6",
             fontsize=14, y=0.995)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gram_dhaka_summary_figure.png"), dpi=110, bbox_inches="tight")
print("saved gram_dhaka_summary_figure.png")
