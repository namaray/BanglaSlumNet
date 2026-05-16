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

LOCS = ["korail", "mirpur", "old_dhaka"]

TITLES = {
    "korail": "Korail (real informal settlement)",
    "mirpur": "Mirpur (mixed formal/informal)",
    "old_dhaka": "Old Dhaka (dense formal historic core)",
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

    colors = {"korail": "#d62728", "mirpur": "#ff7f0e", "old_dhaka": "#1f77b4"}
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
            vals = df[df["location"] == loc]["mean_prob"].values
        else:
            vals = np.concatenate(all_probs[loc])

        if len(vals) == 0:
            print(f"[warn] no values for {loc}, skipping violin")
            continue

        pos = positions[loc]
        bp = ax.violinplot(vals, positions=[pos], widths=0.6,
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
    main()