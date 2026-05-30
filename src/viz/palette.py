"""
Consistent color palette for all paper figures.
Navy / teal / steel-blue / slate with fixed semantic mapping.
Import from here; never define colors in plots.py or qualitative.py directly.
"""

# Hex colors
NAVY   = "#1B2A4A"
TEAL   = "#2A9D8F"
STEEL  = "#4A7FB5"
SLATE  = "#6C757D"
CORAL  = "#E76F51"
SAND   = "#E9C46A"

# Semantic mapping (consistent across all figures)
CONFIG_COLORS = {
    "full":         NAVY,     # BanglaSlumNet-full (primary result)
    "vlm_lang":     TEAL,     # + language
    "vlm_visual":   STEEL,    # VLM visual only
    "baseline_cnn": SLATE,    # optical-only baseline
    "gram":         CORAL,    # GRAM zero-shot
}

REGION_COLORS = {
    "korail":            TEAL,
    "bhashantek":        STEEL,
    "karail_extension":  NAVY,
    "old_dhaka":         CORAL,
    "gulshan_baridhara": SAND,
}

CHANNEL_COLORS = {
    "viirs":       NAVY,
    "worldpop":    TEAL,
    "ghspop":      STEEL,
    "osm_roads":   SLATE,
    "wb_poverty":  CORAL,
    "ghsl_builtup": SAND,
}

# Figure style
DPI = 300
FONT_SIZE = 9           # single-column CVPR width
FONT_SIZE_WIDE = 10     # double-column / slides
LINE_WIDTH = 1.5
MARKER_SIZE = 5

def apply_style(wide: bool = False):
    """Apply consistent matplotlib rcParams."""
    import matplotlib.pyplot as plt
    fs = FONT_SIZE_WIDE if wide else FONT_SIZE
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs,
        "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1,
        "legend.fontsize": fs - 1,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
