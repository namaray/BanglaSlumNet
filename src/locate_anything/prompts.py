"""
Centralized prompt templates for LocateAnything grounding.
All prompt strings live here; source code imports from this module.
Every results JSON records PROMPT_VERSION for reproducibility.
"""

PROMPT_VERSION = "v1"

# Neutral prompt: used for vlm_visual config (no concept discrimination)
NEUTRAL = (
    "Locate all dense built-up residential areas in this satellite image."
)

# Discriminative prompts for vlm_lang and full configs
SLUM_DISCRIMINATIVE = (
    "Locate all areas of dense informal settlement: clusters of small irregular "
    "rooftops with varied materials, narrow unpaved gaps between structures, "
    "no organized road grid, buildings packed with no setbacks or courtyards."
)

FORMAL_DISCRIMINATIVE = (
    "Locate all areas of formal urban housing: regular building footprints with "
    "uniform roof materials, organized street grid with wide paved roads, "
    "visible setbacks or courtyards between buildings."
)

# Label-validation prompts (Direction B — used by label_validator.py)
LV_SLUM = SLUM_DISCRIMINATIVE
LV_FORMAL = FORMAL_DISCRIMINATIVE

# Map config names to prompt strings
PROMPT_BY_CONFIG = {
    "vlm_visual": NEUTRAL,
    "vlm_lang_slum": SLUM_DISCRIMINATIVE,
    "vlm_lang_formal": FORMAL_DISCRIMINATIVE,
}


def get_prompt(config: str, role: str = "slum") -> str:
    """Return the appropriate prompt string for a given model config and role."""
    if config == "vlm_visual":
        return NEUTRAL
    elif config in ("vlm_lang", "full"):
        return SLUM_DISCRIMINATIVE if role == "slum" else FORMAL_DISCRIMINATIVE
    else:
        return NEUTRAL
