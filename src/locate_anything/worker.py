"""
LocateAnythingWorker — interface to nvidia/LocateAnything-3B.
Ported from the HF model card with adaptations per §7.1:
  - load_in_4bit option (default False; enable only on <24 GB GPUs)
  - extract_visual_features() for Direction A feature extraction
  - Drive-backed model cache (never re-download per Colab session)

IMPORTANT: This model uses trust_remote_code=True.
MagiAttention is NOT installed — falls back to PyTorch SDPA automatically on Ampere (Colab A100).

TODO_VERIFY at first model load: run the "Inspect model internals" notebook cell and
confirm that the vision encoder attribute path found by _resolve_vision_encoder()
matches the actual module tree printed there.
"""

import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image

# These imports only succeed after the model is loaded; guarded below.
_transformers_available = False
try:
    from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
    _transformers_available = True
except ImportError:
    warnings.warn("transformers not installed. Install requirements_colab.txt first.")

MODEL_ID = "nvidia/LocateAnything-3B"

# Known candidate attribute paths for the vision encoder in VLM architectures
# (Qwen-VL / EAGLE / similar). The resolver tries these in order.
_VISION_ENCODER_CANDIDATES = [
    "vision_model",
    "visual",
    "vision_tower",
    "visual_encoder",
    "image_encoder",
    "vit",
]


class LocateAnythingWorker:
    """
    Wrapper around nvidia/LocateAnything-3B.
    Loads the model once and caches it for the session.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False,
        cache_dir: Optional[str] = None,
    ):
        if not _transformers_available:
            raise RuntimeError("transformers==4.57.1 is required. Run: pip install transformers==4.57.1")

        self.model_id = model_id
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_4bit = load_in_4bit
        self.cache_dir = cache_dir

        self.model = None
        self.processor = None
        self._vision_encoder = None
        self._hook_handle = None
        self._hook_output = None

    def load(self):
        """Load model and processor. Call once; idempotent."""
        if self.model is not None:
            return

        print(f"Loading {self.model_id} (trust_remote_code=True) ...")

        quant_config = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except ImportError:
                warnings.warn("bitsandbytes not installed; falling back to BF16 full precision.")

        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            quantization_config=quant_config,
            cache_dir=self.cache_dir,
            device_map=self.device if quant_config else None,
        )

        if quant_config is None:
            self.model = self.model.to(self.device)

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        print("Model loaded.")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        generation_mode: str = "slow",
        max_new_tokens: int = 512,
    ) -> str:
        """
        Run grounding generation for an image + text prompt.
        Returns raw generated text (unparsed boxes).
        generation_mode: 'slow' (NTP/AR), 'fast' (MTP), 'hybrid'.
        NOTE: MTP fast mode requires MagiAttention which is unavailable on Colab A100.
        On Ampere, use 'slow' or 'hybrid'; 'fast' falls back via SDPA automatically.
        """
        self.load()
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                generation_mode=generation_mode,
            )
        text = self.processor.decode(outputs[0], skip_special_tokens=False)
        return text

    @staticmethod
    def parse_boxes(text: str, image_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Parse <box><x1><y1><x2><y2></box> tokens from generated text.
        Coordinates are normalized integers in [0, 1000]; converts to pixel space.
        Returns list of (x1, y1, x2, y2) in pixel coordinates.
        """
        W, H = image_size
        pattern = r"<box>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</box>"
        boxes = []
        for m in re.finditer(pattern, text):
            x1, y1, x2, y2 = [int(v) for v in m.groups()]
            px1 = int(x1 / 1000.0 * W)
            py1 = int(y1 / 1000.0 * H)
            px2 = int(x2 / 1000.0 * W)
            py2 = int(y2 / 1000.0 * H)
            boxes.append((px1, py1, px2, py2))
        return boxes

    @staticmethod
    def rasterize_boxes(
        boxes: List[Tuple[int, int, int, int]],
        image_size: Tuple[int, int],
    ) -> "torch.Tensor":
        """Convert list of pixel boxes to a binary coverage mask [H, W]."""
        H, W = image_size[1], image_size[0]
        mask = torch.zeros(H, W, dtype=torch.float32)
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            mask[y1:y2, x1:x2] = 1.0
        return mask

    # ── Feature extraction (Direction A) ──────────────────────────────────────

    def _resolve_vision_encoder(self):
        """
        Try to find the MoonViT encoder in the model's attribute tree.
        Tries known candidate names; falls back to regex search over named modules.
        Returns the encoder module or None (signals grounding_map fallback).

        TODO_VERIFY: Run the "Inspect model internals" cell in the notebook
        and confirm the resolved path matches what is printed there.
        """
        self.load()
        for candidate in _VISION_ENCODER_CANDIDATES:
            if hasattr(self.model, candidate):
                enc = getattr(self.model, candidate)
                print(f"[LocateAnything] Vision encoder found at: model.{candidate}")
                return enc

        # Regex fallback: look for a module whose name suggests a vision encoder
        vision_pattern = re.compile(r"(vision|visual|vit|moonvit|image_enc)", re.IGNORECASE)
        for name, module in self.model.named_modules():
            if vision_pattern.search(name) and "." not in name:
                print(f"[LocateAnything] Vision encoder found via regex: model.{name}")
                return module

        warnings.warn(
            "[LocateAnything] Could not resolve vision encoder automatically. "
            "Falling back to grounding_map feature mode. "
            "Run the 'Inspect model internals' notebook cell and check attribute names."
        )
        return None

    def register_feature_hook(self) -> bool:
        """
        Register a forward hook on the last block of the vision encoder
        to capture patch embeddings [B, N_patches, D].
        Returns True if successful (hidden_state mode), False (grounding_map fallback).
        """
        enc = self._resolve_vision_encoder()
        if enc is None:
            return False

        # Find the last sequential block or layer
        last_block = None
        for name, module in enc.named_modules():
            if hasattr(module, "forward") and list(module.parameters(recurse=False)):
                last_block = module
        if last_block is None:
            # Try the encoder itself as the hook target
            last_block = enc

        def _hook(module, input, output):
            # output may be a tensor or a tuple; capture the first tensor
            if isinstance(output, tuple):
                self._hook_output = output[0].detach()
            elif isinstance(output, torch.Tensor):
                self._hook_output = output.detach()

        self._hook_handle = last_block.register_forward_hook(_hook)
        print("[LocateAnything] Feature hook registered on vision encoder.")
        return True

    def remove_feature_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def extract_visual_features(
        self,
        image: Image.Image,
        prompt: str,
        feature_grid_hw: Tuple[int, int] = (16, 16),
    ) -> Optional[torch.Tensor]:
        """
        Extract MoonViT patch features for a single image + prompt.
        Returns [D, H_f, W_f] float32 tensor on CPU, or None on failure.
        The caller should fall back to grounding_map mode on None.
        """
        self.load()
        if self._hook_handle is None:
            success = self.register_feature_hook()
            if not success:
                return None

        self._hook_output = None
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs, output_hidden_states=True)

        if self._hook_output is None:
            warnings.warn("[LocateAnything] Hook did not fire. Falling back to grounding_map.")
            return None

        feats = self._hook_output  # [1, N_patches, D] or [1, D, H, W]
        if feats.dim() == 3:
            B, N, D = feats.shape
            H_f, W_f = feature_grid_hw
            if N != H_f * W_f:
                # Use nearest square root if grid doesn't match
                import math
                side = int(math.isqrt(N))
                H_f = W_f = side
            feats = feats[0].T.reshape(D, H_f, W_f)  # [D, H_f, W_f]
        elif feats.dim() == 4:
            feats = feats[0]  # [D, H_f, W_f]
        else:
            warnings.warn(f"[LocateAnything] Unexpected feature shape: {feats.shape}")
            return None

        return feats.cpu().float()

    def extract_grounding_map_features(
        self,
        image: Image.Image,
        prompts: List[str],
        generation_mode: str = "slow",
        max_new_tokens: int = 512,
        out_size: Tuple[int, int] = (256, 256),
    ) -> torch.Tensor:
        """
        Fallback feature mode: rasterize grounding box predictions into dense channels.
        Returns [len(prompts), H, W] float32 tensor — one coverage mask per prompt.
        This uses only the public generate() API and is robust to internal changes.
        """
        import torch.nn.functional as F
        maps = []
        for prompt in prompts:
            text = self.generate(image, prompt, generation_mode=generation_mode,
                                 max_new_tokens=max_new_tokens)
            boxes = self.parse_boxes(text, image.size)
            mask = self.rasterize_boxes(boxes, image.size)
            # Resize to out_size
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                 size=out_size, mode="bilinear",
                                 align_corners=False).squeeze()
            maps.append(mask)
        return torch.stack(maps, dim=0)  # [len(prompts), H, W]

    def inspect_model_internals(self):
        """Print model config and module tree for debugging feature extractor setup."""
        self.load()
        print("=" * 60)
        print("MODEL CONFIG:")
        print(self.model.config)
        print("\nTOP-LEVEL ATTRIBUTES:")
        for attr in dir(self.model):
            if not attr.startswith("_"):
                obj = getattr(self.model, attr, None)
                if isinstance(obj, torch.nn.Module):
                    print(f"  model.{attr}: {type(obj).__name__}")
        print("\nNAMED MODULES (depth ≤ 2):")
        for name, mod in self.model.named_modules():
            if name.count(".") <= 1:
                print(f"  {name}: {type(mod).__name__}")
        print("=" * 60)
