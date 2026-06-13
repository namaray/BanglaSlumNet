"""
LocateAnythingWorker — interface to nvidia/LocateAnything-3B.

Ported from the official HF model-card usage (chat-template + process_vision_info +
the model's custom generate signature) with BanglaSlumNet adaptations per §7.1:
  - load_in_4bit option (default False; enable only on <24 GB GPUs)
  - Drive-backed model cache (never re-download per Colab session)
  - extract_visual_features() / extract_grounding_map_features() for Direction A
  - decord stub injected before load (image-only; see _compat.ensure_decord)

Model facts confirmed at load (P1.2):
  vision encoder : model.vision_model   (MoonViT, hidden_size 1152)
  connector      : model.mlp1
  language model : model.language_model  (Qwen2.5-3B)
  box format     : <box><x1><y1><x2><y2></box>, coords normalized to [0,1000]
"""

import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

_transformers_available = False
try:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    _transformers_available = True
except ImportError:
    warnings.warn("transformers not installed. Install requirements_colab.txt first.")

MODEL_ID = "nvidia/LocateAnything-3B"

# Confirmed at P1.2: vision encoder is at model.vision_model. Others kept as fallback.
_VISION_ENCODER_CANDIDATES = [
    "vision_model", "visual", "vision_tower", "visual_encoder", "image_encoder", "vit",
]


class LocateAnythingWorker:
    """Wrapper around nvidia/LocateAnything-3B. Loads once, serves grounding + features."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False,
        cache_dir: Optional[str] = None,
    ):
        if not _transformers_available:
            raise RuntimeError("transformers==4.57.1 required. pip install transformers==4.57.1")
        self.model_id = model_id
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_4bit = load_in_4bit
        self.cache_dir = cache_dir

        self.model = None
        self.processor = None
        self.tokenizer = None
        self._vision_encoder = None
        self._hook_handle = None
        self._hook_output = None

    # ── Loading ────────────────────────────────────────────────────────────────
    def load(self):
        if self.model is not None:
            return

        # LocateAnything's remote code requires `decord`; inject a stub if absent
        # (image-only use, never video). See _compat.ensure_decord.
        from ._compat import ensure_decord
        ensure_decord()

        print(f"Loading {self.model_id} (trust_remote_code=True) ...")
        quant_config = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            except ImportError:
                warnings.warn("bitsandbytes not installed; using BF16 full precision.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, cache_dir=self.cache_dir)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, cache_dir=self.cache_dir)
        self.model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=self.dtype,
            quantization_config=quant_config, cache_dir=self.cache_dir,
            device_map=self.device if quant_config else None,
        )
        if quant_config is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded.")

    # ── Input construction (matches model card) ─────────────────────────────────
    def _build_inputs(self, image: Image.Image, question: str):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)
        return inputs

    # ── Grounding generation ────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "slow",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> str:
        """Run a grounding prompt; return the raw answer string."""
        self.load()
        inputs = self._build_inputs(image, question)
        pixel_values = inputs["pixel_values"].to(self.dtype)
        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )
        answer = response[0] if isinstance(response, tuple) else response
        return answer if isinstance(answer, str) else str(answer)

    def generate(self, image: Image.Image, prompt: str,
                 generation_mode: str = "slow", max_new_tokens: int = 512) -> str:
        """Back-compat alias used by label_validator / feature_extractor."""
        return self.predict(image, prompt, generation_mode=generation_mode,
                            max_new_tokens=max_new_tokens)

    # ── Box parsing (model card format) ─────────────────────────────────────────
    @staticmethod
    def parse_boxes(answer: str, image_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """Parse <box><x1><y1><x2><y2></box> (coords in [0,1000]) -> pixel tuples."""
        W, H = image_size
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append((int(x1 / 1000 * W), int(y1 / 1000 * H),
                          int(x2 / 1000 * W), int(y2 / 1000 * H)))
        return boxes

    @staticmethod
    def rasterize_boxes(boxes: List[Tuple[int, int, int, int]],
                        image_size: Tuple[int, int]) -> torch.Tensor:
        """List of pixel boxes -> binary coverage mask [H, W]."""
        W, H = image_size
        mask = torch.zeros(H, W, dtype=torch.float32)
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1.0
        return mask

    # ── Feature extraction (Direction A) ────────────────────────────────────────
    def _resolve_vision_encoder(self):
        self.load()
        for cand in _VISION_ENCODER_CANDIDATES:
            if hasattr(self.model, cand):
                enc = getattr(self.model, cand)
                if isinstance(enc, torch.nn.Module):
                    return enc
        vision_pat = re.compile(r"(vision|visual|vit|moonvit|image_enc)", re.IGNORECASE)
        for name, module in self.model.named_modules():
            if vision_pat.search(name) and "." not in name:
                return module
        warnings.warn("[LocateAnything] Could not resolve vision encoder; use grounding_map mode.")
        return None

    def register_feature_hook(self) -> bool:
        enc = self._resolve_vision_encoder()
        if enc is None:
            return False
        self._vision_encoder = enc

        def _hook(module, inp, out):
            # MoonViT returns a list[Tensor] (native-resolution, per-image); also
            # handle tuple/tensor. Capture the first tensor found.
            t = out
            while isinstance(t, (list, tuple)) and len(t) > 0:
                t = t[0]
            if isinstance(t, torch.Tensor):
                self._hook_output = t.detach()

        self._hook_handle = enc.register_forward_hook(_hook)
        print("[LocateAnything] Feature hook registered on vision encoder.")
        return True

    def remove_feature_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def extract_visual_features(
        self, image: Image.Image, prompt: str,
        feature_grid_hw: Tuple[int, int] = (16, 16),
    ) -> Optional[torch.Tensor]:
        """
        Return MoonViT patch features [D, H_f, W_f] for one image, or None on failure
        (caller then falls back to grounding_map mode). The vision encoder runs during
        generate prefill; we trigger a 1-token generate and capture the hook output.
        Any LM-side error after the vision pass is swallowed — we already have features.
        """
        self.load()
        if self._hook_handle is None and not self.register_feature_hook():
            return None

        self._hook_output = None
        try:
            inputs = self._build_inputs(image, prompt)
            pixel_values = inputs["pixel_values"].to(self.dtype)
            try:
                self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws", None),
                    tokenizer=self.tokenizer, max_new_tokens=1, use_cache=True,
                    generation_mode="slow", do_sample=False, verbose=False,
                )
            except Exception:
                pass  # vision encoder already ran; LM error is irrelevant for features
        except Exception as e:
            warnings.warn(f"[LocateAnything] feature input build failed: {e}")
            return None

        feats = self._hook_output
        if feats is None:
            return None
        return self._reshape_features(feats, inputs.get("image_grid_hws", None), feature_grid_hw)

    @staticmethod
    def _reshape_features(feats: torch.Tensor, image_grid_hws, feature_grid_hw):
        """Reshape captured [.,N,D] or [N,D] patch features to [D, H_f, W_f]."""
        import math
        if feats.dim() == 3:
            feats = feats[0]            # [N, D]
        if feats.dim() != 2:
            warnings.warn(f"[LocateAnything] unexpected feature dims {tuple(feats.shape)}")
            return None
        N, D = feats.shape
        H_f = W_f = None
        if image_grid_hws is not None:
            try:
                g = image_grid_hws[0].tolist() if hasattr(image_grid_hws[0], "tolist") else list(image_grid_hws[0])
                gh, gw = int(g[-2]), int(g[-1])
                for (a, b) in [(gh, gw), (gh // 2, gw // 2)]:
                    if a * b == N:
                        H_f, W_f = a, b
                        break
            except Exception:
                pass
        if H_f is None:
            s = int(math.isqrt(N))
            H_f = W_f = s
            if H_f * W_f != N:
                return feats.t().reshape(D, N, 1).cpu().float()  # last-resort 1D grid
        return feats.t().reshape(D, H_f, W_f).cpu().float()

    @torch.no_grad()
    def extract_grounding_map_features(
        self, image: Image.Image, prompts: List[str],
        generation_mode: str = "slow", max_new_tokens: int = 512,
        out_size: Tuple[int, int] = (256, 256),
    ) -> torch.Tensor:
        """Fallback: rasterize grounding boxes per prompt into dense channels [P,H,W]."""
        import torch.nn.functional as F
        maps = []
        for prompt in prompts:
            text = self.predict(image, prompt, generation_mode=generation_mode,
                                max_new_tokens=max_new_tokens)
            boxes = self.parse_boxes(text, image.size)
            mask = self.rasterize_boxes(boxes, image.size)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=out_size,
                                 mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            maps.append(mask)
        return torch.stack(maps, dim=0)

    # ── Debug ───────────────────────────────────────────────────────────────────
    def inspect_model_internals(self):
        self.load()
        print("=" * 60)
        print("MODEL CONFIG:\n", self.model.config)
        print("\nTOP-LEVEL MODULES:")
        for attr in dir(self.model):
            if not attr.startswith("_"):
                obj = getattr(self.model, attr, None)
                if isinstance(obj, torch.nn.Module):
                    print(f"  model.{attr}: {type(obj).__name__}")
        print("\nNAMED MODULES (depth <= 2):")
        for name, mod in self.model.named_modules():
            if name.count(".") <= 1:
                print(f"  {name}: {type(mod).__name__}")
        print("=" * 60)
