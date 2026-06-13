"""
Cache LocateAnything-3B to Google Drive so it survives Colab disconnects.
Run once; subsequent sessions use the cached weights.

Usage (in Colab after Drive mount):
    python scripts/download_models.py --cache_dir /gdrive/MyDrive/BanglaSlumNet/model_cache
"""

import argparse
import os
from pathlib import Path


def download_locate_anything(cache_dir: str):
    # Ensure `decord` is importable (stub if needed) before transformers loads
    # the model's trust_remote_code modules. Image-only use; see _compat.
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from src.locate_anything._compat import ensure_decord
        ensure_decord()
    except Exception as e:
        print(f"(decord stub bootstrap skipped: {e})")

    from transformers import AutoModel, AutoProcessor

    print(f"Downloading nvidia/LocateAnything-3B to {cache_dir} ...")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    _ = AutoProcessor.from_pretrained(
        "nvidia/LocateAnything-3B",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    _ = AutoModel.from_pretrained(
        "nvidia/LocateAnything-3B",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    print("Download complete. Model cached.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default="/gdrive/MyDrive/BanglaSlumNet/model_cache",
        help="Google Drive path for model cache",
    )
    args = parser.parse_args()
    download_locate_anything(args.cache_dir)


if __name__ == "__main__":
    main()
