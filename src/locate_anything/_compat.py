"""
Compatibility shims for loading LocateAnything-3B on Colab (Python 3.12).

The model's trust_remote_code modules declare `decord` (a video reader) as a
required import, and transformers' check_imports refuses to load the remote code
unless `decord` is importable. decord has no Python 3.12 wheel and building it is
painful. Since BanglaSlumNet uses LocateAnything for IMAGE grounding only (never
video), we inject a lightweight stub module that satisfies the import check.
The stub's video classes raise if actually used, so an accidental video code
path fails loudly rather than silently mis-behaving.
"""

import importlib.machinery
import sys
import types


def ensure_decord() -> bool:
    """
    Ensure a module named `decord` is importable AND has a valid __spec__
    (transformers calls importlib.util.find_spec("decord"), which raises
    ValueError if a stub module's __spec__ is None). If the real package is
    present it is used as-is; otherwise a stub with a proper spec is injected.
    Returns True if the real decord is available, False if the stub was used.
    """
    existing = sys.modules.get("decord")
    if existing is not None:
        # Repair a previously-injected stub that lacks a __spec__.
        if getattr(existing, "__spec__", None) is None:
            existing.__spec__ = importlib.machinery.ModuleSpec("decord", loader=None)
        return getattr(existing, "__version__", "") != "0.0.0-stub"

    try:
        import decord  # noqa: F401
        return True
    except Exception:
        pass

    m = types.ModuleType("decord")
    m.__spec__ = importlib.machinery.ModuleSpec("decord", loader=None)

    class _Bridge:
        def set_bridge(self, *args, **kwargs):
            pass

    class _VideoReader:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "decord stub: video decoding is not supported in this image-only "
                "build. Install a real decord/eva-decord if you need video."
            )

    m.bridge = _Bridge()
    m.VideoReader = _VideoReader
    m.VideoLoader = _VideoReader
    m.cpu = lambda *a, **k: None
    m.gpu = lambda *a, **k: None
    m.__version__ = "0.0.0-stub"
    sys.modules["decord"] = m
    return False
