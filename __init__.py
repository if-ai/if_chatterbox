# Expose node mappings
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# -----------------------------------------------------------------------------
# Make internal model packages importable at top-level so third-party code that
# expects `import s3tokenizer` (or `import s3gen`, etc.) continues to work.
# -----------------------------------------------------------------------------

import importlib, sys

_alias_packages = {
    "s3tokenizer": f"{__name__}.models.s3tokenizer",
    "s3gen": f"{__name__}.models.s3gen",
    "t3": f"{__name__}.models.t3",
}

for public_name, internal_path in _alias_packages.items():
    if public_name not in sys.modules:
        try:
            sys.modules[public_name] = importlib.import_module(internal_path)
        except ModuleNotFoundError:
            # Ignore if the internal module does not exist; will raise later if used
            pass

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']