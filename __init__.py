"""Package init for if_chatterbox.

We expose ComfyUI node mappings *and* create import aliases so that code which
expects top-level packages like ``s3tokenizer`` continues to work when the
models are vendored inside this repository.
"""

import sys, importlib, types
import pathlib

# -----------------------------------------------------------------------------
# 1.  Register alias packages *before* importing sub-modules that rely on them
# -----------------------------------------------------------------------------

_ALIAS_MAP = {
    "s3tokenizer": f"{__name__}.models.s3tokenizer",
    "s3gen": f"{__name__}.models.s3gen",
    "t3": f"{__name__}.models.t3",
}

for public_name, internal_path in _ALIAS_MAP.items():
    if public_name in sys.modules:
        continue  # already provided by user environment

    # Create a lazy proxy module so that `import s3tokenizer` works even if the
    # real internal package fails to import later (e.g., missing deps).
    module_proxy = types.ModuleType(public_name)
    # Expose the filesystem path so that `import s3tokenizer.utils` works via
    # standard import machinery without us eagerly importing every sub-module.
    internal_parts = internal_path.split('.')
    try:
        pkg_dir = pathlib.Path(__file__).parent.joinpath(*internal_parts[1:])
        module_proxy.__path__ = [str(pkg_dir)]
    except Exception:  # pragma: no cover
        pass

    # Optionally attempt eager import; if it fails we retain the proxy.
    try:
        real_mod = importlib.import_module(internal_path)
        sys.modules[public_name] = real_mod
    except (ModuleNotFoundError, ImportError) as e:
        # Keep the proxy module even if import fails due to dependency issues
        sys.modules[public_name] = module_proxy
        print(f"Warning: Failed to import {internal_path}: {e}")
        pass

# -----------------------------------------------------------------------------
# 2.  Import node mappings after aliases are ready
# -----------------------------------------------------------------------------

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]