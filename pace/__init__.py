# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""PACE - Platform Aware Compute Engine.

Importing `pace` loads the bundled `libpace_cpp.so` via
`torch.ops.load_library`, which fires the `TORCH_LIBRARY_FRAGMENT` static
initializers and populates `torch.ops.pace.*` + `torch.classes.pace.*`. The
library ships no CPython extension module; the wheel is consequently
`py3-none-<plat>` and works on any CPython 3.x within the supported range
declared in pyproject.toml.
"""

from pathlib import Path

from .version import __version__, __version_tuple__

try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError("Torch not found, install torch. Refer to README.md.")

# Load libpace_cpp.so once at import time. dlopen() refcounts, so a redundant
# call from e.g. `pace_vllm` later in the same process is a no-op for the
# dynamic linker and the TORCH_LIBRARY_FRAGMENT initializers run exactly once.
_LIB_PATH = Path(__file__).resolve().parent / "lib" / "libpace_cpp.so"
torch.ops.load_library(str(_LIB_PATH))

from . import core
from . import utils
from . import llm
from . import ops
from ._register_fake import *  # noqa: F401,F403

__all__ = ["__version__", "__version_tuple__", "core", "utils", "llm", "ops"]
