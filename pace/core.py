# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Python surface for the three native helpers previously exposed by the
pybind module `pace._C`.

Each helper is now a torch dispatcher op registered on the `pace` namespace
in `csrc/core/core_ops.cpp`:

- `pace.core.thread_bind`        -> `torch.ops.pace.thread_bind`
- `pace.core.pace_logger`        -> `torch.ops.pace.log`
- `pace.core.enable_pace_fusion` -> `torch.ops.pace.enable_fusion`

Callers should prefer the wrappers below over `torch.ops.pace.*` directly so
the C++ schema can evolve without breaking import sites.
"""

from typing import Sequence

import torch


def thread_bind(cores: Sequence[int]) -> None:
    """Bind the calling process's OpenMP team threads to the given CPU core ids."""
    torch.ops.pace.thread_bind(list(cores))


def pace_logger(level: int, message: str) -> None:
    """Emit a log line through PACE's C++ logger.

    Args:
        level: 0=DEBUG, 1=PROFILE, 2=INFO, 3=WARNING, 4=ERROR.
        message: text to log.
    """
    torch.ops.pace.log(level, message)


def enable_pace_fusion(enabled: bool) -> None:
    """Toggle PACE's JIT graph optimization pass."""
    torch.ops.pace.enable_fusion(enabled)


__all__ = ["thread_bind", "pace_logger", "enable_pace_fusion"]
