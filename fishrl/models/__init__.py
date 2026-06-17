"""Model package. `device_of` reads the device a module's parameters live on, so
forward-call sites can place their input tensors there without threading a device
argument everywhere."""
from __future__ import annotations

import torch


def device_of(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device
