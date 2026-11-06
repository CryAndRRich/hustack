from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from common.allowed import TORCHVISION_ARCHS


def replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    name = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    parent, _, attr = name.rpartition(".")
    last = model.get_submodule(name)
    setattr(model.get_submodule(parent), attr, nn.Linear(last.in_features, num_classes))
    return model


def expand_input_channels(model: nn.Module, frames: int) -> nn.Module:
    if frames == 1:
        return model
    name = next(n for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and m.in_channels == 3)
    old = model.get_submodule(name)
    new = nn.Conv2d(3 * frames, old.out_channels, old.kernel_size, old.stride, old.padding, old.dilation, old.groups,
                    old.bias is not None, old.padding_mode)
    with torch.no_grad():
        new.weight.copy_(old.weight.repeat(1, frames, 1, 1) / frames)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    parent, _, attr = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, attr, new)
    return model


def build_model(arch: str, num_classes: int, pretrained: bool = True, log=print) -> nn.Module:
    if arch not in TORCHVISION_ARCHS:
        raise ValueError(f"arch {arch!r} not in the list")
    model = getattr(models, arch)(weights="DEFAULT" if pretrained else None)
    log(f"[model] {arch}: " + ("torchvision DEFAULT weights" if pretrained else "random init"))
    return replace_head(model, num_classes)
