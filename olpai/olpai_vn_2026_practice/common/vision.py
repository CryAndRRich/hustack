from __future__ import annotations

from torch import nn
from torchvision import models

from common.allowed import TORCHVISION_ARCHS


def build_backbone(arch: str, pretrained: bool = True, log=print) -> tuple[nn.Module, int]:
    if arch not in TORCHVISION_ARCHS or arch.startswith("squeezenet"):
        raise ValueError(f"arch {arch!r} not usable (not on the organisers' list, or squeezenet ending in Conv2d)")
    kwargs = {"aux_logits": False} if arch == "googlenet" else {}
    model = getattr(models, arch)(weights="DEFAULT" if pretrained else None, **kwargs)
    log(f"[model] {arch}: " + ("torchvision DEFAULT weights" if pretrained else "random init"))
    name, last = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    parent, _, attr = name.rpartition(".")
    setattr(model.get_submodule(parent), attr, nn.Identity())
    return model, last.in_features
