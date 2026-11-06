from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from common.allowed import TORCHVISION_ARCHS as ALLOWED_ARCHS

WEIGHT_EXTS = {".pth", ".pt", ".bin"}


def find_local_weights(arch: str, root: str | Path) -> Path | None:
    root = Path(root)
    if not root.exists():
        return None
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix in WEIGHT_EXTS]
    official = Path(models.get_model_weights(arch).DEFAULT.url).name
    for p in files:
        if p.name == official:
            return p
    prefixed = sorted(p for p in files if p.stem.lower().split("-")[0] == arch)
    return prefixed[0] if prefixed else None


def replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    if isinstance(model, models.SqueezeNet):
        old = model.classifier[1]
        model.classifier[1] = nn.Conv2d(old.in_channels, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model

    name, last = None, None
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            name, last = module_name, module
    if last is None:
        raise ValueError(f"No final Linear layer found in {type(model).__name__}")
    parent_name, _, attr = name.rpartition(".")
    setattr(model.get_submodule(parent_name), attr, nn.Linear(last.in_features, num_classes))
    return model


def build_model(arch: str, num_classes: int = 2, pretrained: bool = True,
                pretrained_dir: str | None = None, log=print) -> nn.Module:
    if arch not in ALLOWED_ARCHS:
        raise ValueError(f"arch {arch!r} not in the organisers' allowed list")

    kwargs = {"aux_logits": False} if arch == "googlenet" else {}
    local = find_local_weights(arch, pretrained_dir) if pretrained and pretrained_dir else None
    weights = "DEFAULT" if pretrained and local is None else None
    model = getattr(models, arch)(weights=weights, **kwargs)

    if local is not None:
        state = torch.load(local, map_location="cpu", weights_only=False)
        if isinstance(state, nn.Module):
            state = state.state_dict()
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        log(f"[model] loaded offline weights {local} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})")
    elif pretrained:
        log(f"[model] {arch}: torchvision DEFAULT weights")
    else:
        log(f"[model] {arch}: random init")

    return replace_head(model, num_classes)
