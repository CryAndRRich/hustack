from __future__ import annotations

import torch
from torch import nn

from common.vision import build_backbone


class TemporalShift:
    def __init__(self, fold_div: int = 8):
        self.fold_div = fold_div
        self.t = 1

    def __call__(self, module, args):
        x = args[0]
        if self.t <= 1:
            return None
        nt, c, h, w = x.shape
        x = x.reshape(nt // self.t, self.t, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return (out.reshape(nt, c, h, w), *args[1:])


def add_temporal_shift(backbone: nn.Module, shift: TemporalShift) -> int:
    from torchvision.models.efficientnet import FusedMBConv, MBConv
    from torchvision.models.mobilenetv2 import InvertedResidual as InvertedResidualV2
    from torchvision.models.mobilenetv3 import InvertedResidual as InvertedResidualV3
    from torchvision.models.regnet import ResBottleneckBlock
    from torchvision.models.resnet import BasicBlock, Bottleneck

    targets = []
    for m in backbone.modules():
        if isinstance(m, (BasicBlock, Bottleneck)):
            targets.append(m.conv1)
        elif isinstance(m, (MBConv, FusedMBConv, InvertedResidualV3)) and m.use_res_connect:
            targets.append(m.block)
        elif isinstance(m, InvertedResidualV2) and m.use_res_connect:
            targets.append(m.conv)
        elif isinstance(m, ResBottleneckBlock):
            targets.append(m.f)
    for target in targets:
        target.register_forward_pre_hook(shift)
    return len(targets)


class VideoClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int, num_frames: int,
                 dim: int, layers: int, heads: int, dropout: float, channels_last: bool = True,
                 shift: TemporalShift | None = None):
        super().__init__()
        self.backbone = backbone
        self.shift = shift
        self.channels_last = channels_last
        self.proj = nn.Linear(feat_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        frames = x.flatten(0, 1)
        if self.channels_last:
            frames = frames.contiguous(memory_format=torch.channels_last)
        if self.shift is not None:
            self.shift.t = t
        feats = self.backbone(frames).view(b, t, -1)
        tokens = self.encoder(self.proj(feats) + self.pos_embed[:, :t])
        return self.head(self.norm(tokens).mean(1))


def build_video_model(cfg, num_classes: int, pretrained: bool | None = None, log=print) -> VideoClassifier:
    backbone, feat_dim = build_backbone(cfg.arch, pretrained=cfg.pretrained if pretrained is None else pretrained,
                                        log=log)
    shift, tsm_info = None, "no TSM"
    if cfg.tsm:
        shift = TemporalShift(cfg.tsm_fold_div)
        n_blocks = add_temporal_shift(backbone, shift)
        if n_blocks == 0:
            raise ValueError(f"TSM does not support {cfg.arch} (supported: resnet, efficientnet, mobilenet, regnet) — set tsm=False")
        tsm_info = f"TSM {n_blocks} block (1/{cfg.tsm_fold_div} channels per direction)"
    model = VideoClassifier(backbone, feat_dim, num_classes, cfg.num_frames, cfg.temporal_dim,
                            cfg.temporal_layers, cfg.temporal_heads, cfg.dropout, cfg.channels_last, shift)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"[model] {cfg.arch} (features {feat_dim}) + {tsm_info} + temporal transformer | {n_params:.1f}M params")
    return model
