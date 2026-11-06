from __future__ import annotations

import math

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GpuAugment:
    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.size = cfg.image_size
        self.device = device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1) * 255
        self.memory_format = torch.channels_last if cfg.channels_last else torch.contiguous_format

    def train(self, images: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        x = self._to_float(images)
        n = x.size(0)
        zoom = self._rand(n, *cfg.zoom)
        angle = self._rand(n, 0.0, 2 * math.pi) if cfg.rotate else torch.zeros(n, device=self.device)
        reach = (1 - zoom) + cfg.shift
        tx, ty = self._rand(n, -1.0, 1.0) * reach, self._rand(n, -1.0, 1.0) * reach
        cos, sin = angle.cos() * zoom, angle.sin() * zoom
        theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
        grid = F.affine_grid(theta, [n, x.size(1), self.size, self.size], align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        if cfg.jitter_p > 0:
            on = self._rand(n) < cfg.jitter_p
            brightness = self._factor(n, on)
            contrast = self._factor(n, on)
            mean = x.mean((1, 2, 3), keepdim=True)
            x = (((x - mean) * contrast + mean) * brightness).clamp(0, 255)
        return self._normalize(x)

    def eval_base(self, images: torch.Tensor) -> torch.Tensor:
        x = self._to_float(images)
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)
        return x

    def view(self, base: torch.Tensor, k: int) -> torch.Tensor:
        return self._normalize(torch.rot90(base, k, dims=(2, 3)) if k else base)

    def views(self) -> list[int]:
        return [0, 1, 2, 3] if self.cfg.tta_rot90 else [0]

    def _to_float(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.device, non_blocking=True)
        b, t, h, w, c = x.shape
        return x.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w).float()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        frames = x.size(1) // 3
        mean, std = self.mean.repeat(1, frames, 1, 1), self.std.repeat(1, frames, 1, 1)
        return ((x - mean) / std).contiguous(memory_format=self.memory_format)

    def _rand(self, n: int, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        return torch.empty(n, device=self.device).uniform_(lo, hi)

    def _factor(self, n: int, on: torch.Tensor) -> torch.Tensor:
        j = self.cfg.jitter
        factor = torch.where(on, self._rand(n, 1 - j, 1 + j), torch.ones(n, device=self.device))
        return factor.view(n, 1, 1, 1)
