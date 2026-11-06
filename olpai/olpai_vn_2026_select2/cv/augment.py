from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LUMA = (0.299, 0.587, 0.114)


class GpuAugment:
    def __init__(self, image_size: int, device: torch.device, train: bool, channels_last: bool = True,
                 crop_mode: str = "rrc", crop_size: int | None = None,
                 scale=(0.6, 1.0), ratio=(0.9, 1.1), flip_p: float = 0.5,
                 jitter_p: float = 0.3, jitter: float = 0.1, blur_p: float = 0.1):
        self.native_crop = train and crop_mode == "crop"
        side = crop_size if self.native_crop else image_size
        self.size = (side, side)
        self.device = device
        self.train = train
        self.channels_last = channels_last
        self.scale = scale
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        self.flip_p = flip_p
        self.jitter_p = jitter_p
        self.jitter = jitter
        self.blur_p = blur_p
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1) * 255
        self.luma = torch.tensor(LUMA, device=device).view(1, 3, 1, 1)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().contiguous()
        n, _, h, w = x.shape
        if self.native_crop:
            boxes = self._native_crop_boxes(n, h, w)
        elif self.train:
            boxes = self._random_boxes(n, h, w)
        else:
            boxes = self._full_boxes(n, h, w)
        x = roi_align(x, boxes, output_size=self.size, spatial_scale=1.0, sampling_ratio=0, aligned=True)
        if self.train:
            x = self._flip(x)
            x = self._color_jitter(x)
            x = self._blur(x)
        x = (x - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last) if self.channels_last else x

    def _rand(self, n: int, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        return torch.empty(n, device=self.device).uniform_(lo, hi)

    def _full_boxes(self, n: int, h: int, w: int) -> torch.Tensor:
        boxes = torch.tensor([0.0, 0.0, w, h], device=self.device).expand(n, 4)
        return torch.cat([torch.arange(n, device=self.device, dtype=torch.float).unsqueeze(1), boxes], 1)

    def _native_crop_boxes(self, n: int, h: int, w: int) -> torch.Tensor:
        side = self.size[0]
        x1 = torch.randint(0, w - side + 1, (n,), device=self.device).float()
        y1 = torch.randint(0, h - side + 1, (n,), device=self.device).float()
        index = torch.arange(n, device=self.device, dtype=torch.float)
        return torch.stack([index, x1, y1, x1 + side, y1 + side], 1)

    def _random_boxes(self, n: int, h: int, w: int) -> torch.Tensor:
        area = self._rand(n, *self.scale) * h * w
        ratio = torch.exp(self._rand(n, *self.log_ratio))
        crop_w = torch.sqrt(area * ratio).clamp(max=w)
        crop_h = torch.sqrt(area / ratio).clamp(max=h)
        x1 = self._rand(n) * (w - crop_w)
        y1 = self._rand(n) * (h - crop_h)
        index = torch.arange(n, device=self.device, dtype=torch.float)
        return torch.stack([index, x1, y1, x1 + crop_w, y1 + crop_h], 1)

    def _flip(self, x: torch.Tensor) -> torch.Tensor:
        mask = (self._rand(x.size(0)) < self.flip_p).view(-1, 1, 1, 1)
        return torch.where(mask, x.flip(3), x)

    def _factor(self, n: int, mask: torch.Tensor) -> torch.Tensor:
        factor = self._rand(n, 1 - self.jitter, 1 + self.jitter)
        return torch.where(mask, factor, torch.ones_like(factor)).view(-1, 1, 1, 1)

    def _color_jitter(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        mask = self._rand(n) < self.jitter_p
        x = x * self._factor(n, mask)
        gray = (x * self.luma).sum(1, keepdim=True)
        mean = gray.mean((2, 3), keepdim=True)
        x = (x - mean) * self._factor(n, mask) + mean
        gray = (x * self.luma).sum(1, keepdim=True)
        x = gray + (x - gray) * self._factor(n, mask)
        return x.clamp(0, 255)

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        if self.blur_p <= 0:
            return x
        n, c, h, w = x.shape
        sigma = self._rand(n, 0.1, 1.0)
        taps = torch.tensor([-1.0, 0.0, 1.0], device=self.device)
        kernel = torch.exp(-(taps.view(1, 3) ** 2) / (2 * sigma.view(n, 1) ** 2))
        kernel = (kernel / kernel.sum(1, keepdim=True)).repeat_interleave(c, 0)
        flat = x.reshape(1, n * c, h, w)
        flat = F.conv2d(F.pad(flat, (1, 1, 0, 0), mode="reflect"), kernel.view(n * c, 1, 1, 3), groups=n * c)
        flat = F.conv2d(F.pad(flat, (0, 0, 1, 1), mode="reflect"), kernel.view(n * c, 1, 3, 1), groups=n * c)
        mask = (self._rand(n) < self.blur_p).view(-1, 1, 1, 1)
        return torch.where(mask, flat.view(n, c, h, w), x)
