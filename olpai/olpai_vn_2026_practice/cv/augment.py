from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LUMA = (0.299, 0.587, 0.114)
DETECT_FRAMES = 4


class ClipAugment:
    def __init__(self, cfg, device: torch.device, swap: torch.Tensor):
        self.cfg = cfg
        self.size = cfg.image_size
        self.device = device
        self.swap = swap.to(device)
        self.log_ratio = (math.log(cfg.rrc_ratio[0]), math.log(cfg.rrc_ratio[1]))
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1) * 255
        self.luma = torch.tensor(LUMA, device=device).view(1, 3, 1, 1)

    def train_batch(self, clips: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        b, t, h, w, _ = clips.shape
        x = self._frames(clips)
        if cfg.noise_p > 0:
            noisy = self._rand(b) < cfg.noise_p
            x = self._add_noise(x, self._rand(b, *cfg.noise_density) * noisy, t)
            if cfg.denoise:
                x = self._median_clips(x, noisy, t)
        x = self._crop(x, self._random_boxes(b, h, w).repeat_interleave(t, 0))
        flip = self._rand(b) < cfg.hflip_p
        x = torch.where(self._per_clip(flip, t), x.flip(3), x)
        labels = torch.where(flip, self.swap[labels], labels)
        x = self._rotate(x, b, t)
        x = self._color_jitter(x, b, t)
        x = self._grayscale(x, b, t)
        return self._normalize(x, b, t), labels

    def eval_frames(self, clips: torch.Tensor, noise_seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = clips.shape[:2]
        x = self._frames(clips)
        if noise_seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(noise_seed)
            lo, hi = self.cfg.noise_density
            x = self._add_noise(x, torch.rand(b, device=self.device, generator=gen) * (hi - lo) + lo, t, gen)
        if not self.cfg.denoise:
            return x, torch.zeros(b, dtype=torch.bool, device=self.device)
        noisy = self._detect(x, b, t)
        return self._median_clips(x, noisy, t), noisy

    def eval_view(self, x: torch.Tensor, t: int, scale: float = 1.0, flip: bool = False) -> torch.Tensor:
        n, _, h, w = x.shape
        if scale < 1 or (h, w) != (self.size, self.size):
            ch, cw = h * scale, w * scale
            box = torch.tensor([(w - cw) / 2, (h - ch) / 2, (w + cw) / 2, (h + ch) / 2], device=self.device)
            x = self._crop(x, box.expand(n, 4))
        if flip:
            x = x.flip(3)
        return self._normalize(x, n // t, t)

    def _add_noise(self, x: torch.Tensor, density: torch.Tensor, t: int, gen=None) -> torch.Tensor:
        n, c, h, w = x.shape
        rand = lambda *shape: torch.rand(*shape, device=self.device, generator=gen)
        hit = (rand(n, 1, h, w) < self._per_clip(density, t)) & (rand(n, c, h, w) < 0.7)
        return torch.where(hit, rand(n, c, h, w) * 255, x)

    def _detect(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        step = max(1, t // DETECT_FRAMES)
        sample = x.view(b, t, *x.shape[1:])[:, ::step].flatten(0, 1)
        outlier = (sample - median3x3(sample)).abs().amax(1) > self.cfg.denoise_diff
        return outlier.float().mean((1, 2)).view(b, -1).mean(1) > self.cfg.denoise_frac

    def _median_clips(self, x: torch.Tensor, mask: torch.Tensor, t: int) -> torch.Tensor:
        frames = mask.repeat_interleave(t).nonzero().squeeze(1)
        if frames.numel():
            x = x.index_copy(0, frames, median3x3(x.index_select(0, frames)))
        return x

    def _frames(self, clips: torch.Tensor) -> torch.Tensor:
        return clips.to(self.device, non_blocking=True).flatten(0, 1).permute(0, 3, 1, 2).float().contiguous()

    def _crop(self, x: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        index = torch.arange(x.size(0), device=self.device, dtype=torch.float).unsqueeze(1)
        return roi_align(x, torch.cat([index, boxes], 1), output_size=(self.size, self.size),
                         spatial_scale=1.0, sampling_ratio=0, aligned=True)

    def _normalize(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return x.reshape(b, t, 3, self.size, self.size)

    def _rand(self, n: int, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        return torch.empty(n, device=self.device).uniform_(lo, hi)

    def _per_clip(self, values: torch.Tensor, t: int) -> torch.Tensor:
        return values[:, None].expand(-1, t).reshape(-1, 1, 1, 1)

    def _random_boxes(self, n: int, h: int, w: int) -> torch.Tensor:
        area = self._rand(n, *self.cfg.rrc_scale) * h * w
        ratio = (w / h) * torch.exp(self._rand(n, *self.log_ratio))
        crop_w = torch.sqrt(area * ratio).clamp(max=w)
        crop_h = torch.sqrt(area / ratio).clamp(max=h)
        x1 = self._rand(n) * (w - crop_w)
        y1 = self._rand(n) * (h - crop_h)
        return torch.stack([x1, y1, x1 + crop_w, y1 + crop_h], 1)

    def _rotate(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        deg = self.cfg.rotate_deg
        if deg <= 0:
            return x
        angle = (self._rand(b, -deg, deg) * math.pi / 180).repeat_interleave(t)
        cos, sin, zero = angle.cos(), angle.sin(), torch.zeros_like(angle)
        theta = torch.stack([torch.stack([cos, -sin, zero], 1), torch.stack([sin, cos, zero], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def _factor(self, b: int, t: int, mask: torch.Tensor) -> torch.Tensor:
        factor = self._rand(b, 1 - self.cfg.jitter, 1 + self.cfg.jitter)
        return self._per_clip(torch.where(mask, factor, torch.ones_like(factor)), t)

    def _color_jitter(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        if self.cfg.jitter_p <= 0:
            return x
        mask = self._rand(b) < self.cfg.jitter_p
        x = x * self._factor(b, t, mask)
        gray = (x * self.luma).sum(1, keepdim=True)
        mean = gray.mean((2, 3), keepdim=True)
        x = (x - mean) * self._factor(b, t, mask) + mean
        gray = (x * self.luma).sum(1, keepdim=True)
        x = gray + (x - gray) * self._factor(b, t, mask)
        return x.clamp(0, 255)

    def _grayscale(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        if self.cfg.gray_p <= 0:
            return x
        gray = (x * self.luma).sum(1, keepdim=True).expand_as(x)
        return torch.where(self._per_clip(self._rand(b) < self.cfg.gray_p, t), gray, x)


def median3x3(x: torch.Tensor) -> torch.Tensor:
    chunk = 64
    out = torch.empty_like(x)
    h, w = x.shape[-2:]
    for start in range(0, x.size(0), chunk):
        part = F.pad(x[start:start + chunk], (1, 1, 1, 1), mode="reflect")
        shifts = [part[..., i:i + h, j:j + w] for i in range(3) for j in range(3)]
        out[start:start + chunk] = torch.stack(shifts).median(0).values
    return out


def tta_views(cfg) -> list[tuple[float, bool]]:
    flips = (False, True) if cfg.tta_hflip else (False,)
    return [(scale, flip) for scale in cfg.tta_scales for flip in flips]
