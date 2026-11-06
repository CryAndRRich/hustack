from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from common import paths
from common.config import ConfigBase
from common.allowed import TORCHVISION_ARCHS

TRACK = "cv"
MARKER = "label_mapping.pkl"
NUM_CLASSES = 100
SPLITS = ("public_test", "private_test")

ALLOWED_ARCHS = tuple(a for a in TORCHVISION_ARCHS if not a.startswith("squeezenet"))


@dataclass
class Config(ConfigBase):
    data_root: str | None = None
    output_dir: str | None = None
    cache_dir: str | None = None
    split: str = "private_test"
    checkpoint: str | None = None

    arch: str = "efficientnet_b0"
    pretrained: bool = True
    tsm: bool = True
    tsm_fold_div: int = 8
    temporal_dim: int = 512
    temporal_layers: int = 2
    temporal_heads: int = 8
    dropout: float = 0.3

    num_frames: int = 16
    image_size: int = 224
    val_size: float = 0.2
    solid_std: float = 6.0

    batch_size: int = 8
    epochs: int = 30
    lr: float = 5e-4
    backbone_lr_scale: float = 0.2
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    logit_adjust: float = 1.0
    ema_decay: float = 0.998

    rrc_scale: tuple = (0.45, 1.0)
    rrc_ratio: tuple = (0.8, 1.25)
    hflip_p: float = 0.5
    flip_pairs: tuple = (("Rẽ trái", "Rẽ phải"),)
    rotate_deg: float = 10.0
    jitter_p: float = 0.8
    jitter: float = 0.4
    gray_p: float = 0.1
    temporal_scale: tuple = (0.7, 1.0)
    noise_p: float = 0.3
    noise_density: tuple = (0.05, 0.4)

    denoise: bool = True
    denoise_diff: float = 80.0
    denoise_frac: float = 0.005
    noisy_val: bool = True

    tta_scales: tuple = (1.0, 0.85)
    tta_hflip: bool = True

    ddp: bool = True
    amp: bool = True
    channels_last: bool = True
    num_workers: int | None = None

    seed: int = 42
    debug: bool = False

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split must be in {SPLITS}, got {self.split!r}")
        if self.arch not in ALLOWED_ARCHS:
            raise ValueError(f"arch {self.arch!r} not in the allowed list (squeezenet excluded): {ALLOWED_ARCHS}")
        if self.temporal_dim % self.temporal_heads:
            raise ValueError("temporal_dim must be divisible by temporal_heads")
        if self.image_size % 32:
            raise ValueError(f"image_size must be divisible by 32, got {self.image_size}")
        for name in ("rrc_scale", "temporal_scale", "noise_density"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi <= 1:
                raise ValueError(f"{name} must be (lo, hi) with 0 < lo <= hi <= 1, got {getattr(self, name)}")
        if not 0 <= self.noise_p <= 1:
            raise ValueError(f"noise_p must be in [0, 1], got {self.noise_p}")
        if self.logit_adjust < 0 or self.solid_std < 0:
            raise ValueError("logit_adjust and solid_std must be >= 0")
        if not 0 <= self.ema_decay < 1:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if not 0 < self.val_size < 1:
            raise ValueError(f"val_size must be in (0, 1), got {self.val_size}")
        if not self.tta_scales or not all(0 < s <= 1 for s in self.tta_scales):
            raise ValueError(f"tta_scales must be numbers in (0, 1], got {self.tta_scales}")
        self.rrc_scale, self.rrc_ratio = tuple(self.rrc_scale), tuple(self.rrc_ratio)
        self.temporal_scale, self.tta_scales = tuple(self.temporal_scale), tuple(self.tta_scales)
        self.noise_density = tuple(self.noise_density)
        self.flip_pairs = tuple(tuple(pair) for pair in self.flip_pairs)
        if any(len(pair) != 2 for pair in self.flip_pairs):
            raise ValueError(f"flip_pairs must be pairs of class names, got {self.flip_pairs}")

    def debug_variant(self) -> Config:
        return replace(self, image_size=96, num_frames=8, batch_size=4, epochs=1, num_workers=0, ddp=False)


def find_data_root(explicit: str | None = None) -> Path:
    candidates = [paths.PROJECT_ROOT / "data" / TRACK, Path.cwd() / "data" / TRACK]
    return paths.find_data_root(explicit, MARKER, candidates, "OLPAI_CV_DATA_ROOT")


def resolve_output_dir(explicit: str | None = None) -> Path:
    return paths.resolve_output_dir(explicit, TRACK)
