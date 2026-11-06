from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from common import paths
from common.config import ConfigBase
from common.allowed import TORCHVISION_ARCHS

TRACK = "cv"
BTC_DATA_ROOT = Path("/home/user/TACVU1/data")

SPLITS = ("public_test", "private_test")
MONITORS = ("f1", "auc", "loss")
CROP_MODES = ("crop", "rrc")
SOURCE_SIZE = 512

ALLOWED_ARCHS = TORCHVISION_ARCHS


@dataclass
class Config(ConfigBase):
    data_root: str | None = None
    output_dir: str | None = None
    cache_dir: str | None = None
    split: str = "private_test"
    checkpoint: str | None = None

    arch: str = "efficientnet_b0"
    pretrained: bool = True
    pretrained_dir: str | None = None

    image_size: int = 512
    crop_mode: str = "crop"
    train_crop: int = 448
    batch_size: int = 32
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 1
    label_smoothing: float = 0.05
    val_size: float = 0.2
    monitor: str = "f1"
    jpeg_p: float = 0.3
    jitter_p: float = 0.3
    blur_p: float = 0.1
    ema_decay: float = 0.99
    refit: bool = False

    ddp: bool = True
    compile: bool = False
    compile_mode: str = "default"
    amp: bool = True
    channels_last: bool = True
    num_workers: int | None = None
    deterministic: bool = False

    tta_hflip: bool = True
    tune_threshold: bool = True

    seed: int = 42
    debug: bool = False

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split must be in {SPLITS}, got {self.split!r}")
        if self.arch not in ALLOWED_ARCHS:
            raise ValueError(f"arch {self.arch!r} not in the organisers' allowed list")
        if self.monitor not in MONITORS:
            raise ValueError(f"monitor must be in {MONITORS}, got {self.monitor!r}")
        if self.crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be in {CROP_MODES}, got {self.crop_mode!r}")
        if self.crop_mode == "crop" and not 0 < self.train_crop <= SOURCE_SIZE:
            raise ValueError(f"train_crop must be in (0, {SOURCE_SIZE}], got {self.train_crop}")
        if not 0 <= self.ema_decay < 1:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if not 0 < self.val_size < 1:
            raise ValueError(f"val_size must be in (0, 1), got {self.val_size}")

    def debug_variant(self) -> Config:
        return replace(
            self,
            image_size=min(self.image_size, 128),
            train_crop=min(self.train_crop, 128),
            batch_size=8,
            epochs=1,
            num_workers=0,
            compile=False,
            ddp=False,
        )


def find_data_root(explicit: str | None = None) -> Path:
    candidates = [BTC_DATA_ROOT, paths.PROJECT_ROOT / "data" / TRACK, Path.cwd() / "data" / TRACK]
    return paths.find_data_root(explicit, "train/manifest.csv", candidates, "TACVU1_DATA_ROOT")


def resolve_output_dir(explicit: str | None = None) -> Path:
    return paths.resolve_output_dir(explicit, TRACK)
