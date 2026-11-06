from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from common import paths
from common.config import ConfigBase
from common.allowed import TORCHVISION_ARCHS

TRACK = "cv"
BTC_DATA_ROOTS = (Path("/home/user/TACVU2"), Path("/home/data/TACVU2"))
BTC_MODEL_DIR = Path("/cache_model")

CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
SOURCE_HW = (384, 512)
SPLITS = ("public_test", "private_test")
MONITORS = ("f1", "loss")
SPLIT_MODES = ("cluster", "random")

ALLOWED_ARCHS = TORCHVISION_ARCHS


@dataclass
class Config(ConfigBase):
    data_root: str | None = None
    output_dir: str | None = None
    cache_dir: str | None = None
    split: str = "private_test"
    zip_password: str | None = "ckolp26@vku"
    test_root: str | None = None
    checkpoint: str | None = None

    arch: str = "efficientnet_b0"
    pretrained: bool = True
    pretrained_dir: str | None = None

    min_copies: int = 3
    split_mode: str = "cluster"
    versions_per_image: int = 5
    group_arch: str = "resnet18"
    val_size: float = 0.2

    image_size: int = 384
    batch_size: int = 32
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    label_smoothing: float = 0.1
    class_weighted: bool = False
    monitor: str = "f1"
    ema_decay: float = 0.995

    rrc_scale: tuple = (0.35, 1.0)
    hflip_p: float = 0.5
    vflip_p: float = 0.0
    jitter_p: float = 0.5
    jitter: float = 0.2
    blur_p: float = 0.1

    ddp: bool = True
    compile: bool = False
    compile_mode: str = "default"
    amp: bool = True
    channels_last: bool = True
    num_workers: int | None = None
    deterministic: bool = False

    tta_hflip: bool = True

    seed: int = 42
    debug: bool = False

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split must be in {SPLITS}, got {self.split!r}")
        for name in ("arch", "group_arch"):
            if getattr(self, name) not in ALLOWED_ARCHS:
                raise ValueError(f"{name} {getattr(self, name)!r} is not in the organisers' allowed list")
        if self.group_arch.startswith("squeezenet"):
            raise ValueError("group_arch needs a backbone ending in nn.Linear (not squeezenet)")
        if self.monitor not in MONITORS:
            raise ValueError(f"monitor must be in {MONITORS}, got {self.monitor!r}")
        if self.split_mode not in SPLIT_MODES:
            raise ValueError(f"split_mode must be in {SPLIT_MODES}, got {self.split_mode!r}")
        if self.image_size % 32:
            raise ValueError(f"image_size must be divisible by 32, got {self.image_size}")
        if not 0 < self.rrc_scale[0] <= self.rrc_scale[1] <= 1:
            raise ValueError(f"rrc_scale must be (lo, hi) with 0 < lo <= hi <= 1, got {self.rrc_scale}")
        if not 0 <= self.ema_decay < 1:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if not 0 < self.val_size < 1:
            raise ValueError(f"val_size must be in (0, 1), got {self.val_size}")
        self.rrc_scale = tuple(self.rrc_scale)

    @property
    def input_hw(self) -> tuple[int, int]:
        return self.image_size * 3 // 4, self.image_size

    def debug_variant(self) -> Config:
        return replace(self, image_size=128, batch_size=8, epochs=1, num_workers=0, compile=False, ddp=False)


def find_data_root(explicit: str | None = None) -> Path:
    candidates = [*BTC_DATA_ROOTS, paths.PROJECT_ROOT / "data" / TRACK, Path.cwd() / "data" / TRACK]
    return paths.find_data_root(explicit, "train/train.csv", candidates, "TACVU2_DATA_ROOT")


def default_pretrained_dir(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return str(BTC_MODEL_DIR) if BTC_MODEL_DIR.is_dir() else None


def resolve_output_dir(explicit: str | None = None) -> Path:
    return paths.resolve_output_dir(explicit, TRACK)
