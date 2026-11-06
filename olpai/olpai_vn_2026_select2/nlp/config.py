from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from common import paths
from common.config import ConfigBase
from common.allowed import TORCHVISION_ARCHS

TRACK = "nlp"
BTC_DATA_ROOT = Path("/home/user/TACVU2/data")
SPLITS = ("public_test", "private_test")
BOLD_MODES = ("cnn", "stroke")


@dataclass
class Config(ConfigBase):
    data_root: str | None = None
    output_dir: str | None = None
    cache_dir: str | None = None
    split: str = "private_test"
    val_docs: float = 0.1

    bold_mode: str = "cnn"
    bold_fallback: bool = True
    checkpoint: str | None = None
    arch: str = "resnet18"
    pretrained: bool = True
    pretrained_dir: str | None = None
    crop_height: int = 48
    crop_width: int = 320
    neg_per_pos: float = 3.0
    batch_size: int = 256
    epochs: int = 6
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 1

    ddp: bool = True
    compile: bool = False
    compile_mode: str = "default"
    amp: bool = True
    channels_last: bool = True
    num_workers: int | None = None
    decode_workers: int | None = None
    deterministic: bool = False

    seed: int = 42
    debug: bool = False

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split must be in {SPLITS}, got {self.split!r}")
        if self.bold_mode not in BOLD_MODES:
            raise ValueError(f"bold_mode must be in {BOLD_MODES}, got {self.bold_mode!r}")
        if self.arch not in TORCHVISION_ARCHS:
            raise ValueError(f"arch {self.arch!r} not in the organisers' allowed list")
        if not 0 < self.val_docs < 1:
            raise ValueError(f"val_docs must be in (0, 1), got {self.val_docs}")

    def debug_variant(self) -> Config:
        return replace(self, epochs=1, batch_size=32, num_workers=0, compile=False, ddp=False, neg_per_pos=1.0)


def find_data_root(explicit: str | None = None) -> Path:
    candidates = [BTC_DATA_ROOT, paths.PROJECT_ROOT / "data" / TRACK, Path.cwd() / "data" / TRACK]
    return paths.find_data_root(explicit, "training_set/manifest.jsonl", candidates, "TACVU2_DATA_ROOT")


def resolve_output_dir(explicit: str | None = None) -> Path:
    return paths.resolve_output_dir(explicit, TRACK)
