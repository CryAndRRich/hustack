from __future__ import annotations

from dataclasses import dataclass

from common.config import ConfigBase
from common.allowed import TORCHVISION_ARCHS

TRACK = "cv"
MARKER = "train/annotations.csv"
NUM_CLASSES = 4
SPLITS = ("public_test", "private_test")

EXPERIMENT_KEYS = ("arch", "pretrained", "image_size", "frames", "batch_size", "epochs", "lr", "weight_decay",
                   "warmup_epochs", "label_smoothing", "neighbor_smoothing", "ema_decay", "eval_every", "rotate", "zoom",
                   "shift", "jitter_p", "jitter", "tta_rot90")


@dataclass
class Config(ConfigBase):
    data_root: str | None = None
    output_dir: str | None = None
    cache_dir: str | None = None

    negative_aspect: float = 1.15

    folds: int = 5
    run_folds: tuple = ()

    arch: str = "convnext_tiny"
    pretrained: bool = True

    cache_size: int = 384
    image_size: int = 320
    frames: tuple = (0,)

    batch_size: int = 32
    epochs: int = 12
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.1
    neighbor_smoothing: float = 0.0
    ema_decay: float = 0.998
    eval_every: int = 3

    rotate: bool = True
    zoom: tuple = (0.7, 1.0)
    shift: float = 0.1
    jitter_p: float = 0.8
    jitter: float = 0.2

    tta_rot90: bool = True

    smooth: bool = True
    smooth_alphas: tuple = (0.3, 0.5, 0.7, 1.0, 1.3, 1.6)
    smooth_mixes: tuple = (0.0, 0.02, 0.05, 0.1, 0.2)
    smooth_gammas: tuple = (0.0, 0.5, 1.0)
    smooth_max_gap_h: int = 24
    smooth_join_splits: bool = True

    time_budget_min: float = 65.0
    amp: bool = True
    channels_last: bool = True
    gpu_cache: bool = True
    save_models: bool = True

    seed: int = 42

    def __post_init__(self):
        self.run_folds, self.zoom, self.frames = tuple(self.run_folds), tuple(self.zoom), tuple(self.frames)
        if 0 not in self.frames or list(self.frames) != sorted(set(self.frames)):
            raise ValueError(f"frames must be increasing, unique, contain 0; got {self.frames}")
        if self.label_smoothing + self.neighbor_smoothing >= 1:
            raise ValueError("label_smoothing + neighbor_smoothing must be < 1")
        self.smooth_alphas, self.smooth_mixes = tuple(self.smooth_alphas), tuple(self.smooth_mixes)
        self.smooth_gammas = tuple(self.smooth_gammas)
        if self.arch not in TORCHVISION_ARCHS:
            raise ValueError(f"arch {self.arch!r} not in {TORCHVISION_ARCHS}")
        if self.folds < 2 or any(not 0 <= f < self.folds for f in self.run_folds):
            raise ValueError(f"need folds >= 2 and run_folds in [0, folds), got {self.folds}, {self.run_folds}")
        if self.image_size % 32 or self.image_size > self.cache_size:
            raise ValueError(f"image_size must be divisible by 32 and <= cache_size, got {self.image_size}/{self.cache_size}")
        lo, hi = self.zoom
        if not 0 < lo <= hi <= 1:
            raise ValueError(f"zoom must be (lo, hi) with 0 < lo <= hi <= 1, got {self.zoom}")
        if not 0 <= self.ema_decay < 1 or self.negative_aspect <= 1:
            raise ValueError("ema_decay trong [0, 1), negative_aspect > 1")

    @property
    def run_fold_list(self) -> list[int]:
        return list(self.run_folds) or list(range(self.folds))


def check_experiments(experiments: dict[str, dict]) -> None:
    if not experiments:
        raise ValueError("experiments is empty")
    for name, overrides in experiments.items():
        bad = [k for k in overrides if k not in EXPERIMENT_KEYS]
        if bad or not name.isidentifier():
            raise ValueError(f"experiment {name!r}: name must be an identifier, may only override {EXPERIMENT_KEYS}, got {bad}")
