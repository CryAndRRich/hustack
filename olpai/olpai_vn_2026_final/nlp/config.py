from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from common import paths
from common.config import ConfigBase
from common.allowed import HF_MODELS

TRACK = "nlp"
BTC_DATA_ROOTS = (Path("/home/user/TACVU1"), Path("/home/data/TACVU1"))
BTC_HF_HOME = Path("/cache_model/huggingface")

DOMAINS = ("Restaurant", "Laptop", "Hotel", "Books", "Clothing")
SPLITS = ("public_test", "private_test")
GENERATIVE_MODELS = ("google-t5/t5-small", "google-t5/t5-base", "Qwen/Qwen3-0.6B")
MONITORS = ("f1", "loss")


@dataclass
class Config(ConfigBase):
    data_root: str | None = None
    output_dir: str | None = None
    split: str = "private_test"
    zip_password: str | None = "ckolp26@vku"
    test_file: str | None = None
    checkpoint: str | None = None

    model_name: str = "google-t5/t5-base"
    train_on_dev: bool = False

    verbalize: bool = True
    max_source_len: int = 256
    max_target_len: int = 160

    epochs: int = 15
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    monitor: str = "f1"
    eval_from_epoch: int = 3
    gradient_checkpointing: bool = False

    eval_beams: int = 1
    num_beams: int = 4
    gen_batch_size: int = 64

    aspect_fuzzy: float = 0.8
    drop_unknown_category: bool = True

    ddp: bool = True
    amp: bool | None = None
    save_model: bool = True

    seed: int = 42
    debug: bool = False

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split must be in {SPLITS}, got {self.split!r}")
        if self.model_name not in HF_MODELS:
            raise ValueError(f"model {self.model_name!r} is not in the organisers' allowed list {HF_MODELS}")
        if self.model_name not in GENERATIVE_MODELS:
            raise ValueError(f"generative pipeline does not support {self.model_name!r} (only {GENERATIVE_MODELS})")
        if self.monitor not in MONITORS:
            raise ValueError(f"monitor must be in {MONITORS}, got {self.monitor!r}")
        if not 0 < self.aspect_fuzzy <= 1:
            raise ValueError(f"aspect_fuzzy must be in (0, 1], got {self.aspect_fuzzy}")

    @property
    def is_seq2seq(self) -> bool:
        return "t5" in self.model_name.lower()

    def debug_variant(self) -> Config:
        return replace(self, model_name="google-t5/t5-small", epochs=1, eval_from_epoch=1, batch_size=8,
                       num_beams=1, ddp=False, save_model=False)


def find_data_root(explicit: str | None = None) -> Path:
    candidates = [*BTC_DATA_ROOTS, paths.PROJECT_ROOT / "data" / TRACK, Path.cwd() / "data" / TRACK]
    return paths.find_data_root(explicit, "train/Train.json", candidates, "TACVU1_DATA_ROOT")


def setup_hf_offline() -> None:
    if BTC_HF_HOME.is_dir():
        os.environ.setdefault("HF_HOME", str(BTC_HF_HOME))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def resolve_output_dir(explicit: str | None = None) -> Path:
    return paths.resolve_output_dir(explicit, TRACK)
