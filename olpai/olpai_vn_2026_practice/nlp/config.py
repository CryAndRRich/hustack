from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

from common import paths
from common.config import ConfigBase

TRACK = "nlp"
MARKER = "train/train.zh"
SPLITS = ("public_test", "private_test")
TOKENIZERS = ("bpe", "unigram", "char", "word")


@dataclass
class Config(ConfigBase):
    data_root: str | None = None
    output_dir: str | None = None
    split: str = "private_test"
    checkpoint: str | None = None

    val_size: int = 1500
    dedupe_pairs: bool = True
    use_memory: bool = True

    zh_tokenizer: str = "bpe"
    zh_vocab: int = 8000
    vi_tokenizer: str = "bpe"
    vi_vocab: int = 8000
    vi_spm: str | None = None
    subword_dropout: float = 0.1
    max_tokens: int = 128

    d_model: int = 512
    heads: int = 4
    ffn_dim: int = 1024
    enc_layers: int = 6
    dec_layers: int = 6
    dropout: float = 0.3

    batch_tokens: int = 4096
    epochs: int = 180
    lr: float = 7e-4
    warmup_steps: int = 1000
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    clip_norm: float = 1.0
    eval_every: int = 5
    eval_from: int = 10
    patience: int = 6
    average_top: int = 5

    beam: int = 5
    len_penalty: float = 1.0
    max_len_a: float = 1.5
    max_len_b: int = 10
    gen_rows: int = 1024

    ensemble: bool = True
    lp_grid: tuple = (1.6, 2.0, 2.5, 3.0)
    beam_grid: tuple = (10, 15)
    weight_grid: tuple = (0.6, 0.7, 0.8)

    amp: bool = True
    seed: int = 42
    run_seed: int = 0
    debug: bool = False

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split must be in {SPLITS}, got {self.split!r}")
        for name in ("zh_tokenizer", "vi_tokenizer"):
            if getattr(self, name) not in TOKENIZERS:
                raise ValueError(f"{name} must be in {TOKENIZERS}, got {getattr(self, name)!r}")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.val_size < 1 or self.beam < 1 or self.average_top < 1:
            raise ValueError("val_size, beam, average_top must be >= 1")
        if not 0 <= self.subword_dropout < 1:
            raise ValueError(f"subword_dropout must be in [0, 1), got {self.subword_dropout}")
        self.lp_grid, self.beam_grid, self.weight_grid = tuple(self.lp_grid), tuple(self.beam_grid), tuple(self.weight_grid)
        if not self.lp_grid or not self.beam_grid:
            raise ValueError("lp_grid and beam_grid must not be empty")
        if not all(0 < w < 1 for w in self.weight_grid):
            raise ValueError(f"weight_grid must be numbers in (0, 1), got {self.weight_grid}")

    @classmethod
    def from_saved(cls, values: dict, **overrides) -> Config:
        known = {f.name for f in fields(cls)}
        return cls(**{**{k: v for k, v in values.items() if k in known}, **overrides})

    def debug_variant(self) -> Config:
        return replace(self, val_size=200, zh_vocab=1000, vi_vocab=1000, d_model=128, ffn_dim=256,
                       enc_layers=2, dec_layers=2, epochs=2, eval_every=1, eval_from=1, warmup_steps=50,
                       average_top=2, beam=2)


def find_data_root(explicit: str | None = None) -> Path:
    candidates = [paths.PROJECT_ROOT / "data" / TRACK, Path.cwd() / "data" / TRACK]
    return paths.find_data_root(explicit, MARKER, candidates, "OLPAI_NLP_DATA_ROOT")


def resolve_output_dir(explicit: str | None = None) -> Path:
    return paths.resolve_output_dir(explicit, TRACK)
