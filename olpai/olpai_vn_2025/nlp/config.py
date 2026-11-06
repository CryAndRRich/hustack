from __future__ import annotations

from dataclasses import dataclass, field

from common.config import ConfigBase

TRACK = "nlp"
MARKER = "train/correct_alignment.csv"
SPLITS = ("public_test", "private_test")


@dataclass
class Config(ConfigBase):
    data_root: str | None = None
    output_dir: str | None = None

    bi_encoders: tuple = (("labse", "sentence-transformers/LaBSE", ""),
                          ("bge", "BAAI/bge-m3", ""),
                          ("e5", "intfloat/multilingual-e5-large", "query: "))
    reranker: str = "BAAI/bge-reranker-v2-m3"
    encode_max_length: int = 256
    rerank_max_length: int = 384
    encode_batch: int = 128
    rerank_batch: int = 64

    csls_k: int = 4
    k_11: int = 10
    k_11_rev: int = 3
    k_21: int = 5
    k_12_pool: int = 5
    k_12: int = 5

    val_blocks: int = 5

    decoder: dict | None = field(default_factory=lambda: {"a": 0.2, "thr": -0.5, "b21": 0.0, "b12": -0.5, "w_name": 0.0,
                                                          "w_len": 0.5, "tail_frac": 0.01, "tail_pen": 2.0})

    time_budget_min: float = 60.0
    seed: int = 42

    def __post_init__(self):
        self.bi_encoders = tuple(tuple(m) for m in self.bi_encoders)
        if any(len(m) != 3 for m in self.bi_encoders):
            raise ValueError("bi_encoders: each item is (name, repo, prefix)")
        if not self.bi_encoders:
            raise ValueError("need at least one bi-encoder to generate candidates")
        if self.val_blocks < 1 or min(self.k_11, self.k_21, self.k_12_pool, self.k_12, self.csls_k) < 1:
            raise ValueError("val_blocks and all k must be >= 1")
        self.decoder = dict(self.decoder or {})
