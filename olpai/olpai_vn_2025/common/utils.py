from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class Logger:
    def __init__(self, path: Path, started: float | None = None, prefix: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.time() if started is None else started
        self.prefix = prefix

    def __call__(self, message: str) -> None:
        minutes, seconds = divmod(int(time.time() - self.started), 60)
        line = f"[{minutes:02d}:{seconds:02d}] {self.prefix}{message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
