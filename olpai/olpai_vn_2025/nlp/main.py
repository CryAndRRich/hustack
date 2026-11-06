from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from common.jobs import reset_claims, run_workers
from common.utils import Logger
from nlp.align import aggregate
from nlp.config import MARKER, TRACK, Config
from nlp.data import make_pools

AGGREGATE_RESERVE_S = 180
SKIP_FILES = ["*.onnx", "onnx/*", "*.h5", "tf_model*", "flax_model*", "*.msgpack", "openvino/*", "rust_model*", "*.ot"]


def ensure_packages(log) -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        log("[install] pip install sentence-transformers")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)


def download_models(cfg: Config, log) -> dict[str, str]:
    from huggingface_hub import list_repo_files, snapshot_download

    repos = [repo for _, repo, _ in cfg.bi_encoders] + ([cfg.reranker] if cfg.reranker else [])
    out = {}
    for repo in repos:
        started = time.time()
        files = list_repo_files(repo)
        skip = SKIP_FILES + (["pytorch_model.bin"] if "model.safetensors" in files else [])
        out[repo] = snapshot_download(repo, ignore_patterns=skip)
        size = sum(f.stat().st_size for f in Path(out[repo]).rglob("*") if f.is_file()) / 2**30
        log(f"[model] {repo}: {size:.2f} GB ({time.time() - started:.0f}s)")
    return out


def run(base: Config | None = None) -> dict:
    t0 = time.time()
    cfg = (base or Config()).with_paths(TRACK, MARKER)
    out = Path(cfg.output_dir)
    reset_claims(out / "pools")
    log = Logger(out / "train.log", t0)
    log(f"[config] {json.dumps(asdict(cfg), ensure_ascii=False)}")

    pools = make_pools(Path(cfg.data_root), cfg.val_blocks)
    log("[pool] " + " | ".join(f"{p['name']}: {len(p['en'])} EN × {len(p['vi'])} VI" for p in pools))
    ensure_packages(log)
    spec = {"t0": t0, "deadline": t0 + cfg.time_budget_min * 60 - AGGREGATE_RESERVE_S, "output_dir": str(out),
            "base": asdict(cfg), "pools": pools, "model_paths": download_models(cfg, log)}
    run_workers("nlp.worker", spec, out, len(pools), log,
                extra_env={"TOKENIZERS_PARALLELISM": "false", "TRANSFORMERS_VERBOSITY": "error"})
    result = aggregate(out, log)
    log(f"[done] total {(time.time() - t0) / 60:.1f} min")
    return result


if __name__ == "__main__":
    run(Config().with_overrides(sys.argv[1:]))
