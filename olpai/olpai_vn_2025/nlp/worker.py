from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import torch

from common.jobs import claim, worker_setup
from nlp.config import Config
from nlp.data import read_sentences
from nlp.score import BiEncoder, Reranker, score_pool

MIN_POOL_S = 60


def main() -> None:
    spec, out, log, cfg, device = worker_setup(Config)

    started = time.time()
    paths = spec["model_paths"]
    encoders = [BiEncoder(name, paths[repo], prefix, device, cfg.encode_max_length, cfg.encode_batch)
                for name, repo, prefix in cfg.bi_encoders]
    reranker = Reranker(paths[cfg.reranker], device, cfg.rerank_max_length, cfg.rerank_batch) if cfg.reranker else None
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    memory = f", VRAM {torch.cuda.memory_allocated(device) / 2**30:.1f} GB" if device.type == "cuda" else ""
    log(f"{gpu} | loaded {len(encoders)} bi-encoder + {'1' if reranker else '0'} reranker in "
        f"{time.time() - started:.0f}s{memory}")

    texts = {}
    lock_dir = out / "pools" / ".claims"
    for pool in spec["pools"]:
        if not claim(lock_dir, pool["name"]):
            continue
        pool_dir = out / "pools" / pool["name"]
        remaining = spec["deadline"] - time.time()
        if remaining < MIN_POOL_S:
            log(f"[{pool['name']}] SKIP: {remaining:.0f}s left before deadline")
            continue
        try:
            if pool["split"] not in texts:
                texts[pool["split"]] = read_sentences(Path(cfg.data_root), pool["split"])
            en_text, vi_text = texts[pool["split"]]
            info = score_pool(pool, en_text, vi_text, encoders, reranker, cfg, pool_dir, log)
            (pool_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        except Exception:
            log(f"[{pool['name']}] ERROR:\n{traceback.format_exc()}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    log("no pools left")


if __name__ == "__main__":
    main()
