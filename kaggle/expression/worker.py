import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fer import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    cfg = json.loads(args.cfg)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`")
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    log = (lambda s: print(s, flush=True)) if rank == 0 else (lambda s: None)

    t0 = time.time()
    d = np.load(args.data)
    X, y, split, X_te = d["X_tr"], d["y"], d["split"], d["X_te"]
    tr, va = np.where(split == 1)[0], np.where(split == 0)[0]
    if cfg.get("max_train"):
        tr = tr[:cfg["max_train"]]
    log(f"  train {len(tr)}  val {len(va)}  on {world} GPU(s), {cfg['bs'] // world} images per GPU per step")
    res = train_model(cfg, X[tr], y[tr], X[va], y[va], X_te, seed=cfg["seed"], log=log, rank=rank, world=world)
    if rank == 0:
        va_p, te_p, hist, state = res
        out = Path(args.out)
        np.savez(out / "run.npz", va_idx=va, va_probs=va_p, te_probs=te_p)
        (out / "hist.json").write_text(json.dumps(hist))
        torch.save(state, out / "model.pt")
        log(f"  done: val acc ({'ten-crop TTA' if cfg['tta'] else 'centre crop'}) "
            f"{float((va_p.argmax(1) == y[va]).mean()):.4f}  in {time.time() - t0:.0f}s")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
