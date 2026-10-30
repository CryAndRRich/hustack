import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    cfg = json.loads(args.cfg)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    log = (lambda s: print(s, flush=True)) if rank == 0 else (lambda s: None)
    c = Path(args.cache)
    X, Y, Xv, Yv, Xt = (np.load(c / f"{n}.npy", mmap_mode="r") for n in ("train_x", "train_y", "val_x", "val_y", "test_x"))
    if "rows" in cfg:
        n_tr, n_va, n_te = cfg["rows"]
        X, Y, Xv, Yv, Xt = X[:n_tr], Y[:n_tr], Xv[:n_va], Yv[:n_va], Xt[:n_te]
    log(f"  train {len(X)}, val {len(Xv)}, test {len(Xt)} on {world} GPU(s), {cfg['bs'] // world} images per GPU per step")
    train(cfg, X, Y, Xv, Yv, Xt, args.ckpt, Path(args.out), rank, world, log)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
