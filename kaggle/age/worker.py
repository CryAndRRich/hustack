import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune import train


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
    d = dict(np.load(args.data))
    log(f"  train {len(d['X_tr'])}, val {len(d['X_va'])} (monitor {len(d['mon'])}), judge {len(d['X_te'])} "
        f"on {world} GPU(s), {cfg['bs'] // world} images per GPU per step")
    pv, pt, hist, best_epoch = train(cfg, d["X_tr"], d["y_tr"], d["X_va"], d["y_va"], d["mon"], d["X_te"],
                                     rank, world, log)
    if rank == 0:
        out = Path(args.out)
        np.savez(out / "run.npz", va=pv, te=pt)
        (out / "hist.json").write_text(json.dumps({"hist": hist, "best_epoch": best_epoch}))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
