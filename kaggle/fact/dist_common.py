import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import transformers
from packaging.version import Version


def setup():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    timeout = datetime.timedelta(minutes=90)
    try:
        dist.init_process_group("nccl", device_id=dev, timeout=timeout)
    except TypeError:
        dist.init_process_group("nccl", timeout=timeout)
    return dist.get_rank(), dist.get_world_size(), dev


def say(*a):
    if dist.get_rank() == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def sync_max(x, dev):
    t = torch.tensor([float(x)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def equal_shard(n, rank, world, seed):
    perm = np.random.default_rng(seed).permutation(n)
    per = n // world
    return perm[rank * per:(rank + 1) * per]


def lr_scale(progress, warm=0.05):
    return min(1.0, (progress + 1e-3) / warm) * max(0.0, 1.0 - progress)


def dtype_kw(dt):
    return {"dtype": dt} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": dt}


def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def mem_gb():
    return torch.cuda.max_memory_allocated() / 2 ** 30


def budget_batches(order, lens, budget, cap):
    out, a = [], 0
    while a < len(order):
        bs = max(1, min(cap, budget // max(int(lens[order[a]]), 1)))
        out.append(order[a:a + bs])
        a += bs
    return out
