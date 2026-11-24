import dataclasses
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from common.data import COUNTS, build_cache, find_root, list_images, open_cache
from common.stitch import find_groups, members_of

T0 = time.time()
RANKS = 2


def log(msg):
    print(f"[main {time.strftime('%H:%M:%S')} +{(time.time() - T0) / 3600:.2f} h] {msg}", flush=True)


def prepare(cfg, log):
    import torch
    import transformers
    out, work = Path(cfg.out_dir), Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout
    log(f"torch {torch.__version__}, transformers {transformers.__version__}, GPUs {gpus.strip().splitlines()}")
    assert torch.cuda.device_count() == 2, "this notebook needs GPU T4 x2"
    root = find_root(cfg)
    names = {s: list_images(root, s) for s in COUNTS}
    log(f"data root {root}: " + ", ".join(f"{s} {len(v)}" for s, v in names.items()))
    assert {s: len(v) for s, v in names.items()} == COUNTS
    (work / "names.json").write_text(json.dumps(names))
    free = shutil.disk_usage(work).free / 2**30
    log(f"free disk in {work}: {free:.0f} GB")
    assert free > 30, "need > 30 GB free for the decoded cache"
    build_cache(root, work / "cache", names, log)
    return out, work, root, names, open_cache(work / "cache")


def tile_groups(cache, work, log):
    pos = {s: find_groups(cache[f"img_{s}"], list(range(COUNTS[s])), log, s) for s in ("val", "test")}
    (work / "groups.json").write_text(json.dumps(pos))
    return {s: {int(k): v for k, v in g.items()} for s, g in pos.items()}


def write_indices(work, sidx):
    (work / "idx_full.json").write_text(json.dumps({s: list(range(COUNTS[s])) for s in ("val", "test")}))
    (work / "idx_smoke.json").write_text(json.dumps(sidx))


def smoke_idx(pos, n, k):
    groups = members_of(pos)
    if not groups:
        return list(range(min(k, n)))
    g = max(groups.values(), key=len)
    _, ay, ax = pos[g[0]]
    return sorted(sorted(g, key=lambda j: abs(pos[j][1] - ay) + abs(pos[j][2] - ax))[:k])


def stream(proc, path):
    with open(path, "w") as f:
        for line in proc.stdout:
            try:
                print(line, end="", flush=True)
                f.write(line)
            except Exception:
                pass


def run_workers(cfg, mode, cfg_path, module):
    code_dir = str(Path(__file__).resolve().parents[1])
    logs = Path(cfg.work_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for rank in range(RANKS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), PYTHONPATH=code_dir,
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        p = subprocess.Popen([sys.executable, "-u", "-m", module, str(rank), mode, cfg_path],
                             cwd=code_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        th = threading.Thread(target=stream, args=(p, logs / f"gpu{rank}_{mode}.log"), daemon=True)
        th.start()
        procs.append((rank, p, th))
    failed = []
    while any(p.poll() is None for _, p, _ in procs) and not failed:
        time.sleep(10)
        failed = [r for r, p, _ in procs if p.poll() not in (None, 0)]
    for _, p, th in procs:
        if failed and p.poll() is None:
            p.kill()
        p.wait()
        th.join()
    failed = [r for r, p, _ in procs if p.returncode != 0]
    assert not failed, f"{mode} worker(s) on GPU {failed} failed (see logs/)"
    return [json.loads((Path(cfg.work_dir) / mode / f"result_{r}.json").read_text()) for r in range(RANKS)]


def stage(cfg, mode, module):
    path = Path(cfg.work_dir) / ("config_smoke.json" if mode == "smoke" else "config.json")
    path.write_text(cfg.to_json())
    return run_workers(cfg, mode, str(path), module)


def smoke_run(cfg, module, names, sidx, folders):
    work = Path(cfg.work_dir)
    res = stage(cfg, "smoke", module)
    package(work / "smoke" / "pred", [names["test"][i] for i in sidx["test"]], work / "smoke" / "submission_smoke.zip", folders)
    shutil.rmtree(work / "smoke" / "models", ignore_errors=True)
    return res


def full_run(cfg, module, names, folders):
    res = stage(cfg, "full", module)
    package(Path(cfg.work_dir) / "full" / "pred", names["test"], Path(cfg.out_dir) / "submission.zip", folders)
    return res


def timings(smoke_res):
    return {k: max(r["timing"][k] for r in smoke_res) for k in smoke_res[0]["timing"]}


def train_budget(cfg, smoke_res, val_keys, test_keys, what, test_floor=()):
    if cfg.train_hours > 0:
        return cfg.train_hours
    tm = timings(smoke_res)
    for t, v in test_floor:
        tm[t] = max(tm[t], tm[v])
    nv, nt = (COUNTS["val"] + 1) // 2, (COUNTS["test"] + 1) // 2
    inf = cfg.margin * (nv * sum(tm[k] for k in val_keys) + nt * sum(tm[k] for k in test_keys))
    fixed, elapsed = 900, time.time() - T0
    train_s = cfg.session_target_hours * 3600 - elapsed - inf - fixed
    train_s = min(max(train_s, cfg.min_train_hours * 3600), cfg.max_train_hours * 3600)
    total = elapsed + train_s + inf + fixed
    log(f"smoke s/img: { {k: round(v, 2) for k, v in tm.items()} }; inference {inf / 3600:.2f} h; "
        f"{what} {train_s / 3600:.2f} h; expected session {total / 3600:.2f} h")
    assert total < cfg.session_hours * 3600, f"expected session {total / 3600:.2f} h exceeds {cfg.session_hours} h"
    return train_s / 3600


def package(pred_dir, names, zip_path, folders):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for folder in folders:
            z.writestr(folder + "/", "")
            for n in names:
                z.write(pred_dir / folder / n, f"{folder}/{n}")
    with zipfile.ZipFile(zip_path) as z:
        expected = {f + "/" for f in folders} | {f"{f}/{n}" for f in folders for n in names}
        entries = set(z.namelist())
        assert entries == expected, f"zip entries differ: missing {sorted(expected - entries)[:5]} extra {sorted(entries - expected)[:5]}"
        for f in folders:
            for n in names:
                a = np.asarray(Image.open(io.BytesIO(z.read(f"{f}/{n}"))))
                dt = np.uint8 if f == "semantics" else np.uint16
                assert a.shape == (1024, 1024) and a.dtype == dt, f"{f}/{n}: {a.shape} {a.dtype}"
    log(f"validated {zip_path}: {len(names)} images x {folders}, 1024x1024, {zip_path.stat().st_size / 2**20:.0f} MB")


def by_date(paths, names, cand, thr, offset, cfg):
    rows = {}
    for p in paths:
        z = np.load(p)
        o, t = cfg.small_offsets.index(offset), cfg.thresholds.index(thr)
        for i, v in zip(z["idx"], z[cand][:, o, t]):
            if not np.isnan(v):
                rows.setdefault(names[int(i)][:5], []).append(v)
    return {d: float(np.mean(v) * 100) for d, v in sorted(rows.items())}


def fingerprint(path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return hashlib.md5(f.read(16 * 2**20)).hexdigest()


def find_best_models(cfg):
    if cfg.best_models_dir:
        return cfg.best_models_dir
    hits = sorted(p.parent.parent for p in Path("/kaggle/input").glob("**/models/plant_m2f/config.json")
                  if (p.parent.parent / "leaf_eomt" / "config.json").exists())
    assert hits, "best models (models/plant_m2f, models/leaf_eomt) not found under /kaggle/input: attach plantleaf-best"
    seen = {}
    for h in hits:
        seen[str(h)] = {r: fingerprint(h / r if r.endswith(".pt") else h / r / "model.safetensors") for r in cfg.best_fingerprints}
        if seen[str(h)] == cfg.best_fingerprints:
            return str(h)
    raise AssertionError(f"no attached models/ folder holds the weights {cfg.best_fingerprints}; found {seen}: "
                         "attach the latest version of plantleaf-best")


def with_best_models(cfg, log):
    cfg = dataclasses.replace(cfg, best_models_dir=find_best_models(cfg))
    log(f"best models: {cfg.best_models_dir}")
    return cfg


def bundle(cfg, zip_path):
    work = Path(cfg.work_dir)
    files = [p for d in (work, work / "full") if d.exists() for p in sorted(d.glob("*")) if p.is_file() and p.suffix == ".json"]
    files += sorted(work.glob("full/*_val/c*.npz"))
    keep = json.loads((work / "result.json").read_text())["kept_models"] if (work / "result.json").exists() else []
    for sub in ["logs", "code", "full/val_pred"] + [f"full/models/{m}" for m in keep]:
        files += [p for p in sorted((work / sub).rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    with zipfile.ZipFile(zip_path, "w") as z:
        for p in files:
            z.write(p, p.relative_to(work), zipfile.ZIP_STORED if p.suffix == ".safetensors" else zipfile.ZIP_DEFLATED)
    log(f"wrote {zip_path}: {len(files)} files, {zip_path.stat().st_size / 2**20:.0f} MB")


def run(cfg, pipeline, log=log):
    out = Path(cfg.out_dir)
    try:
        pipeline(cfg)
    finally:
        bundle(cfg, out / "results.zip")
        log(f"outputs: {sorted(p.name for p in out.iterdir())}")


def make_log(rank):
    def log(msg):
        print(f"[gpu{rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)
    return log


class Sync:
    def __init__(self, d, rank, log):
        self.d, self.rank, self.log = d, rank, log
        d.mkdir(parents=True, exist_ok=True)

    def barrier(self, name):
        (self.d / f"{name}.{self.rank}").write_text("ok")
        t = time.time()
        while not all((self.d / f"{name}.{r}").exists() for r in range(RANKS)):
            time.sleep(2)
        self.log(f"barrier {name}: waited {time.time() - t:.0f} s")

    def decide(self, name, fn):
        path = self.d / f"{name}.json"
        if self.rank == 0:
            tmp = self.d / f"{name}.tmp"
            tmp.write_text(json.dumps(fn()))
            os.replace(tmp, path)
        while not path.exists():
            time.sleep(2)
        return json.loads(path.read_text())


def per_image(times):
    return float(np.mean(times[1:] if len(times) > 1 else times))


class Worker:
    def __init__(self, config_cls, rank, mode, cfg_path, dirs):
        import torch
        self.rank = rank = int(rank)
        self.cfg = cfg = config_cls.from_json(Path(cfg_path).read_text())
        self.smoke = mode == "smoke"
        random.seed(cfg.seed + rank)
        np.random.seed(cfg.seed + rank)
        torch.manual_seed(cfg.seed + rank)
        self.log = make_log(rank)
        self.work = Path(cfg.work_dir)
        self.out = self.work / mode
        self.names = json.loads((self.work / "names.json").read_text())
        self.idx = json.loads((self.work / f"idx_{mode}.json").read_text())
        self.shard = {s: self.idx[s][rank::RANKS] for s in ("val", "test")}
        self.cache = open_cache(self.work / "cache")
        self.dev = torch.device("cuda")
        self.sync = Sync(self.out / "sync", rank, self.log)
        self.best = Path(cfg.best_models_dir)
        self.dirs = {k: self.out / k for k in dirs}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.res, self.tm = {"rank": rank}, {}

    def groups(self):
        pos = {s: {int(k): v for k, v in g.items() if int(k) in set(self.idx[s])}
               for s, g in json.loads((self.work / "groups.json").read_text()).items()}
        return pos, {s: members_of(pos[s]) for s in pos}

    def quick_idx(self):
        v = self.idx["val"]
        return v[:2] if self.smoke else v[::max(1, len(v) // self.cfg.quick_eval_images)][:self.cfg.quick_eval_images]

    def phase(self, name, n, fn):
        t = time.time()
        r = fn()
        self.tm[name] = (time.time() - t) / max(n, 1)
        return r

    def finish(self, log_keys):
        (self.out / f"result_{self.rank}.json").write_text(json.dumps(self.res))
        self.log(f"worker finished: {json.dumps({k: self.res[k] for k in log_keys})}")
