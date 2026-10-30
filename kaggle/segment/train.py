import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from preprocess import CACHE, build_cache, file_lists, get_data_dir

KAGGLE = Path("/kaggle/working").exists()
CODE = Path("/tmp/seg") if KAGGLE else Path(__file__).resolve().parent
BACKBONE_SRC = ("https://raw.githubusercontent.com/facebookresearch/sapiens2/7e5bae88456ac418ff0e58e74106c9fe192055d4/"
                "sapiens/backbones/standalone/sapiens2.py")
if KAGGLE and not (CODE / "sapiens2.py").exists():
    urllib.request.urlretrieve(BACKBONE_SRC, CODE / "sapiens2.py")
sys.path.insert(0, str(CODE))
from engine import N_CLS, NAMES, confusion, miou, valid_from_image

OUT = Path("/kaggle/working") if KAGGLE else Path(".")
RUNS = Path("/tmp/seg_runs")
VERSION = "v3"
DEADLINE_MIN = 650
CKPT = ("facebook/sapiens2-seg-0.4b", "sapiens2_0.4b_seg.safetensors")
CFG = {"img": 448, "frame": 640, "bs": 12, "lr": 2e-4, "layer_decay": 0.85, "wd": 0.1, "warmup": 200, "clip": 1.0,
       "freeze": 8, "ckpt_blocks": 12, "probe": [10, 40], "lovasz": 1.0, "zoom": [0.75, 1.25], "rot": 15.0,
       "p_rot": 0.5, "p_fg": 0.7, "p_jitter": 0.8, "jitter": 0.2, "eval_every": 5, "post_min": 20,
       "infer_margin": 1.3, "tta": [288, 224, 180], "seed": 0}
SMOKE = {"rows": [64, 32, 16], "steps": 6}
GRID = (0, 16, 32, 64, 128, 256, 512, 1024)
QGRID = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
ARMS = ("ms", "ss")
SEED = 0


def launch(cfg, name):
    run = RUNS / name
    run.mkdir(parents=True, exist_ok=True)
    n_gpu = torch.cuda.device_count()
    log_path = run / "run.log"
    with open(log_path, "w") as log:
        p = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpu}",
                              str(CODE / "worker.py"), "--cache", str(CACHE), "--ckpt", cfg["ckpt"], "--out", str(run),
                              "--cfg", json.dumps(cfg)], stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                                      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
        seen = 0
        while p.poll() is None:
            time.sleep(5)
            lines = log_path.read_text().splitlines()
            for line in lines[seen:]:
                print(line)
            seen = len(lines)
    lines = log_path.read_text().splitlines()
    for line in lines[seen:]:
        print(line)
    if p.returncode != 0:
        raise RuntimeError(f"torchrun failed (exit {p.returncode}):\n" + "\n".join(lines[-40:]))
    return run, json.loads((run / "hist.json").read_text())


def merge(run, kind, n, world):
    parts = [np.load(run / f"{kind}_{r}.npy", mmap_mode="r") for r in range(world)]
    out = np.empty((n,) + parts[0].shape[1:], parts[0].dtype)
    for r, a in enumerate(parts):
        out[r::world] = a
    return out


def apply_pp(top3, valid, drop):
    p = top3[:, 0].long()
    for k in (1, 2):
        p = torch.where(drop.gather(1, p.flatten(1)).view_as(p), top3[:, k].long(), p)
    return torch.where(drop.gather(1, p.flatten(1)).view_as(p) | ~valid, 0, p)


def drop_of(area, conf, a, q):
    return (area > 0) & ((area < a) | (conf < q))


def scores(top3, valid, gt, drop, bs=250):
    u, g = [], []
    for s in range(0, len(top3), bs):
        x, y = miou(confusion(apply_pp(top3[s:s + bs], valid[s:s + bs], drop[s:s + bs]), gt[s:s + bs]))
        u.append(x)
        g.append(y)
    return torch.cat(u), torch.cat(g)


def tune(top3, valid, gt, area, conf):
    dev = top3.device
    a, q = torch.zeros(N_CLS, device=dev), torch.zeros(N_CLS, device=dev)
    cur = scores(top3, valid, gt, drop_of(area, conf, a, q))[0]
    for _ in range(2):
        for c in range(1, N_CLS):
            rows = (area[:, c] > 0).nonzero().squeeze(1)
            if len(rows) == 0:
                continue
            t3, v, g, ar, cf = top3[rows], valid[rows], gt[rows], area[rows], conf[rows]
            base, best = cur[rows].sum().item(), None
            for ta in GRID:
                for tq in QGRID:
                    a2, q2 = a.clone(), q.clone()
                    a2[c], q2[c] = ta, tq
                    new = scores(t3, v, g, drop_of(ar, cf, a2, q2))[0]
                    gain = new.sum().item() - base
                    if gain > 1e-9 and (best is None or gain > best[0]):
                        best = (gain, ta, tq, new)
            if best is not None:
                a[c], q[c] = best[1], best[2]
                cur[rows] = best[3]
    return a, q


def class_table(top3, valid, gt):
    agg = torch.zeros(5, N_CLS, dtype=torch.float64, device=top3.device)
    none = torch.zeros(len(top3), N_CLS, dtype=torch.bool, device=top3.device)
    for s in range(0, len(top3), 250):
        cm = confusion(apply_pp(top3[s:s + 250], valid[s:s + 250], none[s:s + 250]), gt[s:s + 250]).double()
        inter, a_gt, a_pr = cm.diagonal(dim1=1, dim2=2), cm.sum(2), cm.sum(1)
        agg += torch.stack([inter.sum(0), (a_gt + a_pr - inter).sum(0), (a_gt > 0).sum(0),
                            ((a_pr > 0) & (a_gt == 0)).sum(0), ((a_gt > 0) & (a_pr == 0)).sum(0)])
    n = len(top3)
    tab = pd.DataFrame({"iou": (agg[0] / agg[1].clamp(min=1)).cpu().numpy(), "gt_img": (agg[2] / n).cpu().numpy(),
                        "fp_img": (agg[3] / n).cpu().numpy(), "fn_img": (agg[4] / n).cpu().numpy()}, index=NAMES)
    print(tab.round(3).to_string())
    return tab


def write_zip(path, pred):
    t0 = time.time()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        with z.open("submission.csv", "w", force_zip64=True) as f:
            f.write(b"index,pred_mask\n")
            for i, a in enumerate(pred):
                f.write(f'{i},"{json.dumps(a.tolist(), separators=(",", ":"))}"\n'.encode())
    print(f"  wrote {path} ({path.stat().st_size / 2 ** 20:.0f} MiB, {len(pred)} rows) in {time.time() - t0:.0f}s")


def finish(run, hist, names, dest, keep_model):
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world = hist["world"]
    Xv, Yv, Xt = (np.load(CACHE / f"{n}.npy", mmap_mode="r") for n in ("val_x", "val_y", "test_x"))
    n_va = sum(len(np.load(run / f"val_ss_top3_{r}.npy", mmap_mode="r")) for r in range(world))
    n_te = sum(len(np.load(run / f"test_ss_top3_{r}.npy", mmap_mode="r")) for r in range(world))
    gt = torch.from_numpy(np.array(Yv[:n_va])).to(dev)
    valid = valid_from_image(torch.from_numpy(np.array(Xv[:n_va])).to(dev))
    print(f"  letterbox mask from pixels vs val ignore label: {(valid == (gt != 255)).float().mean().item():.5f} agreement")
    none = torch.zeros(n_va, N_CLS, dtype=torch.bool, device=dev)
    half = torch.from_numpy(np.random.default_rng(SEED).permutation(n_va) < n_va // 2).to(dev)
    res, npz = {}, {}
    for arm in ARMS:
        vt = torch.from_numpy(merge(run, f"val_{arm}_top3", n_va, world)).to(dev)
        vs = torch.from_numpy(merge(run, f"val_{arm}_stats", n_va, world)).to(dev)
        print(f"  [{arm}] raw val classes:")
        tab = class_table(vt, valid, gt)
        area, conf = vs[..., 0], vs[..., 1]
        raw_u, raw_g = scores(vt, valid, gt, none)
        u, g = torch.empty_like(raw_u), torch.empty_like(raw_g)
        for fit, ev in ((half, ~half), (~half, half)):
            a, q = tune(vt[fit], valid[fit], gt[fit], area[fit], conf[fit])
            u[ev], g[ev] = scores(vt[ev], valid[ev], gt[ev], drop_of(area[ev], conf[ev], a, q))
        res[arm] = {"raw_u": raw_u, "raw_g": raw_g, "u": u, "g": g, "tab": tab,
                    "thr": tune(vt, valid, gt, area, conf)}
        npz[f"top1_{arm}"], npz[f"stats_{arm}"] = vt[:, 0].cpu().numpy(), vs.cpu().numpy()
        del vt, vs

    def diff(x, y):
        d = (x - y).cpu().numpy()
        return f"{d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}"

    for arm, r in res.items():
        a, q = r["thr"]
        print(f"  [{arm}] val per-image mIoU  raw union {r['raw_u'].mean():.4f} gt-only {r['raw_g'].mean():.4f};  "
              f"pp (cross-fitted) union {r['u'].mean():.4f}, vs raw {diff(r['u'], r['raw_u'])}")
        print(f"  [{arm}] thresholds (all val): " + ", ".join(
            f"{NAMES[c]} {int(a[c])}/{q[c]:.1f}" for c in range(1, N_CLS) if a[c] > 0 or q[c] > 0))
    print(f"  ms - ss: raw {diff(res['ms']['raw_u'], res['ss']['raw_u'])}, pp {diff(res['ms']['u'], res['ss']['u'])}")
    best = max(ARMS, key=lambda k: res[k]["u"].mean().item())
    (dest / "extra").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": names["val"][:n_va], **{f"{k}_{arm}": r[k].cpu().numpy() for arm, r in res.items()
                                                  for k in ("raw_u", "raw_g", "u", "g")}}).to_csv(
        dest / "extra" / f"val_{VERSION}.csv", index=False)
    np.savez_compressed(dest / "extra" / f"val_pred_{VERSION}.npz", **npz)
    (dest / "extra" / f"run_{VERSION}.json").write_text(json.dumps(
        {**hist, "best_arm": best,
         **{arm: {"classes": r["tab"].round(4).to_dict(), "val_union_raw": r["raw_u"].mean().item(),
                  "val_union_pp": r["u"].mean().item(), "thresholds": [t.tolist() for t in r["thr"]]}
            for arm, r in res.items()}}, indent=1))
    del gt, valid
    for arm in ARMS:
        tt, ts = merge(run, f"test_{arm}_top3", n_te, world), merge(run, f"test_{arm}_stats", n_te, world)
        pred = np.empty((n_te, 320, 320), np.uint8)
        for s in range(0, n_te, 250):
            e = min(s + 250, n_te)
            st = torch.from_numpy(ts[s:e]).to(dev)
            v = valid_from_image(torch.from_numpy(np.array(Xt[s:e])).to(dev))
            pred[s:e] = apply_pp(torch.from_numpy(tt[s:e]).to(dev), v,
                                 drop_of(st[..., 0], st[..., 1], *res[arm]["thr"])).cpu().numpy()
        del tt, ts
        write_zip(dest / "submission.zip" if arm == best else dest / "extra" / f"{VERSION}_{arm}.zip", pred)
    print(f"  post-processing took {time.time() - t0:.0f}s")
    if keep_model:
        shutil.move(run / "model.pt", dest / "extra" / f"model_{VERSION}.pt")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def main():
    t0 = time.time()
    d = get_data_dir()
    names = file_lists(d)
    print(f"data {d}: " + ", ".join(f"{k} {len(v)}" for k, v in names.items()))
    with ThreadPoolExecutor(1) as pool:
        download = pool.submit(hf_hub_download, *CKPT, local_dir="/tmp/seg_ckpt")
        build_cache(d, names)
        ckpt = download.result()
    print(f"checkpoint {ckpt} ({Path(ckpt).stat().st_size / 2 ** 30:.2f} GiB), {time.time() - t0:.0f}s so far")
    cfg = {**CFG, "ckpt": ckpt, "deadline": t0 + DEADLINE_MIN * 60}
    run, hist = launch({**cfg, **SMOKE}, "smoke")
    finish(run, hist, names, run / "out", False)
    print(f"smoke ok, {time.time() - t0:.0f}s so far")
    run, hist = launch(cfg, "full")
    best = finish(run, hist, names, OUT, True)
    other = next(a for a in ARMS if a != best)
    print(f"total {(time.time() - t0) / 60:.0f} min (deadline {DEADLINE_MIN})")
    print(f"SUBMIT: submission.zip ({VERSION}: Sapiens2-seg-0.4B, 2x frame, 448 crops, {best} window + flip TTA "
          f"(ms = source windows {CFG['tta']}, ss = {CFG['tta'][1]} only), area+confidence pp); "
          f"extra/{VERSION}_{other}.zip = the {other} arm, same pp")


if __name__ == "__main__":
    main()
