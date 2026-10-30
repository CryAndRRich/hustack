import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from preprocess import find, load_images, load_tables

VERSION = "v2"
SEED = 42
FOLDS, VAL_FOLD = 5, 0
GATE_SE = 2.0
AGE_RANGE = (1.0, 100.0)
AGE_BINS = (0, 20, 30, 40, 50, 60, 70, 101)
STRATA = (20, 25, 30, 35, 40, 50, 60, 70)
INNER = 0.1
MLP = {"hidden": 512, "lr": 1e-3, "wd": 1e-4, "batch": 512, "patience": 15, "seeds": 3, "epochs": 300}
CFG = {"arch": "vit_large_patch14_clip_quickgelu_224.openai", "img": 224, "mean": list(OPENAI_CLIP_MEAN),
       "std": list(OPENAI_CLIP_STD), "epochs": 5, "warmup": 0.3, "bs": 64, "lr": 3e-5, "head_lr_mult": 10.0,
       "layer_decay": 0.8, "wd": 0.05, "drop_path": 0.1, "scale": [0.8, 1.0], "ratio": [0.9, 1.1], "rot": 10.0,
       "p_jitter": 0.5, "jitter": 0.2, "seed": SEED, "age_range": list(AGE_RANGE), "grad_ckpt": True}
SMOKE = {"epochs": 1, "rows": (512, 256, 64)}
KAGGLE = Path("/kaggle/working").exists()
CODE = Path("/tmp/age") if KAGGLE else Path(__file__).resolve().parent
OUT = Path("/kaggle/working") if KAGGLE else Path(".")
RUNS = Path("/tmp/age_runs")


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def subject_groups(names, images):
    n = len(names)
    nid = pd.factorize(names)[0]
    hid = pd.factorize(np.array([hashlib.md5(a.tobytes()).hexdigest() for a in images]))[0]
    k1, k2 = nid.max() + 1, hid.max() + 1
    rows, cols = np.r_[np.arange(n), np.arange(n)], np.r_[n + nid, n + k1 + hid]
    g = coo_matrix((np.ones(2 * n), (rows, cols)), shape=(n + k1 + k2,) * 2)
    return connected_components(g, directed=False)[1][:n]


def split(face, groups):
    sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    va = list(sgkf.split(face, np.digitize(face.age, STRATA), groups))[VAL_FOLD][1]
    tr = np.setdiff1d(np.flatnonzero(face.clean.to_numpy()), va)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    mon_pos, rep_pos = next(gss.split(va, groups=groups[va]))
    return tr, va, np.sort(mon_pos), np.sort(rep_pos)


def _init(gen, seeds, fan_in, fan_out, device):
    k = fan_in ** -0.5
    w = (torch.rand(seeds, fan_in, fan_out, generator=gen, device=device) * 2 - 1) * k
    b = (torch.rand(seeds, 1, fan_out, generator=gen, device=device) * 2 - 1) * k
    return [w.requires_grad_(), b.requires_grad_()]


def _forward(p, x):
    h = torch.relu(torch.baddbmm(p[1], x, p[0]))
    h = torch.relu(torch.baddbmm(p[3], h, p[2]))
    return torch.baddbmm(p[5], h, p[4]).squeeze(-1)


def mlp(X, y, fit, es, outs, device, seed):
    S, B = MLP["seeds"], MLP["batch"]
    tr = np.r_[fit, es]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    ym, ys = y[fit].mean(), y[fit].std()
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    d, h = X.shape[1], MLP["hidden"]
    p = _init(gen, S, d, h, device) + _init(gen, S, h, h, device) + _init(gen, S, h, 1, device)
    opt = torch.optim.AdamW(p, lr=MLP["lr"], weight_decay=MLP["wd"])
    fit_t, es_t = torch.tensor(fit, device=device), torch.tensor(es, device=device)
    best = torch.full((S,), float("inf"), device=device)
    best_p = [q.detach().clone() for q in p]
    stale = torch.zeros(S, dtype=torch.long, device=device)
    for _ in range(MLP["epochs"]):
        order = torch.argsort(torch.rand(S, len(fit), generator=gen, device=device), 1)
        for s in range(0, len(fit), B):
            idx = fit_t[order[:, s:s + B]]
            loss = ((_forward(p, Xt[idx]) - yt[idx]) ** 2).mean(1).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            val = ((_forward(p, Xt[es_t].expand(S, -1, -1)) - yt[es_t]) ** 2).mean(1)
            better = val < best
            best = torch.where(better, val, best)
            for q, bq in zip(p, best_p):
                bq[better] = q.detach()[better]
            stale = torch.where(better, 0, stale + 1)
        if bool((stale >= MLP["patience"]).all()):
            break
    with torch.no_grad():
        return [_forward(best_p, Xt[torch.tensor(o, device=device)].expand(S, -1, -1)).mean(0).cpu().numpy() * ys + ym
                for o in outs]


def reference(face, judge_df, groups, tr, va, device):
    p = find("emb_clip.npy")
    assert p is not None, "attach the dataset age-emb (V1's features)"
    rows = pd.read_csv(p.parent / "emb_rows.csv")
    assert (rows.ID.to_numpy() == np.r_[face.ID, judge_df.ID]).all() and (rows.part == "judge").sum() == len(judge_df)
    X = np.load(p).astype(np.float32)
    y = np.r_[face.age.to_numpy(dtype=float), np.zeros(len(judge_df))]
    gss = GroupShuffleSplit(n_splits=1, test_size=INNER, random_state=SEED)
    fit, es = next(gss.split(tr, groups=groups[tr]))
    pv, pj = mlp(X, y, tr[fit], tr[es], [va, np.arange(len(face), len(X))], device, SEED)
    return np.clip(pv, *AGE_RANGE), np.clip(pj, *AGE_RANGE)


def launch(cfg, arrays, name):
    run = RUNS / name
    run.mkdir(parents=True, exist_ok=True)
    np.savez(run / "data.npz", **arrays)
    n_gpu = torch.cuda.device_count()
    log_path = run / "run.log"
    with open(log_path, "w") as log:
        p = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpu}",
                              str(CODE / "worker.py"), "--data", str(run / "data.npz"), "--out", str(run),
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
    d = np.load(run / "run.npz")
    return d["va"], d["te"], json.loads((run / "hist.json").read_text())


def finetune(cfg, images, jimages, y, tr, va, mon, smoke):
    arrays = {"X_tr": images[tr], "y_tr": y[tr].astype(np.float32), "X_va": images[va],
              "y_va": y[va].astype(np.float32), "mon": mon, "X_te": jimages}
    if smoke:
        n_tr, n_va, n_te = SMOKE["rows"]
        keep = mon[mon < n_va]
        arrays = {"X_tr": arrays["X_tr"][:n_tr], "y_tr": arrays["y_tr"][:n_tr], "X_va": arrays["X_va"][:n_va],
                  "y_va": arrays["y_va"][:n_va], "mon": keep, "X_te": jimages[:n_te]}
        cfg = {**cfg, "epochs": SMOKE["epochs"]}
    t0 = time.time()
    pv, pt, hist = launch(cfg, arrays, "smoke" if smoke else "full")
    assert len(pv) == len(arrays["X_va"]) and len(pt) == len(arrays["X_te"]) and np.isfinite(np.r_[pv, pt]).all()
    print(f"  {'smoke' if smoke else 'full'} fine-tune done in {time.time() - t0:.0f}s")
    return np.clip(pv, *AGE_RANGE), np.clip(pt, *AGE_RANGE), hist


def evaluate(y, clean, rep, mon, preds):
    rc, mc = rep[clean[rep]], mon[clean[mon]]
    table = pd.DataFrame({a: {"rmse_report": rmse(y[rc], p[rc]), "rmse_report_all": rmse(y[rep], p[rep]),
                              "mae_report": float(np.mean(np.abs(y[rc] - p[rc]))), "rmse_monitor": rmse(y[mc], p[mc])}
                          for a, p in preds.items()}).T
    d = (preds["finetune"][rc] - y[rc]) ** 2 - (preds["reference"][rc] - y[rc]) ** 2
    se = d.std(ddof=1) / np.sqrt(len(d))
    gate = {"mse_delta": float(d.mean()), "se": float(se), "passed": bool(d.mean() < -GATE_SE * se), "rows": int(len(d))}
    err = preds["finetune"][rc] - y[rc]
    by_bin = pd.DataFrame({"bin": pd.cut(y[rc], AGE_BINS, right=False), "err": err, "se": err ** 2}).groupby(
        "bin", observed=True).agg(n=("err", "size"), rmse=("se", lambda s: float(np.sqrt(s.mean()))), bias=("err", "mean"))
    return table, gate, by_bin


def write(face, judge_df, va, half, preds, jpreds, table, gate, by_bin, hist):
    extra = OUT / "extra"
    extra.mkdir(exist_ok=True)
    f = face.iloc[va]
    pd.DataFrame({"ID": f.ID, "half": half, "age": f.age, "clean": f.clean, **{a: p.round(3) for a, p in preds.items()}}
                 ).to_csv(extra / f"val_{VERSION}.csv", index=False)
    pd.DataFrame({"ID": judge_df.ID, **{a: p.round(3) for a, p in jpreds.items()}}).to_csv(
        extra / f"judge_{VERSION}.csv", index=False)
    (extra / f"summary_{VERSION}.json").write_text(json.dumps(
        {"version": VERSION, "cfg": CFG, "gate": gate, "table": table.round(5).to_dict("index"), "hist": hist,
         "by_bin": by_bin.round(4).reset_index().astype(str).to_dict("records"),
         "versions": {"torch": torch.__version__, "timm": timm.__version__}}, indent=1))
    if not gate["passed"]:
        print(f"SUBMIT: none — the fine-tune did not beat V1's head by {GATE_SE} SE; keep V1 (releases/v1)")
        return
    sub = pd.DataFrame({"ID": judge_df.ID, "age": jpreds["finetune"]})
    assert len(sub) == len(judge_df) and sub.ID.is_unique and np.isfinite(sub.age).all()
    assert sub.age.between(*AGE_RANGE).all()
    sub.to_csv(OUT / "submission.csv", index=False)
    print(f"SUBMIT: {OUT / 'submission.csv'}  (finetune, report rmse {table.loc['finetune', 'rmse_report']:.4f} vs "
          f"V1 head {table.loc['reference', 'rmse_report']:.4f})")


def main():
    assert torch.cuda.is_available(), "turn on the GPU (T4 x2)"
    print(f"torch {torch.__version__}, timm {timm.__version__}, GPUs: "
          f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    d, lab, judge_df, _ = load_tables()
    t0 = time.time()
    images = load_images(d, "train", lab, np.flatnonzero(lab.face.to_numpy()))
    jimages = load_images(d, "judge", judge_df, np.arange(len(judge_df)))
    print(f"images {images.shape}, judge {jimages.shape} in {time.time() - t0:.0f}s")
    face = lab[lab.face].reset_index(drop=True)
    groups = subject_groups(face.name.to_numpy(), images)
    tr, va, mon, rep = split(face, groups)
    y, clean = face.age.to_numpy(dtype=float), face.clean.to_numpy()
    print(f"groups {len(np.unique(groups))} (names {face.name.nunique()}); train {len(tr)}, val {len(va)} "
          f"(monitor {len(mon)}, report {len(rep)})")
    t0 = time.time()
    ref_va, ref_j = reference(face, judge_df, groups, tr, va, torch.device("cuda:0"))
    torch.cuda.empty_cache()
    print(f"reference (V1 head on the same rows): val rmse {rmse(y[va][clean[va]], ref_va[clean[va]]):.4f} "
          f"in {time.time() - t0:.0f}s")
    mon_clean = mon[clean[va][mon]]
    finetune(CFG, images, jimages, y, tr, va, mon_clean, smoke=True)
    ft_va, ft_j, hist = finetune(CFG, images, jimages, y, tr, va, mon_clean, smoke=False)
    preds = {"finetune": ft_va, "reference": ref_va, "blend": (ft_va + ref_va) / 2}
    jpreds = {"finetune": ft_j, "reference": ref_j, "blend": (ft_j + ref_j) / 2}
    yv, cv = y[va], clean[va]
    table, gate, by_bin = evaluate(yv, cv, rep, mon, preds)
    print(table.round(4).to_string())
    print(f"gate (finetune vs V1 head, clean report rows): mse delta {gate['mse_delta']:+.3f}, se {gate['se']:.3f} "
          f"-> {'PASS' if gate['passed'] else 'fail'}")
    print(f"finetune by age (report):\n{by_bin.round(3).to_string()}")
    half = np.where(np.isin(np.arange(len(va)), mon), "monitor", "report")
    write(face, judge_df, va, half, preds, jpreds, table, gate, by_bin, hist)


if __name__ == "__main__":
    main()
