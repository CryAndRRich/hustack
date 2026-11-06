import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from preprocess import (FIRE_FLOOR, TARGETS, build_features, ext_features, ext_tables, extract_chips, find_data_dir,
                        load_ext, load_tables, work_dir)

warnings.filterwarnings("ignore", message="X does not have valid feature names")
VERSION = "V6"
T_START = time.time()
DEADLINE_S = 70 * 60
REPEATS, FOLDS = 3, 5
WORK = Path("/tmp/obscura")
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=40,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0, verbose=-1, n_jobs=4)
POOL_NAME = {"t_aai": "aai", "t_no2": "no2", "t_co": "co"}
FI = TARGETS.index("t_firms")
AI = TARGETS.index("t_aai")
HI_ELEV = 100
RIDGE_C = (0.03, 0.3, 3.0, 30.0, 300.0)
IMG_HEADS = ["ch4", "no2", "co", "aai", "fire", "firms"]
CNN_HEADS = ["fire", "aai", "no2", "co"]


def site_folds(sites, n_folds, seed):
    rng = np.random.default_rng(seed)
    counts = pd.Series(sites).value_counts()
    order = rng.permutation(counts.index.to_numpy())
    load, fold_of = np.zeros(n_folds), {}
    for s in order:
        f = int(np.argmin(load + rng.random(n_folds) * 1e-6))
        fold_of[s] = f
        load[f] += counts[s]
    return np.array([fold_of[s] for s in sites])


def fit_predict(t, X, y, X_eval, params, seed):
    p = dict(params, random_state=seed)
    keep = ~np.isnan(y)
    if keep.sum() < 30:
        return [(np.zeros(len(x)), None, None) for x in X_eval]
    if t != "t_firms":
        m = lgb.LGBMRegressor(**p).fit(X[keep], y[keep])
        return [(m.predict(x), None, None) for x in X_eval]
    fire = y > FIRE_FLOOR + 0.01
    clf = lgb.LGBMClassifier(**p).fit(X, fire.astype(int))
    if fire.sum() >= 30:
        reg = lgb.LGBMRegressor(**p).fit(X[fire], y[fire])
        mus = [reg.predict(x) for x in X_eval]
    else:
        mus = [np.full(len(x), y[fire].mean() if fire.any() else FIRE_FLOOR) for x in X_eval]
    out = []
    for x, mu in zip(X_eval, mus):
        pf = clf.predict_proba(x)[:, 1]
        out.append((FIRE_FLOOR + pf * (mu - FIRE_FLOOR), pf, mu))
    return out


def run_arm(name, X, Y, X_test, folds, params):
    t0 = time.time()
    n, nt = len(X), len(X_test)
    oof, test = np.zeros((len(folds), n, 5)), np.zeros((nt, 5))
    fire_oof, fire_test = np.zeros((len(folds), n, 2)), np.zeros((nt, 2))
    k = 0
    for r, fold in enumerate(folds):
        for f in range(fold.max() + 1):
            tr_i, va_i = fold != f, fold == f
            for j, t in enumerate(TARGETS):
                (pv, fv, mv), (pt, ft, mt) = fit_predict(t, X[tr_i], Y[tr_i, j], [X[va_i], X_test], params, 100 * r + f)
                oof[r, va_i, j] = pv
                test[:, j] += pt
                if t == "t_firms":
                    fire_oof[r, va_i] = np.c_[fv, mv]
                    fire_test += np.c_[ft, mt]
            k += 1
    print(f"arm {name}: {k} fold fits x 5 targets in {time.time() - t0:.0f}s")
    return {"oof": oof, "test": test / k, "fire_oof": fire_oof, "fire_test": fire_test / k}


def affine_map(name, ext_pred, y, rows, pool):
    ye, pe = pool["y"].get(name), pool["oof"].get(name)
    if ye is None:
        return None
    ke = ~np.isnan(ye)
    b2, c2 = np.polyfit(pe[ke], ye[ke], 1)
    rows = rows & ~np.isnan(y)
    b1, c1 = np.polyfit(ext_pred[rows], y[rows], 1)
    if b1 <= 0 or b2 <= 0:
        print(f"map {name}: slope b1 {b1:.3g} / b2 {b2:.3g} not positive")
        return None
    return b1 / b2, c1, c2


def pooled_target(t, Xc, y, Xct, ext_pred, pool, folds, params):
    name = POOL_NAME[t]
    ye = pool["y"].get(name)
    if ye is None:
        return None
    ke = ~np.isnan(ye)
    Xe, ye = pool["X"][ke], ye[ke]
    oof, test, k, ks = np.zeros((len(folds), len(Xc))), np.zeros(len(Xct)), 0, []
    for r, fold in enumerate(folds):
        for f in range(fold.max() + 1):
            mp = affine_map(name, ext_pred, y, fold != f, pool)
            if mp is None:
                return None
            kk, c1, c2 = mp
            tr = (fold != f) & ~np.isnan(y)
            w = np.r_[np.ones(tr.sum()), np.full(len(ye), tr.sum() / len(ye))]
            m = lgb.LGBMRegressor(**dict(params, random_state=100 * r + f)).fit(
                np.vstack([Xc[tr], Xe]), np.r_[y[tr], c1 + kk * (ye - c2)], sample_weight=w)
            oof[r, fold == f] = m.predict(Xc[fold == f])
            test += m.predict(Xct)
            ks.append(kk)
            k += 1
    print(f"pool {t}: {len(ye)} external rows, raw->z scale {np.mean(ks):.4g} (fold range {min(ks):.4g}..{max(ks):.4g})")
    return oof, test / k


def ridge_path(Z, y, Zev, cs):
    m, mu = Z.mean(0), y.mean()
    U, s, Vt = np.linalg.svd(Z - m, full_matrices=False)
    uy = U.T @ (y - mu)
    out = []
    for c in cs:
        coef = Vt.T @ (s / (s ** 2 + c * len(y)) * uy)
        out.append([(e - m) @ coef + mu for e in Zev])
    return out


def find_chips(data_dir):
    for root in ("/kaggle/input", str(data_dir / "external" / "chips")):
        hits = sorted(Path(p) / f for p, _, files in os.walk(root) for f in files if f.startswith("shard_") and f.endswith(".npz"))
        if hits:
            print(f"external chips: {len(hits)} shards under {hits[0].parent}")
            return hits
    print("external chips not found: image heads and CNN pretraining skipped")
    return []


def ext_chip_labels(shards, e, y):
    keys = pd.concat([pd.DataFrame({"fac_id": z["fac_id"], "date": z["date"].astype(str)}) for z in map(np.load, shards)])
    pos = pd.Series(np.arange(len(e)), index=pd.MultiIndex.from_arrays([e.fac_id.to_numpy(), e.date.astype(str).to_numpy()]))
    idx = pos.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
    assert not np.isnan(idx).any(), "chip shards do not match ext_rows.csv"
    idx = idx.astype(int)
    lab = [y.fire.to_numpy(float)[idx]]
    for name in CNN_HEADS[1:]:
        v = y[name].to_numpy(float)[idx]
        lo, hi = np.nanpercentile(v, [0.5, 99.5])
        v = np.clip(v, lo, hi)
        lab.append((v - np.nanmean(v)) / np.nanstd(v))
    hold = (keys.fac_id.to_numpy() * 2654435761 % 10) == 0
    return idx, np.stack(lab, 1).astype(np.float32), hold


def image_heads(Zc, Ze, y, groups):
    from sklearn.model_selection import GroupKFold
    out = {}
    for name in IMG_HEADS:
        v = y[name].to_numpy(float)
        k = ~np.isnan(v)
        lo, hi = np.percentile(v[k], [0.5, 99.5])
        Zk, v, g = Ze[k], np.clip(v[k], lo, hi), groups[k]
        cv = np.zeros((len(RIDGE_C), len(v)))
        for a, b in GroupKFold(3).split(Zk, groups=g):
            for ci, (p,) in enumerate(ridge_path(Zk[a], v[a], [Zk[b]], RIDGE_C)):
                cv[ci, b] = p
        best = int(np.argmin(((cv - v) ** 2).mean(1)))
        out[f"img_{name}"] = ridge_path(Zk, v, [Zc], [RIDGE_C[best]])[0][0]
        print(f"  image head {name}: {k.sum()} external chips, c {RIDGE_C[best]}, facility-grouped corr {np.corrcoef(cv[best], v)[0, 1]:.3f}")
    return out


def loso_far(Z, sites, q=0.75):
    d = np.zeros(len(Z))
    for u in np.unique(sites):
        m = sites == u
        dd = (Z[m] ** 2).sum(1)[:, None] + (Z[~m] ** 2).sum(1)[None] - 2 * Z[m] @ Z[~m].T
        d[m] = np.sort(np.sqrt(np.clip(dd, 0, None)), 1)[:, :5].mean(1)
    return d > np.quantile(d, q), d


def rmse_cols(Y, P):
    out = []
    for j in range(Y.shape[1]):
        k = ~np.isnan(Y[:, j])
        out.append(np.sqrt(np.mean((Y[k, j] - P[k, j]) ** 2)) if k.any() else np.nan)
    return np.array(out)


def alphas(Y, oof):
    P = oof.mean(0)
    a = []
    for j in range(Y.shape[1]):
        k = ~np.isnan(Y[:, j])
        a.append(np.clip((Y[k, j] * P[k, j]).sum() / max((P[k, j] ** 2).sum(), 1e-12), 0, 1.2))
    return np.array(a)


def score(a, Y, folds):
    a["alpha"] = alphas(Y, a["oof"])
    a["fts"] = np.array([rmse_cols(Y[fold == f], a["oof"][r][fold == f] * a["alpha"])
                         for r, fold in enumerate(folds) for f in range(fold.max() + 1)])
    a["folds_score"] = np.nanmean(a["fts"], axis=1)


def paired(d, label):
    d = d[~np.isnan(d)]
    tt = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
    print(f"{label}: mean {d.mean():+.4f}, t {tt:+.2f}, better in {(d < 0).sum()}/{len(d)} folds")
    return d.mean(), tt


def better(j, d, label):
    m, tt = paired(d, label)
    return m < 0 and tt < -1


def subset_rmse(a, Y, rows):
    return rmse_cols(Y[rows], (a["oof"].mean(0) * a["alpha"])[rows])


def merge(base, cand, js, label, rule):
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    took = []
    for j in js:
        if rule(j, cand["fts"][:, j] - base["fts"][:, j], f"{label} {TARGETS[j]}"):
            out["oof"][..., j], out["test"][:, j] = cand["oof"][..., j], cand["test"][:, j]
            if j == FI:
                out["fire_oof"], out["fire_test"] = cand["fire_oof"].copy(), cand["fire_test"].copy()
            took.append(j)
    print(f"{label}: took {[TARGETS[j] for j in took]}")
    return out, took


def subset_gains(Y, oof, alpha, sites, k, draws=2000, seed=0):
    P = oof.mean(0) * alpha
    us = np.unique(sites)
    idx = {u: np.where(sites == u)[0] for u in us}
    rng = np.random.default_rng(seed)
    G = np.full((draws, 5), np.nan)
    for b in range(draws):
        rows = np.concatenate([idx[u] for u in rng.choice(us, min(k, len(us)), replace=False)])
        for j in range(5):
            y, p = Y[rows, j], P[rows, j]
            m = ~np.isnan(y)
            if m.sum() > 5:
                G[b, j] = np.sqrt(np.mean((y[m] - p[m]) ** 2)) - np.sqrt(np.mean(y[m] ** 2))
    return np.nanpercentile(G, [10, 50, 90], axis=0)


def launch(script, gpu, cfg_path, log, args=()):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.Popen([sys.executable, str(WORK / script), str(cfg_path), *args], env=env,
                            stdout=open(log, "w"), stderr=subprocess.STDOUT)


class Jobs:
    PLAN = [[("embed", "embed_worker.py", ()), ("tabpfn", "icl_worker.py", ("tabpfn",))],
            [("cnn", "cnn_worker.py", ()), ("tabicl", "icl_worker.py", ("tabicl",))]]

    def __init__(self, cfg_path, run_dir, n_gpu):
        self.cfg_path, self.run_dir, self.procs = cfg_path, run_dir, {}
        plan = self.PLAN if n_gpu > 1 else [[job for q in zip(*self.PLAN) for job in q]] if n_gpu else []
        self.queues, self.cur = [list(q) for q in plan], [None] * len(plan)
        if not n_gpu:
            print("no GPU: embedding, CNN, TabPFN and TabICL arms skipped")
        self.pump()

    def pump(self):
        for g, q in enumerate(self.queues):
            if q and (self.cur[g] is None or self.procs[self.cur[g]].poll() is not None):
                name, script, args = q.pop(0)
                self.cur[g] = name
                self.procs[name] = launch(script, g, self.cfg_path, self.run_dir / f"{name}.log", args)

    def wait(self, names=None):
        while time.time() < T_START + DEADLINE_S + 60:
            self.pump()
            queued = [k for q in self.queues for k, _, _ in q]
            running = [k for k, p in self.procs.items() if p.poll() is None]
            if not [k for k in (names or running + queued) if k in running or k in queued]:
                return
            time.sleep(5)

    def wait_file(self, path, name):
        while time.time() < T_START + DEADLINE_S + 60 and not path.exists():
            self.pump()
            if name in self.procs and self.procs[name].poll() is not None:
                return
            time.sleep(5)

    def finish(self):
        self.wait()
        for k, p in self.procs.items():
            if p.poll() is None:
                p.kill()
            print(f"--- {k} worker (exit {p.poll()}), log tail:")
            print("\n".join((self.run_dir / f"{k}.log").read_text().splitlines()[-4:]))


def collect(prefix, out, folds, width):
    oof, test, k = np.zeros((len(folds), len(folds[0]), width)), None, 0
    for r, fold in enumerate(folds):
        for f in range(fold.max() + 1):
            p = out / f"{prefix}_r{r}_f{f}.npz"
            if not p.exists():
                print(f"{prefix}: missing {p.name}, arm skipped")
                return None
            z = np.load(p)
            oof[r, fold == f] = z["val"].reshape(-1, width)
            test = z["test"].reshape(-1, width) + (0 if test is None else test)
            k += 1
    return oof, test / k


def month0(F):
    ang = np.arctan2(F.month_sin.fillna(0).to_numpy(), F.month_cos.fillna(1).to_numpy())
    return (np.round(ang * 6 / np.pi).astype(int) - 1) % 12


def main(smoke):
    data_dir, out = find_data_dir(), work_dir()
    tr, te, ss = load_tables(data_dir)
    if smoke:
        keep = pd.Series(tr.site_key.unique()).sample(8, random_state=0)
        tr, te = tr[tr.site_key.isin(keep)].reset_index(drop=True), te.head(40).reset_index(drop=True)
    tag = "smoke" if smoke else "full"
    n_gpu = torch.cuda.device_count()
    print(f"{tag.upper()}: train {len(tr)} rows / {tr.site_key.nunique()} sites, test {len(te)}, GPUs {n_gpu}, "
          f"elapsed {time.time() - T_START:.0f}s")
    stats, imgs, fms = extract_chips(data_dir / "images", list(tr.sample_id) + list(te.sample_id))
    Fdf = build_features(tr, te, stats)
    ext = load_ext(data_dir)
    if ext is not None:
        efeat, pool = ext_features(Fdf, ext, n_estimators=50 if smoke else 600)
        Edf = pd.concat([Fdf, efeat], axis=1)
    X = Fdf.to_numpy(np.float32)
    n = len(tr)
    Y = tr[TARGETS].to_numpy(float)
    sites = tr.site_key.to_numpy()
    hi = tr.elevation.to_numpy() > HI_ELEV
    print(f"train rows above {HI_ELEV} m: {hi.sum()} from {tr.site_key[hi].nunique()} sites; "
          f"test rows above: {(te.elevation > HI_ELEV).mean():.2f}")
    repeats, n_folds = (1, 2) if smoke else (REPEATS, FOLDS)
    folds = [site_folds(sites, n_folds, seed=r) for r in range(repeats)]

    run_dir = WORK / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    for p in list(run_dir.glob("*.npz")) + list(run_dir.glob("*.lock")):
        p.unlink()
    Xw = Edf.to_numpy(np.float32) if ext is not None else X
    Xi = np.where(np.isnan(Xw), np.nanmedian(Xw[:n], axis=0), Xw).astype(np.float32)
    np.savez(run_dir / "data.npz", X=Xi, Y=Y, folds=np.array(folds), n_tr=n)
    np.save(run_dir / "cnn.npy", imgs)
    np.save(run_dir / "fm.npy", fms)
    np.save(run_dir / "months.npy", month0(Fdf))
    del imgs, fms
    shards = find_chips(data_dir) if ext is not None else []
    shards = shards[:1] if smoke else shards
    if shards:
        ext_y = ext_tables(ext)[1]
        ext_idx, lab, hold = ext_chip_labels(shards, ext, ext_y)
        np.savez(run_dir / "ext_y.npz", y=lab, hold=hold)
        print(f"external chip labels: {len(ext_idx)} chips, fire rate {lab[:, 0].mean():.3f}, hold-out {hold.mean():.2f}")
    cfg = {"data": str(run_dir / "data.npz"), "cnn": str(run_dir / "cnn.npy"), "fm": str(run_dir / "fm.npy"),
           "months": str(run_dir / "months.npy"), "out": str(run_dir), "floor": FIRE_FLOOR,
           "ext_chips": [str(p) for p in shards], "ext_y": str(run_dir / "ext_y.npz"),
           "ext_epochs": 1 if smoke else 20, "seeds": 1 if smoke else 2,
           "deadline": T_START + DEADLINE_S, "n_est": 1 if smoke else 4, "epochs": 1 if smoke else 15}
    cfg_path = run_dir / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    jobs = Jobs(cfg_path, run_dir, n_gpu)

    params = dict(LGB_PARAMS, n_estimators=20) if smoke else LGB_PARAMS
    arms, raw = {"B": run_arm("B", X[:n], Y, X[n:], folds, params)}, {}
    score(arms["B"], Y, folds)
    pick, feats, ext_js = "B", X, set()
    if ext is not None:
        XE = Edf.to_numpy(np.float32)
        arms["E"] = run_arm("E", XE[:n], Y, XE[n:], folds, params)
        score(arms["E"], Y, folds)
        if paired(arms["E"]["folds_score"] - arms["B"]["folds_score"], "E - B")[0] < 0:
            pick, feats = "E", XE
        base = arms[pick]
        Xs = Fdf[pool["cols"]].to_numpy(np.float32)
        P = dict(base, oof=base["oof"].copy(), test=base["test"].copy())
        js = []
        for t in POOL_NAME:
            j = TARGETS.index(t)
            res = pooled_target(t, Xs[:n], Y[:, j], Xs[n:], efeat[f"ext_{POOL_NAME[t]}"].to_numpy()[:n], pool, folds, params)
            if res is not None:
                P["oof"][..., j], P["test"][:, j] = res
                js.append(j)
        score(P, Y, folds)
        raw["P"] = P

        def pool_rule(j, d, label):
            m, tt = paired(d, label)
            if j != AI:
                return m < 0 and tt < -1
            hb, hc = subset_rmse(base, Y, hi)[j], subset_rmse(P, Y, hi)[j]
            print(f"  aai on train rows above {HI_ELEV} m: {pick} {hb:.4f}, pooled {hc:.4f}")
            return tt < 1 and hc < hb
        merged, took = merge(base, P, js, "pooled", pool_rule)
        if took:
            score(merged, Y, folds)
            arms["X"], pick, ext_js = merged, "X", set(took)

    jobs.wait(["embed"])
    emb = np.load(run_dir / "emb.npz") if (run_dir / "emb.npz").exists() else None
    names = [k for k in ("olmo", "olmo_base", "clip") if emb is not None and k in emb.files and np.isfinite(emb[k]).all()]
    far = None
    img = {}
    if names:
        Zc = np.hstack([emb[k] for k in names]).astype(np.float64)
        mu, sd = Zc[:n].mean(0), Zc[:n].std(0) + 1e-6
        Zc = (Zc - mu) / sd
        far, dist = loso_far(Zc[:n], sites)
        print(f"embedding {names} {Zc.shape}; LOSO kNN distance train q50/q75 {np.percentile(dist, [50, 75]).round(2)}")
        ee = np.load(run_dir / "emb_ext.npz") if shards and (run_dir / "emb_ext.npz").exists() else None
        if ee is not None and all(k in ee.files for k in names):
            assert (ee["fac_id"] == ext.fac_id.to_numpy()[ext_idx]).all()
            Ze = (np.hstack([ee[k] for k in names]).astype(np.float64) - mu) / sd
            img.update(image_heads(Zc, Ze, ext_y.iloc[ext_idx].reset_index(drop=True), ext.fac_id.to_numpy()[ext_idx]))
    else:
        print("embedding: no usable output, image heads skipped")
    if shards and n_gpu:
        jobs.wait_file(run_dir / "cnnext.npz", "cnn")
        if (run_dir / "cnnext.npz").exists():
            pred = np.load(run_dir / "cnnext.npz")["pred"]
            img.update({f"cnn_{h}": pred[:, k] for k, h in enumerate(CNN_HEADS)})
        else:
            print("cnn pretraining produced no features")
    if img:
        Idf = pd.DataFrame(img)
        print("image features:", list(Idf.columns), "train corr with targets:",
              {c: [round(float(pd.Series(Idf[c][:n]).corr(pd.Series(Y[:, j] if j < 4 else (Y[:, FI] > FIRE_FLOOR + 0.01).astype(float)))), 3)
                   for j in range(5)] for c in Idf.columns})
        XI = np.hstack([feats, Idf.to_numpy(np.float32)])
        I = run_arm("I", XI[:n], Y, XI[n:], folds, params)
        score(I, Y, folds)
        raw["I"] = I
        merged, took = merge(arms[pick], I, [j for j in range(5) if j not in ext_js], "I", better)
        if took:
            score(merged, Y, folds)
            arms["I"], pick = merged, "I"

    jobs.finish()
    cnn = collect("cnn", run_dir, folds, 1) if n_gpu else None
    if cnn is not None:
        base = arms[pick]
        oof, test = base["oof"].copy(), base["test"].copy()
        pf_o = 0.5 * base["fire_oof"][..., 0] + 0.5 * cnn[0][..., 0]
        pf_t = 0.5 * base["fire_test"][:, 0] + 0.5 * cnn[1][:, 0]
        oof[..., FI] = FIRE_FLOOR + pf_o * (base["fire_oof"][..., 1] - FIRE_FLOOR)
        test[:, FI] = FIRE_FLOOR + pf_t * (base["fire_test"][:, 1] - FIRE_FLOOR)
        arms["F"] = dict(base, oof=oof, test=test)
        score(arms["F"], Y, folds)
        if paired(arms["F"]["folds_score"] - base["folds_score"], f"F - {pick}")[0] < 0:
            pick = "F"
    keep = np.array([j != FI and j not in ext_js for j in range(5)])
    members = 1
    for name, prefix, label in (("T", "tab", "TabPFN"), ("C", "icl", "TabICL")):
        res = collect(prefix, run_dir, folds, 5) if n_gpu else None
        if res is None:
            continue
        base, w = arms[pick], 1 / (members + 1)
        arms[name] = dict(base, oof=np.where(keep, (1 - w) * base["oof"] + w * res[0], base["oof"]),
                          test=np.where(keep, (1 - w) * base["test"] + w * res[1], base["test"]))
        score(arms[name], Y, folds)
        if paired(arms[name]["folds_score"] - base["folds_score"], f"{name} ({label}) - {pick}")[0] < 0:
            pick, members = name, members + 1

    rows = []
    for name, a in list(arms.items()) + [(f"raw_{k}", v) for k, v in raw.items()]:
        rows.append({"arm": name, "cv": a["folds_score"].mean(), "cv_hi": np.nanmean(subset_rmse(a, Y, hi)),
                     "cv_far": np.nanmean(subset_rmse(a, Y, far)) if far is not None else np.nan,
                     **dict(zip(TARGETS, np.nanmean(a["fts"], axis=0))),
                     **{f"a_{t[2:]}": v for t, v in zip(TARGETS, a["alpha"])}})
    print(f"cv = plain fold-mean MCRMSE (gate); cv_hi / cv_far = pooled OOF MCRMSE on train rows above {HI_ELEV} m / in "
          "the farthest embedding-distance quartile (report); raw_* = candidates before the per-target gates:")
    print(pd.DataFrame(rows).set_index("arm").round(4).to_string())
    a = arms[pick]
    for k in (5, 28):
        q = subset_gains(Y, a["oof"], a["alpha"], sites, k)
        print(f"{pick} gain vs zero on random {k}-site subsets (q10/q50/q90):",
              {t[2:]: tuple(float(v) for v in q[:, j].round(3)) for j, t in enumerate(TARGETS)})

    oof_pred = a["oof"].mean(0) * a["alpha"]
    lo, hi_q = np.percentile(oof_pred, 0.5, axis=0), np.percentile(oof_pred, 99.5, axis=0)
    raw_test = a["test"] * a["alpha"]
    pred = np.clip(raw_test, lo, hi_q)
    pred[:, FI] = np.maximum(pred[:, FI], FIRE_FLOOR)
    print("clipped test rows per target:", dict(zip(TARGETS, ((raw_test < lo) | (raw_test > hi_q)).sum(0).tolist())))
    sub = pd.DataFrame(pred, columns=TARGETS)
    sub.insert(0, "sample_id", te.sample_id.to_numpy())
    assert len(sub) == len(te) and np.isfinite(sub[TARGETS].to_numpy()).all()
    if smoke:
        print(f"smoke ok ({time.time() - T_START:.0f}s), pick {pick}")
        return
    assert list(sub.sample_id) == list(ss.sample_id)
    extra_dir = out / "extra" / VERSION
    extra_dir.mkdir(parents=True, exist_ok=True)
    for name, v in list(arms.items()) + [(f"raw_{k}", v) for k, v in raw.items()]:
        np.savez_compressed(extra_dir / f"arm_{name}.npz", oof=v["oof"], test=v["test"], alpha=v["alpha"],
                            fire_oof=v["fire_oof"], fire_test=v["fire_test"], folds=np.array(folds))
        if not name.startswith("raw_"):
            s = pd.DataFrame(np.clip(v["test"] * v["alpha"], lo, hi_q), columns=TARGETS)
            s.insert(0, "sample_id", te.sample_id.to_numpy())
            s.to_csv(extra_dir / f"submission_{name}.csv", index=False)
    for f in ("emb.npz", "emb_ext.npz", "cnnext.npz"):
        if (run_dir / f).exists():
            (extra_dir / f).write_bytes((run_dir / f).read_bytes())
    for log in run_dir.glob("*.log"):
        (extra_dir / log.name).write_text(log.read_text())
    print("prediction sd:", dict(zip(TARGETS, pred.std(0).round(3).tolist())))
    sub.to_csv(out / "submission.csv", index=False)
    print(f"total {time.time() - T_START:.0f}s, picked arm {pick}")
    print(f"SUBMIT: {out / 'submission.csv'}")


if __name__ == "__main__":
    main(smoke=True)
    main(smoke=False)
