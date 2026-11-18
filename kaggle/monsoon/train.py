import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold

from preprocess import ID, TARGET, load

FOLDS, PATIENCE = 5, 200
LGB = dict(objective="l1", learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, seed=0)
V3_BLOCK_MAE = 0.8947
FM_ESTIMATORS = 8
TAU = np.round(np.arange(0.02, 0.99, 0.02), 2)
MID = int(np.flatnonzero(TAU == 0.5)[0])
GRID = np.arange(0.25, 20.01, 0.25)
BAND = np.digitize(GRID, [1.125, 2.125, 3.125, 4.125, 6.125])
TAIL = BAND >= 4
TAIL_PROBE = 0.03
ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def setup():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn", "tabicl"], check=True)
    if Path("/kaggle/working").exists():
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
        try:
            os.environ["TABPFN_TOKEN"] = secrets.get_secret("TABPFN_TOKEN")
        except Exception:
            pass
    os.environ["TABPFN_NO_BROWSER"] = "1"
    if not os.environ.get("TABPFN_TOKEN"):
        os.environ["TABPFN_MODEL_VERSION"] = "v2"
        print("WARNING: no TABPFN_TOKEN secret -> TabPFN v2 (ungated) instead of v3.5", flush=True)
    from importlib.metadata import version
    import torch
    assert torch.cuda.is_available(), "GPU accelerator required"
    print(f"tabpfn {version('tabpfn')} tabicl {version('tabicl')} torch {torch.__version__} "
          f"gpu {torch.cuda.get_device_name(0)} model version {os.environ.get('TABPFN_MODEL_VERSION', 'default (v3.5)')}",
          flush=True)


def clip(p):
    return np.clip(p, 0.25, 20)


def lgb_cv(tr, te, feats, y, folds, rounds=5000):
    yl = np.log1p(y)
    oof, fold_pred, iters = np.zeros(len(tr)), np.zeros(len(te)), []
    for a, b in folds:
        m = lgb.train(LGB, lgb.Dataset(tr[feats].iloc[a], yl[a]), rounds,
                      valid_sets=[lgb.Dataset(tr[feats].iloc[b], yl[b])],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        oof[b] = m.predict(tr[feats].iloc[b], num_iteration=m.best_iteration)
        fold_pred += m.predict(te[feats], num_iteration=m.best_iteration) / len(folds)
        iters.append(m.best_iteration)
    full = lgb.train(LGB, lgb.Dataset(tr[feats], yl), int(np.median(iters)))
    return clip(np.expm1(oof)), clip(np.expm1(full.predict(te[feats]))), clip(np.expm1(fold_pred)), iters


def make(kind, n_est):
    if kind == "tabpfn":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(n_estimators=n_est, device="cuda", ignore_pretraining_limits=True, random_state=0)
    from tabicl import TabICLRegressor
    return TabICLRegressor(n_estimators=n_est, device="cuda", random_state=0)


def quantiles(kind, m, X):
    if kind == "tabpfn":
        return np.stack(m.predict(X, output_type="quantiles", quantiles=list(TAU)), axis=1)
    return np.asarray(m.predict(X, output_type="quantiles", alphas=list(TAU)))


def fm_fit_predict(kind, n_est, X, yl, Xp):
    import torch
    q = quantiles(kind, make(kind, n_est).fit(X, yl), Xp)
    torch.cuda.empty_cache()
    assert q.shape == (len(Xp), len(TAU)), (kind, q.shape)
    return clip(np.expm1(np.sort(q, axis=1)))


def fm_cv(kind, tr, te, feats, y, folds, n_est=FM_ESTIMATORS):
    yl, oof = np.log1p(y), np.zeros((len(tr), len(TAU)))
    for a, b in folds:
        oof[b] = fm_fit_predict(kind, n_est, tr[feats].iloc[a], yl[a], tr[feats].iloc[b])
    return oof, fm_fit_predict(kind, n_est, tr[feats], yl, te[feats])


def grid_probs(q):
    fp, edges = np.r_[0.0, TAU, 1.0], GRID[:-1] + 0.125
    ramp = np.arange(len(TAU)) * 1e-7
    F = np.array([np.interp(edges, np.r_[0.125, r + ramp, 20.125], fp) for r in q])
    return np.diff(np.c_[np.zeros(len(q)), F, np.ones(len(q))], axis=1)


def band_mass(P):
    return np.stack([P[:, BAND == k].sum(1) for k in range(BAND.max() + 1)], axis=1)


def em_prior(P, src, iters=500):
    Pb, q = band_mass(P), src.copy()
    for _ in range(iters):
        post = Pb * (q / src)
        q = (post / post.sum(1, keepdims=True)).mean(0)
    return q


def median_of(P, wb):
    W = P * wb[BAND]
    c = np.cumsum(W, axis=1) / W.sum(1, keepdims=True)
    return GRID[np.minimum((c < 0.5).sum(1), len(GRID) - 1)]


def shift_delta(P, wb):
    return median_of(P, wb) - median_of(P, np.ones_like(wb))


def smooth(pred, g, alpha):
    med = pd.Series(pred).groupby(g.values).transform("median").values
    return np.where(np.isnan(med), pred, alpha * pred + (1 - alpha) * med)


def clustered(d, groups):
    s = pd.Series(d).groupby(groups).sum()
    return d.mean(), np.sqrt((s ** 2).sum()) / len(d)


def smoke(tr, te, feats, y, tr_sd):
    n, f = 300, [(np.arange(200), np.arange(200, 300))]
    t, s, ys = tr.iloc[:n], te.iloc[:50], y[:n]
    lo, lt, _, _ = lgb_cv(t, s, feats, ys, f, rounds=20)
    qs = {k: fm_cv(k, t, s, feats, ys, f, n_est=1) for k in ("tabpfn", "tabicl")}
    P = grid_probs((qs["tabpfn"][0][200:] + qs["tabicl"][0][200:]) / 2)
    assert np.allclose(P.sum(1), 1) and (P >= -1e-9).all()
    src = band_mass(P).mean(0)
    d = shift_delta(P, em_prior(P, src) / src)
    sm = smooth(lo[200:], tr_sd.iloc[200:n].reset_index(drop=True), 0.5)
    for name, v in [("lgb", lo[200:]), ("lgb_test", lt), ("tabpfn", qs["tabpfn"][1]), ("tabicl", qs["tabicl"][1]),
                    ("delta", d), ("smooth", sm)]:
        assert np.isfinite(v).all(), name
    print("smoke ok: lgb, tabpfn, tabicl, EM, smoothing", flush=True)


def write(name, ids, p):
    pd.DataFrame({ID: ids, TARGET: p}).to_csv(OUT / "extra" / f"{name}.csv", index=False)


if __name__ == "__main__":
    setup()
    ids, y, test_ids, tr, te, feats, groups, tr_sd, te_sd = load()
    smoke(tr, te, feats, y, tr_sd)
    (OUT / "extra").mkdir(exist_ok=True)
    folds = list(GroupKFold(FOLDS).split(tr, groups=groups))
    oof, pred, qo, qt = {}, {}, {}, {}

    t0 = time.time()
    oof["lgb"], pred["lgb"], v3, iters = lgb_cv(tr, te, feats, y, folds)
    print(f"lgb block MAE {np.abs(oof['lgb'] - y).mean():.4f} (V3 no_calendar {V3_BLOCK_MAE}) iters {iters} "
          f"full refit {int(np.median(iters))} rounds, rms(full - V3 folds) {np.sqrt(((pred['lgb'] - v3) ** 2).mean()):.4f} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    write("lgb", test_ids, pred["lgb"])
    for kind in ("tabpfn", "tabicl"):
        t0 = time.time()
        qo[kind], qt[kind] = fm_cv(kind, tr, te, feats, y, folds)
        oof[kind], pred[kind] = qo[kind][:, MID], qt[kind][:, MID]
        print(f"{kind} block MAE {np.abs(oof[kind] - y).mean():.4f} [{time.time() - t0:.0f}s]", flush=True)
        write(kind, test_ids, pred[kind])
    oof["blend"] = np.median([oof[k] for k in ("lgb", "tabpfn", "tabicl")], axis=0)
    pred["blend"] = np.median([pred[k] for k in ("lgb", "tabpfn", "tabicl")], axis=0)


    Po = grid_probs((qo["tabpfn"] + qo["tabicl"]) / 2)
    Pt = grid_probs((qt["tabpfn"] + qt["tabicl"]) / 2)
    src = band_mass(Po).mean(0)
    print(f"band prior: train labels {np.bincount(BAND[np.searchsorted(GRID, y)], minlength=6) / len(y)} "
          f"| OOF mean posterior {src.round(3)}")
    q_null = em_prior(Po, src)
    d_null = shift_delta(Po, q_null / src)
    rng = np.random.default_rng(0)
    head, tail = np.flatnonzero(y <= 4), np.flatnonzero(y > 4)
    thin = np.r_[head, rng.choice(tail, int(TAIL_PROBE / (1 - TAIL_PROBE) * len(head)), replace=False)]
    q_thin = em_prior(Po[thin], src)
    d_thin = shift_delta(Po[thin], q_thin / src)
    mae = lambda p, idx: np.abs(clip(p) - y[idx]).mean()
    gain_thin = mae(oof["blend"][thin] + d_thin, thin) - mae(oof["blend"][thin], thin)
    gain_null = mae(oof["blend"] + d_null, np.arange(len(y))) - mae(oof["blend"], np.arange(len(y)))
    q_test = em_prior(Pt, src)
    d_test = shift_delta(Pt, q_test / src)
    ls_ok = 0.015 <= q_thin[4:].sum() <= 0.06 and gain_thin < 0 and gain_null < 0.002
    print(f"EM null check (full OOF): tail {src[4:].sum():.3f} -> {q_null[4:].sum():.3f}, blend MAE change {gain_null:+.4f}\n"
          f"EM synthetic thin-tail ({TAIL_PROBE:.0%} true, n {len(thin)}): estimated tail {q_thin[4:].sum():.3f}, "
          f"blend MAE change {gain_thin:+.4f}\n"
          f"EM on test: band prior {q_test.round(3)}, tail {q_test[4:].sum():.3f} (cap probe ~{TAIL_PROBE}); "
          f"mean delta {d_test.mean():+.3f}, rows changed {(d_test != 0).mean():.2f} | label-shift gate {'PASS' if ls_ok else 'FAIL'}",
          flush=True)
    pred["blend_labelshift"] = clip(pred["blend"] + d_test)


    ae_b = np.abs(oof["blend"] - y)
    res = {a: clustered(np.abs(smooth(oof["blend"], tr_sd, a) - y) - ae_b, groups) for a in ALPHAS}
    print("smoothing alpha -> (OOF MAE change, block-clustered SE): "
          + ", ".join(f"{a}: {m:+.4f}±{s:.4f}" for a, (m, s) in res.items()))
    best = min(res, key=lambda a: res[a][0])
    sm_ok = best < 1 and res[best][0] < -2 * res[best][1]
    print(f"smoothing best alpha {best} | gate {'PASS' if sm_ok else 'FAIL'} | test rows in groups of >=2: "
          f"{te_sd.map(te_sd.value_counts()).ge(2).mean():.3f}", flush=True)
    pred["blend_smooth"] = smooth(pred["blend"], te_sd, best)

    rows = []
    for k in ("lgb", "tabpfn", "tabicl", "blend", "blend_labelshift", "blend_smooth"):
        d = pred[k] - v3
        ae = np.abs(oof[k] - y) if k in oof else np.full(len(y), np.nan)
        rows.append(dict(arm=k, block_mae=ae.mean(), mae_y_le4=ae[y <= 4].mean(), pred_mean=pred[k].mean(),
                         rms_vs_v3=np.sqrt((d ** 2).mean()), lb_sd_bound=np.sqrt((d ** 2).mean() / (0.51 * len(d)))))
        write(k, test_ids, pred[k])
    print("\nblock CV does NOT rank model classes here (FINDINGS); the LB judges. v3 = releases/V3 submission method\n"
          + pd.DataFrame(rows).set_index("arm").round(4).to_string())
    pd.DataFrame({ID: ids, TARGET: y, **oof}).to_csv(OUT / "oof.csv", index=False)
    pd.DataFrame({ID: test_ids, TARGET: pred["blend"]}).to_csv(OUT / "submission.csv", index=False)
    print("SUBMIT submission.csv = extra/blend.csv (median of lgb (full refit), tabpfn, tabicl)")
    print(f"NEXT extra/blend_labelshift.csv only if label-shift gate PASS ({'PASS' if ls_ok else 'FAIL'}); "
          f"extra/blend_smooth.csv only if smoothing gate PASS ({'PASS' if sm_ok else 'FAIL'})")
