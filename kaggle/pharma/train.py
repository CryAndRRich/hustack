import json
import os
import shutil
import signal
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from preprocess import (SEV_CLASSES, TEST_BP, TRAIN_BP, build_static, flatten,
                        fold_target_encodings, load, log, set_breakpoints)

T0 = time.time()
SMOKE = os.environ.get("PHARMA_SMOKE", "0") == "1"
SEED = 2026
N_FOLDS = 2 if SMOKE else 5
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
WORK = Path("/tmp/work/v1")
EXTRA = WORK / "extra"
for p in (WORK, EXTRA):
    p.mkdir(parents=True, exist_ok=True)
BUDGET_S = 600 if SMOKE else 9 * 3600
RESULT = {"smoke": SMOKE, "seed": SEED, "folds": N_FOLDS}


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def budget_left():
    return BUDGET_S - (time.time() - T0)


SEV_INDEX = {c: i for i, c in enumerate(SEV_CLASSES)}


def encode_severity(y):
    y = np.asarray(y)
    unknown = set(np.unique(y)) - set(SEV_INDEX)
    assert not unknown, f"severity value(s) not in SEV_CLASSES: {unknown}"
    code = np.array([SEV_INDEX[v] for v in y], dtype=np.int64)
    assert (np.array(SEV_CLASSES)[code] == y).all(), "severity round-trip failed"
    return code


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", labels=SEV_CLASSES, zero_division=0))


def micro_f1_from_counts(tp, fp, fn):
    return 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)


def masked_rmse(y, p):
    m = y > 0
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def final_score(sev_f1, bin_f1, rmse):
    return 0.4 * sev_f1 + 0.3 * bin_f1 + 0.3 / (1.0 + rmse)


def best_global_threshold(Y, S, n_grid=200):
    y = Y.ravel().astype(np.int8)
    s = S.ravel().astype(np.float64)
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    pos = float(y.sum())
    f1 = 2.0 * tp / (k + pos)
    i = int(np.argmax(f1))
    s_sorted = s[order]
    thr = s_sorted[i] if i + 1 >= len(s_sorted) else (s_sorted[i] + s_sorted[i + 1]) / 2.0
    return float(thr), float(f1[i])


def per_label_thresholds(Y, S, grid=np.linspace(0.02, 0.95, 94)):
    n_lab = Y.shape[1]
    thr = np.full(n_lab, best_global_threshold(Y, S)[0], dtype=np.float64)
    P = S >= thr[None, :]
    tp = float((P & (Y == 1)).sum())
    fp = float((P & (Y == 0)).sum())
    fn = float((~P & (Y == 1)).sum())
    for _ in range(2):
        for j in range(n_lab):
            yj, sj = Y[:, j], S[:, j]
            pj = P[:, j]
            tp_j = float((pj & (yj == 1)).sum())
            fp_j = float((pj & (yj == 0)).sum())
            fn_j = float((~pj & (yj == 1)).sum())
            base = (tp - tp_j, fp - fp_j, fn - fn_j)
            best = (micro_f1_from_counts(tp, fp, fn), thr[j], tp_j, fp_j, fn_j)
            for t in grid:
                q = sj >= t
                a = float((q & (yj == 1)).sum())
                b = float((q & (yj == 0)).sum())
                c = float((~q & (yj == 1)).sum())
                f = micro_f1_from_counts(base[0] + a, base[1] + b, base[2] + c)
                if f > best[0]:
                    best = (f, float(t), a, b, c)
            thr[j] = best[1]
            P[:, j] = sj >= thr[j]
            tp, fp, fn = base[0] + best[2], base[1] + best[3], base[2] + best[4]
    return thr, micro_f1_from_counts(tp, fp, fn)


def best_activity_gate(Y, S, act, n_grid=40):
    g0, f0 = best_global_threshold(Y, S)
    best = (-np.inf, g0, f0)
    for q in np.linspace(0.05, 0.95, n_grid):
        gate = float(np.quantile(act, q))
        keep = act >= gate
        if keep.sum() < 50 or keep.all():
            continue
        t, f = best_global_threshold(Y, S * keep[:, None])
        if f > best[2] + 1e-6:
            best = (gate, t, f)
    return best


def get_lgb():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        log("lightgbm missing: trying pip install")
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], check=True, timeout=600)
            import lightgbm as lgb
            return lgb
        except Exception as e:
            log("lightgbm unavailable:", repr(e)[:200])
            return None


def lgb_params(objective, n_class=None, seed=SEED):
    p = {"objective": objective, "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 40,
         "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
         "verbosity": -1, "seed": seed, "num_threads": os.cpu_count() or 4, "max_bin": 127}
    if n_class:
        p.update({"num_class": n_class, "metric": "multi_logloss"})
    elif objective == "binary":
        p["metric"] = "binary_logloss"
    else:
        p["metric"] = "rmse"
    return p


def fit_lgb(lgb, X, y, Xv, yv, params, rounds, weight=None, early=50):
    ds = lgb.Dataset(X, label=y, weight=weight, free_raw_data=True)
    vs = [lgb.Dataset(Xv, label=yv, reference=ds)] if Xv is not None and len(Xv) else None
    cb = [lgb.early_stopping(early, verbose=False)] if vs else []
    return lgb.train(params, ds, num_boost_round=rounds, valid_sets=vs, callbacks=cb)


def run_severity(lgb, Xtr, ytr, Xva, Xte, rounds):
    code = encode_severity(ytr)
    cnt = np.bincount(code, minlength=3).astype(float)
    inv = len(code) / np.maximum(cnt, 1) / 3.0
    w = inv[code]
    m = fit_lgb(lgb, Xtr, code, None, None, lgb_params("multiclass", n_class=3), rounds, weight=w)
    return m.predict(Xva), (m.predict(Xte) if Xte is not None else None), m


def severity_decision(P, class_bias=None):
    Q = P if class_bias is None else P * class_bias[None, :]
    return np.array(SEV_CLASSES)[np.argmax(Q, 1)]


def tune_class_bias(P, y, n_iter=300, seed=SEED):
    rng = np.random.default_rng(seed)
    best = (macro_f1(y, severity_decision(P)), np.ones(3))
    for _ in range(n_iter):
        b = np.exp(rng.normal(0, 0.7, 3))
        f = macro_f1(y, severity_decision(P, b))
        if f > best[0]:
            best = (f, b)
    return best[1], best[0]


def run_activity(lgb, Xtr, atr, Xva, Xte, rounds):
    m = fit_lgb(lgb, Xtr, atr, None, None, lgb_params("binary"), rounds)
    return m.predict(Xva), (m.predict(Xte) if Xte is not None else None), m


def run_labels(lgb, Xtr, Ytr, Xva, Xte, rounds, n_label_models, effects):
    n_lab = Ytr.shape[1]
    sel = list(range(n_lab)) if n_label_models >= n_lab else \
        list(np.argsort(-Ytr.sum(0))[:n_label_models])
    Sva = np.zeros((len(Xva), n_lab), np.float64)
    Ste = np.zeros((len(Xte), n_lab), np.float64) if Xte is not None else None
    done = []
    for j in sel:
        if budget_left() < 120:
            log(f"  label head: out of budget after {len(done)} labels")
            break
        y = Ytr[:, j]
        if y.sum() < 20 or y.sum() > len(y) - 20:
            continue
        m = fit_lgb(lgb, Xtr, y, None, None, lgb_params("binary"), rounds)
        Sva[:, j] = m.predict(Xva)
        if Ste is not None:
            Ste[:, j] = m.predict(Xte)
        done.append(j)
    return Sva, Ste, done


def run_prr(lgb, Xtr, Rtr, Xva, Xte, rounds, n_prr_models, effects, prr_prior):
    n_lab = Rtr.shape[1]
    sel = list(range(n_lab)) if n_prr_models >= n_lab else list(np.argsort(-(Rtr > 0).sum(0))[:n_prr_models])
    Pva = np.tile(prr_prior.astype(np.float64), (len(Xva), 1))
    Pte = np.tile(prr_prior.astype(np.float64), (len(Xte), 1)) if Xte is not None else None
    done = []
    for j in sel:
        if budget_left() < 120:
            log(f"  prr head: out of budget after {len(done)} labels")
            break
        m_pos = Rtr[:, j] > 0
        if m_pos.sum() < 100:
            continue
        y = np.log1p(Rtr[m_pos, j])
        mdl = fit_lgb(lgb, Xtr[m_pos], y, None, None, lgb_params("regression"), rounds)
        resid = y - mdl.predict(Xtr[m_pos])
        s2 = float(np.var(resid))
        corr = np.exp(s2 / 2.0)
        Pva[:, j] = np.maximum(np.expm1(mdl.predict(Xva)) * corr, 0.01)
        if Pte is not None:
            Pte[:, j] = np.maximum(np.expm1(mdl.predict(Xte)) * corr, 0.01)
        done.append(j)
    return Pva, Pte, done


def one_arm(name, use_position, tr, te, stat_tr, stat_te, Y, R, sev, groups, lgb, cfg, effects):
    log(f"=== arm {name} (position features: {use_position}) ===")
    drop = set()
    if not use_position:
        drop = {k for k in stat_tr if k.startswith("pos_")}
        log(f"  dropping {len(drop)} position features")
    s_tr = {k: v for k, v in stat_tr.items() if k not in drop}
    s_te = {k: v for k, v in stat_te.items() if k not in drop}

    n, n_lab = len(tr), Y.shape[1]
    oof_sev = np.zeros((n, 3))
    oof_act = np.zeros(n)
    oof_lab = np.zeros((n, n_lab))
    oof_prr = np.zeros((n, n_lab))
    te_sev, te_act, te_lab, te_prr, n_used = [], [], [], [], 0
    oof_prior = np.zeros((n, n_lab))
    te_priors = []
    act = (Y.sum(1) > 0).astype(int)
    gkf = GroupKFold(n_splits=N_FOLDS)
    rng = np.random.default_rng(SEED)
    uniq_g = pd.unique(groups)
    perm = rng.permutation(len(uniq_g))
    gmap = dict(zip(uniq_g, perm))
    g_shuf = pd.Series(groups).map(gmap).values

    all_folds = list(gkf.split(np.zeros(n), groups=g_shuf))
    for fold, (itr, iva) in enumerate(all_folds):
        if budget_left() < 300:
            log(f"  stopping before fold {fold}: {budget_left():.0f}s left")
            break
        t0 = time.time()
        (f_tr, f_va), prr_prior = fold_target_encodings(
            tr, itr, [itr, iva], Y, R, sev, m=cfg["te_m"])
        Xtr, names = flatten([{k: v[itr] for k, v in s_tr.items()}, f_tr], len(itr))
        Xva, names_v = flatten([{k: v[iva] for k, v in s_tr.items()}, f_va], len(iva))
        assert names == names_v, "fold feature names differ between train and valid"
        f_te_apply = apply_encodings_to_test(tr, te, itr, Y, R, sev, cfg["te_m"])
        Xte, names_t = flatten([s_te, f_te_apply], len(te))
        assert names == names_t, f"test feature names differ from train: {set(names) ^ set(names_t)}"

        ps, pst, _ = run_severity(lgb, Xtr, sev[itr], Xva, Xte, cfg["rounds_sev"])
        oof_sev[iva] = ps
        pa, pat, _ = run_activity(lgb, Xtr, act[itr], Xva, Xte, cfg["rounds_act"])
        oof_act[iva] = pa
        pl, plt_, done_l = run_labels(lgb, Xtr, Y[itr], Xva, Xte, cfg["rounds_lab"],
                                      cfg["n_label_models"], effects)
        oof_lab[iva] = pl
        pr, prt, done_p = run_prr(lgb, Xtr, R[itr], Xva, Xte, cfg["rounds_prr"],
                                  cfg["n_prr_models"], effects, prr_prior)
        oof_prr[iva] = pr
        oof_prior[iva] = prr_prior[None, :]
        te_priors.append(prr_prior)
        for lst, v in ((te_sev, pst), (te_act, pat), (te_lab, plt_), (te_prr, prt)):
            if v is not None:
                lst.append(v)
        n_used += 1
        log(f"  fold {fold}: {len(itr)} train / {len(iva)} val, {Xtr.shape[1]} features, "
            f"{len(done_l)} label models, {len(done_p)} prr models, {time.time() - t0:.0f}s")

    assert n_used > 0, "no fold completed"
    scored = np.zeros(n, bool)
    for _, iva in all_folds[:n_used]:
        scored[iva] = True
    log(f"  scoring {scored.sum()} of {n} rows ({n_used} folds)")

    bias, f_sev = tune_class_bias(oof_sev[scored], sev[scored])
    S_oof = oof_lab[scored] * oof_act[scored][:, None]
    gate, thr_gate, f_bin_gate = best_activity_gate(Y[scored], S_oof, oof_act[scored])
    thr_g, f_bin_g = best_global_threshold(Y[scored], S_oof)
    thr_pl, f_bin_pl = per_label_thresholds(Y[scored], S_oof)
    use_pl = f_bin_pl > max(f_bin_g, f_bin_gate) + 1e-4
    use_gate = (not use_pl) and f_bin_gate > f_bin_g + 1e-4
    if use_pl:
        f_bin = f_bin_pl
    elif use_gate:
        f_bin = f_bin_gate
        thr_g = thr_gate
    else:
        f_bin = f_bin_g
    log(f"  binary thresholds: global {f_bin_g:.5f} | per-label {f_bin_pl:.5f} | "
        f"gated {f_bin_gate:.5f} (gate {gate:.4f}) -> using "
        f"{'per-label' if use_pl else ('gated' if use_gate else 'global')}")
    base_oof = oof_prior[scored]
    best = (masked_rmse(R[scored], oof_prr[scored]), 1.0)
    for w in np.linspace(0, 1, 21):
        r = masked_rmse(R[scored], w * oof_prr[scored] + (1 - w) * base_oof)
        if r < best[0]:
            best = (r, float(w))
    w_shrink = best[1]
    P_oof = w_shrink * oof_prr[scored] + (1 - w_shrink) * base_oof
    msk = R[scored] > 0
    denom = float((P_oof[msk] ** 2).sum())
    prr_scale = float((P_oof[msk] * R[scored][msk]).sum() / denom) if denom > 0 else 1.0
    prr_scale = float(np.clip(prr_scale, 0.5, 2.0))
    if masked_rmse(R[scored], P_oof * prr_scale) >= masked_rmse(R[scored], P_oof):
        prr_scale = 1.0
    P_oof = P_oof * prr_scale
    rmse = masked_rmse(R[scored], P_oof)
    s = final_score(f_sev, f_bin, rmse)
    log(f"  OOF: sevF1 {f_sev:.4f} | microF1 {f_bin:.4f} (global {f_bin_g:.4f} / perlabel {f_bin_pl:.4f}) "
        f"| maskedRMSE {rmse:.4f} (shrink w={w_shrink:.2f} scale {prr_scale:.3f}) | SCORE {s:.5f}")
    aucs = [roc_auc_score(Y[scored][:, j], S_oof[:, j]) for j in range(n_lab)
            if 0 < Y[scored][:, j].sum() < scored.sum()]
    log(f"  OOF mean label AUC {np.mean(aucs):.4f} | activity AUC "
        f"{roc_auc_score(act[scored], oof_act[scored]):.4f}")

    res = {"arm": name, "use_position": use_position, "sev_f1": f_sev, "bin_f1": f_bin,
           "bin_f1_global": f_bin_g, "bin_f1_perlabel": f_bin_pl, "bin_f1_gated": f_bin_gate,
           "use_per_label_thr": bool(use_pl), "use_activity_gate": bool(use_gate),
           "activity_gate": float(gate) if use_gate else None,
           "prr_rmse": rmse, "prr_shrink_w": w_shrink, "prr_scale": prr_scale,
           "score": s, "folds_used": n_used,
           "n_features": int(Xtr.shape[1]), "mean_label_auc": float(np.mean(aucs)),
           "class_bias": bias.tolist(), "thr_global": thr_g}
    pred = None
    if te_sev:
        Psev = np.mean(te_sev, 0)
        Pact = np.mean(te_act, 0)
        Plab = np.mean(te_lab, 0) * Pact[:, None]
        Pprr = np.mean(te_prr, 0)
        Pprr = (w_shrink * Pprr + (1 - w_shrink) * np.tile(np.mean(te_priors, 0), (len(te), 1))) * prr_scale
        thr = thr_pl if use_pl else np.full(n_lab, thr_g)
        Pbin = (Plab >= thr[None, :]).astype(int)
        if use_gate:
            Pbin[Pact < gate] = 0
        pred = {"sev": severity_decision(Psev, bias), "bin": Pbin,
                "prr": np.maximum(Pprr, 0.0)}
        log(f"  test: active rate {(pred['bin'].sum(1) > 0).mean():.4f}, "
            f"mean labels/row {pred['bin'].sum(1).mean():.2f}, severity "
            f"{pd.Series(pred['sev']).value_counts(normalize=True).round(3).to_dict()}")
    return res, pred, {"sev": oof_sev, "act": oof_act, "lab": oof_lab, "prr": oof_prr, "scored": scored}


def apply_encodings_to_test(tr, te, idx_fit, Y, R, sev, m):
    n_tr = len(tr)
    cat = pd.concat([tr[["kA", "kB"]], te[["kA", "kB"]]], ignore_index=True)
    Yc = np.vstack([Y, np.zeros((len(te), Y.shape[1]), Y.dtype)])
    Rc = np.vstack([R, np.zeros((len(te), R.shape[1]), R.dtype)])
    sevc = np.concatenate([sev, np.full(len(te), SEV_CLASSES[1])])
    (f_te,), _ = fold_target_encodings(cat, idx_fit, [np.arange(n_tr, n_tr + len(te))], Yc, Rc, sevc, m=m)
    return f_te


def write_submission(ss, pred, bin_cols, prr_cols, path):
    sub = pd.DataFrame({"Pair_ID": ss.Pair_ID.values})
    sub["Severity"] = pred["sev"]
    B = pd.DataFrame(pred["bin"].astype(int), columns=bin_cols)
    P = pd.DataFrame(np.round(pred["prr"].astype(float), 6), columns=prr_cols)
    sub = pd.concat([sub, B, P], axis=1)
    assert list(sub.columns) == ["Pair_ID", "Severity"] + bin_cols + prr_cols, "column order wrong"
    assert len(sub) == len(ss) and (sub.Pair_ID.values == ss.Pair_ID.values).all(), "id mismatch"
    assert set(sub.Severity.unique()) <= set(SEV_CLASSES), "severity value outside train classes"
    assert sub[bin_cols].isin([0, 1]).all().all(), "binary column not 0/1"
    assert (sub[prr_cols].values >= 0).all() and np.isfinite(sub[prr_cols].values).all(), "bad PRR"
    sub.to_csv(path, index=False)
    return sub


def main():
    cfg = dict(te_m=5.0, rounds_sev=120, rounds_act=120, rounds_lab=60, rounds_prr=60,
               n_label_models=6, n_prr_models=6) if SMOKE else \
        dict(te_m=5.0, rounds_sev=600, rounds_act=600, rounds_lab=250, rounds_prr=250,
             n_label_models=50, n_prr_models=50)
    log("config:", cfg)
    lgb = get_lgb()
    assert lgb is not None, "lightgbm is required"
    tr, te, ss, bin_cols, prr_cols, effects = load()
    if SMOKE:
        step = max(1, len(tr) // 1500)
        tr = tr.iloc[::step].reset_index(drop=True)
        te = te.iloc[: max(200, len(te) // 8)].reset_index(drop=True)
        ss = ss.iloc[: len(te)].reset_index(drop=True)
        log(f"SMOKE: sampled train {tr.shape} (every {step}th row), test {te.shape}. "
            f"Classes present: {sorted(tr.Severity.unique())}. Scores are NOT meaningful.")
        assert set(tr.Severity.unique()) == set(SEV_CLASSES), "smoke sample lost a severity class"
        set_breakpoints(tuple(int(round(b / step)) for b in TRAIN_BP),
                        tuple(int(round(b * len(te) / 3974)) for b in TEST_BP))
    tr, te, stat_tr, stat_te = build_static(tr, te, n_svd_fp=16 if SMOKE else 48,
                                            n_svd_tx=16 if SMOKE else 64, seed=SEED)
    Y = tr[bin_cols].to_numpy(np.int8)
    R = tr[prr_cols].to_numpy(np.float64)
    sev = tr.Severity.to_numpy()
    assert ((R > 0) == (Y == 1)).all(), "run-1 invariant broken: PRR>0 is no longer exactly Binary==1"
    groups = tr.pair_key.values
    log(f"targets: active rows {float((Y.sum(1) > 0).mean()):.4f}, positive cells {float(Y.mean()):.4f}, "
        f"{len(pd.unique(groups))} distinct pairs")

    code = encode_severity(sev)
    assert list(np.bincount(code, minlength=3)) == [int((sev == c).sum()) for c in SEV_CLASSES], \
        "severity encoding does not reproduce the class counts"
    _P = np.tile(np.array([0.2, 0.5, 0.3]), (len(sev), 1))
    assert set(np.unique(severity_decision(_P))) <= set(SEV_CLASSES), "severity_decision produced a bad class"
    _b, _f = tune_class_bias(_P[:200], sev[:200], n_iter=5)
    assert np.isfinite(_f), "tune_class_bias returned a non-finite score"
    _Y = Y[:500]
    _S = np.random.default_rng(0).random(_Y.shape)
    _t, _g = best_global_threshold(_Y, _S)
    assert 0.0 <= _g <= 1.0 and np.isfinite(_t), "best_global_threshold is broken"
    assert np.isfinite(masked_rmse(R[:500], np.full_like(R[:500], 3.9))), "masked_rmse is broken"
    log(f"preflight ok: severity counts match, micro-F1 sweep returns {_g:.4f} on random scores")

    results, preds = [], {}
    for name, use_pos in (("with_pos", True),):
        if budget_left() < 400:
            log(f"skipping arm {name}: {budget_left():.0f}s left")
            continue
        try:
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(int(max(60, budget_left() - 120)))
            res, pred, oof = one_arm(name, use_pos, tr, te, stat_tr, stat_te, Y, R, sev, groups,
                                     lgb, cfg, effects)
            signal.alarm(0)
            results.append(res)
            if pred is not None:
                preds[name] = pred
            np.savez_compressed(EXTRA / f"oof_{name}.npz", **{k: v for k, v in oof.items()})
        except Timeout:
            signal.alarm(0)
            log(f"arm {name} hit its time budget; keeping what finished")
        except Exception:
            signal.alarm(0)
            log(f"arm {name} FAILED:\n{traceback.format_exc()}")

    if not results:
        log("ALL ARMS FAILED -- writing the constant fallback submission so the run is not wasted")
        prr_mean = np.where((R > 0).sum(0) > 0, R.sum(0) / np.maximum((R > 0).sum(0), 1), 1.0)
        maj = pd.Series(sev).value_counts().idxmax()
        pred = {"sev": np.full(len(te), maj),
                "bin": np.ones((len(te), Y.shape[1]), int),
                "prr": np.tile(prr_mean, (len(te), 1))}
        sub = write_submission(ss, pred, bin_cols, prr_cols, OUT / "submission.csv")
        RESULT["fallback"] = "constant (all arms failed)"
        (WORK / "result.json").write_text(json.dumps(RESULT, indent=1, default=str))
        with zipfile.ZipFile(OUT / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in WORK.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(WORK))
        log(f"constant fallback written: {sub.shape}")
        print("SUBMIT:", OUT / "submission.csv")
        return
    RESULT["arms"] = results
    tab = pd.DataFrame(results)[["arm", "sev_f1", "bin_f1", "prr_rmse", "score", "folds_used"]]
    log("\n" + tab.round(5).to_string(index=False))
    best = max(results, key=lambda r: r["score"])
    RESULT["best_arm"] = best["arm"]
    log(f"best arm by OOF: {best['arm']} ({best['score']:.5f})")
    if len(results) == 2:
        d = results[0]["score"] - results[1]["score"]
        RESULT["position_feature_delta"] = d
        log(f"value of the row-position features: {d:+.5f} final score "
            f"(severity {results[0]['sev_f1'] - results[1]['sev_f1']:+.4f}, "
            f"binary {results[0]['bin_f1'] - results[1]['bin_f1']:+.4f}, "
            f"rmse {results[0]['prr_rmse'] - results[1]['prr_rmse']:+.4f})")

    chosen = best["arm"] if best["arm"] in preds else next(iter(preds))
    sub = write_submission(ss, preds[chosen], bin_cols, prr_cols, OUT / "submission.csv")
    log(f"submission from arm '{chosen}': {sub.shape}")
    for nm, pred in preds.items():
        if nm != chosen:
            write_submission(ss, pred, bin_cols, prr_cols, EXTRA / f"submission_{nm}.csv")
    (WORK / "result.json").write_text(json.dumps(RESULT, indent=1, default=str))
    tab.to_csv(WORK / "arms.csv", index=False)
    try:
        src = Path(__file__).resolve().parent
        for f in src.glob("*.py"):
            shutil.copy(f, EXTRA / f.name)
    except NameError:
        pass
    with zipfile.ZipFile(OUT / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in WORK.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(WORK))
    log("outputs:", sorted(os.listdir(OUT)))
    print("SUBMIT:", OUT / "submission.csv")
    if SMOKE:
        print("smoke ok")


if __name__ == "__main__":
    main()
