import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from gnss import City
from geo import Cache, Lattice, visibility
from preprocess import (VARIANTS, building_variant, gt_agreement, gt_epochs, gt_margins, load, motion_features,
                        positions, signal_features)

WORK = Path("/tmp/work")
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "out"
FOLDS = ("TST", "WP", "MK")
H_GRID = (0.5, 1.5, 2.5)
WINDOWS = (1, 3)
SCALES = (0.85, 0.92, 1.0, 1.08, 1.15)
TEMPER = tuple((w, e) for w in (1.0, 0.5, 0.3, 0.2) for e in (3e-4, 3e-3))
PASSES, ERR_CAP = 4, 30.0
CITIES = {}
ST_CONF, ST_GAIN = 0.4, 0.002
SIG = ["cn0", "el_h", "res_h", "abs_res_h", "rsse_h", "sig_h", "nres_h", "has_phase", "has_l2", "cn02", "cn0_l1l2",
       "sysi", "arc_age", "arc_left", "cn0_m5", "cn0_s5", "cn0_min5", "cn0_m15", "cn0_s15", "cn0_min15", "ph_f5",
       "ph_f15", "cn0_d1", "lock_age", "cdr", "abs_cdr", "abs_cdr_m5", "dcmc", "abs_dcmc", "abs_dcmc_m5", "n_sats",
       "ep_phase", "ep_cn0_med", "cn0_rel", "cn0_rank", "el_rank", "cn0_el"]
NAV = SIG + ["res", "nres", "loo", "abs_loo", "abs_loo_m5", "dres", "abs_dres", "abs_dres_m5", "speed", "across"]
MAP = NAV + ["m_c", "m_max", "m_min", "m_mean", "los8", "los16"]
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(str(msg))


def lgbm(n):
    return lgb.LGBMClassifier(objective="binary", n_estimators=n, learning_rate=0.03, num_leaves=31,
                              min_child_samples=200, colsample_bytree=0.6, subsample=0.8, subsample_freq=1,
                              reg_lambda=5.0, random_state=0, verbose=-1)


def small(n):
    return lgb.LGBMClassifier(objective="binary", n_estimators=n // 2, learning_rate=0.05, num_leaves=7,
                              min_child_samples=500, subsample=0.8, subsample_freq=1, reg_lambda=5.0,
                              monotone_constraints=[1, 1, 0, 0, 0, 0], random_state=0, verbose=-1)


def city_for(bld, variant, scale):
    if (variant, scale) not in CITIES:
        CITIES[variant, scale] = City(building_variant(bld, variant, scale))
    return CITIES[variant, scale]


def smooth(df, p, w):
    if w == 1:
        return p
    s = pd.DataFrame({"p": p, "arc": df.arc.values, "t": df.t.values}, index=df.index).sort_values(["arc", "t"])
    r = s.groupby("arc").p.rolling(w, center=True, min_periods=1).mean().reset_index(level=0, drop=True)
    return r.reindex(df.index).values


def fit_predict(make, xa, ya, xb, n_est, self_train):
    m = make(n_est).fit(xa, ya)
    p = m.predict_proba(xb)[:, 1]
    if self_train:
        sure = np.abs(p - 0.5) >= ST_CONF
        m = make(n_est).fit(pd.concat([xa, xb[sure]]), np.r_[ya, (p[sure] >= 0.5).astype(int)])
        p = m.predict_proba(xb)[:, 1]
    return p, m


def loao(make, tr, te, cols, n_est, self_train):
    y = tr.label.values.astype(int)
    p = np.zeros(len(tr))
    for a in FOLDS:
        va = (tr.area == a).values
        p[va] = fit_predict(make, tr.loc[~va, cols], y[~va], tr.loc[va, cols], n_est, self_train)[0]
    pt, m = fit_predict(make, tr[cols], y, te[cols], n_est, self_train)
    return p, pt, m


def accuracy(tr, p):
    hit = (p >= 0.5) == (tr.label.values == 1)
    return {"pooled": round(float(hit.mean()), 4), **pd.Series(hit).groupby(tr.area.values).mean().round(4).to_dict()}


def record(df, tr_mask, arm, p_tr, p_te, res, oof, tp, t1, m=None, cols=None):
    tr, te = df[tr_mask], df[~tr_mask]
    y = tr.label.values == 1
    res[arm] = {w: accuracy(tr, smooth(tr, p_tr, w)) for w in WINDOWS}
    w = max(res[arm], key=lambda k: res[arm][k]["pooled"])
    oof[arm], tp[arm] = smooth(tr, p_tr, w), smooth(te, p_te, w)
    thr = np.linspace(0.3, 0.7, 41)
    best_thr = thr[np.argmax([((oof[arm] >= c) == y).mean() for c in thr])]
    msg = (f"arm {arm}: LOAO accuracy by window {json.dumps(res[arm])}\n  best window {w}, best OOF threshold "
           f"{best_thr:.2f}, test LOS rate {(tp[arm] >= 0.5).mean():.3f}, {time.time() - t1:.0f}s")
    if m is not None:
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
        msg += "\n  top gain: " + ", ".join(f"{k} {v:.3f}" for k, v in (imp / imp.sum()).head(8).items())
    log(msg)


def fit_arm(df, tr_mask, arm, cols, n_est, res, oof, tp, make=lgbm, self_train=False):
    t1 = time.time()
    p_tr, p_te, m = loao(make, df[tr_mask], df[~tr_mask], cols, n_est, self_train)
    record(df, tr_mask, arm, p_tr, p_te, res, oof, tp, t1, m, cols)


def position_report(df, ep, cols):
    e = gt_epochs(df).merge(ep, on=["seg", "t"], how="left")
    e["area"] = np.array(["TST", "TST", "WP", "MK"])[e.seg.values]
    for name, (sx, sy) in cols.items():
        err = np.hypot(e[sx] - e.x_gt, e[sy] - e.y_gt)
        q = err.groupby(e.area).quantile([0.5, 0.9]).unstack().round(1)
        log(f"horizontal error {name} (m), median / p90 by area:\n{q.to_string()}")
    return e


def choose_buildings(df, bld, t0):
    agree = {}
    for name in VARIANTS:
        by_area, agree[name] = gt_agreement(df, city_for(bld, name, 1.0), 1.5)
        log(f"ray-cast at GT, buildings '{name}', antenna 1.5 m: {by_area} pooled {agree[name]:.4f}")
    variant = max(agree, key=agree.get)
    agree_s = {}
    for s in SCALES:
        by_area, agree_s[s] = gt_agreement(df, city_for(bld, variant, s), 1.5)
        log(f"ray-cast at GT, buildings '{variant}', height scale {s}, antenna 1.5 m: {by_area} pooled "
            f"{agree_s[s]:.4f}")
    scale = max(agree_s, key=agree_s.get)
    city = city_for(bld, variant, scale)
    agree_h = {}
    for h in H_GRID:
        by_area, agree_h[h] = gt_agreement(df, city, h)
        log(f"ray-cast at GT, buildings '{variant}', scale {scale}, antenna {h} m: {by_area} pooled {agree_h[h]:.4f}")
    h_ant = max(agree_h, key=agree_h.get)
    m, y = gt_margins(df, city, h_ant), df.label.values[~df.test.values]
    ok = np.isfinite(m)
    lr = LogisticRegression(C=1e6).fit(np.clip(m[ok], -30, 30)[:, None] / 10, y[ok])
    vis = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
    pts = np.array([-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0])
    log(f"chosen on train GT: buildings '{variant}', height scale {scale}, antenna {h_ant} m; soft visibility "
        f"P(label LOS | margin) at {pts.tolist()} deg: {np.round(visibility(pts, vis), 3).tolist()}; "
        f"{time.time() - t0:.0f}s")
    return city, h_ant, vis, {"variant": variant, "scale": scale, "h_ant": h_ant, "vis": vis}


def run(df, eph, bld, smoke):
    t0 = time.time()
    if smoke:
        keep = df.groupby("seg").t.rank(method="dense") <= 40
        df = df[keep].reset_index(drop=True)
    df = df.copy()
    ep = positions(df, eph, log)
    d = df[np.isfinite(df.el_o)]
    for s, g in d.groupby("sys"):
        daz = np.abs((np.degrees(g.az_o) - g.az_h + 180) % 360 - 180)
        log(f"ephemeris check {s}: median |el ours - host| {np.median(np.abs(np.degrees(g.el_o) - g.el_h)):.3f} deg, "
            f"|az| {np.median(daz):.3f} deg")
    signal_features(df)
    motion_features(df, ep)
    log(f"positions and signal features done {time.time() - t0:.0f}s")
    city, h_ant, vis, setup = choose_buildings(df, bld, t0)

    n_est = 30 if smoke else 600
    tr_mask = (~df.test).values
    gt = gt_epochs(df)
    gtx = gt.merge(ep[["seg", "t"]].reset_index(), on=["seg", "t"])
    gi, garea = gtx["index"].values, np.array(["TST", "TST", "WP", "MK"])[gtx.seg.values]
    cache = Cache(city, h_ant)
    lat = Lattice(df, ep, cache, ep.x.values, ep.y.values, vis)
    lat.base_features(df)
    inside = pd.Series(city.inside(gt.x_gt.values, gt.y_gt.values)).groupby(gt.seg.values).mean().round(4)
    log(f"lattice: {len(cache.keys)} skymask cells; candidates inside buildings {lat.inside[lat.ok].mean():.3f}; "
        f"GT inside by seg {inside.to_dict()}; {time.time() - t0:.0f}s")
    arms, res, oof, tp = {"nav": NAV, "map": MAP}, {}, {}, {}
    for arm in ("nav", "map"):
        fit_arm(df, tr_mask, arm, arms[arm], n_est, res, oof, tp)
    y = df.label.values[tr_mask] == 1
    xy, src = {"smoothed": ("x", "y")}, "nav"
    df["lg_nav"] = 0.0
    df.loc[tr_mask, "lg_nav"], df.loc[~tr_mask, "lg_nav"] = logit(np.clip(oof["nav"], 1e-4, 1 - 1e-4)), \
        logit(np.clip(tp["nav"], 1e-4, 1 - 1e-4))
    for i in range(1, PASSES + 1):
        t1 = time.time()
        if i > 1:
            lat = None
            lat = Lattice(df, ep, cache, ep[f"x_{i - 1}"].values, ep[f"y_{i - 1}"].values, vis)
        p = np.zeros(len(df))
        p[tr_mask], p[~tr_mask] = oof[src], tp[src]
        sm = lat.sm_ll(p)
        best = None
        for w, eps in TEMPER:
            gamma = lat.smooth(sm, w, eps)
            x, yy = lat.mean_position(gamma)
            err = np.hypot(x[gi] - gtx.x_gt.values, yy[gi] - gtx.y_gt.values)
            score = float(np.nanmean(np.minimum(err, ERR_CAP)))
            q = pd.Series(err).groupby(garea).quantile([0.5, 0.9]).unstack().round(1)
            log(f"pass {i} (likelihood from {src}): weight {w}, floor {eps}: mean error capped at {ERR_CAP:.0f} m "
                f"{score:.2f}; median {q[0.5].to_dict()} p90 {q[0.9].to_dict()}")
            if best is None or score < best[0]:
                best = (score, w, eps, gamma)
        _, w, eps, gamma = best
        cols = MAP + lat.post_features(df, ep, gamma, f"_{i}")
        xy[f"pass {i}"] = (f"x_{i}", f"y_{i}")
        tr = df[tr_mask]
        for lab, col in (("posterior LOS > 0.5", tr[f"post_los_{i}"] > 0.5),
                         ("hard posterior LOS > 0.5", tr[f"post_hard_{i}"] > 0.5),
                         ("posterior-mean cell", tr[f"pm_m_{i}"] > 0)):
            hit = col.values == y
            log(f"pass {i} chosen weight {w}, floor {eps}: ray-cast at {lab} agreement "
                f"{pd.Series(hit).groupby(tr.area.values).mean().round(4).to_dict()} pooled {hit.mean():.4f}")
        post = df[f"post_los_{i}"].fillna(0.5).values
        record(df, tr_mask, f"geom{i}", post[tr_mask], post[~tr_mask], res, oof, tp, t1)
        df[f"lg_post_{i}"] = logit(np.clip(post, 1e-4, 1 - 1e-4))
        arms[f"fuse{i}"] = [f"lg_post_{i}", "lg_nav", "el_h", "cn0", f"post_std_{i}", f"pm_m_{i}"]
        arms[f"geo{i}"] = cols
        fit_arm(df, tr_mask, f"fuse{i}", arms[f"fuse{i}"], n_est, res, oof, tp, make=small)
        fit_arm(df, tr_mask, f"geo{i}", cols, n_est, res, oof, tp)
        src = max((f"fuse{i}", f"geo{i}"), key=lambda a: max(v["pooled"] for v in res[a].values()))
        log(f"pass {i} done {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s); {len(cache.keys)} skymask "
            f"cells; next likelihood from {src}")
    epg = position_report(df, ep, xy)
    pooled = {a: max(v["pooled"] for v in r.values()) for a, r in res.items() if a != "nav"}
    trained = max((a for a in pooled if a.startswith(("fuse", "geo")) and not a.startswith("geom")), key=pooled.get)
    make = small if trained.startswith("fuse") else lgbm
    fit_arm(df, tr_mask, trained + "_st", arms[trained], n_est, res, oof, tp, make=make, self_train=True)
    gain = max(v["pooled"] for v in res[trained + "_st"].values()) - pooled[trained]
    log(f"self-training of {trained} on the held-out area: LOAO gain {gain:+.4f} (adopted if >= {ST_GAIN})")
    if gain >= ST_GAIN:
        pooled[trained + "_st"] = pooled[trained] + gain
    best = max(pooled, key=pooled.get)
    tr, te = df[tr_mask], df[~tr_mask]
    t = ep[ep.seg == 4]
    log(f"LOAO pooled by arm: { {a: round(v, 4) for a, v in pooled.items()} }")
    log(f"chosen arm: {best}; test agreement with map arm {((tp['map'] >= 0.5) == (tp[best] >= 0.5)).mean():.4f}; "
        f"test posterior-mean shift median {np.hypot(t[f'x_{PASSES}'] - t.x, t[f'y_{PASSES}'] - t.y).median():.1f} m")
    sub = pd.DataFrame({"ID": te.id.values, "Predict": (tp[best] >= 0.5).astype(int)}).sort_values("ID")
    n_test = int(df.test.sum())
    assert len(sub) == n_test and (sub.ID.values == np.arange(n_test)).all() and sub.Predict.isin([0, 1]).all()
    tag = "smoke" if smoke else "full"
    (WORK / tag).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"seg": tr.seg, "t": tr.t, "sat": tr.sat, "area": tr.area, "label": tr.label.astype(int),
                  **{f"oof_{a}": v for a, v in oof.items()}}).to_csv(WORK / tag / "oof.csv", index=False)
    pd.DataFrame({"ID": te.id.values, **{f"p_{a}": v for a, v in tp.items()}}).sort_values("ID") \
        .to_csv(WORK / tag / "test_probs.csv", index=False)
    epg.to_csv(WORK / tag / "train_epochs.csv", index=False)
    ep.to_csv(WORK / tag / "epochs.csv", index=False)
    (WORK / tag / "cv.json").write_text(json.dumps({"cv": res, "chosen": best, **setup}, indent=1))
    log(f"{tag} run done in {time.time() - t0:.0f}s")
    return sub


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        df, eph, bld = load()
        log(f"loaded {len(df)} rows, {len(bld)} buildings, {time.time() - t0:.0f}s")
        run(df, eph, bld, smoke=True)
        log("smoke ok")
        if os.environ.get("SMOKE") != "1":
            sub = run(df, eph, bld, smoke=False)
            sub.to_csv(OUT / "submission.csv", index=False)
            log(f"SUBMIT: {OUT / 'submission.csv'} (LOS rate {sub.Predict.mean():.3f})")
    finally:
        (WORK / "log.txt").write_text("\n".join(LOG))
        with zipfile.ZipFile(OUT / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in WORK.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(WORK))


if __name__ == "__main__":
    main()
