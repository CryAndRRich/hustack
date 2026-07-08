# Updated on June 30, 2026, 0:13 AM
# Public Score: 0.7572
# Rank: 926/3950

import subprocess
import warnings
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from preprocess import (
    CFG, FORMATIONS, BEAMS, ANCH_OFFS, BEAM_OFFS, SC_OFFS, PF_OFFS,
    SELECTOR_SCALES, FormationPlaneKNN, DenseANCCImputer,
    build_dataset, run_pf_lik_ensemble_scales, run_beam_ensemble,
    selector_well_code, apply_selector_variant, lik_pf, beam_search,
    robust_slope, affine_cal, seg_b_well,
    run_pf_ancc, run_pf_z, multi_scale_ncc, dtw_tvt_estimate,
    _well_traj_features,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

lgb_params = [
    dict(objective="regression", metric="rmse", verbosity=-1, learning_rate=0.04,
         num_leaves=127, feature_fraction=0.7, bagging_fraction=0.85, bagging_freq=1,
         min_child_samples=30, lambda_l1=0.1, lambda_l2=0.1, n_estimators=3000),
    dict(objective="regression", metric="rmse", verbosity=-1, learning_rate=0.03,
         num_leaves=255, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
         min_child_samples=20, lambda_l1=0.2, lambda_l2=0.5, n_estimators=4000),
    dict(objective="regression", metric="rmse", verbosity=-1, learning_rate=0.05,
         num_leaves=63, feature_fraction=0.8, bagging_fraction=0.9, bagging_freq=1,
         min_child_samples=50, lambda_l1=0.05, lambda_l2=0.05, n_estimators=2500),
]
cb_params = [
    dict(iterations=2000, learning_rate=0.04, depth=8, loss_function="RMSE", eval_metric="RMSE",
         l2_leaf_reg=3., rsm=0.7, subsample=0.85, verbose=0, thread_count=-1),
    dict(iterations=3000, learning_rate=0.03, depth=10, loss_function="RMSE", eval_metric="RMSE",
         l2_leaf_reg=5., rsm=0.6, subsample=0.8, verbose=0, thread_count=-1),
]
ridge_params = dict(alpha=5.)


def sg_smooth(df_pred, col="pred", win=21, poly=3):
    result = df_pred.copy()
    for wid, grp in result.groupby("well"):
        idx = grp.index
        vals = grp[col].to_numpy(float)
        n = len(vals)
        if n >= 7:
            w = min(win, n if n % 2 == 1 else n - 1)
            if w % 2 == 0:
                w -= 1
            w = max(w, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
            result.loc[idx, col] = savgol_filter(vals, w, poly)
    return result


def kalman_smooth(vals, r_vec=None, q=0.01):
    n = len(vals)
    x = np.zeros(n)
    p = np.zeros(n)
    r_arr = np.full(n, 1.0) if r_vec is None else np.maximum(r_vec ** 2, 1e-6)
    x[0] = vals[0]
    p[0] = r_arr[0]
    for i in range(1, n):
        xp = x[i - 1]
        pp = p[i - 1] + q
        k = pp / (pp + r_arr[i])
        x[i] = xp + k * (vals[i] - xp)
        p[i] = (1. - k) * pp
    xs = np.zeros(n)
    ps = np.zeros(n)
    xs[-1] = x[-1]
    ps[-1] = p[-1]
    for i in range(n - 2, -1, -1):
        pp = p[i] + q
        g = p[i] / pp
        xs[i] = x[i] + g * (xs[i + 1] - x[i])
        ps[i] = p[i] + g ** 2 * (ps[i + 1] - pp)
    return xs.astype(np.float32)


def apply_pp(pred_df, alpha=1.0, tau=85., w_pf=0.0):
    out = []
    for wid, grp in pred_df.groupby("well"):
        md = grp["md_since"].to_numpy(np.float32)
        base = grp["ridge_pred"].to_numpy(np.float32)
        lkt = float(grp["last_known_tvt"].iloc[0])
        w = np.exp(-md / max(float(tau), 1.)).astype(np.float32)
        blended = base * alpha
        blended = w * lkt + (1. - w) * blended
        if w_pf > 0. and "pf_ancc" in grp.columns:
            pf = grp["pf_ancc"].to_numpy(np.float32)
            blended = (1. - w_pf) * blended + w_pf * pf
        out.append(pd.Series(blended, index=grp.index))
    return pd.concat(out).reindex(pred_df.index).astype(np.float32)


def _optuna_pp_params(pred_df, n_trials=50):
    def obj(trial):
        alpha = trial.suggest_float("alpha", 0.7, 1.3)
        tau = trial.suggest_float("tau", 40., 150.)
        preds = apply_pp(pred_df, alpha=alpha, tau=tau, w_pf=0.0)
        mask = pred_df["target"].notna()
        return float(root_mean_squared_error(pred_df.loc[mask, "target"], preds[mask]))
    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_pipeline_a(train_df, test_df, features, artifacts_path=None):
    X = train_df[features].values.astype(np.float32)
    y = train_df["target"].values.astype(np.float32)
    g = train_df["well"].values
    X_te = test_df[features].values.astype(np.float32)
    cv = GroupKFold(n_splits=CFG.n_splits)
    oof = {}
    te_preds = {}

    for pi, p in enumerate(lgb_params):
        oof_p = np.zeros(len(X), np.float32)
        te_p = np.zeros(len(X_te), np.float32)
        for fold, (tri, vai) in enumerate(cv.split(X, y, g)):
            ds_tr = lgb.Dataset(X[tri], y[tri])
            ds_va = lgb.Dataset(X[vai], y[vai], reference=ds_tr)
            if artifacts_path:
                mf = Path(artifacts_path) / f"lgb_{pi}_fold{fold}.lgb"
                if mf.exists():
                    m = lgb.Booster(model_file=str(mf))
                    oof_p[vai] = m.predict(X[vai])
                    te_p += m.predict(X_te) / CFG.n_splits
                    continue
            m = lgb.train(p, ds_tr, valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(100, verbose=False),
                                     lgb.log_evaluation(-1)])
            oof_p[vai] = m.predict(X[vai])
            te_p += m.predict(X_te) / CFG.n_splits
            if artifacts_path:
                m.save_model(str(Path(artifacts_path) / f"lgb_{pi}_fold{fold}.lgb"))
        oof[f"lgb_{pi}"] = oof_p
        te_preds[f"lgb_{pi}"] = te_p
        print(f"LGB-{pi} OOF {root_mean_squared_error(y, oof_p):.4f}")

    for pi, p in enumerate(cb_params):
        oof_p = np.zeros(len(X), np.float32)
        te_p = np.zeros(len(X_te), np.float32)
        for fold, (tri, vai) in enumerate(cv.split(X, y, g)):
            m = cb.CatBoostRegressor(**p)
            m.fit(X[tri], y[tri], eval_set=(X[vai], y[vai]),
                  early_stopping_rounds=100, verbose=0)
            oof_p[vai] = m.predict(X[vai])
            te_p += m.predict(X_te) / CFG.n_splits
        oof[f"cb_{pi}"] = oof_p
        te_preds[f"cb_{pi}"] = te_p
        print(f"CB-{pi} OOF {root_mean_squared_error(y, oof_p):.4f}")

    xgb_p = dict(objective="reg:squarederror", eval_metric="rmse", verbosity=0,
                 n_estimators=2000, learning_rate=0.04, max_depth=7,
                 subsample=0.85, colsample_bytree=0.7, reg_lambda=1., reg_alpha=0.1,
                 tree_method="hist", device="cpu")
    oof_p = np.zeros(len(X), np.float32)
    te_p = np.zeros(len(X_te), np.float32)
    for fold, (tri, vai) in enumerate(cv.split(X, y, g)):
        m = xgb.XGBRegressor(**xgb_p)
        m.fit(X[tri], y[tri], eval_set=[(X[vai], y[vai])],
              early_stopping_rounds=100, verbose=0)
        oof_p[vai] = m.predict(X[vai])
        te_p += m.predict(X_te) / CFG.n_splits
    oof["xgb"] = oof_p
    te_preds["xgb"] = te_p
    print(f"XGB OOF {root_mean_squared_error(y, oof_p):.4f}")

    et_p = dict(n_estimators=500, max_depth=None, min_samples_leaf=5,
                max_features=0.7, n_jobs=-1, random_state=CFG.seed)
    oof_p = np.zeros(len(X), np.float32)
    te_p = np.zeros(len(X_te), np.float32)
    for fold, (tri, vai) in enumerate(cv.split(X, y, g)):
        m = ExtraTreesRegressor(**et_p)
        m.fit(X[tri], y[tri])
        oof_p[vai] = m.predict(X[vai])
        te_p += m.predict(X_te) / CFG.n_splits
    oof["et"] = oof_p
    te_preds["et"] = te_p
    print(f"ET OOF {root_mean_squared_error(y, oof_p):.4f}")

    oof_df = pd.DataFrame(oof)
    te_df = pd.DataFrame(te_preds)
    scaler = StandardScaler()
    oof_s = scaler.fit_transform(oof_df)
    te_s = scaler.transform(te_df)
    ridge = Ridge(**ridge_params)
    ridge.fit(oof_s, y)
    oof_ridge = ridge.predict(oof_s).astype(np.float32)
    te_ridge = ridge.predict(te_s).astype(np.float32)
    print(f"Ridge OOF {root_mean_squared_error(y, oof_ridge):.4f}")

    pred_tr = train_df[["well", "md_since", "last_known_tvt", "pf_ancc", "target"]].copy()
    pred_tr["ridge_pred"] = oof_ridge
    pred_tr["target"] = y
    best_pp = _optuna_pp_params(pred_tr, n_trials=50)
    print(f"Optuna pp_params: {best_pp}")

    pred_tr["pp_pred"] = apply_pp(pred_tr, **best_pp, w_pf=0.0)
    pred_tr_sg = sg_smooth(pred_tr, col="pp_pred")
    print(f"OOF RMSE after PP+SG: {root_mean_squared_error(y, pred_tr_sg['pp_pred']):.4f}")

    r_vec = test_df["pf_ancc_std"].to_numpy(np.float32) if "pf_ancc_std" in test_df.columns else None

    pred_te = test_df[["well", "id", "md_since", "last_known_tvt", "pf_ancc"]].copy()
    pred_te["ridge_pred"] = te_ridge
    pred_te["pp_pred"] = apply_pp(pred_te, **best_pp, w_pf=0.0)
    pred_te_sg = sg_smooth(pred_te, col="pp_pred")
    for wid, grp in pred_te_sg.groupby("well"):
        idx = grp.index
        rv = r_vec[idx] if r_vec is not None else None
        pred_te_sg.loc[idx, "pp_pred"] = kalman_smooth(grp["pp_pred"].values, rv)

    sub_1_rows = []
    for wid, grp in pred_te_sg.groupby("well"):
        lkt = float(grp["last_known_tvt"].iloc[0])
        tvt = lkt + grp["pp_pred"].to_numpy(float)
        for id_, tv in zip(grp["id"], tvt):
            sub_1_rows.append({"ID": id_, "TVT": tv})
    return pd.DataFrame(sub_1_rows)


_FI_B = None
_DI_B = None


def init_imputers_b(train_well_ids, data_dir):
    global _FI_B, _DI_B
    _FI_B = FormationPlaneKNN(train_well_ids, data_dir)
    _DI_B = DenseANCCImputer(train_well_ids, data_dir)


def build_well_b(hw_path, tw_path, is_train, likpf_map=None):
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path).sort_values("TVT")
    except Exception:
        return None
    if is_train and "TVT" not in hw.columns:
        return None
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    if is_train and hw["TVT"].isna().all():
        return None
    tw_tvt = tw["TVT"].to_numpy(np.float32)
    tw_gr = tw["GR"].to_numpy(np.float32)
    if len(tw_tvt) < 3:
        return None

    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0:
        return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
    pf_use = pf_a.astype(np.float32)
    std_use = std_a.astype(np.float32)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))

    lk = kn.iloc[-1]
    last_tvt = float(lk["TVT_input"])
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)
    traj_fs = _well_traj_features(kn)

    bpaths = {}
    for (bs, mc, es, r, tag) in BEAMS:
        bpaths[tag] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    beam_ref = (bpaths["cons"] + bpaths["sm5"]) / 2.

    ktvt = kn["TVT_input"].to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]
    sc15, sc15s = sc_res[1]
    sc25, sc25s = sc_res[2]
    sc_trust = float(np.clip(len(kn) / 200., 0., 0.6))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc_ens

    dtw8, dtw8s = dtw_tvt_estimate(kgr, ktvt, hgr, hw=8, stride=3)
    dtw15, dtw15s = dtw_tvt_estimate(kgr, ktvt, hgr, hw=15, stride=3)
    dtw25, dtw25s = dtw_tvt_estimate(kgr, ktvt, hgr, hw=25, stride=3)

    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
    a_cal, b_cal = affine_cal(kgr, tw_at_k)
    kmd = kn["MD"].to_numpy(np.float32)
    kz = kn["Z"].to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_k) ** 2)))
    slp_all = robust_slope(kmd, ktvt)
    slp_50 = robust_slope(kmd[-50:], ktvt[-50:])
    slp_z = robust_slope(kz, ktvt)

    swid = wid if is_train else None
    xy_ev = ev[["X", "Y"]].to_numpy(np.float64)
    xy_kn = kn[["X", "Y"]].to_numpy(np.float64)
    form_ev, knn_d = _FI_B.impute(xy_ev, self_wid=swid)
    form_kn, _ = _FI_B.impute(xy_kn, self_wid=swid)
    z_kn = kn["Z"].to_numpy(np.float32)
    z_ev = ev["Z"].to_numpy(np.float32)

    tvt_fs = {}
    form_rmse = {}
    form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        tvt_f = (-z_ev + form_ev[:, fi2] + b_full).astype(np.float32)
        tvt_fw = (-z_ev + form_ev[:, fi2] + b_wls).astype(np.float32)
        tvt_f50 = (-z_ev + form_ev[:, fi2] + b_late).astype(np.float32)
        tvt_fs[f"tvtF_{fn}"] = tvt_f
        tvt_fs[f"tvtFw_{fn}"] = tvt_fw
        tvt_fs[f"tvtF50_{fn}"] = tvt_f50
        tvt_fs[f"bw_{fn}"] = np.float32(b_full)
        tvt_fs[f"bww_{fn}"] = np.float32(b_wls)
        tvt_fs[f"bw50_{fn}"] = np.float32(b_late)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-z_kn + form_kn[:, fi2] + b_full)) ** 2)))
        form_list.append(tvt_f)

    fs = np.stack(form_list, 1)
    form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
    form_std_d = fs.std(1).astype(np.float32)

    d_ancc, d_std, d_dist = _DI_B.impute(xy_ev, self_wid=swid)
    d_kn, d_std_kn, _ = _DI_B.impute(xy_kn, self_wid=swid)
    res_kn = ktvt + z_kn - d_kn
    b_d = float(np.median(ktvt + z_kn - d_kn))
    _, _, _, b_dl, b_dw = seg_b_well(ktvt, z_kn, d_kn)
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
    tvt_densew = (-z_ev + d_ancc + b_dw).astype(np.float32)
    tvt_dense50 = (-z_ev + d_ancc + b_dl).astype(np.float32)
    d_rmse = float(np.sqrt(np.mean(res_kn ** 2)))
    d_bias = float(np.mean(res_kn))

    all_sigs = [pf_use] + list(bpaths.values()) + [sc8, sc15, sc25, sc_ens, tvt_fs["tvtF_ANCC"], tvt_dense]
    sig_std = np.stack(all_sigs, 1).std(1).astype(np.float32)
    sig_mean = (np.stack(all_sigs, 1).mean(1) - last_tvt).astype(np.float32)

    gr_s = pd.Series(gr_full.values)
    rolls = {}
    for w in [5, 21, 51, 101]:
        r = gr_s.rolling(w, center=True, min_periods=1)
        rolls[f"grm{w}"] = r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f"grs{w}"] = r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1, 5, 15, 30]:
        rolls[f"glag{lag}"] = gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f"glead{lag}"] = gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1 = gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_d2 = gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)

    hmd = ev["MD"].to_numpy(np.float32)
    md_since = hmd - float(lk["MD"])
    slp_b_all = (last_tvt + slp_all * md_since).astype(np.float32)
    slp_b_50 = (last_tvt + slp_50 * md_since).astype(np.float32)

    mdd = hw["MD"].diff().replace(0, np.nan)
    dzdmd = (hw["Z"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw["X"].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw["Y"].diff() / mdd).iloc[ev.index].values.astype(np.float32)

    dzdmd_s = pd.Series(dzdmd)
    dtvt_rolls = {}
    for w in [5, 15, 30]:
        dtvt_rolls[f"dzdmd_mean{w}"] = dzdmd_s.rolling(w, center=True, min_periods=1).mean().values.astype(np.float32)
        dtvt_rolls[f"dzdmd_std{w}"] = dzdmd_s.rolling(w, center=True, min_periods=1).std().fillna(0).values.astype(np.float32)

    nh = len(ev)
    frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)

    def sc(v):
        return np.full(nh, np.float32(v), np.float32)

    likpf_feats = {}
    if likpf_map and wid in likpf_map:
        lrow = likpf_map[wid]
        for key in ["pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12"]:
            if key in lrow:
                likpf_feats[f"likpf_{key}_d"] = (np.asarray(lrow[key], np.float32) - last_tvt)
        if "pf_best_ll" in lrow:
            likpf_feats["likpf_best_ll"] = sc(float(lrow["pf_best_ll"]))
        if "pf_ll_spread" in lrow:
            likpf_feats["likpf_ll_spread"] = sc(float(lrow["pf_ll_spread"]))
        if "pf_pt_std" in lrow:
            likpf_feats["likpf_pt_std"] = np.asarray(lrow["pf_pt_std"], np.float32)

    feats = {
        "well": wid, "id": [f"{wid}_{i}" for i in ev.index],
        "last_known_tvt": sc(last_tvt),
        "pf_ancc": pf_use, "pf_ancc_std": std_use,
        "pf_ancc_delta": (pf_use - last_tvt).astype(np.float32),
        "pf_z": (pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        "pf_z_delta": ((pf_z - last_tvt).astype(np.float32) if has_z else sc(0.)),
        "pf_vs_z": ((pf_use - pf_z.astype(np.float32)) if has_z else sc(0.)),
        **{f"beam_{t}_d": (p - np.float32(last_tvt)).astype(np.float32) for t, p in bpaths.items()},
        "beam_mean_d": np.stack([(p - last_tvt) for p in bpaths.values()], 1).mean(1).astype(np.float32),
        "beam_std_d": np.stack([(p - last_tvt) for p in bpaths.values()], 1).std(1).astype(np.float32),
        "beam_med_d": np.median(np.stack([(p - last_tvt) for p in bpaths.values()], 1), 1).astype(np.float32),
        "sc8_d": (sc8 - np.float32(last_tvt)).astype(np.float32), "sc8_sc": sc8s,
        "sc15_d": (sc15 - np.float32(last_tvt)).astype(np.float32), "sc15_sc": sc15s,
        "sc25_d": (sc25 - np.float32(last_tvt)).astype(np.float32), "sc25_sc": sc25s,
        "sc_ens_d": (sc_ens - np.float32(last_tvt)).astype(np.float32),
        "sc_trust": sc(sc_trust), "hyb_d": (hyb_ref - np.float32(last_tvt)).astype(np.float32),
        "dtw8_d": (dtw8 - np.float32(last_tvt)).astype(np.float32), "dtw8_sc": dtw8s,
        "dtw15_d": (dtw15 - np.float32(last_tvt)).astype(np.float32), "dtw15_sc": dtw15s,
        "dtw25_d": (dtw25 - np.float32(last_tvt)).astype(np.float32), "dtw25_sc": dtw25s,
        "sig_std": sig_std, "sig_mean_d": sig_mean,
        **tvt_fs,
        **{f"frm_rmse_{fn}": sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d": form_mean_d, "form_std_d": form_std_d,
        "dense_ancc": d_ancc, "dense_std": d_std, "dense_dist": d_dist,
        "tvt_dense_d": (tvt_dense - last_tvt).astype(np.float32),
        "tvt_densew_d": (tvt_densew - last_tvt).astype(np.float32),
        "tvt_dense50_d": (tvt_dense50 - last_tvt).astype(np.float32),
        "dense_rmse": sc(d_rmse), "dense_bias": sc(d_bias),
        "pf_vs_spatial": (pf_use - tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "pf_vs_dense": (pf_use - tvt_dense).astype(np.float32),
        "cal_a": sc(a_cal), "cal_b": sc(b_cal),
        "pfx_rmse": sc(pfx_rmse), "known_len": sc(len(kn)), "eval_len": sc(nh),
        "slp_all": sc(slp_all), "slp_50": sc(slp_50), "slp_z": sc(slp_z),
        "slp_b_d_all": (slp_b_all - last_tvt).astype(np.float32),
        "slp_b_d_50": (slp_b_50 - last_tvt).astype(np.float32),
        "ktvt_range": sc(float(np.ptp(ktvt))), "ktvt_std": sc(float(ktvt.std())),
        "md_since": md_since, "frac": frac, "frac2": frac ** 2, "sqrt_frac": np.sqrt(frac),
        "z": z_ev,
        "dx": (ev["X"] - float(lk["X"])).to_numpy(np.float32),
        "dy": (ev["Y"] - float(lk["Y"])).to_numpy(np.float32),
        "dz": (z_ev - float(lk["Z"])).astype(np.float32),
        "dxy": np.sqrt((ev["X"] - float(lk["X"])) ** 2 + (ev["Y"] - float(lk["Y"])) ** 2).to_numpy(np.float32),
        "dzdmd": dzdmd, "dxdmd": dxdmd, "dydmd": dydmd,
        "gr": hgr, "gr_d1": gr_d1, "gr_d2": gr_d2,
        **{f"tda{int(o)}": hgr - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr)) for o in ANCH_OFFS},
        **{f"tdbc{int(o)}": hgr - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f"tdsc{int(o)}": hgr - np.interp(sc_ens + o, tw_tvt, tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f"tdpf{int(o)}": hgr - np.interp(pf_use + o, tw_tvt, tw_gr).astype(np.float32) for o in PF_OFFS},
        "tw_range": sc(float(np.ptp(tw_tvt))), "tw_gr_mean": sc(float(tw_gr.mean())),
        **{k: sc(v) for k, v in traj_fs.items()},
        **dtvt_rolls,
        **likpf_feats,
    }
    for k, v in rolls.items():
        feats[k] = v
    result = pd.DataFrame(feats)
    if is_train:
        if "TVT" not in ev.columns or ev["TVT"].isna().all():
            return None
        result["target"] = (ev["TVT"].to_numpy(np.float32) - np.float32(last_tvt))
    return result


def build_likpf(hw_paths, tw_paths, n_particles=CFG.PF_PARTICLES, n_seeds=CFG.PF_SEEDS):
    def _single(hp, tp):
        wid = Path(hp).stem.replace("__horizontal_well", "")
        try:
            hw = pd.read_csv(hp)
            tw = pd.read_csv(tp).sort_values("TVT")
        except Exception:
            return None
        kn = hw[hw["TVT_input"].notna()]
        ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0 or len(kn) < 10:
            return None
        out, ev_idx, q = lik_pf(hw, tw, n_particles=n_particles, n_seeds=n_seeds,
                                scales=(3., 5., 8., 12.), with_quality=True)
        if not out:
            return None
        row = {"wid": wid, "ev_idx": ev_idx}
        for k, v in out.items():
            row[k] = v
        row.update(q)
        return row
    results = Parallel(n_jobs=CFG.n_jobs, prefer="threads")(
        delayed(_single)(hp, tp) for hp, tp in zip(hw_paths, tw_paths))
    return {r["wid"]: r for r in results if r is not None}


def _device():
    use_gpu = CFG.USE_GPU
    if use_gpu == "auto":
        try:
            r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            return "GPU" if r.returncode == 0 else "CPU"
        except Exception:
            return "CPU"
    return "GPU" if str(use_gpu).upper() in ("1", "TRUE", "GPU") else "CPU"


def lgb_configs_b(dev):
    return [
        dict(objective="regression", metric="rmse", verbosity=-1, learning_rate=0.03,
             num_leaves=255, feature_fraction=0.65, bagging_fraction=0.8, bagging_freq=1,
             min_child_samples=20, lambda_l1=0.2, lambda_l2=0.5, n_estimators=5000,
             device="gpu" if dev == "GPU" else "cpu"),
        dict(objective="regression", metric="rmse", verbosity=-1, learning_rate=0.04,
             num_leaves=127, feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1,
             min_child_samples=30, lambda_l1=0.1, lambda_l2=0.1, n_estimators=4000,
             device="gpu" if dev == "GPU" else "cpu"),
    ]


def cb_configs_b(dev):
    base = dict(loss_function="RMSE", eval_metric="RMSE", verbose=0,
                task_type="GPU" if dev == "GPU" else "CPU")
    return [
        dict(iterations=4000, learning_rate=0.03, depth=9, l2_leaf_reg=4., rsm=0.65, subsample=0.8, **base),
        dict(iterations=3000, learning_rate=0.04, depth=7, l2_leaf_reg=3., rsm=0.75, subsample=0.85, **base),
    ]


def train_stack_b(train_df, test_df, features):
    X = train_df[features].values.astype(np.float32)
    y = train_df["target"].values.astype(np.float32)
    g = train_df["well"].values
    X_te = test_df[features].values.astype(np.float32)
    cv = GroupKFold(n_splits=CFG.n_splits)
    dev = _device()
    oof = {}
    te_preds = {}

    for pi, p in enumerate(lgb_configs_b(dev)):
        oof_p = np.zeros(len(X), np.float32)
        te_p = np.zeros(len(X_te), np.float32)
        for fold, (tri, vai) in enumerate(cv.split(X, y, g)):
            ds_tr = lgb.Dataset(X[tri], y[tri])
            ds_va = lgb.Dataset(X[vai], y[vai], reference=ds_tr)
            m = lgb.train(p, ds_tr, valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(-1)])
            oof_p[vai] = m.predict(X[vai])
            te_p += m.predict(X_te) / CFG.n_splits
        oof[f"lgb_{pi}"] = oof_p
        te_preds[f"lgb_{pi}"] = te_p

    for pi, p in enumerate(cb_configs_b(dev)):
        oof_p = np.zeros(len(X), np.float32)
        te_p = np.zeros(len(X_te), np.float32)
        for fold, (tri, vai) in enumerate(cv.split(X, y, g)):
            m = cb.CatBoostRegressor(**p)
            m.fit(X[tri], y[tri], eval_set=(X[vai], y[vai]), early_stopping_rounds=150, verbose=0)
            oof_p[vai] = m.predict(X[vai])
            te_p += m.predict(X_te) / CFG.n_splits
        oof[f"cb_{pi}"] = oof_p
        te_preds[f"cb_{pi}"] = te_p

    xgb_cfg = dict(objective="reg:squarederror", eval_metric="rmse", verbosity=0,
                   n_estimators=3000, learning_rate=0.03, max_depth=8,
                   subsample=0.8, colsample_bytree=0.65, reg_lambda=2., reg_alpha=0.2,
                   tree_method="hist", device="cuda" if dev == "GPU" else "cpu")
    oof_p = np.zeros(len(X), np.float32)
    te_p = np.zeros(len(X_te), np.float32)
    for fold, (tri, vai) in enumerate(cv.split(X, y, g)):
        m = xgb.XGBRegressor(**xgb_cfg)
        m.fit(X[tri], y[tri], eval_set=[(X[vai], y[vai])], early_stopping_rounds=150, verbose=0)
        oof_p[vai] = m.predict(X[vai])
        te_p += m.predict(X_te) / CFG.n_splits
    oof["xgb"] = oof_p
    te_preds["xgb"] = te_p

    oof_df = pd.DataFrame(oof)
    te_df = pd.DataFrame(te_preds)
    scaler = StandardScaler()
    oof_s = scaler.fit_transform(oof_df)
    te_s = scaler.transform(te_df)
    ridge = Ridge(alpha=5.)
    ridge.fit(oof_s, y)
    return ridge.predict(oof_s).astype(np.float32), ridge.predict(te_s).astype(np.float32), scaler, ridge


class PP:
    def __init__(self, alpha=1.0, tau=85.0, w_pf=0.0, w_sub1=0.60,
                 sub2_scale="scale_5", sg_win=61, sg_poly=3):
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.w_pf = float(w_pf)
        self.w_sub1 = float(w_sub1)
        self.sub2_scale = sub2_scale
        self.sg_win = int(sg_win)
        self.sg_poly = int(sg_poly)
        self._tau_w = {}

    def warmup(self, md_since, tau):
        key = (id(md_since), float(tau))
        if key not in self._tau_w:
            self._tau_w[key] = np.exp(-np.asarray(md_since, float) / max(float(tau), 1.)).astype(np.float32)
        return self._tau_w[key]

    def make_prediction(self, df, model_delta, likpf):
        out = []
        for wid, grp in df.groupby("well"):
            idx = grp.index
            md_since = grp["md_since"].to_numpy(np.float32)
            lkt = float(grp["last_known_tvt"].iloc[0])
            w = self.warmup(md_since, self.tau)
            base = lkt + model_delta[idx].astype(np.float32) * self.alpha
            base = w * lkt + (1. - w) * base
            if self.w_pf > 0. and "pf_ancc" in grp.columns:
                pf = grp["pf_ancc"].to_numpy(np.float32)
                base = (1. - self.w_pf) * base + self.w_pf * pf
            if likpf is not None and wid in likpf:
                lrow = likpf[wid]
                sc_key = f"pf_{self.sub2_scale}"
                if sc_key in lrow:
                    lp = np.asarray(lrow[sc_key], np.float32)
                    if len(lp) == len(idx):
                        base = (1. - self.w_sub1) * base + self.w_sub1 * lp
            n = len(base)
            if n >= 7:
                win = min(self.sg_win, n if n % 2 == 1 else n - 1)
                if win % 2 == 0:
                    win -= 1
                win = max(win, self.sg_poly + 2 if (self.sg_poly + 2) % 2 == 1 else self.sg_poly + 3)
                base = savgol_filter(base, win, self.sg_poly)
            out.append(pd.Series(base.astype(np.float32), index=idx))
        return pd.concat(out).reindex(df.index)


def _robfit(s, y, deg=4):
    x = s.to_numpy(float)
    yv = y.to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(yv)
    if mask.sum() < deg + 3:
        return pd.Series(yv)
    xm = x[mask]
    ym = yv[mask]
    xn = (xm - xm.mean()) / (xm.std() + 1e-9)
    P = np.column_stack([xn ** d for d in range(deg, -1, -1)])
    hr = HuberRegressor(epsilon=1.35, max_iter=200)
    hr.fit(P, ym)
    xn_all = (x - xm.mean()) / (xm.std() + 1e-9)
    P_all = np.column_stack([xn_all ** d for d in range(deg, -1, -1)])
    return pd.Series(hr.predict(P_all), index=y.index)


def main_pipeline_b(test_hw_paths, test_tw_paths):
    sub2_rows = {}
    for hw_path, tw_path in zip(test_hw_paths, test_tw_paths):
        wid = Path(hw_path).stem.replace("__horizontal_well", "")
        try:
            hw = pd.read_csv(hw_path)
            tw = pd.read_csv(tw_path).sort_values("TVT")
        except Exception:
            continue
        kn = hw[hw["TVT_input"].notna()]
        ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0:
            for i in kn.index:
                sub2_rows[f"{wid}_{i}"] = float(kn.loc[i, "TVT_input"])
            continue
        if len(kn) == 0:
            continue
        pf_by_scale = run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES,
                                                 n_particles=CFG.PF_PARTICLES, n_seeds=CFG.PF_SEEDS)
        tvt_beam = run_beam_ensemble(hw, tw)
        code, variant, n_eval, z_span = selector_well_code(hw)
        last_known_tvt = float(kn.iloc[-1]["TVT_input"])
        tvt_pred = apply_selector_variant(variant, pf_by_scale, tvt_beam[list(ev.index)], last_known_tvt)
        for i, tvt in zip(ev.index, tvt_pred):
            sub2_rows[f"{wid}_{i}"] = float(tvt)
    return pd.DataFrame({"ID": list(sub2_rows.keys()), "TVT": list(sub2_rows.values())})


def run(data_dir=None, artifacts_dir=None, out_dir=None, w_sp45=None):
    import preprocess as _pre
    data_dir = Path(data_dir or CFG.DATA)
    artifacts_dir = Path(artifacts_dir or CFG.ARTIFACTS)
    out_dir = Path(out_dir or CFG.OUT)
    w_sp45 = float(w_sp45) if w_sp45 is not None else CFG.BLEND_W_SP45

    train_hw_paths = sorted((data_dir / "train").glob("*__horizontal_well.csv"))
    test_hw_paths = sorted((data_dir / "test").glob("*__horizontal_well.csv"))

    def tw_path(hw):
        return Path(str(hw).replace("__horizontal_well", "__typewell"))

    test_tw_paths = [tw_path(p) for p in test_hw_paths]
    train_well_ids = [p.stem.replace("__horizontal_well", "") for p in train_hw_paths]

    train_csv = artifacts_dir / "train_features.csv"
    if train_csv.exists():
        print("Loading train features from artifact...")
        train_df = pd.read_csv(train_csv)
    else:
        _pre._FI = FormationPlaneKNN(train_well_ids, data_dir / "train")
        _pre._DI = DenseANCCImputer(train_well_ids, data_dir / "train")
        train_df = build_dataset(train_hw_paths, is_train=True, label="train")
        train_df.to_csv(out_dir / "train_features.csv", index=False)

    _pre._FI = FormationPlaneKNN(train_well_ids, data_dir / "train")
    _pre._DI = DenseANCCImputer(train_well_ids, data_dir / "train")
    test_df = build_dataset(test_hw_paths, is_train=False, label="test")

    features = [c for c in train_df.columns if c not in {"well", "id", "target"}]
    sub_1 = train_pipeline_a(train_df, test_df, features, str(artifacts_dir))
    sub_2 = main_pipeline_b([str(p) for p in test_hw_paths], [str(p) for p in test_tw_paths])

    merged = sub_1.merge(sub_2, on="ID", suffixes=("_a", "_b"))
    merged["TVT_blend"] = CFG.BLEND_W_A * merged["TVT_a"] + CFG.BLEND_W_B * merged["TVT_b"]

    sp45_sub = merged[["ID", "TVT_blend"]].rename(columns={"TVT_blend": "TVT"}).copy()
    sp45_id = sp45_sub["ID"].str.rsplit("_", n=1, expand=True)
    sp45_sub["well"] = sp45_id[0]
    sp45_sub["row_idx"] = sp45_id[1].astype(int)
    sp45_sub = sp45_sub.sort_values(["well", "row_idx"])
    all_parts = []
    for wid, grp in sp45_sub.groupby("well"):
        s = grp["row_idx"].reset_index(drop=True).astype(float)
        tvt = grp["TVT"].reset_index(drop=True)
        tvt_z = pd.Series(np.zeros(len(s)), name="z")
        for hp in test_hw_paths:
            if hp.stem.replace("__horizontal_well", "") == wid:
                hw_tmp = pd.read_csv(hp)
                ev_tmp = hw_tmp[hw_tmp["TVT_input"].isna()]
                if "Z" in ev_tmp.columns:
                    tvt_z = pd.Series(ev_tmp["Z"].values[:len(s)])
        anchor = float(tvt.iloc[0])
        fit_target = tvt + tvt_z - anchor
        fitted = _robfit(s, fit_target, deg=4)
        blended_tvt = 0.25 * (tvt + tvt_z) + 0.75 * (fitted + anchor - tvt_z.values)
        grp_out = grp.copy()
        grp_out["TVT"] = blended_tvt.values
        all_parts.append(grp_out)
    sp45_sub_proj = pd.concat(all_parts)[["ID", "TVT"]]
    sp45_sub_proj.to_csv(out_dir / "sp45_projection_submission.csv", index=False)
    print("Saved sp45_projection_submission.csv")

    fleongg_path = artifacts_dir / "fleongg_submission.csv"
    if fleongg_path.exists():
        fl = pd.read_csv(fleongg_path)
        final = sp45_sub_proj.merge(fl, on="ID", suffixes=("_sp45", "_fl"))
        final["TVT"] = w_sp45 * final["TVT_sp45"] + (1. - w_sp45) * final["TVT_fl"]
        final[["ID", "TVT"]].to_csv(out_dir / "submission.csv", index=False)
        print(f"Final blend saved (w_sp45={w_sp45})")
    else:
        sp45_sub_proj.to_csv(out_dir / "submission.csv", index=False)
        print("No fleongg artifact found; using sp45 as final submission.")

    guarded_override(out_dir / "submission.csv", data_dir, out_dir)


def guarded_override(submission_path, data_dir, out_dir, rmse_threshold=1.0):
    sub = pd.read_csv(submission_path)
    sub["well"] = sub["ID"].str.rsplit("_", n=1, expand=True)[0]
    test_hw_paths_map = {
        Path(p).stem.replace("__horizontal_well", ""): str(p)
        for p in sorted((data_dir / "test").glob("*__horizontal_well.csv"))
    }
    changed = 0
    for wid, grp in sub.groupby("well"):
        if wid not in test_hw_paths_map:
            continue
        tr_path = data_dir / "train" / f"{wid}__horizontal_well.csv"
        if not tr_path.exists():
            continue
        tr = pd.read_csv(tr_path)
        if "TVT" not in tr.columns or tr["TVT"].isna().all():
            continue
        kn_tr = tr[tr["TVT"].notna()]
        if len(kn_tr) == 0:
            continue
        hw_te = pd.read_csv(test_hw_paths_map[wid])
        ev_te = hw_te[hw_te["TVT_input"].isna()]
        if len(ev_te) == 0:
            continue
        tr_kn = kn_tr.tail(30)
        form_preds = []
        for fn in FORMATIONS:
            if fn in tr_kn.columns and fn in kn_tr.columns:
                b = float(np.median(tr_kn["TVT"].values + tr_kn["Z"].values - tr_kn[fn].values))
                if fn in ev_te.columns:
                    fp = (-ev_te["Z"].values + ev_te[fn].values + b).astype(float)
                    form_preds.append(fp)
        if not form_preds:
            continue
        contact_tvt = np.stack(form_preds, 0).mean(0)
        cur_tvt = grp["TVT"].values
        if len(contact_tvt) != len(cur_tvt):
            continue
        rmse = float(root_mean_squared_error(cur_tvt, contact_tvt))
        if rmse < rmse_threshold:
            sub.loc[grp.index, "TVT"] = contact_tvt
            changed += 1
    if changed > 0:
        sub[["ID", "TVT"]].to_csv(submission_path, index=False)
        print(f"Guarded override: {changed} wells updated.")


if __name__ == "__main__":
    run()
