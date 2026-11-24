import numpy as np
import pandas as pd

from grid import measured_cap

PLANTS = ["G1", "G2", "G3"]


def units_online(raw, events, scada):
    plants = pd.read_csv(raw / "system_info" / "plants.csv").set_index("plant")
    uc = pd.read_csv(raw / "system_info" / "unit_commitment.csv")
    uc = uc.assign(hour=uc["time_utc"].str[:13]).set_index("hour")
    hour = events["time_utc"].str[:13].values
    n = np.stack([uc[f"{g}_units_online"].reindex(hour).values for g in PLANTS], 1)
    missing = np.isnan(n)
    for k, g in enumerate(PLANTS):
        per_unit = np.nanmedian(100 * scada[f"P_{g}"].values / n[:, k])
        guess = np.clip(np.ceil(100 * scada[f"P_{g}"].values / per_unit), 1, plants.loc[g, "units"])
        n[:, k] = np.where(missing[:, k], guess, n[:, k])
    return n, missing.any(1)


def inertia(raw, events, n):
    plants = pd.read_csv(raw / "system_info" / "plants.csv").set_index("plant")
    H = n * plants.loc[PLANTS, "unit_inertia_H_s_on_100MVA"].values
    xd = np.tile(plants.loc[PLANTS, "plant_Xd_transient_pu_100MVA"].values, (len(events), 1))
    return H, xd


def capacitor(br, raw, events, scada, X, ch):
    cap = pd.read_csv(raw / "system_info" / "series_compensation.csv")
    logged = events["time_utc"].str[:10].map(dict(zip(cap["date"], cap["L5-7_series_capacitor"] == "in service")))
    th47 = np.nanmedian(X[:, :20, ch.index("theta4")] - X[:, :20, ch.index("theta7")], 1)
    logged = logged.fillna(False).values.astype(bool)
    return measured_cap(br, scada, th47, logged), logged


def accel_ratio(X, ch, events, acc_pred, usable):
    f = np.stack([X[:, :, ch.index(f"f{p}")] for p in "479"], 2)
    meas = (f[:, 24] - f[:, 20]) / 0.04
    ratio = meas / acc_pred.values
    ratio[~usable] = np.nan
    return pd.DataFrame(ratio, columns=PLANTS).where(lambda r: (r > 0.05) & (r < 20))


def line_reference(ratio, events):
    return ratio.groupby(events["faulted_line"].values).median()


def rolling_ratio(cfg, ratio, events, ref):
    rel = ratio.values / ref.reindex(events["faulted_line"].values).values
    rel = np.where((rel > 0.2) & (rel < 5), rel, np.nan)
    time = pd.to_datetime(events["time_utc"]).values.astype("datetime64[s]").astype(float) / 86400
    order = np.argsort(time)
    ts = time[order]
    lo = np.searchsorted(ts, ts - cfg.inertia_days)
    hi = np.searchsorted(ts, ts + cfg.inertia_days, side="right")
    roll = np.full(rel.shape, np.nan)
    sorted_rel = rel[order]
    with np.errstate(all="ignore"):
        for i in range(len(ts)):
            window = sorted_rel[lo[i]:hi[i]]
            ok = np.isfinite(window).sum(0) >= cfg.inertia_min
            roll[order[i]] = np.where(ok, np.nanmedian(window, 0), np.nan)
    return rel, roll


def fault_voltages(X, ch):
    v = np.stack([X[:, 22:25, ch.index(c)] for c in ("V4", "V7", "V9")], 2)
    with np.errstate(all="ignore"):
        return np.nanmean(v, 1)


def fast_share(cfg, events, t_obs, observable):
    time = pd.to_datetime(events["time_utc"]).values.astype("datetime64[s]").astype(float) / 86400
    line = events["faulted_line"].values
    fast = (t_obs < cfg.fast_clear_s).astype(float)
    out = np.full(len(events), np.nan)
    for l in np.unique(line):
        idx = np.where(line == l)[0]
        ok = idx[observable[idx]]
        d = np.abs(time[idx][:, None] - time[ok][None, :]) <= cfg.share_days
        d &= idx[:, None] != ok[None, :]
        cnt = d.sum(1)
        out[idx] = np.where(cnt >= cfg.share_min, (d * fast[ok][None, :]).sum(1) / np.maximum(cnt, 1), np.nan)
    return out


def clearing_cdf(cfg, t_clear, pott):
    grid = np.arange(0, 0.801, 0.005)
    return grid, [np.searchsorted(np.sort(t_clear[pott == k]), grid, side="right") / max((pott == k).sum(), 1)
                  for k in (0, 1)]


def exceed(cdf, share, cut, cct):
    grid, (dist, pott) = cdf
    w = np.clip((share - 0.3) / 0.55, 0, 1)
    w = np.where(np.isnan(w), 0.5, w)
    F = lambda x: w * np.interp(x, grid, pott) + (1 - w) * np.interp(x, grid, dist)
    surv = np.maximum(1 - F(cut), 1e-6)
    return np.clip((1 - F(np.maximum(cut, cct))) / surv, 0, 1)


FIT = ["cct_fit", "cct_reclose", "x_fit", "rf_fit", "fit_rmse", "rel_G1", "rel_G2", "rel_G3"]


def derived(rows, cdf):
    rows = rows.copy()
    early = rows["cut_s"].values < 0.05 - 1e-9
    for c in FIT:
        rows.loc[early, c] = np.nan
    for k, g in enumerate(PLANTS):
        rows[f"slope_per_p_{g}"] = rows[f"slope_f{'479'[k]}"] / rows[f"P_{g}"]
    cleared = rows["cleared"].values == 1
    seen = np.where(cleared, rows["t_clear"].values, rows["cut_s"].values)
    cct = rows["cct_fit"].fillna(rows["cct_prior_mean"]).values
    rows["margin"] = seen - cct
    rows["margin_prior"] = seen - rows["cct_prior_mean"].values
    rows["margin_prior_min"] = seen - rows["cct_prior_min"].values
    rows["past_cct"] = (~cleared & (rows["cut_s"].values > cct)).astype(float)
    p = exceed(cdf, rows["fast_share"].values, rows["cut_s"].values, cct)
    rows["p_exceed"] = np.where(cleared, (rows["t_clear"].values > cct).astype(float), p)
    rows["p_exceed_prior"] = np.where(cleared, (rows["t_clear"].values > rows["cct_prior_mean"].values).astype(float),
                                      exceed(cdf, rows["fast_share"].values, rows["cut_s"].values,
                                             rows["cct_prior_mean"].values))
    return rows
