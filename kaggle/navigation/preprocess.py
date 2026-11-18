import os
from pathlib import Path

import numpy as np
import pandas as pd

from gnss import (LAT0, LON0, ROT0, Ephemeris, doppler_velocity, ecef_to_geodetic, geodetic_to_ecef,
                  rts_1d, sat_states, spp, to_map)

AREAS = ["TST", "TST", "WP", "MK"]
SEG_ROWS = [8673, 1718, 27752, 35942]
CUTOFF = pd.Timestamp("2021-12-03")
VARIANTS = ("all", "no_new")


def find(name):
    roots = [Path("/kaggle/input")] if Path("/kaggle/input").exists() else []
    try:
        roots.append(Path(__file__).resolve().parent.parent / "data")
    except NameError:
        roots.append(Path.cwd() / "data")
    for root in roots:
        for dirpath, dirs, files in os.walk(root):
            if name in files:
                return Path(dirpath) / name
    raise FileNotFoundError(f"{name} not found; attach the competition data and the navigation-ext dataset")


def load():
    tr = pd.read_csv(find("GNSS_raw_train.csv"), encoding="utf-8-sig").dropna(subset=["GPS_Time(s)"])
    te = pd.read_csv(find("GNSS_raw_test.csv"), encoding="utf-8-sig")
    xtr, xte = pd.read_csv(find("ext_host_train.csv")), pd.read_csv(find("ext_host_test.csv"))
    parts = []
    for raw, x, test in ((tr, xtr, False), (te, xte, True)):
        raw = raw.reset_index(drop=True)
        assert len(raw) == len(x)
        assert (np.mod(raw["GPS_Time(s)"].values, 604800) == x["GPS_Time(s)"].values).all()
        assert (raw["Cnr_L1"].values == x["CNR"].values).all()
        d = pd.DataFrame({
            "t": raw["GPS_Time(s)"].astype(np.int64).values, "sat": raw["Satelite_Code"].values,
            "code": raw["Code_L1"].values, "phase": raw["Phase_L1"].values, "dop": raw["Doppler_L1"].values,
            "cn0": raw["Cnr_L1"].values.astype(float), "code2": raw["code_L2"].values,
            "cn02": raw["Cnr_L2"].values.astype(float), "el_h": x["Elevation"].values, "az_h": x["Azimuth"].values,
            "res_h": x["Pr_Residual"].values, "rsse_h": x["Pr_residual_Root_of_sum_of_square_error"].values,
            "sig_h": x["Pr_Residual_cigma"].values, "test": test})
        if test:
            d["seg"], d["label"], d["id"] = 4, np.nan, np.arange(len(d))
        else:
            jump = np.r_[False, np.abs(np.diff(d.t.values)) > 60]
            d["seg"], d["label"], d["id"] = np.cumsum(jump), raw["Label"].values, -1
            assert d.seg.value_counts().sort_index().tolist() == SEG_ROWS, "unexpected train scenario split"
            d["lat_gt"], d["lon_gt"], d["h_gt"] = raw["Latitude"].values, raw["Longitude"].values, raw["Height"].values
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["area"] = np.where(df.seg < 4, np.array(AREAS)[np.minimum(df.seg, 3)], "KLB")
    df["sys"] = df.sat.str[0]
    df["sysi"] = df.sys.map({"G": 0, "R": 1, "E": 2, "C": 3}).astype(int)
    eph = Ephemeris(sorted(str(find(f"brdc_2021{d}.rnx")) for d in ("137", "138", "141", "337")))
    bld = pd.read_csv(find("ext_buildings.csv"))
    return df.sort_values(["seg", "t", "sat"], kind="stable").reset_index(drop=True), eph, bld


def positions(df, eph, log):
    sp, sv, clk, drift, lam = sat_states(eph, df.sat.values, df.t.values.astype(float), df.code.values)
    df["lam"] = lam
    log(f"satellite states: {np.isfinite(sp[:, 0]).mean():.4f} of rows have ephemeris; missing sats "
        f"{sorted(df.sat[~np.isfinite(sp[:, 0])].unique())}")
    iono = eph.iono.get("GPSA"), eph.iono.get("GPSB")
    if iono[0] is None:
        iono = eph.iono.get("BDSA"), eph.iono.get("BDSB")
    iono = None if iono[0] is None else iono
    groups = [(s, t, idx.values) for (s, t), idx in df.groupby(["seg", "t"]).groups.items()]
    seed = geodetic_to_ecef(LAT0, LON0, 10.0)

    def run(isb_by_seg):
        sols, x0 = {}, {}
        for s, t, idx in groups:
            sol = spp(t, sp[idx], clk[idx], df.sys.values[idx], df.code.values[idx], df.cn0.values[idx], iono,
                      isb_by_seg.get(s, {}), x0.get(s, seed))
            if sol is not None and np.linalg.norm(sol["rx"] - seed) < 2e4:
                sols[(s, t)], x0[s] = sol, sol["rx"]
        return sols

    first = run({})
    isb = {}
    for s in df.seg.unique():
        b = pd.DataFrame([v["bias"] for (ss, _), v in first.items() if ss == s])
        ref = "G" if "G" in b else b.columns[0]
        isb[s] = {c: float(np.nanmedian(b[c] - b[ref])) for c in b.columns}
        log(f"seg {s}: inter-system biases (m) {({k: round(v, 1) for k, v in isb[s].items()})}")
    sols = run(isb)
    log(f"SPP solved {len(sols)} of {len(groups)} epochs")

    feats = {k: np.full(len(df), np.nan) for k in ("res", "nres", "loo", "el_o", "az_o", "dres")}
    for s, t, idx in groups:
        if (s, t) in sols:
            for k, kk in (("res", "res"), ("nres", "nres"), ("loo", "loo"), ("el_o", "el"), ("az_o", "az")):
                feats[k][idx] = sols[(s, t)][kk]

    probe = [g for g in groups if (g[0], g[1]) in sols][:: max(1, len(sols) // 400)]
    score = {}
    for sign in (1.0, -1.0):
        r = [doppler_velocity(sp[i], sv[i], drift[i], lam[i], df.dop.values[i], df.cn0.values[i], sols[(s, t)]["rx"],
                              sign)[1] for s, t, i in probe]
        score[sign] = np.nanmedian(np.abs(np.concatenate(r)))
    sign = min(score, key=score.get)
    log(f"Doppler sign: median |normalised residual| {score}; using {sign:+.0f}")
    df["dop_rr"] = sign * -df.lam * df.dop

    rows = []
    for s, t, idx in groups:
        sol = sols.get((s, t))
        v = None
        if sol is not None:
            v, feats["dres"][idx] = doppler_velocity(sp[idx], sv[idx], drift[idx], lam[idx], df.dop.values[idx],
                                                      df.cn0.values[idx], sol["rx"], sign)
        rec = {"seg": s, "t": t, "n_used": sol["n"] if sol else 0}
        if sol is not None:
            lat, lon, hh = ecef_to_geodetic(sol["rx"])
            rec["x_spp"], rec["y_spp"] = to_map(lat, lon)
            rec["h_spp"] = hh
            cov = ROT0 @ sol["cov"] @ ROT0.T
            rec["vx_spp"], rec["vy_spp"] = cov[0, 0], cov[1, 1]
        if v is not None:
            ve = ROT0 @ v
            rec["ve"], rec["vn"] = ve[0], ve[1]
        rows.append(rec)
    ep = pd.DataFrame(rows)
    for c in ("x_spp", "y_spp", "h_spp", "vx_spp", "vy_spp", "ve", "vn"):
        if c not in ep:
            ep[c] = np.nan
    ep["x"], ep["y"] = np.nan, np.nan
    for s, g in ep.groupby("seg"):
        dt = np.r_[1.0, np.diff(g.t.values)].astype(float)
        piece = np.cumsum(dt > 10)
        for p in np.unique(piece):
            m = piece == p
            gi = g.index[m]
            for c, vc, var in (("x", "ve", "vx_spp"), ("y", "vn", "vy_spp")):
                ep.loc[gi, c] = rts_1d(g[c + "_spp"].values[m], np.maximum(g[var].values[m], 4.0),
                                       g[vc].values[m], dt[m], 0.1)
    ep["speed"] = np.hypot(ep.ve, ep.vn)
    ep["heading"] = np.arctan2(ep.ve, ep.vn)
    for k, v in feats.items():
        df[k] = v
    return ep


def gt_epochs(df):
    g = df[~df.test].groupby(["seg", "t"])[["lat_gt", "lon_gt", "h_gt"]].first().reset_index()
    g["x_gt"], g["y_gt"] = to_map(g.lat_gt.values, g.lon_gt.values)
    return g


def building_variant(bld, name, scale=1.0):
    keep = np.isfinite(bld.top_mpd.values)
    if name == "no_new":
        keep &= (pd.to_datetime(bld.date_create, dayfirst=True) <= CUTOFF).values
    b = bld[keep].copy()
    base = b.base_mpd.fillna(b.top_mpd)
    b["top_mpd"] = base + scale * (b.top_mpd - base)
    return b


def epoch_index(df, ep_keys):
    pos = pd.Series(np.arange(len(ep_keys)), index=pd.MultiIndex.from_frame(ep_keys))
    return pos.reindex(pd.MultiIndex.from_frame(df[["seg", "t"]])).values


def margins(df, ek, msk):
    out = np.full(len(df), np.nan)
    ok = np.isfinite(ek)
    b = np.mod(np.floor(df.az_h.values).astype(int), 360)
    out[ok] = df.el_h.values[ok] - np.degrees(msk[ek[ok].astype(int), b[ok]])
    return out


def gt_margins(df, city, h_ant):
    gt, tr = gt_epochs(df), df[~df.test]
    return margins(tr, epoch_index(tr, gt[["seg", "t"]]), city.masks_at(gt.x_gt.values, gt.y_gt.values, h_ant))


def gt_agreement(df, city, h_ant):
    tr = df[~df.test]
    hit = (gt_margins(df, city, h_ant) > 0) == (tr.label.values == 1)
    return pd.Series(hit).groupby(tr.area.values).mean().round(4).to_dict(), float(hit.mean())


def pieces(ep):
    dt = np.r_[np.inf, np.diff(ep.t.values)]
    return np.cumsum((dt > 10) | (np.r_[1, np.diff(ep.seg.values)] != 0))


def signal_features(df):
    df["abs_res_h"] = df.res_h.abs()
    df["nres_h"] = df.res_h / df.sig_h.replace(0, np.nan)
    df["has_phase"] = (df.phase != 0).astype(float)
    df["has_l2"] = (df.code2 != 0).astype(float)
    df["cn0_l1l2"] = np.where(df.code2 != 0, df.cn0 - df.cn02, np.nan)
    df["abs_loo"] = df.loo.abs()
    df["abs_dres"] = df.dres.abs()

    key = ["seg", "sat"]
    df.sort_values(key + ["t"], inplace=True, kind="stable")
    gap = df.groupby(key).t.diff()
    df["arc"] = (gap.isna() | (gap > 3)).cumsum()
    a = df.groupby("arc")
    df["arc_age"] = df.t - a.t.transform("min")
    df["arc_left"] = a.t.transform("max") - df.t
    for w in (5, 15):
        r = a.cn0.rolling(w, center=True, min_periods=1)
        df[f"cn0_m{w}"] = r.mean().reset_index(level=0, drop=True)
        df[f"cn0_s{w}"] = r.std().reset_index(level=0, drop=True)
        df[f"cn0_min{w}"] = r.min().reset_index(level=0, drop=True)
        df[f"ph_f{w}"] = a.has_phase.rolling(w, center=True, min_periods=1).mean().reset_index(level=0, drop=True)
    df["cn0_d1"] = a.cn0.diff()
    run = (df.has_phase == 0) | (a.has_phase.diff() != 0)
    df["lock_age"] = np.where(df.has_phase > 0, df.groupby(run.cumsum()).cumcount(), -1)

    dt = a.t.diff()
    code_rate = a.code.diff() / dt
    dop_rate = 0.5 * (df.dop_rr + a.dop_rr.shift())
    cdr = (code_rate - dop_rate).where(dt <= 3)
    df["cdr"] = cdr - cdr.groupby([df.seg, df.t]).transform("median")
    df["abs_cdr"] = df.cdr.abs()
    cmc = (df.code - df.lam * df.phase).where(df.phase != 0)
    dcmc = cmc.groupby(df.arc).diff().where((dt <= 1) & (df.lock_age > 0))
    df["dcmc"] = dcmc - dcmc.groupby([df.seg, df.t]).transform("median")
    df["abs_dcmc"] = df.dcmc.abs()
    for c in ("abs_cdr", "abs_dcmc", "abs_loo", "abs_dres"):
        df[f"{c}_m5"] = df.groupby("arc")[c].rolling(5, center=True, min_periods=1).mean() \
            .reset_index(level=0, drop=True)
    df.sort_values(["seg", "t", "sat"], inplace=True, kind="stable")

    e = df.groupby(["seg", "t"])
    df["n_sats"] = e.cn0.transform("count")
    df["ep_phase"] = e.has_phase.transform("mean")
    df["ep_cn0_med"] = e.cn0.transform("median")
    df["cn0_rel"] = df.cn0 - df.ep_cn0_med
    df["cn0_rank"] = e.cn0.rank(pct=True)
    df["el_rank"] = e.el_h.rank(pct=True)
    df["cn0_el"] = df.cn0 - df.groupby("sys").cn0.transform("mean")


def motion_features(df, ep):
    k = ep.set_index(["seg", "t"])[["speed", "heading"]]
    j = k.reindex(pd.MultiIndex.from_frame(df[["seg", "t"]]))
    df["speed"] = j.speed.values
    df["across"] = np.abs(np.sin(np.radians(df.az_h.values) - j.heading.values))

