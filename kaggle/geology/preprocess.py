import sys
import os
import glob
import subprocess
import multiprocessing
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

kb_dir = "/kaggle/input/koolbox-offline"
if not os.path.isdir(kb_dir):
    cand = glob.glob("/kaggle/input/**/koolbox*", recursive=True)
    if cand:
        kb_dir = cand[0]
if os.path.isdir(kb_dir):
    whls = glob.glob(f"{kb_dir}/**/*.whl", recursive=True)
    if whls:
        for w in whls:
            subprocess.run(["pip", "install", "--no-deps", w], check=False)
    else:
        sys.path.insert(0, kb_dir)
        for sub in os.listdir(kb_dir):
            sub_path = os.path.join(kb_dir, sub)
            if os.path.isdir(sub_path):
                sys.path.insert(0, sub_path)


class CFG:
    DATA = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
    ARTIFACTS = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")
    OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

    dataset_path = DATA
    artifacts_path = ARTIFACTS

    seed = 42
    n_splits = 5
    cv = GroupKFold(n_splits=n_splits)
    metric = root_mean_squared_error
    n_jobs = min(8, multiprocessing.cpu_count())

    PF_SEEDS = 128
    PF_PARTICLES = 500
    PF_SCALES = (3., 5., 8., 12.)

    FAST = bool(int(os.environ.get("FAST", "0")))
    N_TRAIN_WELLS = int(os.environ.get("N_TRAIN_WELLS", "0"))
    USE_GPU = os.environ.get("USE_GPU", "auto")
    SHOW_FIGS = os.environ.get("SHOW_FIGS", "0") == "1"

    BLEND_W_A = 0.30
    BLEND_W_B = 0.70
    BLEND_W_SP45 = 0.55

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10
DENSE_SPW = 60
DENSE_K = 20
NCPU = min(4, multiprocessing.cpu_count())

BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    ( 8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]

ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
BEAM_OFFS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], np.float32)
SC_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
PF_OFFS = SC_OFFS.copy()

PF_N = 600
ANCC_N = 600
PF_MOM = 0.993
PF_VN = 0.005
PF_PN = 0.01
PF_GR_SIG_MIN = 10.
PF_GR_SIG_MAX = 60.
PF_GR_SIG_DEF = 30.
PF_GR_WIN = 5
PF_GR_WT = 0.3
PF_RESAMP = 0.5
PF_ROUGH_P = 0.2
PF_ROUGH_V = 0.003
ANCC_ALPHA = 0.998
ANCC_RN = 0.002
ANCC_PN = 0.005
ANCC_IS = 0.3
ANCC_RP = 0.1
ANCC_RR = 0.001

SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}
SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)

BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1), (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3), (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4), (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]


@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1. - t) + grid[i + 1] * t


@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0., 1. / N)
    np2 = np.empty(N)
    na = np.empty(N)
    ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr)
    nt = len(tw_gr)
    MAX = BS * 6
    bidx = np.zeros(BS, np.int64)
    bidx[0] = si
    bcost = np.full(BS, 1e30)
    bcost[0] = 0.
    bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64)
    hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64)
    cC = np.full(MAX, 1e30)
    cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]
        nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]
            cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt:
                    continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni:
                        fnd = ci
                    break
                if fnd >= 0:
                    if tot < cC[fnd]:
                        cC[fnd] = tot
                    cP[fnd] = bi
                else:
                    if nc < MAX:
                        cI[nc] = ni
                    cC[nc] = tot
                    cP[nc] = bi
                    nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]:
                    mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]
        hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]
        bcost[:kept] = cC[:kept]
        bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]:
            best = b
    path = np.zeros(n, np.int64)
    b = best
    for s in range(n - 1, -1, -1):
        path[s] = hI[s, b]
    b = hP[s, b]
    return path


@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N)
    rate = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.
    for i in range(len(md_v)):
        dm = max(md_v[i] - pm, 1.)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.)
            tvt_j = min(tvt_j, vmin + len(gg) * step + 50.)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600. else 0., 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1. / N
        ne = 0.
        for j in range(N):
            ne += w[j] * w[j]
        if 1. / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N):
                w[j] = 1. / N
        tv = 0.
        for j in range(N):
            tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        va = 0.
        for j in range(N):
            va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]
    return pts, std_


@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N,
          MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos = np.empty(N)
    vel = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.
    pz = z_v[0] - 1.
    for i in range(len(md_v)):
        dm = max(md_v[i] - pm, 1.)
        dzd = (z_v[i] - pz) / dm
        ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.)
            pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.)
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                lp = max(np.exp(-0.5 * dp * dp) if dp * dp < 600. else 0., 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es2 = _interp1(gg_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es2) / (gs * 1.5)
                    ls = max(np.exp(-0.5 * ds * ds) if ds * ds < 600. else 0., 1e-300)
                    lk = (1. - GR_WT) * lp + GR_WT * ls
                else:
                    lk = lp
                lk = max(lk, 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1. / N
        ws2 = 0.
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2., 0.005)
            lz = max(np.exp(-0.5 * dv * dv) if dv * dv < 600. else 0., 1e-300)
            w[j] *= lz
            ws2 += w[j]
        if ws2 > 0.:
            for j in range(N):
                w[j] /= ws2
        else:
            for j in range(N):
                w[j] = 1. / N
        ne = 0.
        for j in range(N):
            ne += w[j] * w[j]
        if 1. / ne < RESAMP * N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N):
                w[j] = 1. / N
        wm = 0.
        for j in range(N):
            wm += w[j] * pos[j]
        pts[i] = wm
        va = 0.
        for j in range(N):
            va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]
        pz = z_v[i]
    return pts, std_


@njit(cache=True, nogil=True)
def _pf_lik_allseeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, n_seeds, seed_base,
                     MOM, VN, PN, RP, RR, RESAMP, init_spr):
    n = len(md_v)
    preds = np.empty((n_seeds, n))
    liks = np.empty(n_seeds)
    tmax = vmin + len(gg) * step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N)
        rate = np.empty(N)
        w = np.ones(N) / N
        for j in range(N):
            pos[j] = ls + init_spr * np.random.randn()
            rate[j] = ir + 0.01 * np.random.randn()
        log_lik = 0.0
        prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0:
                dm = 1.0
            for j in range(N):
                rate[j] = MOM * rate[j] + VN * np.random.randn()
                pos[j] += rate[j] * dm + PN * np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin - 100.:
                    tvt_j = vmin - 100.
                if tvt_j > tmax + 100.:
                    tvt_j = tmax + 100.
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                dd = min(d * d, 600.)
                lk = np.exp(-0.5 * dd)
                if lk < 1e-300:
                    lk = 1e-300
                avg_lk += w[j] * lk
                w[j] = w[j] * lk
            if avg_lk < 1e-300:
                avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N):
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1. / N
            neff = 0.0
            for j in range(N):
                neff += w[j] * w[j]
            neff = 1.0 / neff
            if neff < RESAMP * N:
                cum = np.empty(N)
                c = 0.0
                for j in range(N):
                    c += w[j]
                    cum[j] = c
                u0 = np.random.uniform(0., 1. / N)
                newpos = np.empty(N)
                newrate = np.empty(N)
                ci = 0
                for j in range(N):
                    u = u0 + j / N
                    while ci < N - 1 and cum[ci] < u:
                        ci += 1
                    newpos[j] = pos[ci] + RP * np.random.randn()
                    newrate[j] = rate[ci] + RR * np.random.randn()
                for j in range(N):
                    pos[j] = newpos[j]
                    rate[j] = newrate[j]
                    w[j] = 1. / N
            est = 0.0
            for j in range(N):
                est += w[j] * (pos[j] - z_v[i])
            preds[s, i] = est
            prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks


def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min())
    tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(kn) < 20:
        return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn["GR"].values - np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))


def _nn(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr):
        return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v):
        return i - 1
    return i


def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)


def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    pts, std = _pf_ancc(ev["MD"].values.astype(np.float64), ev["Z"].values.astype(np.float64),
                        ev["GR"].values.astype(np.float64), gg, gmin, gst,
                        gs, ls, ir, N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)


def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    dz_k = np.diff(kna["Z"].values)
    dvt = np.diff(kna["TVT_input"].values)
    dmd_k = np.diff(kna["MD"].values)
    m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2] / dmd_k[m2]
        vt = dvt[m2] / dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)
    else:
        beta, icpt, zsig = -1., 0., 0.1
    t2 = kna.tail(20)
    dvt2 = np.diff(t2["TVT_input"].values)
    dmd2 = np.diff(t2["MD"].values)
    m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_sm = hw["GR"].rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    pts, std = _pf_z(ev["MD"].values.astype(np.float64), ev["Z"].values.astype(np.float64),
                     ev["GR"].values.astype(np.float64), gr_sm.loc[ev.index].values.astype(np.float64),
                     gg, gs2, gmin, gst, gs, float(kna["TVT_input"].iloc[-1]), iv,
                     beta, icpt, zsig, N,
                     PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)


def lik_pf(hw, tw, n_particles=CFG.PF_PARTICLES, n_seeds=CFG.PF_SEEDS, scales=CFG.PF_SCALES,
           init_spr=4.5, seed_base=0, with_quality=False):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]
    ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0:
        return {}, np.array([]), {}
    last = kn.iloc[-1]
    ls = float(last.TVT_input) + float(last.Z)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.))
    tail = kn.tail(30)
    dt = np.diff(tail.TVT_input.values)
    dz = np.diff(tail.Z.values)
    dm = np.diff(tail.MD.values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gr_v = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    preds, liks = _pf_lik_allseeds(ev.MD.values.astype(float), ev.Z.values.astype(float), gr_v,
                                   gg, gmin, gst, gs, ls, ir, n_particles, n_seeds, seed_base,
                                   0.998, 0.002, 0.005, 0.1, 0.001, 0.5, init_spr)
    ln = liks - liks.max()
    out = {}
    for sc in scales:
        wts = np.exp(ln / float(sc))
        wts /= wts.sum()
        out[f"pf_scale_{sc:g}"] = (wts[:, None] * preds).sum(0)
    out["pf_mean"] = preds.mean(0)
    q = {}
    if with_quality:
        q = {"pf_best_ll": float(liks.max()) / len(ev), "pf_ll_spread": float(liks.std()),
             "pf_pt_std": preds.std(0).astype(np.float32), "pf_gr_sig": gs}
    return out, ev.index.values, q


def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt)
    sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))
    return tw_tvt[path].astype(np.float32)


def run_beam_ensemble(hw, tw):
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy()
    last_tvt = float(kn.iloc[-1]["TVT_input"])
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)
    hgr = gr_all[ev.index]
    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
                    for (bs, mc, es, r) in BEAM_CONFIGS]
    beam_mean = np.stack(beam_results, 0).mean(0)
    out = hw["TVT_input"].values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2 * hw + 1
        nk = len(kgr)
        nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1)
    scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3. * scores)
    sw /= sw.sum(1, keepdims=True) + 1e-9
    return out, (tvts * sw).sum(1).astype(np.float32)


def dtw_tvt_estimate(kgr, ktvt, hgr, hw=15, stride=3):
    win = 2 * hw + 1
    nk = len(kgr)
    nh = len(hgr)
    if nk < win + 1 or nh == 0:
        return np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)
    kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
    hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
    sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
    if len(sts) == 0:
        return np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)
    C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]]
    Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
    hp = np.pad(hg, hw, mode="edge")
    H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]]
    Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
    dists = np.sqrt(((Hn[:, None, :] - Cn[None, :, :]) ** 2).mean(2))
    best = dists.argmin(1)
    best_dist = dists[np.arange(nh), best]
    best_tvt = ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32)
    score = np.exp(-best_dist).astype(np.float32)
    return best_tvt, score


class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            try:
                df = pd.read_csv(data_dir / f"{wid}__horizontal_well.csv",
                                 usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df.X.median()), "y": float(df.Y.median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df.wid)}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df.x.to_numpy()
        self.ya = self.df.y.to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale
        nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1)
        ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1. / (dk + 1e-3), 0.).astype(np.float64)
        xn = self.xa[ik]
        yn = self.ya[ik]
        fn = self.fa[ik]
        wx = w * xn
        wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1)
        A[:, 0, 1] = (wx * yn).sum(1)
        A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]
        A[:, 1, 1] = (wy * yn).sum(1)
        A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]
        A[:, 2, 1] = A[:, 1, 2]
        A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9
        A[:, 1, 1] += 1e-9
        A[:, 2, 2] += 1e-9
        rhs = np.stack([(wx[:, :, None] * fn).sum(1),
                        (wy[:, :, None] * fn).sum(1),
                        (w[:, :, None] * fn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception:
                    pass
        Xq = xy_q[:, 0]
        Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs, ys, an, wd = [], [], [], []
        for wid in well_ids:
            try:
                df = pd.read_csv(data_dir / f"{wid}__horizontal_well.csv",
                                 usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            s = df.iloc[ix]
            xs.append(s.X.values)
            ys.append(s.Y.values)
            an.append(s.ANCC.values)
            wd.extend([wid] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(an).astype(np.float32)
        self.wids = np.array(wd)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1., self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1)
        ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1. / (dk + 1e-3), 0.)
        sw = w.sum(1)
        safe = np.where(sw < 1e-9, 1., sw)
        a = self.ancc[ik]
        ap = (a * w).sum(1) / safe
        ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((a - ap[:, None]) ** 2 * w).sum(1) / safe
        return (ap.astype(np.float32),
                np.sqrt(np.maximum(var, 0.)).astype(np.float32),
                np.where(vk, dk, np.inf).min(1).astype(np.float32))


def robust_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6:
        return 0.
    return float(np.polyfit(x[m], y[m], 1)[0])


def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1., float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col
    n = len(bv)
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    t1, t2 = n // 3, 2 * n // 3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1 + 1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02 * np.arange(n))
    w /= w.sum()
    return b_full, b_early, b_mid, b_late, float(np.dot(w, bv))


def _lateral_section(kn, threshold=0.05, win=50):
    md = kn["MD"].values.astype(float)
    z = kn["Z"].values.astype(float)
    n = len(md)
    lat_start = 0
    for i in range(n - 1, win, -1):
        seg_z = z[i - win:i]
        seg_md = md[i - win:i]
        dm = seg_md[-1] - seg_md[0]
        if dm < 1e-6:
            continue
        if abs((seg_z[-1] - seg_z[0]) / dm) > threshold:
            lat_start = i
            break
    return max(0, min(lat_start + win, n - 20))


def _well_traj_features(kn):
    ZERO = {
        "well_azimuth_sin": 0., "well_azimuth_cos": 1.,
        "dz_per_md_lateral": 0., "tvt_z_slope_lat": 0., "tvt_minus_az_b": 0.,
        "T_incline": 1., "Gamma_incline": 0., "TQG_incline": 0.,
        "T_azimuth": 1., "Gamma_azimuth": 0., "TQG_azimuth": 0., "TQG_Q3D": 0.,
    }
    if len(kn) < 20:
        return ZERO
    md = kn["MD"].values.astype(float)
    x = kn["X"].values.astype(float)
    y = kn["Y"].values.astype(float)
    z = kn["Z"].values.astype(float)
    tvt = kn["TVT_input"].values.astype(float)
    lat_i = _lateral_section(kn)
    lat_x = x[lat_i:]
    lat_y = y[lat_i:]
    lat_z = z[lat_i:]
    lat_md = md[lat_i:]
    lat_tvt = tvt[lat_i:]
    if len(lat_x) >= 2:
        azi = np.arctan2(lat_x[-1] - lat_x[0], lat_y[-1] - lat_y[0])
        azi_sin = float(np.sin(azi))
        azi_cos = float(np.cos(azi))
    else:
        azi_sin, azi_cos = 0., 1.
    dm_lat = np.diff(lat_md)
    m_lat = dm_lat > 0
    dzdmd_lat = (float(np.median(np.diff(lat_z)[m_lat] / dm_lat[m_lat])) if m_lat.sum() >= 2 else 0.)
    fin = np.isfinite(lat_tvt) & np.isfinite(lat_z)
    tvt_z_sl = 0.
    tvt_minus_az = 0.
    if fin.sum() >= 20 and lat_z[fin].std() > 1.0:
        A = np.column_stack([lat_z[fin], np.ones(int(fin.sum()))])
        c, _, _, _ = np.linalg.lstsq(A, lat_tvt[fin], rcond=None)
        tvt_z_sl = float(c[0])
        intercept = float(c[1])
        tvt_minus_az = float(tvt[-1]) - (tvt_z_sl * float(z[-1]) + intercept)
    stride = max(1, len(md) // 100)
    md_s = md[::stride]
    x_s = x[::stride]
    y_s = y[::stride]
    z_s = z[::stride]
    T_inc, Gamma_inc, TQG_inc = 1., 0., 0.
    T_azi, Gamma_azi, TQG_azi = 1., 0., 0.
    if len(md_s) >= 4:
        dx_s = np.diff(x_s)
        dy_s = np.diff(y_s)
        dz_s = np.diff(z_s)
        dh = np.sqrt(dx_s ** 2 + dy_s ** 2)
        inc_chord = float(np.sqrt((x_s[-1] - x_s[0]) ** 2 + (y_s[-1] - y_s[0]) ** 2 + (z_s[-1] - z_s[0]) ** 2))
        if inc_chord > 1.0:
            inc_arc = float(np.sqrt(dh ** 2 + dz_s ** 2).sum())
            T_inc = inc_arc / inc_chord
            inc_ang = np.arctan2(dz_s, dh + 1e-9)
            Gamma_inc = float(np.abs(np.diff(inc_ang)).sum() / inc_chord)
            TQG_inc = float(np.sqrt(max(T_inc * Gamma_inc, 0.)))
        fwd_dx = x_s[-1] - x_s[0]
        fwd_dy = y_s[-1] - y_s[0]
        azi_chord = float(np.sqrt(fwd_dx ** 2 + fwd_dy ** 2))
        if azi_chord > 1.0:
            nrm = azi_chord + 1e-9
            perp_x = -fwd_dy / nrm
            perp_y = fwd_dx / nrm
            lat_off = (x_s - x_s[0]) * perp_x + (y_s - y_s[0]) * perp_y
            d_lat = np.diff(lat_off)
            azi_arc = float(np.sqrt(dh ** 2 + d_lat ** 2).sum())
            T_azi = azi_arc / azi_chord
            azi_ang = np.arctan2(d_lat, dh + 1e-9)
            Gamma_azi = float(np.abs(np.diff(azi_ang)).sum() / azi_chord)
            TQG_azi = float(np.sqrt(max(T_azi * Gamma_azi, 0.)))
    return {
        "well_azimuth_sin": azi_sin, "well_azimuth_cos": azi_cos,
        "dz_per_md_lateral": dzdmd_lat, "tvt_z_slope_lat": tvt_z_sl, "tvt_minus_az_b": tvt_minus_az,
        "T_incline": T_inc, "Gamma_incline": Gamma_inc, "TQG_incline": TQG_inc,
        "T_azimuth": T_azi, "Gamma_azimuth": Gamma_azi, "TQG_azimuth": TQG_azi,
        "TQG_Q3D": float(np.sqrt(TQG_inc ** 2 + TQG_azi ** 2)),
    }

_FI = None
_DI = None


def build_well(hw_path, tw_path, is_train):
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
    sc_cons = (sc8 + sc15 + sc25) / 3.
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
    form_ev, knn_d = _FI.impute(xy_ev, self_wid=swid)
    form_kn, _ = _FI.impute(xy_kn, self_wid=swid)
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
        tvt_fs[f"bw_early_{fn}"] = np.float32(b_early)
        tvt_fs[f"bw_mid_{fn}"] = np.float32(b_mid)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-z_kn + form_kn[:, fi2] + b_full)) ** 2)))
        form_list.append(tvt_f)

    fs = np.stack(form_list, 1)
    form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
    form_std_d = fs.std(1).astype(np.float32)
    form_rng_d = (fs.max(1) - fs.min(1)).astype(np.float32)

    d_ancc, d_std, d_dist = _DI.impute(xy_ev, self_wid=swid)
    d_kn, d_std_kn, _ = _DI.impute(xy_kn, self_wid=swid)
    b_vd = ktvt + z_kn - d_kn
    _, b_de, b_dm, b_dl, b_dw = seg_b_well(ktvt, z_kn, d_kn)
    b_d = float(np.median(b_vd))
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
    tvt_densew = (-z_ev + d_ancc + b_dw).astype(np.float32)
    tvt_dense50 = (-z_ev + d_ancc + b_dl).astype(np.float32)
    res_kn = ktvt + z_kn - d_kn
    d_rmse = float(np.sqrt(np.mean(res_kn ** 2)))
    d_bias = float(np.mean(res_kn))
    d_nb_std = float(np.mean(d_std_kn))

    all_sigs = [pf_use] + [p for p in bpaths.values()] + [sc8, sc15, sc25, sc_ens, tvt_fs["tvtF_ANCC"], tvt_dense]
    sig_mat = np.stack(all_sigs, 1)
    sig_std = sig_mat.std(1).astype(np.float32)
    sig_mean = (sig_mat.mean(1) - last_tvt).astype(np.float32)

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
    gr_env = gr_s.rolling(21, center=True, min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg = np.sqrt(np.maximum(
        (gr_s ** 2).rolling(21, center=True, min_periods=1).mean(), 0.
    )).iloc[ev.index].values.astype(np.float32)

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
        "sc_cons_d": (sc_cons - np.float32(last_tvt)).astype(np.float32),
        "sc_ens_d": (sc_ens - np.float32(last_tvt)).astype(np.float32),
        "sc_trust": sc(sc_trust), "hyb_d": (hyb_ref - np.float32(last_tvt)).astype(np.float32),
        "dtw8_d": (dtw8 - np.float32(last_tvt)).astype(np.float32), "dtw8_sc": dtw8s,
        "dtw15_d": (dtw15 - np.float32(last_tvt)).astype(np.float32), "dtw15_sc": dtw15s,
        "dtw25_d": (dtw25 - np.float32(last_tvt)).astype(np.float32), "dtw25_sc": dtw25s,
        "sig_std": sig_std, "sig_mean_d": sig_mean,
        **tvt_fs,
        **{f"frm_rmse_{fn}": sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d": form_mean_d, "form_std_d": form_std_d, "form_rng_d": form_rng_d,
        "spatial_ancc_d": (form_ev[:, 0] - np.float32(np.interp(last_tvt, tw_tvt, tw_gr))),
        "spatial_knn_dist": knn_d,
        "dense_ancc": d_ancc, "dense_std": d_std, "dense_dist": d_dist,
        "tvt_dense_d": (tvt_dense - last_tvt).astype(np.float32),
        "tvt_densew_d": (tvt_densew - last_tvt).astype(np.float32),
        "tvt_dense50_d": (tvt_dense50 - last_tvt).astype(np.float32),
        "dense_rmse": sc(d_rmse), "dense_bias": sc(d_bias), "dense_nb_std": sc(d_nb_std),
        "pf_vs_spatial": (pf_use - tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "pf_vs_dense": (pf_use - tvt_dense).astype(np.float32),
        "spatial_vs_dense": (tvt_fs["tvtF_ANCC"] - tvt_dense).astype(np.float32),
        "beam_vs_spatial": (bpaths["cons"] - tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "sc_vs_beam": (sc_ens - bpaths["cons"]).astype(np.float32),
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
        "gr": hgr, "gr_d1": gr_d1, "gr_d2": gr_d2, "gr_env": gr_env, "gr_nrg": gr_nrg,
        "gr_vs_tw_anc": hgr - np.float32(np.interp(last_tvt, tw_tvt, tw_gr)),
        "gr_vs_slp_all": hgr - np.interp(slp_b_all, tw_tvt, tw_gr).astype(np.float32),
        **{f"tda{int(o)}": hgr - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr)) for o in ANCH_OFFS},
        **{f"tdbc{int(o)}": hgr - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f"tdsc{int(o)}": hgr - np.interp(sc_ens + o, tw_tvt, tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f"tdpf{int(o)}": hgr - np.interp(pf_use + o, tw_tvt, tw_gr).astype(np.float32) for o in PF_OFFS},
        "tw_range": sc(float(np.ptp(tw_tvt))), "tw_gr_mean": sc(float(tw_gr.mean())),
        **{k: sc(v) for k, v in traj_fs.items()},
        **dtvt_rolls,
    }
    for k, v in rolls.items():
        feats[k] = v
    result = pd.DataFrame(feats)
    if is_train:
        if "TVT" not in ev.columns or ev["TVT"].isna().all():
            return None
        result["target"] = (ev["TVT"].to_numpy(np.float32) - np.float32(last_tvt))
    return result


def build_dataset(paths, is_train, label):
    args = [(str(p), str(p.parent / f'{p.stem.replace("__horizontal_well", "")}__typewell.csv'), is_train)
            for p in paths
            if (p.parent / f'{p.stem.replace("__horizontal_well", "")}__typewell.csv').exists()]
    res = Parallel(n_jobs=NCPU, prefer="threads", verbose=3)(
        delayed(build_well)(hp, tp, it) for hp, tp, it in args)
    parts = [r for r in res if r is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def tvt_from_contacts(hw_tr, tw_tr, ref_col="EGFDU"):
    tw_g = tw_tr.dropna(subset=["Geology"])
    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]
        ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset


def load_well(wid, split="train"):
    base = CFG.DATA / split
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv").sort_values("TVT")
    return hw, tw


def selector_well_code(hw):
    eval_mask = hw["TVT_input"].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, "Z"].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
    return code, variant, n_eval, z_span


def parse_selector_variant(name):
    parts = name.split("_")
    scale = float(parts[2])
    beam_weight = 0.0
    hold_weight = 0.0
    if "beam" in parts:
        beam_weight = float(parts[parts.index("beam") + 1])
    if "hold" in parts:
        hold_weight = float(parts[parts.index("hold") + 1])
    return scale, beam_weight, hold_weight


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt):
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f"pf_scale_{scale:g}", pf_by_scale.get("pf_scale_8", list(pf_by_scale.values())[0]))
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred


def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        last_val = float(kn.iloc[-1]["TVT_input"]) if len(kn) > 0 else 0.0
        return {f"pf_scale_{s:g}": hw["TVT_input"].fillna(last_val).values.astype(float) for s in scales}
    last = kn.iloc[-1]
    ls = float(last.TVT_input) + float(last.Z)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.))
    tail = kn.tail(30)
    dt = np.diff(tail.TVT_input.values)
    dz = np.diff(tail.Z.values)
    dm = np.diff(tail.MD.values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gr_v = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    preds, liks = _pf_lik_allseeds(ev.MD.values.astype(float), ev.Z.values.astype(float), gr_v,
                                   gg, gmin, gst, gs, ls, ir, n_particles, n_seeds, 0,
                                   0.998, 0.002, 0.005, 0.1, 0.001, 0.5, 4.5)
    liks_n = liks - liks.max()
    out = {}
    for sc in scales:
        wts = np.exp(liks_n / float(sc))
        wts /= wts.sum()
        tvt_full = hw["TVT_input"].values.astype(float).copy()
        tvt_full[list(ev.index)] = (wts[:, None] * preds).sum(0)
        out[f"pf_scale_{sc:g}"] = tvt_full
    out["pf_mean"] = hw["TVT_input"].values.astype(float).copy()
    out["pf_mean"][list(ev.index)] = preds.mean(0)
    return out

_md = np.linspace(1, 50, 20, np.float64)
_z = np.zeros(20, np.float64)
_gr = np.full(20, 50., np.float64)
_gg = np.linspace(45, 55, 100, np.float64)
_pf_ancc(_md, _z, _gr, _gg, 45., 0.1, 20., 50., 0., 8, 0.998, 0.002, 0.005, 0.3, 0.1, 0.001, 0.5)
_pf_z(_md, _z, _gr, _gr, _gg, _gg, 45., 0.1, 20., 50., 0., -1., 0., 0.1, 8, 0.993, 0.005, 0.01, 0.3, 0.2, 0.003, 0.5)
_beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15., 100.)
_pf_lik_allseeds(_md, _z, _gr, _gg, 45., .1, 20., 50., 0., 64, 4, 0, .998, .002, .005, .1, .001, .5, 4.5)
print("trackers compiled.")
