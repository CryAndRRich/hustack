import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_CAMS = [f"cam_{c}" for c in "ABCDEFGH"]
H_MAX, BLANK, EDGE_K, EDGE_W, FIT_W = 15, 30, 3, 15, 10
SHIFTS, SHIFT_SD, RATIO_WIN = np.arange(-8, 9), 4.0, 31
LAT, LON, TZ = 24.4672, 39.6111, 3

def data_dir():
    for root, _, files in sorted(os.walk("/kaggle/input", followlinks=True)):
        if {"train.csv", "test.csv", "sample_submission.csv"} <= set(files):
            return Path(root)
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"

def load():
    d = data_dir()
    tr, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    sample_ids = pd.read_csv(d / "sample_submission.csv", usecols=["id"])["id"]
    holes = te[te["id"].notna()].copy()
    assert holes["people"].isna().all() and te.loc[te["id"].isna(), "people"].notna().all()
    assert set(holes["id"]) == set(sample_ids) and holes["id"].is_unique
    print(f"data {d}: train {len(tr):,}, test {len(te):,}, holes {len(holes):,}")
    return tr, te[te["id"].isna()], holes

def table(rows):
    t = rows.pivot_table(index="minute_of_day", columns="unit", values="people")
    return t.reindex(np.arange(t.index.min(), t.index.max() + 1))

def prayer_times(day):
    d = pd.Timestamp(day)
    g = 2 * math.pi / 365 * (d.dayofyear - 1)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    dec = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
           + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    phi = math.radians(LAT)
    noon = 720 - 4 * LON - eqt + 60 * TZ

    def after_noon(alt):
        c = (math.sin(alt) - math.sin(phi) * math.sin(dec)) / (math.cos(phi) * math.cos(dec))
        return noon + 4 * math.degrees(math.acos(c))

    maghrib = after_noon(math.radians(-0.833))
    asr = after_noon(math.atan(1 / (1 + math.tan(abs(phi - dec)))))
    return {"dhuhr": noon, "asr": asr, "maghrib": maghrib, "isha": maghrib + 90}

def anchors(day):
    p = prayer_times(day)
    return np.array([0.0, p["dhuhr"], p["asr"], p["maghrib"], p["isha"], 1440.0])

def smooth(x):
    s = pd.Series(x)
    m = s.rolling(5, center=True, min_periods=3).median().rolling(3, center=True, min_periods=1).mean()
    return np.where(np.isfinite(x), m.to_numpy(), np.nan)

def prev_next(X):
    n = len(X)
    idx = np.arange(n).reshape((n,) + (1,) * (X.ndim - 1))
    ok = np.isfinite(X)
    prev = np.maximum.accumulate(np.where(ok, idx, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, idx, n)[::-1], axis=0)[::-1]
    return prev, nxt

class Day:
    def __init__(self, name, day, weekday, M):
        self.name, self.day, self.weekday, self.M = name, day, weekday, M
        self.first = int(M.index[0])
        self.X = M.to_numpy(float)
        self.cams = list(M.columns)
        self.prev, self.next = prev_next(self.X)
        self.anch = anchors(day)

    def col(self, cam):
        return self.cams.index(cam)

class Source:
    def __init__(self, name, day, weekday, T):
        self.name, self.day, self.weekday = name, day, weekday
        self.first = int(T.index[0])
        self.anch = anchors(day)
        self.S = {c: smooth(T[c].to_numpy(float)) for c in T.columns if T[c].notna().any()}
        self.lr = {}

    def at(self, cam, minutes):
        return self.read(self.S[cam], minutes)

    def log_ratio(self, h, v):
        key = (h, v)
        if key not in self.lr:
            r = pd.Series(np.log1p(self.S[h]) - np.log1p(self.S[v]))
            self.lr[key] = r.rolling(RATIO_WIN, center=True, min_periods=10).median().to_numpy()
        return self.lr[key]

    def read(self, s, minutes):
        p = minutes - self.first
        i = np.floor(p).astype(int)
        f = p - i
        ok = (i >= 0) & (i + 1 < len(s))
        out = np.full(len(p), np.nan)
        lo, hi = s[i[ok]], s[i[ok] + 1]
        out[ok] = np.where(f[ok] == 0, lo, lo * (1 - f[ok]) + hi * f[ok])
        return out

def source_weight(day, src):
    if day.weekday != src.weekday:
        return 0.3
    gap = abs((pd.Timestamp(day.day) - pd.Timestamp(src.day)).days)
    return 1.0 if gap <= 7 else 0.6

def ratio_curve(target, first, n, sources, h, v):
    t = first + np.arange(n, dtype=float)
    num, den, pool = np.zeros(n), np.zeros(n), []
    for s in sources:
        if s.name == target.name or h not in s.S or v not in s.S:
            continue
        lr = s.log_ratio(h, v)
        val = s.read(lr, np.interp(t, target.anch, s.anch))
        k = np.isfinite(val)
        w = source_weight(target, s)
        num[k] += w * val[k]
        den[k] += w
        pool.append(lr[np.isfinite(lr)])
    if not pool:
        return None
    return np.where(den > 0, num / np.where(den > 0, den, 1), float(np.median(np.concatenate(pool))))

QUALITY = {}

def donor_quality(train_sources, h, v):
    key = (tuple(sorted(s.name for s in train_sources)), h, v)
    if key not in QUALITY:
        QUALITY[key] = _donor_quality(train_sources, h, v)
    return QUALITY[key]

def _donor_quality(train_sources, h, v):
    res = []
    for T in train_sources:
        if h not in T.S or v not in T.S:
            continue
        rho = ratio_curve(T, T.first, len(T.S[h]), train_sources, h, v)
        if rho is not None:
            e = np.log1p(T.S[h]) - np.log1p(T.S[v]) - rho
            res.append(e[np.isfinite(e)])
    e = np.concatenate(res) if res else np.array([])
    return 1.0 / max(float(np.var(e)), 1e-4) if len(e) >= 100 else 0.0

def ratio_estimate(day, h, donors, sources, train_sources, logs=None):
    n = len(day.X)
    num, den, best = np.zeros(n), np.zeros(n), None
    for v in donors:
        q = donor_quality(train_sources, h, v)
        rho = ratio_curve(day, day.first, n, sources, h, v)
        if q <= 0 or rho is None:
            continue
        lv = logs[v] if logs is not None else np.log1p(smooth(day.X[:, day.col(v)]))
        e = lv + rho
        k = np.isfinite(e)
        num[k] += q * e[k]
        den[k] += q
        if best is None or q > best[0]:
            best = (q, e)
    if best is None:
        return np.full(n, np.nan), np.full(n, np.nan)
    return np.where(den > 0, np.expm1(num / np.where(den > 0, den, 1)), np.nan), np.expm1(best[1])

def build_analogs(day, sources, cams):
    srcs = [s for s in sources if s.weekday == day.weekday and s.name != day.name]
    n = len(day.X)
    t = day.first + np.arange(n, dtype=float)
    raw = {}
    for s in srcs:
        tw = np.interp(t, day.anch, s.anch)
        for cam in cams:
            if cam in s.S:
                raw[(cam, s.name)] = np.log1p(s.at(cam, tw))
    off = {}
    for (cam, sn), a in raw.items():
        if cam in day.cams:
            o = np.log1p(day.X[:, day.col(cam)])
            k = np.isfinite(a) & np.isfinite(o)
            if k.sum() >= 30:
                off[(cam, sn)] = float(np.median(o[k] - a[k]))
    for cam, sn in raw:
        if (cam, sn) not in off:
            same = [v for (c2, s2), v in off.items() if s2 == sn and c2 in BASE_CAMS]
            off[(cam, sn)] = float(np.median(same)) if same else 0.0
    out = {}
    for cam in cams:
        num, den = np.zeros(n), np.zeros(n)
        for s in srcs:
            a = raw.get((cam, s.name))
            if a is None:
                continue
            k = np.isfinite(a)
            w = source_weight(day, s)
            num[k] += w * (a[k] + off[(cam, s.name)])
            den[k] += w
        if (den > 0).any():
            out[cam] = np.where(den > 0, np.expm1(num / np.where(den > 0, den, 1)), np.nan)
    for cam in day.cams:
        if cam in out:
            continue
        o = np.log1p(day.X[:, day.col(cam)])
        best = None
        for p, a in out.items():
            if p not in day.cams:
                continue
            q = np.log1p(day.X[:, day.col(p)])
            k = np.isfinite(o) & np.isfinite(q) & np.isfinite(a)
            if k.sum() >= 30:
                r = np.corrcoef(o[k], q[k])[0, 1]
                if best is None or r > best[0]:
                    best = (r, p, float(np.median(o[k] - np.log1p(a[k]))))
        if best is not None:
            out[cam] = np.expm1(np.log1p(out[best[1]]) + best[2])
    return out

def partners(day, min_corr=0.3):
    R = {}
    for c in day.cams:
        x = day.X[:, day.col(c)]
        med = pd.Series(x).rolling(9, center=True, min_periods=5).median().to_numpy()
        R[c] = np.log1p(x) - np.log1p(med)
    out = {}
    for c in day.cams:
        ps = []
        for p in day.cams:
            k = np.isfinite(R[c]) & np.isfinite(R[p])
            if p != c and k.sum() >= 60 and np.corrcoef(R[c][k], R[p][k])[0, 1] >= min_corr:
                ps.append(p)
        if not ps:
            continue
        stack = np.vstack([R[p] for p in ps])
        cnt = np.isfinite(stack).sum(0)
        rp = np.where(cnt > 0, np.nansum(stack, 0) / np.maximum(cnt, 1), np.nan)
        k = np.isfinite(R[c]) & np.isfinite(rp)
        if k.sum() >= 60 and (rp[k] ** 2).sum() > 0:
            out[c] = (ps, float(np.clip((R[c][k] * rp[k]).sum() / (rp[k] ** 2).sum(), 0, 1.5)))
    return out

def edge_median(x, idx):
    v = x[idx]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else np.nan

def gap_arms(x, prev, nxt, t, an):
    n = len(x)
    a, b = prev[t], nxt[t]
    iL = [i for i in range(max(a - EDGE_W + 1, 0), a + 1) if np.isfinite(x[i])][-EDGE_K:] if a >= 0 else []
    iR = [i for i in range(b, min(b + EDGE_W, n)) if np.isfinite(x[i])][:EDGE_K] if b < n else []
    last = x[a] if a >= 0 else x[b]
    raw = x[a] + (x[b] - x[a]) * (t - a) / (b - a) if a >= 0 and b < n else last
    lv, rv = edge_median(x, iL), edge_median(x, iR)
    if iL and iR:
        w = float(np.clip((t - np.mean(iL)) / (np.mean(iR) - np.mean(iL)), 0, 1))
        sm = lv + (rv - lv) * w
    else:
        w = 0.0 if iL else 1.0
        sm = lv if iL else rv
    analog = np.nan
    if an is not None and np.isfinite(an[t]):
        dL = edge_median(np.log1p(x) - np.log1p(an), iL) if iL else np.nan
        dR = edge_median(np.log1p(x) - np.log1p(an), iR) if iR else np.nan
        if np.isfinite(dL) and np.isfinite(dR):
            d = dL + (dR - dL) * w
        else:
            d = dL if np.isfinite(dL) else dR
        analog = float(np.expm1(np.log1p(an[t]) + d)) if np.isfinite(d) else np.nan
    return {"last": last, "raw": raw, "smooth": sm, "analog": analog}, (iL, iR, w)

def shift_stack(la):
    n = len(la)
    S = np.full((len(SHIFTS), n), np.nan)
    for k, s in enumerate(SHIFTS):
        if s >= 0:
            S[k, :n - s] = la[s:]
        else:
            S[k, -s:] = la[:n + s]
    return S

def weighted_median(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, 0.5 * c[-1])])

def soft_bridge(lx, prev, nxt, t, LS):
    n = len(lx)
    a, b = prev[t], nxt[t]
    iL = [i for i in range(max(a - FIT_W + 1, 0), a + 1) if np.isfinite(lx[i])] if a >= 0 else []
    iR = [i for i in range(b, min(b + FIT_W, n)) if np.isfinite(lx[i])] if b < n else []
    if not (iL or iR):
        return np.nan
    m = len(SHIFTS)
    E, cnt = np.zeros(m), np.zeros(m)
    offL, offR = np.full(m, np.nan), np.full(m, np.nan)
    for idx, side in ((iL, "L"), (iR, "R")):
        if not idx:
            continue
        D = lx[idx][None, :] - LS[:, idx]
        off = np.nanmedian(D[:, -EDGE_K:] if side == "L" else D[:, :EDGE_K], axis=1)
        if side == "L":
            offL = off
        else:
            offR = off
        e = np.nanmean(np.abs(D - np.nanmedian(D, axis=1, keepdims=True)), axis=1)
        ok = np.isfinite(e)
        E[ok] += e[ok]
        cnt[ok] += 1
    E = np.where(cnt > 0, E / np.maximum(cnt, 1), np.nan)
    if iL and iR:
        mL, mR = np.mean(iL[-EDGE_K:]), np.mean(iR[:EDGE_K])
        w = float(np.clip((t - mL) / (mR - mL), 0, 1))
    else:
        w = 0.0 if iL else 1.0
    off = np.where(np.isfinite(offL) & np.isfinite(offR), offL + (offR - offL) * w, np.where(np.isfinite(offL), offL, offR))
    pred = np.expm1(LS[:, t] + off)
    tau = max(float(np.nanmedian(E)), 0.01) if np.isfinite(E).any() else 1.0
    lw = -SHIFTS ** 2 / (2 * SHIFT_SD ** 2) - E / tau
    ok = np.isfinite(pred) & np.isfinite(lw)
    if not ok.any():
        return np.nan
    w8 = np.exp(lw[ok] - lw[ok].max())
    return weighted_median(pred[ok], w8)

def partner_arm(day, cam, t, sm, edges, part):
    if cam not in part:
        return sm
    ps, g = part[cam]
    iL, iR, w = edges
    devs = []
    for p in ps:
        xp = day.X[:, day.col(p)]
        lv, rv = edge_median(xp, iL), edge_median(xp, iR)
        line = lv + (rv - lv) * w if np.isfinite(lv) and np.isfinite(rv) else (lv if np.isfinite(lv) else rv)
        if np.isfinite(xp[t]) and np.isfinite(line):
            devs.append(math.log1p(xp[t]) - math.log1p(max(line, 0.0)))
    return float(np.expm1(math.log1p(max(sm, 0.0)) + g * np.mean(devs))) if devs else sm

def simulate(T, hide, rng):
    M = T.drop(columns=[c for c in hide if c in T.columns]).copy()
    n = len(T)
    blank = np.zeros(n, bool)
    nows, now = [], int(rng.integers(20, 27))
    while now + BLANK + 6 <= n - 1:
        nows.append(now)
        blank[now + 1:now + BLANK + 1] = True
        now += int(rng.integers(49, 61))
    M.iloc[blank] = np.nan
    holes = []
    for now in nows:
        for cam in M.columns:
            holes += [("forecast", cam, now + h, now) for h in range(1, H_MAX + 1) if np.isfinite(T[cam].iat[now + h])]
        for cam in hide:
            if cam in T.columns:
                holes += [("hidden-future", cam, now + h, now) for h in range(1, H_MAX + 1)
                          if np.isfinite(T[cam].iat[now + h])]
    for cam in hide:
        if cam in T.columns:
            holes += [("hidden", cam, t, -1) for t in range(n) if not blank[t] and np.isfinite(T[cam].iat[t])]
    holes += add_missing(M, blank, rng, n / 205, 5, 21)
    H = pd.DataFrame(holes, columns=["kind", "unit", "t", "now"])
    H["y"] = [T[c].iat[t] for c, t in zip(H["unit"], H["t"])]
    assert H["y"].notna().all()
    return M, H

def add_missing(M, blocked, rng, rate, lo, hi):
    n, holes = len(M), []
    for cam in M.columns:
        x = M[cam].to_numpy(float)
        taken = blocked.copy()
        for _ in range(rng.poisson(rate)):
            for _ in range(50):
                ln = int(rng.integers(lo, hi))
                s = int(rng.integers(1, n - ln - 1))
                seg = x[s:s + ln]
                if (taken[s - 1:s + ln + 1].any() or not (np.isfinite(x[s - 1]) and np.isfinite(x[s + ln]))
                        or not np.isfinite(seg).all()):
                    continue
                taken[s - 1:s + ln + 1] = True
                holes += [("missing", cam, t, -1) for t in range(s, s + ln)]
                x[s:s + ln] = np.nan
                break
        M[cam] = x
    return holes

def test_holes(holes, first):
    H = pd.DataFrame({"id": holes["id"].to_numpy(), "kind": holes["kind"].to_numpy(), "unit": holes["unit"].to_numpy(),
                      "t": (holes["minute_of_day"] - first).astype(int).to_numpy()})
    H["now"] = np.where(holes["now_minute_of_day"].notna(), holes["now_minute_of_day"].fillna(0) - first, -1).astype(int)
    return H
