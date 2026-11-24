import numpy as np
import pandas as pd

MOMENTS = [0, 50, 100, 125, 150, 175, 200, 250, 300, 400, 500]
PRE = 20
ONSET = 21


def ffill(a):
    a = a.copy()
    for k in range(1, a.shape[1]):
        m = np.isnan(a[:, k])
        a[m, k] = a[m, k - 1]
    return a


def cut_frame(t, ms):
    return int(np.searchsorted(t, ms / 1000 + 1e-6)) - 1


def pmu_features(X, t, ch, cuts_ms, jump):
    n = len(X)
    col = {c: ffill(X[:, :, ch.index(c)]) for c in ch}
    pre = {c: np.nanmedian(col[c][:, :PRE], 1) for c in ch}
    volt = np.stack([col[c] / pre[c][:, None] for c in ("V4", "V7", "V9")], 2)
    raw_volt = np.stack([X[:, :, ch.index(c)] / pre[c][:, None] for c in ("V4", "V7", "V9")], 2)
    raw_f = np.stack([X[:, :, ch.index(f"f{p}")] - 50.0 for p in ("4", "7", "9")], 2)
    rise = np.nan_to_num(np.diff(volt, axis=1), nan=-9).max(2)
    th = {a + b: col[f"theta{a}"] - col[f"theta{b}"] for a, b in (("4", "7"), ("9", "7"), ("4", "9"))}
    fdev = np.stack([col[f"f{p}"] - 50.0 for p in ("4", "7", "9")], 2)
    rows = np.arange(n)
    k = np.array([cut_frame(t, ms) for ms in cuts_ms])
    f = {"cut_s": np.asarray(cuts_ms) / 1000}
    frames = np.arange(X.shape[1])[None, :]
    live = (frames >= ONSET) & (frames <= k[:, None])
    hit = (rise[:, ONSET - 1:] > jump) & live[:, ONSET:]
    cleared = hit.any(1)
    clear_k = np.where(cleared, hit.argmax(1) + ONSET, -1)
    f["cleared"] = cleared.astype(float)
    f["t_clear"] = np.where(cleared, t[np.maximum(clear_k, 0)], np.nan)
    f["since_clear"] = np.where(cleared, f["cut_s"] - f["t_clear"], np.nan)
    f["fault_on_s"] = np.where(cleared, f["t_clear"], f["cut_s"]) - 0.01
    on = (frames >= ONSET) & (frames < np.where(cleared, clear_k, k + 1)[:, None])
    for j, p in enumerate(("4", "7", "9")):
        v = np.where(on, raw_volt[:, :, j], np.nan)
        with np.errstate(all="ignore"):
            f[f"fault_v{p}"] = np.nanmean(v, 1)
        f[f"now_v{p}"] = volt[rows, k, j]
        f[f"now_df{p}"] = fdev[rows, k, j]
    post = (frames >= ONSET) & (frames <= k[:, None])
    for name, a in th.items():
        d = a - np.nanmedian(a[:, :PRE], 1)[:, None]
        f[f"now_th{name}"] = d[rows, k]
        f[f"rate_th{name}"] = d[rows, k] - d[rows, np.maximum(k - 5, 0)]
        with np.errstate(all="ignore"):
            f[f"max_th{name}"] = np.nanmax(np.where(post, np.abs(d), np.nan), 1)
    with np.errstate(all="ignore"):
        spread = np.nanmax(fdev, 2) - np.nanmin(fdev, 2)
        f["now_fspread"] = spread[rows, k]
        f["max_fspread"] = np.nanmax(np.where(post, spread, np.nan), 1)
        f["max_fdev"] = np.nanmax(np.where(post[:, :, None], np.abs(fdev), np.nan), (1, 2))
    for c in ("P_L4-5", "P_L4-6", "P_L7-8"):
        f[f"now_{c}"] = col[c][rows, k] - pre[c]
    for c in ("V4", "V7", "V9", "P_L4-5", "P_L4-6", "P_L7-8"):
        f[f"pre_{c}"] = pre[c]
    for j, p in enumerate(("4", "7", "9")):
        slope = (raw_f[:, ONSET + 3, j] - fdev[:, PRE, j]) / (t[ONSET + 3] - t[PRE])
        f[f"slope_f{p}"] = np.where(k >= ONSET + 3, slope, np.nan)
    f["pre_th47"] = np.nanmedian(th["47"][:, :PRE], 1)
    f["pre_th97"] = np.nanmedian(th["97"][:, :PRE], 1)
    return pd.DataFrame(f)


def train_rows(X, t, ch, ctx, y, jump):
    parts = []
    for ms in MOMENTS:
        k = cut_frame(t, ms)
        Xc = X.copy()
        Xc[:, k + 1:] = np.nan
        feats = pmu_features(Xc, t, ch, [ms] * len(X), jump)
        parts.append(pd.concat([ctx.reset_index(drop=True), feats], axis=1).assign(event=np.arange(len(X)), y=y))
    return pd.concat(parts, ignore_index=True)


def truncated_rows(X, t, ch, ctx, windows, jump):
    parts = []
    for ms in MOMENTS:
        idx = np.where(windows >= ms)[0]
        if not len(idx):
            continue
        Xc = X[idx].copy()
        Xc[:, cut_frame(t, ms) + 1:] = np.nan
        feats = pmu_features(Xc, t, ch, [ms] * len(idx), jump)
        parts.append(pd.concat([ctx.iloc[idx].reset_index(drop=True), feats], axis=1).assign(
            event=idx, own=windows[idx] == ms))
    return pd.concat(parts, ignore_index=True)


def test_rows(X, t, ch, ctx, windows, jump):
    feats = pmu_features(X, t, ch, windows, jump)
    return pd.concat([ctx.reset_index(drop=True), feats], axis=1)
