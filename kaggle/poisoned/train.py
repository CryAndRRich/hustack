import time

import numpy as np
import pandas as pd

from preprocess import (CANDIDATES, LAYERS, N_TRAIN, WIDTH, forward, get_data_dir, load_data, load_heldout, load_weights,
                        metrics, out_dir)
from repair import min_norm_inspan, removal, span_basis

VERSION = "v7"
WRITEUP_URL = "https://www.kaggle.com/competitions/fixing-the-poisoned-well/discussion/link-to-blog"
R_SPAN = 4
SPAN_EPS = 1e-5
VIS_SHARE = 1e-3
NULL_LAMBDA = 1e-9
LEAK_HELDOUT = 0.0019 - 0.0001
N_STARTS = 8
MONTHS = range(1, 13)
FEATS = ["T_air_C", "sm_2in", "sm_20in", "season", "srad_Wm2", "rain24", "rain72"]

def unit(v):
    return v / np.linalg.norm(v)

def undo_rescale_decoys(W, acts, layer):
    i = LAYERS.index(layer)
    prev = LAYERS[i - 1]
    H, w = acts[i].copy(), W[layer + ".weight"].copy()
    live = H.max(0) > 0
    live_prev = acts[i - 1].max(0) > 0 if i > 1 else np.ones(acts[0].shape[1], bool)
    inn = ((W[prev + ".weight"] * live_prev) ** 2).sum(1) + W[prev + ".bias"] ** 2
    out = (w ** 2).sum(0)
    bad = live & (inn > 30 * np.median(inn[live])) & (out < np.median(out[live]) / 30)
    c = np.ones(WIDTH)
    c[bad] = (out[bad] / inn[bad]) ** 0.25
    return H * c, w / c, c, np.where(bad)[0]

def outspan(W, acts, layer):
    i = LAYERS.index(layer)
    H, w, c, bad = undo_rescale_decoys(W, acts, layer)
    li, lo = H.max(0) > 0, acts[i + 1].max(0) > 0
    e, V = span_basis(H[:, li])
    r = min(max(R_SPAN, int((e > SPAN_EPS).sum())), int(li.sum()) - 3)
    wl = w[lo][:, li]
    out = {}
    for k in range(1, min(r + 3, int(li.sum()))):
        R = wl - (wl @ V[:, :k]) @ V[:, :k].T
        U, s, Vt = np.linalg.svd(R)
        v, d = np.zeros(WIDTH), np.zeros(WIDTH)
        v[lo], d[li] = U[:, 0] * s[0], Vt[0]
        d = c * d
        out[k] = {"s": s, "v": v * np.linalg.norm(d), "d": unit(d), "R": np.linalg.norm(R)}
    B = np.zeros((WIDTH, r))
    B[li] = V[:, :r]
    dom = (out[r]["s"][0] - out[r]["s"][1]) / np.linalg.norm(wl)
    return {"r": r, "e": e, "bad": bad, "c": c, "B": c[:, None] * B, "fits": out, "dom": dom}

def band_report(H, d):
    live = H.max(0) > 0
    Hl = H[:, live]
    ev, V = np.linalg.eigh(Hl.T @ Hl / len(Hl))
    c2 = (V.T @ d[live]) ** 2
    edges = [np.inf, 1e-3, 1e-6, 1e-9, 1e-12, -np.inf]
    return " | ".join(f"lambda {edges[j + 1]:.0e}..{edges[j]:.0e}: {c2[(ev < edges[j]) & (ev >= edges[j + 1])].sum() / c2.sum():.3f}"
                      for j in range(len(edges) - 1))

def drop_null_band(H, d):
    live = H.max(0) > 0
    ev, V = np.linalg.eigh(H[:, live].T @ H[:, live] / len(H))
    c = V.T @ d[live]
    c[ev < NULL_LAMBDA] = 0
    out = np.zeros(WIDTH)
    out[live] = V @ c
    return out

def clean_weak_part(W, acts, k, v, q):
    li, lo = acts[5].max(0) > 0, acts[6].max(0) > 0
    _, V = span_basis(acts[5][:, li])
    cols = W["fc6.weight"][lo][:, li] @ V
    S = np.linalg.svd(cols[:, :k] - np.outer(v[lo], q[li] @ V[:, :k]), full_matrices=False)[0]
    dw = np.linalg.lstsq(np.c_[v[lo], S], cols[:, k:], rcond=None)[0][0]
    d = np.zeros(WIDTH)
    d[li] = V[:, k:] @ dw
    return d

def describe(W, X, y, pred, dfw, Xh, ph, meta, w6, name):
    new = forward(W, X, {"fc6": w6})[0]
    newh = forward(W, Xh, {"fc6": w6})[0]
    leak, leakh = pred - new, ph - newh
    m = metrics(new, y)
    tr = dfw["split"].to_numpy() == "train"
    print(f"   {name}: |dW|F {np.linalg.norm(W['fc6.weight'] - w6):.4f} | public bias {m['bias']:+.6f} RMSE {m['RMSE']:.6f}"
          f" NSE {m['NSE']:.4f} | train bias {np.mean(new[tr] - y[tr]):+.6f} val bias {np.mean(new[~tr] - y[~tr]):+.6f}")
    print(f"      held-out removal {leakh.mean() * 1e3:.3f}e-3 (page leak {LEAK_HELDOUT * 1e3:.1f}e-3), sd {leakh.std() * 1e3:.3f}"
          f" | by month " + " ".join(f"{k}:{leakh[meta['datetime'].dt.month == k].mean() * 1e3:+.2f}" for k in (12, 1, 2, 3, 4, 5)))
    month = dfw["datetime"].dt.month.to_numpy()
    print("      public removal by month x1e3: " + " ".join(f"{leak[month == k].mean() * 1e3:+.2f}" for k in MONTHS)
          + f" | T<5C {leak[dfw['T_air_C'] < 5].mean() * 1e3:+.2f} T>=5C {leak[dfw['T_air_C'] >= 5].mean() * 1e3:+.2f}")
    print("      corr(public removal, feature): " + " ".join(f"{k} {np.corrcoef(leak, dfw[k])[0, 1]:+.2f}" for k in FEATS))

def write_submission(path, w_fixed, direction):
    assert w_fixed.shape == (WIDTH, WIDTH) and direction.shape == (WIDTH,)
    assert np.isfinite(w_fixed).all() and np.isfinite(direction).all()
    ids = [f"fc_{r}_{c}" for r in range(WIDTH) for c in range(WIDTH)] + [f"dir_{k}" for k in range(WIDTH)]
    vals = np.concatenate([w_fixed.ravel(), unit(direction)])
    sub = pd.DataFrame({"id": ["pub_" + i for i in ids] + ["prv_" + i for i in ids],
                        "value": np.concatenate([vals, vals]), "writeup_url": WRITEUP_URL})
    assert len(sub) == 6384 and sub["id"].is_unique
    sub.to_csv(path, index=False)

def run(out):
    t0 = time.time()
    d = get_data_dir()
    W = load_weights(d)
    X, y, dfw = load_data(d)
    pred, acts, _ = forward(W, X)
    Xh, meta, gap = load_heldout(d)
    ph, acts_h, _ = forward(W, Xh)
    base = metrics(pred, y)
    print(f"poisoned public: bias {base['bias']:+.6f} RMSE {base['RMSE']:.6f} NSE {base['NSE']:.4f}")
    print(f"held-out rebuilt: {len(Xh)} hours {meta['datetime'].min()} -> {meta['datetime'].max()} | scaler check max |mean - mu|/sig"
          f" {gap.max():.4f} | poisoned mean prediction {ph.mean():.5f}")

    print("\n== step 1, data-free: out-of-span part of every candidate layer (training leaves a flat floor there, a rank-1"
          " edit one dominant singular value)")
    spans = {}
    for layer in CANDIDATES:
        sp = outspan(W, acts, layer)
        spans[layer] = sp
        o = sp["fits"][sp["r"]]
        print(f"{layer}: r {sp['r']}, rescale decoys undone at {sp['bad'].tolist()} | sv {o['s'][:4].round(4).tolist()}"
              f" | dominance (s1 - s2)/|W| {sp['dom']:.4f}")
    layer = max(CANDIDATES, key=lambda k: spans[k]["dom"])
    assert layer == "fc6", "steps 2-3 are written for fc6"
    sp = spans[layer]
    H5, H5h = acts[5], acts_h[5]
    k = int((sp["e"] >= VIS_SHARE).sum())
    v, dperp, B = sp["fits"][k]["v"], sp["fits"][k]["d"], sp["B"][:, :k]
    print(f"\nLAYER {layer} | input span energy shares {sp['e'][:7].round(6).tolist()} -> {k} strong directions (share >= {VIS_SHARE}):"
          f" | |v| {np.linalg.norm(v):.4f}, sv {sp['fits'][k]['s'][:4].round(4).tolist()}")
    for kk in range(max(1, k - 1), k + 4):
        if kk in sp["fits"]:
            print(f"   stability vs k={kk}: |cos d| {abs(dperp @ sp['fits'][kk]['d']):.4f} |cos v| {abs(unit(v) @ unit(sp['fits'][kk]['v'])):.4f}")
    print(f"   d_perp energy by eigenvalue of E[h h^T]: {band_report(H5, sp['fits'][sp['r']]['d'])}")
    out_h = H5h - (H5h @ sp["B"]) @ sp["B"].T
    print(f"   held-out activations outside the public span (r={sp['r']}): energy share {np.sum(out_h ** 2) / np.sum(H5h ** 2):.2e}"
          f" | fc1-fc6 units alive on held-out only: {[int(((acts_h[i].max(0) > 0) & ~(acts[i].max(0) > 0)).sum()) for i in range(1, 7)]}")

    assert not len(sp["bad"]), "clean_weak_part assumes no rescale decoy at the layer's input"
    bias_tr = float(np.mean(pred[:N_TRAIN] - y[:N_TRAIN]))
    rng = np.random.default_rng(0)
    starts = [np.full(k, 0.1)] + [rng.normal(0, 0.3, k) for _ in range(N_STARTS - 1)]
    conds = [(H5h, ph, LEAK_HELDOUT), (H5[:N_TRAIN], pred[:N_TRAIN], bias_tr)]

    def physics(d0, name):
        c, viol = min_norm_inspan(W, layer, v, d0, B, conds, starts)
        d = d0 + B @ c
        print(f"   {name}: c {np.round(c, 4).tolist()} |q| {np.linalg.norm(B @ c):.4f} (residual {viol:.1e}) | removal held-out"
              f" {removal(W, layer, H5h, ph, v, d) * 1e3:.3f} train {removal(W, layer, H5[:N_TRAIN], pred[:N_TRAIN], v, d) * 1e3:.3f}"
              f" (poisoned train bias {bias_tr * 1e3:.3f}) val {removal(W, layer, H5[N_TRAIN:], pred[N_TRAIN:], v, d) * 1e3:.3f} (e-3)")
        return d

    print("\n== step 2, physics (V4): in-span part q for the out-of-span part read as the projection on v")
    d4 = physics(dperp, "V4")
    print("\n== step 3, clean weight in the weak span dims: refit each weak column as v d_j + clean strong output subspace")
    d0 = clean_weak_part(W, acts, k, v, d4 - dperp)
    print(f"   corrected out-of-span part: |d0| {np.linalg.norm(d0):.4f} (d_perp 1), cos(d0, d_perp) {abs(unit(d0) @ dperp):.4f}"
          f" | energy by eigenvalue of E[h h^T]: {band_report(H5, d0)}")
    d_new = physics(d0, VERSION.upper())
    print(f"   cos(d, d_V4) {abs(unit(d_new) @ unit(d4)):.4f}")

    print("\n== step 4: repair")
    w_new = W["fc6.weight"] - np.outer(v, d_new)
    describe(W, X, y, pred, dfw, Xh, ph, meta, w_new, VERSION.upper())

    direction = drop_null_band(H5, d_new)
    print(f"   direction: near-null band (lambda < {NULL_LAMBDA:.0e}) dropped, cos with the edit's d {unit(direction) @ unit(d_new):.5f}"
          f" | {band_report(H5, direction)}")
    (out / "extra").mkdir(exist_ok=True)
    write_submission(out / "submission.csv", w_new, direction)
    np.savez(out / "extra" / f"train_{VERSION}.npz", v=v, dperp=dperp, d0=d0, B=B, k=k, d=d_new, d4=d4, direction=direction)
    print(f"\nSUBMIT: submission.csv = {VERSION}, layer {layer}, edit v (d0 + q)^T: out-of-span part cleaned of the weak-dim clean"
          f" weight, physics-anchored q on the host's held-out inputs (|dW|F {np.linalg.norm(W['fc6.weight'] - w_new):.4f}); direction"
          f" without its near-null band; total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    run(out_dir())
