import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import (BASE_CAMS, Day, Source, add_missing, build_analogs, gap_arms, load, partner_arm, partners,
                        prev_next, ratio_estimate, shift_stack, simulate, soft_bridge, table, test_holes)

SCENARIOS = {"fri": ["cam_B", "cam_G"], "sat": ["cam_D"]}
ARMS = {"forecast": ["last", "analog", "soft"], "missing": ["raw", "smooth", "analog", "partner"],
        "hidden": ["donor", "ratio", "ratio1"], "hidden-future": ["last", "analog", "dfc"]}
BUCKETED = {"forecast", "hidden-future"}
ARM_COLS = sorted({a for v in ARMS.values() for a in v})
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
EXTRA = OUT / "extra" / "V3"
warnings.filterwarnings("ignore", category=RuntimeWarning)

def visible_of(scn):
    return [c for c in BASE_CAMS if c not in SCENARIOS[scn]]

def donors(truth, names, hide, avail):
    res = {}
    for cam in hide:
        best = None
        for d in BASE_CAMS:
            if d in hide or d not in avail:
                continue
            pairs = [(truth[n][cam], truth[n][d]) for n in names if cam in truth[n] and d in truth[n]]
            if not pairs:
                continue
            y = np.concatenate([a[a.notna() & b.notna()].to_numpy() for a, b in pairs])
            x = np.concatenate([b[a.notna() & b.notna()].to_numpy() for a, b in pairs])
            s = y.sum() / x.sum()
            mae = np.abs(y - s * x).mean()
            if best is None or mae < best[2]:
                best = (d, s, mae)
        res[cam] = best[:2]
    return res

def predict_arms(day, H, scn, sources, train_sources, donor):
    hide = SCENARIOS[scn]
    vis = [c for c in visible_of(scn) if c in day.cams]
    an = build_analogs(day, sources, sorted(set(day.cams) | set(hide)))
    part = partners(day)
    out = {a: np.full(len(H), np.nan) for a in ARM_COLS}
    kinds, units, ts = H["kind"].to_numpy(), H["unit"].to_numpy(), H["t"].to_numpy()
    LX = {c: np.log1p(day.X[:, day.col(c)]) for c in day.cams}
    LS = {c: shift_stack(np.log1p(an[c])) for c in day.cams if c in an}
    for i in np.flatnonzero(np.isin(kinds, ["forecast", "missing"])):
        u, j = units[i], day.col(units[i])
        arms, edges = gap_arms(day.X[:, j], day.prev[:, j], day.next[:, j], ts[i], an.get(u))
        if kinds[i] == "forecast":
            arms["soft"] = soft_bridge(LX[u], day.prev[:, j], day.next[:, j], ts[i], LS[u]) if u in LS else np.nan
        else:
            arms["partner"] = partner_arm(day, u, ts[i], arms["smooth"], edges, part)
        for a in ARMS[kinds[i]]:
            out[a][i] = arms[a]
    seen = np.isfinite(day.X).any(1)
    for cam in hide:
        kh = np.flatnonzero((kinds == "hidden") & (units == cam))
        kf = np.flatnonzero((kinds == "hidden-future") & (units == cam))
        if not (len(kh) or len(kf)):
            continue
        ratio, ratio1 = ratio_estimate(day, cam, vis, sources, train_sources)
        d, s = donor[cam]
        j = day.col(d)
        pd_ = day.prev[:, j]
        don = np.where(pd_ >= 0, s * day.X[np.maximum(pd_, 0), j], np.nan)
        out["donor"][kh], out["ratio"][kh], out["ratio1"][kh] = don[ts[kh]], ratio[ts[kh]], ratio1[ts[kh]]
        if not len(kf):
            continue
        h = np.where(seen, np.where(np.isfinite(ratio), ratio, don), np.nan)
        hp, hn = prev_next(h)
        logs = {}
        for v in vis:
            jv = day.col(v)
            lv = np.full(len(h), np.nan)
            for t in np.unique(ts[kf]):
                p = soft_bridge(LX[v], day.prev[:, jv], day.next[:, jv], t, LS[v]) if v in LS else np.nan
                if not np.isfinite(p) and day.prev[t, jv] >= 0:
                    p = day.X[day.prev[t, jv], jv]
                lv[t] = np.log1p(p)
            logs[v] = lv
        dfc = ratio_estimate(day, cam, vis, sources, train_sources, logs=logs)[0]
        for i in kf:
            arms, _ = gap_arms(h, hp, hn, ts[i], an.get(cam))
            out["last"][i], out["analog"][i], out["dfc"][i] = arms["last"], arms["analog"], dfc[ts[i]]
    return pd.DataFrame(out, index=H.index).clip(lower=0)

def references(T, day, H, A, donor):
    ref = np.full(len(H), np.nan)
    for i, (kind, cam, t, now) in enumerate(H[["kind", "unit", "t", "now"]].itertuples(index=False)):
        if kind == "forecast":
            j = day.col(cam)
            ref[i] = day.X[day.prev[now, j], j]
        elif kind == "missing":
            ref[i] = A["raw"].iat[i]
        else:
            d, s = donor[cam]
            x = T[d].to_numpy(float)[:(t if kind == "hidden" else now) + 1]
            x = x[np.isfinite(x)]
            ref[i] = s * x[-1] if len(x) else np.nan
    return ref

def make_sources(truth, test_tabs, meta, names):
    return ([Source(n, *meta[n], truth[n]) for n in names]
            + [Source(n, *meta[n], T) for n, T in test_tabs.items()])

def cv(truth, test_tabs, meta, names, held_out, draws):
    rows = []
    for held in held_out:
        t0 = time.time()
        others = [m for m in names if m != held]
        sources = make_sources(truth, test_tabs, meta, others)
        train_sources = [s for s in sources if s.name in others]
        for scn, hide in SCENARIOS.items():
            donor = donors(truth, others, hide, truth[held].columns)
            for d in range(draws):
                rng = np.random.default_rng(10_000 + 97 * d + (0 if scn == "fri" else 5))
                M, H = simulate(truth[held], hide, rng)
                day = Day(held, *meta[held], M)
                A = predict_arms(day, H, scn, sources, train_sources, donor)
                A["ref"] = references(truth[held], day, H, A, donor)
                A["y"], A["kind"], A["unit"] = H["y"].to_numpy(), H["kind"].to_numpy(), H["unit"].to_numpy()
                A["h"] = np.where(H["now"] >= 0, H["t"] - H["now"], 0)
                A["held"], A["scn"], A["draw"] = held, scn, d
                rows.append(A[np.isfinite(A["ref"])])
        print(f"  cv {held}: {time.time() - t0:.0f}s")
    return pd.concat(rows, ignore_index=True)

def arm_matrix(R, kind):
    A = R[ARMS[kind]].to_numpy(float)
    return np.where(np.isfinite(A), A, A[:, :1])

def buckets(R):
    return np.where(R["kind"].isin(BUCKETED), (R["h"].to_numpy() - 1) // 5, 0)

def row_weights(R):
    den = (R["y"] - R["ref"]).abs().groupby([R["held"], R["scn"], R["kind"]]).transform("sum")
    return 1.0 / np.maximum(den.to_numpy(), 1e-9)

def simplex(k, steps=10):
    return [np.array(c) / steps for c in itertools.product(range(steps + 1), repeat=k) if sum(c) == steps]

def best_weights(A, y, om):
    grid = simplex(A.shape[1])
    errs = [(om * np.abs(A @ w - y)).sum() for w in grid]
    return grid[int(np.argmin(errs))]

def fit_weights(R):
    W, om, bk = {}, row_weights(R), buckets(R)
    for kind in ARMS:
        for b in np.unique(bk[(R["kind"] == kind).to_numpy()]):
            k = ((R["kind"] == kind).to_numpy() & (bk == b))
            W[(kind, b)] = best_weights(arm_matrix(R[k], kind), R["y"].to_numpy()[k], om[k])
    return W

def apply(R, W):
    p, bk = np.full(len(R), np.nan), buckets(R)
    for (kind, b), w in W.items():
        k = (R["kind"] == kind).to_numpy() & (bk == b)
        if k.any():
            p[k] = arm_matrix(R[k], kind) @ w
    return p

def group_ratios(R, p):
    e = pd.DataFrame({"g": list(zip(R["held"], R["scn"])), "kind": R["kind"], "e": np.abs(R["y"] - p),
                      "r": np.abs(R["y"] - R["ref"])})
    s = e.groupby(["kind", "g"])[["e", "r"]].sum()
    return (s["e"] / s["r"]).groupby("kind").mean()

def evaluate(R):
    lines = []
    for kind in ARMS:
        Rk = R[R["kind"] == kind]
        A = arm_matrix(Rk, kind)
        single = {a: group_ratios(Rk, A[:, i])[kind] for i, a in enumerate(ARMS[kind])}
        lines.append(f"  {kind:14s} n {len(Rk):6d}  arms " + ", ".join(f"{a} {v:.4f}" for a, v in single.items()))
    print("\n".join(lines))
    nested = np.full(len(R), np.nan)
    for h in R["held"].unique():
        te = (R["held"] == h).to_numpy()
        nested[te] = apply(R[te], fit_weights(R[~te]))
    W = fit_weights(R)
    for (kind, b), w in W.items():
        print(f"  weights {kind:14s} bucket {b}: {dict(zip(ARMS[kind], w.round(2)))}")
    gr = group_ratios(R, nested)
    print("  nested stack per kind (mean of per-recording ratios): " + ", ".join(f"{k} {v:.4f}" for k, v in gr.items())
          + f"  | score {gr.mean():.4f}")
    return W, nested

def by_group(R, p, keys):
    out = []
    for g, Rg in R.groupby(keys):
        gr = group_ratios(Rg, p[Rg.index])
        out.append({"group": g, **{k: round(v, 3) for k, v in gr.items()}, "mean": round(gr.mean(), 3)})
    print(pd.DataFrame(out).to_string(index=False))

def test_days(te, holes):
    out = []
    for name, g in te.groupby("recording"):
        hh = holes[holes["recording"] == name]
        T = table(g)
        lo = int(min(T.index.min(), hh["minute_of_day"].min()))
        hi = int(max(T.index.max(), hh["minute_of_day"].max()))
        T = T.reindex(np.arange(lo, hi + 1))
        hide = sorted(hh.loc[hh["kind"] == "hidden", "unit"].unique())
        scn = next(k for k, v in SCENARIOS.items() if sorted(v) == hide)
        assert not set(hide) & set(T.columns), "a hidden camera has released rows"
        H = test_holes(hh, lo)
        assert (H["t"] >= 0).all() and (H["now"][H["kind"].isin(["forecast", "hidden-future"])] >= 0).all()
        out.append((name, T, H, scn))
        print(f"test {name}: scenario {scn} (hidden {hide}), {len(H)} holes, cams {list(T.columns)}")
    return out

def pseudo_checks(tests, truth, test_tabs, meta, names):
    sources = make_sources(truth, test_tabs, meta, names)
    train_sources = [s for s in sources if s.name in names]
    print("test-day pseudo holes (released minutes only; report, not used for weights)")
    for name, T, _, scn in tests:
        M = T.copy()
        Hm = pd.DataFrame(add_missing(M, np.zeros(len(M), bool), np.random.default_rng(7), len(M) / 60, 5, 13),
                          columns=["kind", "unit", "t", "now"])
        y = np.array([T[c].iat[t] for c, t in zip(Hm["unit"], Hm["t"])])
        A = predict_arms(Day(name, *meta[name], M), Hm, scn, sources, train_sources, {})
        grp = np.where(Hm["unit"].str[-1].isin(list("IJKLM")), "I-M", "A-H")
        for g in ("A-H", "I-M"):
            k = grp == g
            den = np.abs(y[k] - A["raw"].to_numpy()[k]).sum()
            print(f"  {name} pseudo-missing {g} n {k.sum()}: " + ", ".join(
                f"{a} {np.abs(y[k] - np.where(np.isfinite(A[a].to_numpy()[k]), A[a].to_numpy()[k], A['raw'].to_numpy()[k])).sum() / den:.3f}"
                for a in ARMS["missing"]))
        day = Day(name, *meta[name], T)
        vis = [c for c in visible_of(scn) if c in T.columns]
        res = []
        for c in vis:
            donor = donors(truth, names, [c], [v for v in vis if v != c])
            ratio, ratio1 = ratio_estimate(day, c, [v for v in vis if v != c], sources, train_sources)
            d, s = donor[c]
            yy, ref = T[c].to_numpy(float), s * T[d].to_numpy(float)
            k = np.isfinite(yy) & np.isfinite(ref)
            den = np.abs(yy[k] - ref[k]).sum()
            fill = lambda a: np.where(np.isfinite(a[k]), a[k], ref[k])
            res.append((c, d, np.abs(yy[k] - fill(ratio)).sum() / den, np.abs(yy[k] - fill(ratio1)).sum() / den))
        print(f"  {name} pseudo-hidden (each visible base cam; ratio vs host-style single donor): "
              + ", ".join(f"{c[-1]}<-{d[-1]} {r:.3f}/{r1:.3f}" for c, d, r, r1 in res)
              + f" | mean ratio {np.mean([r for _, _, r, _ in res]):.3f}, ratio1 {np.mean([r for _, _, _, r in res]):.3f}")

def run(smoke):
    tr, te, holes = load()
    names = sorted(tr["recording"].unique())
    truth = {r: table(g) for r, g in tr.groupby("recording")}
    meta = {r: (g["day"].iat[0], g["weekday"].iat[0]) for r, g in pd.concat([tr, te]).groupby("recording")}
    tests = test_days(te, holes)
    test_tabs = {name: T for name, T, _, _ in tests}
    draws = 1 if smoke else 5
    held = [names[0], names[-1]] if smoke else names
    print(f"== {'smoke' if smoke else 'full'}: draws {draws}, held-out {held}")
    R = cv(truth, test_tabs, meta, names, held, draws)
    print("CV (ratios are means of per-recording-and-scenario ratios, as the LB scores one recording)")
    W, nested = evaluate(R)
    print("CV nested stack per scenario")
    by_group(R, nested, ["scn"])
    print("CV nested stack per held-out recording and scenario")
    by_group(R, nested, ["held", "scn"])
    pseudo_checks(tests, truth, test_tabs, meta, names)

    sources = make_sources(truth, test_tabs, meta, names)
    train_sources = [s for s in sources if s.name in names]
    preds = []
    for name, T, H, scn in tests:
        donor = donors(truth, names, SCENARIOS[scn], T.columns)
        A = predict_arms(Day(name, *meta[name], T), H, scn, sources, train_sources, donor)
        A["kind"], A["h"] = H["kind"].to_numpy(), np.where(H["now"] >= 0, H["t"] - H["now"], 0)
        p = apply(A, W)
        print(f"  {name}: donors {donor}; arm NaN share within kind " + ", ".join(
            f"{k}: " + "/".join(f"{A.loc[A['kind'] == k, a].isna().mean():.2f}" for a in ARMS[k]) for k in ARMS))
        preds.append(pd.concat([H[["id"]].reset_index(drop=True), A.reset_index(drop=True)], axis=1).assign(people=p))
    S = pd.concat(preds, ignore_index=True)
    assert S["id"].is_unique and set(S["id"]) == set(holes["id"]) and np.isfinite(S["people"]).all()
    assert (S["people"] >= 0).all()
    print(S.groupby("kind")["people"].describe().round(2).to_string())
    if smoke:
        return
    S = S.set_index("id").loc[holes["id"]].reset_index()
    EXTRA.mkdir(parents=True, exist_ok=True)
    R.assign(nested=nested).to_csv(EXTRA / "cv_rows.csv.gz", index=False)
    S.to_csv(EXTRA / "test_arms.csv", index=False)
    S[["id", "people"]].to_csv(OUT / "submission.csv", index=False)
    print(f"SUBMIT: submission.csv ({len(S)} rows; " + "; ".join(
        f"{k}/{b} {dict(zip(ARMS[k], w.round(2)))}" for (k, b), w in W.items()) + ")")

if __name__ == "__main__":
    run(smoke=True)
    run(smoke=False)
