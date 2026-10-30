import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from model import CensoredNB, expected_sales
from preprocess import GAP_START, build, load

CFG = {"lam": {"sp": 1.0, "sdow": 5.0, "pdow": 5.0, "day": 2.0, "dayp": 10.0}, "lam_x": 1.0, "lam_a": 0.01,
       "future_days": 84, "gap_blocks": 1344, "maxiter": 3000, "shops": None, "margin": 0.02,
       "arms": {"mult_dayp100": {"events": "mult", "extra": [], "lam": {"dayp": 100.0}},
                "dayp1000": {"events": "mult", "extra": ["price_dow"], "lam": {"dayp": 1000.0}},
                "dayp1e4": {"events": "mult", "extra": ["price_dow"], "lam": {"dayp": 1e4}},
                "no_dayp": {"events": "mult", "extra": ["price_dow"], "lam": {"dayp": 1e8}},
                "dayp1000_sdow50": {"events": "mult", "extra": ["price_dow"], "lam": {"dayp": 1000.0, "sdow": 50.0}}},
       "incumbent": {"gap": "dayp1000", "future": ("mult_dayp100", "season")}}
SMOKE = {"maxiter": 30, "shops": 6, "gap_blocks": 60}
VARIANTS = ("level", "season")
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def features(df, E, Es, n_prod, events, extra):
    P = np.eye(n_prod)[df["prod"].values]
    loc = pd.get_dummies(df.location_type).reindex(columns=["residential", "transit"], fill_value=0).values.astype(float)
    t, rain = df.temperature_forecast_c.values / 10, df.rain_forecast_mm.values / 5
    season = df[["season_sin", "season_cos"]].values
    parts = {"disc_prod": df.disc.values[:, None] * P, "disc_loc": df.disc.values[:, None] * loc,
             "temp_loc": loc * t[:, None], "rain_loc": loc * rain[:, None],
             "temp_prod": P[:, 1:] * t[:, None], "rain_prod": P[:, 1:] * rain[:, None],
             "season_prod": (P[:, 1:, None] * season[:, None, :]).reshape(len(df), -1),
             "season_loc": (loc[:, :, None] * season[:, None, :]).reshape(len(df), -1)}
    if "price_dow" in extra:
        parts["disc_dow"] = df.disc.values[:, None] * np.eye(7)[df.dow.values][:, 1:]
    ev = lambda M: (M[:, :, None] * P[:, None, :]).reshape(len(df), -1)
    if events == "mult":
        parts["events"] = ev(E)
    names = [f"{k}{i}" for k, v in parts.items() for i in range(v.shape[1])]
    return np.hstack(list(parts.values())), names, (ev(Es) if events == "add" else None)


def fit_nb(cfg, df, E, Es, arm, rows):
    n_shop, n_prod, n_day = df.shop.max() + 1, df["prod"].max() + 1, df.day.max() + 1
    blocks = {"sp": (n_shop * n_prod, df.shop.values * n_prod + df["prod"].values),
              "sdow": (n_shop * 7, df.shop.values * 7 + df.dow.values),
              "pdow": (n_prod * 7, df["prod"].values * 7 + df.dow.values),
              "day": (n_day, df.day.values),
              "dayp": (n_day * n_prod, df.day.values * n_prod + df["prod"].values)}
    X, names, A = features(df, E, Es, n_prod, arm["events"], arm["extra"])
    lam = {**cfg["lam"], **arm["lam"]}
    m = CensoredNB({b: v[0] for b, v in blocks.items()}, lam, X.shape[1], cfg["lam_x"],
                   0 if A is None else A.shape[1], cfg["lam_a"])
    m.idx, m.X, m.A, m.xnames = {b: v[1] for b, v in blocks.items()}, X, A, names
    Z = m.design({b: v[rows] for b, v in m.idx.items()}, X[rows])
    m.fit(Z, None if A is None else A[rows], df.units_sold.values[rows], df.opening_stock.values[rows], "sp",
          cfg["maxiter"])
    return m


def predict_nb(m, df, rows, theta=None):
    Z = m.design({b: v[rows] for b, v in m.idx.items()}, m.X[rows])
    eta = m.eta(Z, None if m.A is None else m.A[rows], theta)
    return expected_sales(eta, df.opening_stock.values[rows].astype(float), m.r)


def delta_model(m, days_seen, dt, variant):
    delta = m.block("day").copy()
    A = np.column_stack([np.ones(len(dt)), dt.temperature_forecast_c / 10, dt.rain_forecast_mm / 5,
                         np.eye(7)[dt.dow.values][:, 1:]]
                        + ([dt.season_sin, dt.season_cos] if variant == "season" else []))
    coef = np.linalg.lstsq(A[days_seen], delta[days_seen], rcond=None)[0]
    unseen = np.setdiff1d(np.arange(len(dt)), days_seen)
    delta[unseen] = A[unseen] @ coef
    th = m.theta.copy()
    th[m.off["day"]: m.off["day"] + m.n["day"]] = delta
    return th, coef, delta


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def report(m, enames, n_prod):
    beta = dict(zip(m.xnames, m.block("x")))
    for k in ("disc_prod", "disc_loc", "disc_dow", "season_prod", "season_loc"):
        print(f"  {k}: {np.round([v for n, v in beta.items() if n.startswith(k)], 3).tolist()}")
    if m.A is not None:
        g = m.gamma.reshape(len(enames), n_prod)
        print("  additive event visitors per sqrt(1k attendance), by product:")
        for k, name in enumerate(enames):
            print(f"    {name:18s} {np.round(g[k], 2).tolist()}")
    else:
        b = np.array([v for n, v in beta.items() if n.startswith("events")]).reshape(len(enames), n_prod)
        print("  log event effect per 1k attendance at <= 0.5 km, by type:",
              {enames[k]: np.round(b[k], 3).tolist() for k in range(0, len(enames), 5)})


def run(cfg, smoke):
    t0 = time.time()
    t = load()
    df, E, Es, enames = build(t)
    if cfg["shops"]:
        keep = df.shop < cfg["shops"]
        df, E, Es = df[keep].reset_index(drop=True), E[keep.values], Es[keep.values]
    n_prod = df["prod"].max() + 1
    dt = df.groupby("day")[["temperature_forecast_c", "rain_forecast_mm", "dow", "season_sin", "season_cos"]].first().sort_index()
    y = df.units_sold.values
    train = np.flatnonzero(df.split.values == "train")

    rng = np.random.default_rng(0)
    win = df.iloc[train]
    blocks = win[win.date >= GAP_START][["shop", "day"]].drop_duplicates().values
    pick = blocks[rng.choice(len(blocks), min(cfg["gap_blocks"], len(blocks)), replace=False)]
    key = df.shop.values * 10000 + df.day.values
    g_hold = train[np.isin(key[train], pick[:, 0] * 10000 + pick[:, 1])]
    g_fit = np.setdiff1d(train, g_hold)
    assert len(g_hold) == 3 * len(pick)
    cut = df.day.values[train].max() - cfg["future_days"] + 1
    f_hold, f_fit = train[df.day.values[train] >= cut], train[df.day.values[train] < cut]

    scores, folds = {}, {"gap": {"row": g_hold, "y": y[g_hold]}, "future": {"row": f_hold, "y": y[f_hold]}}
    for name, arm in cfg["arms"].items():
        mg = fit_nb(cfg, df, E, Es, arm, g_fit)
        p = predict_nb(mg, df, g_hold)
        scores[name] = {"gap": rmse(p, y[g_hold])}
        folds["gap"][name] = p
        mf = fit_nb(cfg, df, E, Es, arm, f_fit)
        for var in VARIANTS:
            th, coef, _ = delta_model(mf, np.unique(df.day.values[f_fit]), dt, var)
            p = predict_nb(mf, df, f_hold, th)
            scores[name][var] = rmse(p, y[f_hold])
            folds["future"][f"{name}_{var}"] = p
        print(f"{name:18s} gap {scores[name]['gap']:.4f} | future " +
              " ".join(f"{v} {scores[name][v]:.4f}" for v in VARIANTS) + f" | {time.time() - t0:.0f}s")

    inc_g, inc_f = cfg["incumbent"]["gap"], tuple(cfg["incumbent"]["future"])
    g_arm = min(scores, key=lambda a: scores[a]["gap"])
    if scores[inc_g]["gap"] - scores[g_arm]["gap"] < cfg["margin"]:
        g_arm = inc_g
    fut_scores = {(a, v): s[v] for a, s in scores.items() for v in VARIANTS}
    f_arm, var = min(fut_scores, key=fut_scores.get)
    if fut_scores[inc_f] - fut_scores[(f_arm, var)] < cfg["margin"]:
        f_arm, var = inc_f
    mixed = np.sqrt(0.25 * scores[g_arm]["gap"] ** 2 + 0.75 * fut_scores[(f_arm, var)] ** 2)
    print(f"chosen gap: {g_arm} ({scores[g_arm]['gap']:.4f}) | future: {f_arm} / {var} "
          f"({fut_scores[(f_arm, var)]:.4f}) | mixed rmse {mixed:.4f}")

    test = np.flatnonzero(df.split.values != "train")
    fut = df.split.values[test] == "future"
    tg, tf = test[~fut], test[fut]
    seen = np.unique(df.day.values[train])
    fitted, thetas = {}, {}
    for a in dict.fromkeys([g_arm, f_arm]):
        fitted[a] = m = fit_nb(cfg, df, E, Es, cfg["arms"][a], train)
        thetas[a] = {v: delta_model(m, seen, dt, v) for v in VARIANTS}
        print(f"full fit {a}:")
        report(m, enames, n_prod)
    m = fitted[f_arm]
    _, coef_full, delta_full = thetas[f_arm][var]
    print(f"  r {m.r:.2f}; day-effect model coef {np.round(coef_full, 3).tolist()}")
    preds = {"gap": predict_nb(fitted[g_arm], df, tg), "future": predict_nb(m, df, tf, thetas[f_arm][var][0])}
    final = np.empty(len(test))
    final[~fut], final[fut] = preds["gap"], preds["future"]
    print(f"  mean prediction: gap {preds['gap'].mean():.2f}, future {preds['future'].mean():.2f}")

    if smoke:
        print(f"smoke ok ({time.time() - t0:.0f}s)")
        return
    sub = t["sample_submission"][["id"]].merge(pd.DataFrame({"id": df.id.values[test], "units_sold": final}),
                                               on="id", how="left")
    s_stock = sub.merge(df[["id", "opening_stock"]], on="id").opening_stock.values
    assert len(sub) == len(t["test"]) and sub.units_sold.notna().all() and np.isfinite(sub.units_sold).all()
    assert (sub.units_sold >= 0).all() and (sub.units_sold <= s_stock + 1e-9).all()
    sub.to_csv(OUT / "submission.csv", index=False)
    with zipfile.ZipFile(OUT / "extra_v3.1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("test_pred.csv", pd.DataFrame({"id": df.id.values[test], "split": df.split.values[test],
                                                  "pred": final}).to_csv(index=False))
        for k, v in folds.items():
            z.writestr(f"fold_{k}.csv", pd.DataFrame(v).to_csv(index=False))
        z.writestr("coef.json", json.dumps({"scores": scores, "gap_arm": g_arm, "future_arm": f_arm, "var": var, "r": m.r,
                                            "delta_coef": coef_full.tolist(), "delta": delta_full.tolist(),
                                            "beta": dict(zip(m.xnames, m.block("x").tolist())),
                                            "gamma": None if m.A is None else m.gamma.tolist()}, indent=1))
    print(f"{time.time() - t0:.0f}s")
    print(f"SUBMIT: {OUT / 'submission.csv'}")


def main():
    run({**CFG, **SMOKE}, smoke=True)
    run(CFG, smoke=False)


if __name__ == "__main__":
    main()
