import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor

from preprocess import PHYS_INIT, PHYS_TERMS, UN_INIT, load, phys_log, prepare, un_log

N_FOLDS = 5
SEEDS = [42, 7, 2026]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
WATCH = ["Singapore", "Panama", "Lesotho", "Aruba", "Chad", "Nepal", "Angola", "Curacao", "Liechtenstein"]


def rmsle_log(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def country_folds(countries, seed):
    uniq = np.array(sorted(countries.unique()))
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {uniq[i]: k % N_FOLDS for k, i in enumerate(order)}
    return countries.map(fold_of).to_numpy()


def learner(X, y, seed):
    a = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40,
                                      l2_regularization=1.0, random_state=seed).fit(X, y).predict
    b = CatBoostRegressor(depth=4, learning_rate=0.05, iterations=800, l2_leaf_reg=3.0, verbose=0,
                          thread_count=2, random_seed=seed).fit(X, y).predict
    return lambda Z: 0.5 * a(Z) + 0.5 * b(Z)


def fit_phys(T, y, route):
    return {r: least_squares(lambda k: phys_log(T[route == r], k) - y[route == r], PHYS_INIT,
                             bounds=(0.0, 1.0)).x for r in (0, 1)}


def offset(df, T, route, ks):
    out = np.zeros(len(route))
    for r in (0, 1):
        out[route == r] = phys_log(T[route == r], ks[r])
    out[route == 3] = un_log(df[route == 3], UN_INIT)
    return out


def percap_log(df_fit, P_fit, df_ev, P_ev, seed):
    y_pc = np.log(df_fit["co2"].to_numpy() / df_fit["population"].to_numpy() * 1e6)
    rate = np.exp(learner(P_fit, y_pc, seed)(P_ev))
    return np.log1p(rate * df_ev["population"].to_numpy() / 1e6)


def smooth(p, countries, mask, lam):
    cm = pd.Series(np.where(mask, p, np.nan)).groupby(countries).transform("mean").to_numpy()
    return np.where(mask, lam * p + (1.0 - lam) * cm, p)


def own_country_fill(p_log, df, has_e):
    out = p_log.copy()
    pop = df["population"].to_numpy()
    years = df["year"].to_numpy()
    pc = np.log(np.clip(np.expm1(p_log), 1e-6, None) / pop)
    for idx in df.groupby("country").indices.values():
        e_idx = idx[has_e[idx]]
        if len(e_idx) == 0 or len(e_idx) == len(idx):
            continue
        for i in idx[~has_e[idx]]:
            near = e_idx[np.argsort(np.abs(years[e_idx] - years[i]))[:3]]
            out[i] = np.log1p(np.exp(pc[near].mean()) * pop[i])
    return out


def assemble(p_res, p_c, lam, df, off, has_e):
    p_res = smooth(p_res, df["country"].to_numpy(), has_e, lam)
    return own_country_fill(np.where(has_e, off + p_res, p_c), df, has_e)


def residual(X_fit, r_fit, X_ev, route_ev, seed):
    p = learner(X_fit, r_fit, seed)(X_ev)
    return np.where(route_ev == 3, 0.0, p)


def run_cv(tr, X, P, T, route, y, seed):
    folds = country_folds(tr["country"], seed)
    has_e = route != 2
    p_res, p_c, off = np.zeros(len(tr)), np.zeros(len(tr)), np.zeros(len(tr))
    for k in range(N_FOLDS):
        fit, ev = (folds != k) & has_e, (folds == k) & has_e
        ks = fit_phys(T[fit], y[fit], route[fit])
        off[ev] = offset(tr[ev], T[ev], route[ev], ks)
        p_res[ev] = residual(X[fit], y[fit] - offset(tr[fit], T[fit], route[fit], ks), X[ev], route[ev], seed)
        p_c[folds == k] = percap_log(tr[folds != k], P[folds != k], tr[folds == k], P[folds == k], seed)
    return p_res, p_c, off


def main():
    t0 = time.time()
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    tr, te = load()
    X, X_te, P, P_te, _, _, r_tr, r_te, T, T_te = prepare(tr, te)
    y = np.log1p(tr["co2"].to_numpy())
    route, route_te = r_tr.to_numpy(), r_te.to_numpy()
    has_e, has_e_te = route != 2, route_te != 2
    print(f"train {len(tr)} rows, test {len(te)} rows; {X.shape[1]} residual features; routes train "
          f"{np.bincount(route, minlength=4).tolist()} test {np.bincount(route_te, minlength=4).tolist()}")

    runs = [run_cv(tr, X, P, T, route, y, s) for s in SEEDS]
    lam = min(LAMBDAS, key=lambda v: np.mean([rmsle_log(y, assemble(a, b, v, tr, o, has_e)) for a, b, o in runs]))
    ps = [assemble(a, b, lam, tr, o, has_e) for a, b, o in runs]
    sc = [rmsle_log(y, p) for p in ps]
    oof = np.mean(ps, axis=0)
    per_route = [rmsle_log(y[route == r], oof[route == r]) for r in range(4)]
    w_te = np.bincount(route_te, minlength=4) / len(te)
    expected = float(np.sqrt(sum(w_te[r] * per_route[r] ** 2 for r in range(4))))
    print(f"grouped CV {np.mean(sc):.5f} (seeds {' '.join(f'{s:.4f}' for s in sc)}), lambda {lam}; "
          + " ".join(f"route{r} {per_route[r]:.4f}" for r in range(4)) + f"; at test's route mix {expected:.5f}")
    g = pd.DataFrame({"c": tr["country"], "d": oof - y}).groupby("c")["d"]
    bias, rms = g.mean(), g.apply(lambda s: float(np.sqrt((s ** 2).mean())))
    print("watched, mean log error: " + ", ".join(f"{c} {bias[c]:+.2f}" for c in WATCH if c in bias.index))
    print("worst countries: " + ", ".join(f"{c} {v:.3f}" for c, v in rms.nlargest(8).items()))

    ks = fit_phys(T[has_e], y[has_e], route[has_e])
    for r in (0, 1):
        print(f"coefficients route {r}: " + ", ".join(f"{n} {v:.4f}" for n, v in zip(PHYS_TERMS, ks[r])))
    off = offset(tr[has_e], T[has_e], route[has_e], ks)
    off_te = np.zeros(len(te))
    off_te[has_e_te] = offset(te[has_e_te], T_te[has_e_te], route_te[has_e_te], ks)
    p_res = np.zeros(len(te))
    p_res[has_e_te] = np.mean([residual(X[has_e], y[has_e] - off, X_te[has_e_te], route_te[has_e_te], s)
                               for s in SEEDS], axis=0)
    p_c = np.mean([percap_log(tr, P, te, P_te, s) for s in SEEDS], axis=0)
    co2 = np.clip(np.expm1(assemble(p_res, p_c, lam, te, off_te, has_e_te)), 0.0, None)
    sub = pd.DataFrame({"id": te["id"], "co2": co2})
    assert len(sub) == 1219 and sub["co2"].notna().all() and (sub["co2"] >= 0).all()
    sub.to_csv(out_dir / "submission.csv", index=False)
    with zipfile.ZipFile(out_dir / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.json", json.dumps({
            "cv": float(np.mean(sc)), "cv_seeds": sc, "lambda": lam, "route_cv": per_route,
            "expected_test": expected,
            "coefficients": {str(r): dict(zip(PHYS_TERMS, map(float, ks[r]))) for r in ks}}, indent=1))
        z.writestr("oof.csv", pd.DataFrame({"id": tr["id"], "country": tr["country"], "route": route,
                                            "y_log": y, "oof": oof}).to_csv(index=False))
        z.writestr("submission.csv", sub.to_csv(index=False))
    print(f"SUBMIT {out_dir / 'submission.csv'}  (CV {np.mean(sc):.5f})  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
