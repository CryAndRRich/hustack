import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

from preprocess import (CAT_COLS, EXT_CAT_COLS, ID, NAME, TARGET, build_features, ext_features, extra_features,
                        fill_rating_by_name, get_out_dir, load_data, load_external, price_features, restore_price,
                        teacher_wage_features)
from text_embed import embed_frames, tfidf_frames

VERSION = "V3"
BASE = "ref"
ARMS = {
    "ref": dict(feats="c", pop="all"),
    "ref_single": dict(feats="c", pop="single"),
    "f3": dict(feats="f3", pop="all"),
    "f3_single": dict(feats="f3", pop="single"),
    "f3_w": dict(feats="f3", pop="weighted"),
}
REPRODUCE = {"ref": 59.63}
REPRODUCE_TOL = 0.6
N_FOLDS, N_REPEATS, CV_SEED = 5, 3, 20261010
FINAL_SEEDS = [0, 1, 2, 3, 4]
EMB_ALPHAS = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
TFIDF_ALPHAS = [0.3, 1.0, 3.0, 10.0]
CAT_PARAMS = dict(
    iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, loss_function="RMSE", verbose=0,
    allow_writing_files=False, thread_count=-1,
)


def mse(a, b):
    return float(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2))


def make_folds(groups, seed):
    uniq = pd.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % N_FOLDS))
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(N_FOLDS)]


def fit_predict(Xa, ya, Xb, seed, weight=None):
    cats = [c for c in CAT_COLS + EXT_CAT_COLS if c in Xa.columns]
    model = CatBoostRegressor(**CAT_PARAMS, random_seed=seed, cat_features=cats)
    model.fit(Xa, ya, sample_weight=weight)
    return model.predict(Xb), model


def name_lookup(train, p, test):
    by_name = pd.Series(p).groupby(train[NAME].to_numpy()).agg(["min", "max"])
    assert np.allclose(by_name["min"], by_name["max"]), "a tutor name has two different restored prices"
    return test[NAME].map(by_name["min"]).to_numpy()


def person_key(names):
    return (pd.Series(names).str.replace(r"^Репетитор по \S+\s+", "", regex=True)
            .str.replace(r"\s*-\s*фотография\s*$", "", regex=True).str.strip().to_numpy())


def fold_ids(groups, seed):
    f = np.zeros(len(groups), dtype=int)
    for k, (_, b) in enumerate(make_folds(groups, seed)):
        f[b] = k
    return f


def cross_subject_copy(train_person, train_names, p, query_person, query_names, usable):
    df = pd.DataFrame({"person": train_person[usable], "name": train_names[usable], "p": p[usable]})
    out = np.full(len(query_person), np.nan)
    by = df.groupby("person")
    for i, (per, nm) in enumerate(zip(query_person, query_names)):
        if per in by.groups:
            g = by.get_group(per)
            g = g[g["name"] != nm]
            if len(g):
                out[i] = g["p"].mean()
    return out


def ridge_oof(E, y, groups, alpha, seed):
    pred = np.zeros(len(y))
    for a, b in make_folds(groups, seed):
        pred[b] = Ridge(alpha=alpha).fit(E[a], y[a]).predict(E[b])
    return pred


def emb_stack(Ea, ya, ga, Eb, seed, alphas=EMB_ALPHAS):
    inner = {al: ridge_oof(Ea, ya, ga, al, seed) for al in alphas}
    best = min(inner, key=lambda al: mse(ya, inner[al]))
    return inner[best], Ridge(alpha=best).fit(Ea, ya).predict(Eb), best


def main():
    t0 = time.time()
    train, test = load_data()
    y_raw = train[TARGET].to_numpy(dtype=float)
    p, low = restore_price(y_raw)
    print(f"{VERSION}: train {len(train)}, test {len(test)}; prices <1 rub/min restored x100: {low.sum()} ({low.mean():.4f})")

    n_nan = int(train["tutor_rating"].isna().sum() + test["tutor_rating"].isna().sum())
    fill_rating_by_name(train, test)
    n_left = int(train["tutor_rating"].isna().sum() + test["tutor_rating"].isna().sum())
    print(f"rating NaN filled from same-name rows: {n_nan} -> {n_left}")

    counts = pd.concat([train[NAME], test[NAME]]).value_counts()
    single = (train[NAME].map(counts) == 1).to_numpy()
    seen = test[NAME].isin(set(train[NAME])).to_numpy()
    test_single = (test[NAME].map(counts) == 1).to_numpy()
    weight = 1.0 / train[NAME].map(train[NAME].value_counts()).to_numpy()
    print(f"train singletons {single.sum()} (maths {np.mean(train['предмет'][single] == 'matematika'):.3f}), "
          f"repeated {(~single).sum()}; test lookup {seen.sum()}, model rows {(~seen).sum()} "
          f"(singletons {test_single[~seen].mean():.3f})")

    ext = load_external()
    e_tr, e_te = ext_features(train, ext), ext_features(test, ext)
    v2_tr, v2_te = pd.concat([build_features(train), e_tr], axis=1), pd.concat([build_features(test), e_te], axis=1)
    c_tr, c_te = extra_features(train, counts), extra_features(test, counts)
    b_tr, b_te = price_features(train), price_features(test)
    w_tr, w_te = teacher_wage_features(e_tr, ext), teacher_wage_features(e_te, ext)
    feats = {
        "c": (pd.concat([v2_tr, c_tr], axis=1), pd.concat([v2_te, c_te], axis=1)),
        "f3": (pd.concat([v2_tr, c_tr, b_tr, w_tr], axis=1), pd.concat([v2_te, c_te, b_te, w_te], axis=1)),
    }
    for k, (X, Xt) in feats.items():
        assert list(X.columns) == list(Xt.columns) and not X.columns.duplicated().any(), k
        print(f"feature set {k}: {X.shape[1]} columns + stacks")
    print(f"teacher wage known for {w_tr['loc_teacher_wage'].notna().mean():.3f} of train rows; "
          f"price-in-text rows {b_tr['txtp_ppm_med'].notna().sum()} / {b_te['txtp_ppm_med'].notna().sum()}")

    E_tr, E_te = embed_frames(train, test)
    T_tr, T_te = tfidf_frames(train, test)
    print(f"embeddings {E_tr.shape[1]} dims, tf-idf {T_tr.shape[1]} columns  [{time.time() - t0:.0f}s]")
    groups = train[NAME].to_numpy()

    oof = {a: np.full((N_REPEATS, len(train)), np.nan) for a in ARMS}
    stack_oof = {k: np.zeros((N_REPEATS, len(train))) for k in ("emb", "tfidf")}
    for r in range(N_REPEATS):
        for a_idx, b_idx in make_folds(groups, CV_SEED + r):
            a_s = a_idx[single[a_idx]]
            st = {}
            for pop, rows in (("all", a_idx), ("single", a_s)):
                st["emb", pop] = emb_stack(E_tr[rows], p[rows], groups[rows], E_tr[b_idx], seed=CV_SEED + 100 + r)
                st["tfidf", pop] = emb_stack(T_tr[rows], p[rows], groups[rows], T_tr[b_idx], seed=CV_SEED + 200 + r,
                                             alphas=TFIDF_ALPHAS)
            stack_oof["emb"][r, b_idx], stack_oof["tfidf"][r, b_idx] = st["emb", "all"][1], st["tfidf", "all"][1]
            for arm, cfg in ARMS.items():
                pop = "single" if cfg["pop"] == "single" else "all"
                rows = a_s if pop == "single" else a_idx
                X = feats[cfg["feats"]][0]
                stacks = ["emb"] + (["tfidf"] if cfg["feats"] == "f3" else [])
                Xa = X.iloc[rows].assign(**{f"{k}_ridge": st[k, pop][0] for k in stacks})
                Xb = X.iloc[b_idx].assign(**{f"{k}_ridge": st[k, pop][1] for k in stacks})
                w = weight[rows] if cfg["pop"] == "weighted" else None
                oof[arm][r, b_idx], _ = fit_predict(Xa, p[rows], Xb, seed=r, weight=w)
        print(f"repeat {r} (singleton MSE): " + "  ".join(f"{a}={mse(p[single], oof[a][r][single]):.2f}" for a in ARMS)
              + f"  [{time.time() - t0:.0f}s]")

    print("\nOOF MSE vs restored price, grouped by tutor; lookup rows not included")
    print(f"  constant (singleton mean)     singletons {np.var(p[single]):.2f}")
    for k in stack_oof:
        print(f"  ridge on {k:<6} alone          singletons {np.mean([mse(p[single], stack_oof[k][r][single]) for r in range(N_REPEATS)]):.2f}")
    per = {a: np.array([mse(p[single], oof[a][r][single]) for r in range(N_REPEATS)]) for a in ARMS}
    per_all = {a: np.array([mse(p, oof[a][r]) for r in range(N_REPEATS)]) for a in ARMS if ARMS[a]["pop"] != "single"}
    for a, ref in REPRODUCE.items():
        ok = abs(per_all[a].mean() - ref) <= REPRODUCE_TOL
        print(f"  plumbing: {a} all-row OOF {per_all[a].mean():.2f} vs V2.2 logged {ref:.2f} (tol {REPRODUCE_TOL}) -> "
              f"{'REPRODUCED' if ok else 'MISMATCH'}")
    frac = (~seen).mean()
    err_b = (p - np.nanmean(oof[BASE], axis=0)) ** 2
    print(f"  {'arm':<11} {'singleton':>9} {'sd':>5}  {'- ' + BASE + ' per repeat':<24} {'SE':>5}  better/worse  all-rows  -> expected LB")
    for a in ARMS:
        err = (p - np.nanmean(oof[a], axis=0)) ** 2
        dd = (err - err_b)[single]
        d = np.round(per[a] - per[BASE], 2).tolist()
        all_txt = f"{per_all[a].mean():8.2f}" if a in per_all else "       -"
        print(f"  {a:<11} {per[a].mean():9.2f} {per[a].std():5.2f}  {str(d):<24} {dd.std() / np.sqrt(len(dd)):5.2f}  "
              f"{(dd < 0).sum():>5}/{(dd > 0).sum():<5} {all_txt}  -> {frac * per[a].mean():.1f}")
    has_price = b_tr["txtp_ppm_med"].notna().to_numpy() & single
    msk = (e_tr["loc_msk"].to_numpy() == 1) & single
    for m, lab in [(has_price, "price in text"), (msk, "Moscow"), (single & ~msk, "not Moscow")]:
        print(f"  singleton rows {lab:<14} n={m.sum():>4}  " + "  ".join(
            f"{a} {((p - np.nanmean(oof[a], axis=0)) ** 2)[m].mean():.2f}" for a in ARMS))

    best = min(ARMS, key=lambda a: per[a].mean())
    ship = best if per[best].mean() < per[BASE].mean() else BASE
    print(f"  SHIP ARM (lowest singleton OOF): {ship}")

    person_tr, person_te = person_key(train[NAME]), person_key(test[NAME])
    d_copy, d_model = [], []
    for r in range(N_REPEATS):
        f = fold_ids(groups, CV_SEED + r)
        for k in range(N_FOLDS):
            q = np.where((f == k) & single)[0]
            cp = cross_subject_copy(person_tr, groups, p, person_tr[q], groups[q], f != k)
            m = ~np.isnan(cp)
            d_copy += list((p[q][m] - cp[m]) ** 2)
            d_model += list((p[q][m] - oof[ship][r, q][m]) ** 2)
    use_d = len(d_copy) > 0 and np.mean(d_copy) < np.mean(d_model)
    print(f"\nblock D (cross-subject copy) on singleton OOF: {len(d_copy)} row-repeats, copy MSE "
          f"{np.mean(d_copy) if d_copy else float('nan'):.2f} vs model {np.mean(d_model) if d_model else float('nan'):.2f} "
          f"-> {'APPLY' if use_d else 'skip'}")

    test_preds, imp = {}, None
    for a in dict.fromkeys([BASE, ship]):
        cfg = ARMS[a]
        rows = np.where(single)[0] if cfg["pop"] == "single" else np.arange(len(train))
        X, Xt = feats[cfg["feats"]]
        stacks = ["emb"] + (["tfidf"] if cfg["feats"] == "f3" else [])
        add_a, add_b = {}, {}
        for k, (M, Mt, al_grid, sd) in {"emb": (E_tr, E_te, EMB_ALPHAS, 999), "tfidf": (T_tr, T_te, TFIDF_ALPHAS, 998)}.items():
            if k in stacks:
                fa, fb, al = emb_stack(M[rows], p[rows], groups[rows], Mt, seed=CV_SEED + sd, alphas=al_grid)
                add_a[f"{k}_ridge"], add_b[f"{k}_ridge"] = fa, fb
                print(f"  {a}: full-data {k} ridge alpha {al}")
        Xa, Xb = X.iloc[rows].assign(**add_a), Xt.assign(**add_b)
        w = weight[rows] if cfg["pop"] == "weighted" else None
        test_preds[a] = np.zeros(len(test))
        for seed in FINAL_SEEDS:
            tp, model = fit_predict(Xa, p[rows], Xb, seed, weight=w)
            test_preds[a] += tp / len(FINAL_SEEDS)
            if a == ship:
                fi = pd.Series(model.get_feature_importance(), index=Xa.columns) / len(FINAL_SEEDS)
                imp = fi if imp is None else imp + fi
    print(f"top features ({ship}):", ", ".join(f"{k}={v:.1f}" for k, v in imp.sort_values(ascending=False).head(20).items()))
    for lab, cols in [("price (B)", list(b_tr.columns)), ("teacher wage", list(w_tr.columns)), ("extra (C)", list(c_tr.columns)),
                      ("stacks", ["emb_ridge", "tfidf_ridge"])]:
        print(f"  {lab} importance: {imp.reindex(cols).fillna(0).sum():.1f}")
    moved = np.abs(test_preds[ship] - test_preds[BASE])[~seen]
    print(f"test model rows, {ship} vs {BASE}: mean |diff| {moved.mean():.2f}, moved > 2 rub/min: {(moved > 2).sum()}, "
          f"mean {test_preds[ship][~seen].mean():.2f} vs {test_preds[BASE][~seen].mean():.2f} (train singleton mean {p[single].mean():.2f})")

    look = name_lookup(train, p, test)
    final = {a: np.where(seen, look, np.clip(test_preds[a], p.min(), p.max())) for a in test_preds}
    d_test = cross_subject_copy(person_tr, groups, p, person_te, test[NAME].to_numpy(), np.ones(len(train), bool))
    d_rows = ~seen & ~np.isnan(d_test)
    print(f"block D on test: {d_rows.sum()} rows" + (f"; mean |copy - model| {np.abs(d_test[d_rows] - final[ship][d_rows]).mean():.2f}" if d_rows.any() else ""))
    if use_d:
        for a in final:
            final[a] = np.where(d_rows, d_test, final[a])

    out = get_out_dir()
    pd.DataFrame({NAME: train[NAME], "p": p, "single": single, **{f"oof_{a}": np.nanmean(oof[a], axis=0) for a in ARMS},
                  **{f"{k}_ridge": stack_oof[k].mean(axis=0) for k in stack_oof}}).to_csv(out / f"oof_{VERSION.lower()}.csv", index=False)
    for fname, a in [("submission.csv", ship), (f"submission_{BASE}.csv", BASE)]:
        sub = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: final[a]})
        assert len(sub) == len(test) and sub[TARGET].notna().all()
        sub.to_csv(out / fname, index=False)
    print(f"submission.csv = {ship}{' + D' if use_d else ''}: mean {final[ship].mean():.2f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
