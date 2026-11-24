import json
import shutil
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from preprocess import LINES, OUT, TARGETS, WORK, build_features, load, requester_te, say

N_FOLDS = 5
PARAMS = {"objective": "binary", "metric": "auc", "learning_rate": 0.03, "num_leaves": 255, "min_data_in_leaf": 200,
          "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1}
SEEDS = [0, 1]
ROUNDS = 10000
SMOKE_ROUNDS = 20
EARLY_STOP = 150
SHORT = ["Y6", "Y12", "Y24"]
V4_Y48_OOF = 0.90452


def auc(y, s):
    return float(roc_auc_score(y, s))


def feature_aucs(X, y):
    rows = {}
    for c in X.columns:
        v = np.nan_to_num(X[c].values, nan=-1e18)
        rows[c] = {t: auc(y[t], v) for t in TARGETS}
    df = pd.DataFrame(rows).T
    df["mean"] = df[TARGETS].mean(axis=1)
    return df.reindex((df["mean"] - 0.5).abs().sort_values(ascending=False).index)


def fit_cv(X, y, folds, Xte, label, rounds, use):
    oof, iters, imp = np.zeros(len(X)), [], np.zeros(X.shape[1])
    pred = np.zeros(len(Xte))
    for k in range(N_FOLDS):
        trn, val, out = use & (folds != k), use & (folds == k), folds == k
        dtr = lgb.Dataset(X[trn], y[trn])
        dva = lgb.Dataset(X[val], y[val], reference=dtr)
        for seed in SEEDS:
            m = lgb.train({**PARAMS, "seed": seed}, dtr, rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            oof[out] += m.predict(X[out], num_iteration=m.best_iteration) / len(SEEDS)
            pred += m.predict(Xte, num_iteration=m.best_iteration) / (N_FOLDS * len(SEEDS))
            iters.append(int(m.best_iteration))
            imp += m.feature_importance("gain")
        say(f"{label} fold {k}: auc {auc(y[val], oof[val]):.5f} iters {iters[-len(SEEDS):]}")
    say(f"{label} OOF auc {auc(y[use], oof[use]):.5f}")
    return oof, pred, iters, imp


def segments(X, y, final_oof):
    seg = {"x_j=0": X.x_j.values == 0, "x_j>0": X.x_j.values > 0, "y_i=0": X.y_i.values == 0,
           "j own record found": ~np.isnan(X.jt_first.values), "j own record missing": np.isnan(X.jt_first.values)}
    for lo, hi in [(0, 1e2), (1e2, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, np.inf)]:
        seg[f"y_j in [{lo:.0e}, {hi:.0e})"] = (X.y_j.values >= lo) & (X.y_j.values < hi)
    rows = {}
    for name, m in seg.items():
        r = {"share": m.mean()}
        for t in ["Y6", "Y48"]:
            rk = rankdata(final_oof[t]) / len(m)
            r[f"{t}_rate"] = y[t][m].mean()
            r[f"{t}_mean_rank"] = rk[m].mean()
            r[f"{t}_auc"] = auc(y[t][m], final_oof[t][m]) if 0 < y[t][m].sum() < m.sum() else np.nan
        rows[name] = r
    return pd.DataFrame(rows).T


def pack(work, out_zip):
    (work / "log.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(work.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(work))


def run(smoke):
    tag = "SMOKE " if smoke else ""
    work = WORK / ("smoke" if smoke else "full")
    out = WORK / "smoke_out" if smoke else OUT
    for d in [work, out]:
        if d != OUT and d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    rounds = SMOKE_ROUNDS if smoke else ROUNDS
    say(f"{tag}run start")

    allf, is_tr, labels = load(smoke)
    feats, dec_comp, paths = build_features(allf)
    groups = dec_comp[is_tr]
    folds_tr = np.full(is_tr.sum(), -1)
    for k, (_, val) in enumerate(GroupKFold(N_FOLDS).split(groups, groups=groups)):
        folds_tr[val] = k
    say("fold sizes", np.bincount(folds_tr).tolist(), "largest group share",
        round(float(pd.Series(groups).value_counts().iloc[0] / len(groups)), 4))
    fold = np.r_[folds_tr, np.full((~is_tr).sum(), -1)]
    feats = pd.concat([feats, pd.DataFrame(requester_te(allf, is_tr, labels, fold, paths["req"]))], axis=1)
    say("features", feats.shape[1])

    X, Xte = feats[is_tr].reset_index(drop=True), feats[~is_tr].reset_index(drop=True)
    y = {t: labels[t][is_tr].astype(int) for t in TARGETS}
    fa = feature_aucs(X, y)
    fa.to_csv(work / "feature_auc.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:.4f}".format):
        say("single-feature AUC (train; TE is out-of-fold)\n" + fa.to_string())
    oof, pred, cv, imp = {}, {}, {}, {}
    everyone = np.ones(len(X), bool)
    for t in ["Y48"] + SHORT:
        oof[t], pred[t], iters, imp[t] = fit_cv(X, y[t], folds_tr, Xte, f"{tag}{t}", rounds, everyone)
        cv[t] = {"oof_auc": auc(y[t], oof[t]), "iters": iters}
    if not smoke:
        say(f"capacity check: Y48 OOF {cv['Y48']['oof_auc']:.5f} vs V4 {V4_Y48_OOF:.5f}")
    pos = y["Y48"] == 1
    cond, cond_pred = {}, {}
    for t in SHORT:
        cond[t], cond_pred[t], iters, imp[f"cond_{t}"] = fit_cv(X, y[t], folds_tr, Xte, f"{tag}{t} | Y48=1", rounds,
                                                                 pos)
        cv[f"cond_{t}"] = {"auc_on_Y48_pos": auc(y[t][pos], cond[t][pos]),
                           "own_auc_on_Y48_pos": auc(y[t][pos], oof[t][pos]), "iters": iters}

    def arms(t, src, c):
        if t == "Y48":
            return {"own": src[t]}
        own, y48, prod = rankdata(src[t]), rankdata(src["Y48"]), rankdata(src["Y48"] * c[t])
        return {"own": src[t], "own+Y48": own + y48, "product": prod, "own+product": own + prod,
                "own+Y48+product": own + y48 + prod}

    choice, final, final_oof = {}, {}, {}
    for t in TARGETS:
        scores = {a: auc(y[t], v) for a, v in arms(t, oof, cond).items()}
        choice[t] = max(scores, key=scores.get)
        cv[f"arms_{t}"] = scores
        say(f"{tag}{t} arms: " + " | ".join(f"{a} {v:.5f}" for a, v in scores.items()) + f" -> {choice[t]}")
        final_oof[t] = arms(t, oof, cond)[choice[t]]
        final[t] = arms(t, pred, cond_pred)[choice[t]]
    cv["final_mean_auc"] = float(np.mean([auc(y[t], final_oof[t]) for t in TARGETS]))
    cv["own_mean_auc"] = float(np.mean([cv[t]["oof_auc"] for t in TARGETS]))
    cv["choice"] = choice
    say(f"{tag}CV mean AUC own {cv['own_mean_auc']:.5f} final {cv['final_mean_auc']:.5f}")

    n_test = int((~is_tr).sum())
    sub = pd.DataFrame({"id": allf.id.values[~is_tr]})
    for t in TARGETS:
        sub[t] = rankdata(final[t]) / n_test
    sub = sub.sort_values("id").reset_index(drop=True)
    assert len(sub) == n_test and sub.id.is_unique and not sub.isna().any().any()
    assert list(sub.columns) == ["id"] + TARGETS
    assert smoke or (n_test == 362791 and sub.id.tolist() == list(range(1, 362792)))
    sub.to_csv(out / "submission.csv", index=False)
    pd.DataFrame(imp, index=X.columns).assign(total=lambda d: d.sum(axis=1)).sort_values(
        "total", ascending=False).to_csv(work / "importance.csv")
    np.savez_compressed(work / "oof.npz", fold=folds_tr, **{t: oof[t] for t in TARGETS},
                        **{f"cond_{t}": cond[t] for t in SHORT})
    (work / "cv.json").write_text(json.dumps(cv, indent=1))
    seg = segments(X, y, final_oof)
    seg.to_csv(work / "segments.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        say(f"{tag}final OOF by segment\n" + seg.to_string())
    say(f"{tag}SUBMIT: {out / 'submission.csv'}")
    pack(work, out / "results.zip")
    say(f"{tag}run done")


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        run(smoke=True)
        run(smoke=False)
    finally:
        full = WORK / "full"
        if full.exists():
            pack(full, OUT / "results.zip")
        else:
            (OUT / "log.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
