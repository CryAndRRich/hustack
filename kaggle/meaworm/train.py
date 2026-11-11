import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from preprocess import TARGET, load

VERSION = "V1"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
FULL = dict(repeats=3, folds=5, rounds=3000, stop=100)
SMOKE = dict(repeats=1, folds=2, rounds=30, stop=10)
GRID = np.exp(np.linspace(-3, 3, 25))


def params(k, seed):
    return dict(objective="multiclass", num_class=k, learning_rate=0.05, num_leaves=63, min_data_in_leaf=5,
                min_sum_hessian_in_leaf=1e-3, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                verbose=-1, num_threads=os.cpu_count(), seed=seed)


def macro_f1(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    f1 = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)
    return f1.mean(), f1, cm


def fit_weights(P, y, k, sweeps=3):
    w = np.ones(k)
    best = macro_f1(y, (P * w).argmax(1), k)[0]
    for _ in range(sweeps):
        for c in range(k):
            for g in GRID:
                w2 = w.copy()
                w2[c] = g
                s = macro_f1(y, (P * w2).argmax(1), k)[0]
                if s > best + 1e-9:
                    best, w = s, w2
    return w


def crossfit_weights(P, y, k):
    pred = np.empty(len(y), np.int64)
    for a, b in StratifiedKFold(2, shuffle=True, random_state=0).split(P, y):
        pred[b] = (P[b] * fit_weights(P[a], y[a], k)).argmax(1)
    return macro_f1(y, pred, k)[0]


def cv(X, y, Xte, k, cfg):
    oof = np.zeros((cfg["repeats"], len(X), k))
    test = np.zeros((len(Xte), k))
    iters, model = [], None
    for r in range(cfg["repeats"]):
        for f, (tr, va) in enumerate(StratifiedKFold(cfg["folds"], shuffle=True, random_state=r).split(X, y)):
            model = lgb.train(params(k, 100 * r + f), lgb.Dataset(X.iloc[tr], y[tr]), cfg["rounds"],
                              valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                              callbacks=[lgb.early_stopping(cfg["stop"], verbose=False)])
            oof[r, va] = model.predict(X.iloc[va], num_iteration=model.best_iteration)
            test += model.predict(Xte, num_iteration=model.best_iteration) / (cfg["repeats"] * cfg["folds"])
            iters.append(model.best_iteration)
        print(f"  repeat {r}: macro F1 {macro_f1(y, oof[r].argmax(1), k)[0]:.4f}, best iterations {iters[-cfg['folds']:]}",
              flush=True)
    return oof, test, model


def report(y, P, classes):
    k = len(classes)
    _, f1, cm = macro_f1(y, P.argmax(1), k)
    counts = np.bincount(y, minlength=k)
    table = pd.DataFrame({"n": counts, "f1": f1.round(4), "recall": (np.diag(cm) / counts).round(4)}, index=classes)
    print(table.sort_values("n", ascending=False).to_string(), flush=True)
    off = [(cm[i, j], classes[i], classes[j]) for i in range(k) for j in range(k) if i != j and cm[i, j] > 0]
    print("largest confusions (true -> predicted): " + "; ".join(f"{a} -> {b} {n}" for n, a, b in sorted(off)[::-1][:10]),
          flush=True)


def run(train, test, y, classes, feats, cfg, label):
    t0 = time.time()
    k = len(classes)
    X, Xte = train[feats], test[feats]
    oof, ptest, model = cv(X, y, Xte, k, cfg)
    P = oof.mean(0)
    per_rep = [macro_f1(y, o.argmax(1), k)[0] for o in oof]
    base = macro_f1(y, P.argmax(1), k)[0]
    honest = crossfit_weights(P, y, k)
    w = fit_weights(P, y, k)
    print(f"{label}: OOF macro F1 per repeat {np.round(per_rep, 4).tolist()}; repeat-averaged argmax {base:.4f}; "
          f"class weights cross-fit {honest:.4f} (in-sample {macro_f1(y, (P * w).argmax(1), k)[0]:.4f}); "
          f"weights {dict(zip(classes, np.round(w, 2).tolist()))}; {time.time() - t0:.0f}s", flush=True)
    report(y, P, classes)
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("top gain features: " + ", ".join(f"{c} {v / imp.sum():.3f}" for c, v in imp.head(15).items()), flush=True)
    use_w = honest > base
    return dict(oof=oof, test=ptest, w=w if use_w else np.ones(k), use_w=use_w, base=base, honest=honest)


def write(test, classes, res, train, feats):
    k = len(classes)
    pred = (res["test"] * res["w"]).argmax(1)
    sub = pd.DataFrame({"id": test["id"].values, TARGET: np.array(classes)[pred]})
    assert len(sub) == len(test) and sub["id"].is_unique and sub[TARGET].isin(classes).all()
    ext = os.path.join(OUT, "extra", VERSION)
    os.makedirs(ext, exist_ok=True)
    np.savez(os.path.join(ext, "lgb.npz"), oof=res["oof"], test=res["test"], w=res["w"], classes=np.array(classes))
    pd.DataFrame({"id": test["id"].values, TARGET: np.array(classes)[res["test"].argmax(1)]}).to_csv(
        os.path.join(ext, "submission_argmax.csv"), index=False)
    key = lambda d: pd.util.hash_pandas_object(d[feats].astype(str), index=False).values
    lab = pd.Series(train[TARGET].values, index=key(train)).groupby(level=0).agg(lambda s: s.mode().iloc[0])
    match = pd.Series(key(test)).map(lab)
    hit = match.notna().values
    print(f"test rows identical to a train row: {hit.sum()}, prediction agrees with the train label on "
          f"{np.mean(sub[TARGET].values[hit] == match.values[hit]):.4f}", flush=True)
    print("test prediction counts: " + str(sub[TARGET].value_counts().to_dict()), flush=True)
    path = os.path.join(OUT, "submission.csv")
    sub.to_csv(path, index=False)
    print(f"picked {'class-weighted' if res['use_w'] else 'argmax'} (cross-fit {res['honest']:.4f} vs argmax "
          f"{res['base']:.4f}); {k} classes", flush=True)
    return path


def main():
    t0 = time.time()
    print(f"VERSION {VERSION}", flush=True)
    train, test, y, classes, feats = load()
    rng = np.random.default_rng(0)
    counts = np.bincount(y)
    keep = np.concatenate([np.where(y == c)[0] if counts[c] <= 400 else rng.choice(np.where(y == c)[0], 300, replace=False)
                           for c in range(len(classes))])
    s = run(train.iloc[keep].reset_index(drop=True), test.iloc[:2000], y[keep], classes, feats, SMOKE, "smoke")
    write(test.iloc[:2000], classes, s, train.iloc[keep], feats)
    print(f"smoke ok ({time.time() - t0:.0f}s)", flush=True)
    res = run(train, test, y, classes, feats, FULL, "full")
    path = write(test, classes, res, train, feats)
    print(f"total {time.time() - t0:.0f}s", flush=True)
    print(f"SUBMIT: {path}", flush=True)


if __name__ == "__main__":
    main()
