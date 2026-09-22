import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from preprocess import (RAW_CATEGORICAL, IMPUTE_COLS, NA_FLAG_COLS, DIGIT_FEATURES, TARGET,
                        load_data, build_features, impute_features, encode_categoricals,
                        add_imputed_ratios)

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

TIME_BUDGET_MIN = 420
GMM_FIT_ROWS = 100_000
QT_SUBSAMPLE = 200_000

MIN_KEEP_AUC = 0.85


def header(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


def build_views(train_raw, test_raw):
    print("chained GBDT imputation (feature-only, leak-safe)...")
    tr = build_features(train_raw)
    te = build_features(test_raw)
    imp_tr, imp_te = impute_features(train_raw, test_raw)
    tr = pd.concat([tr.reset_index(drop=True), imp_tr], axis=1)
    te = pd.concat([te.reset_index(drop=True), imp_te], axis=1)
    tr = add_imputed_ratios(tr, imp_tr)
    te = add_imputed_ratios(te, imp_te)
    tr, te, _ = encode_categoricals(tr, te)

    for f in (tr, te):
        f["sleep_filled"] = f["sleep_hours"].fillna(tr["sleep_hours"].median())
        for c in RAW_CATEGORICAL:
            f[c + "_code"] = f[c].cat.codes.astype("int16")

    imp = [c + "_imp" for c in IMPUTE_COLS]
    na = ["na_" + c for c in NA_FLAG_COLS]
    dig = [f"digit{p}_{c}" for c, p in DIGIT_FEATURES]
    cat = [c + "_code" for c in RAW_CATEGORICAL]
    ratio = ["i_entertainment_ratio", "i_other_screen_time", "i_weekend_vs_daily",
             "i_social_ratio", "i_work_ratio"]
    screen = [c + "_imp" for c in ("daily_screen_time_hours", "social_media_hours",
                                   "gaming_hours", "work_study_hours", "weekend_screen_time")]
    behav = [c + "_imp" for c in ("notifications_per_day", "app_opens_per_day", "age")]

    views = {
        "core": imp + ["sleep_filled"],
        "core_cat": imp + ["sleep_filled"] + cat,
        "imp_ratio": imp + ["sleep_filled"] + ratio,
        "digits": dig + ["missing_count"],
        "na": na + ["missing_count"],
        "digits_na": dig + na + ["missing_count"],
        "screen": screen + ["sleep_filled"],
        "behav": behav + ["missing_count"],
        "all": imp + ["sleep_filled"] + na + dig + cat + ratio + ["missing_count"],
    }
    out = {}
    for name, cols in views.items():
        missing = [c for c in cols if c not in tr.columns]
        if missing:
            print(f"  view {name}: skipped, missing {missing}")
            continue
        a = tr[cols].to_numpy(np.float64)
        b = te[cols].to_numpy(np.float64)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            print(f"  view {name}: skipped, contains non-finite values")
            continue
        out[name] = (a, b, len(cols))
        print(f"  view {name:<10} {len(cols):>3} columns")
    return out


def _fit_transform(kind, Xtr, others):
    if kind == "std":
        t = StandardScaler().fit(Xtr)
    else:
        t = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                subsample=QT_SUBSAMPLE, random_state=SEED).fit(Xtr)
    return t.transform(Xtr), [t.transform(o) for o in others]


def _score_qda(Xtr, ytr, Xs, reg):
    m = QuadraticDiscriminantAnalysis(reg_param=reg, store_covariance=False).fit(Xtr, ytr)
    return [m.decision_function(x) for x in Xs]


def _score_lda(Xtr, ytr, Xs):
    m = LinearDiscriminantAnalysis().fit(Xtr, ytr)
    return [m.decision_function(x) for x in Xs]


def _score_gnb(Xtr, ytr, Xs, smooth):
    m = GaussianNB(var_smoothing=smooth).fit(Xtr, ytr)
    return [m.predict_log_proba(x)[:, 1] - m.predict_log_proba(x)[:, 0] for x in Xs]


def _score_gmm(Xtr, ytr, Xs, k, cov, rng):
    parts = []
    for cls in (0, 1):
        Xc = Xtr[ytr == cls]
        if len(Xc) > GMM_FIT_ROWS:
            Xc = Xc[rng.choice(len(Xc), GMM_FIT_ROWS, replace=False)]
        g = GaussianMixture(n_components=k, covariance_type=cov, reg_covar=1e-4,
                            max_iter=25, tol=1e-3, random_state=SEED).fit(Xc)
        parts.append([g.score_samples(x) for x in Xs])
    return [p1 - p0 for p0, p1 in zip(parts[0], parts[1])]


def _score_binned_nb(Xtr, ytr, Xs, bins, alpha=1.0):
    out = [np.zeros(len(x)) for x in Xs]
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    for j in range(Xtr.shape[1]):
        edges = np.unique(np.quantile(Xtr[:, j], qs))
        if len(edges) == 0:
            continue
        btr = np.digitize(Xtr[:, j], edges)
        nb = len(edges) + 1
        c1 = np.bincount(btr[ytr == 1], minlength=nb) + alpha
        c0 = np.bincount(btr[ytr == 0], minlength=nb) + alpha
        lo = np.log(c1 / c1.sum()) - np.log(c0 / c0.sum())
        for i, x in enumerate(Xs):
            out[i] += lo[np.digitize(x[:, j], edges)]
    return out

SPECS = (
    [("binned_nb", dict(bins=b), None, 0) for b in (16, 32, 64)]
    + [("gnb", dict(smooth=s), t, 0) for s in (1e-9, 1e-3) for t in ("std", "rg")]
    + [("lda", dict(), t, 1) for t in ("std", "rg")]
    + [("qda", dict(reg=r), t, 1) for r in (0.0, 1e-3, 1e-2, 1e-1) for t in ("std", "rg")]
    + [("gmm", dict(k=k, cov=c), t, 2)
       for k in (2, 3) for c in ("diag", "full") for t in ("std", "rg")]
)


def run_spec(family, kw, transform, Xtr_full, Xte_full, y, folds, rng):
    oof = np.zeros(len(y))

    def score(Xa, ya, targets):
        if transform is not None:
            Xa, targets = _fit_transform(transform, Xa, targets)
        if family == "qda":
            return _score_qda(Xa, ya, targets, kw["reg"])
        if family == "lda":
            return _score_lda(Xa, ya, targets)
        if family == "gnb":
            return _score_gnb(Xa, ya, targets, kw["smooth"])
        if family == "gmm":
            return _score_gmm(Xa, ya, targets, kw["k"], kw["cov"], rng)
        if family == "binned_nb":
            return _score_binned_nb(Xa, ya, targets, kw["bins"])
        raise ValueError(family)

    for trn, val in folds:
        oof[val] = score(Xtr_full[trn], y[trn], [Xtr_full[val]])[0]
    test = score(Xtr_full, y, [Xte_full])[0]
    if not (np.isfinite(oof).all() and np.isfinite(test).all()):
        raise ValueError("non-finite scores")
    return oof, test


def main():
    t0 = time.time()
    train_raw, test_raw, _ = load_data()
    y = train_raw[TARGET].to_numpy(np.int8)
    n, n_te = len(train_raw), len(test_raw)
    print(f"train {n:,} | test {n_te:,} | positive rate {y.mean():.6f}")

    header("FEATURE VIEWS")
    views = build_views(train_raw, test_raw)
    if not views:
        raise RuntimeError("no usable feature view was built")

    folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
                 .split(np.zeros(n), y))
    fold_id = np.empty(n, dtype=np.int8)
    for f, (_, va) in enumerate(folds):
        fold_id[va] = f
    np.save(OUTPUT_DIR / "fold_id.npy", fold_id)
    print(f"\nfold sizes {np.bincount(fold_id).tolist()} — fold_id.npy saved so consumers can "
          f"verify the split rather than assume it")

    plan = [(cost, vname, family, kw, tr)
            for family, kw, tr, cost in SPECS for vname in views]
    plan.sort(key=lambda r: r[0])
    print(f"\n{len(plan)} specs planned across {len(views)} views, budget {TIME_BUDGET_MIN} min")

    cap = len(plan)
    OOF = np.empty((n, cap), dtype=np.float32)
    TEST = np.empty((n_te, cap), dtype=np.float32)
    rows, kept, skipped, failed = [], 0, 0, 0
    rng = np.random.default_rng(SEED)

    header("BUILDING MEMBERS")
    for i, (cost, vname, family, kw, tr) in enumerate(plan, start=1):
        if (time.time() - t0) / 60 > TIME_BUDGET_MIN:
            print(f"\ntime budget reached after {i - 1} specs — saving what is done")
            break
        Xtr, Xte, _ = views[vname]
        tag = "_".join(filter(None, [family, *(f"{k}{v}" for k, v in kw.items()), tr, vname]))
        tag = f"g{kept + 1:03d}_{tag}".replace(".", "").replace("-", "")
        t1 = time.time()
        try:
            oof, test = run_spec(family, kw, tr, Xtr, Xte, y, folds, rng)
            auc = roc_auc_score(y, oof)
        except Exception as exc:
            failed += 1
            print(f"  [{i}/{len(plan)}] {tag:<44} FAILED {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t1
        if auc < 0.5:
            oof, test, auc = -oof, -test, 1.0 - auc
            tag += "_flip"
        if auc < MIN_KEEP_AUC:
            skipped += 1
            print(f"  [{i}/{len(plan)}] {tag:<44} auc={auc:.5f} below {MIN_KEEP_AUC} — dropped "
                  f"({dt:.0f}s)")
            continue
        OOF[:, kept] = oof.astype(np.float32)
        TEST[:, kept] = test.astype(np.float32)
        rows.append(dict(id=tag, family=family, view=vname, transform=tr or "none",
                         solo_oof_auc=round(float(auc), 7), seconds=round(dt, 1)))
        kept += 1
        print(f"  [{i}/{len(plan)}] {tag:<44} auc={auc:.5f}  ({dt:.0f}s)  kept={kept}")

    if kept == 0:
        raise RuntimeError("no member survived; nothing to save")

    header("SAVING")
    np.save(OUTPUT_DIR / "oof.npy", OOF[:, :kept])
    np.save(OUTPUT_DIR / "test.npy", TEST[:, :kept])
    meta = pd.DataFrame(rows)
    meta.to_csv(OUTPUT_DIR / "members.csv", index=False)
    print(f"kept {kept}   dropped {skipped} (auc<{MIN_KEEP_AUC})   failed {failed}")
    print(f"oof.npy {OOF[:, :kept].shape} float32 "
          f"({OOF[:, :kept].nbytes / 1024**2:.0f}MB), test.npy {TEST[:, :kept].shape}")
    print(f"solo AUC {meta.solo_oof_auc.min():.5f} .. {meta.solo_oof_auc.max():.5f}")
    print(f"\nper family:\n{meta.groupby('family').solo_oof_auc.agg(['size','max']).to_string()}")
    print(f"\nper view:\n{meta.groupby('view').solo_oof_auc.agg(['size','max']).to_string()}")

    above = int((meta.solo_oof_auc >= 0.90).sum())
    print(f"\n{above}/{kept} members clear train.py's MIN_SOLO_AUC=0.90 screen.")
    print("Publish /kaggle/working as a Kaggle Dataset, then attach it to the V6 notebook.")
    print("No train.py change is needed: oof.npy + test.npy + members.csv IS layout 5, which")
    print("load_pool() already reads. Add the dataset slug to AUDITED_TRUSTED_SOURCES so these")
    print("members also count in the audited pool — fold_id.npy above is what earns that.")
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    main()
