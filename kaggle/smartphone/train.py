import gc
import glob
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import ks_2samp, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "addicted_label"
ID_COL = "id"
SEED = 42
N_SPLITS = 5
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

AUTO_SMOKE = True

MIN_SOLO_AUC = 0.90
MAX_KS = 0.05
KS_SAMPLE = 40_000

CORRECTOR_SUBSTRINGS = ("perp",)

LEVEL2_PREFIXES = ("naji_blend",)
LEVEL2_SUBSTRINGS = ("blend",)
LEVEL2_NAMES = {
    "pub_ravi": "level-2 stack (szymonkapiski, 'the score I would not pick')",
    "sixmember_meta": "level-2: crossfit logistic stack over beicicc members already in the pool",
    "sixmember_equal_rank": "level-2: equal-weight rank blend of beicicc members already in pool",
}

KNOWN_DIFFERENT_FOLDS = {
    "pub_evg": "StratifiedKFold(n_splits=10), read from source by szymonkapiski",
    "foldsafe_te_xgb_10f": "10-fold, stated in the stream name (boltuzamaki)",
    "xgb_te_4fold": "4-fold, stated in the stream name (boltuzamaki)",
}

REQUIRE_FOLD_MATCH = True

AUDITED_TRUSTED_SOURCES = ("beicicc", "dariushafshar", "raykkretzschmar", "szymonkapiski",
                           "adarsh1077", "generative-members")
AUDITED_NAME_ALLOW = ("_5fold",)
WRITE_AUDITED_SUBMISSION = True

META_C_DEFAULT = 0.03
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
C_SEARCH_ROWS = 200_000
C_MIN_GAIN = 5e-5
MAX_ITER = 5000
TOL = 1e-5

USE_SUBSPACE_ARM = False
USE_META_LGB_ARM = False
TIME_BUDGET_MIN = 300
_T_START = time.time()
_CKPT = None
ARMS_ON_AUDITED = False
N_BAGS = 8
POOL_SUBSPACE_FRACTION = 0.7
ARM_MIN_GAIN = 1e-4
CALIBRATE_ARM_GATE = True
ARM_GATE_BOUNDS = (2e-5, 5e-4)

USE_REGIME_ARM = True
REGIME_MODE = "onehot3"
REGIME_EDGES = (0, 3)
REGIME_MAX_ITER = 400
REGIME_MAX_COLS = 1200
REGIME_TOP_K = 120

OWN_SOURCE_MARKER = "generative-members"
AB_PUBLIC_ONLY = True
META_LGB_PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, n_estimators=1200,
                       num_leaves=31, max_depth=6, min_child_samples=200, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=5.0,
                       n_jobs=-1, verbosity=-1)

EXPECTED_OOF_TO_LB_GAP = 0.0011

USE_RANK_FUSION = True
FUSION_GRID = tuple(round(w, 2) for w in np.arange(0.0, 1.01, 0.05))
FUSION_MIN_GAIN = None

_SMOKE = dict(n_splits=2, c_grid=(META_C_DEFAULT,), max_iter=200, n_bags=2,
              meta_trees=60, audited=False)


def calibrate_arm_gate():
    global ARM_MIN_GAIN
    hits = sorted(glob.glob("/kaggle/input/**/noise_floor_scenarios.csv", recursive=True))
    if not hits:
        print(f"noise_floor_scenarios.csv not attached -> keeping ARM_MIN_GAIN={ARM_MIN_GAIN}")
        return
    try:
        df = pd.read_csv(hits[0])
        print(f"measured noise floor from {hits[0]}:")
        print(df.to_string(index=False))
        col = next((c for c in ("resolvable_95pct_near_identical",
                                "sd_paired_delta_near_identical") if c in df.columns), None)
        if col is None:
            print("  no recognised resolvable-delta column -> keeping the default")
            return
        row = (df.iloc[(df["public_fraction"] - 0.20).abs().argsort().iloc[0]]
               if "public_fraction" in df.columns else df.iloc[0])
        pub = float(row[col])
        if col.startswith("sd_"):
            pub *= 1.96
        n_pub = float(row["n_public_rows"]) if "n_public_rows" in df.columns else 59_260.0
        derived = pub * np.sqrt(n_pub / 691_369.0)
        print(f"  {col} = {pub:.7f} on {n_pub:,.0f} public rows")
        print(f"  -> OOF-scale resolvable delta = {derived:.7f}")
        lo, hi = ARM_GATE_BOUNDS
        if not (lo <= derived <= hi):
            print(f"  outside sanity bounds {ARM_GATE_BOUNDS} -> keeping "
                  f"ARM_MIN_GAIN={ARM_MIN_GAIN}")
            return
        print(f"  ARM_MIN_GAIN {ARM_MIN_GAIN:.7f} -> {derived:.7f} (measured, not guessed)")
        ARM_MIN_GAIN = derived
    except Exception as exc:
        print(f"  could not use it ({type(exc).__name__}: {exc}) -> keeping the default")

    for extra in sorted(glob.glob("/kaggle/input/**/flag_gain_null_band.csv", recursive=True)):
        try:
            d = pd.read_csv(extra)
            keep = [c for c in ("flag", "mean_gain_3_seeds", "null_band_max",
                                "ratio_to_null_max", "verdict") if c in d.columns]
            print(f"\nmissingness-flag null band ({extra}) — context for the regime arm:")
            print(d[keep].to_string(index=False) if keep else d.head().to_string(index=False))
        except Exception:
            pass


def get_data_dir() -> Path:
    kaggle_dir = Path("/kaggle/input/competitions/playground-series-s6e8")
    if kaggle_dir.exists():
        return kaggle_dir
    for hit in sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True)):
        d = Path(hit).parent
        if (d / "test.csv").exists() and (d / "sample_submission.csv").exists():
            return d
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def source_of(path) -> str:
    parts = Path(path).parts
    try:
        i = parts.index("input")
    except ValueError:
        return "?"
    tail = parts[i + 1:]
    return tail[1] if tail and tail[0] == "datasets" and len(tail) > 1 else (tail[0] if tail else "?")


def rank01(v):
    return (rankdata(v, method="average") - 0.5) / len(v)


def rank_gauss(v):
    return ndtri(np.clip(rank01(v), 1e-7, 1.0 - 1e-7)).astype(np.float32)


def strict_rank01(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    order = np.argsort(v, kind="mergesort")
    out = np.empty(len(v), dtype=np.int64)
    out[order] = np.arange(len(v), dtype=np.int64)
    return (out + 0.5) / len(v)


def header(text):
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


def build_folds(y):
    return list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
                .split(np.zeros(len(y)), y))


def verify_fold_maps(y):
    n = len(y)
    fold_id = np.empty(n, dtype=np.int8)
    for f, (_, va) in enumerate(build_folds(y)):
        fold_id[va] = f
    hits = sorted(set(glob.glob("/kaggle/input/**/*fold_id*.npy", recursive=True))
                  | set(glob.glob("/kaggle/input/**/folds_*.npy", recursive=True)))
    ok, bad = 0, []
    for p in hits:
        try:
            ext = np.load(p).astype(int).ravel()
        except Exception:
            continue
        if len(ext) != n:
            continue
        if ext.min() == 1:
            ext = ext - 1
        if np.array_equal(ext, fold_id):
            ok += 1
        else:
            bad.append(p)
    print(f"fold maps checked: {ok} identical, {len(bad)} mismatched")
    for p in bad:
        print(f"  MISMATCH {p}")
    if bad and REQUIRE_FOLD_MATCH:
        raise RuntimeError(
            f"{len(bad)} published fold map(s) disagree with StratifiedKFold({N_SPLITS}, "
            f"shuffle=True, random_state={SEED}). Every member is joined by row position, so a "
            f"differing partition invalidates the join. Detach that dataset or set "
            f"REQUIRE_FOLD_MATCH = False after deciding it is safe."
        )
    return ok


def load_pool(n_train, n_test, train_ids, test_ids):
    members, dropped_shape = {}, 0

    def claim(raw_name, src_path, layout, loader):
        src = source_of(src_path)
        name = f"{src.split('/')[0]}__{raw_name}"
        if name in members:
            return
        members[name] = dict(source=src, raw=raw_name, layout=layout, loader=loader,
                             path=str(src_path))

    for op in sorted(glob.glob("/kaggle/input/**/*.npy", recursive=True)):
        base, d = Path(op).stem, os.path.dirname(op)
        if "fold" in base.lower():
            continue
        if base.startswith("oof_"):
            key, tp, lay = base[4:], os.path.join(d, f"test_{base[4:]}.npy"), "npy:oof_X"
        elif base.endswith("_oof"):
            key, tp, lay = base[:-4], os.path.join(d, f"{base[:-4]}_test.npy"), "npy:X_oof"
        else:
            continue
        if not os.path.exists(tp):
            continue
        try:
            a, b = np.load(op, mmap_mode="r"), np.load(tp, mmap_mode="r")
        except Exception:
            continue
        if a.ndim != 1 or a.shape != (n_train,) or b.shape != (n_test,):
            dropped_shape += 1
            continue

        def npy_loader(o=op, t=tp):
            return lambda: (np.load(o).astype(np.float64).ravel(),
                            np.load(t).astype(np.float64).ravel())

        claim(key, op, lay, npy_loader(op, tp))

    for meta_path in sorted(glob.glob("/kaggle/input/**/members.csv", recursive=True)):
        d = os.path.dirname(meta_path)
        op, tp = os.path.join(d, "oof.npy"), os.path.join(d, "test.npy")
        if not (os.path.exists(op) and os.path.exists(tp)):
            continue
        try:
            a, b = np.load(op, mmap_mode="r"), np.load(tp, mmap_mode="r")
            meta = pd.read_csv(meta_path)
        except Exception:
            continue
        if a.ndim != 2 or a.shape[0] != n_train or b.shape[0] != n_test \
                or a.shape[1] != b.shape[1]:
            continue
        col = next((c for c in ("id", "member", "name") if c in meta.columns), None)
        names = meta[col].astype(str).tolist() if col is not None else []
        if len(names) != a.shape[1]:
            names = [f"m{j + 1:02d}" for j in range(a.shape[1])]

        def col_loader(o=op, t=tp, j=0):
            return lambda: (np.load(o, mmap_mode="r")[:, j].astype(np.float64),
                            np.load(t, mmap_mode="r")[:, j].astype(np.float64))

        for j, nm in enumerate(names):
            claim(nm, op, "npy:2-D+members.csv", col_loader(op, tp, j))

    for op in sorted(glob.glob("/kaggle/input/**/*oof*.parquet", recursive=True)):
        tp = os.path.join(os.path.dirname(op), Path(op).name.replace("oof", "test"))
        if not os.path.exists(tp):
            continue
        try:
            import pyarrow.parquet as pq
            ocols = set(pq.ParquetFile(op).schema.names)
            tcols = set(pq.ParquetFile(tp).schema.names)
        except Exception:
            try:
                ocols = set(pd.read_parquet(op).columns)
                tcols = set(pd.read_parquet(tp).columns)
            except Exception:
                continue
        if ID_COL not in ocols or ID_COL not in tcols:
            continue
        shared = sorted((ocols & tcols) - {ID_COL})

        def pq_loader(o=op, t=tp, c=""):
            return lambda: (
                pd.read_parquet(o, columns=[ID_COL, c]).set_index(ID_COL)
                  .reindex(train_ids)[c].to_numpy(np.float64),
                pd.read_parquet(t, columns=[ID_COL, c]).set_index(ID_COL)
                  .reindex(test_ids)[c].to_numpy(np.float64))

        for c in shared:
            claim(c, op, "parquet+id", pq_loader(op, tp, c))

    for op in sorted(glob.glob("/kaggle/input/**/*_blend_oof_predictions.csv", recursive=True)):
        k = Path(op).name.split("_")[0]
        d = os.path.dirname(op)
        tp = next((c for c in (os.path.join(d, f"{k}_blend_submission.csv"),
                               os.path.join(d, f"{k}_blend_submission.csv.csv"))
                   if os.path.exists(c)), None)
        if tp is None:
            continue

        def csv_loader(o=op, t=tp):
            return lambda: (
                pd.read_csv(o).set_index(ID_COL).reindex(train_ids).iloc[:, 0]
                  .to_numpy(np.float64),
                pd.read_csv(t).set_index(ID_COL).reindex(test_ids).iloc[:, 0]
                  .to_numpy(np.float64))

        claim(f"naji_blend{k}", op, "blend csv pair", csv_loader(op, tp))

    print(f"collected {len(members)} members "
          f"({dropped_shape} npy pairs skipped on shape)")
    return members


def screen_pool(members, y, n_test):
    n = len(y)
    rng = np.random.default_rng(0)
    ia = rng.choice(n, min(KS_SAMPLE, n), replace=False)
    ib = rng.choice(n_test, min(KS_SAMPLE, n_test), replace=False)

    rows, seen, kept = [], {}, []
    t0 = time.time()
    names = sorted(members)
    for i, name in enumerate(names, start=1):
        info = members[name]
        raw = info["raw"]
        corrector = any(s in raw.lower() for s in CORRECTOR_SUBSTRINGS)
        rec = dict(member=name, source=info["source"], layout=info["layout"],
                   solo_auc=np.nan, ks=np.nan, verdict="keep", corrector=corrector)
        try:
            o, t = info["loader"]()
            if o.shape != (n,) or t.shape != (n_test,):
                rec["verdict"] = "DROP shape"
            elif not (np.isfinite(o).all() and np.isfinite(t).all()):
                rec["verdict"] = "DROP nonfinite"
            else:
                rec["solo_auc"] = float(roc_auc_score(y, o))
                rec["ks"] = float(ks_2samp(rank01(o)[ia], rank01(t)[ib]).statistic)
                rec["lo"], rec["hi"] = float(o.min()), float(o.max())
                h = hashlib.md5(np.ascontiguousarray(o).tobytes()).hexdigest()
                if h in seen:
                    rec["verdict"] = f"DROP dup of {seen[h]}"
                elif raw in LEVEL2_NAMES or raw.startswith(LEVEL2_PREFIXES) \
                        or any(s in raw.lower() for s in LEVEL2_SUBSTRINGS):
                    rec["verdict"] = "DROP level-2"
                elif raw in KNOWN_DIFFERENT_FOLDS:
                    rec["verdict"] = "DROP fold protocol"
                elif not corrector and rec["solo_auc"] < MIN_SOLO_AUC:
                    rec["verdict"] = f"DROP auc<{MIN_SOLO_AUC}"
                elif not corrector and rec["ks"] > MAX_KS:
                    rec["verdict"] = f"DROP ks={rec['ks']:.3f}"
                else:
                    seen[h] = name
            del o, t
        except Exception as exc:
            rec["verdict"] = f"ERROR {type(exc).__name__}: {exc}"
        if rec["verdict"] == "keep":
            kept.append(name)
        rows.append(rec)
        if i % 50 == 0 or i == len(names):
            print(f"  screened {i}/{len(names)}  [{time.time() - t0:.0f}s]")

    report = pd.DataFrame(rows).sort_values("solo_auc", ascending=False, na_position="last")
    dropped = report[~report.verdict.eq("keep")]
    print(f"\nkept {len(kept)}, dropped {len(dropped)}")
    if len(dropped):
        print(dropped[["member", "source", "solo_auc", "ks", "verdict"]].to_string(index=False))
    if "lo" in report.columns:
        wide = report[report.lo.fillna(0) < -0.05]
        print(f"\n{len(wide)} members carry values well outside [0, 1] (raw logits / decision "
              f"scores). Rank-gauss handles these; V5's clip-then-logit would have destroyed "
              f"them -- see docs/FINDINGS.md §14.2.")
    return kept, report


def audited_subset(kept, members, report):
    out = []
    for name in kept:
        path = members[name]["path"].lower()
        raw = name.split("__", 1)[-1].lower()
        trusted = any(s.lower() in path for s in AUDITED_TRUSTED_SOURCES)
        if trusted or any(a in raw for a in AUDITED_NAME_ALLOW):
            out.append(name)
    return out


def build_matrix(members, kept, n_train, n_test):
    G = np.empty((n_train, len(kept)), dtype=np.float32)
    Gt = np.empty((n_test, len(kept)), dtype=np.float32)
    t0 = time.time()
    for j, name in enumerate(kept):
        o, t = members[name]["loader"]()
        G[:, j] = rank_gauss(o)
        Gt[:, j] = rank_gauss(t)
        del o, t
        if (j + 1) % 50 == 0 or j + 1 == len(kept):
            print(f"  transformed {j + 1}/{len(kept)}  [{time.time() - t0:.0f}s]")
    assert np.isfinite(G).all() and np.isfinite(Gt).all()
    print(f"pool: train {G.shape}, test {Gt.shape}, "
          f"{(G.nbytes + Gt.nbytes) / 1024 ** 3:.2f}GB")
    return G, Gt


def _fit_logit(X, y, C, max_iter=None):
    return LogisticRegression(C=C, max_iter=MAX_ITER if max_iter is None else max_iter,
                              solver="lbfgs", tol=TOL).fit(X, y)


def search_C(G, y, c_grid, max_iter=None):
    if len(c_grid) == 1:
        print(f"  C fixed at {c_grid[0]:g}")
        return c_grid[0]
    idx = np.arange(len(y))
    if C_SEARCH_ROWS < len(y):
        idx, _ = train_test_split(idx, train_size=C_SEARCH_ROWS, random_state=SEED, stratify=y)
    i_fit, i_score = train_test_split(idx, train_size=0.6, random_state=SEED, stratify=y[idx])
    scaler = StandardScaler().fit(G[i_fit])
    Xf, Xs = scaler.transform(G[i_fit]), scaler.transform(G[i_score])
    scores = {}
    for C in c_grid:
        m = _fit_logit(Xf, y[i_fit], C, max_iter)
        scores[C] = roc_auc_score(y[i_score], m.decision_function(Xs))
        print(f"  C={C:<7g} holdout AUC={scores[C]:.6f}")
    best = max(scores, key=scores.get)
    if META_C_DEFAULT in scores and scores[best] - scores[META_C_DEFAULT] < C_MIN_GAIN:
        print(f"  best C={best:g} beats incumbent {META_C_DEFAULT:g} by "
              f"{scores[best] - scores[META_C_DEFAULT]:+.6f} < {C_MIN_GAIN} -> keeping incumbent")
        return META_C_DEFAULT
    print(f"  adopting C={best:g}")
    return best


def nested_stack(G, Gt, y, folds, C, max_iter=None, label="stack"):
    oof = np.zeros(len(y))
    for f, (trn, val) in enumerate(folds, start=1):
        scaler = StandardScaler().fit(G[trn])
        m = _fit_logit(scaler.transform(G[trn]), y[trn], C, max_iter)
        oof[val] = m.decision_function(scaler.transform(G[val]))
        print(f"  [{label}] fold {f}: AUC={roc_auc_score(y[val], oof[val]):.7f} "
              f"(iters={int(np.max(m.n_iter_))})")
    auc = roc_auc_score(y, oof)
    scaler = StandardScaler().fit(G)
    m = _fit_logit(scaler.transform(G), y, C, max_iter)
    test = m.decision_function(scaler.transform(Gt))
    print(f"  [{label}] nested OOF AUC = {auc:.7f}")
    return auc, oof, test


def subspace_arm(G, Gt, y, folds, C, n_bags, max_iter=None):
    rng = np.random.default_rng(SEED)
    keep_n = max(2, int(round(POOL_SUBSPACE_FRACTION * G.shape[1])))
    oof_acc = np.zeros(len(y))
    test_acc = np.zeros(Gt.shape[0])
    for b in range(n_bags):
        cols = rng.choice(G.shape[1], size=keep_n, replace=False)
        auc, o, t = nested_stack(G[:, cols], Gt[:, cols], y, folds, C, max_iter,
                                 label=f"bag{b + 1}")
        oof_acc += rank01(o) / n_bags
        test_acc += rank01(t) / n_bags
    auc = roc_auc_score(y, oof_acc)
    print(f"  [subspace] nested OOF AUC = {auc:.7f} ({n_bags} bags, {keep_n}/{G.shape[1]} cols)")
    return auc, oof_acc, test_acc


def missing_regime(data_dir, n_train, n_test):
    out = []
    for fn, expected in (("train.csv", n_train), ("test.csv", n_test)):
        df = pd.read_csv(data_dir / fn)
        feats = [c for c in df.columns if c not in (ID_COL, TARGET)]
        mc = df[feats].isna().sum(axis=1).to_numpy()
        assert len(mc) == expected, f"{fn}: {len(mc)} rows != {expected}"
        lo, hi = REGIME_EDGES
        out.append((mc, np.where(mc <= lo, 0, np.where(mc <= hi, 1, 2)).astype(np.int8)))
    (mc_tr, r_tr), (mc_te, r_te) = out
    print(f"missingness regimes (0 missing / 1-{hi} / {hi + 1}+):")
    for r in (0, 1, 2):
        print(f"  regime {r}: train {int((r_tr == r).sum()):>7,} "
              f"({(r_tr == r).mean():6.2%})   test {int((r_te == r).sum()):>7,} "
              f"({(r_te == r).mean():6.2%})")
    print("  the train/test share difference is exactly why regime-specific weights can pay: a "
          "single global weighting is fitted to train's mix and applied to test's "
          "(docs/FINDINGS.md §1).")
    return mc_tr, r_tr, mc_te, r_te


def _regime_design(G, regime, mc, cols=None):
    sub = G if cols is None else G[:, cols]
    if REGIME_MODE == "scalar":
        z = ((mc - mc.mean()) / (mc.std() + 1e-9)).astype(np.float32)
        return np.hstack([G, sub * z[:, None]])
    blocks = [G]
    for r in (1, 2):
        blocks.append(sub * (regime == r).astype(np.float32)[:, None])
    return np.hstack(blocks)


def regime_top_cols(G, y, folds, C):
    if not REGIME_TOP_K or REGIME_TOP_K >= G.shape[1]:
        return None
    trn = folds[0][0]
    scaler = StandardScaler().fit(G[trn])
    m = _fit_logit(scaler.transform(G[trn]), y[trn], C, REGIME_MAX_ITER)
    order = np.argsort(-np.abs(m.coef_.ravel()))
    cols = np.sort(order[:REGIME_TOP_K])
    print(f"  [regime] interacting the top {len(cols)} of {G.shape[1]} members by |coef| "
          f"(fold-1 train rows only)")
    print("  [regime] CAVEAT: this subset was chosen once on fold-1 train, which overlaps folds "
          "2-5's val rows.\n"
          "           Column-level selection leak -> if this arm clears the gate, treat the "
          "delta as an UPPER BOUND (see regime_top_cols docstring).")
    return cols


def regime_arm(G, Gt, y, folds, C, reg_tr, mc_tr, reg_te, mc_te, max_iter=None):
    cols = regime_top_cols(G, y, folds, C)
    n_int = G.shape[1] if cols is None else len(cols)
    n_design = G.shape[1] + 2 * n_int if REGIME_MODE == "onehot3" else G.shape[1] + n_int
    if n_design > REGIME_MAX_COLS:
        print(f"  [regime] design would be {n_design} columns > REGIME_MAX_COLS="
              f"{REGIME_MAX_COLS} -> arm skipped (cost guard, not a measurement)")
        return None
    cap = REGIME_MAX_ITER if max_iter is None else min(max_iter, REGIME_MAX_ITER)
    X = _regime_design(G, reg_tr, mc_tr, cols)
    Xt = _regime_design(Gt, reg_te, mc_te, cols)
    print(f"  [regime] design {X.shape[1]} columns ({REGIME_MODE}), "
          f"{(X.nbytes + Xt.nbytes) / 1024 ** 3:.2f}GB, max_iter={cap}")
    auc, oof, test = nested_stack(X, Xt, y, folds, C, cap, "regime")
    del X, Xt
    gc.collect()
    return auc, oof, test


def meta_lgb_arm(G, Gt, y, folds, n_estimators):
    import lightgbm as lgb
    params = dict(META_LGB_PARAMS, n_estimators=n_estimators)
    oof = np.zeros(len(y))
    test = np.zeros(Gt.shape[0])
    for f, (trn, val) in enumerate(folds, start=1):
        m = lgb.LGBMClassifier(random_state=SEED, **params)
        m.fit(G[trn], y[trn], eval_set=[(G[val], y[val])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[val] = m.predict_proba(G[val])[:, 1]
        test += m.predict_proba(Gt)[:, 1] / len(folds)
        print(f"  [meta_lgb] fold {f}: AUC={roc_auc_score(y[val], oof[val]):.7f} "
              f"(best_iter={m.best_iteration_})")
    auc = roc_auc_score(y, oof)
    print(f"  [meta_lgb] nested OOF AUC = {auc:.7f}")
    return auc, oof, test


def rank_fusion(oof_a, test_a, oof_b, test_b, y, folds, label_a, label_b, gate):
    ra, rb = rank01(oof_a), rank01(oof_b)
    fused = np.zeros(len(y))
    picks = []
    for trn, val in folds:
        best_w, best_s = None, -np.inf
        for w in FUSION_GRID:
            s = roc_auc_score(y[trn], w * ra[trn] + (1 - w) * rb[trn])
            if s > best_s:
                best_w, best_s = w, s
        fused[val] = best_w * ra[val] + (1 - best_w) * rb[val]
        picks.append(best_w)
    auc = roc_auc_score(y, fused)
    a_auc, b_auc = roc_auc_score(y, ra), roc_auc_score(y, rb)
    solo = max(a_auc, b_auc)
    print(f"  [fusion] {label_a} {a_auc:.7f} | {label_b} {b_auc:.7f}")
    print(f"  [fusion] per-fold w({label_a}) = {picks}  (nested: chosen on 4 folds, applied to the 5th)")
    print(f"  [fusion] nested OOF {auc:.7f}  ({auc - solo:+.7f} vs the better endpoint)")
    if auc - solo < gate:
        print(f"  [fusion] below gate {gate:.7f} -> not shipped; the endpoints are too similar "
              f"for a rank blend to add anything")
        return None
    w_final = max(FUSION_GRID, key=lambda w: roc_auc_score(y, w * ra + (1 - w) * rb))
    print(f"  [fusion] shipping w={w_final:.2f} on {label_a}, {1 - w_final:.2f} on {label_b}")
    return auc, fused, w_final * rank01(test_a) + (1 - w_final) * rank01(test_b), w_final


def write_submission(test_scores, test_ids, sample_submission, filename):
    pred = strict_rank01(test_scores)
    sub = sample_submission.copy()
    assert sub[ID_COL].equals(pd.Series(test_ids, name=ID_COL)), \
        "sample_submission id order differs from test.csv"
    sub[TARGET] = pred
    assert len(sub) == len(test_ids)
    assert sub[TARGET].notna().all() and np.isfinite(sub[TARGET]).all()
    assert (sub[TARGET] > 0).all() and (sub[TARGET] < 1).all()
    assert sub[TARGET].nunique() == len(sub), "strict rank must give one value per row"
    sub.to_csv(OUTPUT_DIR / filename, index=False)
    print(f"wrote {filename}  {sub.shape}")
    return sub


def budget_left(need_min, what):
    spent = (time.time() - _T_START) / 60
    if spent + need_min > TIME_BUDGET_MIN:
        print(f"  [budget] {spent:.0f} min spent, {what} needs ~{need_min} more, "
              f"ceiling is {TIME_BUDGET_MIN} -> skipped to protect the submission")
        return False
    return True


def run(G, Gt, y, folds, c_grid, max_iter, n_bags, meta_trees, label, arms=True,
        regime=None):
    header(f"META-MODEL — {label}  ({G.shape[1]} members)")
    print("C search:")
    C = search_C(G, y, c_grid, max_iter)
    print()
    base_auc, base_oof, base_test = nested_stack(G, Gt, y, folds, C, max_iter, "rank-gauss")

    best = ("rank_gauss_logistic", base_auc, base_oof, base_test)
    if _CKPT is not None and label == "full pool":
        _CKPT(base_test, base_oof, base_auc)
    if not arms:
        print(f"\n-> {label}: shipping rank_gauss_logistic, nested OOF {base_auc:.7f} "
              f"(arms skipped)")
        print(f"   expected public LB ~{base_auc + EXPECTED_OOF_TO_LB_GAP:.5f}")
        return (*best, base_auc)
    if USE_SUBSPACE_ARM:
        print()
        a, o, t = subspace_arm(G, Gt, y, folds, C, n_bags, max_iter)
        if a - base_auc >= ARM_MIN_GAIN:
            best = ("subspace_bagged", a, o, t)
        else:
            print(f"  subspace arm {a - base_auc:+.7f} vs incumbent, "
                  f"below ARM_MIN_GAIN={ARM_MIN_GAIN:.7f} -> not adopted")
    if USE_REGIME_ARM and regime is not None and budget_left(45, "the regime arm"):
        print()
        got = regime_arm(G, Gt, y, folds, C, *regime, max_iter=max_iter)
        if got is not None:
            a, o, t = got
            if a - base_auc >= ARM_MIN_GAIN and a > best[1]:
                best = ("regime_stack", a, o, t)
            else:
                print(f"  regime arm {a - base_auc:+.7f} vs incumbent, "
                      f"below ARM_MIN_GAIN={ARM_MIN_GAIN:.7f} -> not adopted")
            del got, o, t
            gc.collect()
    if USE_META_LGB_ARM and budget_left(70, "the meta-LightGBM arm"):
        print()
        a, o, t = meta_lgb_arm(G, Gt, y, folds, meta_trees)
        if a - base_auc >= ARM_MIN_GAIN and a > best[1]:
            best = ("meta_lgb", a, o, t)
        else:
            print(f"  meta-LightGBM arm {a - base_auc:+.7f} vs incumbent, "
                  f"below ARM_MIN_GAIN={ARM_MIN_GAIN:.7f} -> not adopted")

    method, auc, oof, test = best
    print(f"\n-> {label}: shipping {method}, nested OOF {auc:.7f}")
    print(f"   expected public LB ~{auc + EXPECTED_OOF_TO_LB_GAP:.5f} "
          f"(gap +{EXPECTED_OOF_TO_LB_GAP:.4f}, stable across every pool measured so far)")
    return method, auc, oof, test, base_auc


def main():
    t0 = time.time()
    data_dir = get_data_dir()
    train = pd.read_csv(data_dir / "train.csv", usecols=[ID_COL, TARGET])
    test = pd.read_csv(data_dir / "test.csv", usecols=[ID_COL])
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    y = train[TARGET].to_numpy(np.int8)
    train_ids, test_ids = train[ID_COL].to_numpy(), test[ID_COL].to_numpy()
    n, n_test = len(train), len(test)
    print(f"data: {data_dir}")
    print(f"train {n:,} | test {n_test:,} | positive rate {y.mean():.6f}")

    header("1. FOLD VERIFICATION")
    if verify_fold_maps(y) == 0:
        print("no published fold map found -- positional alignment is assumed, not verified.")

    header("2. DECISION THRESHOLD, FROM A MEASURED NOISE FLOOR")
    if CALIBRATE_ARM_GATE:
        calibrate_arm_gate()
    else:
        print(f"calibration off -> ARM_MIN_GAIN={ARM_MIN_GAIN}")

    regime = None
    if USE_REGIME_ARM:
        print()
        mc_tr, r_tr, mc_te, r_te = missing_regime(data_dir, n, n_test)
        regime = (r_tr, mc_tr, r_te, mc_te)

    header("3. POOL")
    members = load_pool(n, n_test, train_ids, test_ids)
    if not members:
        raise RuntimeError(
            "no members loaded. V6 trains nothing on raw features, so there is no fallback. "
            "Attach the OOF libraries listed in docs/NOTEBOOKS_SUMMARY.md round 3 and rerun."
        )

    header("4. SCREENING")
    kept, report = screen_pool(members, y, n_test)
    report.to_csv(OUTPUT_DIR / "pool_screen_report.csv", index=False)
    if not kept:
        raise RuntimeError("every member was screened out; check pool_screen_report.csv")
    print("\nkept per source:")
    print(report[report.verdict.eq("keep")].groupby("source")
          .agg(members=("member", "size"), best=("solo_auc", "max"),
               worst=("solo_auc", "min")).sort_values("members", ascending=False).to_string())
    best_single = report[report.verdict.eq("keep")].solo_auc.max()
    best_single_name = report[report.verdict.eq("keep")].iloc[0].member
    print(f"\nbest single kept member: {best_single_name} {best_single:.7f}")

    header("5. RANK-GAUSS TRANSFORM")
    G, Gt = build_matrix(members, kept, n, n_test)
    folds_full = build_folds(y)

    global _CKPT

    def _checkpoint(test_scores, oof, auc):
        write_submission(test_scores, test_ids, sample_submission, "submission.csv")
        pd.DataFrame({ID_COL: train_ids, TARGET: y, "oof_pred": oof}) \
            .to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
        print(f"  [checkpoint] base stack saved (OOF {auc:.7f}) — the run now has a valid "
              f"submission even if an arm below is killed")

    _CKPT = _checkpoint

    own_names = {nm for nm, info in members.items()
                 if OWN_SOURCE_MARKER in info["path"].lower()}
    ours = [i for i, nm in enumerate(kept) if nm in own_names]
    public = [i for i, nm in enumerate(kept) if nm not in own_names]
    if ours:
        own = report[report.member.isin([kept[i] for i in ours])]
        print(f"\nOUR OWN GENERATIVE MEMBERS: {len(ours)} kept of {len(own_names)} loaded "
              f"({len(kept)} total members in the pool)")
        print(f"  solo AUC {own.solo_auc.min():.5f} .. {own.solo_auc.max():.5f}")
        dropped_own = report[(~report.verdict.eq("keep")) & report.member.isin(own_names)]
        if len(dropped_own):
            print(f"  {len(dropped_own)} of ours were screened out, by reason:")
            print(dropped_own.verdict.str.split().str[:2].str.join(" ")
                  .value_counts().to_string())
            print("  (a generative member models the FEATURE density, and missingness is what "
                  "shifts train->test, so KS drops here are expected — docs/FINDINGS.md §1)")
    else:
        print(f"\nno members matched OWN_SOURCE_MARKER={OWN_SOURCE_MARKER!r} — the generative "
              f"factory dataset is not attached, so the public-only A/B is skipped. Everything "
              f"else runs normally.")

    if AUTO_SMOKE:
        header("SMOKE PASS — trivial settings, result discarded")
        smoke_folds = list(StratifiedKFold(_SMOKE["n_splits"], shuffle=True, random_state=SEED)
                           .split(np.zeros(n), y))
        run(G, Gt, y, smoke_folds, _SMOKE["c_grid"], _SMOKE["max_iter"],
            _SMOKE["n_bags"], _SMOKE["meta_trees"], "SMOKE",
            regime=regime)
        print("\nsmoke pass OK — starting the real run")

    header("6. THE A/B — did our own members add anything?")
    public_auc = None
    if AB_PUBLIC_ONLY and ours and len(public) >= 20:
        print(f"public-only pool: {len(public)} members (this should reproduce V6's 0.9701086 if "
              f"the public libraries have not changed version)")
        C_pub = search_C(G[:, public], y, C_GRID)
        public_auc, _, _ = nested_stack(G[:, public], Gt[:, public], y, folds_full, C_pub,
                                        label="public-only")
    else:
        print("skipped — nothing of ours in the pool to attribute a delta to.")

    method, auc, oof, test_scores, full_base_auc = run(
        G, Gt, y, folds_full, C_GRID, None, N_BAGS, META_LGB_PARAMS["n_estimators"],
        "full pool", regime=regime)

    if public_auc is not None:
        d = full_base_auc - public_auc
        print(f"\npublic-only  {len(public):>4} members  nested OOF {public_auc:.7f}")
        print(f"+ our {len(ours):<3}    {len(kept):>4} members  nested OOF {full_base_auc:.7f}   "
              f"({d:+.7f})   [plain rank-gauss on both sides]")
        verdict = ("ADDS SIGNAL" if d >= ARM_MIN_GAIN else
                   "within the resolvable floor — treat as no effect" if d > -ARM_MIN_GAIN else
                   "HURTS — ship the public-only pool instead")
        print(f"-> our generative members: {verdict}")
        print("   §13's ladder is the reference: 10/20/30 weak arrays measured -0.0000120 / "
              "-0.0000122 / -0.0000047, turning positive only around 50.")

    header("7. SUBMISSION")
    print(f"full pool     : {len(kept)} members, nested OOF {auc:.7f} ({method})")
    print(f"best single   : {best_single:.7f} ({best_single_name})")
    print(f"gain over best single: {auc - best_single:+.7f}")
    write_submission(test_scores, test_ids, sample_submission, "submission.csv")
    pd.DataFrame({ID_COL: train_ids, TARGET: y, "oof_pred": oof}) \
        .to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

    if WRITE_AUDITED_SUBMISSION:
        header("8. AUDITED POOL — the second submission slot")
        aud = audited_subset(kept, members, report)
        print(f"{len(aud)} of {len(kept)} members survive the provenance restriction "
              f"(docs/DESIGN.md 'V6'): sources with a verified fold map, published training "
              f"code, or a single stated 5-fold protocol. Dropped: the library its own author "
              f"documents as a 5/10-fold mix (except streams named *_5fold), and one shipping "
              f"no provenance metadata.")
        if len(aud) < 20 or len(aud) == len(kept):
            print("-> not a meaningfully different pool; skipping.")
        else:
            cols = [kept.index(nm) for nm in aud]
            _, a_auc, a_oof, a_test, _ = run(G[:, cols], Gt[:, cols], y, folds_full,
                                             C_GRID, None, N_BAGS,
                                             META_LGB_PARAMS["n_estimators"], "audited pool",
                                             arms=ARMS_ON_AUDITED, regime=regime)
            write_submission(a_test, test_ids, sample_submission, "submission_audited.csv")
            pd.DataFrame({ID_COL: train_ids, TARGET: y, "oof_pred": a_oof}) \
                .to_csv(OUTPUT_DIR / "oof_predictions_audited.csv", index=False)
            print(f"\nfull pool    {len(kept):>4} members  nested OOF {auc:.7f}")
            print(f"audited pool {len(aud):>4} members  nested OOF {a_auc:.7f}  "
                  f"({a_auc - auc:+.7f})")
            print("The audited pool is expected to score slightly lower and to be the better "
                  "private-leaderboard bet (docs/FINDINGS.md §13, and both top public authors "
                  "say this is the trade they would take). Submit submission.csv as the "
                  "score-maximising slot and submission_audited.csv as the provenance slot.")

            if USE_RANK_FUSION:
                header("9. RANK FUSION — full x audited")
                print("The 0.9712x public tier is rank blends of two strong CSVs. Ours are the "
                      "two endpoints above: same frozen split, different member sets, so the "
                      "weight is chosen on OOF rather than on the leaderboard.")
                gate = FUSION_MIN_GAIN if FUSION_MIN_GAIN is not None else ARM_MIN_GAIN
                got = rank_fusion(oof, test_scores, a_oof, a_test, y, folds_full,
                                  "full", "audited", gate)
                if got is None:
                    print("-> keeping the two separate submissions.")
                else:
                    f_auc, f_oof, f_test, f_w = got
                    write_submission(f_test, test_ids, sample_submission,
                                     "submission_fusion.csv")
                    pd.DataFrame({ID_COL: train_ids, TARGET: y, "oof_pred": f_oof}) \
                        .to_csv(OUTPUT_DIR / "oof_predictions_fusion.csv", index=False)
                    print(f"\nfused        nested OOF {f_auc:.7f}   "
                          f"expected public LB ~{f_auc + EXPECTED_OOF_TO_LB_GAP:.5f}")
                    print("Three files now exist. submission_fusion.csv is the score-maximising "
                          "slot; submission_audited.csv stays the provenance slot. Note the "
                          "fusion inherits the full pool's unverifiable members at weight "
                          f"{f_w:.2f}, so it is NOT the conservative choice.")

    print(f"\ntotal {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
