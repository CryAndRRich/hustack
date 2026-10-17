import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from preprocess import (
    CATEGORICAL,
    CLASS_NAMES,
    GROUP,
    ID,
    LABELS,
    TARGET,
    build_features,
    get_output_dir,
    load_external,
    load_land_mask,
    load_raw,
    topo_table,
)

VERSION = "V2.1"
AUTO_SMOKE = True

N_FOLDS = 5
N_REPEATS = 3
N_SEEDS_FULL = 5
CV_SEED = 20260927
CB_PARAMS = dict(
    loss_function="MultiClass", iterations=800, learning_rate=0.03, depth=4, l2_leaf_reg=5.0,
    verbose=0, allow_writing_files=False, thread_count=-1,
)
SMOKE_SETTINGS = dict(folds=2, repeats=1, seeds=1, iterations=20)

SHIP_ORDER = ["v2.1"]
GATE_BASELINE = "v2_full"
SHIP_GATE_SE = 1.0

SHIP_OVERRIDE = "v2.1"

REPRODUCE = {"v1": {"acc": 0.9612, "weightedF1": 0.9533}, "v2_full": {"acc": 0.9644, "weightedF1": 0.9565}}

RUN_LOG = "run_log.txt"
N_CLASSES = len(CLASS_NAMES)


class _Tee:

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s):
        if not self.log.closed:
            self.log.write(s)
        return self.stream.write(s)

    def flush(self):
        if not self.log.closed:
            self.log.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def make_folds(y: np.ndarray, groups: np.ndarray, n_folds: int, n_repeats: int) -> list[list]:
    reps = []
    for r in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=CV_SEED + r)
        reps.append(list(skf.split(np.zeros(len(y)), y, groups)))
    return reps


def _proba(model, X: pd.DataFrame) -> np.ndarray:

    p = model.predict_proba(X)
    full = np.zeros((len(X), N_CLASSES))
    full[:, np.asarray(model.classes_, dtype=int)] = p
    return full


def fit_catboost(X: pd.DataFrame, y: np.ndarray, seed: int, iterations: int):
    params = {**CB_PARAMS, "iterations": iterations, "random_seed": seed}
    model = CatBoostClassifier(**params, cat_features=[c for c in CATEGORICAL if c in X.columns])
    model.fit(X, y)
    return model


def field_average(P: np.ndarray, groups: np.ndarray) -> np.ndarray:

    return pd.DataFrame(P).groupby(groups).transform("mean").to_numpy()


def _clip_rows(P: np.ndarray) -> np.ndarray:
    Q = np.clip(P, 1e-6, 1)
    return Q / Q.sum(1, keepdims=True)


def scores(y: np.ndarray, P: np.ndarray) -> dict:
    pred = P.argmax(1)
    return {
        "weightedF1": f1_score(y, pred, average="weighted", zero_division=0),
        "acc": accuracy_score(y, pred),
        "macroF1": f1_score(y, pred, average="macro", labels=list(range(N_CLASSES)), zero_division=0),
        "logloss": log_loss(y, _clip_rows(P), labels=list(range(N_CLASSES))),
    }


def print_scores(name: str, per_repeat: list[dict]) -> None:
    keys = per_repeat[0].keys()
    parts = [f"{k} {np.mean([s[k] for s in per_repeat]):.4f}±{np.std([s[k] for s in per_repeat]):.4f}"
             for k in keys]
    print(f"  {name:10s} " + "  ".join(parts))


def print_confusion(y: np.ndarray, pred: np.ndarray) -> None:
    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
    print("    rows = true, cols = predicted  " + "  ".join(f"{c[:8]:>8s}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c:31s}" + "  ".join(f"{v:8d}" for v in cm[i]))


def cv_arm(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, folds: list, iterations: int) -> list[np.ndarray]:
    oofs = []
    for r, rep in enumerate(folds):
        oof = np.full((len(y), N_CLASSES), np.nan)
        for tr, va in rep:
            m = fit_catboost(X.iloc[tr], y[tr], seed=CV_SEED + r, iterations=iterations)
            oof[va] = _proba(m, X.iloc[va])
        assert not np.isnan(oof).any()
        oofs.append(field_average(oof, groups))
    return oofs


def rule_geo(X: pd.DataFrame) -> np.ndarray:
    land = X["geo_is_land"].to_numpy()
    pred = np.where(land == 0, LABELS["OFFSHORE"], LABELS["ONSHORE"])
    P = np.zeros((len(X), N_CLASSES))
    P[np.arange(len(X)), pred] = 1.0
    return P


def paired_stats(y: np.ndarray, Pa: np.ndarray, Pb: np.ndarray) -> tuple[float, float, int, int]:
    d = (Pa.argmax(1) == y).astype(float) - (Pb.argmax(1) == y).astype(float)
    se = d.std(ddof=1) / np.sqrt(len(d)) if d.any() else 0.0
    return float(d.mean()), float(se), int((d > 0).sum()), int((d < 0).sum())


def paired(y: np.ndarray, Pa: np.ndarray, Pb: np.ndarray) -> str:
    m, se, won, lost = paired_stats(y, Pa, Pb)
    t = m / se if se > 0 else 0.0
    dw = scores(y, Pa)["weightedF1"] - scores(y, Pb)["weightedF1"]
    return f"acc {m:+.4f} (SE {se:.4f}, t {t:+.2f}, rows won/lost {won}/{lost}), weightedF1 {dw:+.4f}"


def choose_ship(y: np.ndarray, P: dict[str, np.ndarray]) -> str:

    base = GATE_BASELINE if GATE_BASELINE in P else "v1"
    for name in SHIP_ORDER:
        if name not in P:
            print(f"  {name}: not available (its external file is missing)")
            continue
        m, se, _, _ = paired_stats(y, P[name], P[base])
        if m >= -SHIP_GATE_SE * se:
            print(f"  -> ship {name}: acc delta vs {base} {m:+.4f} >= gate {-SHIP_GATE_SE * se:+.4f}")
            return name
        print(f"  {name} REJECTED: acc delta vs {base} {m:+.4f} < gate {-SHIP_GATE_SE * se:+.4f}")
    print(f"  -> ship {base} (no arm cleared the gate)")
    return base


def row_view(fs, i: int, split: str) -> str:
    df, X = (fs.train, fs.X_tr) if split == "train" else (fs.test, fs.X_te)
    parts = [f"{df[GROUP].iat[i]}", f"{df['Country'].iat[i]}", f"{df['Basin name'].iat[i]}",
             f"coords={df['coord_source'].iat[i]}", f"geo={df['geo_source'].iat[i]}"]
    for c in ("geo_is_land", "geo_coast_km", "topo_elev_0", "topo_min_5", "topo_below0_ring_8"):
        if c in X.columns:
            v = X[c].iat[i]
            parts.append(f"{c.replace('topo_', 't_').replace('geo_', 'g_')}={'nan' if pd.isna(v) else round(float(v), 2)}")
    return " | ".join(parts)


def topo_label_table(fs, y: np.ndarray) -> None:
    if "topo_elev_0" not in fs.X_tr.columns:
        return
    order = ["<-50", "-50..-10", "-10..0", "0..10", "10..50", ">50", "no coords"]
    b = pd.cut(fs.X_tr["topo_elev_0"], [-np.inf, -50, -10, 0, 10, 50, np.inf], labels=order[:-1])
    b = b.astype(object).where(b.notna(), "no coords")
    lab = pd.Series(y, index=fs.X_tr.index).map(dict(enumerate(CLASS_NAMES)))
    tab = pd.crosstab(b.rename("elev_0 (m)"), lab.rename("label"))
    print("\n  train labels by GEBCO elevation at the field (m, negative = below sea level):")
    print("    " + tab.reindex([o for o in order if o in tab.index]).to_string().replace("\n", "\n    "))


def refit_predict(fs, cols: list[str], y: np.ndarray, n_seeds: int, iters: int, importance: bool) -> np.ndarray:
    P = np.zeros((len(fs.X_te), N_CLASSES))
    for s in range(n_seeds):
        m = fit_catboost(fs.X_tr[cols], y, seed=CV_SEED + 100 + s, iterations=iters)
        P += _proba(m, fs.X_te[cols]) / n_seeds
        if s == 0 and importance:
            imp = pd.Series(m.get_feature_importance(), index=cols).sort_values(ascending=False)
            print("\n  feature importance (seed 0, top 25):")
            for c, v in imp.head(25).items():
                print(f"    {v:6.2f}  {c}")
    return field_average(P, fs.test[GROUP].to_numpy())


def apply_lookup(fs, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    known = fs.train.groupby(GROUP)["y"].agg(lambda s: s.mode().iloc[0])
    hit = fs.test[GROUP].isin(known.index).to_numpy()
    out = pred.copy()
    out[hit] = fs.test.loc[hit, GROUP].map(known).to_numpy()
    return out, hit


def pipeline(arms: dict, smoke: bool) -> None:
    t0 = time.time()
    n_folds = SMOKE_SETTINGS["folds"] if smoke else N_FOLDS
    n_repeats = SMOKE_SETTINGS["repeats"] if smoke else N_REPEATS
    n_seeds = SMOKE_SETTINGS["seeds"] if smoke else N_SEEDS_FULL
    iters = SMOKE_SETTINGS["iterations"] if smoke else CB_PARAMS["iterations"]

    ref = arms["v1"][0]
    y = ref.train["y"].to_numpy()
    groups = ref.train[GROUP].to_numpy()
    for fs, _ in arms.values():
        assert (fs.train["y"].to_numpy() == y).all() and (fs.train[GROUP].to_numpy() == groups).all()
        assert (fs.test[GROUP].to_numpy() == ref.test[GROUP].to_numpy()).all()
    folds = make_folds(y, groups, n_folds, n_repeats)
    print(f"\n=== CV: StratifiedGroupKFold by field, {n_folds} folds x {n_repeats} repeats, "
          f"{iters} iterations, arms {list(arms)} ===")

    oofs = {}
    for name, (fs, cols) in arms.items():
        t = time.time()
        oofs[name] = cv_arm(fs.X_tr[cols], y, groups, folds, iters)
        print(f"  {name:10s} {len(cols)} features, {time.time() - t:.0f}s")
    P = {name: np.mean(o, axis=0) for name, o in oofs.items()}

    print("\n  per-repeat mean ± sd (OOF, field-averaged); weightedF1 is the LB metric:")
    rules = {}
    if ref.geo_cols:
        rules["rule_geo"] = rule_geo(ref.X_tr)
        if "v1+coords" in arms:
            rules["rule_geo+c"] = rule_geo(arms["v1+coords"][0].X_tr)
    for name, Pr in rules.items():
        print_scores(name, [scores(y, Pr)])
    for name, o in oofs.items():
        print_scores(name, [scores(y, Pp) for Pp in o])
    n = len(y)
    print(f"  accuracy SE at n={n}: ~{np.sqrt(0.96 * 0.04 / n):.4f} at acc 0.96; one train row = {1 / n:.4f}")

    print("\n  repeat-averaged OOF, pairwise:")
    for a in P:
        for b in P:
            if list(P).index(a) > list(P).index(b):
                print(f"    {a:8s} vs {b:8s} {paired(y, P[a], P[b])}")
    if not smoke:
        for name, ref_s in REPRODUCE.items():
            if name not in P:
                continue
            s1 = scores(y, P[name])
            same = abs(s1["acc"] - ref_s["acc"]) < 6e-5 and abs(s1["weightedF1"] - ref_s["weightedF1"]) < 6e-5
            print(f"  plumbing: {name} arm acc {s1['acc']:.4f} / weightedF1 {s1['weightedF1']:.4f} vs recorded "
                  f"{ref_s['acc']} / {ref_s['weightedF1']} -> {'REPRODUCED' if same else 'DIFFERENT -- inputs changed?'}")

    grp = arms[list(arms)[-1]][0].train["geo_source"].to_numpy()
    sub = np.array([not (g in ("own", "none")) for g in grp])
    if sub.any():
        print(f"\n  OOF on the {sub.sum()} train rows whose geography came from their group "
              f"(no coordinates of their own):")
        for name in P:
            ok = (P[name].argmax(1) == y)[sub]
            print(f"    {name:8s} correct {ok.sum()}/{sub.sum()}")

    print("\n  ship rule:")
    ship = choose_ship(y, P)
    if SHIP_OVERRIDE is not None and SHIP_OVERRIDE in P and SHIP_OVERRIDE != ship:
        print(f"  -> OVERRIDE: shipping {SHIP_OVERRIDE} instead of {ship} (SHIP_OVERRIDE, see its comment)")
        ship = SHIP_OVERRIDE
    fs_ship, cols_ship = arms[ship]
    print(f"\n  shipped arm: {ship} -- repeat-averaged OOF:")
    for k, v in scores(y, P[ship]).items():
        print(f"    {k:11s} {v:.4f}")
    print_confusion(y, P[ship].argmax(1))

    topo_label_table(fs_ship, y)

    base = GATE_BASELINE if GATE_BASELINE in P and GATE_BASELINE != ship else "v1"
    pv1, ps = P[base].argmax(1), P[ship].argmax(1)
    changed = np.where(pv1 != ps)[0]
    print(f"\n  OOF rows where {ship} and {base} disagree: {len(changed)}")
    for i in changed:
        tag = "FIXED " if ps[i] == y[i] else ("BROKEN" if pv1[i] == y[i] else "both wrong")
        print(f"    {tag} true {CLASS_NAMES[y[i]]:16s} {base}={CLASS_NAMES[pv1[i]][:8]:8s} {ship}={CLASS_NAMES[ps[i]][:8]:8s} "
              f"p={np.round(P[ship][i], 2).tolist()}  {row_view(fs_ship, i, 'train')}")

    wrong = np.where(ps != y)[0]
    print(f"\n  {ship} OOF errors: {len(wrong)} rows")
    for i in wrong[:40]:
        print(f"    true {CLASS_NAMES[y[i]]:16s} p={np.round(P[ship][i], 2).tolist()}  {row_view(fs_ship, i, 'train')}")

    P_te = refit_predict(fs_ship, cols_ship, y, n_seeds, iters, importance=True)
    pred, hit = apply_lookup(fs_ship, P_te.argmax(1))
    print(f"\n  lookup: {hit.sum()} test rows are fields seen in train; "
          f"model agrees with the train label on {(P_te.argmax(1)[hit] == pred[hit]).sum()} of them")
    preds = {ship: pred}
    for name, (fs, cols) in arms.items():
        if name != ship:
            preds[name] = apply_lookup(fs, refit_predict(fs, cols, y, n_seeds, iters, importance=False).argmax(1))[0]

    changes = []
    for other in [a for a in arms if a != ship]:
        moved = np.where(pred != preds[other])[0]
        print(f"\n  test rows whose prediction changed vs {other}: {len(moved)} "
              f"(each public row is worth ~0.011 of weighted-F1)")
        for i in moved:
            print(f"    idx {i:3d}  {other}={CLASS_NAMES[preds[other][i]][:8]:8s} -> {ship}={CLASS_NAMES[pred[i]][:8]:8s} "
                  f"p={np.round(P_te[i], 2).tolist()}  {row_view(fs_ship, i, 'test')}")
            changes.append({"vs": other, ID: i, GROUP: fs_ship.test[GROUP].iat[i], "before": CLASS_NAMES[preds[other][i]],
                            "after": CLASS_NAMES[pred[i]], "geo_source": fs_ship.test["geo_source"].iat[i]})
    no_coords = np.where(fs_ship.X_te["Latitude"].isna().to_numpy() & ~hit)[0]
    print(f"\n  test rows WITHOUT coordinates of their own (and not looked up): {len(no_coords)}")
    for i in no_coords:
        print(f"    idx {i:3d}  {CLASS_NAMES[pred[i]][:8]:8s} p={np.round(P_te[i], 2).tolist()}  {row_view(fs_ship, i, 'test')}")

    for name, pr in preds.items():
        print(f"  predicted test counts {name:8s}: {dict(zip(CLASS_NAMES, np.bincount(pr, minlength=N_CLASSES).tolist()))}")
    write_outputs(fs_ship, y, P[ship], P_te, pred, ship, pd.DataFrame(changes))
    for name, pr in preds.items():
        write_submission(pr, get_output_dir() / f"submission_{name}.csv")
    print(f"  wrote submission_<arm>.csv for {list(preds)}; submission.csv = {ship}")
    print(f"\n  pipeline {'(smoke) ' if smoke else ''}took {time.time() - t0:.0f}s")


def write_outputs(fs, y, P_oof, P_te, pred, ship, changes: pd.DataFrame) -> None:
    out = get_output_dir()
    tag = f"{VERSION.lower()}_{ship.replace('+', '_')}"
    write_submission(pred, out / "submission.csv")

    oof = pd.DataFrame({"row": np.arange(len(fs.train)), GROUP: fs.train[GROUP], "y": y,
                        "coord_source": fs.train["coord_source"]})
    for k, c in enumerate(CLASS_NAMES):
        oof[f"p_{c}"] = P_oof[:, k]
    oof["pred"] = P_oof.argmax(1)
    oof.to_csv(out / f"oof_{tag}.csv", index=False)

    te = pd.DataFrame({ID: np.arange(len(fs.test)), GROUP: fs.test[GROUP], "coord_source": fs.test["coord_source"],
                       "geo_source": fs.test["geo_source"]})
    for k, c in enumerate(CLASS_NAMES):
        te[f"p_{c}"] = P_te[:, k]
    te["pred"] = pred
    te.to_csv(out / f"test_proba_{tag}.csv", index=False)
    changes.to_csv(out / f"test_changes_{tag}.csv", index=False)
    print(f"\n  wrote submission.csv, oof_{tag}.csv, test_proba_{tag}.csv, test_changes_{tag}.csv to {out}")


def write_submission(pred: np.ndarray, path) -> None:
    sub = pd.DataFrame({ID: np.arange(len(pred)), TARGET: pred.astype(int)})
    assert len(sub) == 133
    assert list(sub.columns) == [ID, TARGET]
    assert (sub[ID].to_numpy() == np.arange(len(sub))).all()
    assert sub[TARGET].notna().all() and set(sub[TARGET].unique()) <= set(LABELS.values())
    sub.to_csv(path, index=False)


def make_arms(fs_v1, fs_ext, fs_grp) -> dict:
    base_geo = fs_v1.base_cols + fs_v1.geo_cols
    arms = {"v1": (fs_v1, base_geo)}
    if fs_ext is not None and fs_ext.topo_cols:
        arms["v2_full"] = (fs_ext, base_geo + fs_ext.topo_cols)
        arms["v2.1"] = (fs_grp, base_geo + fs_grp.topo_cols)
    return arms


def _run() -> None:
    global AUTO_SMOKE
    print(f"{VERSION}")
    train, test = load_raw()
    globe = load_land_mask()
    ext_coords, rings = load_external()
    topo = topo_table(rings) if rings is not None else None
    fs_v1 = build_features(train, test, globe, None, topo, "v1 coordinates")
    fs_ext = fs_grp = None
    if ext_coords is not None:
        fs_ext = build_features(train, test, globe, ext_coords, topo, "external coordinates")
        fs_grp = build_features(train, test, globe, ext_coords, topo, "external coordinates + group geography",
                                group_impute=True)
    arms = make_arms(fs_v1, fs_ext, fs_grp)
    print("\narms: " + ", ".join(f"{k} ({len(c)} features)" for k, (_, c) in arms.items()))

    if AUTO_SMOKE:
        print("########## SMOKE ##########")
        pipeline(arms, smoke=True)
        AUTO_SMOKE = False
        print("########## SMOKE OK -- real run follows ##########")
    pipeline(arms, smoke=False)


def main() -> None:

    real = sys.stdout, sys.stderr
    with open(get_output_dir() / RUN_LOG, "w", buffering=1) as log:
        sys.stdout, sys.stderr = _Tee(real[0], log), _Tee(real[1], log)
        try:
            _run()
        except BaseException:
            import traceback
            traceback.print_exc(file=log)
            raise
        finally:
            sys.stdout, sys.stderr = real

if __name__ == "__main__":
    main()
