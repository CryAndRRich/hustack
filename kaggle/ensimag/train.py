import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from families import FAMILIES, blend, fit_blend_weights, fit_family
from preprocess import OK_LIMIT, build_features, get_data_dir, get_ext_dir, load_data, load_external

VERSION = "v5.1"
AUTO_SMOKE = True
POS_MW = 50
FOLDS = [("2022", 2021, 2022), ("2023", 2022, 2023)]
TEST_LIKE_MONTHS = 8
GATE_T = {"2022": 2.0, "2023": 2.0}
SELECT_FOLD = "2023"
JAN22 = ("2022-01-01", "2022-01-31")

V4_SPEC = dict(groups=["v1", "error", "error2"], exclude=JAN22)
ARMS = {("v4" if fam == "lgbm" else fam): dict(V4_SPEC, family=fam) for fam in FAMILIES}


TREE_ARMS = ["v4", "xgb", "cat", "hgb", "forest"]
BLENDS = {"blend_all": dict(members=list(ARMS), weights="equal"),
          "blend_trees": dict(members=TREE_ARMS, weights="equal"),
          "blend_fit": dict(members=list(ARMS), weights="fit")}
EXTRA_SUBMISSION = "blend_all"
INCUMBENT = "v4"
REFERENCE = ("ref_oof_v4.csv", "v4_nojan22")
REF_MIN_SIGN_AGREE = 0.99
FULL = dict(num_boost_round=400, learning_rate=0.03, xgb_rounds=600, cat_iters=800, hgb_iters=400, forest_trees=200,
            mlp_epochs=8, kernel_comp=500, linear_iter=1000)
SMOKE = dict(num_boost_round=5, learning_rate=0.3, xgb_rounds=5, cat_iters=5, hgb_iters=5, forest_trees=4,
             mlp_epochs=1, kernel_comp=20, linear_iter=20)
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def pnl(forecast, spread, ok):
    return np.where(ok, np.where(forecast >= 0, 1.0, -1.0) * spread * POS_MW, 0.0)


def make_arms(groups, ldate):

    arms = {}
    for a, spec in ARMS.items():
        feats = [f for gname in spec["groups"] for f in groups[gname]]
        drop = spec.get("drop", [])
        assert set(drop) <= set(feats), f"{a}: drops columns that are not features: {set(drop) - set(feats)}"
        feats = [f for f in feats if f not in drop]
        assert len(feats) == len(set(feats)), f"{a}: duplicated features"
        keep = np.ones(len(ldate), dtype=bool)
        if "exclude" in spec:
            lo, hi = (pd.Timestamp(d) for d in spec["exclude"])
            keep = ~((ldate >= lo) & (ldate <= hi))
        arms[a] = (feats, keep, spec["family"])
        print(f"arm {a} ({spec['family']}): {len(feats)} features, {int((~keep).sum())} train rows excluded")
    return arms


def score_rows(forecast, spread, ok, days, inc_forecast):
    p = pnl(forecast, spread, ok)
    perfect = POS_MW * np.abs(spread[ok]).sum()
    daily = pd.Series(p).groupby(days).sum()
    d = pd.Series(p - pnl(inc_forecast, spread, ok)).groupby(days).sum()
    n = len(daily)
    t = d.sum() / (d.std(ddof=1) * np.sqrt(n)) if d.std(ddof=1) > 0 else 0.0
    return {"pnl_meur": p.sum() / 1e6, "capture": p.sum() / perfect,
            "capture_sd": daily.std(ddof=1) * np.sqrt(n) / perfect, "t_vs_incumbent": float(t),
            "long_share": float((forecast >= 0).mean())}


def fit_arm(arm, arms, x_fit, spread_fit, x_pred, cfg, failed, seconds, label):
    feats, _, family = arms[arm]
    t0 = time.time()
    try:
        pred, imp = fit_family(family, x_fit, spread_fit, x_pred, feats, cfg)
        assert np.isfinite(pred).all(), f"{arm}: non-finite forecast"
    except Exception:
        print(f"   {arm}: {label} FAILED — dropped\n{traceback.format_exc()}")
        failed.add(arm)
        return None, None
    seconds[arm] = seconds.get(arm, 0.0) + time.time() - t0
    print(f"   {arm} ({family}, {len(feats)} features) {label}: {time.time() - t0:.1f}s, long share {(pred >= 0).mean():.3f}")
    return pred, imp


def add_blends(members, oof, test_pred, spread, ok, years):

    weights = {}
    for name, spec in BLENDS.items():
        mem = [m for m in spec["members"] if m in members]
        if len(mem) < 2:
            print(f"   {name}: fewer than 2 surviving members — skipped")
            continue
        prev_rows, fold_w = None, {}
        for fold, _, score_year in FOLDS:
            rows = years == score_year
            w = None
            if spec["weights"] == "fit" and prev_rows is not None:
                fit_rows = prev_rows & ok
                w = fit_blend_weights({m: oof.loc[fit_rows, m].values for m in mem}, mem, spread[fit_rows])
            fold_w[fold] = w
            oof.loc[rows, name] = blend({m: oof.loc[rows, m].values for m in mem}, mem, w)
            prev_rows = rows
        w = None
        if spec["weights"] == "fit":
            fit_rows = oof["fold"].ne("").values & ok
            w = fit_blend_weights({m: oof.loc[fit_rows, m].values for m in mem}, mem, spread[fit_rows])
        fold_w["final"] = w
        test_pred[name] = blend(test_pred, mem, w)
        weights[name] = fold_w
        if spec["weights"] == "fit":
            print(f"   {name} weights:", {k: None if v is None else {m: round(x, 3) for m, x in v.items()} for k, v in fold_w.items()})
    return weights


def validate(arms, oof, spread, ok, years, meta):
    rows, days = [], meta["ldate"].values
    for fold, _, score_year in FOLDS:
        for label, m in ((fold, years == score_year),
                         (f"{fold}_jan-aug", (years == score_year) & (meta["month"].values <= TEST_LIKE_MONTHS))):
            for arm in arms:
                r = score_rows(oof.loc[m, arm].values, spread[m], ok[m], days[m], oof.loc[m, INCUMBENT].values)
                rows.append({"arm": arm, "eval": label, **r})
    return pd.DataFrame(rows)


def passes(res, arm):
    t = res[res["arm"] == arm].set_index("eval")["t_vs_incumbent"]
    return all(fold in t.index and t[fold] > GATE_T[fold] for fold, _, _ in FOLDS)


def fold_capture(res, arm, fold):
    return float(res[(res["arm"] == arm) & (res["eval"] == fold)]["capture"].iloc[0])


def check_reference(oof, ref_path, ref_col):
    if not ref_path.exists():
        print(f"reference check SKIPPED: {ref_path.name} not in the dataset")
        return
    ref = pd.read_csv(ref_path, parse_dates=["date"], usecols=["date", ref_col]).rename(columns={ref_col: "released"})
    j = oof[["date", INCUMBENT]].merge(ref, on="date", how="inner").dropna()
    agree = float(((j[INCUMBENT] >= 0) == (j["released"] >= 0)).mean())
    diff = float(np.abs(j[INCUMBENT] - j["released"]).max())
    print(f"reference check vs the released {INCUMBENT} OOF: {len(j)} rows, sign agreement {agree:.5f}, max |diff| {diff:.2e}")
    assert agree >= REF_MIN_SIGN_AGREE, f"the {INCUMBENT} arm does not reproduce its release — the pipeline changed"


def write_submission(path, test, sample, forecast):
    sub = pd.DataFrame({"ID": test["ID"].values, "forecast": forecast})
    assert len(sub) == len(sample), "row count differs from sample.csv"
    assert list(sub.columns) == list(sample.columns), "columns differ from sample.csv"
    assert (sub["ID"].values == sample["ID"].values).all(), "ID order differs from sample.csv"
    assert np.isfinite(sub["forecast"].values).all(), "non-finite forecast"
    sub.to_csv(path, index=False)


def run(cfg, out_dir: Path, tag: str):
    t_run = time.time()
    print(f"\n========== {tag} run {VERSION}: {cfg} ==========")
    train, test, sample = load_data(get_data_dir())
    ext_dir = get_ext_dir()
    x_tr, x_te, meta, groups = build_features(train, test, load_external(ext_dir))
    spread = train["spread"].values
    ok = np.abs(train["imbalances"].values) <= OK_LIMIT
    years = meta["year"].values
    print(f"scored train rows (|imb| <= {OK_LIMIT:g}): {ok.sum()} of {len(ok)}")

    oof = pd.DataFrame({"date": meta["date"], "spread": spread, "imbalances": train["imbalances"].values,
                        "ok": ok, "fold": ""})
    failed, seconds = set(), {}
    arms = make_arms(groups, pd.DatetimeIndex(meta["ldate"]))
    for fold, fit_to, score_year in FOLDS:
        smask = years == score_year
        oof.loc[smask, "fold"] = fold
        print(f"\n-- fold {fold}: fit ≤ {fit_to}, score {score_year} ({smask.sum()} rows)")
        for arm, (_, keep, _) in arms.items():
            if arm not in failed:
                fm = ok & keep & (years <= fit_to)
                pred, _ = fit_arm(arm, arms, x_tr[fm], spread[fm], x_tr[smask], cfg, failed, seconds, f"fold {fold}")
                if pred is not None:
                    oof.loc[smask, arm] = pred
    if INCUMBENT in failed:
        raise RuntimeError(f"the incumbent {INCUMBENT} failed — nothing to gate against")
    if tag == "FULL":
        check_reference(oof, ext_dir / REFERENCE[0], REFERENCE[1])

    print("\n-- final fit on all scored train rows")
    test_pred, imps = {}, {}
    for arm, (_, keep, _) in arms.items():
        if arm not in failed:
            pred, imp = fit_arm(arm, arms, x_tr[ok & keep], spread[ok & keep], x_te, cfg, failed, seconds, "final")
            if pred is not None:
                test_pred[arm] = pred
                if imp is not None:
                    imps[arm] = imp
    if INCUMBENT in failed:
        raise RuntimeError(f"the incumbent {INCUMBENT} failed in the final fit")
    members = [a for a in arms if a not in failed]
    test_pred = {a: p for a, p in test_pred.items() if a in members}
    print(f"\n-- blends over {members}")
    blend_weights = add_blends(members, oof, test_pred, spread, ok, years)

    alive = members + [b for b in BLENDS if b in test_pred]
    res = validate(alive, oof, spread, ok, years, meta)
    passing = [a for a in alive if a != INCUMBENT and passes(res, a)]
    chosen = max(passing, key=lambda a: fold_capture(res, a, SELECT_FOLD)) if passing else INCUMBENT
    print(f"\n## gate: t vs {INCUMBENT} above {GATE_T} per fold -> passing {passing}; chosen: {chosen}")
    if chosen == INCUMBENT:
        print(f"   {VERSION} submission = {INCUMBENT} re-fit — the release's arm, not a bit-identical file (FINDINGS §18)")
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 200):
        print(f"\n## validation (capture = PnL / perfect-foresight PnL on scored rows; t = paired day-block vs {INCUMBENT})")
        print(res.round(4).to_string(index=False))
    tp = pd.DataFrame(test_pred)[alive]
    print("\ntest sign agreement with chosen arm:", {a: round(float(((tp[a] >= 0) == (tp[chosen] >= 0)).mean()), 3) for a in tp})
    corr = oof.loc[oof["fold"].ne("") & oof["ok"], members].corr(method="spearman")
    print("OOF Spearman correlation between family forecasts:\n" + corr.round(2).to_string())

    extra = out_dir / "extra"
    extra.mkdir(parents=True, exist_ok=True)
    path = out_dir / "submission.csv"
    write_submission(path, test, sample, test_pred[chosen])
    if EXTRA_SUBMISSION in test_pred:
        write_submission(extra / f"submission_{EXTRA_SUBMISSION}.csv", test, sample, test_pred[EXTRA_SUBMISSION])
    oof.to_csv(extra / f"oof_{VERSION}.csv", index=False)
    tp.insert(0, "ID", test["ID"].values)
    tp.to_csv(extra / f"test_arms_{VERSION}.csv", index=False)
    res.to_csv(extra / f"validation_{VERSION}.csv", index=False)
    pd.DataFrame(imps).to_csv(extra / f"importance_{VERSION}.csv")
    json.dump({"version": VERSION, "cfg": cfg, "chosen": chosen, "passing": passing, "failed": sorted(failed),
               "seconds": seconds, "blend_weights": blend_weights,
               f"capture_{SELECT_FOLD}": {a: fold_capture(res, a, SELECT_FOLD) for a in alive}},
              open(extra / f"summary_{VERSION}.json", "w"), indent=1)
    print(f"\n{tag} run done in {time.time() - t_run:.0f}s")
    return path, chosen


def main():
    if AUTO_SMOKE:
        smoke_dir = WORK / "smoke_tmp"
        run(SMOKE, smoke_dir, "SMOKE")
        shutil.rmtree(smoke_dir, ignore_errors=True)
        print("smoke pass OK — outputs discarded")
    path, chosen = run(FULL, WORK, "FULL")
    print(f"\n{VERSION} chosen arm: {chosen}")
    print(f"SUBMIT: {path}")


if __name__ == "__main__":
    main()
