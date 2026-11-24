import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from arms import ARMS, CANDIDATE, REFERENCE
from blend import fairness, lp_weights, nested_blend, paired_se
from config import Config, smoke
from data import ID, TARGET, features, load
from runtime import log, output_dir, start_log, write_results


def cross_fit(cfg, X, y, Xte, fold):
    oof, test = {}, {}
    for arm, fit in ARMS.items():
        t0 = time.time()
        oof[arm], fold_test = np.zeros(len(y)), np.zeros(len(Xte))
        for f in range(cfg.folds):
            tr = fold != f
            predict = fit(cfg, X[tr].reset_index(drop=True), y[tr])
            oof[arm][~tr] = predict(X[~tr])
            fold_test += predict(Xte) / cfg.folds
        test[arm] = fold_test if arm == REFERENCE else fit(cfg, X, y)(Xte)
        oof[arm], test[arm] = np.clip(oof[arm], cfg.lo, cfg.hi), np.clip(test[arm], cfg.lo, cfg.hi)
        assert np.isfinite(oof[arm]).all() and np.isfinite(test[arm]).all(), arm
        per_fold = [np.abs(y - oof[arm])[fold == f].mean() for f in range(cfg.folds)]
        log(f"{arm:16s} OOF MAE {np.abs(y - oof[arm]).mean():.4f} folds {np.round(per_fold, 3).tolist()}"
            f" ({time.time() - t0:.0f} s)")
    return oof, test


def choose(cfg, y, fold, oof, test):
    maes = {a: float(np.abs(y - oof[a]).mean()) for a in ARMS}
    pairs = {a: paired_se(y - oof[REFERENCE], y - oof[a], cfg.bootstrap, cfg.seed) for a in ARMS if a != REFERENCE}
    for a, (gain, se) in pairs.items():
        log(f"{a:16s} gain vs {REFERENCE} {gain:+.4f} (SE {se:.4f})")
    gain, se = pairs[CANDIDATE]
    best = CANDIDATE if gain > cfg.margin_se * se else REFERENCE
    log(f"pre-registered candidate {CANDIDATE}: {'passes' if best == CANDIDATE else 'fails'} the {cfg.margin_se} SE gate")
    others = [a for a in pairs if a != CANDIDATE and pairs[a][0] > cfg.margin_se * pairs[a][1]]
    log("other arms past the gate (reported, not submitted):", others)
    arms = list(ARMS)
    P, Pte = np.column_stack([oof[a] for a in arms]), np.column_stack([test[a] for a in arms])
    blend_oof = np.clip(nested_blend(P, y, fold), cfg.lo, cfg.hi)
    w = lp_weights(P, y)
    blend_test = np.clip(Pte @ w, cfg.lo, cfg.hi)
    b_gain, b_se = paired_se(y - oof[best], y - blend_oof, cfg.bootstrap, cfg.seed)
    log("blend weights", {a: round(float(x), 3) for a, x in zip(arms, w)})
    log(f"nested blend {np.abs(y - blend_oof).mean():.4f}, gain vs {best} {b_gain:+.4f} (SE {b_se:.4f})")
    pick = "blend" if b_gain > cfg.margin_se * b_se else best
    log("SUBMITTED CHOICE:", pick)
    final_oof, final_test = (blend_oof, blend_test) if pick == "blend" else (oof[best], test[best])
    return pick, final_oof, final_test, {
        "arm_mae": maes, "gain_vs_reference": pairs, "pick": pick, "blend_nested_mae": float(np.abs(y - blend_oof).mean()),
        "blend_gain": b_gain, "blend_gain_se": b_se, "blend_weights": dict(zip(arms, map(float, w)))}


def run_once(cfg, final):
    tr, te, ss = load()
    if cfg.smoke_rows:
        tr = tr.sample(cfg.smoke_rows, random_state=cfg.seed).reset_index(drop=True)
    X, Xte = features(tr, te)
    y = tr[TARGET].to_numpy(float)
    fold = np.zeros(len(y), int)
    skf = StratifiedKFold(cfg.folds, shuffle=True, random_state=cfg.seed)
    for f, (_, va) in enumerate(skf.split(X, tr.region_tier)):
        fold[va] = f
    oof, test = cross_fit(cfg, X, y, Xte, fold)
    if final:
        ref = np.abs(y - oof[REFERENCE]).mean()
        assert abs(ref - cfg.reference_oof_mae) < 5e-4, f"V1 reference does not reproduce: {ref:.4f}"
    summary = {"config": cfg.__dict__}
    pick, final_oof, final_test, summary["choice"] = choose(cfg, y, fold, oof, test)
    summary["fairness"] = fairness(y, final_oof, {c: tr[c] for c in ("region_tier", "vulnerability_group")})
    sub = pd.DataFrame({ID: te[ID], "predicted_resolution_days": np.round(final_test, 3)})
    assert len(sub) == len(ss) and (sub[ID].values == ss[ID].values).all()
    assert sub.predicted_resolution_days.between(cfg.lo, cfg.hi).all() and sub.notna().all().all()
    extra = Path(cfg.work) / ("extra" if final else "smoke")
    extra.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: tr[ID], "y": y, "fold": fold, **oof, "submitted": final_oof}).to_csv(extra / "oof.csv",
                                                                                            index=False)
    pd.DataFrame({ID: te[ID], **test, "submitted": final_test}).to_csv(extra / "test_preds.csv", index=False)
    if final:
        path = output_dir() / "submission.csv"
        sub.to_csv(path, index=False)
        log(f"SUBMIT: {path} ({pick}, OOF MAE {np.abs(y - final_oof).mean():.4f})")
    return summary


def run():
    cfg = Config()
    start_log(cfg.work)
    summary = {}
    try:
        log("== smoke")
        summary["smoke"] = run_once(smoke(cfg), final=False)
        log("== full")
        summary["full"] = run_once(cfg, final=True)
    except Exception:
        summary["error"] = traceback.format_exc()
        log(summary["error"])
        raise
    finally:
        write_results(cfg.work, cfg.code_dir, summary)


if __name__ == "__main__":
    run()
