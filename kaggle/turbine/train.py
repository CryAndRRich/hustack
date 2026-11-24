import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

import adversarial
import catalog
import config
import data
import evaluate
import models
import runtime
import submit
from runtime import log, section, step

CODE_DIR = "/tmp/code"


def holdout_split(t):
    days = sorted(set(t["d"]))
    if len(days) < 2:
        raise RuntimeError("need at least two distinct `d` values for a time holdout, got %s" % days)
    val_days = set(days[-min(config.VAL_DAYS, len(days) // 2):])
    va = np.array([x in val_days for x in t["d"]])
    val_ids = set(t["ids"][va])
    tr = ~va & np.array([x not in val_ids for x in t["ids"]])
    log("holdout: days %s..%s (%d rows); training days %s..%s (%d rows, %d dropped as shared users)"
        % (min(val_days), max(val_days), int(va.sum()), days[0], days[-len(val_days) - 1] if len(days) > len(val_days) else "-",
           int(tr.sum()), int((~va & ~tr).sum())))
    return np.flatnonzero(tr), np.flatnonzero(va)


def reorder(t, order):
    for k in ("X", "ids", "inst", "rev", "d", "sw"):
        t[k] = t[k][order]


def pipeline(smoke):
    tag = "SMOKE" if smoke else "FULL"
    work = os.path.join(config.WORK, "smoke") if smoke else config.WORK
    os.makedirs(os.path.join(work, "extra"), exist_ok=True)
    rounds = config.SMOKE_ROUNDS if smoke else config.MAX_ROUNDS
    files = runtime.find_inputs()

    section("%s 1 -- data" % tag)
    dia = {k: runtime.dialect(files[k]) for k in ("train", "cand", "cat_tr", "cat_ca")}
    cats, code = catalog.load_mapping(files["mapping"])
    t = data.load_training(files["train"], dia["train"], cats, smoke)
    cand = data.load_candidates(files["cand"], dia["cand"], t["spec"], smoke)
    cat_names = catalog.feature_names(cats)
    keys_tr = pd.DataFrame({"hashed_aaid": t["ids"],
                            "date": (pd.to_datetime(pd.Series(t["d"])) - pd.Timedelta(days=4)).dt.strftime("%Y-%m-%d")})
    t["X"] = np.hstack([t["X"], catalog.attach(keys_tr, files["cat_tr"], dia["cat_tr"], code, len(cats), True)])
    cand["X"] = np.hstack([cand["X"], catalog.attach(pd.DataFrame({"hashed_aaid": cand["ids"]}), files["cat_ca"],
                                                      dia["cat_ca"], code, len(cats), False)])
    names = t["names"] + cat_names
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate feature names")
    if t["X"].shape[1] != cand["X"].shape[1] or t["X"].shape[1] != len(names):
        raise RuntimeError("feature matrices disagree: %s vs %s vs %d names"
                           % (t["X"].shape, cand["X"].shape, len(names)))
    step("matrices: training %s, candidates %s" % (t["X"].shape, cand["X"].shape))
    k = submit.k_for(cand)
    frac = k / len(cand["ids"])

    section("%s 2 -- holdout and shift" % tag)
    tr_idx, va_idx = holdout_split(t)
    reorder(t, np.concatenate([tr_idx, va_idx]))
    n_tr = len(tr_idx)
    t["iw"], adv = adversarial.fit(t["X"], t["sw"], cand["X"], names, min(rounds, config.ADV_ROUNDS))

    section("%s 3 -- holdout fits" % tag)
    best, val_pred = {}, {}
    for name in models.MODELS:
        model, best[name] = models.fit_holdout(name, t["X"], t, n_tr, names, rounds)
        val_pred[name] = model.predict(t["X"][n_tr:], num_iteration=best[name])
        del model
    va = slice(n_tr, None)
    scores = models.arm_scores(val_pred)
    rev_va, sw_va, iw_va = t["rev"][va], t["sw"][va], t["iw"][va]
    plain = evaluate.table(scores, rev_va, sw_va, frac)
    shifted = evaluate.table(scores, rev_va, sw_va * iw_va, frac)
    log("\nholdout capture at frac=%.5f (%d holdout rows, %d payers, %.2f USD)"
        % (frac, len(rev_va), int((rev_va > 0).sum()), rev_va.sum()))
    log("%-12s %10s %10s" % ("arm", "plain", "shifted"))
    for arm in list(scores) + ["oracle", "random"]:
        log("%-12s %10.4f %10.4f" % (arm, plain[arm], shifted[arm]))
    chosen, wins = evaluate.gate(scores, rev_va, sw_va * iw_va, frac, config.CANDIDATE_ARM,
                                 50 if smoke else config.BOOTSTRAP, config.GATE_WIN, config.SEED)
    log("gate: share of resamples beating %s: %s -> submit %s"
        % (config.CANDIDATE_ARM, {a: round(s, 3) for a, s in wins.items()}, chosen))

    section("%s 4 -- refit on all rows and score candidates" % tag)
    cand_pred, imp = {}, {}
    for name in models.MODELS:
        model = models.fit_full(name, t["X"], t, names, max(1, int(round(best[name] * config.REFIT_FACTOR))))
        cand_pred[name] = model.predict(cand["X"])
        imp[name] = models.importance(model, names)
        if name == config.CANDIDATE_ARM:
            models.log_top(model, names, "top features of the %s refit (gain):" % name)
        del model
    cand_scores = models.arm_scores(cand_pred)

    section("%s 5 -- files" % tag)
    forced = submit.forced_mask(cand)
    for arm, s in cand_scores.items():
        submit.write(os.path.join(work, "extra", "submission_%s.csv" % arm), submit.top_ids(cand["ids"], s, k), cand["ids"], k)
        submit.write(os.path.join(work, "extra", "submission_%s_forced598.csv" % arm),
                     submit.top_ids(cand["ids"], s, k, forced), cand["ids"], k)
    overlap = {a: len(set(submit.top_ids(cand["ids"], s, k)) & set(submit.top_ids(cand["ids"], cand_scores[chosen], k))) / k
               for a, s in cand_scores.items()}
    log("overlap of each arm's top-%d with the submitted arm: %s" % (k, {a: round(v, 3) for a, v in overlap.items()}))
    log("candidates with ad_revenue inside the submitted top-%d: %d of %d"
        % (k, int(np.isin(np.flatnonzero(forced), np.argsort(-cand_scores[chosen])[:k]).sum()), int(forced.sum())))

    np.savez_compressed(os.path.join(work, "holdout_predictions.npz"), ids=t["ids"][va].astype(str), rev=rev_va,
                        sw=sw_va, iw=iw_va, inst=t["inst"][va], **{"score_" + a: s for a, s in scores.items()})
    pd.DataFrame({n: dict(imp[n]) for n in imp}).reindex(names).to_csv(os.path.join(work, "feature_importance.csv"))
    result = {"tag": tag, "frac": frac, "k": k, "n_train_rows": n_tr, "n_holdout_rows": len(rev_va),
              "n_features": len(names), "best_iteration": best, "capture_plain": plain, "capture_shifted": shifted,
              "gate_wins": wins, "submitted_arm": chosen, "adversarial": adv, "overlap_with_submitted": overlap,
              "config": {k_: v for k_, v in vars(config).items() if k_.isupper()}}
    with open(os.path.join(work, "result.json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    sub_src = os.path.join(work, "extra", "submission_%s_forced598.csv" % chosen)
    if not smoke:
        dst = os.path.join(runtime.out_dir(), "submission.csv")
        shutil.copy(sub_src, dst)
        log("SUBMIT: %s  (arm %s with every candidate holding ad_revenue forced in)" % (dst, chosen))
    return result


def run():
    runtime.init()
    t0 = time.time()
    try:
        pipeline(smoke=True)
        log("\nsmoke pass completed in %.0fs; starting the full run" % (time.time() - t0))
        shutil.rmtree(os.path.join(config.WORK, "smoke"), ignore_errors=True)
        pipeline(smoke=False)
        log("\ntotal %.0fs" % (time.time() - t0))
    finally:
        runtime.finish(CODE_DIR)


if __name__ == "__main__":
    sys.exit(run())
