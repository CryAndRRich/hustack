import itertools
import json
import shutil
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from models import dissim_fit, dissim_select, et_arm, gin_arm, kernel_arm, kernel_select, lgb_arm, svm_feat_arm
from preprocess import N_CLASSES, OUT, WORK, load, log

warnings.filterwarnings("ignore", message="X does not have valid feature names")
N_SPLITS = 5
N_REPEATS = 3
BLEND_TOP = 4
BLEND_STEPS = 10


def macro_f1(y, P):
    return f1_score(y, P.argmax(1), average="macro")


def balanced_labels(P, quota):
    slots = np.repeat(np.arange(N_CLASSES), quota)
    rows, cols = linear_sum_assignment(-np.log(np.clip(P, 1e-12, 1))[:, slots])
    out = np.empty(len(P), dtype=np.int64)
    out[rows] = slots[cols]
    return out


def balanced_oof_f1(P, y, folds):
    pred = np.empty(len(y), dtype=np.int64)
    for va in folds:
        pred[va] = balanced_labels(P[va], np.bincount(y[va], minlength=N_CLASSES))
    return f1_score(y, pred, average="macro")


def choose_blend(oof, y, folds, names):
    best = None
    for w in itertools.product(range(BLEND_STEPS + 1), repeat=len(names)):
        if sum(w) != BLEND_STEPS:
            continue
        f = balanced_oof_f1(sum(wi / BLEND_STEPS * oof[a][0] for wi, a in zip(w, names)), y, folds[0])
        if best is None or f > best[0]:
            best = (f, {a: wi / BLEND_STEPS for wi, a in zip(w, names) if wi})
    return best


def scores(P, y, folds):
    return ([macro_f1(y, P[r]) for r in range(N_REPEATS)],
            [balanced_oof_f1(P[r], y, folds[r]) for r in range(N_REPEATS)])


def run():
    t0 = time.time()
    graphs, y, F, K, D_wwl, D_fgw, sub = load()
    n_tr = len(y)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=0)
    splits = list(rskf.split(np.zeros(n_tr), y))
    folds = [[va for _, va in splits[r * N_SPLITS:(r + 1) * N_SPLITS]] for r in range(N_REPEATS)]
    sel_folds = splits[:N_SPLITS]
    kern, dist, selected = {}, {}, {}
    for name, Ds in (("wwl", D_wwl), ("fgw", D_fgw)):
        scales = {k: np.median(D[:n_tr, :n_tr][D[:n_tr, :n_tr] > 0]) for k, D in Ds.items()}
        f1_sel, key, gamma, C = kernel_select({k: D[:n_tr, :n_tr] for k, D in Ds.items()}, scales, y, sel_folds, name)
        kern[name] = np.exp(-gamma * Ds[key] / scales[key])
        dist[name] = Ds[key] / scales[key]
        selected[name] = {"key": key, "gamma": gamma, "C": C, "select_f1_rep0": float(f1_sel)}
        log(f"{name} selection on repeat 0: {key}, gamma {gamma}, C {C}, macro F1 {f1_sel:.4f}  [{time.time() - t0:.0f}s]")
    f1_sel, C_dis = dissim_select([dist["fgw"][:n_tr, :n_tr], dist["wwl"][:n_tr, :n_tr]], y, sel_folds)
    selected["dissim"] = {"C": C_dis, "select_f1_rep0": float(f1_sel)}
    log(f"dissim selection on repeat 0: C {C_dis}, macro F1 {f1_sel:.4f}  [{time.time() - t0:.0f}s]")
    arms = {"et": lambda tr, s: et_arm(F, y, tr, s),
            "lgb": lambda tr, s: lgb_arm(F, y, tr, s),
            "svm_feat": lambda tr, s: svm_feat_arm(F, y, tr, s),
            "wl": lambda tr, s: kernel_arm(K, y, tr, s),
            "wwl": lambda tr, s: kernel_arm(kern["wwl"], y, tr, s, C=selected["wwl"]["C"]),
            "fgw": lambda tr, s: kernel_arm(kern["fgw"], y, tr, s, C=selected["fgw"]["C"]),
            "dissim": lambda tr, s: dissim_fit([dist["fgw"], dist["wwl"]], y, tr, C_dis),
            "gin": lambda tr, s: gin_arm(graphs, y, tr, s)}
    n_te = len(graphs) - n_tr
    assert n_te % N_CLASSES == 0
    test_quota = np.full(N_CLASSES, n_te // N_CLASSES)
    oof = {a: np.zeros((N_REPEATS, n_tr, N_CLASSES)) for a in arms}
    test = {a: np.zeros((n_te, N_CLASSES)) for a in arms}
    for k, (tr, va) in enumerate(splits):
        rep = k // N_SPLITS
        line = []
        for a, fit in arms.items():
            P = fit(tr, k)
            assert P.shape == (len(graphs), N_CLASSES)
            oof[a][rep, va] = P[va]
            test[a] += P[n_tr:] / (N_SPLITS * N_REPEATS)
            line.append(f"{a} {macro_f1(y[va], P[va]):.3f}")
        log(f"fold {k:2d} (rep {rep}): " + ", ".join(line) + f"  [{time.time() - t0:.0f}s]")

    summary = {}
    for a in arms:
        am, bal = scores(oof[a], y, folds)
        summary[a] = {"argmax": am, "balanced": bal}
        log(f"OOF macro F1 {a:9s}: argmax {np.mean(am):.4f} ± {np.std(am):.4f}, balanced {np.round(bal, 4).tolist()}, "
            f"clean (rep 1-2) balanced {np.mean(bal[1:]):.4f}")
    top = sorted(arms, key=lambda a: -summary[a]["balanced"][0])[:BLEND_TOP]
    f1_sel, w = choose_blend(oof, y, folds, top)
    log(f"blend chosen on repeat 0 from {top}: " + ", ".join(f"{a} {v:.1f}" for a, v in w.items())
        + f", balanced F1 {f1_sel:.4f}")
    blend_oof = sum(v * oof[a] for a, v in w.items())
    blend_test = sum(v * test[a] for a, v in w.items())
    am, bal = scores(blend_oof, y, folds)
    summary["blend"] = {"weights": w, "argmax": am, "balanced": bal}
    gain = np.subtract(bal, summary["fgw"]["balanced"])[1:]
    log(f"OOF macro F1 blend    : argmax {np.round(am, 4).tolist()}, balanced {np.round(bal, 4).tolist()}; "
        f"clean (rep 1-2) balanced {np.mean(bal[1:]):.4f}, gain over fgw alone {np.round(gain, 4).tolist()}")
    summary["selection"] = selected

    extra = WORK / "extra"
    extra.mkdir(parents=True, exist_ok=True)
    argmax_pred = blend_test.argmax(1)
    pred = balanced_labels(blend_test, test_quota)
    assert (np.bincount(pred, minlength=N_CLASSES) == test_quota).all()
    log(f"test: argmax class counts {np.bincount(argmax_pred, minlength=N_CLASSES).tolist()}, "
        f"balanced changes {int((pred != argmax_pred).sum())} of {n_te} labels")
    for path, p in ((OUT / "submission.csv", pred), (extra / "submission_argmax.csv", argmax_pred)):
        out = pd.DataFrame({"graph_id": sub.graph_id, "label": p + 1})
        assert len(out) == len(sub) and list(out.graph_id) == list(sub.graph_id)
        assert out.label.between(1, N_CLASSES).all()
        out.to_csv(path, index=False)
    np.savez_compressed(extra / "oof.npz", y=y, **{f"oof_{a}": oof[a] for a in arms})
    for a in arms:
        pd.DataFrame(test[a], columns=[f"p{c + 1}" for c in range(N_CLASSES)]).assign(graph_id=sub.graph_id) \
            .to_csv(extra / f"test_{a}.csv", index=False)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"done in {time.time() - t0:.0f}s")
    log(f"SUBMIT: {OUT / 'submission.csv'}")


def write_results():
    with zipfile.ZipFile(OUT / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(WORK.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(WORK))


if __name__ == "__main__":
    shutil.rmtree(WORK, ignore_errors=True)
    try:
        run()
    finally:
        write_results()
