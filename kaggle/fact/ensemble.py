import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import LABELS, N_EVID_SUBMIT, NEI, OUT, covered, find_data_dir, load_claims, log, tune_offsets

INPUT = Path("/kaggle/input")
PROB_COLS = {"V1": ["p_sup", "p_ref", "p_nei"],
             "V2 epoch1": ["epoch1_sup", "epoch1_ref", "epoch1_nei"],
             "V2 final": ["final_sup", "final_ref", "final_nei"]}
MEANS = {"V1 + V2 epoch1": ["V1", "V2 epoch1"], "V1 + V2 epoch1 + V2 final": ["V1", "V2 epoch1", "V2 final"]}


def find_runs(root):
    runs = {}
    for dirpath, _, files in os.walk(root):
        if {"verifier_probs.parquet", "rerank_dev_test.parquet"} <= set(files):
            cols = pd.read_parquet(Path(dirpath) / "verifier_probs.parquet").columns
            runs["V1" if "p_sup" in cols else "V2"] = Path(dirpath)
    assert set(runs) == {"V1", "V2"}, f"attach the V1 extra dataset and the V2 train notebook output; found {runs}"
    return runs


def prob_sets(runs, claims):
    sets = {}
    for run in ("V1", "V2"):
        p = pd.read_parquet(runs[run] / "verifier_probs.parquet").set_index("qid")
        assert (p["id"] == claims.loc[p.index, "id"]).all(), f"{run}: qid -> id differs from this notebook's claims"
        for name, cols in PROB_COLS.items():
            if name.startswith(run):
                sets[name] = pd.DataFrame(p[cols].to_numpy(np.float64), index=p.index)
    for name, parts in MEANS.items():
        sets[name] = sum(sets[k] for k in parts) / len(parts)
    return sets


def zscore(c):
    g = c.groupby("qid")["score"]
    return (c["score"] - g.transform("mean")) / (g.transform("std").fillna(0.0) + 1e-6)


def rankings(runs):
    c = {}
    for run in ("V1", "V2"):
        d = pd.read_parquet(runs[run] / "rerank_dev_test.parquet", columns=["qid", "title", "sid", "score"])
        assert np.isfinite(d["score"]).all(), f"{run}: unscored dev/test candidates"
        d[run] = zscore(d)
        c[run] = d.drop(columns="score")
    m = c["V1"].merge(c["V2"], on=["qid", "title", "sid"], how="outer")
    log(f"candidates: {len(c['V1'])} V1, {len(c['V2'])} V2, {m[['V1', 'V2']].notna().all(1).sum()} in both")
    m["V1 + V2"] = m[["V1", "V2"]].fillna(-3.0).mean(1)
    tops = {}
    for name in ("V1", "V2", "V1 + V2"):
        d = m.dropna(subset=[name]).sort_values(["qid", name], ascending=[True, False]).groupby("qid").head(N_EVID_SUBMIT)
        tops[name] = {q: list(zip(g["title"], g["sid"].astype(int))) for q, g in d.groupby("qid", sort=False)}
    return tops


if __name__ == "__main__":
    claims = load_claims(find_data_dir())
    runs = find_runs(INPUT)
    log(f"runs: {runs}")
    sets = prob_sets(runs, claims)
    tops = rankings(runs)
    dev = claims[claims["split"] == "dev"]
    test = claims[claims["split"] == "test"]
    y = dev["y"].to_numpy()
    sr = y != NEI

    grid = []
    for ev, top in tops.items():
        ok = np.array([covered(set(top.get(q, [])), gs) for q, gs in dev["groups"].items()])
        log(f"evidence {ev}: dev gold group within top-{N_EVID_SUBMIT} {ok[sr].mean():.4f}")
        for name, pr in sets.items():
            logp = np.log(pr.loc[dev.index].to_numpy() + 1e-9)
            f, bs, br = tune_offsets(logp, y, ok)
            raw = np.mean((logp.argmax(1) == y) & (~sr | ok))
            log(f"dev evidence {ev} | probs {name}: label acc {np.mean(logp.argmax(1) == y):.4f}  FEVER raw {raw:.4f}  "
                f"tuned {f:.4f} (offsets S {bs:+.1f} R {br:+.1f})")
            grid.append({"evidence": ev, "probs": name, "dev_fever": f, "S": bs, "R": br})

    def predict(g):
        logpt = np.log(sets[g["probs"]].loc[test.index].to_numpy() + 1e-9)
        return np.argmax(logpt + np.array([g["S"], g["R"], 0.0]), 1), [tops[g["evidence"]].get(q, []) for q in test.index]

    best = max(grid, key=lambda g: g["dev_fever"])
    base = next(g for g in grid if g["evidence"] == "V1" and g["probs"] == "V1")
    log(f"chosen: evidence {best['evidence']}, probs {best['probs']}, dev FEVER {best['dev_fever']:.4f} "
        f"(V1 alone {base['dev_fever']:.4f})")
    pred, ev_test = predict(best)
    pred_v1, ev_v1 = predict(base)
    new_block = test["id"].to_numpy() >= 32000
    for nm, msk in (("original-test ids", ~new_block), ("new ids >= 32000", new_block)):
        log(f"test {nm}: n={msk.sum()} label shares {np.round(np.bincount(pred[msk], minlength=3) / msk.sum(), 3).tolist()}  "
            f"same label as V1 {np.mean(pred[msk] == pred_v1[msk]):.3f}  "
            f"same evidence as V1 {np.mean([a == b for a, b, m in zip(ev_test, ev_v1, msk) if m]):.3f}")

    sample = pd.read_csv(find_data_dir() / "sample_submission.csv")
    lab_of = dict(zip(test["id"], pred))
    ev_of = dict(zip(test["id"], ev_test))
    sub = pd.DataFrame({"id": sample["id"]})
    sub["predicted_label"] = [LABELS[lab_of[i]] for i in sub["id"]]
    sub["predicted_evidence"] = [json.dumps([[t, int(s)] for t, s in ev_of[i]], ensure_ascii=False) for i in sub["id"]]
    assert len(sub) == 6000 and sub["id"].is_unique and set(sub["id"]) == set(test["id"])
    assert all(1 <= len(json.loads(e)) <= N_EVID_SUBMIT for e in sub["predicted_evidence"])
    sub.to_csv(OUT / "submission.csv", index=False)
    (OUT / "extra").mkdir(exist_ok=True)
    (OUT / "extra" / "choice.json").write_text(json.dumps({"chosen": best, "grid": grid}, default=float, indent=1))
    log(f"SUBMIT: {OUT / 'submission.csv'}")
