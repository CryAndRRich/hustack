import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from data import find_raw, load_split, repair_pmu, repair_scada
from ensemble import blend, search
from features import test_rows, train_rows, truncated_rows
from metric import decide
from models import domain_weights, lgb_bag, lgb_fit, protocol
from pseudo import arm_scores, labelled
from regime import derived
from runtime import run_session
from seqnet import sequences
from tables import event_tables

FAMILIES = ["lgbm", "catboost", "tabpfn", "gru"]
WORKERS = {"0": "tabpfn", "1": "catboost,gru"}


def setup_tabpfn():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn"], check=False)
    if Path("/kaggle/working").exists():
        from kaggle_secrets import UserSecretsClient
        for key in ("HF_TOKEN", "TABPFN_TOKEN"):
            try:
                os.environ[key] = UserSecretsClient().get_secret(key)
            except Exception:
                print(f"secret {key} not available")
    os.environ["TABPFN_NO_BROWSER"] = "1"
    if not os.environ.get("TABPFN_TOKEN"):
        os.environ["TABPFN_MODEL_VERSION"] = "v2"
        print("WARNING: no TABPFN_TOKEN secret -> TabPFN v2 (ungated)")
    try:
        from importlib.metadata import version
        print(f"tabpfn {version('tabpfn')}, model version {os.environ.get('TABPFN_MODEL_VERSION', 'default')}")
    except Exception as e:
        print(f"tabpfn not installed: {e!r}")


def best_arm(report):
    key = max(report, key=lambda k: report[k]["score"])
    return key, report[key]


def cross_fit(cfg, rows, trunc, cols, rounds, pi):
    held = ~trunc["own"].values
    reports, oofs = {}, {}
    for weight in cfg.test_weights:
        oof = np.full(len(trunc), np.nan)
        for f in range(cfg.folds):
            part = trunc[trunc["fold"] != f]
            model = lgb_fit(cfg, pd.concat([rows, labelled(part, part["q"].values, cfg, weight)], ignore_index=True),
                            cols, rounds=rounds)
            m = (trunc["fold"] == f).values
            oof[m] = model.predict(trunc.loc[m, cols])
        ev = trunc[held]
        reports[weight] = arm_scores(cfg, ev, oof[held], (ev["q"].values >= 0.5).astype(int), pi)
        oofs[weight] = oof
        key, arm = best_arm(reports[weight])
        print(f"pseudo CV test weight {weight}: best {key} score {arm['score']}", flush=True)
    return reports, oofs


def self_label(cfg, rows, trunc, cols, rounds, q, chosen):
    pi = float(q.mean())
    rng = np.random.default_rng(cfg.seed)
    fold = dict(zip(chosen, rng.permutation(len(chosen)) % cfg.folds))
    trunc["fold"] = trunc["event"].map(fold)
    for r in range(cfg.pseudo_rounds):
        reports, oofs = cross_fit(cfg, rows, trunc, cols, rounds, pi)
        weight = max(reports, key=lambda w: best_arm(reports[w])[1]["score"])
        print(f"round {r + 1}: chosen test weight {weight}, {best_arm(reports[weight])[0]}", flush=True)
        if r + 1 < cfg.pseudo_rounds:
            own = trunc["own"].values
            refit = pd.Series(oofs[weight][own], index=trunc.loc[own, "event"].values)
            q_new = 0.5 * (trunc["q"].values + refit.reindex(trunc["event"].values).values)
            print(f"round {r + 2} labels: {np.mean((q_new >= 0.5) != (trunc['q'].values >= 0.5)):.4f} flip")
            trunc["q"] = q_new
            pi = float(refit.reindex(chosen).mul(0.5).add(0.5 * q).mean())
    return weight, pi


def worker_log(work, families, name):
    return work / f"log_{families.replace(',', '_')}_{name}.txt"


def run_families(cfg, data, work, name, rounds):
    data_dir, out_dir = Path(cfg.scratch_dir) / f"data_{name}", work / f"preds_{name}"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump({k: v for k, v in data.items() if k != "seq"}, open(data_dir / "data.pkl", "wb"))
    (data_dir / "config.json").write_text(json.dumps(asdict(cfg)))
    np.savez(data_dir / "seq.npz", **data["seq"])
    start = time.time()
    procs = []
    for gpu, families in WORKERS.items():
        log = open(worker_log(work, families, name), "w")
        procs.append((subprocess.Popen([sys.executable, str(Path(cfg.code_dir) / "worker.py"), families,
                                        str(data_dir), str(out_dir)], stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}), log))
    preds = {"lgbm": protocol(cfg, lambda c, tr, cols, Xs: lgb_bag(c, tr, cols, rounds, Xs, monotone=True), data,
                              lambda m: print(f"[{time.time() - start:6.0f}s] lgbm {name}: {m}", flush=True))}
    for p, log in procs:
        p.wait()
        log.close()
    for f in FAMILIES[1:]:
        path = out_dir / f"{f}.npz"
        if path.exists():
            preds[f] = dict(np.load(path))
    for families in WORKERS.values():
        print(f"{name} worker {families} log tail:\n{worker_log(work, families, name).read_text()[-800:]}")
    print(f"{name}: families {sorted(preds)} in {time.time() - start:.0f}s", flush=True)
    return preds


def subset(data):
    rng = np.random.default_rng(0)
    small = dict(data)
    events = rng.choice(data["trunc"]["event"].unique(), 300, replace=False)
    small["trunc"] = data["trunc"][data["trunc"]["event"].isin(events)].reset_index(drop=True)
    small["rows"] = data["rows"].iloc[np.sort(rng.choice(len(data["rows"]), 3000, replace=False))].reset_index(drop=True)
    small["is_val"] = (small["rows"]["val"] == 1).values
    small["test"] = data["test"].iloc[np.sort(rng.choice(len(data["test"]), 400, replace=False))].reset_index(drop=True)
    return small


def solve(cfg, work, submission):
    extra = work / "extra"
    setup_tabpfn()
    raw = find_raw()
    print(f"data: {raw}")
    Xtr, t, ch, ev_tr, sc_tr = load_split(raw, "train")
    Xte, t_te, ch_te, ev_te, sc_te = load_split(raw, "test")
    assert ch == ch_te and np.allclose(t, t_te)
    print(f"repaired PMU 4 delay on {repair_pmu(cfg, Xte, ch, ev_te)} test events, "
          f"{repair_pmu(cfg, Xtr, ch, ev_tr)} train events")
    sc_te, fixed = repair_scada(cfg, sc_te, ev_te)
    sc_tr, fixed_tr = repair_scada(cfg, sc_tr, ev_tr)
    print(f"repaired SCADA: test {fixed}, train {fixed_tr}")
    splits = {"train": {"X": Xtr, "events": ev_tr, "scada": sc_tr}, "test": {"X": Xte, "events": ev_te, "scada": sc_te}}
    tables, cdf, physics_info = event_tables(cfg, raw, t, ch, splits)
    ctx_tr, ctx_te = tables["train"], tables["test"]
    windows = ev_te["window_ms"].values
    is_val_event = (ev_tr["time_utc"] >= cfg.val_from).values
    rows = derived(train_rows(Xtr, t, ch, ctx_tr, ev_tr["unstable"].values, cfg.jump), cdf)
    rows = rows.assign(w=1.0, base_w=1.0, src=0, val=is_val_event[rows["event"].values].astype(int))
    test = derived(test_rows(Xte, t, ch, ctx_te, windows, cfg.jump), cdf).assign(src=1, event=np.arange(len(ev_te)))
    cols = [c for c in test.columns if c not in ("event", "y", "w", "base_w", "own", "src", "val")]
    print(f"train rows {len(rows)}, test rows {len(test)}, features {len(cols)}", flush=True)
    (extra / "events_test.csv").write_text(ctx_te.assign(event_id=ev_te["event_id"]).to_csv(index=False))

    is_val = (rows["val"] == 1).values
    probe = lgb_fit(cfg, rows[~is_val], cols, va=rows[is_val])
    rounds = int(probe.best_iteration * 1.1) + 1
    dw = domain_weights(cfg, rows[is_val], test, cols)
    print(f"probe iterations {probe.best_iteration}; domain weights sd {dw.std():.2f}", flush=True)

    q_own = lgb_bag(cfg, rows, cols, rounds, [test])[0]
    chosen = np.where(windows >= cfg.pseudo_min_ms)[0]
    q = q_own[chosen]
    trunc = derived(truncated_rows(Xte[chosen], t, ch, ctx_te.iloc[chosen], windows[chosen], cfg.jump), cdf)
    trunc["q"] = q[trunc["event"].values]
    trunc["event"] = chosen[trunc["event"].values]
    trunc["src"] = 1
    print(f"pseudo events {len(chosen)}: mean q {q.mean():.4f}; truncated rows {len(trunc)}", flush=True)
    weight, pi = self_label(cfg, rows, trunc, cols, rounds, q, chosen)

    seq_tr, seq_te = sequences(Xtr, ch), sequences(Xte, ch)
    scale = np.where(np.arange(seq_tr.shape[2]) % 2 == 0, np.std(seq_tr, (0, 1)) + 1e-6, 1.0).astype(np.float32)
    data = {"rows": rows, "trunc": trunc, "test": test, "cols": cols, "is_val": is_val, "weight": weight,
            "seq": {"train": seq_tr, "test": seq_te, "scale": scale}}
    smoke = run_families(cfg, subset(data), work, "smoke", rounds)
    preds = run_families(cfg, data, work, "full", rounds)
    for f in FAMILIES:
        if f not in preds:
            print(f"WARNING: family {f} missing from the ensemble (in smoke: {f in smoke})")

    names = [f for f in FAMILIES if f in preds]
    held = ~trunc["own"].values
    val_rows = rows[is_val]
    sets = [{"preds": [preds[n]["oof"][held] for n in names], "window": trunc["cut_s"].values[held] * 1000,
             "y": (trunc["q"].values[held] >= 0.5).astype(int), "pi": pi},
            {"preds": [preds[n]["val"] for n in names], "window": val_rows["cut_s"].values * 1000,
             "y": val_rows["y"].values, "pi": float(np.average(preds["lgbm"]["val"], weights=dw)), "weight": dw}]
    (total, w, safety, per), single = search(cfg, names, sets)
    print(f"single families (combined score): { {k: round(v, 2) for k, v in single.items()} }")
    print(f"ensemble weights {dict(zip(names, w.round(2)))}, combined {total:.3f}, safety per moment {safety}, "
          f"per moment { {k: round(v, 1) for k, v in per.items()} }", flush=True)

    p_test = blend([preds[n]["test"] for n in names], w)
    s_row = np.array([safety[int(m)] for m in windows])
    sample = pd.read_csv(raw / "test_data" / "sample_submission.csv")
    files = {"ensemble": decide(p_test, windows, pi, s_row), "ensemble_safety1.0": decide(p_test, windows, pi, 1.0)}
    for n in names:
        files[f"{n}_safety1.0"] = decide(preds[n]["test"], windows, pi, 1.0)
    calls = {}
    for name, dec in files.items():
        sub = pd.DataFrame({"event_id": ev_te["event_id"], "decision": dec})
        assert list(sub.columns) == list(sample.columns) and (sub["event_id"].values == sample["event_id"].values).all()
        assert sub["decision"].isin(["stable", "unstable", "wait"]).all() and len(sub) == 12100
        sub.to_csv(extra / f"submission_{name}.csv", index=False)
        calls[name] = {d: int(n) for d, n in sub["decision"].value_counts().items()}
        if name == "ensemble":
            sub.to_csv(submission, index=False)
    print("test calls", json.dumps(calls))
    pd.DataFrame({"event_id": ev_te["event_id"], "window_ms": windows, "q_labeller": q_own, "p": p_test,
                  **{f"p_{n}": preds[n]["test"] for n in names}}).to_csv(extra / "test_p.csv", index=False)
    return {"physics": physics_info, "families": names, "single": single, "weights": dict(zip(names, w.tolist())),
            "safety": {str(k): v for k, v in safety.items()}, "combined": total,
            "per_moment": {str(k): v for k, v in per.items()}, "test_weight": weight, "pi": pi, "rounds": rounds,
            "calls": calls}


def run(cfg=None):
    run_session(cfg or Config(), solve)
