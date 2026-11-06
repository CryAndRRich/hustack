import gc
import itertools
import os
import shutil
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from preprocess import FUTURE, HIT, ID, MISS, N_FUTURE, PERIOD, TARGET, Features, auc, clean, load, out_dir

VERSION = "v2"
P0 = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 500,
      "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
      "max_bin": 63, "metric": "None", "verbose": -1}
BASE = {"target": "b12", "undersample": 0.2, "money_rank": False, "params": {}}
CFG = {
    "seed": 17,
    "start": 201901,
    "folds": {"A": (202105, 202107), "B": (202103, 202105), "C": (202102, 202104)},
    "tune_folds": ("B", "C"),
    "final_end": 202107,
    "arms": {"b2": {"target": "b2"}, "rank": {"money_rank": True}, "us50": {"undersample": 0.5}},
    "tune_space": {"num_leaves": (63, 127, 255, 511, 1023), "min_data_in_leaf": (100, 300, 1000, 3000),
                   "feature_fraction": (0.3, 0.5, 0.7), "learning_rate": (0.02, 0.05),
                   "lambda_l2": (1.0, 10.0, 100.0)},
    "tune_n": 10,
    "tune_budget_min": 100,
    "gate_mean": 5.0,
    "gate_worst": -10.0,
    "smooth": 2001,
    "round_to": 500,
    "rounds": 5000,
    "early_stop": 150,
    "round_mult": 1.1,
    "final_seeds": 10,
    "budget_min": 600,
}
SMOKE = {"frac": 0.03, "rounds": 15, "early_stop": 5, "final_seeds": 2, "smooth": 51, "round_to": 10,
         "tune_n": 2}


def curve(score, b2):
    order = np.argsort(-score, kind="stable")
    return np.cumsum(np.where(b2[order], HIT, MISS)) / 1e6


def smooth(g, w):
    w = max(1, min(w, len(g)))
    return np.convolve(g, np.ones(w) / w, mode="same")


def make_spec(over):
    return {**BASE, **over, "params": {**P0, **over.get("params", {})}}


def expected_count(p, undersample):
    odds = p / (1 - p) * undersample
    return float((odds / (1 + odds)).sum())


class Run:
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.feats = {False: Features(df)}
        if any(a.get("money_rank") for a in cfg["arms"].values()):
            self.feats[True] = Features(df, money_rank=True, like=self.feats[False])
        self.period = df[PERIOD].to_numpy()
        self.target = df[TARGET].to_numpy()
        self.ids = df[ID].to_numpy()
        self.b1, self.b2 = self.target == "BAJA+1", self.target == "BAJA+2"
        self.u = np.random.default_rng(cfg["seed"]).random(len(df))
        self.test = {f: np.flatnonzero(self.period == m) for f, (_, m) in cfg["folds"].items()}
        self.half = {f: np.random.default_rng(cfg["seed"] + m).random(len(self.test[f])) < 0.5
                     for f, (_, m) in cfg["folds"].items()}
        self.results, self.preds, self.specs = [], {}, {}
        self.t0 = time.time()

    def minutes(self):
        return (time.time() - self.t0) / 60

    def pos(self, spec):
        return self.b2 | self.b1 if spec["target"] == "b12" else self.b2

    def train_rows(self, spec, end):
        lab = pd.notna(self.target)
        keep = (self.target != "CONTINUA") | (self.u < spec["undersample"])
        return np.flatnonzero((self.period >= self.cfg["start"]) & (self.period <= end) & lab & keep)

    def fit(self, spec, rows, seed, rounds, monitor=None):
        feats = self.feats[spec["money_rank"]]
        params = {**spec["params"], "seed": seed, "num_threads": os.cpu_count()}
        pos = self.pos(spec)
        dtrain = lgb.Dataset(feats.matrix(rows), label=pos[rows].astype(np.float32), feature_name=feats.names,
                             free_raw_data=True)
        if monitor is None:
            return lgb.train(params, dtrain, rounds)
        b2m = self.b2[monitor]
        dmon = lgb.Dataset(feats.matrix(monitor), label=pos[monitor].astype(np.float32), reference=dtrain)
        return lgb.train(params, dtrain, rounds, valid_sets=[dmon], valid_names=["monitor"],
                         feval=lambda p, d: ("auc_b2", auc(p, b2m), True),
                         callbacks=[lgb.early_stopping(self.cfg["early_stop"], verbose=False),
                                    lgb.log_evaluation(500)])

    def evaluate(self, name, spec, fold):
        cfg = self.cfg
        test, half = self.test[fold], self.half[fold]
        rows = self.train_rows(spec, cfg["folds"][fold][0])
        t = time.time()
        model = self.fit(spec, rows, cfg["seed"], cfg["rounds"], test[half])
        p = model.predict(self.feats[spec["money_rank"]].matrix(test), num_iteration=model.best_iteration)
        b2 = self.b2[test]
        g_full = curve(p, b2)
        s_full = smooth(g_full, cfg["smooth"])
        s_rep = smooth(curve(p[~half], b2[~half]), cfg["smooth"] // 2 + 1)
        n_opt = int(np.argmax(s_full)) + 1
        r = {"run": name, "fold": fold, "train_rows": len(rows), "best_iter": model.best_iteration,
             "auc_report": auc(p[~half], b2[~half]), "smax_report_x2": 2 * s_rep.max(),
             "smax_full": s_full.max(), "n_opt": n_opt, "n_rows": len(test), "n_b2": int(b2.sum()),
             "expected": expected_count(p, spec["undersample"]), "minutes": (time.time() - t) / 60}
        self.results.append(r)
        self.preds[(name, fold)] = p
        self.specs[name] = spec
        print(f"[{name}/{fold}] {len(rows):,} rows -> iter {r['best_iter']}, AUC {r['auc_report']:.4f}, "
              f"report x2 {r['smax_report_x2']:.1f} M, full {r['smax_full']:.1f} M at N={n_opt}, "
              f"E[pos] {r['expected']:.0f} vs b2 {r['n_b2']}, {r['minutes']:.1f} min, total {self.minutes():.0f} min",
              flush=True)
        del model
        gc.collect()
        return r

    def table(self):
        return pd.DataFrame(self.results)

    def delta(self, name, folds):
        t = self.table().pivot_table(index="run", columns="fold", values="smax_report_x2")
        return {f: t.at[name, f] - t.at["base", f] for f in folds}

    def tune(self, start):
        cfg = self.cfg
        keys = list(cfg["tune_space"])
        grid = list(itertools.product(*cfg["tune_space"].values()))
        rng = np.random.default_rng(cfg["seed"])
        picks = rng.choice(len(grid), min(cfg["tune_n"], len(grid)), replace=False)
        scores = []
        for i, j in enumerate(picks):
            if self.minutes() > start + cfg["tune_budget_min"]:
                print(f"tuning budget reached after {i} configs")
                break
            params = dict(zip(keys, grid[j]))
            spec = make_spec({"params": params})
            for f in cfg["tune_folds"]:
                self.evaluate(f"tune{i}", spec, f)
            d = self.delta(f"tune{i}", cfg["tune_folds"])
            scores.append((float(np.mean(list(d.values()))), i, params))
            print(f"tune{i} {params}: mean delta vs base on {cfg['tune_folds']} {scores[-1][0]:.1f}", flush=True)
        t = pd.DataFrame([{"run": f"tune{i}", "score": s, **p} for s, i, p in scores])
        best = max(scores, key=lambda s: s[0]) if scores else None
        if best is not None and best[0] <= 0:
            print(f"no config beats base on {cfg['tune_folds']} (best {best[0]:.1f}): no tuned arm")
            best = None
        return t, best

    def final(self, spec, rounds, n_rule, work):
        cfg = self.cfg
        rows = self.train_rows(spec, cfg["final_end"])
        future = np.flatnonzero(self.period == FUTURE)
        x = self.feats[spec["money_rank"]].matrix(future)
        preds, imp = [], None
        for k in range(cfg["final_seeds"]):
            if k and self.minutes() > cfg["budget_min"]:
                print(f"budget {cfg['budget_min']} min reached: stopping at {k} seeds")
                break
            t = time.time()
            model = self.fit(spec, rows, cfg["seed"] + 1000 * (k + 1), rounds)
            preds.append(model.predict(x))
            if imp is None:
                imp = pd.Series(model.feature_importance("gain"), index=model.feature_name())
            print(f"final seed {k + 1}: {len(rows):,} rows, {rounds} rounds, {(time.time() - t) / 60:.1f} min",
                  flush=True)
            del model
            gc.collect()
        prob = np.mean(preds, axis=0)
        out = pd.DataFrame({ID: self.ids[future], "prob": prob, **{f"prob_s{k + 1}": p for k, p in enumerate(preds)}})
        out.to_csv(work / "prob_future.csv", index=False, float_format="%.8g")
        imp = imp.sort_values(ascending=False)
        imp.to_csv(work / "importance_gain.csv", header=["gain"])
        print(f"top features (gain share):\n{(imp / imp.sum()).head(25).round(4).to_string()}")
        rule, const = n_rule
        scale = len(future) if rule == "share" else expected_count(prob, spec["undersample"])
        n = int(round(const * scale / cfg["round_to"]) * cfg["round_to"])
        n = min(max(n, cfg["round_to"]), len(future))
        print(f"N rule '{rule}': {const:.5f} x {scale:,.0f} -> N = {n}")
        return out, n


def gated(r, cfg):
    adopted = {}
    for name in cfg["arms"]:
        d = r.delta(name, cfg["folds"])
        ok = np.mean(list(d.values())) >= cfg["gate_mean"] and min(d.values()) >= cfg["gate_worst"]
        print(f"gate {name}: deltas {({k: round(float(v), 1) for k, v in d.items()})} -> {'ADOPT' if ok else 'reject'}")
        if ok:
            adopted[name] = np.mean(list(d.values()))
    if "tuned" in r.specs:
        d = r.delta("tuned", ["A"])["A"]
        ok = d >= cfg["gate_mean"]
        print(f"gate tuned (held-out fold A only): delta {d:.1f} -> {'ADOPT' if ok else 'reject'}")
        if ok:
            adopted["tuned"] = d
    return adopted


def n_rule(r, name):
    cfg = r.cfg
    rows = []
    for f in cfg["folds"]:
        x = r.table().query("run == @name and fold == @f").iloc[0]
        rows.append({"fold": f, "n_rows": x.n_rows, "expected": x.expected, "n_opt": x.n_opt,
                     "g": curve(r.preds[(name, f)], r.b2[r.test[f]])})
    out = []
    for i, row in enumerate(rows):
        rest = rows[:i] + rows[i + 1:]
        for rule, key in (("share", "n_rows"), ("expected", "expected")):
            n = int(np.mean([q["n_opt"] / q[key] for q in rest]) * row[key])
            n = min(max(n, 1), len(row["g"]))
            out.append({"fold": row["fold"], "rule": rule, "N": n, "gain": row["g"][n - 1],
                        "gain_at_opt": row["g"][row["n_opt"] - 1]})
    t = pd.DataFrame(out)
    mean = t.groupby("rule")["gain"].mean()
    rule = "expected" if mean["expected"] > mean["share"] else "share"
    key = "n_rows" if rule == "share" else "expected"
    const = float(np.mean([q["n_opt"] / q[key] for q in rows]))
    print(f"N rules, leave-one-fold-out realized gain:\n{t.round(1).to_string(index=False)}\n"
          f"mean {mean.round(1).to_dict()} -> '{rule}', constant {const:.5f}")
    return (rule, const), t


def ensembles(r, names, folds):
    out = []
    for a, b in itertools.combinations(names, 2):
        d = []
        for f in folds:
            test, half = r.test[f], r.half[f]
            s = rankdata(r.preds[(a, f)]) + rankdata(r.preds[(b, f)])
            rep = smooth(curve(s[~half], r.b2[test][~half]), r.cfg["smooth"] // 2 + 1).max() * 2
            base = r.table().query("run == 'base' and fold == @f").smax_report_x2.iloc[0]
            d.append(rep - base)
        out.append({"pair": f"{a}+{b}", **{f: v for f, v in zip(folds, d)}, "mean": np.mean(d)})
    t = pd.DataFrame(out).sort_values("mean", ascending=False) if out else pd.DataFrame()
    print(f"rank-average pairs (report x2 delta vs base, not used for shipping):\n{t.round(1).to_string(index=False)}")
    return t


def run_pipeline(r, work):
    cfg = r.cfg
    folds = list(cfg["folds"])
    for f in folds:
        r.evaluate("base", make_spec({}), f)
    for name, over in cfg["arms"].items():
        for f in folds:
            r.evaluate(name, make_spec(over), f)
    tune_table, best = r.tune(r.minutes())
    if best is not None:
        name = f"tune{best[1]}"
        r.evaluate("tuned", make_spec({"params": best[2]}), "A")
        res = r.table()
        for f in cfg["tune_folds"]:
            r.preds[("tuned", f)] = r.preds[(name, f)]
            r.results.append({**res[(res.run == name) & (res.fold == f)].iloc[0].to_dict(), "run": "tuned"})
        print(f"tuned = {name} {best[2]}")
    adopted = gated(r, cfg)
    ship = max(adopted, key=adopted.get) if adopted else "base"
    if len(adopted) > 1:
        over = {k: v for name in adopted if name != "tuned" for k, v in cfg["arms"][name].items()}
        if "tuned" in adopted:
            over["params"] = best[2]
        for f in folds:
            r.evaluate("combo", make_spec(over), f)
        honest = ["A"] if "tuned" in adopted else folds
        d_combo = float(np.mean(list(r.delta("combo", honest).values())))
        print(f"combo {sorted(adopted)}: honest mean delta {d_combo:.1f} vs best single '{ship}' {adopted[ship]:.1f}")
        if d_combo >= adopted[ship]:
            ship = "combo"
    print(f"SHIP spec '{ship}': {r.specs[ship]}", flush=True)
    res = r.table()
    with pd.option_context("display.width", 250):
        print(res.round(3).to_string(index=False))
    res.to_csv(work / "backtest.csv", index=False)
    tune_table.to_csv(work / "tuning.csv", index=False)
    full = [n for n in r.specs if all((n, f) in r.preds for f in folds)]
    ens = ensembles(r, [n for n in full if n != "combo"], folds)
    ens.to_csv(work / "ensembles.csv", index=False)
    pred = pd.concat([pd.DataFrame({"fold": f, ID: r.ids[r.test[f]], TARGET: r.target[r.test[f]],
                                    "monitor": r.half[f], **{n: r.preds[(n, f)] for n in full}}) for f in folds])
    pred.to_csv(work / "backtest_preds.csv", index=False, float_format="%.6g")
    rule, rule_table = n_rule(r, ship)
    rule_table.to_csv(work / "n_rule.csv", index=False)
    it = res[res.run == ship]["best_iter"].mean()
    rounds = max(1, int(it * cfg["round_mult"]))
    prob, n = r.final(r.specs[ship], rounds, rule, work)
    return prob, n, ship


def write_submission(prob, n, path):
    sub = pd.DataFrame({ID: prob[ID].to_numpy(), "Predicted": 0})
    sub.loc[np.argsort(-prob["prob"].to_numpy(), kind="stable")[:n], "Predicted"] = 1
    sub.to_csv(path, index=False)
    return sub


def check_submission(sub, ids, n):
    assert len(sub) == len(ids) and sub[ID].is_unique and set(sub[ID]) == set(ids), "submission ids"
    assert set(sub["Predicted"].unique()) <= {0, 1} and int(sub["Predicted"].sum()) == n


def run():
    t0 = time.time()
    print(f"{VERSION} CFG: {CFG}\nP0: {P0}\nBASE: {BASE}")
    df = clean(load())
    rng = np.random.default_rng(CFG["seed"])
    ids = df[ID].unique()
    small = df[df[ID].isin(rng.choice(ids, int(len(ids) * SMOKE["frac"]), replace=False))].reset_index(drop=True)
    smoke_cfg = {**CFG, **{k: v for k, v in SMOKE.items() if k != "frac"}}
    small_future = small.loc[small[PERIOD] == FUTURE, ID].to_numpy()
    with tempfile.TemporaryDirectory() as tmp:
        prob, n, _ = run_pipeline(Run(small, smoke_cfg), Path(tmp))
        sub = write_submission(prob, n, Path(tmp) / "submission.csv")
        check_submission(sub, small_future, n)
    print(f"smoke ok ({(time.time() - t0) / 60:.1f} min)\n", flush=True)
    del small
    future_ids = df.loc[df[PERIOD] == FUTURE, ID].to_numpy()
    assert len(future_ids) == N_FUTURE, len(future_ids)
    r = Run(df, CFG)
    del df
    gc.collect()
    work = out_dir() / f"extra_{VERSION}"
    work.mkdir(exist_ok=True)
    prob, n, ship = run_pipeline(r, work)
    sub = write_submission(prob, n, out_dir() / "submission.csv")
    check_submission(sub, future_ids, n)
    zipped = shutil.make_archive(str(out_dir() / f"extra_{VERSION}"), "zip", work)
    shutil.rmtree(work)
    print(f"wrote {zipped}")
    print(f"SUBMIT submission.csv  ({VERSION}: spec {ship}, top {n} of {len(sub)} sent; "
          f"total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    run()
