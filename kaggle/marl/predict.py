import multiprocessing as mp
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import runtime
from runtime import log

SMOKE_ROWS, SMOKE_GAMES = 8, 4
VAL_ROUND, VAL_MAX, VAL_BUDGET = 50, 400, 900.0
TEST_ROUND, TEST_MAX, TEST_BUDGET = 500, 4000, 3600.0
FOCUS_BAND, FOCUS_ROUND, FOCUS_MAX, FOCUS_BUDGET = 0.1, 1000, 20000, 2700.0
MATCH_FLOOR, AUC_FLOOR = 0.95, 0.95
LEGACY = {"ronin", "kiln", "ironjaw", "vesper", "hex", "cog"}
DROP = {"id", "state_seed", "split_label", "rollout_count", "win_count", "draw_count", "loss_count",
        "mean_final_hp_delta", "mean_rollout_turns", "win_probability"}


class Tally:
    def __init__(self):
        self.c = {}

    def add(self, res):
        table, i, variant, counts = res
        a = self.c.setdefault((table, variant), {}).setdefault(i, np.zeros(5))
        a += counts

    def counts(self, table, variant, size):
        a = np.array([self.c.get((table, variant), {}).get(i, np.zeros(5)) for i in range(size)])
        assert (a[:, 3] > 0).all(), "rows without rollouts: %d" % (a[:, 3] == 0).sum()
        return a


def prob(a, row_player_is_p2, flip):
    mine = np.where(row_player_is_p2 & ~flip, a[:, 1], a[:, 0])
    return (mine + 0.5 * a[:, 2]) / a[:, 3]


def simulate(play, jobs, budget, tally):
    deadline = runtime.elapsed() + budget
    done = 0
    sys.stdout.flush()
    pool = mp.get_context("fork").Pool(runtime.n_cpu())
    try:
        for res in pool.imap_unordered(play, jobs, chunksize=4):
            tally.add(res)
            done += 1
            if runtime.elapsed() > deadline:
                log("budget %.0fs reached after %d/%d jobs" % (budget, done, len(jobs)))
                break
    finally:
        pool.terminate()
        pool.join()
    return done


def jobs_for(table, indices, variants, per_round, max_games, seed0):
    jobs, k = [], 0
    for r in range(max_games // per_round):
        for v in variants:
            for i in indices:
                jobs.append((table, int(i), v, per_round, seed0 + k))
                k += 1
    return jobs


def scores(y, p):
    yb = (y >= 0.5).astype(int)
    auc = roc_auc_score(yb, p) if 0 < yb.sum() < len(yb) else float("nan")
    return {"auc": float(auc), "mse": float(np.mean((p - y) ** 2)), "bias": float(np.mean(p - y)), "n": int(len(y))}


def write_submission(path, sample, test, pred):
    sub = sample[["id"]].copy()
    sub["win_probability"] = sub["id"].map(pd.Series(np.clip(pred, 0, 1), index=test["id"].to_numpy()))
    assert len(sub) == 700 and sub["win_probability"].notna().all()
    sub.to_csv(path, index=False)


def rebuild(df, name, index, find, replay, build, character):
    states, swapped, how = [], [], []
    for _, r in df.iterrows():
        chars = (character(r, "player"), character(r, "opp"))
        seed, sw = find(index, chars, (r["player_loadout"].split("|"), r["opp_loadout"].split("|")))
        g = None
        if seed is not None:
            g, bot, stag = replay(r, seed, sw)
        if g is None:
            states.append(build(r))
            swapped.append(False)
            how.append("v1" if seed is None else "v1_nomatch")
        else:
            states.append(g)
            swapped.append(sw)
            how.append(bot + ("" if stag else "_stagdiff"))
    how = pd.Series(how)
    log("REBUILD %s: %s; swapped %d" % (name, how.value_counts().to_dict(), int(np.sum(swapped))))
    return states, np.array(swapped), how


def run():
    try:
        root = runtime.data_root()
        runtime.import_engine(root)
        import rollout
        from seeds import find, seed_index
        from state import build, character, replay

        train, test, sample = (pd.read_csv(root / f"{n}.csv") for n in ("train", "test", "sample_submission"))
        log("data", root, "train", train.shape, "test", test.shape, "cpus", runtime.n_cpu())
        assert set(sample["id"]) == set(test["id"])
        index = seed_index()
        log("seed index: %d seeds" % len(index))
        st_tr, sw_tr, how_tr = rebuild(train, "train", index, find, replay, build, character)
        st_te, sw_te, how_te = rebuild(test, "test", index, find, replay, build, character)
        exact_tr = ~how_tr.str.startswith("v1").to_numpy()
        exact_te = ~how_te.str.startswith("v1").to_numpy()
        if exact_tr.mean() < MATCH_FLOOR:
            raise RuntimeError("only %.3f of train rows replay exactly: warmup protocol not reproduced" % exact_tr.mean())
        rollout.STATES["train"], rollout.STATES["test"] = st_tr, st_te
        y = train["win_probability"].to_numpy()

        smoke = Tally()
        t0 = runtime.elapsed()
        smoke_jobs = (jobs_for("train", range(SMOKE_ROWS), tuple(rollout.BOTS), SMOKE_GAMES, SMOKE_GAMES, 1)
                      + jobs_for("test", range(SMOKE_ROWS), ("heu",), SMOKE_GAMES, SMOKE_GAMES, 2))
        simulate(rollout.play, smoke_jobs, 1e9, smoke)
        per_game = (runtime.elapsed() - t0) * runtime.n_cpu() / (len(smoke_jobs) * SMOKE_GAMES)
        log("SMOKE %.2f ms/game/cpu; projected val %.0fs, test %.0fs (before budgets)" % (
            1000 * per_game, per_game * len(train) * len(rollout.BOTS) * VAL_MAX / runtime.n_cpu(),
            per_game * len(test) * TEST_MAX / runtime.n_cpu()))

        val = Tally()
        simulate(rollout.play, jobs_for("train", range(len(train)), tuple(rollout.BOTS), VAL_ROUND, VAL_MAX, 10**6),
                 VAL_BUDGET, val)
        chars = pd.DataFrame({s: [character(r, s) for _, r in train.iterrows()] for s in ("player", "opp")})
        groups = {s: (train["split_label"] == s).to_numpy() for s in ("easy", "mid", "hard")}
        groups["legacy"] = (chars["player"].isin(LEGACY) | chars["opp"].isin(LEGACY)).to_numpy()
        groups["swapped"] = sw_tr
        groups["poise0_one_side"] = ((train["player_poise"] == 0) ^ (train["opp_poise"] == 0)).to_numpy()
        groups["not_exact"] = ~exact_tr
        table, mc_train = {}, {}
        for v in rollout.BOTS:
            a = val.counts("train", v, len(train))
            for flip in (False, True):
                arm = v + ("_enginep1" if flip else "")
                p = prob(a, sw_tr, np.full(len(train), flip))
                mc_train[arm] = p
                row = {"all": scores(y, p), "games_min": int(a[:, 3].min())}
                row.update({g: scores(y[m], p[m]) for g, m in groups.items() if m.sum() > 1})
                table[arm] = row
                log("VAL %-13s AUC %.4f MSE %.5f bias %+.4f | " % (arm, row["all"]["auc"], row["all"]["mse"],
                                                                     row["all"]["bias"])
                    + " ".join("%s %.5f" % (g, row[g]["mse"]) for g in groups if g in row))
        noise = float(np.mean(y * (1 - y) / train["rollout_count"].to_numpy()))
        log("label-noise MSE floor %.5f; V1 reference MSE 0.00477 (mid 0.0085)" % noise)
        chosen = min(table, key=lambda k: table[k]["all"]["mse"])
        variant, flip = chosen.split("_")[0], chosen.endswith("_enginep1")
        log("chosen", chosen)
        if table[chosen]["all"]["auc"] < AUC_FLOOR:
            raise RuntimeError("train AUC %.4f < %.2f" % (table[chosen]["all"]["auc"], AUC_FLOOR))

        tst = Tally()
        simulate(rollout.play, jobs_for("test", range(len(test)), (variant,), TEST_ROUND, TEST_MAX, 10**8),
                 TEST_BUDGET, tst)
        a = tst.counts("test", variant, len(test))
        p_base = prob(a, sw_te, np.full(len(test), flip))
        focus = np.where(np.abs(p_base - 0.5) < FOCUS_BAND)[0]
        log("TEST base: games/row min %d; %d rows within %.2f of 0.5 get up to %d games" % (
            a[:, 3].min(), len(focus), FOCUS_BAND, FOCUS_MAX))
        if len(focus):
            simulate(rollout.play, jobs_for("test", focus, (variant,), FOCUS_ROUND, FOCUS_MAX - int(a[:, 3].min()),
                                            10**9), FOCUS_BUDGET, tst)
        a = tst.counts("test", variant, len(test))
        p_test = prob(a, sw_te, np.full(len(test), flip))
        moved = np.abs(p_test - p_base)[focus] if len(focus) else np.zeros(1)
        log("TEST final: games/row min %d max %d; focus rows moved mean %.4f max %.4f; p mean %.3f" % (
            a[:, 3].min(), a[:, 3].max(), moved.mean(), moved.max(), p_test.mean()))

        feats = [c for c in test.columns if c not in DROP and pd.api.types.is_numeric_dtype(test[c])
                 and train[c].nunique() > 1]
        xtr = train[feats].assign(mc=mc_train[chosen]).to_numpy(float)
        xte = test[feats].assign(mc=p_test).to_numpy(float)
        oof, stack = np.zeros(len(train)), np.zeros(len(test))
        for tr_i, va_i in KFold(5, shuffle=True, random_state=0).split(xtr):
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3).fit(xtr[tr_i], y[tr_i])
            oof[va_i] = m.predict(xtr[va_i])
            stack += m.predict(xte) / 5
        log("STACK OOF AUC %.4f MSE %.5f vs MC alone AUC %.4f MSE %.5f" % (
            scores(y, oof)["auc"], scores(y, oof)["mse"], table[chosen]["all"]["auc"], table[chosen]["all"]["mse"]))

        extra = runtime.WORK / "extra"
        extra.mkdir(parents=True, exist_ok=True)
        write_submission(extra / "submission_stack.csv", sample, test, stack)
        pd.DataFrame({"id": train["id"], "label": y, "split": train["split_label"], "swapped": sw_tr,
                      "rebuild": how_tr, **{f"mc_{k}": v for k, v in mc_train.items()}, "stack_oof": oof}).to_csv(
            runtime.WORK / "train_mc.csv", index=False)
        pd.DataFrame({"id": test["id"], "mc": p_test, "mc_base": p_base, "games": a[:, 3], "swapped": sw_te,
                      "rebuild": how_te, "stack": stack}).to_csv(runtime.WORK / "test_mc.csv", index=False)
        sub = runtime.out_dir() / "submission.csv"
        write_submission(sub, sample, test, p_test)
        runtime.save_json({"val": table, "chosen": chosen, "label_noise_mse": noise, "stack_oof": scores(y, oof),
                           "rebuild_train": how_tr.value_counts().to_dict(),
                           "rebuild_test": how_te.value_counts().to_dict()}, "result.json")
        log("SUBMIT:", sub, "(V2 exact-state MC, %s, %d+ games/state, %d exact test states)" % (
            chosen, a[:, 3].min(), exact_te.sum()))
    finally:
        runtime.save_code()
        runtime.zip_results()


if __name__ == "__main__":
    run()
