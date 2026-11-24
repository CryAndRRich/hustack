import math
import os
import shutil
import statistics
import time
import traceback
from pathlib import Path

import pandas as pd

from check import challenger_table, replay
from config import Config
from policy import submission
from runtime import find_files, log, output_dir, start_log, write_results

SCHEDULE_HEADER = ["Match"] + [f"Player {i}" for i in range(1, 7)]
HOST_SCHEDULE = 'schedule = pd.read_csv("/kaggle/input/datasets/chrissantoslang/schedule/schedule.csv")'
HOST_NAME = 'strategy_name = "My_strategy" # <-- set this to your strategy name'
HOST_DEF_START = "def submission(\n"
HOST_DEF_END = '  return "TODO: Code me"\n'


def find_schedule():
    cands = [p for p in find_files(lambda f: f.endswith(".csv"))
             if list(pd.read_csv(p, nrows=0, encoding="utf-8-sig").columns) == SCHEDULE_HEADER]
    for p in cands:
        t = pd.read_csv(p, encoding="utf-8-sig")
        strats = t.iloc[:, 1:].apply(lambda r: tuple(sorted({n[:-1] for n in r})), axis=1)
        log(f"schedule candidate {p}: {len(t)} matches, {strats.nunique()} pairings, "
            f"{int((strats.str.len() == 1).sum())} self matches")
    host = [p for p in cands if p.name == "schedule.csv"]
    assert len(host) == 1, "attach the host's schedule dataset (chrissantoslang/schedule, file schedule.csv)"
    return host[0]


def precheck(schedule):
    found = [p for p in find_files(lambda f: f.endswith(".csv"))
             if {"Match", "Player", "Strategy", "Position", "Winner"} <= set(pd.read_csv(p, nrows=0).columns)]
    if not found:
        log("precheck: no host results file found")
        return None
    res = pd.read_csv(found[0])
    sched = pd.read_csv(schedule, encoding="utf-8-sig")
    lineup = res.sort_values(["Match", "Position"]).groupby("Match")["Player"].apply(tuple)
    same = lineup.to_dict() == {m: tuple(r) for m, *r in sched.itertuples(index=False)}
    log(f"precheck {found[0].name}: {res['Match'].nunique()} matches; lineup equals schedule: {same}")
    return {"lineup_equals_schedule": same, "replay": replay(res, log)}


def host_source(cfg, schedule):
    src = (Path(cfg.code_dir) / "host.py").read_text()
    for old in (HOST_SCHEDULE, HOST_NAME, HOST_DEF_START, HOST_DEF_END):
        assert src.count(old) == 1, old
    a = src.index(HOST_DEF_START)
    b = src.index(HOST_DEF_END) + len(HOST_DEF_END)
    src = src[:a] + "submission = entry\n" + src[b:]
    src = src.replace(HOST_SCHEDULE, f"schedule = pd.read_csv({str(schedule)!r})")
    return src.replace(HOST_NAME, f"strategy_name = {cfg.strategy_name!r}")


def logged(cfg, run):
    state = {"t": time.time(), "matches": 0}

    def entry(position, round, history, cache={}):
        if round == 1 and not cache.get("team"):
            state["matches"] += 1
            if time.time() - state["t"] > cfg.log_every_s:
                state["t"] = time.time()
                log(f"run {run}: challenger match {state['matches']}")
        return submission(position, round, history, cache)

    return entry


def tournament(cfg, src, run):
    folder = Path(cfg.work) / f"run{run}"
    folder.mkdir(parents=True, exist_ok=True)
    here = os.getcwd()
    os.chdir(folder)
    try:
        ns = {"entry": logged(cfg, run), "__name__": "host"}
        t = time.time()
        exec(compile(src, "host.py", "exec"), ns)
    finally:
        os.chdir(here)
    skill = float(ns["stats"].at["challenger", "skill"])
    log(f"run {run}: skill {skill:.5f} in {(time.time() - t) / 60:.1f} min")
    return folder, skill, ns["results"]


def check_submission(path):
    sub = pd.read_csv(path)
    assert list(sub.columns) == ["id", "skill"] and len(sub) == 1 and sub["id"].iloc[0] == 0, sub
    assert math.isfinite(float(sub["skill"].iloc[0])), sub
    return float(sub["skill"].iloc[0])


def run():
    cfg = Config()
    start_log(cfg.work)
    summary = {"config": vars(cfg)}
    extra = []
    try:
        schedule = find_schedule()
        summary["schedule"] = str(schedule)
        try:
            summary["precheck"] = precheck(schedule)
        except Exception:
            summary["precheck"] = traceback.format_exc()
            log("precheck failed (non-fatal):", summary["precheck"])
        src = host_source(cfg, schedule)
        skills = []
        for k in range(1, cfg.runs + 1):
            folder, skill, results = tournament(cfg, src, k)
            skills.append(skill)
            extra += [(f"extra/run{k}/{p.name}", p) for p in sorted(folder.glob("*.csv"))]
            table = challenger_table(results)
            summary[f"run{k}"] = {"skill": skill, "vs": table}
            log(f"run {k} vs opponents:", {o: (t["win"], t["draw"], t["loss"]) for o, t in table.items()})
            if k == 1:
                sub = output_dir() / "submission.csv"
                shutil.copy(folder / f"{cfg.strategy_name}_submission.csv", sub)
                assert check_submission(sub) == skill
                for o, t in table.items():
                    log(f"  {o:10s} W/D/L {t['win']}/{t['draw']}/{t['loss']}  min ours {t['our_min']:.3f} "
                        f"theirs {t['their_min']:.3f}  mean ours {t['our_mean']:.3f} theirs {t['their_mean']:.3f}")
                summary["check"] = replay(results, log)
        summary["skills"] = skills
        if len(skills) > 1:
            log(f"skill over {len(skills)} runs: mean {statistics.mean(skills):.5f} "
                f"sd {statistics.stdev(skills):.5f} min {min(skills):.5f} max {max(skills):.5f}")
        log(f"SUBMIT: {output_dir() / 'submission.csv'} (run 1, skill {skills[0]:.5f})")
    except Exception:
        summary["error"] = traceback.format_exc()
        log(summary["error"])
        raise
    finally:
        write_results(cfg.work, cfg.code_dir, extra, summary)


if __name__ == "__main__":
    run()
