import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from plan import plan, to_design
from sim import BUDGET, KEYS, PV_KWP_PER_M2, PVT_M2, ROOF_M2, capex, check_references, find_data, load_weather, roof, simulate, valid

VERSION = "V2"
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
FULL = dict(hours=8760, n_lp=3, levels=4, max_evals=500)
SMOKE = dict(hours=168, n_lp=1, levels=1, max_evals=4)
SIZES = {"pv_kwp": 0.05, "pvt_panels": 1, "flat_m2": 0.1, "tube_m2": 0.1, "hp_kw": 0.1, "battery_kwh": 0.1,
         "tank_l": 10, "mgso4_kwh": 10, "lime_kwh": 10}
UNIT_COST = {"pv_kwp": 1200, "pvt_panels": 1361, "flat_m2": 327, "tube_m2": 679, "hp_kw": 1000, "battery_kwh": 600,
             "tank_l": 1.5, "mgso4_kwh": 5.3, "lime_kwh": 3.7}
UNIT_ROOF = {"pv_kwp": 1 / PV_KWP_PER_M2, "pvt_panels": PVT_M2, "flat_m2": 1.0, "tube_m2": 1.0}
SETTINGS = [(0.0, 0, 0.0)] + [(1.0, g, sv) for g in (0.0, 1.0) for sv in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)]
DESIGNS = ("A", "B", "C")
V1_DESIGNS = {
    "A": dict(pv_kwp=2.5, hp_kw=3.9, tank_l=300, control=1, store_value=0.05, smart_goal=0),
    "B": dict(pv_kwp=4.75, hp_kw=4.0, tank_l=480, lime_kwh=20, control=1, store_value=0.02, smart_goal=1),
    "C": dict(pv_kwp=12.15, tube_m2=4.7, hp_kw=16, battery_kwh=30, tank_l=2160, mgso4_kwh=1530, lime_kwh=3000,
              control=1, store_value=0.02, smart_goal=1)}


def snap(d, ranges, down=False):
    out = dict(d)
    for k, step in SIZES.items():
        q = np.floor(d[k] / step + 1e-6) if down else np.round(d[k] / step)
        out[k] = float(np.clip(round(q * step, 4), *ranges[k]))
    out["store_value"] = float(np.clip(round(d["store_value"], 2), *ranges["store_value"]))
    if out["pvt_panels"] <= 0 or out["control"] == 1:
        out["hp_source"] = 0.0
    return out


def feasible(kind, d, ranges):
    return valid(d, ranges) and (kind == "A" or capex(d) <= BUDGET[kind])


def part(kind, d, w, base):
    r = simulate(d, w)
    return (r["cost10"] / base["cost10"]) if kind == "A" else r["bought"] / base["bought"]


def repairs(kind, cand, moved, ranges):
    if feasible(kind, cand, ranges):
        return [cand]
    over_cost = capex(cand) - BUDGET[kind] if kind != "A" else 0.0
    over_roof = roof(cand) - (ROOF_M2 - 1e-6)
    out = []
    for j in SIZES:
        if j == moved or cand[j] <= 0 or (over_roof > 0 and j not in UNIT_ROOF):
            continue
        cut = max(over_cost / UNIT_COST[j] if over_cost > 0 else 0.0, over_roof / UNIT_ROOF[j] if over_roof > 0 else 0.0)
        v = np.floor((cand[j] - cut) / SIZES[j] - 1e-9) * SIZES[j]
        fixed = snap({**cand, j: max(v, 0.0)}, ranges)
        if feasible(kind, fixed, ranges):
            out.append(fixed)
    return out


def sweep(kind, d, w, ranges, base):
    out = []
    for control, goal, sv in SETTINGS:
        cand = snap({**d, "control": control, "smart_goal": goal, "store_value": sv}, ranges, down=True)
        if feasible(kind, cand, ranges):
            out.append((part(kind, cand, w, base), cand))
    return sorted(out, key=lambda x: x[0])


def local_search(kind, best, fbest, w, ranges, base, levels, max_evals, t0):
    seen, evals = {tuple(best.values())}, 0
    for lv in range(levels):
        improved = True
        while improved and evals < max_evals:
            improved = False
            for k in list(SIZES) + ["store_value", "smart_goal"]:
                if k == "smart_goal":
                    moves = [1.0 - best[k]] if best["control"] == 1 else []
                else:
                    step = SIZES.get(k, 0.01)
                    delta = max(step, (ranges[k][1] - ranges[k][0]) * 0.1 / 2 ** lv)
                    moves = [best[k] + delta, best[k] - delta, 0.0]
                for v in moves:
                    if v == best[k] or not ranges[k][0] <= v <= ranges[k][1]:
                        continue
                    for cand in repairs(kind, snap({**best, k: v}, ranges), k, ranges):
                        key = tuple(cand.values())
                        if key in seen or evals >= max_evals:
                            continue
                        seen.add(key)
                        evals += 1
                        f = part(kind, cand, w, base)
                        if f < fbest - 1e-9:
                            best, fbest, improved = cand, f, True
                            print(f"[{kind}] level {lv} eval {evals}: {fbest:.5f} via {k} -> {cand[k]:g} "
                                  f"({time.time() - t0:.0f}s)", flush=True)
        print(f"[{kind}] level {lv} done: {fbest:.5f} after {evals} evals ({time.time() - t0:.0f}s)", flush=True)
    return fbest, best


def solve_design(kind, w, ranges, base, hours, n_lp, levels, max_evals):
    t0 = time.time()
    lps = plan(kind, w, ranges, BUDGET.get(kind, np.inf), hours)
    norm = base["cost10"] if kind == "A" else base["bought"]
    bound = lps[0]["objective"] / norm
    print(f"[{kind}] LP bound (perfect foresight, full year): part {bound:.5f} ({time.time() - t0:.0f}s)", flush=True)
    starts = [("lp" + "".join(str(v) for v in r["on"].values()), to_design(r["sizes"])) for r in lps[:n_lp]]
    starts.append(("v1", {**to_design({}), **{k: float(v) for k, v in V1_DESIGNS[kind].items()}}))
    rows, best = [], None
    for name, d in starts:
        res = sweep(kind, d, w, ranges, base)
        if not res:
            print(f"[{kind}] {name}: no feasible setting", flush=True)
            continue
        f, cand = res[0]
        rows.append({"design": kind, "start": name, "part": f, **cand})
        print(f"[{kind}] {name}: best part {f:.5f} with control {cand['control']:g}, goal {cand['smart_goal']:g}, "
              f"store_value {cand['store_value']:g} ({time.time() - t0:.0f}s)", flush=True)
        if best is None or f < best[0]:
            best = (f, cand)
    fbest, dbest = local_search(kind, best[1], best[0], w, ranges, base, levels, max_evals, t0)
    rows.append({"design": kind, "start": "final", "part": fbest, **dbest})
    return kind, fbest, dbest, bound, rows


def run(w, ranges, hours, n_lp, levels, max_evals, smoke):
    t0 = time.time()
    base = simulate({k: 0.0 for k in KEYS}, w)
    print(f"{'smoke' if smoke else 'full'}: doing nothing: bill {base['bill']:.2f}, bought {base['bought']:.1f}")
    out = Parallel(n_jobs=3)(delayed(solve_design)(k, w, ranges, base, hours, n_lp, levels, max_evals)
                             for k in DESIGNS)
    res = {k: (f, d, b) for k, f, d, b, _ in out}
    for k in DESIGNS:
        f, d, b = res[k]
        r = simulate(d, w)
        print(f"  {k}: part {f:.5f} (LP bound {b:.5f}), capex {r['capex']:.0f}, bill {r['bill']:.1f}, "
              f"bought {r['bought']:.1f}, import {r['import']:.0f}, export {r['export']:.0f}, gas {r['gas']:.0f}")
        print("     " + ", ".join(f"{x}={d[x]:g}" for x in KEYS if d[x]))
    print(f"expected public score {np.mean([res[k][0] for k in DESIGNS]):.5f}, LP bound "
          f"{np.mean([res[k][2] for k in DESIGNS]):.5f} ({(time.time() - t0) / 60:.1f} min)")
    if smoke:
        return
    extra = OUT / "extra" / VERSION
    extra.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for *_, rows in out for r in rows]).to_csv(extra / "candidates.csv", index=False)
    (extra / "designs.json").write_text(json.dumps({k: res[k][1] for k in DESIGNS}, indent=1))
    ids, vals = [], []
    for year in ("pub", "priv"):
        for k in DESIGNS:
            d = res[k][1]
            assert feasible(k, d, ranges)
            for x in KEYS:
                ids.append(f"{year}_{k}_{x}")
                vals.append(d[x])
    sub = pd.DataFrame({"id": ids, "value": vals})
    assert len(sub) == 78 and sub["id"].is_unique
    sub.to_csv(OUT / "submission.csv", index=False)
    print(f"SUBMIT: {OUT / 'submission.csv'}")


if __name__ == "__main__":
    t0 = time.time()
    data = find_data()
    w = load_weather(data)
    cat = pd.read_csv(data / "catalogue.csv")
    assert list(cat["name"]) == KEYS
    ranges = {r["name"]: (float(r["min"]), float(r["max"])) for _, r in cat.iterrows()}
    print("reference check (ours - host):")
    worst = check_references(w, pd.read_csv(data / "reference_results.csv"))
    print(f"  worst |diff| {worst:.2f}" + ("" if worst < 2 else "  <-- WARNING: physics does not match the host"))
    run(w, ranges, smoke=True, **SMOKE)
    print("smoke ok")
    run(w, ranges, smoke=False, **FULL)
    print(f"total {(time.time() - t0) / 60:.1f} min")
