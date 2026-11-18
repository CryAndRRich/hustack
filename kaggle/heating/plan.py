import time
from itertools import product

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from sim import (BAT_C, BAT_EFF, BOILER, CAR_CAP, CAR_KW, CAR_START, CHEM, COLLECTORS, KEYS, P_EXP, P_GAS, P_IMP,
                 PV_KWP_PER_M2, ROOF_M2, T_COLLECTOR, TANK_KEEP, TANK_KWH_PER_L, V, NV, cop)

SIZE_VARS = ["pv_kwp", "flat_m2", "tube_m2", "hp_kw", "battery_kwh", "tank_l", "mgso4_kwh", "lime_kwh"]
LIN_COST = {"pv_kwp": 1200, "flat_m2": 327, "tube_m2": 679, "hp_kw": 1000, "battery_kwh": 600, "tank_l": 1.5,
            "mgso4_kwh": 5.3, "lime_kwh": 3.7}
FIXED = {"flat_m2": 1500, "tube_m2": 1500, "hp_kw": 9000, "tank_l": 500}
SWITCHED = ("flat_m2", "tube_m2", "tank_l")


def per_m2_heat(w):
    dt = T_COLLECTOR - w["T"]
    return {k: np.maximum(0.0, e0 * w["G"] - (a1 * dt + a2 * dt ** 2) / 1000) for k, (e0, a1, a2) in COLLECTORS.items()}


def design_lp(kind, w, on, ranges, budget, hours):
    n = hours
    nh = n * NV
    ncol = nh + len(SIZE_VARS)
    t = np.arange(n)
    col = lambda k: t * NV + V[k]
    sz = lambda k: nh + SIZE_VARS.index(k)
    c55, csup, c65 = (np.asarray(x)[:n] for x in (cop(w["T"], 55.0), cop(w["T"], w["TS"]), cop(w["T"], 65.0)))
    qm2 = {k: v[:n] for k, v in per_m2_heat(w).items()}
    G, DW, SH, EL, DRV, HOME = (w[k][:n] for k in ("G", "DW", "SH", "EL", "DRV", "HOME"))
    R, C, X = [], [], []

    def put(block, var, coef, lag=0):
        tt = t[lag:]
        R.append(block * n + tt)
        C.append(col(var)[tt - lag])
        X.append(np.broadcast_to(np.asarray(coef, float), (n,))[lag:] if np.ndim(coef) else np.full(len(tt), float(coef)))

    def put_size(block, var, coef):
        R.append(block * n + t)
        C.append(np.full(n, sz(var)))
        X.append(np.broadcast_to(np.asarray(coef, float), (n,)).copy())

    for k in ("c_dw", "k_dw", "l_dw", "h_dw", "g_dw"):
        put(0, k, 1)
    for k in ("c_sh", "k_sh", "m_sh", "l_sh", "h_sh", "g_sh"):
        put(1, k, 1)
    put(2, "S", 1); put(2, "c_tk", -TANK_KEEP); put(2, "h_tk", -TANK_KEEP); put(2, "k_dw", TANK_KEEP)
    put(2, "k_sh", TANK_KEEP); put(2, "S", -TANK_KEEP, lag=1)
    put(3, "M", 1); put(3, "c_mg", -CHEM); put(3, "m_sh", 1); put(3, "M", -1, lag=1)
    put(4, "L", 1); put(4, "r", -CHEM); put(4, "l_dw", 1); put(4, "l_sh", 1); put(4, "L", -1, lag=1)
    put(5, "B", 1); put(5, "bc", -BAT_EFF); put(5, "bd", 1 / BAT_EFF); put(5, "B", -1, lag=1)
    put(6, "C", 1); put(6, "ch", -1); put(6, "els", -1); put(6, "C", -1, lag=1)
    put(7, "imp", 1); put(7, "bd", 1); put(7, "exp", -1); put(7, "h_dw", -1 / c55); put(7, "h_sh", -1 / csup)
    put(7, "h_tk", -1 / c65); put(7, "ch", -1); put(7, "r", -1); put(7, "bc", -1)
    put_size(7, "pv_kwp", 0.85 * G)
    b_eq = np.concatenate([DW, SH, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), -DRV, EL])
    b_eq[6 * n] += CAR_START
    A_eq = csr_matrix((np.concatenate(X), (np.concatenate(R), np.concatenate(C))), shape=(8 * n, ncol))
    R, C, X = [], [], []
    for k in ("c_dw", "c_sh", "c_tk", "c_mg"):
        put(0, k, 1)
    put_size(0, "flat_m2", -qm2["flat_m2"]); put_size(0, "tube_m2", -qm2["tube_m2"])
    for k in ("h_dw", "h_sh", "h_tk"):
        put(1, k, 1)
    put_size(1, "hp_kw", -1)
    put(2, "bc", 1); put_size(2, "battery_kwh", -BAT_C)
    put(3, "bd", 1); put_size(3, "battery_kwh", -BAT_C)
    put(4, "S", 1); put_size(4, "tank_l", -TANK_KEEP * TANK_KWH_PER_L)
    put(5, "M", 1); put_size(5, "mgso4_kwh", -1)
    put(6, "L", 1); put_size(6, "lime_kwh", -1)
    put(7, "B", 1); put_size(7, "battery_kwh", -1)
    rows, cols, vals = list(np.concatenate(R)), list(np.concatenate(C)), list(np.concatenate(X))
    b_ub = list(np.zeros(8 * n))
    roof_coef = {"pv_kwp": 1 / PV_KWP_PER_M2, "flat_m2": 1.0, "tube_m2": 1.0}
    for k, v in roof_coef.items():
        rows.append(8 * n); cols.append(sz(k)); vals.append(v)
    b_ub.append(ROOF_M2 - 1e-3)
    fixed = FIXED["hp_kw"] + sum(FIXED[k] for k in SWITCHED if on[k])
    if kind != "A":
        for k, v in LIN_COST.items():
            rows.append(8 * n + 1); cols.append(sz(k)); vals.append(v)
        b_ub.append(budget - fixed)
    A_ub = csr_matrix((vals, (rows, cols)), shape=(len(b_ub), ncol))
    hi = np.full((n, NV), np.inf)
    hi[:, V["ch"]] = CAR_KW * HOME
    hi[:, V["C"]] = CAR_CAP
    size_hi = [0.0 if k in SWITCHED and not on[k] else ranges[k][1] for k in SIZE_VARS]
    bounds = np.c_[np.zeros(ncol), np.r_[hi.ravel(), size_hi]]
    cost = np.zeros(ncol)
    hourly = cost[:nh].reshape(n, NV)
    if kind == "A":
        hourly[:, V["imp"]] = hourly[:, V["els"]] = 10 * P_IMP
        hourly[:, V["exp"]] = -10 * P_EXP
        hourly[:, V["g_dw"]] = hourly[:, V["g_sh"]] = 10 * P_GAS / BOILER
        for k, v in LIN_COST.items():
            cost[sz(k)] = v
    else:
        hourly[:, V["imp"]] = hourly[:, V["els"]] = 1.0
        hourly[:, V["exp"]] = -1e-4
        hourly[:, V["g_dw"]] = hourly[:, V["g_sh"]] = 1 / BOILER
        for k, v in LIN_COST.items():
            cost[sz(k)] = 1e-6 * v
    t0 = time.time()
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    assert res.status == 0, f"design LP {kind} {on}: {res.message}"
    x = res.x
    h = x[:nh].reshape(n, NV)
    sizes = {k: float(x[sz(k)]) for k in SIZE_VARS}
    imp = float(h[:, V["imp"]].sum() + h[:, V["els"]].sum())
    gas = float((h[:, V["g_dw"]] + h[:, V["g_sh"]]).sum() / BOILER)
    exp = float(h[:, V["exp"]].sum())
    capex = fixed + sum(LIN_COST[k] * sizes[k] for k in SIZE_VARS)
    bill = P_IMP * imp - P_EXP * exp + P_GAS * gas
    return {"sizes": sizes, "on": on, "bought": imp + gas, "bill": bill, "capex": capex, "cost10": capex + 10 * bill,
            "seconds": time.time() - t0}


def plan(kind, w, ranges, budget, hours):
    out = []
    for flags in product((0, 1), repeat=len(SWITCHED)):
        on = dict(zip(SWITCHED, flags))
        r = design_lp(kind, w, on, ranges, budget, hours)
        r["objective"] = r["cost10"] if kind == "A" else r["bought"]
        out.append(r)
        print(f"[{kind}] LP {''.join(map(str, flags))} (flat, tube, tank): objective {r['objective']:.1f}, capex "
              f"{r['capex']:.0f}, bought {r['bought']:.1f} ({r['seconds']:.0f}s)", flush=True)
    return sorted(out, key=lambda r: r["objective"])


def to_design(sizes):
    d = {k: 0.0 for k in KEYS}
    d.update(sizes)
    return d
