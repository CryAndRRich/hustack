from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

KEYS = ["pv_kwp", "pvt_panels", "flat_m2", "tube_m2", "hp_kw", "hp_source", "battery_kwh", "tank_l", "mgso4_kwh",
        "lime_kwh", "control", "store_value", "smart_goal"]
P_IMP, P_EXP, P_GAS = 0.25, 0.04, 0.30
BOILER = 0.888
TANK_KWH_PER_L, TANK_KEEP = 0.0407, 0.9975
CHEM = 0.9
BAT_EFF, BAT_C = 0.95, 0.5
CAR_CAP, CAR_KW, CAR_START, CAR_RESERVE = 60.0, 11.0, 30.0, 25.0
PVT_M2, PV_KWP_PER_M2, ROOF_M2 = 2.04, 0.2206, 60.0
COLLECTORS = {"flat_m2": (0.757, 4.069, 0.020), "tube_m2": (0.514, 1.158, 0.005)}
T_COLLECTOR = 60.0
BUDGET = {"B": 20000.0, "C": 100000.0}
REF_COLS = {"grid_import_kwh": "import", "export_kwh": "export", "gas_kwh": "gas", "hp_heat_kwh": "hp_heat",
            "hp_electricity_kwh": "hp_el", "energy_bill_eur": "bill", "capex_eur": "capex"}


def find_data():
    root = Path("/kaggle/input")
    hits = sorted(root.rglob("catalogue.csv")) if root.exists() else []
    if hits:
        return hits[0].parent
    try:
        return Path(__file__).resolve().parent.parent / "data" / "raw"
    except NameError:
        return Path.cwd() / "data" / "raw"


def load_weather(d):
    df = pd.read_csv(d / "train_2024.csv")
    assert len(df) == 8760 and df[["T_out_C", "g_roof35_Wm2", "space_heat_kwh"]].notna().all().all()
    w = {"T": df["T_out_C"], "G": df["g_roof35_Wm2"] / 1000, "SH": df["space_heat_kwh"], "DW": df["hot_water_kwh"],
         "EL": df["household_el_kwh"], "TS": df["supply_temp_C"], "HOME": df["car_home"], "DRV": df["car_drive_kwh"]}
    return {k: v.to_numpy(float) for k, v in w.items()}


def eta(ts, tk):
    e35 = np.interp(ts, [-7, 2, 7], [0.4998, 0.6042, 0.6419])
    e55 = np.interp(ts, [-7, 7], [0.4387, 0.5554])
    f = np.clip((np.asarray(tk, float) - 35) / 20, 0, 1)
    return (1 - f) * e35 + f * e55


def cop(ts, tk):
    th, tc = np.asarray(tk, float) + 5 + 273.15, np.asarray(ts, float) - 5 + 273.15
    return np.maximum(1.0, eta(ts, tk) * th / np.maximum(th - tc, 1.0))


def cop_s(ts, tk):
    if ts <= -7:
        e35 = 0.4998
    elif ts <= 2:
        e35 = 0.4998 + (ts + 7) / 9 * (0.6042 - 0.4998)
    elif ts <= 7:
        e35 = 0.6042 + (ts - 2) / 5 * (0.6419 - 0.6042)
    else:
        e35 = 0.6419
    e55 = 0.4387 if ts <= -7 else 0.5554 if ts >= 7 else 0.4387 + (ts + 7) / 14 * (0.5554 - 0.4387)
    f = min(max((tk - 35) / 20, 0.0), 1.0)
    th, tc = tk + 5 + 273.15, ts - 5 + 273.15
    return max(1.0, ((1 - f) * e35 + f * e55) * th / max(th - tc, 1.0))


def pvt_source(t, g, q_src, area):
    return t + (0.3098 * g - q_src / area) / 0.04735


def pvt_cop(t, g, area, q, tk):
    c = cop_s(t, tk)
    for _ in range(2):
        c = cop_s(pvt_source(t, g, q * (1 - 1 / c), area), tk)
    return c


def collector_heat(d, w):
    dt = T_COLLECTOR - w["T"]
    q = np.zeros(len(dt))
    for k, (e0, a1, a2) in COLLECTORS.items():
        if d[k] > 0:
            q += d[k] * np.maximum(0.0, e0 * w["G"] - (a1 * dt + a2 * dt ** 2) / 1000)
    return q


def pv_power(d, w):
    return (d["pv_kwp"] + 0.45 * d["pvt_panels"]) * w["G"] * 0.85


def capex(d):
    c = 1200 * d["pv_kwp"] + 1361 * d["pvt_panels"] + 600 * d["battery_kwh"] + 5.3 * d["mgso4_kwh"] + 3.7 * d["lime_kwh"]
    c += (327 * d["flat_m2"] + 1500) if d["flat_m2"] > 0 else 0
    c += (679 * d["tube_m2"] + 1500) if d["tube_m2"] > 0 else 0
    c += (9000 + 1000 * d["hp_kw"]) if d["hp_kw"] > 0 else 0
    c += (500 + 1.5 * d["tank_l"]) if d["tank_l"] > 0 else 0
    return c


def roof(d):
    return d["pv_kwp"] / PV_KWP_PER_M2 + PVT_M2 * d["pvt_panels"] + d["flat_m2"] + d["tube_m2"]


def valid(d, ranges):
    ok = all(ranges[k][0] <= d[k] <= ranges[k][1] for k in KEYS)
    ok &= roof(d) <= ROOF_M2 - 1e-6
    ok &= not (d["hp_source"] == 1 and d["pvt_panels"] <= 0)
    ok &= d["control"] in (0, 1) and d["smart_goal"] in (0, 1) and d["hp_source"] in (0, 1)
    return bool(ok)


def money(d, imp, exp, gas, hp_heat, hp_el):
    bill = P_IMP * imp - P_EXP * exp + P_GAS * gas
    cx = capex(d)
    return {"import": imp, "export": exp, "gas": gas, "bought": imp + gas, "hp_heat": hp_heat, "hp_el": hp_el,
            "bill": bill, "capex": cx, "cost10": cx + 10 * bill}


def simulate(d, w):
    return simulate_smart(d, w) if d["control"] == 1 else simulate_rule(d, w)


def simulate_rule(d, w):
    n = len(w["T"])
    T, G, SH, DW, EL, TS, HOME, DRV = (w[k].tolist() for k in ("T", "G", "SH", "DW", "EL", "TS", "HOME", "DRV"))
    Q, PV = collector_heat(d, w).tolist(), pv_power(d, w).tolist()
    hp, bat = float(d["hp_kw"]), float(d["battery_kwh"])
    tank_cap, mg_cap, lime_cap = d["tank_l"] * TANK_KWH_PER_L, float(d["mgso4_kwh"]), float(d["lime_kwh"])
    pvt = d["hp_source"] == 1
    area = PVT_M2 * d["pvt_panels"]
    if not pvt:
        c55, csup, c65 = cop(w["T"], 55.0).tolist(), cop(w["T"], w["TS"]).tolist(), cop(w["T"], 65.0).tolist()
    tank = mg = lime = b = 0.0
    car = CAR_START
    imp = exp = gas = hp_heat = hp_el = 0.0
    for i in range(n):
        dw, sh, q = DW[i], SH[i], Q[i]
        if q > 0:
            u = min(q, dw); dw -= u; q -= u
            u = min(q, sh); sh -= u; q -= u
            if tank_cap > 0:
                u = min(q, tank_cap - tank); tank += u; q -= u
            if mg_cap > 0:
                mg += min(CHEM * q, mg_cap - mg)
        if tank > 0:
            u = min(tank, dw); tank -= u; dw -= u
            u = min(tank, sh); tank -= u; sh -= u
        if mg > 0:
            u = min(mg, sh); mg -= u; sh -= u
        if lime > 0:
            u = min(lime, dw); lime -= u; dw -= u
            u = min(lime, sh); lime -= u; sh -= u
        cap, e_hp = hp, 0.0
        if cap > 0 and dw > 0:
            u = min(dw, cap)
            e_hp += u / (pvt_cop(T[i], G[i], area, u, 55.0) if pvt else c55[i])
            cap -= u; dw -= u; hp_heat += u
        if cap > 0 and sh > 0:
            u = min(sh, cap)
            e_hp += u / (pvt_cop(T[i], G[i], area, u, TS[i]) if pvt else csup[i])
            cap -= u; sh -= u; hp_heat += u
        gas += (dw + sh) / BOILER
        car -= DRV[i]
        if car < 0:
            imp -= car; car = 0.0
        need = min(CAR_KW, CAR_RESERVE - car) if HOME[i] and car < CAR_RESERVE else 0.0
        car += need
        s = PV[i] - EL[i] - e_hp - need
        if s > 0:
            if HOME[i] and car < CAR_CAP:
                u = min(s, CAR_KW - need, CAR_CAP - car); car += u; s -= u
            if bat > 0:
                u = min(s, BAT_C * bat, (bat - b) / BAT_EFF); b += u * BAT_EFF; s -= u
            if cap > 0 and tank_cap > 0 and tank < tank_cap and s > 0:
                c = cop_s(pvt_source(T[i], G[i], cap * (1 - 1 / 3), area), 65.0) if pvt else c65[i]
                u = min(cap, tank_cap - tank, s * c); tank += u; s -= u / c; e_hp += u / c; hp_heat += u
            if lime_cap > 0 and s > 0:
                u = min(s, (lime_cap - lime) / CHEM); lime += CHEM * u; s -= u
            exp += s
        else:
            if bat > 0:
                u = min(-s, BAT_C * bat, b * BAT_EFF); b -= u / BAT_EFF; s += u
            imp -= s
        hp_el += e_hp
        tank *= TANK_KEEP
    return money(d, imp, exp, gas, hp_heat, hp_el)


V = {k: i for i, k in enumerate(["c_dw", "c_sh", "c_tk", "c_mg", "k_dw", "k_sh", "m_sh", "l_dw", "l_sh", "h_dw", "h_sh",
                                 "h_tk", "g_dw", "g_sh", "bc", "bd", "ch", "els", "r", "imp", "exp", "S", "M", "L", "B",
                                 "C"])}
NV = len(V)


def window_lp(d, w, sl, init, c55, csup, c65, q, pv):
    n = sl.stop - sl.start
    idx = lambda k, t: t * NV + V[k]
    rows, cols, vals, rhs = [], [], [], []
    r = 0

    def eq(terms, b):
        nonlocal r
        for k, t, v in terms:
            rows.append(r); cols.append(idx(k, t)); vals.append(v)
        rhs.append(b)
        r += 1

    S0, M0, L0, B0, C0 = init
    for t in range(n):
        h = sl.start + t
        eq([("c_dw", t, 1), ("k_dw", t, 1), ("l_dw", t, 1), ("h_dw", t, 1), ("g_dw", t, 1)], w["DW"][h])
        eq([("c_sh", t, 1), ("k_sh", t, 1), ("m_sh", t, 1), ("l_sh", t, 1), ("h_sh", t, 1), ("g_sh", t, 1)], w["SH"][h])
        prev = lambda k: [(k, t - 1, -1.0)] if t else []
        eq([("S", t, 1), ("c_tk", t, -TANK_KEEP), ("h_tk", t, -TANK_KEEP), ("k_dw", t, TANK_KEEP),
            ("k_sh", t, TANK_KEEP)] + [(k, tt, v * TANK_KEEP) for k, tt, v in prev("S")], 0 if t else TANK_KEEP * S0)
        eq([("M", t, 1), ("c_mg", t, -CHEM), ("m_sh", t, 1)] + prev("M"), 0 if t else M0)
        eq([("L", t, 1), ("r", t, -CHEM), ("l_dw", t, 1), ("l_sh", t, 1)] + prev("L"), 0 if t else L0)
        eq([("B", t, 1), ("bc", t, -BAT_EFF), ("bd", t, 1 / BAT_EFF)] + prev("B"), 0 if t else B0)
        eq([("C", t, 1), ("ch", t, -1), ("els", t, -1)] + prev("C"), -w["DRV"][h] + (0 if t else C0))
        eq([("imp", t, 1), ("bd", t, 1), ("exp", t, -1), ("h_dw", t, -1 / c55[h]), ("h_sh", t, -1 / csup[h]),
            ("h_tk", t, -1 / c65[h]), ("ch", t, -1), ("r", t, -1), ("bc", t, -1)], w["EL"][h] - pv[h])
    A_eq = csr_matrix((vals, (rows, cols)), shape=(r, n * NV))
    ub_rows, ub_cols, ub_rhs = [], [], []
    for t in range(n):
        for k in ("c_dw", "c_sh", "c_tk", "c_mg"):
            ub_rows.append(2 * t); ub_cols.append(idx(k, t))
        for k in ("h_dw", "h_sh", "h_tk"):
            ub_rows.append(2 * t + 1); ub_cols.append(idx(k, t))
        ub_rhs += [q[sl.start + t], d["hp_kw"]]
    A_ub = csr_matrix((np.ones(len(ub_rows)), (ub_rows, ub_cols)), shape=(2 * n, n * NV))
    tank_cap = d["tank_l"] * TANK_KWH_PER_L
    hi = np.full((n, NV), np.inf)
    hi[:, V["bc"]] = hi[:, V["bd"]] = BAT_C * d["battery_kwh"]
    hi[:, V["ch"]] = CAR_KW * w["HOME"][sl]
    if d["lime_kwh"] <= 0:
        hi[:, V["r"]] = 0.0
    hi[:, V["S"]], hi[:, V["M"]], hi[:, V["L"]] = TANK_KEEP * tank_cap, d["mgso4_kwh"], d["lime_kwh"]
    hi[:, V["B"]], hi[:, V["C"]] = d["battery_kwh"], CAR_CAP
    for k in ("h_dw", "h_sh", "h_tk"):
        hi[:, V[k]] = d["hp_kw"]
    cost = np.zeros((n, NV))
    sv = d["store_value"]
    if d["smart_goal"] == 0:
        cost[:, V["imp"]], cost[:, V["exp"]], cost[:, V["els"]] = P_IMP, -P_EXP, P_IMP + 0.01
        cost[:, V["g_dw"]] = cost[:, V["g_sh"]] = P_GAS / BOILER
        car_credit = P_IMP
    else:
        cost[:, V["imp"]], cost[:, V["exp"]], cost[:, V["els"]] = 1.0, -0.0001, 1.01
        cost[:, V["g_dw"]] = cost[:, V["g_sh"]] = 1 / BOILER
        car_credit = 1.0
    for k in ("S", "M", "L", "B"):
        cost[-1, V[k]] -= sv
    cost[-1, V["C"]] -= car_credit
    res = linprog(cost.ravel(), A_ub=A_ub, b_ub=ub_rhs, A_eq=A_eq, b_eq=rhs,
                  bounds=np.c_[np.zeros(n * NV), hi.ravel()], method="highs")
    assert res.status == 0, f"window at hour {sl.start}: {res.message}"
    return res.x.reshape(n, NV)


def smart_cops(d, w):
    if d["hp_source"] == 1:
        area = PVT_M2 * d["pvt_panels"]
        ts = pvt_source(w["T"], w["G"], 0.49 * d["hp_kw"], area)
    else:
        ts = w["T"]
    return cop(ts, 55.0), cop(ts, w["TS"]), cop(ts, 65.0)


def simulate_smart(d, w):
    c55, csup, c65 = smart_cops(d, w)
    q, pv = collector_heat(d, w), pv_power(d, w)
    n = len(w["T"])
    init = (0.0, 0.0, 0.0, 0.0, CAR_START)
    tot = np.zeros(NV)
    hp_el = 0.0
    for start in range(0, n, 24):
        x = window_lp(d, w, slice(start, min(start + 48, n)), init, c55, csup, c65, q, pv)[:24]
        tot += x.sum(0)
        h = np.arange(start, start + len(x))
        hp_el += float((x[:, V["h_dw"]] / c55[h] + x[:, V["h_sh"]] / csup[h] + x[:, V["h_tk"]] / c65[h]).sum())
        last = x[-1]
        init = (last[V["S"]], last[V["M"]], last[V["L"]], last[V["B"]], last[V["C"]])
    gas = (tot[V["g_dw"]] + tot[V["g_sh"]]) / BOILER
    return money(d, tot[V["imp"]] + tot[V["els"]], tot[V["exp"]], gas,
                 tot[V["h_dw"]] + tot[V["h_sh"]] + tot[V["h_tk"]], hp_el)


def check_references(w, ref):
    worst = 0.0
    for _, row in ref.iterrows():
        d = {k: float(row[k]) for k in KEYS}
        r = simulate(d, w)
        diffs = {v: r[v] - row[k] for k, v in REF_COLS.items()}
        worst = max(worst, max(abs(x) for x in diffs.values()))
        print(f"  {row['reference'][:44]:44s} " + " ".join(f"{v} {x:+8.1f}" for v, x in diffs.items()), flush=True)
    return worst
