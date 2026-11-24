from pathlib import Path

import numpy as np
import pandas as pd

LINES = ["L4-5", "L4-6", "L5-7", "L6-9", "L7-8", "L8-9", "L5-6"]
PMU4 = ["V4", "theta4", "f4", "P_L4-5", "P_L4-6"]


def find_raw(root="/kaggle/input"):
    hits = sorted(Path(root).rglob("train_pmu.npz"))
    assert hits, f"train_pmu.npz not found under {root}"
    return hits[0].parent.parent


def load_split(raw, split):
    folder = raw / ("training_data" if split == "train" else "test_data")
    z = np.load(folder / f"{split}_pmu.npz")
    events = pd.read_csv(folder / f"{split}_events.csv")
    scada = pd.read_csv(folder / f"{split}_scada.csv")
    assert (z["event_id"] == events["event_id"].values).all() and (scada["event_id"] == events["event_id"]).all()
    return z["X"].astype(np.float64), z["t"], [str(c) for c in z["channels"]], events, scada


def repair_pmu(cfg, X, ch, events):
    time = events["time_utc"].values
    late = (time >= cfg.pmu4_late_from) & (time < cfg.pmu4_late_to)
    k = cfg.pmu4_late_frames
    for c in PMU4:
        j = ch.index(c)
        X[late, :-k, j] = X[late, k:, j]
        X[late, -k:, j] = np.nan
    return int(late.sum())


def repair_scada(cfg, scada, events):
    scada = scada.copy()
    time = events["time_utc"].values
    flip = time >= cfg.p45_flip_from
    scada.loc[flip, "P_L4-5"] *= -1
    stuck = scada["P_G2"].eq(cfg.pg2_stuck).values
    scada.loc[stuck, "P_G2"] = scada.loc[stuck, "P_L7-8"] - scada.loc[stuck, "P_L5-7"] + cfg.bus7_loss
    return scada, {"p45_flipped": int(flip.sum()), "pg2_stuck": int(stuck.sum())}


def state_on(log, key, value_col, dates):
    log = log.sort_values("effective_from")
    out = np.full(len(dates), None, dtype=object)
    for _, r in log.iterrows():
        m = (dates >= r["effective_from"]) & (key == r["line"])
        out[m] = r[value_col]
    return out


def context(raw, events, scada, n, H, cap_on, share):
    info = raw / "system_info"
    hourly = pd.read_csv(info / "hourly_system.csv")
    prot = pd.read_csv(info / "protection_log.csv")
    dates = events["time_utc"].str[:10].values
    hour = events["time_utc"].str[:13].values
    line = events["faulted_line"].values
    ctx = scada.drop(columns="event_id").reset_index(drop=True)
    for name in LINES:
        ctx[f"line_{name}"] = (line == name).astype(float)
    ctx["reclose"] = (state_on(prot, line, "auto_reclose", dates) == "yes").astype(float)
    ctx["fast_share"] = share
    ctx["cap_l57"] = cap_on.astype(float)
    ctx["l56_in_service"] = (dates >= "2026-12-01").astype(float)
    hourly = hourly.assign(hour=hourly["time_utc"].str[:13]).set_index("hour")
    for c in ["load_MW", "wind_MW", "solar_MW", "interchange_MW"]:
        ctx[c] = hourly[c].reindex(hour).values
    for k, g in enumerate(["G1", "G2", "G3"]):
        ctx[f"{g}_units"] = n[:, k]
        ctx[f"H_{g}"] = H[:, k]
    ctx["inertia"] = H.sum(1)
    ctx["alarms"] = events["alarms_in_hour"].values
    return ctx
