import numpy as np
import pandas as pd

from data import context, state_on
from features import pmu_features
from grid import branches
from physics import physics
from regime import (PLANTS, accel_ratio, capacitor, clearing_cdf, fast_share, fault_voltages, inertia, line_reference,
                    rolling_ratio, units_online)


def auc(score, y):
    ok = np.isfinite(score)
    score, y = score[ok], y[ok]
    r = np.empty(len(score))
    r[np.argsort(score)] = np.arange(1, len(score) + 1)
    pos = y == 1
    return round(float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())), 4)


def event_tables(cfg, raw, t, ch, splits):
    br = branches(raw)
    windows = splits["test"]["events"]["window_ms"].values
    own = pmu_features(splits["test"]["X"], t, ch, windows, cfg.jump)
    t_obs = {"train": splits["train"]["events"]["t_clear_s"].values, "test": own["t_clear"].values}
    observable = {"train": None, "test": windows >= cfg.fast_clear_s * 1000}
    usable = {"train": None, "test": windows >= 50}
    first = {}
    for name, s in splits.items():
        X, ev, sc = s["X"], s["events"], s["scada"]
        n, uc_missing = units_online(raw, ev, sc)
        H, xd = inertia(raw, ev, n)
        cap_on, logged = capacitor(br, raw, ev, sc, X, ch)
        obs = np.ones(len(ev), bool) if observable[name] is None else observable[name]
        share = fast_share(cfg, ev, t_obs[name], obs)
        l56 = (ev["time_utc"].str[:10] >= "2026-12-01").values
        vf = fault_voltages(X, ch)
        acc = physics(cfg, br, sc, ev, cap_on, l56, xd, H, vf, full=False)
        ok = np.ones(len(ev), bool) if usable[name] is None else usable[name]
        ratio = accel_ratio(X, ch, ev, acc[[f"acc_pred_{g}" for g in PLANTS]], ok)
        print(f"{name}: unit commitment imputed {uc_missing.mean():.3f}, capacitor measured on {cap_on.mean():.3f} "
              f"(log {logged.mean():.3f}, disagree {np.mean(cap_on != logged):.3f}), "
              f"fast share mean {np.nanmean(share):.3f}", flush=True)
        first[name] = dict(n=n, uc_missing=uc_missing, H=H, xd=xd, cap_on=cap_on, share=share, l56=l56, vf=vf,
                           ratio=ratio)
    ref = line_reference(first["train"]["ratio"], splits["train"]["events"])
    print("acceleration ratio reference by line (train median):", ref.round(3).to_dict())
    damping, horizon = select_dynamics(cfg, br, splits["train"], first["train"])
    tables = {}
    for name, s in splits.items():
        ev, sc, d = s["events"], s["scada"], first[name]
        rel, roll = rolling_ratio(cfg, d["ratio"], ev, ref)
        Hc = d["H"] if name == "train" else d["H"] / np.clip(np.where(np.isfinite(roll), roll, 1.0), *cfg.inertia_clip)
        month = ev["time_utc"].str[:7].values
        summary = pd.DataFrame(roll, columns=PLANTS).groupby(month).median().round(3)
        print(f"{name}: rolling inertia error (measured/predicted acceleration) by month\n{summary.to_string()}")
        ctx = context(raw, ev, sc, d["n"], Hc, d["cap_on"], d["share"]).assign(uc_missing=d["uc_missing"].astype(float))
        for k, g in enumerate(PLANTS):
            ctx[f"roll_{g}"] = roll[:, k]
            ctx[f"rel_{g}"] = rel[:, k]
        phys = physics(cfg, br, sc, ev, d["cap_on"], d["l56"], d["xd"], Hc, d["vf"], damping=damping, horizon=horizon)
        tables[name] = pd.concat([ctx, phys.drop(columns=[f"acc_pred_{g}" for g in PLANTS])], axis=1)
    ev_tr = splits["train"]["events"]
    prot = pd.read_csv(raw / "system_info" / "protection_log.csv")
    pott = state_on(prot, ev_tr["faulted_line"].values, "scheme", ev_tr["time_utc"].str[:10].values) == "POTT"
    cdf = clearing_cdf(cfg, ev_tr["t_clear_s"].values, pott.astype(int))
    y = ev_tr["unstable"].values
    tc = ev_tr["t_clear_s"].values
    phys_auc = {c: auc(tc - tables["train"][c].values, y) for c in ("cct_fit", "cct_prior_mean")}
    print(f"physics check on train: AUC(t_clear - cct) {phys_auc}, AUC(t_clear) {auc(tc, y):.4f}")
    return tables, cdf, {"auc": phys_auc, "damping": damping, "horizon": horizon}


def select_dynamics(cfg, br, split, d):
    ev, sc = split["events"], split["scada"]
    idx = np.sort(np.random.default_rng(cfg.seed).choice(len(ev), min(cfg.dynamics_events, len(ev)), replace=False))
    sub = ev.iloc[idx].reset_index(drop=True)
    tc, y = sub["t_clear_s"].values, sub["unstable"].values
    scores = {}
    for damping in cfg.damping_grid:
        for horizon in cfg.horizon_grid:
            out = physics(cfg, br, sc.iloc[idx].reset_index(drop=True), sub, d["cap_on"][idx], d["l56"][idx],
                          d["xd"][idx], d["H"][idx], d["vf"][idx], damping=damping, horizon=horizon, prior=False)
            scores[(damping, horizon)] = auc(tc - out["cct_fit"].values, y)
    best = max(scores, key=scores.get)
    print("dynamics selection on train, AUC(t_clear - cct_fit):", {f"D{k[0]} h{k[1]}": v for k, v in scores.items()},
          "->", best, flush=True)
    return best
