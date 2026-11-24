import numpy as np
import pandas as pd

from grid import PMU_BUS, Machine, electrical_power, fault_bus_voltages, kron

SCADA = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "P_G1", "Q_G1", "P_G2", "Q_G2", "P_G3", "Q_G3",
         "P_L4-5", "P_L4-6", "P_L5-7", "P_L6-9", "P_L7-8", "P_L8-9"]


def locate(m, vfault, xs, rfs):
    n = len(vfault)
    best, bx, brf = np.full(n, np.inf), np.full(n, 0.5), np.zeros(n)
    for x in xs:
        for rf in rfs:
            v = np.abs(fault_bus_voltages(m.fault(np.full(n, x), np.full(n, rf)), m.E)[:, PMU_BUS])
            err = np.nanmean((v - vfault) ** 2, 1)
            better = err < best
            best, bx, brf = np.where(better, err, best), np.where(better, x, bx), np.where(better, rf, brf)
    return bx, brf, np.sqrt(best)


def physics(cfg, br, scada, events, cap_on, l56_on, xd, H, vfault, full=True, damping=None, horizon=None, prior=True):
    damping = cfg.damping if damping is None else damping
    horizon = cfg.horizon if horizon is None else horizon
    out = pd.DataFrame(index=range(len(events)), dtype=float)
    lines = events["faulted_line"].values
    for line in np.unique(lines):
        sel = np.where(lines == line)[0]
        sc = scada.iloc[sel][SCADA].reset_index(drop=True)
        bad = sc.isna().any(axis=1).values | ~np.isfinite(H[sel]).all(1)
        sc = sc.fillna(sc.median())
        Hs = np.where(np.isfinite(H[sel]), H[sel], np.nanmedian(H[sel], 0))
        m = Machine(br, sc, line, cap_on[sel], l56_on[sel], xd[sel], Hs, damping)
        n = len(sel)
        known = np.isfinite(vfault[sel]).any(1)
        bx, brf, rmse = locate(m, vfault[sel], cfg.locate_x, cfg.locate_rf)
        accel = (m.Pm - electrical_power(kron(m.fault(bx, brf))[0], m.Emag, m.delta0)) / m.M / (2 * np.pi)
        for k in range(3):
            out.loc[sel, f"acc_pred_G{k + 1}"] = np.where(bad | ~known, np.nan, accel[:, k])
        if not full:
            continue
        fit = m.cct(m.fault(bx, brf), horizon=horizon)
        if not prior:
            out.loc[sel, "cct_fit"] = np.where(bad | ~known, np.nan, fit)
            continue
        priors = np.stack([m.cct(m.fault(np.full(n, x), np.full(n, rf)), horizon=horizon)
                          for x in cfg.prior_x for rf in cfg.prior_rf], 1)
        reclose = m.cct(m.fault(bx, brf), reclose=(cfg.reclose_dead_s, cfg.reclose_refault_s), horizon=horizon)
        res = {"cct_prior_mean": priors.mean(1), "cct_prior_min": priors.min(1), "cct_prior_max": priors.max(1),
               "cct_fit": np.where(known, fit, np.nan), "cct_reclose": np.where(known, reclose, np.nan),
               "x_fit": np.where(known, bx, np.nan), "rf_fit": np.where(known, brf, np.nan),
               "fit_rmse": np.where(known, rmse, np.nan)}
        for k, v in res.items():
            out.loc[sel, k] = np.where(bad, np.nan, v)
        print(f"physics {line}: {n} events, {bad.sum()} without full SCADA, median cct_fit {np.nanmedian(fit):.3f}, "
              f"median fit rmse {np.nanmedian(rmse):.4f}", flush=True)
    return out
