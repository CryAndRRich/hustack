import numpy as np
import pandas as pd

from metric import WAIT, decide, score


def labelled(rows, q, cfg, weight):
    q = np.clip(q, 0, 1)
    hard = (q <= cfg.hard_lo) | (q >= cfg.hard_hi)
    rows = rows.assign(base_w=weight)
    base = rows.assign(y=(q >= 0.5).astype(int), w=weight)[hard]
    soft = rows[~hard]
    pos = soft.assign(y=1, w=weight * q[~hard])
    neg = soft.assign(y=0, w=weight * (1 - q[~hard]))
    return pd.concat([base, pos, neg], ignore_index=True)


def no_stable(cfg, rows):
    settled = rows["cleared"].eq(1) & (rows["since_clear"] >= 0.3 + cfg.reclose_settle_s)
    safe = rows["cleared"].eq(1) & (rows["t_clear"] < rows["cct_reclose"] - cfg.reclose_margin)
    return (rows["reclose"].values == 1) & ~settled.values & ~safe.values


def arm_scores(cfg, rows, p, y, pi, weight=None):
    out = {}
    window = rows["cut_s"].values * 1000
    for override in (True, False):
        blocked = no_stable(cfg, rows) if override else None
        for k in cfg.safety:
            dec = decide(p, window, pi, k, blocked)
            total, per = score(dec, y, window, weight)
            out[f"safety{k}_override{int(override)}"] = {
                "safety": k, "override": override, "score": round(total, 3),
                "per_moment": {m: round(v, 2) for m, v in per.items()}, "calls": float(np.mean(dec != WAIT))}
    return out
