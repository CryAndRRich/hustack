import os
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_FILE, TEST_FILE = "covid_train_2026.csv", "covid_test_2026.csv"
TARGET = "tested_positive"
FEATS = ["cli", "ili", "hh_cmnty_cli", "nohh_cmnty_cli", "wearing_mask", "travel_outside_state",
         "work_outside_home", "shop", "restaurant", "spent_time", "large_event", "public_transit",
         "anxious", "depressed", "felt_isolated", "worried_become_ill", "worried_finances"]
DAYS = ("", ".1", ".2")


def find_data():
    for dirpath, _, files in os.walk("/kaggle/input"):
        if TRAIN_FILE in files and TEST_FILE in files:
            return Path(dirpath)
    try:
        return Path(__file__).resolve().parent.parent / "data" / "raw"
    except NameError:
        return Path.cwd() / "data" / "raw"


def load():
    d = find_data()
    tr, te = pd.read_csv(d / TRAIN_FILE), pd.read_csv(d / TEST_FILE)
    assert len(tr) == 2000 and len(te) == 700 and TARGET + ".2" not in te
    states = list(tr.columns[1:41])
    df = pd.concat([tr.assign(split="train"), te.assign(split="test")], ignore_index=True)
    assert (df[states].sum(axis=1) == 1).all()
    df["state"] = df[states].to_numpy().argmax(1)
    print(f"data {d}: train {len(tr)}, test {len(te)}, states {len(states)}")
    return df


def build_panel(df):
    cols = FEATS + [TARGET]
    d0, d1 = (df[[c + s for c in cols]].to_numpy() for s in DAYS[:2])
    d2 = df[[c + DAYS[2] for c in FEATS]].to_numpy()
    y2 = df[TARGET + DAYS[2]].to_numpy()
    assert np.isfinite(d0).all() and np.isfinite(d1).all() and np.isfinite(d2).all()
    first = {}
    for i, (st, v) in enumerate(zip(df["state"], d0)):
        assert (st, v.tobytes()) not in first, "duplicate day block"
        first[(st, v.tobytes())] = i
    nxt = np.array([first.get((st, v.tobytes()), -1) for st, v in zip(df["state"], d1)])
    linked = nxt[nxt >= 0]
    assert len(set(linked)) == len(linked)
    segments = []
    for h in sorted(set(range(len(df))) - set(linked)):
        order = [h]
        while nxt[order[-1]] >= 0:
            order.append(nxt[order[-1]])
        segments.append(order)
    assert sum(map(len, segments)) == len(df)

    parts, seg, t = [], np.empty(len(df), int), np.empty(len(df), int)
    for s, order in enumerate(segments):
        last = order[-1]
        vals = np.vstack([d0[order], d1[last], np.append(d2[last], y2[last])])
        for k, i in enumerate(order):
            assert np.array_equal(d1[i], vals[k + 1]) and np.array_equal(d2[i], vals[k + 2, :-1])
            assert np.isnan(y2[i]) or y2[i] == vals[k + 2, -1]
        seg[order], t[order] = s, np.arange(len(order)) + 2
        p = pd.DataFrame(vals, columns=cols)
        p.insert(0, "t", np.arange(len(vals)))
        p.insert(0, "state", df["state"].iat[last])
        p.insert(0, "seg", s)
        parts.append(p)
    panel = pd.concat(parts, ignore_index=True)
    start = panel.groupby("seg").size().cumsum().shift(fill_value=0).to_numpy()
    df["seg"], df["t"] = seg, t
    df["y_chain"] = panel[TARGET].to_numpy()[start[seg] + t]

    tr, te = df["split"] == "train", df["split"] == "test"
    assert (df.loc[tr, "y_chain"] == df.loc[tr, TARGET + ".2"]).all()
    n_open = int(te.sum() - df.loc[te, "y_chain"].notna().sum())
    assert n_open == panel[TARGET].isna().sum()
    print(f"segments {len(segments)} (states {df['state'].nunique()}), days per segment "
          f"{panel.groupby('seg').size().min()}-{panel.groupby('seg').size().max()}; "
          f"test targets from the chain {int(te.sum()) - n_open}/{int(te.sum())}, chain-end forecasts {n_open}")
    return panel
