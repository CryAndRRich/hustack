from pathlib import Path

import numpy as np
import pandas as pd

TARGET, ID = "PeakLoadMW", "RecordID"
STATIC_NUM = ["Homes", "Shops", "GridAge", "Stalls", "ApartmentPct", "CapacityMW"]
STATIC_CAT = ["City", "Area", "FeederGroup", "YearBand", "HomesBand", "ShopBand", "AgeBand", "BackupBand"]
ROW_NUM = ["DayOffset", "Lighting", "Stalls", "TempConn", "AwayPct", "BackupPct", "TempC", "Humidity", "Cloud",
           "Outages30", "OutageFlag", "MaintFlag", "EventLevel", "Footfall", "HomeActivity"]
ROW_CAT = ["Slot", "Festival", "Weekday", "MarketBand", "DiwaliPhase"]
SLOTS = ["00-03", "03-06", "06-09", "09-12", "12-15", "15-18", "18-21", "21-24"]
FEEDER_MEAN = ["Lighting", "TempConn", "AwayPct", "BackupPct", "Outages30", "MaintFlag", "EventLevel", "Footfall",
               "HomeActivity", "TempC"]
MIN_KEY_ROWS = 20
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAYS = np.arange(-7, 8)
FESTIVAL_DAYS = {"Choti Diwali": [-1], "Diwali": [0], "Govardhan Puja": [1], "Bhai Dooj": [2, 3],
                 "Normal": [d for d in DAYS if d not in (-1, 0, 1, 2)]}
PHASE_DAYS = {"Pre": range(-7, -3), "LeadUp": range(-3, 0), "Diwali": [0], "After": range(1, 4), "Post": range(4, 8)}
DAY_CONST = ["AwayPct", "Stalls", "Outages30"]
DAY_SOFT = ["Lighting", "EventLevel", "TempConn"]
SLOT_SOFT = ["TempC", "Humidity", "Footfall", "HomeActivity", "Cloud"]
WEATHER = ["TempC", "Humidity", "Cloud"]


def find(name):
    root = Path("/kaggle/input")
    hits = sorted(root.rglob(name)) if root.exists() else []
    if hits:
        return hits[0].parent
    try:
        return Path(__file__).resolve().parent.parent / "data" / "raw"
    except NameError:
        return Path.cwd() / "data" / "raw"


def load():
    d = find("data_dictionary.csv")
    tr, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    assert tr[ID].is_unique and te[ID].is_unique and not tr[ID].isin(te[ID]).any()
    assert tr[TARGET].notna().all() and (tr[TARGET] > 0).all()
    print(f"train {tr.shape}, test {te.shape}, target mean {tr[TARGET].mean():.4f} sd {tr[TARGET].std():.4f}")
    return tr, te


def mode(s):
    m = s.mode()
    return m.iloc[0] if len(m) else np.nan


def mismatch(al, prof):
    score = np.zeros((len(al), len(prof)))
    for c in STATIC_CAT:
        v, p = al[c].to_numpy(object), prof[c].to_numpy(object)
        obs = al[c].notna().to_numpy()
        score[obs] += v[obs, None] != p[None, :]
    for c in STATIC_NUM:
        v, p = al[c].to_numpy(float), prof[c].to_numpy(float)
        obs = ~np.isnan(v)
        score[obs] += np.minimum(5 * np.abs(v[obs, None] - p[None, :]) / np.maximum(np.abs(p[None, :]), 1e-3), 1)
    return score


def assign_feeders(al):
    vc = al["Homes"].value_counts()
    key = al["Homes"].where(al["Homes"].isin(vc[vc >= MIN_KEY_ROWS].index))
    prof = al.assign(key=key).dropna(subset=["key"]).groupby("key")[STATIC_NUM + STATIC_CAT].agg(mode)
    score = mismatch(al, prof)
    has = key.notna().to_numpy()
    blind = score[has].argmin(1)
    print(f"feeders {len(prof)}; held-out check on {has.sum()} rows with a clean Homes value: "
          f"{(prof.index.to_numpy()[blind] == key[has].to_numpy()).mean():.4f} recovered without it")
    srt = np.sort(score, 1)
    fid = score.argmin(1)
    fid[has] = pd.Index(prof.index).get_indexer(key[has])
    margin = srt[:, 1] - srt[:, 0]
    print(f"rows with an ambiguous feeder (margin < 0.5, no clean Homes): {((margin < 0.5) & ~has).sum()}")
    return fid, prof.reset_index(drop=True)


def constrain(C, labels, mapping):
    for lab, ds in mapping.items():
        m = (labels == lab).to_numpy()
        new = C[m] & np.isin(DAYS, list(ds))[None, :]
        C[m] = np.where(new.any(1, keepdims=True), new, C[m])


def recover_days(al, fid):
    d_obs = al["DayOffset"].to_numpy(float)
    wd = al["Weekday"].map({w: i for i, w in enumerate(WEEKDAYS)}).to_numpy(float)
    k = ~np.isnan(d_obs) & ~np.isnan(wd)
    w0 = pd.Series((wd[k] - d_obs[k]) % 7).groupby(fid[k]).agg(lambda s: s.mode().iloc[0])
    print(f"weekday of DayOffset 0 agrees with its feeder's mode on {np.mean((wd[k] - d_obs[k]) % 7 == w0.reindex(fid[k]).to_numpy()):.4f} of rows")
    C = np.ones((len(al), len(DAYS)), bool)
    r = (wd - w0.reindex(fid).to_numpy()) % 7
    m = ~np.isnan(r)
    C[m] = (DAYS[None, :] % 7) == r[m, None]
    constrain(C, al["Festival"], FESTIVAL_DAYS)
    constrain(C, al["DiwaliPhase"], PHASE_DAYS)
    lab = pd.DataFrame({"f": fid, "d": d_obs}).join(al[DAY_CONST + DAY_SOFT]).dropna(subset=["d"])
    const = lab.groupby(["f", "d"])[DAY_CONST].agg(mode)
    soft = lab.groupby(["f", "d"])[DAY_SOFT].mean()
    sd = lab.groupby(["f", "d"])[DAY_SOFT].std().median().to_numpy()
    score = np.zeros(C.shape)
    X, Y = al[DAY_CONST].to_numpy(float), al[DAY_SOFT].to_numpy(float)
    for j, d in enumerate(DAYS):
        idx = pd.MultiIndex.from_arrays([fid, np.full(len(al), float(d))])
        P, M = const.reindex(idx).to_numpy(float), soft.reindex(idx).to_numpy(float)
        score[:, j] += np.where(np.isnan(X) | np.isnan(P), 0, np.where(np.isclose(X, P), -3, 1)).sum(1)
        score[:, j] += np.where(np.isnan(Y) | np.isnan(M), 0, np.minimum(((Y - M) / sd) ** 2, 9) / 2).sum(1)
    score = np.where(C, score, np.inf)
    best = DAYS[score.argmin(1)].astype(float)
    tied = (score == score.min(1, keepdims=True)).sum(1) > 1
    best[tied] = np.nan
    obs = ~np.isnan(d_obs)
    best[obs] = d_obs[obs]
    print(f"DayOffset known {obs.mean():.3f} -> {np.mean(~np.isnan(best)):.3f}; unknown rows narrowed to one day "
          f"by labels alone: {np.mean(C[~obs].sum(1) == 1):.3f}")
    return best, w0.reindex(fid).to_numpy()


def recover_slots(al, fid, day):
    slot = al["Slot"].map({s: i for i, s in enumerate(SLOTS)}).to_numpy(float)
    known0 = np.mean(~np.isnan(slot))
    for _ in range(5):
        df = pd.DataFrame({"f": fid, "d": day, "s": slot})
        taken = df.dropna().groupby(["f", "d"]).s.agg(set)
        todo = df.index[df.s.isna() & df.d.notna()]
        free = [set(range(len(SLOTS))) - taken.get((df.f[i], df.d[i]), set()) for i in todo]
        hit = [(i, f.pop()) for i, f in zip(todo, free) if len(f) == 1]
        if not hit:
            break
        for i, s in hit:
            slot[i] = s
    print(f"Slot known {known0:.3f} -> {np.mean(~np.isnan(slot)):.3f}")
    return slot


def build(tr, te, grid):
    al = pd.concat([tr.drop(columns=TARGET), te], ignore_index=True)
    fid, prof = assign_feeders(al)
    X = pd.DataFrame({"feeder": fid})
    for c in STATIC_NUM + STATIC_CAT:
        X["p_" + c] = prof[c].to_numpy()[fid]
    for c in ROW_NUM:
        X[c] = al[c].to_numpy(float)
    for c in ROW_CAT:
        X[c] = al[c].to_numpy(object)
    X["Slot"] = X["Slot"].map({s: i for i, s in enumerate(SLOTS)})
    for c in FEEDER_MEAN:
        X["fm_" + c] = X.groupby("feeder")[c].transform("mean")
    if grid:
        X["day"], X["year"] = recover_days(al, fid)
        X["slot"] = recover_slots(al, fid, X["day"].to_numpy())
        for keys, cols in ((["feeder", "day"], DAY_CONST + DAY_SOFT), (["feeder", "slot"], SLOT_SOFT),
                           (["p_City", "year", "day", "slot"], WEATHER)):
            name = "_".join(k[0] for k in keys)
            for c in cols:
                X[f"{name}_{c}"] = X.groupby(keys)[c].transform("mean")
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
    n = len(tr)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)
