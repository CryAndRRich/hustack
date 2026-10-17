import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "AffordabilityPercentageTrue"
ID = "Zip"
MONTH = "Month"

TRAIN_MONTHS = ["5/1/2026", "6/1/2026"]
TEST_MONTH = "7/1/2026"

BURDEN_THRESHOLD = 0.30

SMOOTHING = 20.0

NEIGHBOUR_K = 10

NUMERIC = [
    "SampleSize", "RentIndex", "AnnualMedianHouseholdIncome", "OwnerPercent",
    "MedianHomeLengthofResidence", "AvgNumberofChildren", "AvgNumberofAdults",
    "AvgHouseholdSize", "AvgGenerationsinHousehold", "YoungAdultinHouseholdPercent",
    "SeniorAdultinHouseholdPercent",
]

CATEGORICAL = ["State", "City", "Metro"]

DATA_DIR = Path("/kaggle/input/competitions/predicting-housing-affordability-in-the-u-s")
EXTERNAL_DIR = Path("/kaggle/input/datasets/cryandrrich/houseafford/external_data")
STORE_DIR = Path("/kaggle/input/datasets/cryandrrich/houseafford/model_weights")
OUTPUT_DIR = Path("/kaggle/working")


def get_data_dir() -> Path:
    return DATA_DIR


def get_output_dir() -> Path:
    return OUTPUT_DIR

EXTERNAL_FILES = [
    "zip_geo.csv", "zip_irs.csv", "zip_zhvi.csv", "zip_zhvi_tier.csv",
    "zip_zhvi_type.csv", "zip_qcew.csv", "zip_fhfa_hpi.csv", "zip_acs_value.csv",
]


def load_external() -> dict[str, pd.DataFrame]:

    d = EXTERNAL_DIR
    if not d.exists():
        print(f"WARNING: {d} not found; falling back to competition columns only.")
        return {}
    print(f"external data: {d}")
    out, absent = {}, []
    for f in EXTERNAL_FILES:
        key = f.replace("zip_", "").replace(".csv", "")
        if (d / f).exists():
            out[key] = pd.read_csv(d / f, dtype={ID: str})
        else:
            absent.append(f)
    if absent:
        print(f"  NOTE: {len(absent)} file(s) not in the dataset, their features will be skipped:")
        for f in absent:
            print(f"    - {f}")
    return out


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    d = get_data_dir()
    kw = {"dtype": {ID: str, "State": str}}
    train = pd.read_csv(d / "train.csv", **kw)
    test = pd.read_csv(d / "test.csv", **kw)
    sample = pd.read_csv(d / "sample_submission.csv", **kw)
    return train, test, sample

ACS_OPEN_TOP = 2.0


def _bins(df: pd.DataFrame, prefix: str) -> tuple:

    cols, lo, hi = [], [], []
    for c in df.columns:
        if not c.startswith(prefix + "_") or c == prefix + "_total":
            continue
        tail = c[len(prefix) + 1:]
        if tail == "nocash":
            a, b = 0.0, 0.0
        else:
            a_s, b_s = tail.split("_")
            a = float(a_s)
            b = a * ACS_OPEN_TOP if b_s == "inf" else float(b_s) + 1.0
        cols.append(c)
        lo.append(a)
        hi.append(b)
    counts = df[cols].to_numpy(dtype=float)
    tot = counts.sum(axis=1, keepdims=True)
    share = np.where(tot > 0, counts / np.where(tot > 0, tot, 1.0), np.nan)
    order = np.argsort(lo, kind="stable")
    return np.array(lo)[order], np.array(hi)[order], share[:, order]


def _cdf(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, share: np.ndarray) -> np.ndarray:

    width = np.where(hi > lo, hi - lo, 1.0)
    frac = np.clip((x[:, :, None] - lo) / width, 0.0, 1.0)
    frac = np.where(hi > lo, frac, (x[:, :, None] >= lo).astype(float))
    return (frac * share[:, None, :]).sum(axis=2)

VAL_MULTIPLES = (2, 3, 4, 5)


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()
    monthly_income = out["AnnualMedianHouseholdIncome"] / 12.0
    out["cost_ratio"] = out["RentIndex"] / monthly_income.replace(0, np.nan)
    out["log_rent"] = np.log1p(out["RentIndex"])
    out["log_income"] = np.log1p(out["AnnualMedianHouseholdIncome"])
    out["income_per_person"] = out["AnnualMedianHouseholdIncome"] / out["AvgHouseholdSize"].replace(0, np.nan)
    out["zip3"] = out[ID].str[:3]

    out["log_ratio_gap"] = np.log(out["cost_ratio"].where(out["cost_ratio"] > 0))
    out["log_ratio_gap"] = out["log_ratio_gap"] - math.log(BURDEN_THRESHOLD)

    out["zip_num"] = out[ID].astype(int)
    return out

ZHVI_BY_MONTH = {TRAIN_MONTHS[0]: "zhvi_may", TRAIN_MONTHS[1]: "zhvi_jun", TEST_MONTH: "zhvi_jul"}
TIER_BY_MONTH = {
    "bot": {TRAIN_MONTHS[0]: "bot_may", TRAIN_MONTHS[1]: "bot_jun", TEST_MONTH: "bot_jul"},
    "top": {TRAIN_MONTHS[0]: "top_may", TRAIN_MONTHS[1]: "top_jun", TEST_MONTH: "top_jul"},
}

TYPE_KEYS = ["br1", "br2", "br3", "br4", "sfr", "condo"]

TYPE_BY_MONTH = {
    k: {TRAIN_MONTHS[0]: f"{k}_may", TRAIN_MONTHS[1]: f"{k}_jun", TEST_MONTH: f"{k}_jul"}
    for k in TYPE_KEYS
}


def merge_external(df: pd.DataFrame, ext: dict[str, pd.DataFrame]) -> pd.DataFrame:

    out = df.copy()
    if not ext:
        return out

    if "geo" in ext:

        geo_cols = [c for c in (ID, "lat", "lon", "county") if c in ext["geo"].columns]
        out = out.merge(ext["geo"][geo_cols], on=ID, how="left")
        if "county" in out.columns:
            out["county"] = out["county"].fillna("").astype(str)
    if "irs" in ext:
        out = out.merge(ext["irs"], on=ID, how="left")
    if "zhvi" in ext:
        out = out.merge(ext["zhvi"], on=ID, how="left")

        zh = pd.Series(np.nan, index=out.index, dtype=float)
        for month, col in ZHVI_BY_MONTH.items():
            if col in out.columns:
                sel = out[MONTH] == month
                zh = zh.where(~sel, out[col])
        out["zhvi"] = zh
        out = out.drop(columns=[c for c in ZHVI_BY_MONTH.values() if c in out.columns])

        annual = out["AnnualMedianHouseholdIncome"].replace(0, np.nan)
        out["price_to_income"] = out["zhvi"] / annual
        out["log_price_to_income"] = np.log(out["price_to_income"].where(out["price_to_income"] > 0))
        out["rent_to_price"] = out["RentIndex"] * 12.0 / out["zhvi"].replace(0, np.nan)
        out["log_zhvi"] = np.log1p(out["zhvi"])

    if "zhvi_type" in ext:
        out = out.merge(ext["zhvi_type"], on=ID, how="left")
        annual = out["AnnualMedianHouseholdIncome"].replace(0, np.nan)
        for key, mapping in TYPE_BY_MONTH.items():
            col = pd.Series(np.nan, index=out.index, dtype=float)
            for month, src in mapping.items():
                if src in out.columns:
                    col = col.where(out[MONTH] != month, out[src])
            out[f"z_{key}"] = col
            out[f"pti_{key}"] = col / annual
            out = out.drop(columns=[c for c in mapping.values() if c in out.columns])

        if "z_sfr" in out.columns and "z_condo" in out.columns:
            out["condo_over_sfr"] = out["z_condo"] / out["z_sfr"].replace(0, np.nan)
        if "z_br2" in out.columns and "z_br4" in out.columns:
            out["br4_over_br2"] = out["z_br4"] / out["z_br2"].replace(0, np.nan)

    if "qcew" in ext:
        cols = [c for c in ext["qcew"].columns if c != "county_fips"]
        out = out.merge(ext["qcew"][cols], on=ID, how="left")

        if "qcew_avg_annual_pay" in out.columns:
            pay = out["qcew_avg_annual_pay"].replace(0, np.nan)
            out["wage_to_income"] = pay / out["AnnualMedianHouseholdIncome"].replace(0, np.nan)
            if "zhvi" in out.columns:
                out["price_to_wage"] = out["zhvi"] / pay

    if "acs_value" in ext:

        vcols = [c for c in ext["acs_value"].columns if c.startswith("val_")]
        out = out.merge(ext["acs_value"][[ID] + vcols], on=ID, how="left")
        lo, hi, share = _bins(out, "val")
        income = out["AnnualMedianHouseholdIncome"].to_numpy(dtype=float)
        within = _cdf(np.outer(income, VAL_MULTIPLES), lo, hi, share)
        for k, m in enumerate(VAL_MULTIPLES):
            out[f"val_share_le_{m}x"] = 100.0 * within[:, k]
        out = out.drop(columns=vcols)

    if "fhfa_hpi" in ext:
        out = out.merge(ext["fhfa_hpi"], on=ID, how="left")

    if "zhvi_tier" in ext:
        out = out.merge(ext["zhvi_tier"], on=ID, how="left")
        for tier, mapping in TIER_BY_MONTH.items():
            col = pd.Series(np.nan, index=out.index, dtype=float)
            for month, src in mapping.items():
                if src in out.columns:
                    col = col.where(out[MONTH] != month, out[src])
            out[f"z_{tier}"] = col
            out = out.drop(columns=[c for c in mapping.values() if c in out.columns])

        annual = out["AnnualMedianHouseholdIncome"].replace(0, np.nan)
        out["pti_bot"] = out["z_bot"] / annual
        out["pti_top"] = out["z_top"] / annual
        if "zhvi" in out.columns:
            out["bot_over_mid"] = out["z_bot"] / out["zhvi"].replace(0, np.nan)
        out["top_over_bot"] = out["z_top"] / out["z_bot"].replace(0, np.nan)
    return out


def add_neighbour_features(fit: pd.DataFrame, frames: list[pd.DataFrame],
                           k: int = NEIGHBOUR_K) -> list[pd.DataFrame]:

    prior = float(fit[TARGET].mean())
    lookup = {}
    for state, g in fit.groupby("State"):
        g = g.sort_values("zip_num")
        lookup[state] = (g["zip_num"].to_numpy(), g[TARGET].to_numpy(dtype=float))

    out = []
    for df in frames:
        vals = np.full(len(df), prior, dtype=float)
        for state, idx in df.groupby("State").indices.items():
            entry = lookup.get(state)
            if entry is None:
                continue
            nums, tgts = entry
            n = len(nums)
            own = df["zip_num"].to_numpy()[idx]
            pos = np.searchsorted(nums, own)

            lo = pos - 1
            hi = pos.copy()
            hi += (hi < n) & (nums[np.clip(hi, 0, n - 1)] == own)
            total = np.zeros(len(idx))
            taken = np.zeros(len(idx), dtype=int)

            for _ in range(k):
                has_lo = lo >= 0
                has_hi = hi < n
                d_lo = np.where(has_lo, own - nums[np.clip(lo, 0, n - 1)], np.inf)
                d_hi = np.where(has_hi, nums[np.clip(hi, 0, n - 1)] - own, np.inf)
                pick_lo = has_lo & (~has_hi | (d_lo <= d_hi))
                pick_hi = has_hi & ~pick_lo
                total += np.where(pick_lo, tgts[np.clip(lo, 0, n - 1)], 0.0)
                total += np.where(pick_hi, tgts[np.clip(hi, 0, n - 1)], 0.0)
                taken += pick_lo.astype(int) + pick_hi.astype(int)
                lo = lo - pick_lo.astype(int)
                hi = hi + pick_hi.astype(int)

            vals[idx] = np.where(taken > 0, total / np.maximum(taken, 1), prior)
        res = df.copy()
        res["nbr_mean"] = vals
        out.append(res)
    return out

GEO_K = 10

GEO_MIN_KM = 0.5


def _unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:

    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def add_geo_neighbour_features(fit: pd.DataFrame, frames: list[pd.DataFrame],
                               k: int = GEO_K) -> list[pd.DataFrame]:

    prior = float(fit[TARGET].mean())
    empty = {"nbr_geo_mean": prior, "nbr_geo_std": 0.0, "nbr_geo_dist": np.nan}

    if not {"lat", "lon"}.issubset(fit.columns):
        return [df.assign(**empty) for df in frames]

    have = fit.dropna(subset=["lat", "lon"])
    if not len(have):
        return [df.assign(**empty) for df in frames]

    agg = have.groupby(have[ID])[TARGET].mean()
    coords = have.groupby(have[ID])[["lat", "lon"]].first()
    ref = coords.join(agg)
    ref_ids = ref.index.to_numpy()
    ref_xyz = _unit_xyz(ref["lat"].to_numpy(), ref["lon"].to_numpy())
    ref_y = ref[TARGET].to_numpy(dtype=float)
    n_ref = len(ref_ids)

    out = []
    for df in frames:
        res = df.copy()
        mean = np.full(len(df), prior)
        std = np.zeros(len(df))
        dist_mean = np.full(len(df), np.nan)

        has_xy = {"lat", "lon"}.issubset(df.columns)
        ok = ((df["lat"].notna() & df["lon"].notna()).to_numpy() if has_xy
              else np.zeros(len(df), dtype=bool))
        if ok.any() and n_ref > 1:
            q_idx = np.flatnonzero(ok)
            q_xyz = _unit_xyz(df["lat"].to_numpy()[ok], df["lon"].to_numpy()[ok])
            q_ids = df[ID].to_numpy()[ok]
            id_pos = {z: i for i, z in enumerate(ref_ids)}

            for start in range(0, len(q_idx), 512):
                sl = slice(start, start + 512)
                dots = np.clip(q_xyz[sl] @ ref_xyz.T, -1.0, 1.0)
                d_km = np.arccos(dots) * 6371.0

                for r, z in enumerate(q_ids[sl]):
                    pos = id_pos.get(z)
                    if pos is not None:
                        d_km[r, pos] = np.inf

                kk = min(k, n_ref - 1)
                nn = np.argpartition(d_km, kk - 1, axis=1)[:, :kk]
                nd = np.take_along_axis(d_km, nn, axis=1)
                ny = ref_y[nn]

                finite = np.isfinite(nd)
                w = np.where(finite, 1.0 / np.maximum(nd, GEO_MIN_KM), 0.0)
                tot = w.sum(axis=1)
                rows = q_idx[sl]
                good = tot > 0
                mean[rows[good]] = (ny * w).sum(axis=1)[good] / tot[good]
                std[rows[good]] = np.where(finite, ny, np.nan)[good].std(axis=1)
                with np.errstate(invalid="ignore"):
                    dist_mean[rows[good]] = np.where(finite, nd, np.nan)[good].mean(axis=1)

        res["nbr_geo_mean"] = mean
        res["nbr_geo_std"] = np.nan_to_num(std, nan=0.0)
        res["nbr_geo_dist"] = dist_mean
        out.append(res)
    return out

FEAT_NBR_COLS = ["price_to_income", "log_income", "OwnerPercent"]
FEAT_NBR_K = 10


def add_feature_neighbour_features(fit: pd.DataFrame, frames: list[pd.DataFrame],
                                   k: int = FEAT_NBR_K) -> list[pd.DataFrame]:

    cols = [c for c in FEAT_NBR_COLS if c in fit.columns]
    if len(cols) < 2:
        return [df.assign(feat_nbr_std=np.nan) for df in frames]

    have = fit.dropna(subset=cols)
    if len(have) < k + 1:
        return [df.assign(feat_nbr_std=np.nan) for df in frames]

    agg = have.groupby(have[ID])[cols].mean()
    tgt = have.groupby(have[ID])[TARGET].mean()
    ref = agg.join(tgt)
    ref_ids = ref.index.to_numpy()

    mu = ref[cols].mean()
    sd = ref[cols].std(ddof=0).replace(0.0, 1.0)
    ref_z = ((ref[cols] - mu) / sd).to_numpy(dtype=float)
    ref_y = ref[TARGET].to_numpy(dtype=float)
    n_ref = len(ref_ids)

    out = []
    for df in frames:
        std = np.full(len(df), np.nan)
        ok = df[cols].notna().all(axis=1).to_numpy() if set(cols).issubset(df.columns) \
            else np.zeros(len(df), dtype=bool)
        if ok.any() and n_ref > 1:
            q_idx = np.flatnonzero(ok)
            q_z = ((df.loc[ok, cols] - mu) / sd).to_numpy(dtype=float)
            q_ids = df[ID].to_numpy()[ok]
            id_pos = {z: i for i, z in enumerate(ref_ids)}
            kk = min(k, n_ref - 1)

            for start in range(0, len(q_idx), 512):
                sl = slice(start, start + 512)
                d = ((q_z[sl][:, None, :] - ref_z[None, :, :]) ** 2).sum(axis=2)

                for r, z in enumerate(q_ids[sl]):
                    pos = id_pos.get(z)
                    if pos is not None:
                        d[r, pos] = np.inf
                nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
                std[q_idx[sl]] = ref_y[nn].std(axis=1)

        res = df.copy()
        res["feat_nbr_std"] = std
        out.append(res)
    return out


def add_relative_features(fit: pd.DataFrame, frames: list[pd.DataFrame],
                         k: int = GEO_K) -> list[pd.DataFrame]:

    src = "price_to_income"
    if src not in fit.columns or not {"lat", "lon"}.issubset(fit.columns):
        return list(frames)

    have = fit.dropna(subset=["lat", "lon", src])
    if len(have) < 2:
        return [df.assign(pti_relative=np.nan) for df in frames]

    ref = have.groupby(have[ID]).agg(lat=("lat", "first"), lon=("lon", "first"),
                                     v=(src, "mean"))
    ref_xyz = _unit_xyz(ref["lat"].to_numpy(), ref["lon"].to_numpy())
    ref_v = ref["v"].to_numpy(dtype=float)
    n_ref = len(ref)

    out = []
    for df in frames:
        rel = np.full(len(df), np.nan)
        ok = (np.zeros(len(df), dtype=bool)
              if not {"lat", "lon", src}.issubset(df.columns)
              else (df["lat"].notna() & df["lon"].notna() & df[src].notna()).to_numpy())
        if ok.any() and n_ref > 1:
            q_idx = np.flatnonzero(ok)
            q_xyz = _unit_xyz(df["lat"].to_numpy()[ok], df["lon"].to_numpy()[ok])
            own = df[src].to_numpy(dtype=float)[ok]
            for start in range(0, len(q_idx), 512):
                sl = slice(start, start + 512)
                d = np.arccos(np.clip(q_xyz[sl] @ ref_xyz.T, -1.0, 1.0)) * 6371.0
                kk = min(k, n_ref)
                nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
                local = np.nanmean(ref_v[nn], axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel[q_idx[sl]] = own[sl] / np.where(local > 0, local, np.nan)
        res = df.copy()
        res["pti_relative"] = np.where(np.isfinite(rel), rel, np.nan)
        out.append(res)
    return out

OOF_FOLDS = 5
OOF_SEED = 20260921


def oof_folds(train: pd.DataFrame, n_folds: int = OOF_FOLDS,
              seed: int = OOF_SEED) -> pd.Series:

    zips = np.sort(train[ID].unique())
    rng = np.random.default_rng(seed)
    assign = rng.permutation(len(zips)) % n_folds
    lookup = dict(zip(zips, assign))
    return train[ID].map(lookup).astype(int)


def oof_fingerprint(train: pd.DataFrame, n_folds: int = OOF_FOLDS,
                    seed: int = OOF_SEED) -> str:

    folds = oof_folds(train, n_folds, seed)
    payload = f"{len(train)}|{n_folds}|{seed}|".encode()
    payload += pd.util.hash_pandas_object(
        train[ID].astype(str) + ":" + folds.astype(str), index=False
    ).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]
