import numpy as np
import pandas as pd

from runtime import find_csvs, log

ID, TARGET = "record_id", "resolution_days"
CATS = ["province_group", "region_tier", "court_jurisdiction", "service_channel", "dispute_category",
        "vulnerability_group", "gender_category"]
NUMS = ["distance_to_court_km", "travel_cost_idr_thousands", "digital_connectivity_index", "legal_aid_available",
        "fee_waiver_prodeo", "digital_intake_score", "paralegal_hours", "case_complexity_index"]
NAN_COLS = ["digital_intake_score", "paralegal_hours"]


def load():
    found = {}
    for p in find_csvs():
        cols = set(pd.read_csv(p, nrows=0).columns)
        if {ID, TARGET} <= cols:
            found.setdefault("train", p)
        elif {ID, "predicted_resolution_days"} == cols:
            found.setdefault("sample", p)
        elif {ID, *CATS, *NUMS} <= cols:
            found.setdefault("test", p)
    assert set(found) == {"train", "test", "sample"}, f"competition csv files not found: {found}"
    tr, te, ss = (pd.read_csv(found[k]) for k in ("train", "test", "sample"))
    log("files", {k: str(v) for k, v in found.items()}, "shapes", tr.shape, te.shape, ss.shape)
    assert len(tr) == 8500 and len(te) == 1500 and (ss[ID].values == te[ID].values).all()
    assert tr[TARGET].notna().all() and not set(tr[ID]) & set(te[ID])
    return tr, te, ss


def features(tr, te):
    both = add_features(pd.concat([tr, te], ignore_index=True))
    for c in CAT_FEATURES:
        both[c] = both[c].astype(str).astype("category")
    return both.iloc[:len(tr)].reset_index(drop=True), both.iloc[len(tr):].reset_index(drop=True)


def add_features(df):
    out = df[CATS + NUMS].copy()
    for c in NAN_COLS:
        out[c + "_nan"] = out[c].isna().astype(float)
    out["log_distance"] = np.log(out.distance_to_court_km)
    out["log_cost"] = np.log(out.travel_cost_idr_thousands)
    out["cost_per_km"] = out.travel_cost_idr_thousands / out.distance_to_court_km
    out["cost_excess"] = out.travel_cost_idr_thousands - 5.0 * out.distance_to_court_km
    out["tier_channel"] = out.region_tier.astype(str) + "|" + out.service_channel.astype(str)
    return out


CAT_FEATURES = CATS + ["tier_channel"]
NUM_FEATURES = NUMS + [c + "_nan" for c in NAN_COLS] + ["log_distance", "log_cost", "cost_per_km", "cost_excess"]
