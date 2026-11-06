from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Revenue"
RAW_COLS = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "SpecialDay", "Month",
            "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jul": 7, "Aug": 8, "Sep": 9,
          "Oct": 10, "Nov": 11, "Dec": 12}
CAT_COLS = ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "MonthCat"]
RARE_MIN = 20
PAGES = [("Administrative", "Administrative_Duration"), ("Informational", "Informational_Duration"),
         ("ProductRelated", "ProductRelated_Duration")]


def find_files():
    roots = [Path("/kaggle/input")]
    try:
        roots.append(Path(__file__).resolve().parent.parent / "data")
    except NameError:
        roots.append(Path.cwd() / "data")
    found = {}
    for root in roots:
        for p in sorted(root.rglob("*.csv")) if root.exists() else []:
            cols = set(pd.read_csv(p, nrows=1).columns)
            if cols == {"Id", TARGET}:
                kind = "sample"
            elif "ExitRates" in cols:
                kind = "train" if TARGET in cols else "test"
            else:
                continue
            found.setdefault(kind, p)
        if len(found) == 3:
            break
    missing = {"train", "test", "sample"} - set(found)
    assert not missing, f"competition files not found: {missing}; attach the competition data"
    print({k: str(v) for k, v in found.items()})
    return found


def load():
    f = find_files()
    tr, te, ss = pd.read_csv(f["train"]), pd.read_csv(f["test"]), pd.read_csv(f["sample"])
    for df in (tr, te):
        missing = set(RAW_COLS + ["Id"]) - set(df.columns)
        assert not missing, f"missing columns {missing}"
    assert (ss["Id"].values == te["Id"].values).all(), "sample_submission order differs from test"
    assert tr[TARGET].isin([0, 1]).all()
    print(f"train {tr.shape}, test {te.shape}, positive rate {tr[TARGET].mean():.4f}")
    return tr, te


def features(tr, te):
    both = pd.concat([tr[RAW_COLS], te[RAW_COLS]], ignore_index=True)
    f = pd.DataFrame(index=both.index)
    for cnt, dur in PAGES:
        f[cnt] = both[cnt].astype(float)
        f[dur + "_log"] = np.log1p(both[dur].clip(lower=0))
        f[cnt + "_per_page_log"] = np.log1p(both[dur].clip(lower=0) / both[cnt].clip(lower=1))
    pages = both[[c for c, _ in PAGES]].sum(axis=1).astype(float)
    dur = both[[d for _, d in PAGES]].clip(lower=0).sum(axis=1)
    f["pages_total"] = pages
    f["dur_total_log"] = np.log1p(dur)
    f["dur_per_page_log"] = np.log1p(dur / pages.clip(lower=1))
    f["product_share"] = both["ProductRelated"] / pages.clip(lower=1)
    f["admin_share"] = both["Administrative"] / pages.clip(lower=1)
    f["BounceRates"] = both["BounceRates"]
    f["ExitRates"] = both["ExitRates"]
    f["exit_minus_bounce"] = both["ExitRates"] - both["BounceRates"]
    f["exit_over_bounce"] = both["ExitRates"] / (both["BounceRates"] + 1e-3)
    f["exit_x_pages"] = both["ExitRates"] * pages
    f["bounce_eq_exit"] = (both["BounceRates"] == both["ExitRates"]).astype(float)
    f["zero_duration"] = (dur == 0).astype(float)
    f["bot_like"] = ((both["BounceRates"] >= 0.2) & (both["ExitRates"] >= 0.2)).astype(float)
    f["SpecialDay"] = both["SpecialDay"]
    month = both["Month"].map(MONTHS)
    assert month.notna().all(), f"unknown months {set(both['Month']) - set(MONTHS)}"
    f["month_num"] = month.astype(float)
    f["q4"] = (month >= 10).astype(float)
    f["Weekend"] = both["Weekend"].astype(str).str.lower().eq("true").astype(float)

    n_tr = len(tr)
    for c in CAT_COLS:
        raw = both["Month"] if c == "MonthCat" else both[c]
        s = raw.astype(str)
        counts = s.iloc[:n_tr].value_counts()
        keep = set(counts[counts >= RARE_MIN].index)
        s = s.where(s.isin(keep), "rare")
        f[c] = pd.Categorical(s, categories=sorted(keep | {"rare"}))
    num = [c for c in f.columns if c not in CAT_COLS]
    f[num] = f[num].astype(float)
    assert np.isfinite(f[num].values).all(), "non-finite feature"
    X_tr, X_te = f.iloc[:n_tr].reset_index(drop=True), f.iloc[n_tr:].reset_index(drop=True)
    print(f"features: {len(num)} numeric + {len(CAT_COLS)} categorical")
    return X_tr, X_te, CAT_COLS
