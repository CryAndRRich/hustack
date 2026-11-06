from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "CO2 uptake at 0.15 bar and 298K"
EPS = 1e-6


def find(name):
    root = Path("/kaggle/input")
    hits = sorted(root.rglob(name)) if root.exists() else []
    if hits:
        return hits[0].parent
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def add_ratios(a):
    a["inaccessible"] = (a["ASA [m^2/cm^3]"] == 0).astype(int)
    a["asa_share"] = a["ASA [m^2/cm^3]"] / (a.total_SA_volumetric + EPS)
    a["poav_share"] = a.POAVF / (a.total_POV_volumetric + EPS)
    a["di_df"] = a.Di / (a.Df + EPS)
    a["dif_di"] = a.Dif / (a.Di + EPS)
    return ["inaccessible", "asa_share", "poav_share", "di_df", "dif_di"]


def load():
    d = find("test.csv")
    tr, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    e = find("ext_core2019.csv")
    ex, hy = pd.read_csv(e / "ext_core2019.csv"), pd.read_csv(e / "ext_hypo.csv")
    base = [c for c in te.columns if c != "MOFname"]
    assert len(base) == 272 and TARGET in tr and set(base) <= set(tr.columns), (len(base), d)
    n0 = len(tr)
    tr = tr.drop_duplicates(subset=["MOFname", *base, TARGET]).reset_index(drop=True)
    assert tr.MOFname.is_unique, "duplicate names with differing rows"
    assert te.MOFname.is_unique and len(te) == 2000
    assert not ex.MOFname.isin(pd.concat([tr.MOFname, te.MOFname])).any() and set(base) <= set(ex.columns)
    assert set(base) <= set(hy.columns) and hy.MOFname.is_unique
    for x, v in ((tr, 0), (ex, 1), (hy, 2), (te, 0)):
        x["y"], x["ext"] = (np.nan if x is te else x[TARGET].astype(float)), v
    a = pd.concat([x[["MOFname", *base, "y", "ext"]] for x in (tr, ex, hy, te)], ignore_index=True)
    a["hypothetical"] = (a.ext == 2).astype(int)
    features = base + add_ratios(a) + ["hypothetical"]
    a["group"] = a.MOFname.str.upper().str[:6].where(a.ext < 2, a.MOFname)
    print(f"train {n0} -> {len(tr)} after dropping exact duplicates, external {len(ex)}, hypothetical {len(hy)}, "
          f"test {len(te)}, features {len(features)}")
    print(f"y mean {tr.y.mean():.4f} std {tr.y.std():.4f}; groups {a.group[a.y.notna()].nunique()}")
    return a, features
