import pandas as pd
from pathlib import Path

TARGET = "RecoveryTimeHours"
ID = "RecordID"
BLOCK = ["Year", "Season", "MonthPart"]
CALENDAR = ["Year", "MonthPart", "Weekday"]
SITE_DAY = ["Rain1dP90", "Rain3dP90", "Rain7dP90", "LevelMedian24h", "SiteType", "Zone"]


def find(name, column):
    for p in Path("/kaggle/input").rglob(name):
        if column in pd.read_csv(p, nrows=1).columns:
            return p
    try:
        return Path(__file__).resolve().parent.parent / "data" / name
    except NameError:
        return Path.cwd() / "data" / name


def site_day(df):
    key = df[SITE_DAY].astype(str).agg("|".join, axis=1)
    return key.where(df[SITE_DAY].notna().all(axis=1))


def load():
    tr, te = pd.read_csv(find("train.csv", TARGET)), pd.read_csv(find("test.csv", ID))
    raw = [c for c in te.columns if c != ID]
    cats = [c for c in raw if tr[c].dtype == object]
    groups = tr[BLOCK].astype(str).agg("|".join, axis=1).values
    for df in (tr, te):
        df["nan_count"] = df[raw].isna().sum(axis=1)
    tr_sd, te_sd = site_day(tr), site_day(te)
    for c in cats:
        levels = sorted(set(tr[c].dropna()) | set(te[c].dropna()))
        tr[c] = pd.Categorical(tr[c], categories=levels)
        te[c] = pd.Categorical(te[c], categories=levels)
    feats = [c for c in raw if c not in CALENDAR] + ["nan_count"]
    print(f"train {tr.shape} test {te.shape} | features {len(feats)} (no calendar) | block groups {len(set(groups))} | "
          f"site-day keyed rows train {tr_sd.notna().mean():.3f} test {te_sd.notna().mean():.3f}")
    return tr[ID], tr[TARGET].values, te[ID], tr, te, feats, groups, tr_sd, te_sd
