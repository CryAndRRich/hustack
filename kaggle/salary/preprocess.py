from pathlib import Path

import pandas as pd

TARGET = "Annual_salary"
ID = "SR_no"
LOW_CUT = 1.4e6


def get_data_dir() -> Path:
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for p in sorted(kaggle.rglob("Placement_Train.csv")):
            with open(p, encoding="utf-8") as f:
                if "Annual_salary" in f.readline():
                    return p.parent
        raise FileNotFoundError("no Placement_Train.csv with Annual_salary under /kaggle/input")
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def load():
    d = get_data_dir()
    tr = pd.read_csv(d / "Placement_Train.csv")
    te = pd.read_csv(d / "Placement_Test.csv")
    ss = pd.read_csv(d / "Plcement_Sample_Submission.csv")
    assert len(tr) == 84 and len(te) == 52 and (ss[ID].values == te[ID].values).all()
    return tr, te, ss


def features(df):
    return pd.DataFrame({"spec_mkt": (df.specialisation == "Marketing").astype(float),
                         "spec_ops": (df.specialisation == "Operations").astype(float),
                         "GD_Percent": df.GD_Percent.astype(float)}, index=df.index)


def onehot(a, b):
    cols = [c for c in b.columns if c not in (ID, TARGET)]
    x = pd.get_dummies(pd.concat([a[cols], b[cols]]), drop_first=True).astype(float)
    return x.iloc[:len(a)], x.iloc[len(a):]


def strata(tr):
    return (tr[TARGET] < LOW_CUT).astype(int).astype(str) + "_" + tr.specialisation


def distinct_test(te):
    key = te.drop(columns=[ID]).astype(str).agg("|".join, axis=1)
    return ~key.duplicated(keep="first").values, key.values
