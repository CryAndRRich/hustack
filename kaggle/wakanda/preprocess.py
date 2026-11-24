from pathlib import Path

import numpy as np
import pandas as pd

N_GENES = 500
DEMO = ["sex", "age", "height", "weight"]


def find_data():
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hits = sorted(kaggle.rglob("gen_data.csv"))
        if hits:
            return hits[0]
    try:
        base = Path(__file__).resolve().parent.parent / "data"
    except NameError:
        base = Path.cwd() / "data"
    hits = sorted(base.rglob("gen_data.csv"))
    if not hits:
        raise FileNotFoundError("gen_data.csv not found under /kaggle/input or data/")
    return hits[0]


def confounders(df):
    sex = df["sex"].to_numpy(float)
    age = df["age"].to_numpy(float)
    h = df["height"].to_numpy(float)
    w = df["weight"].to_numpy(float)
    bmi = w / h ** 2
    cols = {"sex": sex, "age": age, "height": h, "weight": w,
            "age2": age ** 2, "height2": h ** 2, "weight2": w ** 2, "bmi": bmi, "bmi2": bmi ** 2,
            "age_sex": age * sex, "height_sex": h * sex, "weight_sex": w * sex}
    Z = np.column_stack([(v - v.mean()) / v.std() for v in cols.values()])
    return np.column_stack([np.ones(len(df)), Z]), ["intercept", *cols]


def linear_confounders(df):
    Z = df[DEMO].to_numpy(float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    return np.column_stack([np.ones(len(df)), Z])


def residualise(M, C):
    coef, *_ = np.linalg.lstsq(C, M, rcond=None)
    return M - C @ coef


def load():
    path = find_data()
    df = pd.read_csv(path)
    genes = [f"Gene_{j}" for j in range(1, N_GENES + 1)]
    assert list(df.columns) == ["wellbeing_index", *genes, *DEMO], "unexpected columns"
    assert df.notna().all().all(), "missing values"
    G = df[genes].to_numpy(float)
    assert set(np.unique(G)) <= {0.0, 1.0}, "genes are not 0/1"
    assert (G.std(0) > 0).all(), f"constant genes: {np.flatnonzero(G.std(0) == 0) + 1}"
    y = df["wellbeing_index"].to_numpy(float)
    C, cnames = confounders(df)
    print(f"data {path}: n={len(df)} genes={G.shape[1]} confounder columns={C.shape[1]}")
    Gt = residualise(G, C)
    return {"df": df, "y": y, "G": G, "C": C, "cnames": cnames, "C_lin": linear_confounders(df), "Gt": Gt, "A": Gt.T @ Gt}


def project(y, data, C=None):
    if C is None:
        Gt, C, A = data["Gt"], data["C"], data["A"]
    else:
        Gt = residualise(data["G"], C)
        A = Gt.T @ Gt
    yt = residualise(y[:, None], C)[:, 0]
    return {"A": A, "b": Gt.T @ yt, "yy": float(yt @ yt), "dof": len(y) - np.linalg.matrix_rank(C)}
