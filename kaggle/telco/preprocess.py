from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

TARGET, ID = "Churn", "customerID"
TENURE_MAX = 72
YN = ["No", "Yes"]
NET = ["No", "Yes", "No internet service"]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CATS = {"gender": ["Female", "Male"], "Partner": YN, "Dependents": YN, "PhoneService": YN,
        "MultipleLines": ["No", "Yes", "No phone service"], "InternetService": ["DSL", "Fiber optic", "No"],
        **{c: NET for c in ADDONS}, "Contract": ["Month-to-month", "One year", "Two year"], "PaperlessBilling": YN,
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]}
PAYMENT = {"electronic": "Electronic check", "mailed": "Mailed check", "bank": "Bank transfer (automatic)",
           "credit": "Credit card (automatic)"}
SENIOR = {"0": 0.0, "1": 1.0, "0.0": 0.0, "1.0": 1.0, "no": 0.0, "yes": 1.0}
TENURE_BINS = [-1, 0, 2, 6, 12, 24, 48, TENURE_MAX]
TE_GROUPS = [("Contract", "InternetService", "PaymentMethod"), ("Contract", "tenure_bin")]
TE_SMOOTH = 20


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
    d = find("sample_submission.csv")
    tr, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    assert tr[ID].is_unique and te[ID].is_unique and not tr[ID].isin(te[ID]).any()
    assert set(tr[TARGET].unique()) == {0, 1}
    print(f"train {tr.shape}, test {te.shape}, churn rate {tr[TARGET].mean():.4f}")
    return tr, te


def number(s):
    t = s.astype(str).str.replace("VND", "", regex=False).str.strip()
    t = t.mask(t.isin(["", "nan", "None"]))
    v = pd.to_numeric(t, errors="coerce")
    bad = t.notna() & v.isna()
    assert not bad.any(), f"unparsed values: {t[bad].unique()[:5]}"
    return v


def clean(df):
    d = df.copy()
    for c in CATS:
        d[c] = d[c].astype(str).str.strip()
    pm = d.PaymentMethod.str.lower()
    d["PaymentMethod"] = pm.map(lambda s: next((v for k, v in PAYMENT.items() if k in s), s))
    for c, allowed in CATS.items():
        bad = ~d[c].isin(allowed)
        assert not bad.any(), f"{c}: unknown values {d.loc[bad, c].unique()[:5]}"
    sc = d.SeniorCitizen.astype(str).str.strip().str.lower()
    d["SeniorCitizen"] = sc.map(SENIOR)
    assert d.SeniorCitizen.notna().sum() == sc.isin(list(SENIOR)).sum()
    d["tenure"] = number(d.tenure)
    d.loc[(d.tenure < 0) | (d.tenure > TENURE_MAX), "tenure"] = np.nan
    d["MonthlyCharges"] = number(d.MonthlyCharges)
    d["TotalCharges"] = number(d.TotalCharges)
    return d


def services(d):
    cols = [d.PhoneService.eq("Yes"), d.MultipleLines.eq("Yes"), d.InternetService.eq("DSL"),
            d.InternetService.eq("Fiber optic")] + [d[c].eq("Yes") for c in ADDONS]
    return np.column_stack([np.ones(len(d))] + [c.to_numpy(float) for c in cols])


class Imputer:
    def __init__(self, mc_fill="ratio"):
        self.mc_fill = mc_fill

    def fit(self, d):
        ok = d.MonthlyCharges.notna()
        self.price = np.linalg.lstsq(services(d[ok]), d.MonthlyCharges[ok].to_numpy(), rcond=None)[0]
        self.tenure_med = d.groupby("Contract").tenure.median()
        self.senior = d.SeniorCitizen.mean()
        return self

    def mc_hat(self, d):
        return pd.Series(services(d) @ self.price, index=d.index)

    def transform(self, d):
        d = d.copy()
        t, mc, tc = d.tenure, d.MonthlyCharges, d.TotalCharges
        d["tenure"] = t.fillna(np.clip(np.round(tc / mc), 1, TENURE_MAX))
        hat = self.mc_hat(d)
        ratio = (tc / d.tenure).where(d.tenure > 0)
        fill = {"ratio": ratio.fillna(hat), "services": hat, "avg": ((ratio + hat) / 2).fillna(hat)}[self.mc_fill]
        d["MonthlyCharges"] = mc.fillna(fill)
        d["tenure"] = d.tenure.fillna(np.clip(np.round(tc / d.MonthlyCharges), 1, TENURE_MAX))
        d["tenure"] = d.tenure.fillna(d.Contract.map(self.tenure_med))
        d["TotalCharges"] = tc.fillna(d.MonthlyCharges * d.tenure)
        d["SeniorCitizen"] = d.SeniorCitizen.fillna(self.senior)
        d["mc_hat"] = hat
        assert not d[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]].isna().any().any()
        return d


def one_hot(d):
    return pd.concat([pd.DataFrame({f"{c}={v}": d[c].eq(v).astype(float) for v in vals}, index=d.index)
                      for c, vals in CATS.items()], axis=1)


def target_encode(a, b, y, seed=0):
    prior = y.mean()

    def encode(src, ys, dst, key):
        g = pd.DataFrame({"k": src[key], "y": ys}).groupby("k").y.agg(["sum", "count"])
        p = dst[key].map((g["sum"] + TE_SMOOTH * prior) / (g["count"] + TE_SMOOTH)).fillna(prior)
        return np.log(p / (1 - p)).to_numpy()

    out_a, out_b = {}, {}
    for grp in TE_GROUPS:
        name = "te_" + "_".join(grp)
        ka, kb = a[list(grp)].astype(str).agg("|".join, axis=1), b[list(grp)].astype(str).agg("|".join, axis=1)
        A, B = pd.DataFrame({"k": ka}), pd.DataFrame({"k": kb})
        col = np.empty(len(a))
        for i, j in StratifiedKFold(5, shuffle=True, random_state=seed).split(A, y):
            col[j] = encode(A.iloc[i], y[i], A.iloc[j], "k")
        out_a[name], out_b[name] = col, encode(A, y, B, "k")
    return pd.DataFrame(out_a, index=a.index), pd.DataFrame(out_b, index=b.index)


def features(d):
    t, mc, tc = d.tenure, d.MonthlyCharges, d.TotalCharges
    blocks = {
        "num": pd.DataFrame({"tenure": t, "MonthlyCharges": mc, "TotalCharges": tc, "SeniorCitizen": d.SeniorCitizen}),
        "onehot": one_hot(d),
        "log": pd.DataFrame({"log_tc": np.log1p(tc), "log_tenure": np.log1p(t)}),
        "eng": pd.DataFrame({"n_services": services(d)[:, 1:].sum(1), "mc_resid": mc - d.mc_hat,
                             "drift": (tc / t.where(t > 0) - mc).fillna(0)}, index=d.index),
    }
    return blocks


def make(tr, te, y, cfg):
    if cfg.get("starter"):
        cols = ["tenure", "MonthlyCharges"]
        f = lambda df: df[cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0)
        return f(tr), f(te)
    a, b = clean(tr), clean(te)
    imp = Imputer(cfg.get("mc_fill", "ratio")).fit(a)
    a, b = imp.transform(a), imp.transform(b)
    fa, fb = features(a), features(b)
    keep = ["num"] + [k for k in ("onehot", "log", "eng") if cfg.get(k)]
    Xa, Xb = pd.concat([fa[k] for k in keep], axis=1), pd.concat([fb[k] for k in keep], axis=1)
    if cfg.get("spline"):
        sp = SplineTransformer(n_knots=6, degree=3, extrapolation="linear").fit(a[["tenure", "MonthlyCharges"]])
        names = sp.get_feature_names_out()
        Xa = Xa.join(pd.DataFrame(sp.transform(a[["tenure", "MonthlyCharges"]]), columns=names, index=a.index))
        Xb = Xb.join(pd.DataFrame(sp.transform(b[["tenure", "MonthlyCharges"]]), columns=names, index=b.index))
    if cfg.get("te"):
        a["tenure_bin"] = pd.cut(a.tenure, TENURE_BINS).astype(str)
        b["tenure_bin"] = pd.cut(b.tenure, TENURE_BINS).astype(str)
        ta, tb = target_encode(a, b, np.asarray(y))
        Xa, Xb = Xa.join(ta), Xb.join(tb)
    if cfg.get("scale"):
        sc = StandardScaler().fit(Xa)
        Xa = pd.DataFrame(sc.transform(Xa), columns=Xa.columns, index=Xa.index)
        Xb = pd.DataFrame(sc.transform(Xb), columns=Xb.columns, index=Xb.index)
    if cfg.get("inter"):
        assert cfg.get("scale"), "interactions are built on standardised columns"
        for X, d in ((Xa, a), (Xb, b)):
            for v in CATS["Contract"]:
                X[f"tenure_x_{v}"] = X.tenure * d.Contract.eq(v)
            fiber = d.InternetService.eq("Fiber optic")
            X["mc_x_fiber"], X["tenure_x_fiber"] = X.MonthlyCharges * fiber, X.tenure * fiber
            X["echeck_x_m2m"] = (d.PaymentMethod.eq("Electronic check") & d.Contract.eq("Month-to-month")).astype(float)
    assert list(Xa.columns) == list(Xb.columns) and np.isfinite(Xa.to_numpy()).all() and np.isfinite(Xb.to_numpy()).all()
    return Xa, Xb
