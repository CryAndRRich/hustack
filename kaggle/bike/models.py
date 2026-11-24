import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import features

REF = "v2_ref"
EXTREME_DATES = ("2012-10-29", "2012-10-30")
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
INNER_FOLDS = 5
FLOORS = (0, 300, 600, 900)


def arms():
    v2 = {"knot": 21, "calendar": False, "inter": True, "hinge": True, "scale": True}
    out = {REF: {"kind": "ridge", "f": v2, "drop": True, "nb": 2}}
    for k in (14, 21):
        v3 = {**v2, "knot": k, "v3": True}
        out[f"ridge_v3_k{k}"] = {"kind": "ridge", "f": v3, "drop": True, "nb": None}
        out[f"ridge_v3_k{k}_nb2"] = {"kind": "ridge", "f": v3, "drop": True, "nb": 2}
        out[f"split_v3_k{k}"] = {"kind": "ridge_split", "f": v3, "drop": True, "nb": None}
        out[f"split_v3_k{k}_nb2"] = {"kind": "ridge_split", "f": v3, "drop": True, "nb": 2}
    return out


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))


def _base(spec, tr, te):
    if spec["drop"]:
        tr = tr[~tr["date"].isin(EXTREME_DATES)]
    Xtr, Xte = features(tr, **spec["f"]).to_numpy(), features(te, **spec["f"]).to_numpy()
    if spec["kind"] == "ridge":
        return _ridge().fit(Xtr, tr["rentals"]).predict(Xte)
    if spec["kind"] == "ridge_split":
        return sum(_ridge().fit(Xtr, tr[c]).predict(Xte) for c in ("casual", "registered"))
    raise ValueError(spec["kind"])


def _kernel(t_from, r_from, t_to, h, exclude_self):
    w = np.exp(-0.5 * ((t_to[:, None] - t_from[None, :]) / h) ** 2)
    if exclude_self:
        np.fill_diagonal(w, 0.0)
    return (w @ r_from) / np.maximum(w.sum(1), 1e-12)


def fit_predict(spec, tr, te):
    pred = _base(spec, tr, te)
    h = spec["nb"]
    if h is not None:
        inner = np.zeros(len(tr))
        for a, b in KFold(INNER_FOLDS, shuffle=True, random_state=0).split(tr):
            inner[b] = _base(spec, tr.iloc[a], tr.iloc[b])
        r = tr["rentals"].to_numpy(float) - inner
        t_tr, t_te = tr["t"].to_numpy(float), te["t"].to_numpy(float)
        s_tr = _kernel(t_tr, r, t_tr, h, exclude_self=True)
        lam = float(s_tr @ r / max(s_tr @ s_tr, 1e-12))
        pred = pred + lam * _kernel(t_tr, r, t_te, h, exclude_self=False)
    return np.clip(pred, 0.0, None)
