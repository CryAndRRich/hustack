import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from design import BINARY, FULL_CATS, LEAN_CATS, LINEAR_INPUTS, Design
from runtime import log
from tobit import fit_tobit


def qr_linear(cfg, Xtr, ytr):
    design = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), LINEAR_INPUTS),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), FULL_CATS),
    ])
    model = make_pipeline(design, QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"))
    model.fit(Xtr, ytr)
    return model.predict


def ols(cats, cond):
    def fit(cfg, Xtr, ytr):
        d = Design(cats, cond).fit(Xtr)
        b = np.linalg.lstsq(d.transform(Xtr)[0], ytr, rcond=None)[0]
        return lambda X: d.transform(X)[0] @ b
    return fit


def tobit(cats, cond, extras=False):
    def fit(cfg, Xtr, ytr):
        d = Design(cats, cond, extras, cfg.spline_knots).fit(Xtr)
        D, pen = d.transform(Xtr)
        lam = choose_lambda(cfg, Xtr, ytr, cats, cond) if extras else 0.0
        b, sigma = fit_tobit(D, ytr, cfg.lo, pen, lam)
        log(f"    tobit cats={len(cats)} cond={cond} extras={extras} lambda={lam:g} sigma={sigma:.4f}"
            f" params={D.shape[1]}")
        return lambda X: np.maximum(d.transform(X)[0] @ b, cfg.lo)
    return fit


def choose_lambda(cfg, Xtr, ytr, cats, cond):
    fold = np.random.default_rng(cfg.seed).permutation(len(ytr)) % cfg.inner_folds
    scores = {lam: 0.0 for lam in cfg.lambdas}
    for f in range(cfg.inner_folds):
        a, v = fold != f, fold == f
        d = Design(cats, cond, True, cfg.spline_knots).fit(Xtr[a])
        Da, pen = d.transform(Xtr[a])
        Dv = d.transform(Xtr[v])[0]
        for lam in cfg.lambdas:
            b, _ = fit_tobit(Da, ytr[a], cfg.lo, pen, lam)
            scores[lam] += np.abs(ytr[v] - np.maximum(Dv @ b, cfg.lo)).sum()
    return min(scores, key=scores.get)


ARMS = {
    "qr_linear": qr_linear,
    "ols_full": ols(FULL_CATS, False),
    "tobit_full": tobit(FULL_CATS, False),
    "tobit_lean": tobit(LEAN_CATS, False),
    "tobit_full_cond": tobit(FULL_CATS, True),
    "tobit_lean_cond": tobit(LEAN_CATS, True),
    "tobit_spline": tobit(FULL_CATS, True, extras=True),
}
REFERENCE, CANDIDATE = "qr_linear", "tobit_full"
