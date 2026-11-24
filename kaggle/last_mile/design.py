import numpy as np
from sklearn.preprocessing import SplineTransformer

from data import CAT_FEATURES, NAN_COLS, NUM_FEATURES

BINARY = [c for c in NUM_FEATURES if c in ("legal_aid_available", "fee_waiver_prodeo") or c.endswith("_nan")]
CONTINUOUS = [c for c in NUM_FEATURES if c not in BINARY]
LINEAR_INPUTS = [c for c in CONTINUOUS if c != "cost_excess"]
SPLINE_INPUTS = [c for c in LINEAR_INPUTS if not c.startswith("log_") and c != "travel_cost_idr_thousands"]
FULL_CATS = [c for c in CAT_FEATURES if c not in ("region_tier", "service_channel")]
LEAN_CATS = ["vulnerability_group", "tier_channel"]
CHANNEL = "service_channel"


def one_hot(X, cols):
    blocks = [(X[c].cat.codes.to_numpy()[:, None] == np.arange(1, len(X[c].cat.categories))).astype(float)
              for c in cols]
    return np.hstack(blocks)


class Design:

    def __init__(self, cats, cond_impute, extras=False, knots=8):
        self.cats, self.cond_impute, self.extras, self.knots = cats, cond_impute, extras, knots

    def _numeric(self, X):
        N = X[LINEAR_INPUTS].copy()
        for c in NAN_COLS:
            miss = N[c].isna().to_numpy()
            if self.cond_impute and miss.any():
                N.loc[miss, c] = self._predictors(X[miss]) @ self.imp[c]
            N[c] = N[c].fillna(self.median[c])
        return N

    def _predictors(self, X):
        P = X[[c for c in LINEAR_INPUTS if c not in NAN_COLS]].to_numpy(float)
        return np.hstack([np.ones((len(X), 1)), P, X[BINARY].to_numpy(float), one_hot(X, CAT_FEATURES)])

    def fit(self, X):
        self.median = X[NAN_COLS].median()
        if self.cond_impute:
            self.imp = {}
            for c in NAN_COLS:
                ok = X[c].notna().to_numpy()
                self.imp[c] = np.linalg.lstsq(self._predictors(X[ok]), X.loc[ok, c].to_numpy(float), rcond=None)[0]
        N = self._numeric(X).to_numpy(float)
        self.mean, self.sd = N.mean(0), N.std(0)
        if self.extras:
            self.spline = SplineTransformer(n_knots=self.knots, degree=3, knots="quantile", include_bias=False)
            self.s_mean = self.spline.fit_transform(self._numeric(X)[SPLINE_INPUTS]).mean(0)
        return self

    def transform(self, X):
        Nf = self._numeric(X)
        N = (Nf.to_numpy(float) - self.mean) / self.sd
        base = np.hstack([np.ones((len(X), 1)), N, X[BINARY].to_numpy(float), one_hot(X, self.cats)])
        if not self.extras:
            return base, np.zeros(base.shape[1])
        S = self.spline.transform(Nf[SPLINE_INPUTS]) - self.s_mean
        ch = one_hot(X, [CHANNEL])
        inter = np.hstack([ch[:, [k]] * N for k in range(ch.shape[1])])
        D = np.hstack([base, S, inter])
        pen = np.r_[np.zeros(base.shape[1]), np.ones(S.shape[1] + inter.shape[1])]
        return D, pen
