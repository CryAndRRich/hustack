import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ALPHAS = [1.0, 3.0, 10.0, 30.0]


def sparse_designs(text, base):
    t0 = time.time()
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True,
                           strip_accents="unicode").fit_transform(text.name + " . " + text.desc)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=100000,
                           sublinear_tf=True).fit_transform(text.name)
    amen = CountVectorizer(tokenizer=lambda s: s.split("\t"), token_pattern=None, lowercase=False, binary=True,
                           min_df=3).fit_transform(text.amen)
    size = ("acc_" + base.accommodates.astype(str) + " bed_" + base.bedrooms.astype(str) + " beds_"
            + base.beds.astype(str) + " bath_" + base.baths.astype(str) + "_" + base.bath_shared.astype(str))
    cats = CountVectorizer(token_pattern=r"\S+", lowercase=False, binary=True).fit_transform(text.cats + " " + size)
    m_text = sp.hstack([word, char, amen]).tocsr()
    m_full = sp.hstack([word, char, amen, cats]).tocsr()
    print(f"sparse designs: text {m_text.shape[1]} cols, full {m_full.shape[1]} cols ({time.time() - t0:.0f}s)")
    return {"ridge_text": m_text, "ridge_full": m_full}


def fit_ridge(M, y, alpha):
    mu = y.mean()
    return Ridge(alpha=alpha, solver="sparse_cg", fit_intercept=False).fit(M, y - mu), mu


def choose_alpha(M, y, seed):
    a, b = next(KFold(5, shuffle=True, random_state=seed).split(y))
    scores = {}
    for alpha in ALPHAS:
        r, mu = fit_ridge(M[a], y[a], alpha)
        scores[alpha] = float(np.sqrt(np.mean((r.predict(M[b]) + mu - y[b]) ** 2)))
    best = min(scores, key=scores.get)
    print("  alpha " + ", ".join(f"{k:g}: {v:.4f}" for k, v in scores.items()) + f" -> {best:g}")
    return best


def ridge_features(designs, alphas, fit_rows, y_fit, apply_rows, inner):
    fit, apps = {}, [{} for _ in apply_rows]
    for name, M in designs.items():
        Mf = M[fit_rows]
        f = np.zeros(len(fit_rows))
        for a, b in inner:
            r, mu = fit_ridge(Mf[a], y_fit[a], alphas[name])
            f[b] = r.predict(Mf[b]) + mu
        fit[name] = f
        r, mu = fit_ridge(Mf, y_fit, alphas[name])
        for rows, app in zip(apply_rows, apps):
            app[name] = r.predict(M[rows]) + mu
    return fit, apps
