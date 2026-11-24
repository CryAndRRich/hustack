import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from config import FULL_FEATURES, MOFF_FEATURES, RIDGE_GRID, SEED, V1_RIDGE_GRID, V2_FEATURES
from evaluate import auroc
from runtime import log
FEATURE_SETS = {'logit_full': FULL_FEATURES, 'logit_moff': MOFF_FEATURES, 'logit_v2': V2_FEATURES}
ARM_GRID = {'logit_full': V1_RIDGE_GRID}

def _fit_one(x, y, c):
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=c, penalty='l2', solver='lbfgs', max_iter=5000, class_weight='balanced', random_state=SEED)
    model.fit(scaler.transform(x), y)
    return (scaler, model)

def _predict(scaler, model, x):
    return model.decision_function(scaler.transform(x))

def _oof(x, y, groups, c):
    out = np.zeros(len(y))
    for g in np.unique(groups):
        held = groups == g
        scaler, model = _fit_one(x[~held], y[~held], c)
        out[held] = _predict(scaler, model, x[held])
    return out

def _choose_c(x, y, groups, grid=None):
    grid = grid or RIDGE_GRID
    best_c, best = (grid[0], -np.inf)
    for c in grid:
        score = auroc(y, _oof(x, y, groups, c))
        if score > best:
            best_c, best = (c, score)
    return best_c

def logistic_arm(name, train_feat, y, groups, test_feat):
    columns = FEATURE_SETS[name]
    grid = ARM_GRID.get(name)
    x = train_feat[columns].values
    xt = test_feat[columns].values
    oof = np.zeros(len(y))
    for g in np.unique(groups):
        held = groups == g
        c = _choose_c(x[~held], y[~held], groups[~held], grid)
        scaler, model = _fit_one(x[~held], y[~held], c)
        oof[held] = _predict(scaler, model, x[held])
    c_full = _choose_c(x, y, groups, grid)
    scaler, model = _fit_one(x, y, c_full)
    log('%s: C=%s coef=%s' % (name, c_full, dict(zip(columns, np.round(model.coef_[0], 3).tolist()))))
    return (oof, _predict(scaler, model, xt), {'C': c_full, 'columns': columns, 'coef': model.coef_[0].tolist()})

def rank_mean(parts):
    return np.mean([rankdata(p) / len(p) for p in parts], axis=0)

def run_all(train_feat, y, groups, test_feat):
    oof, test, meta = ({}, {}, {})
    oof['cfd'] = train_feat['cfd'].values
    test['cfd'] = test_feat['cfd'].values
    oof['cfd_dp'] = train_feat['cfd'].values * train_feat['distance_factor'].values
    test['cfd_dp'] = test_feat['cfd'].values * test_feat['distance_factor'].values
    for name in FEATURE_SETS:
        oof[name], test[name], meta[name] = logistic_arm(name, train_feat, y, groups, test_feat)
    oof['rank_blend'] = rank_mean([oof['logit_full'], oof['logit_v2']])
    test['rank_blend'] = rank_mean([test['logit_full'], test['logit_v2']])
    return (oof, test, meta)
