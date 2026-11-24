import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import features
from config import SEED, TRANSFER_MAX_ITER
from runtime import log

def _within_guide_z(guides, reads):
    target = np.zeros(len(reads))
    for g in np.unique(guides):
        m = guides == g
        values = np.log(reads[m])
        sd = values.std()
        target[m] = (values - values.mean()) / sd if sd > 0 else 0.0
    return target

def fit(external, smoke=False):
    if smoke:
        keep = external['guide'].drop_duplicates().head(8)
        external = external[external['guide'].isin(keep)].reset_index(drop=True)
    x = features.build(external)[features.SEQUENCE_FEATURES].values
    y = _within_guide_z(external['guide'].values, external['reads'].values.astype(float))
    log('transfer: %d rows, %d guides' % (len(external), external['guide'].nunique()))
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=1.0, random_state=SEED).fit(scaler.transform(x), y)
    gbdt = HistGradientBoostingRegressor(max_iter=TRANSFER_MAX_ITER, max_depth=4, learning_rate=0.06, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED).fit(x, y)
    log('transfer: ridge R2 %.4f, gbdt R2 %.4f (in-sample, diagnostic only)' % (ridge.score(scaler.transform(x), y), gbdt.score(x, y)))
    return (scaler, ridge, gbdt)

def apply(models, feat):
    scaler, ridge, gbdt = models
    x = feat[features.SEQUENCE_FEATURES].values
    return {'tx_ridge': ridge.predict(scaler.transform(x)), 'tx_gbdt': gbdt.predict(x)}
