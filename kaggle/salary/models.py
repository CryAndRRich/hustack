import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from preprocess import LOW_CUT, TARGET, features, onehot

SG = ["spec_mkt", "spec_ops", "GD_Percent"]


def spec_mean(a, b):
    m = a.groupby("specialisation")[TARGET].mean()
    return b.specialisation.map(m).fillna(a[TARGET].mean()).values


def ridge_sg(a, b):
    xa, xb = features(a)[SG].values, features(b)[SG].values
    sx = StandardScaler().fit(xa)
    ya = a[TARGET].values
    mu, sd = ya.mean(), ya.std()
    m = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(sx.transform(xa), (ya - mu) / sd)
    return m.predict(sx.transform(xb)) * sd + mu


def low_rule(df):
    mk = (df.specialisation == "Marketing").values
    return np.where(mk, df.Avg_GPA_Comm.values <= 2.9, df.Avg_GPA_CT.values <= 2.8)


def v1(a, b):
    low = (a[TARGET] < LOW_CUT).values
    hi = a[~low].reset_index(drop=True)
    high = 0.5 * spec_mean(hi, b) + 0.5 * ridge_sg(hi, b)
    return np.where(low_rule(b), a.loc[low, TARGET].mean(), high)


def trees12(a, b):
    xa, xb = onehot(a, b)
    ms = ([RandomForestRegressor(random_state=s) for s in range(5)] + [ExtraTreesRegressor(random_state=s) for s in range(5)]
          + [GradientBoostingRegressor(random_state=0), XGBRegressor(random_state=0, verbosity=0)])
    return np.mean([m.fit(xa, a[TARGET].values).predict(xb) for m in ms], axis=0)


def boost3(a, b):
    xa, xb = onehot(a, b)
    ms = [GradientBoostingRegressor(random_state=0), XGBRegressor(random_state=0, verbosity=0),
          CatBoostRegressor(random_seed=0, verbose=0, thread_count=4, allow_writing_files=False)]
    return np.mean([m.fit(xa, a[TARGET].values).predict(xb) for m in ms], axis=0)


def lgb5(a, b):
    xa, xb = onehot(a, b)
    p = dict(verbose=-1, min_child_samples=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_estimators=300, learning_rate=0.03)
    return np.mean([LGBMRegressor(random_state=s, **p).fit(xa, a[TARGET].values).predict(xb) for s in range(5)], axis=0)


MEMBERS = {"v1": v1, "trees12": trees12, "boost3": boost3, "lgb5": lgb5}

WEIGHTS = {"v1": 0.15, "trees12": 0.45, "boost3": 0.2, "lgb5": 0.2}
