from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "addicted_label"
ID_COL = "id"
EPS = 1e-5

RAW_NUMERIC = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
RAW_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]

COMPONENT_COLS = ["social_media_hours", "gaming_hours", "work_study_hours"]

IMPUTE_COLS = [
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "app_opens_per_day",
    "notifications_per_day",
    "age",
]

NA_FLAG_COLS = [
    "daily_screen_time_hours",
    "social_media_hours",
    "weekend_screen_time",
    "gaming_hours",
    "work_study_hours",
]

IMPUTE_ROUNDS = 2
IMPUTE_N_ESTIMATORS = 250

ADD_IMPUTED_RATIOS = False

DIGIT_FEATURES = [
    ("daily_screen_time_hours", 1),
    ("daily_screen_time_hours", 2),
    ("social_media_hours", 1),
    ("gaming_hours", 1),
    ("sleep_hours", 2),
    ("weekend_screen_time", 1),
    ("weekend_screen_time", 2),
]


def get_data_dir() -> Path:
    kaggle_dir = Path("/kaggle/input/competitions/playground-series-s6e8")
    if kaggle_dir.exists():
        return kaggle_dir
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def load_data(nrows=None):
    data_dir = get_data_dir()
    train = pd.read_csv(data_dir / "train.csv", nrows=nrows)
    test = pd.read_csv(data_dir / "test.csv", nrows=nrows)
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv", nrows=nrows)
    return train, test, sample_submission


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()

    o["missing_count"] = o[RAW_NUMERIC + RAW_CATEGORICAL].isna().sum(axis=1)

    o["entertainment_hours"] = o["social_media_hours"] + o["gaming_hours"]
    o["entertainment_ratio"] = o["entertainment_hours"] / (o["daily_screen_time_hours"] + EPS)
    o["screen_to_sleep_ratio"] = o["daily_screen_time_hours"] / (o["sleep_hours"] + EPS)
    o["other_screen_time"] = (
        o["daily_screen_time_hours"] - o["social_media_hours"] - o["gaming_hours"] - o["work_study_hours"]
    )
    o["weekend_vs_daily"] = o["weekend_screen_time"] - o["daily_screen_time_hours"]
    o["notifications_per_open"] = o["notifications_per_day"] / (o["app_opens_per_day"] + EPS)
    o["work_ratio"] = o["work_study_hours"] / (o["daily_screen_time_hours"] + EPS)
    o["social_ratio"] = o["social_media_hours"] / (o["daily_screen_time_hours"] + EPS)
    o["social_share_of_entertainment"] = o["social_media_hours"] / (o["entertainment_hours"] + EPS)
    o["screen_sum"] = o["daily_screen_time_hours"] + o["weekend_screen_time"]
    o["weekend_ratio"] = o["weekend_screen_time"] / (o["daily_screen_time_hours"] + EPS)

    o["comp_sum_known"] = o[COMPONENT_COLS].sum(axis=1, min_count=1)
    o["comp_n_known"] = o[COMPONENT_COLS].notna().sum(axis=1).astype("int8")
    o["daily_minus_lower"] = o["daily_screen_time_hours"] - o["comp_sum_known"]
    o["social_upper"] = o["daily_screen_time_hours"] - o[["gaming_hours", "work_study_hours"]].sum(axis=1, min_count=1)
    o["gaming_upper"] = o["daily_screen_time_hours"] - o[["social_media_hours", "work_study_hours"]].sum(axis=1, min_count=1)
    o["work_upper"] = o["daily_screen_time_hours"] - o[["social_media_hours", "gaming_hours"]].sum(axis=1, min_count=1)

    for col in NA_FLAG_COLS:
        o["na_" + col] = o[col].isna().astype("int8")

    cents_cache = {}
    for col, place in DIGIT_FEATURES:
        if col not in cents_cache:
            cents_cache[col] = np.rint(o[col].values * 100)
        cents = cents_cache[col]
        digit = (cents // 10) % 10 if place == 1 else cents % 10
        o[f"digit{place}_{col}"] = np.where(np.isnan(cents), -1, digit).astype("int8")

    return o


def impute_features(train_raw, test_raw, rounds=IMPUTE_ROUNDS, n_estimators=IMPUTE_N_ESTIMATORS,
                    verbose=True):
    import lightgbm as lgb

    n_train = len(train_raw)
    cols = RAW_NUMERIC + RAW_CATEGORICAL
    both = pd.concat([train_raw[cols], test_raw[cols]], ignore_index=True)
    base = both[RAW_NUMERIC].copy()
    cat_codes = pd.DataFrame(
        {c: both[c].astype("category").cat.codes.astype("int16") for c in RAW_CATEGORICAL},
        index=both.index,
    )

    work = base.copy()
    for col in RAW_NUMERIC:
        work[col] = work[col].fillna(base[col].median())

    lo, hi = base.min(), base.max()
    for r in range(rounds):
        for col in IMPUTE_COLS:
            missing = base[col].isna().values
            if not missing.any():
                continue
            predictors = [c for c in RAW_NUMERIC if c != col]
            X = pd.concat([work[predictors], cat_codes], axis=1)
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators, learning_rate=0.08, num_leaves=63,
                min_child_samples=40, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                verbosity=-1, n_jobs=-1,
            )
            model.fit(X[~missing], base.loc[~missing, col])
            work.loc[missing, col] = np.clip(model.predict(X[missing]), lo[col], hi[col])
        if verbose:
            print(f"  imputation round {r + 1}/{rounds} done")

    work.columns = [c + "_imp" for c in work.columns]
    return (work.iloc[:n_train].reset_index(drop=True),
            work.iloc[n_train:].reset_index(drop=True))


def add_imputed_ratios(out: pd.DataFrame, imp: pd.DataFrame) -> pd.DataFrame:
    d = imp["daily_screen_time_hours_imp"].values
    s = imp["social_media_hours_imp"].values
    g = imp["gaming_hours_imp"].values
    w = imp["work_study_hours_imp"].values
    wk = imp["weekend_screen_time_imp"].values

    out["i_entertainment_ratio"] = (s + g) / (d + EPS)
    out["i_other_screen_time"] = d - s - g - w
    out["i_weekend_vs_daily"] = wk - d
    out["i_social_ratio"] = s / (d + EPS)
    out["i_work_ratio"] = w / (d + EPS)
    return out


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()
    for col in RAW_CATEGORICAL:
        combined = pd.concat([train[col], test[col]], ignore_index=True).astype("string").fillna("missing")
        codes, categories = pd.factorize(combined, sort=True)
        train[col] = pd.Categorical.from_codes(codes[: len(train)], categories=categories)
        test[col] = pd.Categorical.from_codes(codes[len(train):], categories=categories)
    return train, test, RAW_CATEGORICAL


def prepare(train_raw: pd.DataFrame, test_raw: pd.DataFrame, use_imputation: bool = True,
            impute_rounds: int = IMPUTE_ROUNDS, impute_n_estimators: int = IMPUTE_N_ESTIMATORS):
    train_feat = build_features(train_raw)
    test_feat = build_features(test_raw)

    if use_imputation:
        print("Chained GBDT imputation (feature-only, leak-safe)...")
        imp_train, imp_test = impute_features(
            train_raw, test_raw, rounds=impute_rounds, n_estimators=impute_n_estimators
        )
        train_feat = pd.concat([train_feat.reset_index(drop=True), imp_train], axis=1)
        test_feat = pd.concat([test_feat.reset_index(drop=True), imp_test], axis=1)
        if ADD_IMPUTED_RATIOS:
            train_feat = add_imputed_ratios(train_feat, imp_train)
            test_feat = add_imputed_ratios(test_feat, imp_test)

    train_feat, test_feat, cat_cols = encode_categoricals(train_feat, test_feat)

    y = train_feat[TARGET].astype("int8")
    feature_cols = [c for c in train_feat.columns if c not in (ID_COL, TARGET)]
    return train_feat[feature_cols], y, test_feat[feature_cols], cat_cols
