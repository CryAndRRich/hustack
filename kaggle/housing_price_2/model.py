# Updated on June 30, 2025, 03:43 PM
# Public Score: 360007.28
# Rank: 155/394

import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score
from bayes_opt import BayesianOptimization
from catboost import CatBoostRegressor

from preprocess import Data

def winkler_score_batch(y_true: np.ndarray, 
                        y_lower: np.ndarray, 
                        y_upper: np.ndarray) -> np.ndarray:
    interval_width = y_upper - y_lower
    penalty = np.zeros_like(y_true, dtype=float)

    below = y_true < y_lower
    above = y_true > y_upper
    in_interval = ~below & ~above
    coverage = np.mean(in_interval)
    alpha = 1.0 - coverage
    factor = 0.0 if alpha == 0 else 2.0 / alpha

    penalty[below] = factor * (y_lower[below] - y_true[below])
    penalty[above] = factor * (y_true[above] - y_upper[above])

    return interval_width + penalty


def winkler_scorer(estimator, 
                   X: np.ndarray, 
                   y: np.ndarray) -> float:
    preds = estimator.predict(X)
    lower, upper = preds[:, 0], preds[:, 1]
    return np.mean(winkler_score_batch(y.to_numpy(), lower, upper))


class DoubleQuantileCat(Pipeline, BaseEstimator, RegressorMixin):
    def __init__(self, 
                 params, 
                 alpha_lower: float = 0.05, 
                 alpha_upper: float = 0.95) -> None:
        self.params = params
        self.alpha_lower = alpha_lower
        self.alpha_upper = alpha_upper

        self.preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features)
        ])

        steps = [
            ("preprocessor", self.preprocessor),
            ("regressor", None)
        ]
        super().__init__(steps=steps)

    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray) -> None:
        X_t = self.named_steps["preprocessor"].fit_transform(X, y)

        lower_params = self.params.copy()
        lower_params.update({
            "loss_function": f"Quantile:alpha={self.alpha_lower}",
            "verbose": False
        })
        self.model_lower_ = CatBoostRegressor(**lower_params)
        self.model_lower_.fit(X_t, y)

        upper_params = self.params.copy()
        upper_params.update({
            "loss_function": f"Quantile:alpha={self.alpha_upper}",
            "verbose": False
        })
        self.model_upper_ = CatBoostRegressor(**upper_params)
        self.model_upper_.fit(X_t, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = self.named_steps["preprocessor"].transform(X)
        lower = self.model_lower_.predict(X_t)
        upper = self.model_upper_.predict(X_t)
        return np.vstack([lower, upper]).T


class HousingPrice2Model():
    def __init__(self, 
                 data_path: str, 
                 data_processed_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        data = pd.read_csv(data_path)

        self.train_data = data[data["id"] <= 199999].copy()
        self.test_data = data[data["id"] > 199999].copy()

        self.test_ids = self.test_data["id"]
        self.X_train = self.train_data.drop(columns=["sale_price", "id"])
        self.y_train = self.train_data["sale_price"]
        self.X_test = self.test_data.drop(columns=["sale_price", "id"])

        self.categorical_features = (
            self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        )
        self.categorical_features += ["grade", "fbsmt_grade", "condition"]
        self.numerical_features = [c for c in self.X_train.columns if c not in self.categorical_features]

    def _bayes_opt(self) -> None:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        def cat_cv_winkler(depth: float, 
                           learning_rate: float, 
                           subsample: float, 
                           colsample_bylevel: float,
                           l2_leaf_reg: float, 
                           bagging_temperature: float, 
                           random_strength: float, 
                           border_count: float) -> float:
            params = {
                "iterations": 1000,
                "depth": int(depth),
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bylevel": colsample_bylevel,
                "l2_leaf_reg": l2_leaf_reg,
                "bagging_temperature": bagging_temperature,
                "random_strength": random_strength,
                "border_count": int(border_count),
                "random_seed": 42
            }
            model = DoubleQuantileCat(params, alpha_lower=0.05, alpha_upper=0.95)
            scores = cross_val_score(model, self.X_train, self.y_train,
                                     cv=cv, scoring=winkler_scorer, n_jobs=1)
            return -scores.mean()

        pbounds = {
            "depth": (3, 10),
            "learning_rate": (0.01, 0.1),
            "subsample": (0.5, 1.0),
            "colsample_bylevel": (0.5, 1.0),
            "l2_leaf_reg": (1, 10),
            "bagging_temperature": (0.0, 1.0),
            "random_strength": (0.0, 1.0),
            "border_count": (32, 255)
        }

        optimizer = BayesianOptimization(
            f=cat_cv_winkler, pbounds=pbounds,
            random_state=42, verbose=2
        )
        optimizer.maximize(init_points=5, n_iter=15)

        best = optimizer.max["params"]
        best["depth"] = int(best["depth"])
        best["border_count"] = int(best["border_count"])

        self.best_params_cat = best

    def train(self) -> None:
        self.model = DoubleQuantileCat(self.best_params_cat, alpha_lower=0.05, alpha_upper=0.95)
        self.model.fit(self.X_train, self.y_train)

    def predict(self) -> None:
        pi = self.model.predict(self.X_test)
        pi_lower = np.expm1(pi[:, 0])
        pi_upper = np.expm1(pi[:, 1])

        submission = pd.DataFrame({
            "id": self.test_ids,
            "pi_lower": pi_lower,
            "pi_upper": pi_upper
        })
        submission.to_csv("housing_price2_submission.csv", index=False)
        print("Submission saved to housing_price2_submission.csv")


if __name__ == "__main__":
    data_path = "housing_price_2/data"

    data = Data(data_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = HousingPrice2Model(data_path, data_processed_path)
    model.train()
    model.predict()