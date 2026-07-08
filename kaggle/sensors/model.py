# Updated on June 12, 2026, 8:01 PM
# Public Score: 0.89201
# Private Score: 0.89144
# Rank: 1/39

import os
import warnings
import numpy as np
import pandas as pd
import optuna
from tqdm.auto import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.base import BaseEstimator, TransformerMixin

from preprocess import Data

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class MKLTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.5, gamma=1.0, degree=3, coef0=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X, y=None):
        self.X_train_ = X
        return self

    def transform(self, X):
        K_rbf = rbf_kernel(X, self.X_train_, gamma=self.gamma)
        K_poly = polynomial_kernel(X, self.X_train_, degree=self.degree, coef0=self.coef0, gamma=None)
        return self.alpha * K_rbf + (1 - self.alpha) * K_poly


class SensorsModel():
    def __init__(self,
                 data_path: str,
                 data_processed_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        data = pd.read_csv(data_path)

        train_data = data[data["faulty"].notna()].copy()
        test_data = data[data["faulty"].isna()].copy()

        self.test_ids = test_data["id"].astype(int)
        self.y = train_data["faulty"].values.astype(int)
        self.X_train = train_data.drop(columns=["id", "faulty"]).values
        self.X_test = test_data.drop(columns=["id", "faulty"]).values

    def _build_pipeline(self, params: dict) -> Pipeline:
        scaler = RobustScaler() if params["scaler"] == "robust" else StandardScaler()
        return Pipeline([
            ("scaler", scaler),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("power", PowerTransformer(method="yeo-johnson")),
            ("selector", SelectKBest(score_func=f_classif, k=params["k"])),
            ("mkl", MKLTransformer(
                alpha=params["alpha"],
                gamma=params["gamma"],
                degree=params["degree"],
                coef0=params["coef0"]
            )),
            ("svm", SVC(
                kernel="precomputed",
                C=params["C"],
                class_weight={0: 1.0, 1: params["weight_class_1"]},
                probability=True,
                cache_size=1500
            ))
        ])

    def _optuna_base(self, n_trials: int = 500) -> None:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def objective(trial) -> float:
            params = {
                "scaler": trial.suggest_categorical("scaler", ["robust", "standard"]),
                "k": trial.suggest_int("k", 50, 200),
                "weight_class_1": trial.suggest_float("weight_class_1", 1.0, 5.0),
                "C": trial.suggest_float("C", 1.0, 1000, log=True),
                "alpha": trial.suggest_float("alpha", 0.0, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
                "degree": trial.suggest_int("degree", 2, 3),
                "coef0": trial.suggest_float("coef0", -1.0, 1.0)
            }
            pipe = self._build_pipeline(params)
            oof_probas = cross_val_predict(
                pipe, self.X_train, self.y, cv=skf, method="predict_proba", n_jobs=-1
            )[:, 1]
            return max(
                f1_score(self.y, (oof_probas >= t).astype(int), average="macro")
                for t in np.linspace(0.2, 0.8, 50)
            )

        pbar = tqdm(total=n_trials, desc="Base model search")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda s, t: pbar.update(1)])
        self.study_base = study

    def _build_oof_matrix(self, top_n: int = 10, n_splits: int = 10) -> None:
        top_trials = sorted(
            self.study_base.trials,
            key=lambda t: t.value if t.value is not None else 0,
            reverse=True
        )[:top_n]

        self.oof_matrix = np.zeros((self.X_train.shape[0], top_n))
        self.test_matrix = np.zeros((self.X_test.shape[0], top_n))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2026)

        for i, trial in enumerate(top_trials):
            print(f"Model {i + 1}/{top_n} (F1={trial.value:.4f})")
            test_probas = np.zeros(self.X_test.shape[0])
            for tr_idx, va_idx in skf.split(self.X_train, self.y):
                pipe = self._build_pipeline(trial.params)
                pipe.fit(self.X_train[tr_idx], self.y[tr_idx])
                self.oof_matrix[va_idx, i] = pipe.predict_proba(self.X_train[va_idx])[:, 1]
                test_probas += pipe.predict_proba(self.X_test)[:, 1] / n_splits
            self.test_matrix[:, i] = test_probas

    def _optuna_meta(self, n_trials: int = 100) -> None:
        top_n = self.oof_matrix.shape[1]

        def objective(trial) -> float:
            weights = np.array([trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(top_n)])
            if np.sum(weights) == 0:
                return 0.0
            weights /= np.sum(weights)
            blend = np.dot(self.oof_matrix, weights)
            thresh = trial.suggest_float("threshold", 0.2, 0.8)
            return f1_score(self.y, (blend >= thresh).astype(int), average="macro")

        pbar = tqdm(total=n_trials, desc="Meta weight search")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda s, t: pbar.update(1)])

        best = study.best_params
        self.best_threshold = best.pop("threshold")
        raw_weights = np.array([best[f"w_{i}"] for i in range(top_n)])
        self.final_weights = raw_weights / np.sum(raw_weights)

        print(f"Meta F1: {study.best_value:.4f}")
        print(f"Threshold: {self.best_threshold:.4f}")

    def predict(self) -> None:
        final_probas = np.dot(self.test_matrix, self.final_weights)
        preds = (final_probas >= self.best_threshold).astype(int)

        submission = pd.DataFrame({
            "id": self.test_ids,
            "faulty": preds
        })
        outfile = os.path.join(self.data_dir, "sensors_submission.csv")
        submission.to_csv(outfile, index=False)
        print("Submission saved to sensors_submission.csv!")


if __name__ == "__main__":
    data_path = "sensors/data"

    data = Data(data_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = SensorsModel(data_path, data_processed_path)
    model._optuna_base(n_trials=500)
    model._build_oof_matrix(top_n=10, n_splits=10)
    model._optuna_meta(n_trials=100)
    model.predict()
