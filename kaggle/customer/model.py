# Updated on June 12, 2026, 8:02 PM
# Public Score: 0.89197
# Private Score: 0.89033
# Rank: 4/24

import os
import warnings
import numpy as np
import pandas as pd
import optuna
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

from preprocess import Data

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class CustomerModel():
    def __init__(self,
                 data_path: str,
                 data_processed_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        data = pd.read_csv(data_path)

        train_data = data[data["satisfaction"].notna()].copy()
        test_data = data[data["satisfaction"].isna()].copy()

        self.test_ids = test_data["id"]

        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(train_data["satisfaction"])

        drop_cols = ["id", "satisfaction"]
        self.X_train = train_data.drop(columns=drop_cols)
        self.X_test = test_data.drop(columns=drop_cols)

        self.cat_features = self.X_train.select_dtypes(include=["object"]).columns.tolist()

    def _optuna_search(self, n_trials: int = 150) -> None:
        X_opt_tr, X_opt_va, y_opt_tr, y_opt_va = train_test_split(
            self.X_train, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        def objective(trial) -> float:
            threshold = trial.suggest_float("threshold", 0.20, 0.80)
            params = {
                "iterations": trial.suggest_int("iterations", 1000, 3000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
                "depth": trial.suggest_int("depth", 4, 12),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 100, log=True),
                "random_strength": trial.suggest_float("random_strength", 1e-4, 10.0, log=True),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
                "eval_metric": "Accuracy",
                "task_type": "CPU",
                "random_seed": 42,
                "verbose": 0
            }
            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

            model = CatBoostClassifier(**params)
            model.fit(
                X_opt_tr, y_opt_tr,
                cat_features=self.cat_features,
                eval_set=(X_opt_va, y_opt_va),
                early_stopping_rounds=100
            )
            probas = model.predict_proba(X_opt_va)[:, 1]
            return accuracy_score(y_opt_va, (probas >= threshold).astype(int))

        pbar = tqdm(total=n_trials, desc="Optuna search")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda s, t: pbar.update(1)])

        best = study.best_params
        self.best_threshold = best.pop("threshold")
        best.update({
            "eval_metric": "Accuracy",
            "task_type": "CPU",
            "random_seed": 2026,
            "verbose": 0
        })
        self.best_params = best
        print(f"Best val accuracy: {study.best_value:.5f}")
        print(f"Best threshold: {self.best_threshold:.4f}")

    def train(self, n_splits: int = 10) -> None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.test_probas = np.zeros(len(self.X_test))
        cv_scores = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(self.X_train, self.y)):
            print(f"Fold {fold + 1}/{n_splits}...")
            X_tr, y_tr = self.X_train.iloc[tr_idx], self.y[tr_idx]
            X_va, y_va = self.X_train.iloc[va_idx], self.y[va_idx]

            model = CatBoostClassifier(**self.best_params)
            model.fit(
                X_tr, y_tr,
                cat_features=self.cat_features,
                eval_set=(X_va, y_va),
                early_stopping_rounds=100
            )

            val_probas = model.predict_proba(X_va)[:, 1]
            score = accuracy_score(y_va, (val_probas >= self.best_threshold).astype(int))
            cv_scores.append(score)
            print(f"  Acc (thresh={self.best_threshold:.3f}): {score:.5f}")

            self.test_probas += model.predict_proba(self.X_test)[:, 1] / n_splits

        print(f"CV accuracy: {np.mean(cv_scores):.5f}")

    def predict(self) -> None:
        preds_num = (self.test_probas >= self.best_threshold).astype(int)
        preds_text = self.label_encoder.inverse_transform(preds_num)

        submission = pd.DataFrame({
            "ID": self.test_ids,
            "satisfaction": preds_text
        })
        outfile = os.path.join(self.data_dir, "customer_submission.csv")
        submission.to_csv(outfile, index=False)
        print("Submission saved to customer_submission.csv!")


if __name__ == "__main__":
    data_path = "customer/data"

    data = Data(data_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = CustomerModel(data_path, data_processed_path)
    model._optuna_search(n_trials=150)
    model.train(n_splits=10)
    model.predict()
