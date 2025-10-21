# Updated on April 23, 2025, 05:22 PM
# Public Score: 0.53957
# Rank: 45/217

import os
from typing import List, Tuple

import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization

from preprocess import Data

class LoanModel():
    def __init__(self, 
                 data_path: str, 
                 data_processed_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        data = pd.read_csv(data_path)

        self.train_data = data[data["Id"] <= 7499].copy()
        self.test_data = data[data["Id"] > 7499].copy()

        self.test_ids = self.test_data["Id"]
        self.X_train = self.train_data.drop(columns=["Credit Default", "Id"])
        self.y_train = self.train_data["Credit Default"]
        self.X_test = self.test_data.drop(columns=["Credit Default", "Id"])

        self.numerical_features = (
            self.X_train.select_dtypes(include=["int64", "float64"])
            .columns.tolist()
        )
        self.categorical_features = (
            self.X_train.select_dtypes(include=["object", "category"])
            .columns.tolist()
        )

        self._build_pipeline()

    def _build_pipeline(self) -> None:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer([
            ("num", num_transformer, self.numerical_features),
            ("cat", cat_transformer, self.categorical_features)
        ])

        self._bayes_opt()

        self.pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("cat", CatBoostClassifier(
                **self.best_params_cat,
                random_seed=42,
                eval_metric="AUC",
                verbose=False
            ))
        ])

    def _bayes_opt(self) -> None:
        def cat_cv(iterations: float,
                   learning_rate: float,
                   depth: float,
                   l2_leaf_reg: float,
                   subsample: float) -> float:
            iters = int(iterations)
            d = int(depth)
            model = CatBoostClassifier(
                iterations=iters,
                learning_rate=learning_rate,
                depth=d,
                l2_leaf_reg=l2_leaf_reg,
                subsample=subsample,
                random_seed=42,
                eval_metric="F1",
                verbose=False
            )
            pipe = Pipeline([
                ("preprocessor", self.preprocessor),
                ("cat", model)
            ])
            return cross_val_score(
                pipe, self.X_train, self.y_train,
                cv=3, scoring="f1"
            ).mean()

        pbounds = {
            "iterations": (100, 500),
            "learning_rate": (0.01, 0.3),
            "depth": (4, 10),
            "l2_leaf_reg": (1, 10),
            "subsample": (0.6, 1.0)
        }

        optimizer = BayesianOptimization(
            f=cat_cv,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        optimizer.maximize(init_points=5, n_iter=15)

        params = optimizer.max["params"]
        params["iterations"] = int(params["iterations"])
        params["depth"] = int(params["depth"])
        self.best_params_cat = params
        print("Best CatBoost params:", self.best_params_cat)

    def train(self) -> None:
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=0.1,
            stratify=self.y_train,
            random_state=42
        )
        self.preprocessor.fit(X_tr, y_tr)
        X_tr_enc = self.preprocessor.transform(X_tr)
        X_val_enc = self.preprocessor.transform(X_val)

        model = CatBoostClassifier(
            **self.best_params_cat,
            random_seed=42,
            eval_metric="AUC"
        )
        model.fit(
            X_tr_enc, y_tr,
            eval_set=(X_val_enc, y_val),
            early_stopping_rounds=20,
            verbose=50
        )

        val_probs = model.predict_proba(X_val_enc)[:, 1]
        best_thresh, best_f1 = self._threshold_optim(y_val, val_probs)
        print(f"Best F1 = {best_f1:.4f} at threshold = {best_thresh}")

        self.model = model
        self.best_thresh = best_thresh

    def _threshold_optim(self,
                         y_true: pd.Series,
                         y_prob: pd.Series|List|Tuple) -> Tuple[float, float]:
        def objective(thresh: float) -> float:
            preds = (y_prob > thresh).astype(int)
            return -f1_score(y_true, preds)

        result = minimize_scalar(
            objective,
            bounds=(0.1, 0.9),
            method="bounded"
        )
        return result.x, -result.fun

    def predict(self) -> None:
        X_test_enc = self.preprocessor.transform(self.X_test)
        probs = self.model.predict_proba(X_test_enc)[:, 1]
        preds = (probs > self.best_thresh).astype(int)

        self.submission = pd.DataFrame({
            "Id": self.test_ids,
            "Credit Default": preds
        })
        outfile = os.path.join(self.data_dir, "loan_submission.csv")
        self.submission.to_csv(outfile, index=False)
        print("Submission saved to loan_submission.csv!")


if __name__ == "__main__":
    data_path = "loan/data"

    data = Data(data_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = LoanModel(data_path, data_processed_path)
    model.train()
    model.predict()