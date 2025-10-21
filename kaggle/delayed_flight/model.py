# Updated on May 15, 2025, 9:58 PM
# Public Score: 0.73750
# Rank: 1186/2205

import os

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

from preprocess import Data

class DelayedFLightModel():
    def __init__(self,
                 data_path: str,
                 data_processed_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        data = pd.read_csv(data_path)

        self.train_data = data[data["id"] > 99999].copy()
        self.test_data = data[data["id"] <= 99999].copy()

        self.test_ids = self.test_data["id"]
        self.X_train = self.train_data.drop(columns=["dep_delayed_15min", "id"])
        self.y_train = self.train_data["dep_delayed_15min"]
        self.X_test = self.test_data.drop(columns=["dep_delayed_15min", "id"])

        self.numerical_features = (
            self.X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        )
        self.categorical_features = (
            self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
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
            ("xgb", XGBClassifier(
                **self.best_params_xgb,
                eval_metric="auc",
                random_state=42,
                n_jobs= -1
            ))
        ])

    def _bayes_opt(self) -> None:
        def xgb_cv(n_estimators: float,
                   learning_rate: float,
                   max_depth: float,
                   subsample: float,
                   colsample_bytree: float,
                   reg_alpha: float,
                   reg_lambda: float) -> float:
            n_est = int(n_estimators)
            md = int(max_depth)
            model = XGBClassifier(
                n_estimators=n_est,
                learning_rate=learning_rate,
                max_depth=md,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                eval_metric="auc",
                random_state=42,
                n_jobs= -1
            )
            pipe = Pipeline([
                ("preprocessor", self.preprocessor),
                ("xgb", model)
            ])
            return cross_val_score(
                pipe, self.X_train, self.y_train,
                cv=3, scoring="roc_auc"
            ).mean()

        pbounds = {
            "n_estimators": (100, 500),
            "learning_rate": (0.01, 0.3),
            "max_depth": (3, 10),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "reg_alpha": (0, 5),
            "reg_lambda": (0, 5)
        }

        optimizer = BayesianOptimization(
            f=xgb_cv,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        optimizer.maximize(init_points=5, n_iter=15)

        params = optimizer.max["params"]
        params["n_estimators"] = int(params["n_estimators"])
        params["max_depth"] = int(params["max_depth"])
        self.best_params_xgb = params
        print("Best XGB params:", self.best_params_xgb)

    def train(self) -> None:
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self) -> None:
        probs = self.pipeline.predict_proba(self.X_test)[:, 1]

        self.submission = pd.DataFrame({
            "id": self.test_ids,
            "dep_delayed_15min": probs
        })
        outfile = os.path.join(self.data_dir, "delayed_flight_submission.csv")
        self.submission.to_csv(outfile, index=False)
        print("Submission saved to delayed_flight_submission.csv!")

if __name__ == "__main__":
    data_path = "delayed_flight/data"
    data = Data(data_path)
    data_processed_path = data.save_csv()

    model = DelayedFLightModel(data_path, data_processed_path)
    model.train()
    model.predict()