# Updated on May 31, 2025, 6:32 PM
# Public Score: 0.05777
# Rank: 1476/4183

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

from preprocess import Data

class CaloriesModel():
    def __init__(self,
                 data_path: str,
                 data_processed_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        data = pd.read_csv(data_path)

        self.train_data = data[(data["id"] < 750000) | (data["id"] > 5000000)].copy()
        self.test_data = data[(data["id"] >= 750000) & (data["id"] < 5000000)].copy()

        self.test_ids = self.test_data["id"].astype(int).values
        self.X_train = self.train_data.drop(columns=["Calories", "id"])
        self.y_train = self.train_data["Calories"]
        self.X_test = self.test_data.drop(columns=["Calories", "id"])

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
            ("xgb", XGBRegressor(
                **self.best_params_xgb,
                random_state=42,
                tree_method="auto"
            ))
        ])

    def _bayes_opt(self) -> None:
        cv = KFold(n_splits=7, shuffle=True, random_state=42)
        
        def xgb_cv(learning_rate: float,
                   max_depth: float,
                   subsample: float,
                   colsample_bytree: float,
                   gamma: float) -> float:
            md = int(max_depth)
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=learning_rate,
                max_depth=md,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                random_state=42,
                n_jobs= -1
            )
            pipe = Pipeline([
                ("preprocessor", self.preprocessor),
                ("xgb", model)
            ])
        
            neg_mse = cross_val_score(pipe, self.X_train, self.y_train, cv=cv,
                                scoring="neg_mean_squared_error").mean()
            rmse = np.sqrt(-neg_mse)
            return -rmse

        pbounds = {
            "max_depth": (3, 10),
            "learning_rate": (0.01, 0.3),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "gamma": (0.0, 5.0)
        }

        optimizer = BayesianOptimization(
            f=xgb_cv,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        optimizer.maximize(init_points=5, n_iter=15)

        params = optimizer.max["params"]
        params["max_depth"] = int(params["max_depth"])
        self.best_params_xgb = params
        print("Best XGB params:", self.best_params_xgb)

    def train(self) -> None:
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self) -> None:
        preds = self.pipeline.predict(self.X_test)

        self.submission = pd.DataFrame({
            "id": self.test_ids,
            "Calories": np.expm1(preds)
        })
        outfile = os.path.join(self.data_dir, "calories_submission.csv")
        self.submission.to_csv(outfile, index=False)
        print("Submission saved to calories_submission.csv!")

if __name__ == "__main__":
    data_path = "calories/data"
    data = Data(data_path)
    data_processed_path = data.save_csv()

    model = CaloriesModel(data_path, data_processed_path)
    model.train()
    model.predict()
