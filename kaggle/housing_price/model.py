# Updated on March 18, 2025, 12:47 PM
# Public Score: 13291.40430
# Rank: 103/7023

import os
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess import Data

class HousingPriceModel():
    def __init__(self, 
                 data_path: str, 
                 data_processed_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        
        data = pd.read_csv(data_path)

        self.train_data = data[data["Id"] <= 1460].copy()
        self.test_data = data[data["Id"] > 1460].copy()
        
        self.y_train = self.train_data.pop("SalePrice")
        
        self.test_data.pop("SalePrice")

        for col in self.train_data.select_dtypes(include=["object", "category"]).columns:
            self.train_data[col] = self.train_data[col].fillna("None").astype("category")

    def _build_pipeline(self) -> None:
        self.pipeline_gbr = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("model", GradientBoostingRegressor(
                random_state=42,
                learning_rate=0.04792803357072395,
                max_depth=3,
                min_samples_leaf=1,
                min_samples_split=13,
                n_estimators=732
            ))
        ])

        self.pipeline_cat = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("model", CatBoostRegressor(
                verbose=0,
                random_state=42,
                depth=5,
                iterations=816,
                l2_leaf_reg=1.3901484116545029,
                learning_rate=0.049212652272621164
            ))
        ])

    def train(self) -> None:
        X_train, X_val, y_train_train, y_train_val = train_test_split(
            self.train_data, self.y_train, test_size=0.2, random_state=42
        )

        self.categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numeric_features = [col for col in X_train.columns if col not in self.categorical_features]

        self.preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), self.numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features)
        ])

        self._build_pipeline()

        self.pipeline_gbr.fit(X_train, np.log(y_train_train))
        self.pipeline_cat.fit(X_train, np.log(y_train_train))

        val_preds_gbr = np.expm1(self.pipeline_gbr.predict(X_val))
        val_preds_cat = np.expm1(self.pipeline_cat.predict(X_val))

        self.best_w_gbr, self.best_w_cat = self.find_best_ensemble_weights(
            y_true=np.array(y_train_val),
            y_pred_gbr=val_preds_gbr,
            y_pred_cat=val_preds_cat,
            step=0.01
        )
        print("Best ensemble weights: GBR: {:.2f}, CatBoost: {:.2f}".format(self.best_w_gbr, self.best_w_cat))

    def find_best_ensemble_weights(self, 
                                   y_true: np.ndarray, 
                                   y_pred_gbr: np.ndarray, 
                                   y_pred_cat: np.ndarray, 
                                   step: float = 0.01) -> Tuple[float, float]:
        best_w = 0.0
        best_rmsle = float("inf")
        for w in np.arange(0, 1 + step, step):
            ensemble_pred = w * y_pred_gbr + (1 - w) * y_pred_cat
            current_rmsle: float = np.sqrt(np.mean((np.log(ensemble_pred) - np.log(y_true)) ** 2))
            if current_rmsle < best_rmsle:
                best_rmsle = current_rmsle
                best_w = w
        return best_w, 1 - best_w

    def predict(self) -> None:
        self.pipeline_gbr.fit(self.train_data, np.log(self.y_train))
        self.pipeline_cat.fit(self.train_data, np.log(self.y_train))

        test_preds_gbr = np.expm1(self.pipeline_gbr.predict(self.test_data))
        test_preds_cat = np.expm1(self.pipeline_cat.predict(self.test_data))
        
        ensemble_test_preds = self.best_w_gbr * test_preds_gbr + self.best_w_cat * test_preds_cat

        surprise_c = 2700
        submission = pd.DataFrame({
            "Id": self.test_data.index + 1,
            "SalePrice": ensemble_test_preds + surprise_c
        })
        
        output_file = os.path.join(self.data_dir, "housing_submission.csv")
        submission.to_csv(output_file, index=False)
        print("Submission saved to housing_submission.csv!")


if __name__ == "__main__":
    data_path = "housing_price/data"

    data = Data(data_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = HousingPriceModel(data_path, data_processed_path)
    model.train()
    model.predict()
