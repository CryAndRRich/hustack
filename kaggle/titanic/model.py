# Updated on March 15, 2025, 10:51 AM
# Public Score: 0.79665
# Rank: 837/15256

import os
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization

from preprocess import Data


def to_numeric_func(x: pd.DataFrame) -> pd.DataFrame:
    return x.apply(pd.to_numeric, errors="coerce")


class TitanicModel():
    def __init__(self, 
                 data_path: str, 
                 data_processed_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        data = pd.read_csv(data_path)

        self.train_data = data[data["PassengerId"] <= 891]
        self.test_data = data[data["PassengerId"] > 891]

        self.features = [
            "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "IsAdult", 
            "FamSize", "hasNanny", "TicketNumber", "TicketLetter", "CabinLet", "CabinNum"
        ]

        self.X = self.train_data[self.features]
        self.y = self.train_data["Survived"]

        self.X_test = self.test_data[self.features]
        self.test_id = self.test_data["PassengerId"]

        self.numeric_features = [
            "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "IsAdult",
            "FamSize", "hasNanny", "TicketNumber", "CabinNum"
        ]
        self.categorical_features = [col for col in self.features if col not in self.numeric_features]
        print("Numeric features:", self.numeric_features)
        print("Categorical features:", self.categorical_features)

        self._build_pineline()

    def _build_pineline(self) -> None:
        self.numeric_transformer = Pipeline(steps=[
            ("to_numeric", FunctionTransformer(to_numeric_func)),
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        self.categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ("num", self.numeric_transformer, self.numeric_features),
            ("cat", self.categorical_transformer, self.categorical_features)
        ])

        self.pipeline_xgb = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("xgb", XGBClassifier(n_estimators=20, random_state=42, eval_metric="logloss"))
        ])

        pca_columns = ["Sex", "Age", "SibSp", "Parch", "Fare"]
        self.catboost_remainder = [col for col in self.features if col not in pca_columns]

        self.preprocessor_cat_pca = ColumnTransformer(
            transformers=[
                ("pca", PCA(n_components=3), pca_columns)
            ],
            remainder="passthrough"
        )

        self._bayes_opt()

        self.pipeline_cat = Pipeline(steps=[
            ("preprocessing", self.preprocessor_cat_pca),
            ("cat", CatBoostClassifier(
                **self.best_params_cat,
                random_seed=42,
                verbose=False,
                cat_features=self.cat_feature_indices
            ))
        ])

    def _bayes_opt(self) -> None:
        X_cat_pca_transformed = self.preprocessor_cat_pca.fit_transform(self.X)
        self.cat_feature_indices = list(range(
            X_cat_pca_transformed.shape[1] - len(self.catboost_remainder),
            X_cat_pca_transformed.shape[1]
        ))

        def cat_cv(iterations: float, learning_rate: float, depth: float) -> float:
            iterations_int = int(iterations)
            depth_int = int(depth)
            model = CatBoostClassifier(
                iterations=iterations_int,
                learning_rate=learning_rate,
                depth=depth_int,
                random_seed=42,
                verbose=False,
                cat_features=self.cat_feature_indices
            )
            pipe = Pipeline(steps=[
                ("preprocessing", self.preprocessor_cat_pca),
                ("cat", model)
            ])
            scores = cross_val_score(pipe, self.X, self.y, cv=3, scoring="roc_auc")
            return scores.mean()

        cat_bo = BayesianOptimization(
            f=cat_cv,
            pbounds={
                "iterations": (100, 500),
                "learning_rate": (0.01, 0.3),
                "depth": (3, 8)
            },
            random_state=42,
            verbose=2
        )
        cat_bo.maximize(init_points=5, n_iter=15)

        self.best_params_cat = cat_bo.max["params"]
        self.best_params_cat["iterations"] = int(self.best_params_cat["iterations"])
        self.best_params_cat["depth"] = int(self.best_params_cat["depth"])
        print("Best CatBoost parameters:", self.best_params_cat)

    def train(self) -> None:
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.pipeline_xgb.fit(X_train_split, y_train_split)
        self.pipeline_cat.fit(X_train_split, y_train_split)

        val_pred_xgb = self.pipeline_xgb.predict_proba(X_val)[:, 1]
        val_pred_cat = self.pipeline_cat.predict_proba(X_val)[:, 1]

        self.best_w_xgb, self.best_w_cat = self.find_best_ensemble_weights(
            y_val, val_pred_xgb, val_pred_cat, step=0.01
        )
        print("Best ensemble weights: XGB: {:.2f}, CatBoost+PCA: {:.2f}".format(self.best_w_xgb, self.best_w_cat))

    def find_best_ensemble_weights(self, 
                                   y_true: np.ndarray,
                                   y_pred_xgb: np.ndarray,
                                   y_pred_cat: np.ndarray,
                                   step: float = 0.01) -> Tuple[float, float]:
        best_w = 0.0
        best_acc = 0.0
        for w in np.arange(0, 1 + step, step):
            ensemble_prob = w * y_pred_xgb + (1 - w) * y_pred_cat
            ensemble_pred = (ensemble_prob > 0.5).astype(int)
            acc = accuracy_score(y_true, ensemble_pred)
            if acc > best_acc:
                best_w, best_acc = w, acc
        return best_w, 1 - best_w

    def predict(self) -> None:
        test_pred_xgb = self.pipeline_xgb.predict_proba(self.X_test)[:, 1]
        test_pred_cat = self.pipeline_cat.predict_proba(self.X_test)[:, 1]

        ensemble_test_prob = self.best_w_xgb * test_pred_xgb + self.best_w_cat * test_pred_cat
        final_predictions = (ensemble_test_prob > 0.5).astype(int)

        self.submission = pd.DataFrame({
            "PassengerId": self.test_id,
            "Survived": final_predictions
        })

        output_file = os.path.join(self.data_dir, "titanic_submission.csv")
        self.submission.to_csv(output_file, index=False)
        print("Submission saved to titanic_submission.csv!")

if __name__ == "__main__":
    data_path = "titanic/data"

    data = Data(data_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = TitanicModel(data_path, data_processed_path)
    model.train()
    model.predict()