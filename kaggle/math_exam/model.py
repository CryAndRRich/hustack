# Updated on May 17, 2025, 09:50 PM
# Public Score: 0.78406
# Rank: 9/151

import os
import pandas as pd

from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

from preprocess import Data

class MathExamModel():
    def __init__(self, 
                 data_path: str, 
                 data_processed_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)
        csv_path = os.path.join(self.data_dir, data_processed_path)
        data = pd.read_csv(csv_path)

        self.train_data = data[data["Id"] <= 9999].copy()
        self.test_data = data[data["Id"] > 9999].copy()

        self.y_train = self.train_data.pop("mean_exam_points")
        self.train_data.pop("Id")

        self.test_ids = self.test_data["Id"].astype(int).values
        self.test_data.drop(columns=["Id", "mean_exam_points"], inplace=True, errors="ignore")

        self._bayes_opt()

    def train(self) -> None:
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=self.best_params["max_depth"],
            learning_rate=self.best_params["learning_rate"],
            subsample=self.best_params["subsample"],
            colsample_bytree=self.best_params["colsample_bytree"],
            gamma=self.best_params["gamma"],
            random_state=42,
            tree_method="auto"
        )
        self.model.fit(self.train_data, self.y_train)

    def _bayes_opt(self) -> None:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        def xgb_cv(max_depth: float,
                   learning_rate: float,
                   subsample: float,
                   colsample_bytree: float,
                   gamma: float) -> float:
            params = {
                "n_estimators": 100,
                "max_depth": int(max_depth),
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "gamma": gamma,
                "random_state": 42,
                "tree_method": "auto"
            }
            model = XGBRegressor(**params)
            score = cross_val_score(model, self.train_data, self.y_train,
                                    cv=cv, scoring="r2").mean()
            return score

        bounds = {
            "max_depth": (3, 10),
            "learning_rate": (0.01, 0.3),
            "subsample": (0.5, 1),
            "colsample_bytree": (0.5, 1),
            "gamma": (0, 5)
        }

        optimizer = BayesianOptimization(f=xgb_cv, pbounds=bounds, random_state=42)
        optimizer.maximize(init_points=5, n_iter=15)

        best = optimizer.max["params"]
        best["max_depth"] = int(best["max_depth"])
        self.best_params = best
        print("Best XGBoost params:", self.best_params)

    def predict(self) -> None:
        preds = self.model.predict(self.test_data)
        adjusted = [x - 0.1 if x > 40 else x + 0.1 for x in preds]

        submission = pd.DataFrame({
            "Id": self.test_ids,
            "mean_exam_points": adjusted
        })

        output_file = os.path.join(self.data_dir, "math_exam_submission.csv")
        submission.to_csv(output_file, index=False)
        print("Submission saved to math_exam_submission.csv!")


if __name__ == "__main__":
    data_path = "math_exam/data"

    data = Data(data_path)
    data.data_processed()
    processed = data.save_csv()

    model = MathExamModel(data_path, processed)
    model.train()
    model.predict()