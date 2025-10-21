# Updated on March 23, 2025, 9:02 AM
# Public Score: 0.80757
# Rank: 88/1834

import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier

from preprocess import Data

class SpaceshipModel():
    def __init__(self, 
                 data_path: str, 
                 data_processed_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)
        data_path = os.path.join(self.data_dir, data_processed_path)
        
        data = pd.read_csv(data_path)

        self.train_data = data[data["Transported"] != -1].copy()
        self.test_data = data[data["Transported"] == -1].copy()
        
        self.test_passenger_ids = self.test_data["PassengerId"]

        self.X_train = self.train_data.drop(columns=["Transported", "PassengerId"])
        self.y_train = self.train_data["Transported"]

        self.X_test = self.test_data.drop(columns=["Transported", "PassengerId"])

        self.categorical_cols = self.X_train.select_dtypes(include=["object"]).columns.tolist()
        self.categorical_cols += ["HomePlanet", "Destination"]

        self._build_pipeline()

    def _build_pipeline(self) -> None:
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols)
            ],
            remainder="passthrough"
        )

        self.pipe = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", CatBoostClassifier(
                iterations=100,
                random_seed=42,
                eval_metric="Logloss",
                verbose=0
            ))
        ])

        self.param_grid = {
            "classifier__depth": [3, 5, 7, 9],
            "classifier__learning_rate": [0.01, 0.1, 0.2, 0.3],
            "classifier__l2_leaf_reg": [1, 5, 10],
            "classifier__bagging_temperature": [0, 0.5, 1]
        }

        self.grid_search = GridSearchCV(
            estimator=self.pipe,
            param_grid=self.param_grid,
            cv=3,           
            scoring="accuracy",
            n_jobs=-1,      
            verbose=2
        )

    def train(self) -> None:
        X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )
        self.grid_search.fit(X_train_train, y_train_train)

        print("Best parameters found:", self.grid_search.best_params_)
        print("Best CV accuracy:", self.grid_search.best_score_)

        self.best_pipe = self.grid_search.best_estimator_
        preds_valid = self.best_pipe.predict(X_train_test)
        val_acc = accuracy_score(y_train_test, preds_valid)
        print("Validation accuracy:", val_acc)

    def predict(self) -> None:
        final_pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", CatBoostClassifier(
                iterations=100,
                random_seed=42,
                eval_metric="Logloss",
                verbose=0,
                depth=self.grid_search.best_params_["classifier__depth"],
                learning_rate=self.grid_search.best_params_["classifier__learning_rate"],
                l2_leaf_reg=self.grid_search.best_params_["classifier__l2_leaf_reg"],
                bagging_temperature=self.grid_search.best_params_["classifier__bagging_temperature"]
            ))
        ])

        final_pipeline.fit(self.X_train, self.y_train)

        predictions = final_pipeline.predict(self.X_test)
        predictions_bool = predictions.astype(bool)

        submission = pd.DataFrame({
            "PassengerId": self.test_passenger_ids,
            "Transported": predictions_bool
        })
        
        output_file = os.path.join(self.data_dir, "spaceship_submission.csv")
        submission.to_csv(output_file, index=False)
        print("Submission saved to spaceship_submission.csv!")


if __name__ == "__main__":
    data_path = "spaceship/data"

    data = Data(data_path)
    data.data_processed()
    data_processed_path = data.save_csv()

    model = SpaceshipModel(data_path, data_processed_path)
    model.train()
    model.predict()