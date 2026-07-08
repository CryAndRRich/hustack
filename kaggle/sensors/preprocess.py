import os
import numpy as np
import pandas as pd


class Data():
    def __init__(self, data_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)

        train_file_path = os.path.join(self.data_dir, "train.csv")
        test_file_path = os.path.join(self.data_dir, "test.csv")
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)

        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    def __drop_correlated(self) -> None:
        train_rows = self.data[self.data["faulty"].notna()]
        feature_cols = [c for c in train_rows.columns if c not in ["id", "faulty"]]

        corr_matrix = train_rows[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.98)]

        self.data = self.data.drop(columns=to_drop)

    def data_processed(self) -> None:
        self.__drop_correlated()

    def save_csv(self) -> str:
        output_file = os.path.join(self.data_dir, "processed_data.csv")
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return "processed_data.csv"


if __name__ == "__main__":
    data_path = "sensors/data"
    data = Data(data_path)
    data.data_processed()
    data.save_csv()
