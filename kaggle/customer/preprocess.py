import os
import pandas as pd


class Data():
    def __init__(self, data_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)

        train_file_path = os.path.join(self.data_dir, "train_dataset.csv")
        test_file_path = os.path.join(self.data_dir, "test_dataset_exam.csv")
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)

        for df in [train_data, test_data]:
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)

        train_data = train_data.dropna(subset=["satisfaction"]).reset_index(drop=True)
        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    def __feature_engineer(self) -> None:
        self.data["Arrival Delay in Minutes"] = self.data["Arrival Delay in Minutes"].fillna(
            self.data["Departure Delay in Minutes"]
        )
        self.data["Total_Delay"] = (
            self.data["Departure Delay in Minutes"] + self.data["Arrival Delay in Minutes"]
        )

        rating_cols = [
            c for c in self.data.columns
            if self.data[c].dtype in ["int64", "float64"]
            and self.data[c].max() <= 5 and self.data[c].min() >= 0
            and c not in ["id"]
        ]
        self.data["Total_Score"] = self.data[rating_cols].sum(axis=1)
        self.data["Count_Zero_Ratings"] = (self.data[rating_cols] == 0).sum(axis=1)

        for col in self.data.select_dtypes(include=["object"]).columns:
            if col != "satisfaction":
                self.data[col] = self.data[col].astype(str)

    def data_processed(self) -> None:
        self.__feature_engineer()

    def save_csv(self) -> str:
        output_file = os.path.join(self.data_dir, "processed_data.csv")
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return "processed_data.csv"

if __name__ == "__main__":
    data_path = "customer/data"
    data = Data(data_path)
    data.data_processed()
    data.save_csv()
