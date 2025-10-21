import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

class Data():
    def __init__(self, data_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)

        train_file_path = os.path.join(self.data_dir, "train.csv")
        test_file_path = os.path.join(self.data_dir, "test.csv")
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)

        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

        categorical_cols = ["Years in current job", "Purpose", "Term"]
        numerical_cols = ["Months since last delinquent", "Bankruptcies", "Credit Default"]

        for col in categorical_cols:
            self.data[col] = self.data[col].fillna("None").astype("category")
        for col in numerical_cols:
            self.data[col] = self.data[col].fillna(0)

    def __drop_uninformative(self) -> None:
        mi_scores = self.mi_scores_dataset()
        cols_to_keep = mi_scores[mi_scores > 0.0].index.tolist()
        cols_to_keep += ["Id", "Credit Default"]
        self.data = self.data.loc[:, self.data.columns.isin(cols_to_keep)]

    def mi_scores_dataset(self) -> pd.Series:
        df = self.data[self.data["Id"] <= 7499].copy()
        y = df.pop("Credit Default")  
        df.pop("Id") 

        for colname in df.select_dtypes(["object", "category"]):
            df[colname], _ = df[colname].factorize()

        discrete_features = [pd.api.types.is_integer_dtype(t) for t in df.dtypes]
        mi = mutual_info_classif(df, y, discrete_features=discrete_features, random_state=0)

        mi_scores = pd.Series(mi, index=df.columns, name="MI Scores").sort_values(ascending=False)
        return mi_scores

    def baseline_score_dataset(self, model: XGBClassifier = XGBClassifier()) -> None:
        df = self.data[self.data["Id"] <= 7499].copy()
        y = df.pop("Credit Default")
        df.pop("Id")

        for col in df.select_dtypes(["category"]):
            df[col] = df[col].cat.codes

        metrics = ["accuracy", "precision", "recall", "f1"]
        for metric in metrics:
            score = cross_val_score(model, df, y, cv=5, scoring=metric).mean()
            print(f"Baseline {metric.capitalize()}: {score:.5f}")

    def __home_processed(self) -> None:
        def filter_home(val):
            if "Rent" in val:
                return "Rent"
            if "Mortgage" in val:
                return "Mortgage"
            return "Home"

        self.data["Home Ownership"] = (
            self.data["Home Ownership"].apply(filter_home).astype("category")
        )
        self.data["hasHome"] = (self.data["Home Ownership"] == "Home").astype(int)

    def __job_processed(self) -> None:
        self.data["Annual Income"] = (
            self.data.groupby("Home Ownership", observed=False)["Annual Income"]
                .transform(lambda x: x.fillna(x.median()))
        )
        self.data["Monthly Income"] = self.data["Annual Income"] / 12.0

        self.data["Income Group"] = pd.qcut(
            self.data["Annual Income"], q=4,
            labels=["Low", "Medium", "High", "Very High"]
        )

        self.data["Log Annual Income"] = np.log1p(self.data["Annual Income"])
        self.data["Log Monthly Income"] = np.log1p(self.data["Monthly Income"])
        self.data["Log Monthly Debt"] = np.log1p(self.data["Monthly Debt"])

        self.data["Income after Debt"] = self.data["Monthly Income"] - self.data["Monthly Debt"]
        self.data["Dept Income Ratio"] = self.data["Monthly Debt"] / self.data["Monthly Income"]

        def filter_year(val):
            tok = str(val).split()[0]
            if tok == "<":
                return 0.5
            if tok == "10+":
                return 10.5
            if tok == "None":
                return np.nan
            return int(tok)

        self.data["Years in current job"] = (
            self.data["Years in current job"].apply(filter_year)
        )
        self.data["Years in current job"] = (
            self.data.groupby("Home Ownership", observed=False)["Years in current job"]
                .transform(lambda x: x.fillna(x.median()))
        )

    def __credit_processed(self) -> None:
        self.data["Has Liens"] = (self.data["Tax Liens"] > 0).astype(int)
        self.data["Has Problems"] = (self.data["Number of Credit Problems"] > 0).astype(int)
        self.data["Bankrupted"] = (self.data["Bankruptcies"] > 0).astype(int)

        self.data["Log Maximum Open Credit"] = np.log1p(self.data["Maximum Open Credit"])
        self.data["Log Current Loan Amount"] = np.log1p(self.data["Current Loan Amount"])
        self.data["Log Current Credit Balance"] = np.log1p(self.data["Current Credit Balance"])

        self.data["Year since last delinquent"] = self.data["Months since last delinquent"] / 12.0
        self.data["No delinquent"] = (self.data["Months since last delinquent"] == 0).astype(int)

        self.data["Months of Credit History"] = (self.data["Years of Credit History"] * 12)
        self.data["Credit History Group"] = pd.cut(
            self.data["Years of Credit History"],
            bins=[0, 5, 10, 20, 50],
            labels=["Very Short", "Short", "Medium", "Long"]
        )

        self.data["Credit Score"] = (
            self.data.groupby("Home Ownership", observed=False)["Credit Score"]
                .transform(lambda x: x.fillna(x.median()))
        )
        self.data["Log Credit Score"] = np.log1p(self.data["Credit Score"])

        self.data["Debt Score Interaction"] = (self.data["Dept Income Ratio"] * self.data["Credit Score"])

        self.data["Credit Score Group"] = pd.qcut(
            self.data["Credit Score"], q=4,
            labels=["Bad", "Average", "Good", "Excellent"]
        )

    def __k_means(self) -> None:
        cluster_features = [
            "Log Monthly Income", "Log Credit Score",
            "Debt Score Interaction", "Log Current Loan Amount",
            "Log Maximum Open Credit"
        ]
        X_scaled = self.data[cluster_features]
        X_scaled = (X_scaled - X_scaled.mean()) / X_scaled.std()

        kmeans = KMeans(n_clusters=10, n_init=50, random_state=0)
        self.data["Cluster"] = kmeans.fit_predict(X_scaled)

        distances = kmeans.transform(X_scaled)
        dist_cols = [f"Centroid{i}" for i in range(distances.shape[1])]
        dist_df = pd.DataFrame(distances, columns=dist_cols, index=self.data.index)
        self.data = self.data.join(dist_df)

    def data_processed(self) -> None:
        self.__home_processed()
        self.__job_processed()
        self.__credit_processed()
        self.__k_means()
        self.__drop_uninformative()

    def save_csv(self) -> str:
        output_file = os.path.join(self.data_dir, "processed_data.csv")
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return "processed_data.csv"


if __name__ == "__main__":
    data_path = "loan/data"
    data = Data(data_path)

    data.data_processed()
    data.mi_scores_dataset()
    data.baseline_score_dataset()
    data.save_csv()