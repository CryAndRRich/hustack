import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})
    
    df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)
    
    df.rename(columns={
        "1stFlrSF": "FirstFlrSF",
        "2ndFlrSF": "SecondFlrSF",
        "3SsnPorch": "Threeseasonporch",
        "FireplaceQu": "FireplaceQual"
    }, inplace=True)
    
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(df[name].median())
    
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    
    return df


features_nom = [
    "MSSubClass", "MSZoning", "Street", "Alley", "LandContour", 
    "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", 
    "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", 
    "MasVnrType", "Foundation", "Heating", "CentralAir", "GarageType", 
    "MiscFeature", "SaleType", "SaleCondition"
]

five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(10))

ordered_levels = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQual": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}

ordered_levels = {key: ["None"] + value for key, value in ordered_levels.items()}


def encode(df: pd.DataFrame) -> pd.DataFrame:
    for name in features_nom:
        df[name] = df[name].astype("category")
        if "None" not in df[name].cat.categories:
            df[name] = df[name].cat.add_categories("None")
    
    for name, levels in ordered_levels.items():
        try:
            df[name] = df[name].astype(pd.CategoricalDtype(levels, ordered=True))
        except Exception as e:
            continue
    return df


class Data():
    def __init__(self, data_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)

        train_file_path = os.path.join(self.data_dir, "train.csv")
        test_file_path = os.path.join(self.data_dir, "test.csv")

        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        
        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
        
        self.data = clean(self.data)
        self.data = encode(self.data)
        self.data = impute(self.data)

    def __drop_uninformative(self) -> None:
        mi_scores = self.mi_scores_dataset()
        cols_to_keep = mi_scores[mi_scores > 10e-6].index.tolist()
        cols_to_keep += ["Id", "SalePrice"]
        self.data = self.data.loc[:, self.data.columns.isin(cols_to_keep)]

    def mi_scores_dataset(self) -> pd.Series:
        X = self.data[self.data["Id"] <= 1460].copy()
        y = X.pop("SalePrice")
        X.pop("Id")
        
        for colname in X.select_dtypes(["object", "category"]):
            X[colname], _ = X[colname].factorize()
        
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    def baseline_score_dataset(self, model: XGBRegressor = XGBRegressor()) -> None:
        X = self.data[self.data["Id"] <= 1460].copy()
        y = X.pop("SalePrice")
        X.pop("Id")
        
        for colname in X.select_dtypes(["category"]):
            X[colname] = X[colname].cat.codes
        
        log_y = np.log(y)
        
        rmsle = np.sqrt(-cross_val_score(
            model, X, log_y, cv=5, scoring="neg_mean_squared_error"
        ).mean())
        
        mae = -cross_val_score(
            model, X, log_y, cv=5, scoring="neg_mean_absolute_error"
        ).mean()
        
        print(f"Baseline RMSLE: {rmsle:.5f}")
        print(f"Baseline MAE (log scale): {mae:.5f}")

    def __age_processed(self) -> None:
        self.data["HouseAge"] = np.maximum(self.data["YrSold"] - self.data["YearBuilt"], 0)
        self.data["HouseRemodAge"] = np.maximum(self.data["YrSold"] - self.data["YearRemodAdd"], 0)

        self.data["HouseCond"] = (self.data["HouseAge"] / 20).round().astype(int).clip(upper=4)
        self.data["HouseCond"] = self.data["HouseCond"].astype(pd.CategoricalDtype(list(range(5)), ordered=True))

        self.data["GarageAge"] = self.data["YrSold"] - self.data["GarageYrBlt"]

    def __area_processed(self) -> None:
        self.data["TotalArea"] = self.data["GrLivArea"] + self.data["TotalBsmtSF"]
        self.data["TotalSF"] = self.data["FirstFlrSF"] + self.data["SecondFlrSF"] + self.data["BsmtFinSF1"] + self.data["BsmtFinSF2"]
        
        self.data["LivAreaRatio"] = self.data["GrLivArea"] / self.data["TotalSF"]
        self.data["LivLotRatio"] = self.data["GrLivArea"] / self.data["LotArea"]
        self.data["RmsPerSqft"] = self.data["TotRmsAbvGrd"] / self.data["GrLivArea"]
        self.data["BedroomRatio"] = self.data["BedroomAbvGr"] / self.data["GrLivArea"]
        self.data["FlrRatio"] = self.data["FirstFlrSF"] / (self.data["FirstFlrSF"] + self.data["SecondFlrSF"])
        self.data["Spaciousness"] = (self.data["FirstFlrSF"] + self.data["SecondFlrSF"]) / self.data["TotRmsAbvGrd"]

        bldg_area = pd.get_dummies(self.data["BldgType"], prefix="Bldg").mul(self.data["GrLivArea"], axis=0)
        self.data = self.data.join(bldg_area)

        self.data["MedNhbdArea"] = self.data.groupby("Neighborhood", observed=False)["GrLivArea"].transform("median")

        self.data["LgGrLivArea"] = np.log1p(self.data["GrLivArea"])
        self.data["LgMasVnrArea"] = np.log1p(self.data["MasVnrArea"])

    def __quality_processed(self) -> None:
        self.data["OverallQual"] = self.data["OverallQual"].cat.remove_categories("None")
        self.data["OverallQual"] = pd.to_numeric(self.data["OverallQual"], errors="coerce")
        self.data["OverallCond"] = pd.to_numeric(self.data["OverallCond"], errors="coerce")

        self._fill_overallqual()
        
        self.data["OverallAvg"] = ((self.data["OverallQual"] + self.data["OverallCond"]) / 2).round().astype("int")
        self.data["QualSF"] = self.data["TotalSF"] * self.data["OverallQual"]
        self.data["NhbQual"] = self.data.groupby("Neighborhood", observed=False)["OverallQual"].transform("median")

        self.data["OverallAvg"] = self.data["OverallAvg"].astype("category")
        self.data["OverallQual"] = self.data["OverallQual"].astype("category")
        self.data["OverallCond"] = self.data["OverallCond"].astype("category")

        quality_features = ["ExterQual", "KitchenQual", "BsmtQual", "FireplaceQual", "GarageQual"]
        qual_mapping = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}

        for feat in quality_features:
            self.data[feat] = self.data[feat].astype(str).str.strip().replace("NA", np.nan)
            self.data[feat + "Num"] = self.data[feat].map(qual_mapping)
            median_val = self.data[feat + "Num"].median()
            self.data[feat + "Num"] = self.data[feat + "Num"].fillna(median_val)
        
        num_cols = [feat + "Num" for feat in quality_features]
        self.data["CompositeQual"] = self.data[num_cols].mean(axis=1)

        self.data["ExterQual"] = self.data["ExterQual"].astype("category")
        self.data["KitchenQual"] = self.data["KitchenQual"].astype("category")
        self.data["BsmtQual"] = self.data["BsmtQual"].astype("category")
        self.data["FireplaceQual"] = self.data["FireplaceQual"].astype("category")
        self.data["GarageQual"] = self.data["GarageQual"].astype("category")
    
    def _fill_overallqual(self) -> None:
        features = [
            "ExterQual", "BsmtQual", "KitchenQual", "Neighborhood", 
            "GrLivArea", "GarageArea", "YearBuilt", "YearRemodAdd"
        ]

        known = self.data[self.data["OverallQual"].notna()].copy()
        unknown = self.data[self.data["OverallQual"].isna()].copy()
        
        X_train = pd.get_dummies(known[features])
        y_train = known["OverallQual"]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        X_unknown = pd.get_dummies(unknown[features])
        X_unknown = X_unknown.reindex(columns=X_train.columns, fill_value=0)
        
        predicted = model.predict(X_unknown)
        
        self.data.loc[self.data["OverallQual"].isna(), "OverallQual"] = np.round(predicted).astype(int)
    
    def __basement_processed(self) -> None:
        qual_mapping = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        self.data["BsmtQual"] = self.data["BsmtQual"].astype(str)
        self.data["BsmtCond"] = self.data["BsmtCond"].astype(str)
        self.data["BsmtQualNum"] = self.data["BsmtQual"].map(qual_mapping)
        self.data["BsmtCondNum"] = self.data["BsmtCond"].map(qual_mapping)
        
        self.data["BsmtQual"] = self.data["BsmtQual"].astype("category")
        self.data["BsmtCond"] = self.data["BsmtCond"].astype("category")
        
        self.data["BsmtQualNum"] = self.data["BsmtQualNum"].fillna(0)
        self.data["BsmtCondNum"] = self.data["BsmtCondNum"].fillna(0)
        self.data["BsmtQualityScore"] = np.round((self.data["BsmtQualNum"] + self.data["BsmtCondNum"]) / 2.0).astype(int)
    
    def __porch_processed(self) -> None:
        self.data["PorchTypes"] = self.data[[ 
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
            "Threeseasonporch",
            "ScreenPorch",
        ]].gt(0.0).sum(axis=1)

    def __bath_processed(self) -> None:
        self.data["TotalBath"] = (
            self.data["FullBath"] +
            0.5 * self.data["HalfBath"] +
            self.data["BsmtFullBath"] +
            0.5 * self.data["BsmtHalfBath"]
        )

    def __k_means(self) -> None:
        cluster_features = ["LotArea", "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF", "GrLivArea"]
        
        X_scaled = self.data.loc[:, cluster_features]
        X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
        
        kmeans = KMeans(n_clusters=20, n_init=50, random_state=0)
        
        self.data["Cluster"] = kmeans.fit_predict(X_scaled)
        
        distances = kmeans.transform(X_scaled)
        
        distances_df = pd.DataFrame(
            distances, 
            columns=[f"Centroid{i}" for i in range(distances.shape[1])],
            index=self.data.index
        )
        
        self.data = self.data.join(distances_df)

    def data_processed(self) -> None:
        self.__age_processed()
        self.__area_processed()
        self.__quality_processed()
        self.__basement_processed()
        self.__porch_processed()
        self.__bath_processed()
        self.__k_means()
        self.__drop_uninformative()

    def save_csv(self) -> str:
        self.output = "processed_data.csv"
        output_file = os.path.join(self.data_dir, self.output)
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return self.output


if __name__ == "__main__":
    data_path = "housing_price/data"
    data = Data(data_path)
    
    data.data_processed()
    data.baseline_score_dataset()
    data.save_csv()