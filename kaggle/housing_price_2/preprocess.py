import numpy as np 
import pandas as pd
import os

from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor
pd.set_option("future.no_silent_downcasting", True)


class Data():
    def __init__(self, data_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)

        train_file_path = os.path.join(self.data_dir, "train.csv")
        test_file_path = os.path.join(self.data_dir, "test.csv")
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)

        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

        for col in self.data.select_dtypes(include=["int64", "float64"]).columns:
            self.data[col] = self.data[col].fillna(0)
        for col in self.data.select_dtypes(include=["object", "category"]).columns:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
            
        self.data["sale_price"] = np.log1p(self.data["sale_price"])

    def __drop_uninformative(self) -> None:
        self.data = self.data.drop(
            columns = ["sale_date", "view_rainier", "view_olympics",
                       "view_cascades", "view_territorial", "view_skyline",
                       "view_sound", "view_lakewash", "view_lakesamm",
                       "view_otherwater", "view_other", "wfnt", "golf",
                       "greenbelt", "noise_traffic"]
        )
        
        mi_scores = self.mi_scores_dataset()
        cols_to_keep = mi_scores[mi_scores > 0.0].index.tolist()
        cols_to_keep += ["id", "sale_price"]
        self.data = self.data.loc[:, self.data.columns.isin(cols_to_keep)]
        print(cols_to_keep)

    def mi_scores_dataset(self) -> pd.Series:
        df = self.data[self.data["id"] <= 199999].copy()
        y = df.pop("sale_price") 
        df.pop("id") 

        categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
        categorical_features += ["grade", "fbsmt_grade", "condition"]

        for colname in categorical_features:
            df[colname], _ = df[colname].factorize()

        discrete_features = [pd.api.types.is_integer_dtype(t) for t in df.dtypes]
        mi = mutual_info_regression(df, y, discrete_features=discrete_features, random_state=0)

        mi_scores = pd.Series(mi, index=df.columns, name="MI Scores").sort_values(ascending=False)
        return mi_scores

    def baseline_score_dataset(self, model: XGBRegressor = XGBRegressor()) -> None:
        df = self.data[self.data["id"] <= 199999].copy()
        y = df.pop("sale_price")
        df.pop("id")

        categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
        categorical_features += ["grade", "fbsmt_grade", "condition"]

        for colname in categorical_features:
            df[colname] = df[colname].astype("category")
            df[colname] = df[colname].cat.codes

        scoring = {
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_mean_squared_error"
        }

        for name, scorer in scoring.items():
            score = cross_val_score(model, df, y, cv=5, scoring=scorer).mean()
            if name == "RMSE":
                score = np.sqrt(-score)
            else: 
                score = -score
            print(f"Baseline {name}: {score:.5f}")

    def __time_processed(self) -> None:
        self.data["sale_date"] = pd.to_datetime(self.data["sale_date"])
        self.data["sale_year"] = self.data["sale_date"].dt.year
        self.data["join_sale_year"] = self.data["join_year"] - self.data["sale_year"]
        self.data["age"] = self.data["sale_year"] - self.data["year_built"]
        self.data["renovated"] = np.where(self.data["year_reno"] > 0, 1, 0)
        self.data["years_since_reno"] = np.where(self.data["renovated"], self.data["sale_year"] - self.data["year_reno"], 0)

    def __house_processed(self) -> None:
        self.data["bath"] = (
            self.data["bath_full"] +
            0.75 * self.data["bath_3qtr"] + 
            0.5 * self.data["bath_half"]
        )
        
        self.data["living_sqft"] = np.sqrt(
            self.data["sqft"] ** 2 + 
            self.data["sqft_fbsmt"] ** 2
        )
        self.data["total_living_sqft"] = self.data["sqft"] + self.data["sqft_fbsmt"]
        self.data["total_non_living_sqft"] = self.data["sqft_lot"] + self.data["gara_sqft"] + self.data["garb_sqft"]
        self.data["total_rooms"] = self.data["beds"] + self.data["bath"] 

        self.data["view"] = (
            self.data["view_rainier"] + self.data["view_olympics"] + 
            self.data["view_cascades"] + self.data["view_territorial"] + 
            self.data["view_skyline"] + self.data["view_sound"] + 
            self.data["view_lakewash"] + self.data["view_lakesamm"] + 
            self.data["view_otherwater"] + self.data["view_other"] +
            self.data["wfnt"] + self.data["golf"] +
            self.data["greenbelt"] + self.data["noise_traffic"]
        )
        
        self.data["stories_living_interact"] = self.data["stories"] * self.data["living_sqft"]
        self.data["stories_tot_living_interact"] = self.data["stories"] * self.data["total_living_sqft"]
        
        self.data["total_val"] = self.data["land_val"] + self.data["imp_val"]
        self.data["mul_val_log"] = np.log1p(self.data["land_val"] * self.data["imp_val"])
        self.data["view_mul_val"] = np.log1p(self.data["total_val"] * self.data["view"])
        
        self.data["grade_mul_log_living"] = self.data["grade"] * np.log1p(self.data["living_sqft"])
        self.data["grade_all"] = (self.data["grade"] + self.data["fbsmt_grade"]) / 2
        self.data["grade_cond_str"] = self.data["grade"].astype(str) + "_" + self.data["condition"].astype(str)

        def get_sale_warns(x):
            lst = x.strip().split(" ")
            if lst[0] == "":
                return 0
            return len(lst)
        self.data["sale_warning_num"] = self.data["sale_warning"].astype(str).apply(get_sale_warns)
        self.data["condition_per_warning"] = self.data["condition"] / (self.data["sale_warning_num"] + 1)
        self.data["val_per_warning"] = self.data["total_val"] / (self.data["sale_warning_num"] + 1)

    def __subdivision_processed(self) -> None:
        subdivisions = [
            "GREEN LAKE", "GILMAN", "HOMECROFT", "SEA VIEW", "CHEROKEE", "MC MICKEN", "DENNY",
            "BALLARD", "SALMON", "YESLERS", "KIRKLAND", "SEATTLE", "BOSTON COMPANYS", "LAKE UNION",
            "BOULEVARD", "SOUTH PARK", "CARLETON PARK", "MOUNT BAKER PARK", "BURKE", "JOVITA",
            "HILLMANS", "RAINIER BEACH", "BRYN MAWR", "BIGELOWS", "KILBOURNES", "INGLEWOOD",
            "OVERLAND", "MAYNARDS", "COTTAGE", "NORTH PARK", "MAPLE", "HIGH POINT", "NELSON",
            "PLEASANT VALLEY", "RENTON", "MOORLAND", "TIMBERLANE", "NORTHEND", "LAKE FOREST",
            "CEDARHURST", "REDMOND", "SQUIRES", "SNOQUALMIE", "YORK", "UNIVERSITY", "LICTON",
            "RENAISSANCE", "QUEEN", "WOODLAND PARK", "OLYMPIC", "SOUTHERN PACIFIC", "SKYWAY",
            "JOVITA", "WHITE", "LAKEWOOD", "BRIDGES", "WESTHOLME", "LAKE SIDE", "HAWTHORNE", 
            "SOMERVILLE", "TIMBERLINE", "VICTORY", "TWIN LAKES", "MADISON", "PONTIAC", "LAKE RIDGE", 
            "WILDERNESS", "HALLER LAKE", "MERIDIAN", "WALLA", "BURNS", "ROXBURY", "AUBURN", 
            "HOMEWOOD", "LAGO", "KINGSGATE", "NORMANDY", "BOWMAN", "BLUE", "MOORLANDS", "EDES", 
            "FAIRWOOD", "LAURELHURST", "EXPOSITION", "GREENE", "KLAHANIE", "DES MOINES",
            "SUNRISE", "PATRICK", "GUNTHERS", "LOYAL", "KATESRIDGE", "LAWS", "STIXRUDS", 
            "BALTIMORE", "FAUNTLEROY", "LAKE HILLS", "ALDERWOOD", "BEVERLY", "EUCLID", "SNOQUALMIE", 
            "TROSSACHS", "JACKSONS", "CUMBERLAND", "SYLVAN", "GREENWOOD", "HOME GARDENS", "CLAREMONT",
            "GEORGETOWN", "MORNINGSIDE", "PEARL", "SOUTH SHORE", "HILL TRACT", "BEAVERDAM", "EVANS",
            "BROADMOOR", "FRONT STREET", "CROWN HILL", "HAMBLETS", "EASTWOOD", "REGENCY",
            "RIDGECREST", "VALLEY", "SANDER", "TAMARACK", "MOUNTAIN PARK", "NEWCASTLE"
        ]

        def normalize_subdivision(x):
            for name in subdivisions:
                if name in x:
                    return name
            lst = x.split(" ")
            if len(lst) == 1 or lst[1] in ("ADD", "DIV", "NO"):
                return lst[0]
            return lst[0] + " " + lst[1]

        self.data["subdivision"] = self.data["subdivision"].astype(str).apply(normalize_subdivision).astype("category")

        def simplify(colname, threshold=100):
            counts = self.data[colname].value_counts()
            rare_values = counts[counts < threshold].index
            self.data[colname] = self.data[colname].apply(lambda x: "Other" if x in rare_values else x)

        simplify("subdivision")
        self.data["zoning_subdiv"] = self.data["zoning"].astype(str) + "_" + self.data["subdivision"].astype(str)
        self.data["submar_subdiv"] = self.data["submarket"].astype(str) + "_" + self.data["subdivision"].astype(str)

    def __location_processed(self) -> None:
        self.data["lat_grid"] = (self.data["latitude"] // 0.01).astype(int)
        self.data["lon_grid"] = (self.data["longitude"] // 0.01).astype(int)
        self.data["subdiv_lat"] = self.data["subdivision"].astype(str) + "_" + self.data["lat_grid"].astype(str)
        self.data["subdiv_lon"] = self.data["subdivision"].astype(str) + "_" + self.data["lon_grid"].astype(str)
        self.data["latlon_grid"] = self.data["lat_grid"].astype(str) + "_" + self.data["lon_grid"].astype(str)

    def __k_means(self) -> None:
        cluster_features = ["latitude", "longitude", "join_sale_year", "total_val"]
        X_scaled = self.data.loc[:, cluster_features]
        X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
        kmeans = KMeans(n_clusters=25, n_init=100, random_state=42)
        self.data["cluster"] = kmeans.fit_predict(X_scaled)

    def data_processed(self) -> None:
        self.__time_processed()
        self.__house_processed()
        self.__subdivision_processed()
        self.__location_processed()
        self.__k_means() 
        self.__drop_uninformative()
    
    def save_csv(self) -> str:
        output_file = os.path.join(self.data_dir, "processed_data.csv")
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return "processed_data.csv"


if __name__ == "__main__":
    data_path = "housing_price2/data"
    data = Data(data_path)

    data.data_processed()
    data.mi_scores_dataset()
    data.baseline_score_dataset()
    data.save_csv()