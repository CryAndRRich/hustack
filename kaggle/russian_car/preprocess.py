import os
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from data.supplemental_english import * # type: ignore

class Data():
    def __init__(self, data_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)

        train_file_path = os.path.join(self.data_dir, "train.csv")
        test_file_path = os.path.join(self.data_dir, "test.csv")

        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
        
        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    def __drop_uninformative(self) -> None:
        cols_to_keep = self.mi_scores_dataset().index.tolist()
        cols_to_keep += ["id", "price"] 
        self.data = self.data.loc[:, self.data.columns.isin(cols_to_keep)]

    def mi_scores_dataset(self) -> pd.Series:
        X = self.data[self.data["id"] <= 51635].copy()
        y = X.pop("price") 

        for colname in X.select_dtypes(["object", "category"]):
            X[colname], _ = X[colname].factorize()
            
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        mi_scores = mi_scores[mi_scores > 10e-6]  

        return mi_scores

    def baseline_score_dataset(self, model: XGBRegressor = XGBRegressor()) -> None:
        X = self.data[self.data["id"] <= 51635].copy()
        y = X.pop("price")  
        
        log_y = np.log(y)

        for colname in X.select_dtypes(["object", "category"]):
            X[colname], _ = X[colname].factorize()
        
        rmsle = np.sqrt(-cross_val_score(
            model, X, log_y, cv=5, scoring="neg_mean_squared_error"
        ).mean())
        
        mae = -cross_val_score(
            model, X, y, cv=5, scoring="neg_mean_absolute_error"
        ).mean()

        def smape(y_true, y_pred):
            y_true_orig = np.exp(y_true)
            y_pred_orig = np.exp(y_pred)
            return np.mean(2 * np.abs(y_pred_orig - y_true_orig) / (np.abs(y_true_orig) + np.abs(y_pred_orig) + 1e-8)) * 100

        smape_scorer = make_scorer(smape, greater_is_better=False)

        smape_score = -cross_val_score(model, X, y, cv=5, scoring=smape_scorer).mean()

        print(f"Baseline RMSLE: {rmsle:.5f}")
        print(f"Baseline MAE: {mae:.5f}")
        print(f"Baseline SMAPE: {smape_score:.5f}")
    
    def __plate_processed(self):
        def split_plate(row):
            first_letter = row[0]
            number = row[1:4]
            second_letter = row[4:6]
            region_code = row[6:]
            return first_letter, int(number), second_letter, int(region_code), first_letter + second_letter

        self.data[["first_letter", "number", "second_letter", "region_code", "combine"]] = self.data["plate"].apply(
            lambda x: pd.Series(split_plate(x))
        )

        self.data = self.data.drop(columns=["plate"])

        def has_repeated(x):
            x = str(x)
            return int(len(set(x)) < len(x))
        
        def has_sequential_numbers(num):
            n_str = str(num).zfill(3)  
            seq_up = ["123", "234", "345", "456", "567", "678", "789"]
            seq_down = ["987", "876", "765", "654", "543", "432", "321"]
            return any(seq in n_str for seq in seq_up) or any(seq in n_str for seq in seq_down)
        
        def has_mirror(x):
            x = str(x)
            return int(x == x[::-1])
        
        self.data["has_repeated_letters"] = self.data["combine"].apply(has_repeated)
        self.data["has_mirror_letters"] = self.data["combine"].apply(has_mirror)

        self.data["has_repeated_numbers_number"] = self.data["number"].apply(has_repeated)
        self.data["has_sequential_numbers_number"] = self.data["number"].apply(has_sequential_numbers)
        self.data["has_mirror_numbers_number"] = self.data["number"].apply(has_mirror)

        self.data["has_repeated_numbers_region_code"] = self.data["region_code"].apply(has_repeated)
        self.data["has_sequential_numbers_region_code"] = self.data["region_code"].apply(has_sequential_numbers)
        self.data["has_mirror_numbers_region_code"] = self.data["region_code"].apply(has_mirror)

    def __date_processed(self):
        import pandas as pd

        self.data["date"] = pd.to_datetime(self.data["date"])

        self.data["year"] = self.data["date"].dt.year
        self.data["month"] = self.data["date"].dt.month
        self.data["day"] = self.data["date"].dt.day

        self.data["days_from_initial_listing"] = self.data.groupby("combine")["date"].transform("min")
        self.data["days_from_initial_listing"] = (self.data["date"] - self.data["days_from_initial_listing"]).dt.days
        
        self.data["months_from_initial_listing"] = round(self.data["days_from_initial_listing"] / 30, 3)
        self.data["years_from_initial_listing"] = round(self.data["months_from_initial_listing"] / 12, 3)

        self.data["year_end"] = (self.data["month"] == 12).astype(int)

        self.data["listing_num"] = self.data.groupby("combine")["date"].rank(method="dense").astype(int)
        self.data["date"] = self.data["date"].dt.date
    
    def __supplemental_processed(self):
        def supple_region_detail(row):
            region_code = row["region_code"]

            for region_name, codes in REGION_CODES.items():
                if str(region_code) in codes:
                    return region_name
            return "None"
            
        def supple_government_detail(row):
            letter = row["combine"]
            region_code = row["region_code"]

            for (code_letters, _, region), details in GOVERNMENT_CODES.items():
                if letter == code_letters and region_code == int(region):
                    return details[2], details[3]
            
            return 0, 0

        self.data[["region_name"]] = self.data.apply(
            lambda x: pd.Series(supple_region_detail(x)),
            axis=1
        )

        self.data[["advantage_on_road", "significance"]] = self.data.apply(
            lambda x: pd.Series(supple_government_detail(x)),
            axis=1
        )

    def __price_processed(self):
        self.data["price"] = np.log1p(self.data["price"])

        available = "ABEKMHOPCTYX"
        exceptional_str = set()
        for i in available:
            for j in available:
                exceptional_str.add(i + j + i)
        exceptional_num = set([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 
            10, 20, 30, 40, 50, 60, 70, 80, 90,
            100, 200, 300, 400, 500, 600, 700, 800, 900, 
            123, 234, 345, 456, 567, 678, 789, 
            987, 876, 765, 654, 543, 432, 321
        ])
        for i in range(1, 10):
            for j in range(10):
                exceptional_num.add(i * 100 + j * 10 + i)

        def check_higher_price(row):
            combine = row["combine"]
            number = row["number"]
            region_code = row["region_code"]
            if combine in exceptional_str or number in exceptional_num or region_code in exceptional_num:
                return 1
            return 0
        
        self.data[["prestigious"]] = self.data.apply(
            lambda x: pd.Series(check_higher_price(x)),
            axis=1
        )

        valid_prices = self.data.dropna(subset=["price"])

        avg_price_by_region = valid_prices.groupby("region_code")["price"].mean()
        self.data["avg_price_by_region"] = self.data["region_code"].map(avg_price_by_region)

        avg_price_by_number = valid_prices.groupby("number")["price"].mean()
        self.data["avg_price_by_number"] = self.data["number"].map(avg_price_by_number)

        avg_price_by_first_letter = valid_prices.groupby("first_letter")["price"].mean()
        self.data["avg_price_by_first_letter"] = self.data["first_letter"].map(avg_price_by_first_letter)
        
        avg_price_by_second_letter = valid_prices.groupby("second_letter")["price"].mean()
        self.data["avg_price_by_second_letter"] = self.data["second_letter"].map(avg_price_by_second_letter)

        self.data["avg_price_by_letter"] = np.sqrt(
            self.data["avg_price_by_first_letter"] * self.data["avg_price_by_second_letter"]
        )

        avg_price_by_combine = valid_prices.groupby("combine")["price"].mean()
        self.data["avg_price_by_combine"] = self.data["combine"].map(avg_price_by_combine)

        avg_price_by_year = valid_prices.groupby("year")["price"].mean()
        self.data["avg_price_by_year"] = self.data["year"].map(avg_price_by_year)

        avg_price_by_prestigious = valid_prices.groupby("prestigious")["price"].mean()
        self.data["avg_price_by_prestigious"] = self.data["prestigious"].map(avg_price_by_prestigious)

    def data_processed(self) -> None:
        self.__plate_processed()
        self.__date_processed()
        self.__supplemental_processed()
        self.__price_processed()
        self.__drop_uninformative()

    def save_csv(self) -> str:
        self.output = "processed_data.csv"
        output_file = os.path.join(self.data_dir, self.output)
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return self.output

if __name__ == "__main__":
    data_path = "russian_car/data"
    data = Data(data_path)
    
    data.data_processed()
    data.baseline_score_dataset()
    data.save_csv()