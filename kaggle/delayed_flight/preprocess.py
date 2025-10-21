import os
import numpy as np
import pandas as pd
import datetime

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
pd.set_option('future.no_silent_downcasting', True)

class Data():
    def __init__(self, data_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)

        train_file_path = os.path.join(self.data_dir, "train.csv")
        test_file_path = os.path.join(self.data_dir, "test.csv")
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)

        self.data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

        self.data["dep_delayed_15min"] = self.data["dep_delayed_15min"].replace({"N": 0, "Y": 1}).fillna(-1).astype(int)

        for col in self.data.select_dtypes(include=["object", "category"]).columns:
            self.data[col] = self.data[col].astype("category")

    def __drop_uninformative(self) -> None:
        mi_scores = self.mi_scores_dataset()
        cols_to_keep = mi_scores[mi_scores > 10 ** -5].index.tolist()
        cols_to_keep += ["id", "dep_delayed_15min"]
        self.data = self.data.loc[:, self.data.columns.isin(cols_to_keep)]

    def mi_scores_dataset(self) -> pd.Series:
        df = self.data[self.data["id"] > 99999].copy()
        y = df.pop("dep_delayed_15min") 
        df.pop("dep_delayed_15min") 

        for colname in df.select_dtypes(["object", "category"]):
            df[colname], _ = df[colname].factorize()

        discrete_features = [pd.api.types.is_integer_dtype(t) for t in df.dtypes]
        mi = mutual_info_classif(df, y, discrete_features=discrete_features, random_state=0)

        mi_scores = pd.Series(mi, index=df.columns, name="MI Scores").sort_values(ascending=False)
        return mi_scores

    def baseline_score_dataset(self, model: XGBClassifier = XGBClassifier()) -> None:
        df = self.data[self.data["id"] > 99999].copy()
        y = df.pop("dep_delayed_15min")
        df.pop("id")

        for col in df.select_dtypes(["category"]):
            df[col] = df[col].cat.codes

        metrics = ["f1", "roc_auc"]
        for metric in metrics:
            score = cross_val_score(model, df, y, cv=5, scoring=metric).mean()
            print(f"Baseline {metric.capitalize()}: {score:.5f}")

    def __time_processed(self) -> None:
        self.data["Month"] = self.data["Month"].apply(lambda x: x.split("-")[1]).astype(int)
        def check_season(x: int) -> int:
            if x == 12 or 1 <= x <= 2: 
                return "Winter"
            elif 3 <= x <= 5: 
                return "Spring"
            elif 6 <= x <= 8: 
                return "Summer"
            else:
                return "Fall"

        self.data["SeasonType"] = self.data["Month"].apply(check_season).astype("category")

        self.data["DayofMonth"] = self.data["DayofMonth"].apply(lambda x: x.split("-")[1]).astype(int)

        self.data["DayOfWeek"] = self.data["DayOfWeek"].apply(lambda x: x.split("-")[1]).astype(int)
        self.data["Weekend"] = self.data["DayOfWeek"].apply(lambda x: 1 if x > 5 else 0)

        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        day_in_year = {1: 0}
        for num in range(1, 12): 
            day_in_year[num + 1] = sum(days_in_month[0:num])

        self.data["DayInYear"] = self.data["Month"].map(day_in_year) + self.data["DayofMonth"]
        self.data["DIYSin"] = np.sin(2 * np.pi * self.data["DayInYear"] / 365.0)
        self.data["DIYCos"] = np.cos(2 * np.pi * self.data["DayInYear"] / 365.0)

        self.data["Year"] = "Y" + (2018 - (self.data["DayInYear"] - self.data["DayOfWeek"]) % 7).astype(str)
        self.data["Year"] = self.data["Year"].astype("category")

        def check_holiday(month, day, year) -> str:
            fixed_holidays = {
                (1, 1): "New Year's Day",
                (7, 4): "Independence Day",
                (11, 11): "Veterans Day",
                (12, 25): "Christmas Day"
            }

            year = int(year[1:])
            try:
                date = datetime.date(year, month, day)
            except ValueError:
                return "Non-Holiday"

            memorial_day = max(
                datetime.date(year, 5, d)
                for d in range(25, 32)
                if datetime.date(year, 5, d).weekday() == 0
            )

            floating_holidays = {
                memorial_day: "Memorial Day",
            }

            if (month, day) in fixed_holidays:
                return fixed_holidays[(month, day)]
            elif date in floating_holidays:
                return floating_holidays[date]
            else:
                return "Non-Holiday"

        self.data["HolidayType"] = self.data.apply(
            lambda row: check_holiday(row["Month"], 
                                      row["DayofMonth"],
                                      row["Year"]), 
            axis=1
        ).astype("category")

        self.data["IsHoliday"] = (self.data["HolidayType"] != "Non-Holiday").astype(int)

        self.data["DepHour"] = (self.data["DepTime"] // 100) % 24 + (self.data["DepTime"] % 100) / 60.0
        self.data["HourSin"] = np.sin(2 * np.pi * self.data["DepHour"] / 24.0)
        self.data["HourCos"] = np.cos(2 * np.pi * self.data["DepHour"] / 24.0)
        self.data["HourS"] = np.square(self.data["DepHour"])

        def check_daytime(x: int) -> str:
            if 459 < x <= 1159:
                return "Morning"
            elif 1159 < x <= 1659:
                return "Afternoon"
            elif 1659 < x <= 2059:
                return "Evening"
            return "Midnight"
        
        def deptime_bin(x: int) -> str:
            if x <= 600:
                return "vem"
            elif 600 < x <= 900:
                return "m"
            elif 900 < x <= 1200:
                return "mm"
            elif 1200 < x <= 1500:
                return "maf"
            elif 1500 < x <= 1800:
                return "af"
            elif 1800 < x <= 2100:
                return "n"
            elif 2100 < x <= 2400:
                return "nn"
            return "lm"

        self.data["TimeType"] = self.data["DepTime"].apply(check_daytime).astype("category")
        self.data["DepTimeBin"] = self.data["DepTime"].apply(deptime_bin).astype("category")

        self.data = self.data.drop(columns=["TimeType", "DepTime", "DayInYear"])

    def __code_processed(self) -> None:
        self.data["Route"] = self.data["Origin"].astype(str) + self.data["Dest"].astype(str)
        self.data["CarrierOrigin"] = self.data["UniqueCarrier"].astype(str) + self.data["Origin"].astype(str)
        self.data["CarrierDest"] = self.data["UniqueCarrier"].astype(str) + self.data["Dest"].astype(str)

        self.data["Route"] = self.data["Route"].astype("category")
        self.data["CarrierOrigin"] = self.data["Route"].astype("category")
        self.data["CarrierDest"] = self.data["Route"].astype("category")

        self.data["DepHourDest"] = self.data["DepHour"].astype(str) + "_" + self.data["Dest"].astype(str)
        self.data["DepHourOrigin"] = self.data["DepHour"].astype(str) + "_" + self.data["Origin"].astype(str)
        self.data["DepHourCarrier"] = self.data["DepHour"].astype(str) + "_" + self.data["UniqueCarrier"].astype(str)
        self.data["DestDepHourCarrier"] = self.data["Dest"].astype(str) + "_" + self.data["DepHour"].astype(str) + "_" + self.data["UniqueCarrier"].astype(str)

        self.data["DepHourDest"] = self.data["DepHourDest"].astype("category")
        self.data["DepHourOrigin"] = self.data["DepHourOrigin"].astype("category")
        self.data["DepHourCarrier"] = self.data["DepHourCarrier"].astype("category")
        self.data["DestDepHourCarrier"] = self.data["DestDepHourCarrier"].astype("category")

        self.data = self.data.drop(columns=["DepHour"])

        df = self.data[self.data["dep_delayed_15min"] != -1].copy()
        origin_delay_rate = df.groupby("Origin", observed=False)["dep_delayed_15min"].apply(lambda x: (x == 1).mean())
        dest_delay_rate = df.groupby("Dest", observed=False)["dep_delayed_15min"].apply(lambda x: (x == 1).mean())
        carrier_delay_rate = df.groupby("UniqueCarrier", observed=False)["dep_delayed_15min"].apply(lambda x: (x == 1).mean())
        self.data["OriginDelayRate"] = self.data["Origin"].map(origin_delay_rate)
        self.data["DestDelayRate"] = self.data["Dest"].map(dest_delay_rate)
        self.data["CarrierDelayRate"] = self.data["UniqueCarrier"].map(carrier_delay_rate)

    def __distance_processed(self) -> None:
        self.data["DistanceS"] = np.square(self.data["Distance"])
        self.data["HourDist"] = self.data["HourS"] * self.data["DistanceS"]

        self.data.loc[self.data["Distance"] <= 500 , "DistBin"] = "very short"
        self.data.loc[(self.data["Distance"] > 500) & (self.data["Distance"] <= 1000), "DistBin"] = "short"
        self.data.loc[(self.data["Distance"] > 1000) & (self.data["Distance"] <= 1500), "DistBin"] = "mid"
        self.data.loc[(self.data["Distance"] > 1500) & (self.data["Distance"] <= 2000), "DistBin"] = "midlong"
        self.data.loc[(self.data["Distance"] > 2000) & (self.data["Distance"] <= 2500), "DistBin"] = "long"
        self.data.loc[self.data["Distance"] > 2500, "DistBin"] = "very long"
        self.data["DistBin"] = self.data["DistBin"].astype("category")
        self.data = self.data.drop(columns=["Distance"])
    
    def data_processed(self) -> None:
        self.__time_processed()
        self.__code_processed()
        self.__distance_processed()
        self.__drop_uninformative()

    def save_csv(self) -> str:
        output_file = os.path.join(self.data_dir, "processed_data.csv")
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return "processed_data.csv"

if __name__ == "__main__":
    data_path = "delayed_flight/data"
    data = Data(data_path)

    data.data_processed()
    data.mi_scores_dataset()
    data.baseline_score_dataset()
    data.save_csv()