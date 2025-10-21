import os
from typing import Tuple, Any

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

    def __name_processed(self) -> None:
        self.data["Title"] = self.data["Name"].str.extract(
            r",\s*(Mrs|Mr|Master|Miss|Don|Rev|Dr|Ms|Major|Lady|Sir|Col|Mme|Mlle|Capt|the Countess|Jonkheer)",
            expand=False
        )

        male_mask = self.data["Title"].isin(["Don", "Rev", "Dr", "Major", "Sir", "Col", "Capt", "Jonkheer"])
        female_mask = self.data["Title"].isin(["Lady", "Ms", "Mme", "Mlle", "the Countess"])
        self.data.loc[male_mask, "Title"] = "Mr"
        self.data.loc[female_mask, "Title"] = "Mrs"

        self.data["IsAdult"] = self.data.apply(self._check_adult, axis=1)

        self.data.loc[(self.data["Title"] == "Miss") & (self.data["IsAdult"] == 1), "Title"] = "Mrs"
        self.data.loc[(self.data["Title"] == "Mrs") & (self.data["IsAdult"] == 0), "Title"] = "Miss"
        self.data.loc[(self.data["Title"] == "Master") & (self.data["IsAdult"] == 1), "Title"] = "Mr"
        self.data.loc[(self.data["Title"] == "Mr") & (self.data["IsAdult"] == 0), "Title"] = "Master"

    def _check_adult(self, row: pd.Series) -> int:
        if pd.notnull(row["Age"]):
            return 1 if row["Age"] >= 21 else 0
        else:
            return 0 if row["Title"] in ["Master", "Miss"] else 1

    def __age_processed(self) -> None:
        self.avg_age: dict = {}
        for title in ["Mr", "Mrs", "Master", "Miss"]:
            valid_ages = self.data[(self.data["Title"] == title) & (self.data["Age"].notnull())]["Age"]
            mean_age = valid_ages.mean()

            def get_avg(age: float, male: bool) -> float:
                age = int(age * 10) // 10
                if age - int(age) < 0.5:
                    return int(age) + 0.5 if male else int(age)
                else:
                    return int(age + 1) if male else int(age) + 0.5

            if title in ["Mrs", "Miss"]:
                self.avg_age[title] = get_avg(mean_age, False)
            else:
                self.avg_age[title] = get_avg(mean_age, True)

        self.data["Age"] = self.data.apply(self._fill_age, axis=1)

    def _fill_age(self, row: pd.Series) -> int:
        if pd.notnull(row["Age"]):
            return row["Age"]
        title = row["Title"]
        if title in self.avg_age and not np.isnan(self.avg_age[title]):
            return self.avg_age[title]
        return 28

    def __sex_processed(self) -> None:
        self.data["Sex"] = self.data["Sex"].str.lower().map({"male": 1, "female": 0})

    def __relative_processed(self) -> None:
        self.data["FamSize"] = self.data["SibSp"] + self.data["Parch"] + 1

        self.data["hasNanny"] = 0
        self.data.loc[(self.data["Title"].isin(["Master", "Miss"])) & (self.data["FamSize"] == 1), "hasNanny"] = 1

    def __ticket_processed(self) -> None:
        self.data["TicketLetter"], self.data["TicketNumber"] = zip(*self.data["Ticket"].apply(self._split_ticket))

    def _split_ticket(self, ticket: Any) -> Tuple[str, str]:
        if not isinstance(ticket, str) or not ticket.strip():
            return "No", "0"
        items = ticket.split()
        ticket_item = items[0] if len(items) > 1 else "No"
        ticket_number = items[-1]
        return ticket_item, ticket_number

    def __fare_processed(self) -> None:
        self.data.loc[self.data["Fare"] == 0, "Fare"] = 1
        self.avg_fare_by_pclass = self.data.groupby("Pclass")["Fare"].mean()
        self.data["Fare"] = self.data.apply(self._fill_fare, axis=1)

    def _fill_fare(self, row: pd.Series) -> int:
        if pd.isnull(row["Fare"]):
            return self.avg_fare_by_pclass[row["Pclass"]]
        else:
            return row["Fare"]

    def __cabin_processed(self) -> None:
        self.data["Cabin"] = self.data["Cabin"].fillna("No")
        self.data[["CabinLet", "CabinNum"]] = self.data["Cabin"].apply(self._fill_cabin)

    def _fill_cabin(self, cabin: str) -> pd.Series:
        parts = cabin.split()
        if len(parts) < 2 and parts[-1].lower() == "no":
            return pd.Series(["N", -1])
        elif len(parts) < 2:
            let = parts[-1][0]
            num_str = parts[-1][1:]
            num_val = 0 if num_str == "" else int(num_str)
            return pd.Series([let, num_val])
        else:
            temp_letters = []
            temp_numbers = []
            for part in parts:
                if part == "F":
                    temp_letters.append("F")
                    temp_numbers.append(0)
                else:
                    temp_letters.append(part[0])
                    num_str = part[1:]
                    temp_numbers.append(int(num_str) if num_str else 0)
            if "F" in temp_letters:
                return pd.Series(["F", 0])
            else:
                avg_num = int(sum(temp_numbers) / len(temp_numbers))
                return pd.Series([temp_letters[-1], avg_num])

    def __embarked_processed(self) -> None:
        self.data["Embarked"] = self.data["Embarked"].fillna(self.data["Embarked"].mode()[0])

    def data_processed(self) -> None:
        self.__name_processed()
        self.__age_processed()
        self.__sex_processed()
        self.__relative_processed()
        self.__ticket_processed()
        self.__fare_processed()
        self.__cabin_processed()
        self.__embarked_processed()
        self.data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

    def save_csv(self) -> str:
        self.output = "processed_data.csv"
        output_file = os.path.join(self.data_dir, self.output)
        self.data.to_csv(output_file, index=False)
        print("Data saved to processed_data.csv!")
        return self.output

if __name__ == "__main__":
    data_path = "titanic/data"
    data = Data(data_path)
    data.data_processed()
    data.save_csv()
