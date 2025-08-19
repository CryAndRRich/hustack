import pandas as pd

def biggest_window(user_visits: pd.DataFrame) -> pd.DataFrame:
    user_visits["visit_date"] = pd.to_datetime(user_visits["visit_date"])
    user_visits = user_visits.sort_values(["user_id", "visit_date"])
    user_visits["next_date"] = user_visits.groupby("user_id")["visit_date"].shift(-1)
    user_visits["next_date"] = user_visits["next_date"].fillna(pd.Timestamp("2021-01-01"))
    user_visits["gap"] = (user_visits["next_date"] - user_visits["visit_date"]).dt.days
    result = user_visits.groupby("user_id")["gap"].max().reset_index(name="biggest_window")
    return result.sort_values("user_id").reset_index(drop=True)
