import pandas as pd

def report_contiguous_dates(failed: pd.DataFrame, succeeded: pd.DataFrame) -> pd.DataFrame:
    failed = failed.rename(columns={"fail_date": "date"}).copy()
    succeeded = succeeded.rename(columns={"success_date": "date"}).copy()
    failed["date"] = pd.to_datetime(failed["date"], errors="coerce")
    succeeded["date"] = pd.to_datetime(succeeded["date"], errors="coerce")

    failed["period_state"] = "failed"
    succeeded["period_state"] = "succeeded"

    combined = pd.concat([failed, succeeded], ignore_index=True)
    combined = combined[(combined["date"] >= "2019-01-01") & (combined["date"] <= "2019-12-31")]

    combined = combined.sort_values("date").reset_index(drop=True)

    combined["grp"] = (combined["period_state"] != combined["period_state"].shift()).cumsum()

    result = combined.groupby(["grp", "period_state"], as_index=False).agg(
        start_date=("date", "min"),
        end_date=("date", "max")
    )

    return result[["period_state", "start_date", "end_date"]]
