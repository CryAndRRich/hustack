import pandas as pd

def find_subtasks(tasks: pd.DataFrame, executed: pd.DataFrame) -> pd.DataFrame:
    all_subtasks = tasks.loc[tasks.index.repeat(tasks["subtasks_count"])].copy()
    all_subtasks["subtask_id"] = all_subtasks.groupby("task_id").cumcount() + 1
    result = all_subtasks[["task_id", "subtask_id"]].merge(
        executed, on=["task_id", "subtask_id"], how="left", indicator=True
    )
    return result[result["_merge"] == "left_only"][["task_id", "subtask_id"]].reset_index(drop=True)
