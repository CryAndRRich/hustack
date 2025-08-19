import pandas as pd

def find_quiet_students(student: pd.DataFrame, exam: pd.DataFrame) -> pd.DataFrame:
    max_min = exam.groupby("exam_id")["score"].agg(["max", "min"]).reset_index()
    
    merged = exam.merge(max_min, on="exam_id")
    
    merged["quiet"] = (merged["score"] != merged["max"]) & (merged["score"] != merged["min"])
    
    quiet_students = merged.groupby("student_id")["quiet"].all()
    quiet_students = quiet_students[quiet_students].index.tolist()
    
    result = student[student["student_id"].isin(quiet_students)].sort_values("student_id").reset_index(drop=True)
    
    return result
