WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < (SELECT MAX(subtasks_count) FROM Tasks)
),
AllSubtasks AS (
    SELECT t.task_id, n.n AS subtask_id
    FROM Tasks t
    JOIN numbers n ON n.n <= t.subtasks_count
),
MissingSubtasks AS (
    SELECT a.task_id, a.subtask_id
    FROM AllSubtasks a
    LEFT JOIN Executed e
    ON a.task_id = e.task_id AND a.subtask_id = e.subtask_id
    WHERE e.subtask_id IS NULL
)
SELECT * FROM MissingSubtasks;
