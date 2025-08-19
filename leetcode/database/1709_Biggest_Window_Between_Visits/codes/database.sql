WITH ordered_visits AS (
    SELECT 
        user_id,
        visit_date,
        LEAD(visit_date) OVER (PARTITION BY user_id ORDER BY visit_date) AS next_date
    FROM UserVisits
),
final_dates AS (
    SELECT 
        user_id,
        visit_date,
        COALESCE(next_date, DATE '2021-01-01') AS end_date
    FROM ordered_visits
),
differences AS (
    SELECT 
        user_id,
        DATEDIFF(end_date, visit_date) AS gap
    FROM final_dates
)
SELECT 
    user_id,
    MAX(gap) AS biggest_window
FROM differences
GROUP BY user_id
ORDER BY user_id;
