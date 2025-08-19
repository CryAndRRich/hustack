WITH all_dates AS (
    SELECT fail_date AS date, 'failed' AS period_state FROM Failed
    UNION ALL
    SELECT success_date, 'succeeded' FROM Succeeded
),
filtered_dates AS (
    SELECT * FROM all_dates
    WHERE date BETWEEN '2019-01-01' AND '2019-12-31'
),
numbered AS (
    SELECT 
        date,
        period_state,
        ROW_NUMBER() OVER (PARTITION BY period_state ORDER BY date) AS rn
    FROM filtered_dates
),
grouped AS (
    SELECT 
        period_state,
        DATE_SUB(date, INTERVAL rn DAY) AS grp
    FROM numbered
),
final_group AS (
    SELECT 
        date,
        period_state,
        DATE_SUB(date, INTERVAL ROW_NUMBER() OVER (PARTITION BY period_state ORDER BY date) DAY) AS grp
    FROM filtered_dates
)
SELECT 
    period_state,
    MIN(date) AS start_date,
    MAX(date) AS end_date
FROM final_group
GROUP BY period_state, grp
ORDER BY start_date;
