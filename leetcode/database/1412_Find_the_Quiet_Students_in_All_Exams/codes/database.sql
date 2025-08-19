SELECT s.student_id, s.student_name
FROM Student s
JOIN Exam e ON s.student_id = e.student_id
GROUP BY s.student_id, s.student_name
HAVING COUNT(*) = SUM(
    CASE 
        WHEN e.score NOT IN (
            (SELECT MAX(score) FROM Exam e2 WHERE e2.exam_id = e.exam_id),
            (SELECT MIN(score) FROM Exam e2 WHERE e2.exam_id = e.exam_id)
        )
        THEN 1 ELSE 0
    END
)
ORDER BY s.student_id;
