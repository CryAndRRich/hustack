WITH product_counts AS (
    SELECT
        customer_id,
        product_id,
        COUNT(*) AS order_count
    FROM Orders
    GROUP BY customer_id, product_id
),
max_counts AS (
    SELECT
        customer_id,
        MAX(order_count) AS max_order_count
    FROM product_counts
    GROUP BY customer_id
)
SELECT
    pc.customer_id,
    pc.product_id,
    p.product_name
FROM product_counts pc
JOIN max_counts mc ON pc.customer_id = mc.customer_id AND pc.order_count = mc.max_order_count
JOIN Products p ON pc.product_id = p.product_id
ORDER BY pc.customer_id, pc.product_id;
