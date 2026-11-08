import sqlite3
import pandas as pd

DB_PATH = "data/instamart.db"

# query = """
# SELECT
#     customer_id,
#     SUM(order_amount) AS total_spend
# FROM orders
# WHERE status = 'completed'
#   AND order_amount > 500
# GROUP BY customer_id
# HAVING SUM(order_amount) > 8000
# ORDER BY total_spend DESC;
# """


# query = """
# SELECT
#     c.city,
#     p.category,
#     ROUND(SUM(o.order_amount), 2) AS revenue_inr
# FROM orders AS o
# JOIN customers AS c
#     ON o.customer_id = c.customer_id
# JOIN products AS p
#     ON o.product_id = p.product_id
# WHERE o.status = 'completed'
# GROUP BY
#     c.city,
#     p.category
# ORDER BY
#     c.city,
#     revenue_inr DESC;
# """

# query = """
# SELECT
#     product_id,
#     product_name,
#     price_inr
# FROM products
# WHERE price_inr > (
#     SELECT AVG(price_inr)
#     FROM products
# )
# ORDER BY price_inr DESC;
# """

# query = """
# SELECT
#     c.customer_id,
#     c.city
# FROM customers AS c
# LEFT JOIN orders AS o
#     ON c.customer_id = o.customer_id
#     AND o.status = 'completed'
# WHERE o.order_id IS NULL
# ORDER BY c.customer_id;
# """

# query = """
# WITH customer_spend AS (
#     SELECT
#         customer_id,
#         SUM(order_amount) AS total_spend
#     FROM orders
#     WHERE status = 'completed'
#     GROUP BY customer_id
# )
# SELECT
#     customer_id,
#     ROUND(total_spend, 2) AS total_spend
# FROM customer_spend
# WHERE total_spend > (
#     SELECT AVG(total_spend)
#     FROM customer_spend
# )
# ORDER BY total_spend DESC;
# """

# query = """
# WITH customer_spend AS (
#     SELECT
#         c.customer_id,
#         c.city,
#         SUM(o.order_amount) AS total_spend
#     FROM customers AS c
#     JOIN orders AS o
#         ON c.customer_id = o.customer_id
#     WHERE o.status = 'completed'
#     GROUP BY c.customer_id, c.city
# )
# SELECT
#     customer_id,
#     city,
#     ROUND(total_spend, 2) AS total_spend,
#     RANK() OVER (
#         PARTITION BY city
#         ORDER BY total_spend DESC
#     ) AS spend_rank,
#     DENSE_RANK() OVER (
#         PARTITION BY city
#         ORDER BY total_spend DESC
#     ) AS dense_rank,
#     ROW_NUMBER() OVER (
#         PARTITION BY city
#         ORDER BY total_spend DESC, customer_id
#     ) AS row_num
# FROM customer_spend
# ORDER BY city, spend_rank;
# """

query = """
WITH daily_revenue AS (
    SELECT
        DATE(order_ts) AS order_date,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY DATE(order_ts)
)
SELECT
    order_date,
    ROUND(daily_revenue, 2) AS daily_revenue,
    ROUND(
        SUM(daily_revenue) OVER (
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY order_date;
"""

with sqlite3.connect(DB_PATH) as connection:
    result = pd.read_sql_query(query, connection)

# print(result.to_string(index=False))
# print("\nNumber of products:", len(result))

print(result)
print("\nNumber of rows:", len(result))

# print("\nNumber of rows:", len(result))
# print("Total revenue (INR):", round(result["revenue_inr"].sum(), 2))
# print("\nLargest revenue cell:")
# print(result.loc[result["revenue_inr"].idxmax()].to_string())

# print(
#     "Rows where spend_rank differs from row_num:",
#     (result["spend_rank"] != result["row_num"]).sum()
# )

print(
    "Final cumulative revenue:",
    result["cumulative_revenue"].iloc[-1]
)