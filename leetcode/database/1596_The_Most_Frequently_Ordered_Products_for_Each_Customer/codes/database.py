import pandas as pd

def most_frequently_products(customers: pd.DataFrame, orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    product_counts = orders.groupby(["customer_id", "product_id"]).size().reset_index(name="order_count")
    max_counts = product_counts.groupby("customer_id")["order_count"].transform("max")
    frequent_products = product_counts[product_counts["order_count"] == max_counts]
    result = frequent_products.merge(products[["product_id", "product_name"]], on="product_id")
    return result[["customer_id", "product_id", "product_name"]].sort_values(["customer_id", "product_id"]).reset_index(drop=True)
