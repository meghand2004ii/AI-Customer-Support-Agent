import pandas as pd

# Load cleaned datasets
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

completed_orders = orders[
    orders["status"] == "completed"
].copy()

# Join completed orders with customers to get city
orders_with_city = completed_orders.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
)

# Join with products to get category
orders_with_details = orders_with_city.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# Part A
# Revenue pivot table with grand total

revenue_pivot = pd.pivot_table(
    orders_with_details,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    margins=True,
    margins_name="Grand Total"
)

revenue_pivot = revenue_pivot.round(2)

print("Revenue Pivot Table (INR):")
print(revenue_pivot)

#Per-city aggregation using groupby

city_summary = orders_with_city.groupby("city").agg(
    revenue_sum=("order_amount", "sum"),
    revenue_mean=("order_amount", "mean"),
    completed_order_count=("order_amount", "count")
)

city_summary = city_summary.round(2)

print("\nCompleted Order Statistics by City:")
print(city_summary)

# Verify the total against SQL

sql_total_revenue = 2998668.80

pandas_total_revenue = orders_with_details["order_amount"].sum()

print("\nSQL Total Revenue (INR):", sql_total_revenue)
print("Pandas Grand Total (INR):", round(pandas_total_revenue, 2))
print(
    "Totals match:",
    round(pandas_total_revenue, 2) == sql_total_revenue
)

# Part B
# Descriptive statistics and outlier detection
# Use completed order amounts
amounts = completed_orders["order_amount"]

# Calculate descriptive statistics
mean_amount = amounts.mean()
median_amount = amounts.median()
skewness = amounts.skew()

q1 = amounts.quantile(0.25)
q3 = amounts.quantile(0.75)

# Calculate interquartile range
iqr = q3 - q1

# Calculate 1.5 x IQR fences
lower_fence = q1 - (1.5 * iqr)
upper_fence = q3 + (1.5 * iqr)

# Identify outliers
outliers_below = amounts[amounts < lower_fence]
outliers_above = amounts[amounts > upper_fence]

# Print results
print("\nCompleted Order Amount Statistics")

print("Mean:", round(mean_amount, 2))
print("Median:", round(median_amount, 2))
print("Skewness:", round(skewness, 2))

print("Q1:", round(q1, 2))
print("Q2 (Median):", round(median_amount, 2))
print("Q3:", round(q3, 2))

print("IQR:", round(iqr, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))

print("Outliers below lower fence:", len(outliers_below))
print("Outliers above upper fence:", len(outliers_above))
print("Total outliers:", len(outliers_below) + len(outliers_above))


# Correlation analysis

correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation_data = completed_orders[correlation_columns].copy()

# Calculate Pearson correlation matrix
correlation_matrix = correlation_data.corr()

print("\nCorrelation Matrix")
print(correlation_matrix.round(3))

# Get each unique variable pair and its correlation
correlation_pairs = correlation_matrix.unstack()

# Remove self-correlations and duplicate pairs
correlation_pairs = correlation_pairs[
    correlation_pairs.index.get_level_values(0)
    < correlation_pairs.index.get_level_values(1)
]

# Find pair with largest absolute correlation
strongest_pair = correlation_pairs.abs().idxmax()
strongest_value = correlation_pairs.loc[strongest_pair]

print("\nStrongest correlation pair:")
print(strongest_pair)
print("Correlation:", round(strongest_value, 3))
print("Absolute correlation:", round(abs(strongest_value), 3))