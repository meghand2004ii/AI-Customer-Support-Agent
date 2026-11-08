
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load clean datasets
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# 2. Define cutoff and observation windows
cutoff = pd.Timestamp("2025-09-01 00:00:00")
september_end = pd.Timestamp("2025-10-01 00:00:00")

# Orders strictly before the cutoff
pre_cutoff = orders[orders["order_ts"] < cutoff].copy()

# Completed orders before cutoff
completed_pre = pre_cutoff[
    pre_cutoff["status"] == "completed"
].copy()

# September completed orders: used ONLY for the label
september_completed = orders[
    (orders["status"] == "completed")
    & (orders["order_ts"] >= cutoff)
    & (orders["order_ts"] < september_end)
].copy()

# 3. Define cohort
# Every customer with at least one completed order before the cutoff
cohort_ids = completed_pre["customer_id"].unique()

cohort = pd.DataFrame({
    "customer_id": cohort_ids
})

# 4. Create lapse labels
# Lapsed = 1 if no completed September order
september_active_ids = set(
    september_completed["customer_id"].unique()
)

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(september_active_ids)
).astype(int)

# 5. Completed-order behavioural features
# All features below use only pre-cutoff orders

# Recency, frequency, monetary
rfm_features = (
    completed_pre.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean")
    )
    .reset_index()
)

rfm_features["recency"] = (
    cutoff - rfm_features["last_order_ts"]
).dt.days

rfm_features = rfm_features.drop(
    columns=["last_order_ts"]
)

# 6. Cancellation rate
# Cancelled orders / all final-status orders
# strictly before cutoff
final_status_orders = pre_cutoff[
    pre_cutoff["status"].isin(["completed", "cancelled"])
].copy()

cancellation_features = (
    final_status_orders.assign(
        is_cancelled=(
            final_status_orders["status"] == "cancelled"
        ).astype(int)
    )
    .groupby("customer_id")
    .agg(
        cancelled_orders=("is_cancelled", "sum"),
        final_status_orders=("order_id", "nunique")
    )
    .reset_index()
)

cancellation_features["cancellation_rate"] = (
    cancellation_features["cancelled_orders"]
    / cancellation_features["final_status_orders"]
)

cancellation_features = cancellation_features[
    ["customer_id", "cancellation_rate"]
]

# 7. Favourite product category
# Based only on completed pre-cutoff orders
completed_with_category = completed_pre.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_counts = (
    completed_with_category.groupby(
        ["customer_id", "category"]
    )
    .size()
    .reset_index(name="category_order_count")
)

# Sort to make category tie-breaking deterministic
category_counts = category_counts.sort_values(
    ["customer_id", "category_order_count", "category"],
    ascending=[True, False, True]
)

favourite_category = (
    category_counts.drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favourite_category"})
)

# 8. Customer attributes and tenure
customer_features = customers[
    ["customer_id", "city", "signup_date", "age", "gender"]
].copy()

customer_features["tenure_days"] = (
    cutoff - customer_features["signup_date"]
).dt.days

customer_features = customer_features.drop(
    columns=["signup_date"]
)

# 9. Combine all features
dataset = (
    cohort
    .merge(rfm_features, on="customer_id", how="left")
    .merge(
        cancellation_features,
        on="customer_id",
        how="left"
    )
    .merge(
        favourite_category,
        on="customer_id",
        how="left"
    )
    .merge(
        customer_features,
        on="customer_id",
        how="left"
    )
)

# 10. Validate cohort and label references
print("\nLapse cohort summary")
print("--------------------")
print("Cohort customers:", len(dataset))
print("Lapsed customers:", int(dataset["lapsed"].sum()))
print(
    "Lapse rate (%):",
    round(dataset["lapsed"].mean() * 100, 2)
)

# 11. Required stratified train/test split
X = dataset.drop(columns=["customer_id", "lapsed"])
y = dataset["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain/test split")
print("----------------")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

print("\nTraining label counts:")
print(y_train.value_counts().sort_index())

print("\nTest label counts:")
print(y_test.value_counts().sort_index())

# 12. Save feature dataset
dataset.to_csv(
    "outputs/lapse_features.csv",
    index=False
)

print("\nSaved outputs/lapse_features.csv")
