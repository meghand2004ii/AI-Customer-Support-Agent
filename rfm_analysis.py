import pandas as pd

# 1. Load cleaned orders
orders = pd.read_csv("data/clean/orders.csv")

# Convert order timestamps to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")

# 2. Keep completed orders only
completed_orders = orders[
    orders["status"].str.strip().str.lower() == "completed"
].copy()

# 3. Set the analysis snapshot date
snapshot_date = pd.Timestamp("2025-10-01")

# 4. Calculate RFM metrics for each customer
rfm = (
    completed_orders
    .groupby("customer_id")
    .agg(
        last_order_date=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    )
    .reset_index()
)

rfm["recency"] = (
    snapshot_date - rfm["last_order_date"].dt.normalize()
).dt.days

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary", "last_order_date"]
]

# Sort customers by recency (most recent first)
rfm = rfm.sort_values("recency").reset_index(drop=True)

# 5. Save the RFM table
rfm.to_csv("outputs/rfm_table.csv", index=False)

# 6. Display basic validation and preview
print("Completed orders:", len(completed_orders))
print("Customers with completed orders:", len(rfm))
print("\nRFM table preview:")
print(rfm.head(10))

print("\nRFM summary:")
print(rfm[["recency", "frequency", "monetary"]].describe())

print("\nMissing values:")
print(rfm.isna().sum())

print("\nSaved to outputs/rfm_table.csv")
