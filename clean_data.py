from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def clean_customers():
    customers = pd.read_csv(RAW_DIR / "customers.csv")

    # Standardise city names: remove surrounding spaces and title-case.
    customers["city"] = (
        customers["city"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    # Standardise gender spellings.
    customers["gender"] = (
        customers["gender"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    gender_map = {
        "m": "male",
        "male": "male",
        "man": "male",
        "f": "female",
        "female": "female",
        "woman": "female",
        "": "unknown",
        "nan": "unknown",
        "unknown": "unknown",
    }

    customers["gender"] = customers["gender"].map(gender_map).fillna("unknown")

    # Convert age to numeric and treat ages outside 10–100 as missing.
    customers["age"] = pd.to_numeric(
        customers["age"],
        errors="coerce"
    )

    customers.loc[
        ~customers["age"].between(10, 100),
        "age"
    ] = pd.NA

    # Fill missing ages with the median age of the customer's city.
    city_medians = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_medians)

    # Convert signup date to datetime.
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"],
        errors="coerce"
    )

    # Save the cleaned customer dataset.
    customers.to_csv(CLEAN_DIR / "customers.csv", index=False)

    print("\nCUSTOMER CLEANING RESULTS")
    print("-" * 40)
    print("Rows:", len(customers))
    print("\nCity counts:")
    print(customers["city"].value_counts().sort_index())
    print("\nGender counts:")
    print(customers["gender"].value_counts())
    print("\nMissing ages:", customers["age"].isna().sum())
    print(
        "Age range:",
        customers["age"].min(),
        "to",
        customers["age"].max()
    )


def clean_products():
    products = pd.read_csv(RAW_DIR / "products.csv")

    products["product_id"] = products["product_id"].astype(str)
    products["price_inr"] = pd.to_numeric(
        products["price_inr"],
        errors="coerce"
    )

    products.to_csv(CLEAN_DIR / "products.csv", index=False)

    print("\nPRODUCT CLEANING RESULTS")
    print("-" * 40)
    print("Rows:", len(products))
    print("Missing values:")
    print(products.isna().sum())


def clean_orders():
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    products = pd.read_csv(CLEAN_DIR / "products.csv")

    # 1. Convert order timestamp to datetime.
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"],
        errors="coerce"
    )

    # 2. Remove exact duplicate rows.
    orders = orders.drop_duplicates()

    # 3. Keep the latest record for each order_id.
    orders = orders.sort_values("order_ts")
    orders = orders.drop_duplicates(
        subset="order_id",
        keep="last"
    )

    # Join product prices and categories for cleaning.
    orders = orders.merge(
        products[["product_id", "price_inr", "category"]],
        on="product_id",
        how="left"
    )

    # 4. Convert order_amount to numeric.
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"],
        errors="coerce"
    )

    # Fill missing amounts using product price, quantity and discount.
    calculated_amount = (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)

    orders["order_amount"] = orders["order_amount"].fillna(
        calculated_amount
    )

    # 5. Set negative delivery times to missing.
    orders.loc[
        orders["delivery_minutes"] < 0,
        "delivery_minutes"
    ] = pd.NA

    # Cancelled orders were not delivered; their delivery time stays missing.
    cancelled_mask = orders["status"] == "cancelled"
    orders.loc[cancelled_mask, "delivery_minutes"] = pd.NA

    # Impute completed orders using the product category median.
    completed_mask = orders["status"] == "completed"

    category_medians = (
        orders.loc[completed_mask]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )

    orders.loc[completed_mask, "delivery_minutes"] = (
        orders.loc[completed_mask, "delivery_minutes"]
        .fillna(category_medians)
    )

    # Remove helper columns and save.
    orders = orders.drop(columns=["price_inr", "category"])

    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"],
        errors="coerce"
    )

    orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

    print("\nORDER CLEANING RESULTS")
    print("-" * 40)
    print("Rows:", len(orders))
    print("\nStatus counts:")
    print(orders["status"].value_counts())
    print("\nMissing order amounts:", orders["order_amount"].isna().sum())
    print(
        "Missing completed delivery times:",
        orders.loc[completed_mask, "delivery_minutes"].isna().sum()
    )
    print(
        "Missing cancelled delivery times:",
        orders.loc[cancelled_mask, "delivery_minutes"].isna().sum()
    )
    print(
        "Completed revenue (INR):",
        round(
            orders.loc[orders["status"] == "completed", "order_amount"].sum(),
            2
        )
    )


if __name__ == "__main__":
    clean_customers()
    clean_products()
    clean_orders()