
from pathlib import Path
import sqlite3
import pandas as pd

CLEAN_DIR = Path("data/clean")
DB_PATH = Path("data/instamart.db")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Read the cleaned datasets.
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Store order timestamps in SQLite as YYYY-MM-DD HH:MM:SS text.
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce"
).dt.strftime("%Y-%m-%d %H:%M:%S")

# Create the database and write the tables.
with sqlite3.connect(DB_PATH) as connection:
    customers.to_sql(
        "customers",
        connection,
        if_exists="replace",
        index=False
    )

    products.to_sql(
        "products",
        connection,
        if_exists="replace",
        index=False
    )

    orders.to_sql(
        "orders",
        connection,
        if_exists="replace",
        index=False
    )

    # Verify that the tables were created.
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        connection
    )

    print("DATABASE CREATED:", DB_PATH)
    print("\nTables:")
    print(tables.to_string(index=False))

    for table_name in ["customers", "products", "orders"]:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS row_count FROM {table_name}",
            connection
        )["row_count"].iloc[0]

        print(f"{table_name}: {count} rows")
