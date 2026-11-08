
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

files = [
    "customers.csv",
    "products.csv",
    "orders.csv",
    "tickets.csv",
    "policy_eval.csv",
    "agent_scenarios.csv",
]

for filename in files:
    path = RAW_DIR / filename
    df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print(f"FILE: {filename}")
    print("=" * 60)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nExact duplicate rows: {df.duplicated().sum()}")

    if filename == "orders.csv":
        print("\nOrder status counts:")
        print(df["status"].value_counts(dropna=False).to_string())

        print("\nFirst five orders:")
        print(df.head().to_string(index=False))
