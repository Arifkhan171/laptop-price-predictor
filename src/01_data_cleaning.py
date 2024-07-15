# ============================================================
# STEP 1 — DATA CLEANING
# ============================================================
# - Remove unnecessary columns
# - Fix data types (Ram: "8GB" -> 8, Weight: "2.0kg" -> 2.0)
# - Check duplicates and missing values
# ============================================================

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Load dataset
df = pd.read_csv("../data/laptop_data.csv")

print("=" * 55)
print("INITIAL DATASET INFO")
print("=" * 55)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicated rows: {df.duplicated().sum()}")
print(f"\nSample data:\n{df.head(3)}")

# ── Drop unnamed index column ────────────────────────────────
df.drop(columns=["Unnamed: 0"], inplace=True)
print("\nDropped 'Unnamed: 0' column")

# ── Fix Ram column: "8GB" → 8 ───────────────────────────────
df["Ram"] = df["Ram"].str.replace("GB", "").astype("int32")
print(f"Ram cleaned — unique values: {sorted(df['Ram'].unique())}")

# ── Fix Weight column: "2.0kg" → 2.0 ───────────────────────
df["Weight"] = df["Weight"].str.replace("kg", "").astype("float64")
print(f"Weight cleaned — range: {df['Weight'].min()} to {df['Weight'].max()} kg")

print("\n" + "=" * 55)
print("CLEANED DATASET")
print("=" * 55)
print(df.dtypes)
print(f"\nShape after cleaning: {df.shape}")

# Save cleaned data for next steps
df.to_csv("../data/laptop_cleaned.csv", index=False)
print("\nSaved cleaned data to data/laptop_cleaned.csv")
