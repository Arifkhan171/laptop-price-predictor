# ============================================================
# STEP 2 — FEATURE ENGINEERING
# ============================================================
# Transforming raw columns into usable numeric features:
# - Extract Touchscreen flag from ScreenResolution
# - Extract IPS flag from ScreenResolution
# - Parse screen resolution into x_res and y_res
# - Calculate PPI (Pixels Per Inch) from resolution + screen size
# - Clean and parse Memory column into HDD/SSD/Hybrid/Flash
# ============================================================

import numpy as np
import pandas as pd

df = pd.read_csv("../data/laptop_cleaned.csv")

print("=" * 55)
print("FEATURE ENGINEERING")
print("=" * 55)

# ── 1. Touchscreen flag ──────────────────────────────────────
df["Touchscreen"] = df["ScreenResolution"].apply(
    lambda x: 1 if "Touchscreen" in x else 0)
print(f"Touchscreen — value counts:\n{df['Touchscreen'].value_counts()}")

# ── 2. IPS Panel flag ───────────────────────────────────────
df["IPS"] = df["ScreenResolution"].apply(
    lambda x: 1 if "IPS" in x else 0)
print(f"\nIPS Panel — value counts:\n{df['IPS'].value_counts()}")

# ── 3. Screen resolution → x_res, y_res ─────────────────────
new = df["ScreenResolution"].str.split("x", n=1, expand=True)
df["x_res"] = new[0].str.replace(",", "").str.findall(
    r"(\d+\.?\d+)").apply(lambda x: x[0]).astype("int")
df["y_res"] = new[1].astype("int")
print(f"\nResolution parsed — x_res range: {df['x_res'].min()}-{df['x_res'].max()}")
print(f"   y_res range: {df['y_res'].min()}-{df['y_res'].max()}")

# ── 4. PPI (Pixels Per Inch) ────────────────────────────────
df["PPI"] = (((df["x_res"] ** 2) + (df["y_res"] ** 2)) ** 0.5 / df["Inches"])
print(f"\nPPI calculated — range: {df['PPI'].min():.1f} to {df['PPI'].max():.1f}")

# Drop raw screen columns (replaced by PPI)
df.drop(columns=["ScreenResolution", "Inches", "x_res", "y_res"], inplace=True)
print("Dropped raw screen columns (replaced by PPI)")

# ── 5. Memory column → HDD, SSD, Hybrid, Flash_Storage ──────
df["Memory"] = df["Memory"].astype(str).str.replace(r"\.0", "", regex=False)
df["Memory"] = df["Memory"].str.replace("GB", "", regex=False)
df["Memory"] = df["Memory"].str.replace("TB", "000", regex=False)

new_mem = df["Memory"].str.split("+", n=1, expand=True)
df["first"]  = new_mem[0].str.strip()
df["second"] = new_mem[1].fillna("0").str.strip()

# Binary flags for storage types
for col, key in [("Layer1HDD", "HDD"), ("Layer1SSD", "SSD"),
                  ("Layer1Hybrid", "Hybrid"), ("Layer1Flash", "Flash Storage")]:
    df[col] = df["first"].apply(lambda x: 1 if key in x else 0)

for col, key in [("Layer2HDD", "HDD"), ("Layer2SSD", "SSD"),
                  ("Layer2Hybrid", "Hybrid"), ("Layer2Flash", "Flash Storage")]:
    df[col] = df["second"].apply(lambda x: 1 if key in x else 0)

df["first"]  = df["first"].str.replace(r"\D", "", regex=True).astype(int)
df["second"] = df["second"].str.replace(r"\D", "", regex=True).astype(int)

df["HDD"]           = df["first"] * df["Layer1HDD"]   + df["second"] * df["Layer2HDD"]
df["SSD"]           = df["first"] * df["Layer1SSD"]   + df["second"] * df["Layer2SSD"]
df["Hybrid"]        = df["first"] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"]
df["Flash_Storage"] = df["first"] * df["Layer1Flash"]  + df["second"] * df["Layer2Flash"]

df.drop(columns=["first", "second", "Memory",
                  "Layer1HDD", "Layer1SSD", "Layer1Hybrid", "Layer1Flash",
                  "Layer2HDD", "Layer2SSD", "Layer2Hybrid", "Layer2Flash",
                  "Hybrid", "Flash_Storage"], inplace=True)

print(f"\nMemory parsed — HDD range: 0-{df['HDD'].max()} GB")
print(f"   SSD range: 0-{df['SSD'].max()} GB")

print(f"\nShape after feature engineering: {df.shape}")
print(f"Columns: {list(df.columns)}")

df.to_csv("../data/laptop_engineered.csv", index=False)
print("\nSaved to data/laptop_engineered.csv")
