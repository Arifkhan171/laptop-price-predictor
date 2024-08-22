# ============================================================
# STEP 3 — FEATURE GENERATION
# ============================================================
# Creating NEW features from existing columns:
# - Cpu brand: extract brand from full CPU name
# - Gpu brand: extract brand from full GPU name
# - Operating System: group 7 OS values into 3 categories
# - Log Price: transform skewed price to normal distribution
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/laptop_engineered.csv")

print("=" * 55)
print("FEATURE GENERATION")
print("=" * 55)

# ── 1. CPU Brand from CPU name ───────────────────────────────
print(f"Unique CPU values (sample): {df['Cpu'].unique()[:5]}")

df["Cpu_name"] = df["Cpu"].apply(lambda x: " ".join(x.split()[0:3]))

def fetch_processor(text):
    """Group CPU names into 4 categories"""
    if text in ["Intel Core i5", "Intel Core i7", "Intel Core i3"]:
        return text
    elif text.split()[0] == "Intel":
        return "Other Intel Processor"
    else:
        return "AMD Processor"

df["Cpu_brand"] = df["Cpu_name"].apply(fetch_processor)
df.drop(columns=["Cpu", "Cpu_name"], inplace=True)

print(f"\nCPU brand generated:")
print(df["Cpu_brand"].value_counts())

# ── 2. GPU Brand from GPU name ───────────────────────────────
df["Gpu_brand"] = df["Gpu"].apply(lambda x: x.split()[0])
print(f"\nGPU brand generated:")
print(df["Gpu_brand"].value_counts())

# Remove ARM GPUs (only 4 rows — too few to model)
df = df[df["Gpu_brand"] != "ARM"]
df.drop(columns=["Gpu"], inplace=True)
print("   Removed 4 ARM GPU rows")

# ── 3. Operating System categories ──────────────────────────
print(f"\nOriginal OS values: {df['OpSys'].unique()}")

def categorize_os(inp):
    """Group OS into 3 clean categories"""
    if inp in ["Windows 10", "Windows 7", "Windows 10 S"]:
        return "Windows"
    elif inp in ["macOS", "Mac OS X"]:
        return "Mac"
    else:
        return "Others/Linux"

df["OS"] = df["OpSys"].apply(categorize_os)
df.drop(columns=["OpSys"], inplace=True)

print(f"\nOS categories generated:")
print(df["OS"].value_counts())

# ── 4. Log Price — normalize skewed price distribution ───────
print(f"\nPrice distribution skewness (before): {df['Price'].skew():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["Price"], kde=True, ax=axes[0])
axes[0].set_title("Price Distribution (Original)")

sns.histplot(np.log(df["Price"]), kde=True, ax=axes[1], color="green")
axes[1].set_title("Price Distribution (Log Transformed)")

plt.tight_layout()
plt.savefig("../data/price_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Log Price skewness (after):  {np.log(df['Price']).skew():.3f}")
print("Log transformation reduces skewness significantly")

print(f"\nFinal shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")

df.to_csv("../data/laptop_final.csv", index=False)
print("\nSaved to data/laptop_final.csv")
