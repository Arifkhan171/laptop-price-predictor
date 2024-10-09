# ============================================================
# STEP 4 — FEATURE SELECTION
# ============================================================
# Selecting the most important features for the model:
# - Correlation analysis (numeric features vs Price)
# - Feature importance from Random Forest
# - Identifying and dropping low-importance features
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/laptop_final.csv")

print("=" * 55)
print("FEATURE SELECTION")
print("=" * 55)
print(f"Dataset shape: {df.shape}")
print(f"Features: {list(df.drop('Price', axis=1).columns)}")

# ── 1. Correlation — numeric features vs Price ───────────────
numeric_cols = ["Ram", "Weight", "Touchscreen", "IPS",
                "PPI", "HDD", "SSD"]
correlations = df[numeric_cols + ["Price"]].corr()["Price"].drop("Price")

print("\nCorrelation with Price (numeric features):")
print(correlations.sort_values(ascending=False).round(3))

plt.figure(figsize=(8, 5))
correlations.sort_values().plot(kind="barh", color="steelblue")
plt.title("Feature Correlation with Price")
plt.xlabel("Pearson Correlation")
plt.tight_layout()
plt.savefig("../data/feature_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 2. Random Forest Feature Importance ─────────────────────
X = df.drop(columns=["Price"])
y = np.log(df["Price"])

categorical_cols = ["Company", "TypeName", "Cpu_brand", "Gpu_brand", "OS"]
numeric_features  = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(sparse_output=False,
                          drop="first",
                          handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=2)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=2))
])
pipe.fit(X_train, y_train)

# Get feature names after one-hot encoding
cat_names = pipe.named_steps["preprocessor"]\
    .named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
all_features = cat_names + numeric_features

importances = pd.Series(
    pipe.named_steps["model"].feature_importances_,
    index=all_features
).sort_values(ascending=False)

print("\nTop 10 most important features (Random Forest):")
print(importances.head(10).round(4))

plt.figure(figsize=(10, 6))
importances.head(15).plot(kind="barh", color="darkorange")
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("../data/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nKey findings:")
print("   - RAM, SSD, PPI are the most important numeric features")
print("   - CPU brand significantly impacts price")
print("   - Weight and HDD have low importance")
print("   - All features kept for final model (tree-based models handle this well)")
