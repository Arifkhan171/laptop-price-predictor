# ============================================================
# STEP 5 — MODEL TRAINING & EVALUATION
# ============================================================
# - Train Random Forest Regressor with tuned hyperparameters
# - Evaluate with R2 Score and MAE
# - Compare multiple algorithms
# - Save trained model as pickle file
# ============================================================

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Load final dataset
df = pd.read_csv("../data/laptop_final.csv")

X = df.drop(columns=["Price"])
y = np.log(df["Price"])   # log transform for better distribution

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=2)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# Preprocessing
categorical_cols  = ["Company", "TypeName", "Cpu_brand", "Gpu_brand", "OS"]
numeric_features  = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(sparse_output=False,
                          drop="first",
                          handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_features)
])

# ── Compare multiple algorithms ──────────────────────────────
models = {
    "Linear Regression":     LinearRegression(),
    "Ridge Regression":      Ridge(alpha=10),
    "Random Forest":         RandomForestRegressor(n_estimators=100,
                                                    random_state=2,
                                                    max_samples=0.9,
                                                    max_features=0.13,
                                                    max_depth=19),
    "Extra Trees":           ExtraTreesRegressor(n_estimators=100, random_state=2),
    "Gradient Boosting":     GradientBoostingRegressor(n_estimators=100, random_state=2),
}

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

results = {}
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {"R2": round(r2, 4), "MAE": round(mae, 4)}
    print(f"{name:25s} → R2: {r2:.4f} | MAE: {mae:.4f}")

# ── Best model: Random Forest ────────────────────────────────
print("\n" + "=" * 55)
print("BEST MODEL: Random Forest Regressor")
print("=" * 55)

best_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100,
                                     random_state=2,
                                     max_samples=0.9,
                                     max_features=0.13,
                                     max_depth=19))
])
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 Score:  {r2:.4f} ({r2*100:.2f}%)")
print(f"MAE:       {mae:.4f}")
print(f"Model explains {r2*100:.1f}% of price variance")

# Actual vs Predicted plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", label="Perfect Prediction")
plt.xlabel("Actual Log Price")
plt.ylabel("Predicted Log Price")
plt.title(f"Random Forest — Actual vs Predicted (R2={r2:.4f})")
plt.legend()
plt.tight_layout()
plt.savefig("../data/actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()

# Save model and dataframe
pickle.dump(df, open("../data/df.pkl", "wb"))
pickle.dump(best_pipe, open("../data/pipe.pkl", "wb"))
print("\nModel saved to data/pipe.pkl")
print("DataFrame saved to data/df.pkl")
