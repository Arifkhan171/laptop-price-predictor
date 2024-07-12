# 💻 Laptop Price Predictor

A Machine Learning project that predicts laptop prices based on hardware specifications — built with Random Forest Regressor achieving **88%+ R2 score** on test data.

---

## 📌 About This Project

This project predicts laptop prices using real-world data of 1300+ laptops. The pipeline covers the full ML workflow: data cleaning, feature engineering, feature generation, feature selection, model training and a Streamlit web app for live predictions.

---

## 📁 Project Structure

```
laptop-price-predictor/
│
├── src/
│   ├── 01_data_cleaning.py        # Clean raw data, fix data types
│   ├── 02_feature_engineering.py  # Transform raw columns into features
│   ├── 03_feature_generation.py   # Create new features from existing ones
│   ├── 04_feature_selection.py    # Select most important features
│   └── 05_model_training.py       # Train, compare and save the model
│
├── data/
│   └── laptop_data.csv            # Raw dataset (1303 laptops)
│
├── app.py                         # Streamlit web application
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔧 Step 1 — Data Cleaning

**File:** `src/01_data_cleaning.py`

Raw data problems fixed:

| Column | Problem | Fix |
|--------|---------|-----|
| Unnamed: 0 | Useless index column | Dropped |
| Ram | Stored as "8GB" (string) | Converted to integer 8 |
| Weight | Stored as "2.0kg" (string) | Converted to float 2.0 |

```python
df["Ram"]    = df["Ram"].str.replace("GB", "").astype("int32")
df["Weight"] = df["Weight"].str.replace("kg", "").astype("float64")
```

---

## ⚙️ Step 2 — Feature Engineering

**File:** `src/02_feature_engineering.py`

Transforming raw columns into numeric features the model can use:

| New Feature | Source Column | Method |
|-------------|--------------|--------|
| Touchscreen | ScreenResolution | Extract "Touchscreen" keyword → 0 or 1 |
| IPS | ScreenResolution | Extract "IPS" keyword → 0 or 1 |
| x_res, y_res | ScreenResolution | Parse "1920x1080" → 1920 and 1080 |
| PPI | x_res, y_res, Inches | `sqrt(x²+y²) / screen_size` |
| HDD | Memory | Parse "1TB HDD+256GB SSD" → 1000 |
| SSD | Memory | Parse "1TB HDD+256GB SSD" → 256 |

**PPI Formula:**
```python
df["PPI"] = (((df["x_res"]**2) + (df["y_res"]**2))**0.5) / df["Inches"]
```

**Why PPI instead of raw resolution?**
PPI (Pixels Per Inch) combines screen size and resolution into one meaningful metric — a 1920x1080 screen on 13" is much sharper than on 17".

---

## 🧪 Step 3 — Feature Generation

**File:** `src/03_feature_generation.py`

Creating **new features** from existing columns by applying domain knowledge:

**CPU Brand — grouped from 100+ CPU names into 4 categories:**
```
Intel Core i3  →  "Intel Core i3"
Intel Core i5  →  "Intel Core i5"
Intel Core i7  →  "Intel Core i7"
All others     →  "Other Intel" or "AMD Processor"
```

**GPU Brand — extracted from full GPU name:**
```
"Nvidia GeForce 940MX"  →  "Nvidia"
"Intel HD Graphics 620" →  "Intel"
"AMD Radeon RX 580"     →  "AMD"
```

**Operating System — grouped 7 values into 3:**
```
Windows 10, Windows 7, Windows 10 S  →  "Windows"
macOS, Mac OS X                       →  "Mac"
Linux, Chrome OS, Others              →  "Others/Linux"
```

**Log Price — normalize the skewed price column:**
```python
y = np.log(df["Price"])   # reduces skewness from 1.8 to ~0.1
```

---

## 📊 Step 4 — Feature Selection

**File:** `src/04_feature_selection.py`

Identifying which features matter most:

**Correlation with Price (numeric features):**
| Feature | Correlation |
|---------|------------|
| RAM | 0.67 |
| SSD | 0.58 |
| PPI | 0.51 |
| IPS | 0.44 |
| Touchscreen | 0.31 |
| Weight | 0.18 |
| HDD | -0.12 |

**Random Forest Feature Importance:**
- CPU brand and RAM are the top predictors
- SSD capacity strongly impacts price
- PPI (display quality) is significant
- HDD has low importance (cheap storage)

---

## 🤖 Step 5 — Model Training

**File:** `src/05_model_training.py`

**Algorithm Comparison:**

| Model | R2 Score | MAE |
|-------|----------|-----|
| Linear Regression | 0.71 | 0.31 |
| Ridge Regression | 0.72 | 0.30 |
| Extra Trees | 0.86 | 0.17 |
| Gradient Boosting | 0.84 | 0.18 |
| **Random Forest** | **0.88** | **0.16** |

**Best Model: Random Forest Regressor**
```python
RandomForestRegressor(
    n_estimators = 100,
    max_samples  = 0.9,
    max_features = 0.13,
    max_depth    = 19,
    random_state = 2
)
```

**Final Results:**
- R2 Score: **0.88** (model explains 88% of price variance)
- MAE: **0.16** (on log scale)

---

## 🌐 Streamlit Web App

**File:** `app.py`

Interactive web app where users select laptop specs and get instant price prediction.

**Input Features:**
- Brand, Type, RAM, Weight
- Touchscreen, IPS Display
- Screen Size, Resolution, CPU, GPU
- HDD, SSD, Operating System

---

## 🚀 How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/Arifkhan171/laptop-price-predictor.git
cd laptop-price-predictor
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run pipeline steps in order:**
```bash
python src/01_data_cleaning.py
python src/02_feature_engineering.py
python src/03_feature_generation.py
python src/04_feature_selection.py
python src/05_model_training.py
```

**4. Launch web app:**
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

---

## 📂 Dataset

- **Source:** Kaggle — Laptop Price Dataset
- **Size:** 1303 laptops × 13 columns
- **Features:** Company, Type, RAM, Weight, Screen, CPU, GPU, Storage, OS, Price

---

## 👤 Author

**Arif Khan**
Final Year CS Student | University of Loralai | ML • Deep Learning • Agentic AI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/arif-khan-71a711376)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/Arifkhan171)

---

> ⭐ Part of a complete Data Science & AI portfolio — from ML basics to Agentic AI systems.
