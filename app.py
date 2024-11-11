# ============================================================
# STREAMLIT WEB APP — Laptop Price Predictor
# ============================================================
# Run with: streamlit run app.py
# Make sure to run src/05_model_training.py first
# to generate data/pipe.pkl and data/df.pkl
# ============================================================

import streamlit as st
import pickle
import numpy as np

# Load trained model and dataframe
pipe = pickle.load(open("data/pipe.pkl", "rb"))
df   = pickle.load(open("data/df.pkl",   "rb"))

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")
st.title("💻 Laptop Price Predictor")
st.markdown("Enter the laptop specifications below to get an estimated price.")
st.markdown("---")

# Input fields
col1, col2 = st.columns(2)

with col1:
    company     = st.selectbox("Brand",         df["Company"].unique())
    laptop_type = st.selectbox("Type",          df["TypeName"].unique())
    ram         = st.selectbox("RAM (GB)",      [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight      = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0)
    touchscreen = st.selectbox("Touchscreen",   ["No", "Yes"])
    ips         = st.selectbox("IPS Display",   ["No", "Yes"])

with col2:
    screen_size = st.slider("Screen Size (inches)", 10.0, 18.0, 13.0)
    resolution  = st.selectbox("Screen Resolution", [
        "1920x1080", "1366x768", "1600x900",
        "3840x2160", "3200x1800", "2880x1800",
        "2560x1600", "2560x1440", "2304x1440"])
    cpu         = st.selectbox("CPU Brand",     df["Cpu_brand"].unique())
    hdd         = st.selectbox("HDD (GB)",      [0, 128, 256, 512, 1024, 2048])
    ssd         = st.selectbox("SSD (GB)",      [0, 8, 128, 256, 512, 1024])
    gpu         = st.selectbox("GPU Brand",     df["Gpu_brand"].unique())
    os          = st.selectbox("Operating System", df["OS"].unique())

st.markdown("---")

if st.button("🔍 Predict Price", use_container_width=True):
    # Encode binary inputs
    touchscreen_val = 1 if touchscreen == "Yes" else 0
    ips_val         = 1 if ips == "Yes"         else 0

    # Calculate PPI from resolution and screen size
    x_res = int(resolution.split("x")[0])
    y_res = int(resolution.split("x")[1])
    ppi   = ((x_res ** 2) + (y_res ** 2)) ** 0.5 / screen_size

    # Build query array
    query = np.array([[company, laptop_type, ram, weight,
                       touchscreen_val, ips_val, ppi,
                       cpu, hdd, ssd, gpu, os]])

    # Predict (model trained on log price)
    log_price     = pipe.predict(query)[0]
    predicted_price = int(np.exp(log_price))

    st.success(f"### 💰 Estimated Price: ₹ {predicted_price:,}")
    st.info(f"Specifications: {ram}GB RAM | {ssd}GB SSD | {hdd}GB HDD | {cpu} | {gpu}")
