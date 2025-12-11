# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Load model & mapping
# ---------------------------
import traceback
import sys

@st.cache_resource
def load_model_safe():
    try:
        model = joblib.load("qc_ordinal_best.joblib")
        mapping = joblib.load("int_to_label.joblib")
        # normalize mapping keys to int (toleransi kalau tersimpan sebagai strings)
        mapping = {int(k): v for k, v in mapping.items()}
        return {"model": model, "mapping": mapping, "error": None}
    except Exception as e:
        tb = traceback.format_exc()
        # print ke terminal
        print("ERROR saat load model:", file=sys.stderr)
        print(tb, file=sys.stderr)
        # return error object agar UI bisa tunjukkan pesan
        return {"model": None, "mapping": None, "error": str(e) + "\n\n" + tb}

res = load_model_safe()
if res["error"]:
    st.error("Terjadi error saat load model — lihat detail di terminal. Pesan singkat:\n\n" + res["error"][:1000])
    st.stop()  # hentikan eksekusi lebih lanjut agar UI tidak blank tanpa pesan

model = res["model"]
int_to_label = res["mapping"]

# ============================
# Build labels_order safely
# ============================

# Jika mapping tidak ada, gunakan fallback
if int_to_label is None:
    labels_order = ["Not pass -2", "Not pass -1", "Pass", "Not pass +1", "Not pass +2"]
else:
    # Normalisasi: ubah key ke int jika key string
    try:
        int_to_label = {int(k): v for k, v in int_to_label.items()}
    except Exception:
        pass

    # Bangun urutan label: sort by integer key
    try:
        labels_order = [int_to_label[i] for i in sorted(int_to_label.keys())]
    except Exception:
        labels_order = ["Not pass -2", "Not pass -1", "Pass", "Not pass +1", "Not pass +2"]

# Debug (opsional)
# st.write("labels_order:", labels_order)


# ---------------------------
# Page layout
# ---------------------------
st.set_page_config(page_title="QC Ordinal Classifier", layout="centered")
st.title("QC Ordinal Classifier — Demo")
st.markdown("Masukkan fitur produk, lalu tekan **Prediksi**")

# ---------------------------
# Input form
# ---------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.01)
        bulk_density = st.number_input("Bulk density", min_value=0.0, max_value=2.0, value=0.55, step=0.001)
        fat = st.number_input("Fat (%)", min_value=0.0, max_value=50.0, value=6.0, step=0.01)
    with col2:
        protein = st.number_input("Protein (%)", min_value=0.0, max_value=50.0, value=3.2, step=0.01)
        plant = st.selectbox("Plant", options=["Plant-1", "Plant-2"])
        shift = st.selectbox("Shift", options=["Pagi", "Siang", "Malam"])

    submitted = st.form_submit_button("Prediksi")

# ---------------------------
# Predict & show result
# ---------------------------
if submitted:
    # prepare dataframe (must match feature order expected by pipeline)
    df_input = pd.DataFrame([{
        "moisture": moisture,
        "bulk_density": bulk_density,
        "fat": fat,
        "protein": protein,
        "plant": plant,
        "shift": shift
    }])

    # predict (regressor => round => map)
    pred_reg = model.predict(df_input)
    pred_int = int(np.rint(pred_reg).clip(min(int_to_label.keys()), max(int_to_label.keys())))
    pred_label = int_to_label[int(pred_int)]

    st.success(f"Prediksi: **{pred_label}** (Ordinal level: {pred_int})")

    # show small diagnostics
    st.write("Predicted (raw regressor output):", np.round(float(pred_reg), 3))

    # show a simple bar of distance to Pass (if you want)
    # find integer value of 'Pass'
    pass_int = None
    for k, v in int_to_label.items():
        if v == "Pass":
            pass_int = int(k)
            break

    if pass_int is not None:
        dist_to_pass = abs(pred_int - pass_int)
        st.info(f"Jarak ke kelas 'Pass': {dist_to_pass} level")

    # Optional: show a small table with the input
    st.write("Input yang diberikan:")
    st.table(df_input.T)

    