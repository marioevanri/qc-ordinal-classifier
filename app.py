# app.py
# Streamlit app untuk QC Ordinal Classifier
# File ini sudah berisi patch lengkap: safe model loading, input normalization,
# fallback labels_order, disabled confusion matrix (placeholder), dan UI yang
# menampilkan raw regressor output (ordinal score) + interpreted label.
#
# Requirements (simpan juga di requirements.txt):
# streamlit
# pandas
# scikit-learn==1.6.1
# joblib
# matplotlib
# seaborn
# numpy

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback
import sys
import matplotlib.pyplot as plt

# ---------------------------
# Helper: safe model loader
# ---------------------------
@st.cache_resource
def load_model_safe():
    """Load model and mapping, return dict with model, mapping, error.
    Uses joblib.load and captures exceptions to show in UI/terminal.
    """
    try:
        model = joblib.load("qc_ordinal_best.joblib")
        mapping = joblib.load("int_to_label.joblib")
        # Normalize mapping keys to int if saved as strings
        try:
            mapping = {int(k): v for k, v in mapping.items()}
        except Exception:
            # if mapping is already in good form, leave it
            pass

        return {"model": model, "mapping": mapping, "error": None}
    except Exception as e:
        tb = traceback.format_exc()
        # print to terminal for debugging
        print("ERROR saat load model:\n", tb, file=sys.stderr)
        return {"model": None, "mapping": None, "error": str(e) + "\n\n" + tb}


# ---------------------------
# App start
# ---------------------------
st.set_page_config(page_title="QC Ordinal Classifier", layout="centered")
st.title("QC Ordinal Classifier — Demo")
st.markdown("Masukkan fitur produk, lalu tekan **Prediksi**")

res = load_model_safe()
if res["error"]:
    st.error("Terjadi error saat load model — lihat terminal untuk detail.\n\n" + res["error"])
    st.stop()

model = res["model"]
int_to_label = res["mapping"]

# ---------------------------
# Build labels_order safely (DO NOT remove)
# ---------------------------
if int_to_label is None:
    # fallback labels (ubah bila pipeline training berbeda)
    labels_order = ["Not pass -2", "Not pass -1", "Pass", "Not pass +1", "Not pass +2"]
else:
    try:
        # ensure keys are ints
        int_to_label = {int(k): v for k, v in int_to_label.items()}
    except Exception:
        pass
    try:
        labels_order = [int_to_label[i] for i in sorted(int_to_label.keys())]
    except Exception:
        labels_order = ["Not pass -2", "Not pass -1", "Pass", "Not pass +1", "Not pass +2"]

# show label mapping for debugging (can comment out)
# st.write("Label mapping:", int_to_label)

# ---------------------------
# Input form
# ---------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.01)
        bulk_density = st.number_input("Bulk density", min_value=0.0, max_value=10.0, value=0.55, step=0.001)
        fat = st.number_input("Fat (%)", min_value=0.0, max_value=100.0, value=6.0, step=0.01)
    with col2:
        protein = st.number_input("Protein (%)", min_value=0.0, max_value=100.0, value=3.2, step=0.01)
        plant = st.selectbox("Plant", options=["Plant-1", "Plant-2"])  # keep same values as in training
        shift = st.selectbox("Shift", options=["Pagi", "Siang", "Malam"])  # example

    submitted = st.form_submit_button("Prediksi")

# ---------------------------
# Helper: normalize input df to match training feature names
# ---------------------------
def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and ensure dtypes align with training pipeline expectations.
    Adjust the mapping below to match exact column names used during training.
    """
    # mapping from UI names -> training column names (case sensitive)
    rename_map = {
        "moisture": "moisture",
        "bulk_density": "bulk_density",
        "fat": "fat",
        "protein": "protein",
        "plant": "plant",
        "shift": "shift",
        # If your pipeline used different names, change here.
    }

    # Ensure column names exist; lower-case keys expected from df creation
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # If pipeline expects numeric values for some columns and they are strings, try convert
    for col in df.columns:
        # skip category columns
        if col in ["plant", "shift"]:
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    return df

# ---------------------------
# Prediction block
# ---------------------------
if submitted:
    df_input = pd.DataFrame([{"moisture": moisture,
                              "bulk_density": bulk_density,
                              "fat": fat,
                              "protein": protein,
                              "plant": plant,
                              "shift": shift}])

    # normalize column names & types to match training
    df_input = normalize_input_df(df_input)

    st.write("Input yang diberikan:")
    st.table(df_input.T)

    # Try prediction inside try/except to show helpful errors
    try:
        # model may be a pipeline that accepts dataframe with original column names
        raw_pred = model.predict(df_input)
        # raw_pred might be array-like
        raw_value = float(np.ravel(raw_pred)[0])

        # round to nearest integer inside label range
        min_int = min(int_to_label.keys()) if int_to_label else 1
        max_int = max(int_to_label.keys()) if int_to_label else len(labels_order)
        pred_int = int(np.rint(raw_value).clip(min_int, max_int))
        pred_label = int_to_label.get(pred_int, str(pred_int)) if int_to_label else labels_order[pred_int - 1]

        # distance to 'Pass' (if exists in mapping)
        pass_int = None
        for k, v in (int_to_label or {}).items():
            if v == "Pass":
                pass_int = int(k)
                break

        dist_to_pass = None
        if pass_int is not None:
            dist_to_pass = abs(pred_int - pass_int)

        # Show result
        st.success(f"Prediksi: {pred_label} (Ordinal level: {pred_int})")
        st.write("Predicted (raw regressor output):", round(raw_value, 3))
        if dist_to_pass is not None:
            st.info(f"Jarak ke kelas 'Pass': {dist_to_pass} level")

        # Optional: show simple bar indicator for the raw score on ordinal scale
        try:
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.set_xlim(min_int - 0.5, max_int + 0.5)
            ax.hlines(0, min_int - 0.5, max_int + 0.5, color="#ddd", linewidth=6)
            ax.plot([raw_value], [0], marker="o", markersize=14)
            ax.set_yticks([])
            ax.set_xticks(list(range(min_int, max_int + 1)))
            ax.set_xticklabels([int_to_label.get(i, str(i)) for i in range(min_int, max_int + 1)], rotation=45)
            ax.set_title("Ordinal position (raw regressor output)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
        except Exception:
            pass

    except Exception as e:
        tb = traceback.format_exc()
        st.error("Terjadi error saat memprediksi — lihat terminal untuk detail.\n\n" + str(e))
        print(tb, file=sys.stderr)

# ---------------------------
# Placeholder for confusion matrix (removed)
# ---------------------------
st.markdown("---")
st.subheader("Visualisasi: Confusion matrix (dinonaktifkan)")
st.info("Confusion matrix dinonaktifkan di versi ini. Jika Anda ingin menampilkannya kembali, aktifkan blok heatmap di app.py dan pastikan 'labels_order' dan data test tersedia.")

# ---------------------------
# Footer / help
# ---------------------------
st.markdown("---")
st.markdown("**Catatan:** Pastikan `qc_ordinal_best.joblib` dan `int_to_label.joblib` ada di folder yang sama dengan `app.py`. Jika joblib.load gagal karena mismatch scikit-learn versi, pasang scikit-learn yang sama dengan versi saat training (contoh: `pip install scikit-learn==1.6.1`).")
