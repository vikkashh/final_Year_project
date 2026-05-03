import warnings
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Bolt Condition Predictor", layout="wide")

st.markdown(
    """
<style>
.title { text-align: center; font-size: 2.2rem; font-weight: 700; color: #1f77b4; }
.subtitle { text-align: center; color: #4b5563; margin-bottom: 0.75rem; }
.result-card {
    padding: 1rem;
    border-radius: 12px;
    background: linear-gradient(90deg, #dbeafe, #ecfeff);
    border: 1px solid #bfdbfe;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Bolt Health Monitoring and Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload vibration Excel data + select wind amplitude to predict condition and level.</div>',
    unsafe_allow_html=True,
)

col_img1, col_img2 = st.columns(2)
with col_img1:
    st.image(
        "https://images.unsplash.com/photo-1504917595217-d4dc5ebe6122?w=1200",
        caption="Structural monitoring",
        use_container_width=True,
    )
with col_img2:
    st.image(
        "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=1200",
        caption="Wind and vibration context",
        use_container_width=True,
    )


def load_pickle(path: str):
    return joblib.load(path)


def extract_features(arr: np.ndarray) -> dict:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "var": float(np.var(arr)),
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "mav": float(np.mean(np.abs(arr))),
    }


def create_feature_windows(df: pd.DataFrame, amplitude: float, windows: int = 30) -> pd.DataFrame:
    if df.shape[1] < 24:
        raise ValueError("Input Excel must contain at least 24 vibration columns (8 sensors x 3 axes).")

    raw_data = df.iloc[:, :24].apply(pd.to_numeric, errors="coerce")
    raw_data = raw_data.dropna(axis=0, how="any")
    if raw_data.empty:
        raise ValueError("No valid numeric vibration rows found in the first 24 columns.")

    xyz = raw_data.to_numpy()
    magnitude = []
    for i in range(0, 24, 3):
        x, y, z = xyz[:, i], xyz[:, i + 1], xyz[:, i + 2]
        mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        magnitude.append(mag)
    magnitude = np.array(magnitude).T  # shape: [n_samples, 8]

    windows = int(max(1, min(windows, len(magnitude))))
    window_size = len(magnitude) // windows
    if window_size == 0:
        windows = 1
        window_size = len(magnitude)

    output_rows = []
    for w in range(windows):
        start = w * window_size
        end = len(magnitude) if w == windows - 1 else (w + 1) * window_size
        chunk = magnitude[start:end]
        if len(chunk) == 0:
            continue

        row = {}
        for sensor_idx in range(8):
            f = extract_features(chunk[:, sensor_idx])
            for k, v in f.items():
                row[f"M{sensor_idx + 1}_{k}"] = v
        row["Amplitude"] = float(amplitude)
        output_rows.append(row)

    if not output_rows:
        raise ValueError("Could not generate feature windows from uploaded data.")
    return pd.DataFrame(output_rows)


@st.cache_resource(show_spinner=False)
def load_assets():
    assets = {
        "scaler": load_pickle("scaler.pkl"),
        "label_encoder": load_pickle("label_encoder.pkl"),
        "columns": load_pickle("columns.pkl"),
    }

    model_files = {
        "SVM": "svm.pkl",
        "LightGBM": "lgb.pkl",
        "XGBoost": "xgb.pkl",
        "Random Forest": "rf.pkl",
    }

    models = {}
    missing = []
    for name, path in model_files.items():
        if Path(path).exists():
            try:
                models[name] = load_pickle(path)
            except Exception as ex:
                missing.append(f"{name} ({ex})")
        else:
            missing.append(name)

    assets["models"] = models
    assets["missing_models"] = missing
    return assets


assets = load_assets()

st.sidebar.header("Prediction Inputs")
amplitude = st.sidebar.selectbox("Wind amplitude", [0.5, 1.0, 2.0], index=0)
windows = st.sidebar.slider("Number of feature windows", 5, 60, 30)

uploaded_file = st.file_uploader("Upload vibration Excel file", type=["xlsx", "xls"])

if assets["missing_models"]:
    st.warning(
        "Some models are unavailable and will be skipped: "
        + ", ".join(assets["missing_models"])
    )

if not assets["models"]:
    st.error("No model could be loaded. Ensure at least one model file is present.")
    st.stop()

if uploaded_file is None:
    st.info("Upload an Excel file to start prediction.")
    st.stop()

df_raw = pd.read_excel(uploaded_file)
st.subheader("Uploaded Data Preview")
st.dataframe(df_raw.head(), use_container_width=True)

if st.button("Run Prediction", type="primary"):
    with st.spinner("Processing vibration data and running models..."):
        time.sleep(1)
        try:
            df_features = create_feature_windows(df_raw, amplitude=amplitude, windows=windows)
            x_new = df_features.reindex(columns=assets["columns"], fill_value=0)
            x_scaled = assets["scaler"].transform(x_new)
        except Exception as ex:
            st.error(f"Preprocessing error: {ex}")
            st.stop()

        prediction_table = pd.DataFrame({"Window": np.arange(1, len(df_features) + 1)})
        for model_name, model in assets["models"].items():
            try:
                pred = model.predict(x_scaled)
                pred_labels = assets["label_encoder"].inverse_transform(pred.astype(int))
                prediction_table[model_name] = pred_labels
            except Exception as ex:
                prediction_table[model_name] = f"Error: {str(ex)[:60]}"

        model_cols = [c for c in prediction_table.columns if c != "Window"]
        vote_df = prediction_table[model_cols].copy()
        vote_df = vote_df.replace(to_replace=r"^Error:.*", value=np.nan, regex=True)
        prediction_table["Combined Model"] = vote_df.mode(axis=1, dropna=True)[0]

    st.success("Prediction completed successfully.")

    st.subheader("Model-wise and Combined Results (Table)")
    st.dataframe(prediction_table, use_container_width=True)

    st.subheader("Final Output")
    final_counts = prediction_table["Combined Model"].value_counts(dropna=True)
    if final_counts.empty:
        st.error("Combined prediction could not be generated due to model prediction errors.")
    else:
        final_result = final_counts.idxmax()
        confidence = (final_counts.max() / final_counts.sum()) * 100
        st.markdown(
            f"""
            <div class="result-card">
                <h4>Final Predicted Bolt Condition/Level: <span style="color:#0f766e;">{final_result}</span></h4>
                <p>Dominant class frequency across windows: <b>{confidence:.2f}%</b></p>
                <p>Total windows analyzed: <b>{len(prediction_table)}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Combined Prediction Distribution")
            st.bar_chart(final_counts)
        with col2:
            st.markdown("#### Per-model Agreement (Window level)")
            agreement = vote_df.notna().sum(axis=1).value_counts().sort_index()
            agreement.index = agreement.index.astype(str) + " valid model(s)"
            st.bar_chart(agreement)

        st.markdown("#### Class Percentage")
        st.dataframe(
            (final_counts / final_counts.sum() * 100).reset_index().rename(
                columns={"index": "Class", "Combined Model": "Percentage"}
            ),
            use_container_width=True,
        )