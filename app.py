import streamlit as st
import pandas as pd
import pickle
import time

# Page config
st.set_page_config(page_title="Structural Health Monitoring", layout="wide")

# ---- Custom CSS (Animation + Styling) ----
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00ffe5;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    color: black;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    animation: fadeIn 1s ease-in;
}
@keyframes fadeIn {
    0% {opacity: 0;}
    100% {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown('<p class="title">🔧 Structural Health Monitoring System</p>', unsafe_allow_html=True)

# ---- Image Banner ----
st.image("https://images.unsplash.com/photo-1581093588401-22c5b7b5c88c", width="stretch")

st.write("### Upload vibration data and analyze bolt condition")

# ---- Load Models ----
rf = pickle.load(open("rf.pkl", "rb"))
svm = pickle.load(open("svm.pkl", "rb"))
xgb = pickle.load(open("xgb.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---- Sidebar ----
st.sidebar.header("⚙️ Settings")
amplitude = st.sidebar.selectbox("Select Amplitude", [0.5, 1, 2])

# ---- File Upload ----
uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.write("### 📊 Data Preview")
    st.dataframe(df.head())

    if st.button("🚀 Run Prediction"):

        with st.spinner("Analyzing vibration data..."):
            time.sleep(2)

            X = df[columns]
            X_scaled = scaler.transform(X)

            # Predictions
            rf_pred = label_encoder.inverse_transform(rf.predict(X_scaled))
            svm_pred = label_encoder.inverse_transform(svm.predict(X_scaled))
            xgb_pred = label_encoder.inverse_transform(xgb.predict(X_scaled))

            result_df = pd.DataFrame({
                "RF_Model": rf_pred,
                "SVM_Model": svm_pred,
                "XGB_Model": xgb_pred
            })

            # Final prediction (majority voting)
            result_df["Final"] = result_df.mode(axis=1)[0]

        st.success("✅ Prediction Completed!")

        # ---- Show Table ----
        st.write("### 🤖 Model Comparison")
        st.dataframe(result_df)

        # ---- Summary ----
        final_summary = result_df["Final"].value_counts().idxmax()

        st.markdown(
            f'<div class="card">Final Predicted Condition: {final_summary}</div>',
            unsafe_allow_html=True
        )

        # ---- Chart ----
        st.write("### 📈 Prediction Distribution")
        st.bar_chart(result_df["Final"].value_counts())

else:
    st.info("👆 Please upload an Excel file to begin")