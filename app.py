import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="ML Prediction Dashboard", layout="wide")

st.title("📊 Machine Learning Prediction Dashboard")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = load_model("churn_model.h5")

# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # --------------------------------------------------
    # SELECT TARGET COLUMN (Dynamic)
    # --------------------------------------------------
    st.subheader("🎯 Select Target Column")

    target_column = st.selectbox(
        "Choose the target column (what you want to predict)",
        data.columns
    )

    # --------------------------------------------------
    # SECTION 1 — DATA ANALYSIS
    # --------------------------------------------------
    st.header("📊 Dataset Analysis")

    col1, col2 = st.columns(2)

    # Target Distribution
    with col1:
        st.subheader("Target Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(x=target_column, data=data, ax=ax1)
        st.pyplot(fig1)

    # Correlation Heatmap (Numeric Only)
    with col2:
        st.subheader("Feature Correlation")
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)

        if not numeric_data.empty:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.heatmap(numeric_data.corr(), cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)
        else:
            st.warning("No numeric columns available for heatmap.")

    # --------------------------------------------------
    # DATA PREPROCESSING
    # --------------------------------------------------
    data = data.fillna(0)

    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col].astype(str))

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------
    try:
        y_pred_prob = model.predict(X_scaled).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

        cm = confusion_matrix(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # --------------------------------------------------
        # SECTION 2 — MODEL PERFORMANCE
        # --------------------------------------------------
        st.header("🤖 Model Performance")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Confusion Matrix")
            fig3, ax3 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
            st.pyplot(fig3)

        with col4:
            st.subheader("ROC Curve")
            fig4, ax4 = plt.subplots()
            ax4.plot(fpr, tpr)
            ax4.plot([0,1],[0,1],'r--')
            ax4.set_title(f"AUC = {roc_auc:.2f}")
            st.pyplot(fig4)

        # --------------------------------------------------
        # SECTION 3 — PREDICTION INSIGHTS
        # --------------------------------------------------
        st.header("📈 Prediction Insights")

        col5, col6 = st.columns(2)

        with col5:
            st.subheader("Prediction Probability Distribution")
            fig5, ax5 = plt.subplots()
            sns.histplot(y_pred_prob, kde=True, ax=ax5)
            st.pyplot(fig5)

        with col6:
            st.subheader("Actual vs Predicted")
            comparison_df = pd.DataFrame({
                "Actual": y.values,
                "Predicted": y_pred
            })
            st.dataframe(comparison_df.head())

        # --------------------------------------------------
        # KPI METRICS
        # --------------------------------------------------
        st.header("📌 Key Metrics")

        total_samples = len(data)
        positive_rate = (y.sum() / total_samples) * 100 if y.sum() != 0 else 0

        col7, col8 = st.columns(2)

        with col7:
            st.metric("Total Records", total_samples)

        with col8:
            st.metric("Positive Class %", f"{positive_rate:.2f}%")

    except Exception as e:
        st.error("Model prediction failed. Make sure uploaded dataset matches training features.")
