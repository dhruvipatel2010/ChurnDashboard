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
st.set_page_config(page_title="ML Dashboard", layout="wide")
st.title("📊 Machine Learning Dashboard")

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
    # TARGET COLUMN SELECTION
    # --------------------------------------------------
    st.subheader("🎯 Select Target Column")
    target_column = st.selectbox("Choose target column", data.columns)

    # --------------------------------------------------
    # SECTION 1 — DATA ANALYSIS
    # --------------------------------------------------
    st.header("📊 Dataset Analysis")

    col1, col2 = st.columns(2)

    # Target Distribution
    with col1:
        fig1, ax1 = plt.subplots()
        sns.countplot(x=target_column, data=data, ax=ax1)
        ax1.set_title("Target Distribution")
        st.pyplot(fig1)

    # Correlation Heatmap
    with col2:
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        if not numeric_data.empty:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.heatmap(numeric_data.corr(), cmap="coolwarm", ax=ax2)
            ax2.set_title("Correlation Heatmap")
            st.pyplot(fig2)

    # --------------------------------------------------
    # EXTRA FEATURE DISTRIBUTION
    # --------------------------------------------------
    st.header("📈 Feature Distribution")

    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) > 0:
        feature_selected = st.selectbox("Select feature to visualize", numeric_columns)

        col3, col4 = st.columns(2)

        with col3:
            fig3, ax3 = plt.subplots()
            sns.histplot(data[feature_selected], kde=True, ax=ax3)
            ax3.set_title(f"{feature_selected} Distribution")
            st.pyplot(fig3)

        with col4:
            fig4, ax4 = plt.subplots()
            sns.boxplot(x=data[feature_selected], ax=ax4)
            ax4.set_title(f"{feature_selected} Boxplot")
            st.pyplot(fig4)

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

        col5, col6 = st.columns(2)

        with col5:
            fig5, ax5 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax5)
            ax5.set_title("Confusion Matrix")
            st.pyplot(fig5)

        with col6:
            fig6, ax6 = plt.subplots()
            ax6.plot(fpr, tpr)
            ax6.plot([0,1],[0,1],'r--')
            ax6.set_title(f"ROC Curve (AUC = {roc_auc:.2f})")
            st.pyplot(fig6)

        # --------------------------------------------------
        # SECTION 3 — PREDICTION INSIGHTS
        # --------------------------------------------------
        st.header("📌 Prediction Insights")

        col7, col8 = st.columns(2)

        with col7:
            fig7, ax7 = plt.subplots()
            sns.histplot(y_pred_prob, kde=True, ax=ax7)
            ax7.set_title("Prediction Probability Distribution")
            st.pyplot(fig7)

        with col8:
            comparison_df = pd.DataFrame({
                "Actual": y.values,
                "Predicted": y_pred
            })
            st.dataframe(comparison_df.head())

    except Exception:
        st.warning("Model prediction failed. Dataset structure must match training features.")
