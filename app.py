# -------------------------------
# STREAMLIT ML DASHBOARD
# -------------------------------
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
st.title("📊 ML Prediction Dashboard — Colorful & Smart")

# Use Google-style colors
sns.set_theme(style="whitegrid")
colors = ["#FF4C4C", "#4285F4", "#0F9D58", "#F4B400", "#AB47BC"]

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
try:
    model = load_model("churn_model.h5")
except Exception as e:
    st.warning("⚠️ Model file 'churn_model.h5' not found. Upload your model in the same folder.")

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
    target_column = st.selectbox(
        "Choose the target column (what you want to predict)",
        data.columns
    )

    # Fill missing values
    data = data.fillna(0)

    # Encode categorical variables
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col].astype(str))

    # Features & Target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Standardize
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

        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)

        # --------------------------
        # ROW 1 — DATA INSIGHTS
        # --------------------------
        st.header("📊 Row 1 — Data Insights")
        col1, col2, col3 = st.columns(3)

        # Target Distribution (Pie Chart)
        with col1:
            st.subheader("Target Distribution")
            counts = y.value_counts()
            fig1, ax1 = plt.subplots(figsize=(3,3))
            ax1.pie(counts, labels=counts.index, colors=colors[:len(counts)], autopct='%1.1f%%')
            st.pyplot(fig1)

        # Feature Correlation
        with col2:
            st.subheader("Feature Correlation")
            if not numeric_data.empty:
                fig2, ax2 = plt.subplots(figsize=(3,3))
                sns.heatmap(numeric_data.corr(), cmap="coolwarm", ax=ax2)
                st.pyplot(fig2)
            else:
                st.warning("No numeric columns available for correlation.")

        # Prediction Probability Distribution
        with col3:
            st.subheader("Prediction Probability")
            fig3, ax3 = plt.subplots(figsize=(3,3))
            sns.histplot(y_pred_prob, kde=True, color="#4285F4", ax=ax3)
            st.pyplot(fig3)

        # --------------------------
        # ROW 2 — MODEL PERFORMANCE
        # --------------------------
        st.header("🤖 Row 2 — Model Performance")
        col4, col5, col6 = st.columns(3)

        # Confusion Matrix
        with col4:
            st.subheader("Confusion Matrix")
            fig4, ax4 = plt.subplots(figsize=(3,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4)
            st.pyplot(fig4)

        # ROC Curve
        with col5:
            st.subheader("ROC Curve")
            fig5, ax5 = plt.subplots(figsize=(3,3))
            ax5.plot(fpr, tpr, color="#0F9D58")
            ax5.plot([0,1],[0,1],'r--')
            ax5.set_title(f"AUC = {roc_auc:.2f}")
            st.pyplot(fig5)

        # Actual vs Predicted
        with col6:
            st.subheader("Actual vs Predicted")
            comparison_df = pd.DataFrame({"Actual": y.values, "Predicted": y_pred})
            st.dataframe(comparison_df.head())

        # --------------------------
        # ROW 3 — EXTRA INSIGHTS
        # --------------------------
        st.header("📈 Row 3 — Extra Insights")
        col7, col8, col9 = st.columns(3)

        # Positive Class %
        with col7:
            st.subheader("Positive Class %")
            positive_rate = (y.sum() / len(y)) * 100 if y.sum() != 0 else 0
            st.metric("Positive Class %", f"{positive_rate:.2f}%")

        # Total Records
        with col8:
            st.subheader("Total Records")
            st.metric("Total Records", len(data))

        # Feature Value Counts (first numeric or categorical)
        with col9:
            st.subheader("Feature Value Counts")
            if not data.empty:
                feature = data.columns[0]
                counts = data[feature].value_counts()
                fig9, ax9 = plt.subplots(figsize=(3,3))
                ax9.bar(counts.index, counts.values, color="#F4B400")
                ax9.set_xticklabels(counts.index, rotation=45)
                st.pyplot(fig9)

    except Exception as e:
        st.error(f"Model prediction failed. Check dataset vs training features. Error: {e}")
