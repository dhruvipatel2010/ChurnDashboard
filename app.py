# -------------------------------
# STREAMLIT ML DASHBOARD — MULTIPLE CSV
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="ML Prediction Dashboard", layout="wide")
st.title("📊 ML Prediction Dashboard — Multi-CSV Friendly")

sns.set_theme(style="whitegrid")
colors = ["#FF4C4C", "#4285F4", "#0F9D58", "#F4B400", "#AB47BC"]

# -------------------------------
# LOAD MODEL & EXPECTED FEATURES
# -------------------------------
try:
    model = load_model("churn_model.h5")
    with open("model_features.json", "r") as f:
        expected_features = json.load(f)
except Exception as e:
    st.warning("⚠️ Model or 'model_features.json' not found. Upload both files in same folder.")
    st.stop()

# -------------------------------
# MULTIPLE FILE UPLOADER
# -------------------------------
uploaded_files = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        st.markdown(f"---\n## 📁 File: {file.name}")
        try:
            data = pd.read_csv(file)
        except:
            st.error("Cannot read CSV.")
            continue

        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        target_column = st.selectbox(f"Choose target column for {file.name}", data.columns, key=file.name)

        # -------------------------------
        # Align features with model
        # -------------------------------
        missing_features = [f for f in expected_features if f not in data.columns]
        extra_features = [f for f in data.columns if f not in expected_features + [target_column]]

        if missing_features:
            st.warning(f"Missing features (will fill with 0): {missing_features}")
            for f in missing_features:
                data[f] = 0  # Fill missing features

        if extra_features:
            st.info(f"Ignoring extra columns: {extra_features}")

        # Keep only expected features
        X = data[expected_features]
        y = data[target_column]

        # Encode categorical
        le = LabelEncoder()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col].astype(str))

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -------------------------------
        # MODEL PREDICTION
        # -------------------------------
        try:
            y_pred_prob = model.predict(X_scaled).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)

            cm = confusion_matrix(y, y_pred)
            fpr, tpr, _ = roc_curve(y, y_pred_prob)
            roc_auc = auc(fpr, tpr)

            numeric_data = X.select_dtypes(include=[np.number]).fillna(0)

            # --------------------------
            # ROW 1 — DATA INSIGHTS
            # --------------------------
            st.header("📊 Row 1 — Data Insights")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Target Distribution")
                counts = y.value_counts()
                fig1, ax1 = plt.subplots(figsize=(3,3))
                ax1.pie(counts, labels=counts.index, colors=colors[:len(counts)], autopct='%1.1f%%')
                st.pyplot(fig1)

            with col2:
                st.subheader("Feature Correlation")
                if not numeric_data.empty:
                    fig2, ax2 = plt.subplots(figsize=(3,3))
                    sns.heatmap(numeric_data.corr(), cmap="coolwarm", ax=ax2)
                    st.pyplot(fig2)
                else:
                    st.warning("No numeric columns for correlation.")

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

            with col4:
                st.subheader("Confusion Matrix")
                fig4, ax4 = plt.subplots(figsize=(3,3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4)
                st.pyplot(fig4)

            with col5:
                st.subheader("ROC Curve")
                fig5, ax5 = plt.subplots(figsize=(3,3))
                ax5.plot(fpr, tpr, color="#0F9D58")
                ax5.plot([0,1],[0,1],'r--')
                ax5.set_title(f"AUC = {roc_auc:.2f}")
                st.pyplot(fig5)

            with col6:
                st.subheader("Actual vs Predicted")
                comparison_df = pd.DataFrame({"Actual": y.values, "Predicted": y_pred})
                st.dataframe(comparison_df.head())

            # --------------------------
            # ROW 3 — EXTRA INSIGHTS
            # --------------------------
            st.header("📈 Row 3 — Extra Insights")
            col7, col8, col9 = st.columns(3)

            with col7:
                positive_rate = (y.sum() / len(y)) * 100 if y.sum() != 0 else 0
                st.metric("Positive Class %", f"{positive_rate:.2f}%")

            with col8:
                st.metric("Total Records", len(data))

            with col9:
                feature = X.columns[0]
                counts = X[feature].value_counts()
                fig9, ax9 = plt.subplots(figsize=(3,3))
                ax9.bar(counts.index, counts.values, color="#F4B400")
                ax9.set_xticklabels(counts.index, rotation=45)
                st.pyplot(fig9)

        except Exception as e:
            st.error(f"Model prediction failed for {file.name}. Error: {e}")
