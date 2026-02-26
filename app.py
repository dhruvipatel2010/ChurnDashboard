# -------------------------------
# FLEXIBLE ML DASHBOARD — MULTI-CSV SAFE WITH ACCURACY
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Flexible ML Dashboard", layout="wide")
st.title("📊 Flexible ML Prediction Dashboard — Multi-CSV Friendly")

sns.set_theme(style="whitegrid")
colors = ["#FF4C4C", "#4285F4", "#0F9D58", "#F4B400", "#AB47BC"]

# -------------------------------
# LOAD MODEL
# -------------------------------
try:
    model = load_model("churn_model.h5")
except Exception as e:
    st.warning("⚠️ Model file 'churn_model.h5' not found. Upload your trained model in the same folder.")
    st.stop()

# -------------------------------
# MULTIPLE CSV UPLOADER
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

        # -------------------------------
        # AUTO-DETECT TARGET COLUMN
        # -------------------------------
        binary_cols = [col for col in data.columns if sorted(data[col].dropna().unique()) == [0,1]]
        if binary_cols:
            target_column = binary_cols[0]
            st.info(f"Auto-selected target column: {target_column}")
        else:
            target_column = st.selectbox(f"Choose target column for {file.name}", data.columns, key=file.name)

        # -------------------------------
        # DYNAMIC FEATURE SELECTION
        # -------------------------------
        feature_cols = [col for col in data.columns if col != target_column]
        X = data[feature_cols].copy()
        y = data[target_column]

        # Fill missing values
        X = X.fillna(0)

        # Encode categorical automatically
        le = LabelEncoder()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col].astype(str))

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -------------------------------
        # DETECT TARGET TYPE
        # -------------------------------
        if len(y.unique()) == 2:
            target_type = "binary"
        else:
            target_type = "multiclass"

        # -------------------------------
        # MODEL PREDICTION & DIAGRAMS
        # -------------------------------
        try:
            y_pred_prob = model.predict(X_scaled).flatten()

            # Binary vs multiclass handling
            if target_type == "binary":
                y_pred = (y_pred_prob > 0.5).astype(int)
            else:  # multiclass: take argmax across classes
                y_pred = np.argmax(y_pred_prob.reshape(-1, len(y.unique())), axis=1)

            # ----------------------
            # METRICS
            # ----------------------
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='weighted')

            st.header("📌 Metrics")
            col_acc, col_f1 = st.columns(2)
            col_acc.metric("Accuracy", f"{accuracy*100:.2f}%")
            col_f1.metric("F1 Score", f"{f1:.2f}")

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
                sns.heatmap(confusion_matrix(y, y_pred, labels=y.unique()), annot=True, fmt="d", cmap="Blues", ax=ax4)
                st.pyplot(fig4)

            with col5:
                if target_type == "binary":
                    st.subheader("ROC Curve")
                    fpr, tpr, _ = roc_curve(y, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    fig5, ax5 = plt.subplots(figsize=(3,3))
                    ax5.plot(fpr, tpr, color="#0F9D58")
                    ax5.plot([0,1],[0,1],'r--')
                    ax5.set_title(f"AUC = {roc_auc:.2f}")
                    st.pyplot(fig5)
                else:
                    st.info("ROC curve skipped (multiclass target)")

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
                positive_rate = (y.sum() / len(y)) * 100 if target_type=="binary" else 0
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
            st.error(f"❌ Model prediction failed for {file.name}. Error: {e}")
