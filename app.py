# -------------------------------
# FASHIONABLE AUTO-DIAGRAM ML DASHBOARD
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
st.set_page_config(page_title="Auto Diagram ML Dashboard", layout="wide")
st.title("📊 Auto-Diagram ML Dashboard — Upload Any CSV")

sns.set_theme(style="whitegrid")
colors = ["#FF4C4C", "#4285F4", "#0F9D58", "#F4B400", "#AB47BC"]

# -------------------------------
# LOAD MODEL (OPTIONAL)
# -------------------------------
try:
    model = load_model("churn_model.h5")
    model_input_features = 9
except:
    st.warning("⚠️ Model file 'churn_model.h5' not found. Predictions will be skipped.")
    model = None

# -------------------------------
# CSV UPLOADER
# -------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    st.markdown(f"---\n## 📁 File: {uploaded_file.name}")
    try:
        data = pd.read_csv(uploaded_file)
    except:
        st.error("Cannot read CSV.")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # -------------------------------
    # SELECT TARGET COLUMN
    # -------------------------------
    target_column = st.selectbox("Choose target column", data.columns)

    # -------------------------------
    # PREPROCESS FEATURES
    # -------------------------------
    feature_cols = [col for col in data.columns if col != target_column]
    X = data[feature_cols].copy()
    y = data[target_column]

    # Fill missing values
    X = X.fillna(0)

    # Encode categorical
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col].astype(str))

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Detect target type
    target_type = "binary" if len(y.unique()) == 2 else "multiclass"

    numeric_data = X.select_dtypes(include=[np.number]).fillna(0)
    categorical_data = X.select_dtypes(exclude=[np.number])

    # -------------------------------
    # ROW 1 — DATA INSIGHTS
    # -------------------------------
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
        st.subheader("First Feature Histogram")
        if not numeric_data.empty:
            fig3, ax3 = plt.subplots(figsize=(3,3))
            sns.histplot(numeric_data[numeric_data.columns[0]], kde=True, color="#4285F4", ax=ax3)
            st.pyplot(fig3)

    # -------------------------------
    # ROW 2 — MODEL PERFORMANCE (optional)
    # -------------------------------
    st.header("🤖 Row 2 — Model Performance (if available)")
    col4, col5, col6 = st.columns(3)
    can_predict = model is not None and X_scaled.shape[1] == model_input_features
    if can_predict:
        try:
            y_pred_prob = model.predict(X_scaled).flatten()
            if target_type=="binary":
                y_pred = (y_pred_prob > 0.5).astype(int)
            else:
                y_pred = np.argmax(y_pred_prob.reshape(-1, len(y.unique())), axis=1)

            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='weighted')

            with col4:
                st.subheader("Confusion Matrix")
                fig4, ax4 = plt.subplots(figsize=(3,3))
                sns.heatmap(confusion_matrix(y, y_pred, labels=y.unique()), annot=True, fmt="d", cmap="Blues", ax=ax4)
                st.pyplot(fig4)
            with col5:
                st.subheader("ROC Curve")
                if target_type=="binary":
                    fpr, tpr, _ = roc_curve(y, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    fig5, ax5 = plt.subplots(figsize=(3,3))
                    ax5.plot(fpr, tpr, color="#0F9D58")
                    ax5.plot([0,1],[0,1],'r--')
                    ax5.set_title(f"AUC = {roc_auc:.2f}")
                    st.pyplot(fig5)
                else:
                    st.info("ROC curve only for binary target.")
            with col6:
                st.subheader("Actual vs Predicted")
                comparison_df = pd.DataFrame({"Actual": y.values, "Predicted": y_pred})
                st.dataframe(comparison_df.head())
        except:
            st.info("Model prediction failed — skipping performance diagrams.")
    else:
        with col4:
            st.info("Confusion Matrix not available.")
        with col5:
            st.info("ROC Curve not available.")
        with col6:
            st.info("Actual vs Predicted table not available.")

    # -------------------------------
    # ROW 3 — EXTRA INSIGHTS
    # -------------------------------
    st.header("📈 Row 3 — Extra Insights")
    col7, col8, col9 = st.columns(3)
    with col7:
        positive_rate = (y.sum()/len(y))*100 if target_type=="binary" else 0
        st.metric("Positive Class %", f"{positive_rate:.2f}%")
    with col8:
        st.metric("Total Records", len(data))
    with col9:
        if not categorical_data.empty:
            first_cat = categorical_data.columns[0]
            counts = categorical_data[first_cat].value_counts()
            fig9, ax9 = plt.subplots(figsize=(3,3))
            ax9.bar(counts.index, counts.values, color="#F4B400")
            ax9.set_xticklabels(counts.index, rotation=45)
            st.pyplot(fig9)
        else:
            st.info("No categorical columns for bar chart.")
