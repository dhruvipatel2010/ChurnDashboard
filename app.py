import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI ML Deep Learning Dashboard", layout="wide")
st.title("🤖 AI + ML + Deep Learning Dashboard — Colorful 3x3 Diagrams")

sns.set_theme(style="whitegrid")
colors = ["#FF4C4C", "#4285F4", "#0F9D58", "#F4B400", "#AB47BC"]

# -------------------------
# LOAD OR CREATE MODEL (Optional)
# -------------------------
try:
    model = load_model("churn_model.h5")
    model_input_features = 9
except:
    st.warning("⚠️ Deep learning model not found. Predictions will be skipped.")
    model = None

# -------------------------
# CSV UPLOADER
# -------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # -------------------------
    # SELECT TARGET COLUMN
    # -------------------------
    target_column = st.selectbox("Select Target Column", data.columns)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Fill missing and encode
    X = X.fillna(0)
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col].astype(str))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    target_type = "binary" if len(y.unique()) == 2 else "multiclass"

    numeric_data = X.select_dtypes(include=[np.number])
    categorical_data = X.select_dtypes(exclude=[np.number])

    # -------------------------
    # ROW 1 — DATA INSIGHTS
    # -------------------------
    st.header("📊 Row 1 — Data Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Target Distribution")
        counts = y.value_counts()
        fig1, ax1 = plt.subplots(figsize=(3,3))
        ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors[:len(counts)])
        st.pyplot(fig1)
    with col2:
        st.subheader("Feature Correlation")
        if not numeric_data.empty:
            fig2, ax2 = plt.subplots(figsize=(3,3))
            sns.heatmap(numeric_data.corr(), cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)
        else:
            st.info("No numeric features for heatmap.")
    with col3:
        st.subheader("First Feature Histogram")
        if not numeric_data.empty:
            fig3, ax3 = plt.subplots(figsize=(3,3))
            sns.histplot(numeric_data[numeric_data.columns[0]], kde=True, color="#4285F4", ax=ax3)
            st.pyplot(fig3)

    # -------------------------
    # ROW 2 — MODEL PERFORMANCE / PREDICTIONS
    # -------------------------
    st.header("🤖 Row 2 — Deep Learning Predictions")
    col4, col5, col6 = st.columns(3)
    can_predict = model is not None and X_scaled.shape[1] == model_input_features

    if can_predict:
        try:
            y_pred_prob = model.predict(X_scaled).flatten()
            if target_type=="binary":
                y_pred = (y_pred_prob > 0.5).astype(int)
            else:
                y_pred = np.argmax(y_pred_prob.reshape(-1,len(y.unique())), axis=1)

            # Confusion Matrix
            with col4:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y, y_pred)
                fig4, ax4 = plt.subplots(figsize=(3,3))
                sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax4)
                st.pyplot(fig4)

            # ROC Curve
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

            # Actual vs Predicted
            with col6:
                st.subheader("Actual vs Predicted")
                st.dataframe(pd.DataFrame({"Actual": y, "Predicted": y_pred}).head())

        except:
            for col in [col4, col5, col6]:
                col.info("Prediction failed.")

    else:
        for col in [col4, col5, col6]:
            col.info("Model prediction not available.")

    # -------------------------
    # ROW 3 — EXTRA INSIGHTS
    # -------------------------
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
