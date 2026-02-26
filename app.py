import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model

st.title("Customer Churn Prediction Dashboard")

model = load_model("churn_model.h5")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col])

    X = data.drop("returned", axis=1)
    y = data["returned"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_pred_prob = model.predict(X_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)

    cm = confusion_matrix(y, y_pred)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr)
    ax2.plot([0,1],[0,1],'r--')
    ax2.set_title(f"AUC = {roc_auc:.2f}")
    st.pyplot(fig2)