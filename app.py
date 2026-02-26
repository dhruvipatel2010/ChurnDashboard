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
st.set_page_config(page_title="Universal ML Dashboard", layout="wide")
st.title("📊 Universal CSV Analytics Dashboard")

# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload ANY CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # --------------------------------------------------
    # SECTION 1 — BASIC DATA VISUALS (ALWAYS WORK)
    # --------------------------------------------------
    st.header("📊 Basic Data Analysis")

    col1, col2 = st.columns(2)

    # --- Column Count Chart ---
    with col1:
        fig1, ax1 = plt.subplots()
        data.count().plot(kind="bar", ax=ax1)
        ax1.set_title("Non-Null Values Per Column")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

    # --- Missing Values Chart ---
    with col2:
        fig2, ax2 = plt.subplots()
        data.isnull().sum().plot(kind="bar", color="red", ax=ax2)
        ax2.set_title("Missing Values Per Column")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

    # --------------------------------------------------
    # SECTION 2 — NUMERIC ANALYSIS (IF EXISTS)
    # --------------------------------------------------
    st.header("📈 Numeric Analysis")

    numeric_data = data.select_dtypes(include=[np.number])

    if not numeric_data.empty:

        col3, col4 = st.columns(2)

        # Histogram of first numeric column
        with col3:
            first_numeric = numeric_data.columns[0]
            fig3, ax3 = plt.subplots()
            sns.histplot(numeric_data[first_numeric], kde=True, ax=ax3)
            ax3.set_title(f"{first_numeric} Distribution")
            st.pyplot(fig3)

        # Correlation Heatmap
        with col4:
            fig4, ax4 = plt.subplots(figsize=(6,4))
            sns.heatmap(numeric_data.corr(), cmap="coolwarm", ax=ax4)
            ax4.set_title("Correlation Heatmap")
            st.pyplot(fig4)

    else:
        st.warning("No numeric columns available for numeric analysis.")

    # --------------------------------------------------
    # SECTION 3 — TARGET BASED ANALYSIS (OPTIONAL)
    # --------------------------------------------------
    st.header("🎯 Optional Target Analysis")

    target_column = st.selectbox("Select target column (optional)", ["None"] + list(data.columns))

    if target_column != "None":

        # Encode text safely
        temp_data = data.copy().fillna(0)
        le = LabelEncoder()
        for col in temp_data.columns:
            if temp_data[col].dtype == 'object':
                temp_data[col] = le.fit_transform(temp_data[col].astype(str))

        X = temp_data.drop(target_column, axis=1)
        y = temp_data[target_column]

        if len(np.unique(y)) <= 10:  # classification style
            col5, col6 = st.columns(2)

            # Target Distribution
            with col5:
                fig5, ax5 = plt.subplots()
                sns.countplot(x=y, ax=ax5)
                ax5.set_title("Target Distribution")
                st.pyplot(fig5)

            # Simple Correlation with Target
            with col6:
                numeric_temp = temp_data.select_dtypes(include=[np.number])
                if not numeric_temp.empty:
                    corr_with_target = numeric_temp.corr()[target_column].sort_values()
                    fig6, ax6 = plt.subplots()
                    corr_with_target.plot(kind="barh", ax=ax6)
                    ax6.set_title("Feature Correlation with Target")
                    st.pyplot(fig6)

        else:
            st.info("Target column not suitable for classification visuals.")
