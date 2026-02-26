import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Complete AI Dashboard", layout="wide")

st.title("🔥 Complete Universal AI Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("File Uploaded Successfully ✅")
    st.dataframe(df.head())

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # ============================
    # TABS
    # ============================
    tab1, tab2, tab3 = st.tabs(["📊 Basic Diagrams", "📈 Advanced Diagrams", "🤖 Models"])

    # ======================================
    # TAB 1 → BASIC DIAGRAMS
    # ======================================
    with tab1:

        col1, col2 = st.columns(2)

        # PIE
        with col1:
            st.subheader("🥧 Pie Chart")
            if categorical_cols:
                col_pie = st.selectbox("Pie Column", categorical_cols, key="pie")
                values = df[col_pie].value_counts()
                fig1, ax1 = plt.subplots(figsize=(5,5))
                ax1.pie(values, labels=values.index, autopct='%1.1f%%')
                st.pyplot(fig1)

        # DONUT
        with col2:
            st.subheader("🍩 Donut Chart")
            if categorical_cols:
                col_donut = st.selectbox("Donut Column", categorical_cols, key="donut")
                values2 = df[col_donut].value_counts()
                fig2, ax2 = plt.subplots(figsize=(5,5))
                ax2.pie(values2, labels=values2.index, autopct='%1.1f%%')
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                fig2.gca().add_artist(centre_circle)
                st.pyplot(fig2)

        # =============================
        # 🆕 3D STYLE PIE (NEW ADDED)
        # =============================
        st.subheader("🎨 3D Style Pie Chart")

        if categorical_cols:
            col_3d = st.selectbox("3D Pie Column", categorical_cols, key="3d_pie")
            values3d = df[col_3d].value_counts()

            explode = [0.05] * len(values3d)

            fig3d, ax3d = plt.subplots(figsize=(6,5))
            ax3d.pie(
                values3d,
                labels=values3d.index,
                autopct='%1.1f%%',
                shadow=True,
                explode=explode
            )
            ax3d.set_title("3D Effect Pie Chart")
            st.pyplot(fig3d)

        col3, col4 = st.columns(2)

        # HISTOGRAM
        with col3:
            st.subheader("📊 Histogram")
            if numeric_cols:
                col_hist = st.selectbox("Histogram Column", numeric_cols, key="hist")
                fig3, ax3 = plt.subplots(figsize=(6,4))
                sns.histplot(df[col_hist], kde=True, ax=ax3)
                st.pyplot(fig3)

        # BOX
        with col4:
            st.subheader("📦 Box Plot")
            if numeric_cols:
                col_box = st.selectbox("Box Column", numeric_cols, key="box")
                fig4, ax4 = plt.subplots(figsize=(6,4))
                sns.boxplot(y=df[col_box], ax=ax4)
                st.pyplot(fig4)

    # ======================================
    # TAB 2 → ADVANCED DIAGRAMS
    # ======================================
    with tab2:

        col5, col6 = st.columns(2)

        # LINE
        with col5:
            st.subheader("📈 Line Chart")
            if numeric_cols:
                col_line = st.selectbox("Line Column", numeric_cols, key="line")
                st.line_chart(df[col_line])

        # CIRCULAR BAR
        with col6:
            st.subheader("🌀 Circular Bar Plot")
            if categorical_cols:
                col_circ = st.selectbox("Circular Column", categorical_cols, key="circular")
                values4 = df[col_circ].value_counts()
                angles = np.linspace(0, 2*np.pi, len(values4), endpoint=False)

                fig6 = plt.figure(figsize=(5,5))
                ax6 = fig6.add_subplot(111, polar=True)
                ax6.bar(angles, values4.values)
                ax6.set_xticks(angles)
                ax6.set_xticklabels(values4.index)
                st.pyplot(fig6)

        # HEATMAP
        st.subheader("🔥 Correlation Heatmap")
        if len(numeric_cols) > 1:
            fig7, ax7 = plt.subplots(figsize=(6,5))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax7)
            st.pyplot(fig7)

    # ======================================
    # TAB 3 → MODELS
    # ======================================
    with tab3:

        st.subheader("🤖 Universal Smart Model")

        target_col = st.selectbox("Select Target Column", df.columns)

        if st.button("Train Model"):

            df_model = df.copy()

            # Remove ID-like columns
            for col in df_model.columns:
                if df_model[col].nunique() == len(df_model):
                    df_model.drop(columns=[col], inplace=True)

            if target_col not in df_model.columns:
                st.error("Selected target looks like ID column.")
            else:

                X = df_model.drop(columns=[target_col])
                y = df_model[target_col]

                if y.dtype == "object":
                    y = pd.factorize(y)[0]

                X = pd.get_dummies(X)

                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                num_classes = len(np.unique(y))

                if num_classes == 2:
                    output_units = 1
                    activation = "sigmoid"
                    loss_function = "binary_crossentropy"
                else:
                    output_units = num_classes
                    activation = "softmax"
                    loss_function = "sparse_categorical_crossentropy"

                model = Sequential([
                    Dense(128, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    Dense(output_units, activation=activation)
                ])

                model.compile(
                    optimizer='adam',
                    loss=loss_function,
                    metrics=['accuracy']
                )

                model.fit(
                    X_train, y_train,
                    epochs=30,
                    validation_split=0.2,
                    verbose=0
                )

                loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

                st.success(f"Model Accuracy: {accuracy*100:.2f}%")

else:
    st.info("Upload a CSV file to start.")
