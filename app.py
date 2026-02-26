import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

st.set_page_config(page_title="Complete AI Dashboard", layout="wide")

st.title("🔥 Complete Universal AI Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("File Uploaded Successfully ✅")
    st.dataframe(df.head())

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    tab1, tab2, tab3 = st.tabs(["📊 Basic Diagrams", "📈 Advanced Diagrams", "🤖 Models"])

    # ==================================================
    # TAB 1 → BASIC DIAGRAMS (3 ROW ALIGNMENT SMALL)
    # ==================================================
    with tab1:

        st.subheader("📊 Basic Visualization Panel")

        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        row3_col1, row3_col2, row3_col3 = st.columns(3)

        # ================= Row 1 =================
        with row1_col1:
            if categorical_cols:
                st.caption("🥧 Pie Chart")
                col_pie = st.selectbox("Pie Column", categorical_cols, key="pie")
                values = df[col_pie].value_counts()
                fig, ax = plt.subplots(figsize=(3,3))
                ax.pie(values, labels=None, autopct='%1.1f%%')
                st.pyplot(fig)

        with row1_col2:
            if categorical_cols:
                st.caption("🍩 Donut Chart")
                col_donut = st.selectbox("Donut Column", categorical_cols, key="donut")
                values2 = df[col_donut].value_counts()
                fig, ax = plt.subplots(figsize=(3,3))
                ax.pie(values2, labels=None, autopct='%1.1f%%')
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                fig.gca().add_artist(centre_circle)
                st.pyplot(fig)

        with row1_col3:
            if categorical_cols:
                st.caption("🎨 3D Pie Chart")
                col_3d = st.selectbox("3D Pie Column", categorical_cols, key="3d_pie")
                values3d = df[col_3d].value_counts()
                explode = [0.05] * len(values3d)
                fig, ax = plt.subplots(figsize=(3,3))
                ax.pie(values3d, autopct='%1.1f%%', shadow=True, explode=explode)
                st.pyplot(fig)

        # ================= Row 2 =================
        with row2_col1:
            if numeric_cols:
                st.caption("📊 Histogram")
                col_hist = st.selectbox("Histogram Column", numeric_cols, key="hist")
                fig, ax = plt.subplots(figsize=(3,3))
                sns.histplot(df[col_hist], kde=True, ax=ax)
                st.pyplot(fig)

        with row2_col2:
            if numeric_cols:
                st.caption("📦 Box Plot")
                col_box = st.selectbox("Box Column", numeric_cols, key="box")
                fig, ax = plt.subplots(figsize=(3,3))
                sns.boxplot(y=df[col_box], ax=ax)
                st.pyplot(fig)

        with row2_col3:
            if numeric_cols:
                st.caption("📈 Line Chart")
                col_line = st.selectbox("Line Column", numeric_cols, key="line_basic")
                fig, ax = plt.subplots(figsize=(3,3))
                ax.plot(df[col_line])
                st.pyplot(fig)

        # ================= Row 3 =================
        with row3_col1:
            if categorical_cols:
                st.caption("🌀 Circular Bar")
                col_circ = st.selectbox("Circular Column", categorical_cols, key="circular")
                values4 = df[col_circ].value_counts()
                angles = np.linspace(0, 2*np.pi, len(values4), endpoint=False)
                fig = plt.figure(figsize=(3,3))
                ax = fig.add_subplot(111, polar=True)
                ax.bar(angles, values4.values)
                st.pyplot(fig)

        with row3_col2:
            if len(numeric_cols) > 1:
                st.caption("🔥 Heatmap")
                fig, ax = plt.subplots(figsize=(3,3))
                sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

        with row3_col3:
            st.caption("📊 Data Summary")
            st.write(df.describe())

    # ==================================================
    # TAB 2 → ADVANCED (UNCHANGED CLEAN)
    # ==================================================
    with tab2:
        st.subheader("📈 Advanced Analysis")
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.pairplot(df[numeric_cols])
            st.pyplot(fig)

    # ==================================================
    # TAB 3 → MODEL
    # ==================================================
    with tab3:

        st.subheader("🤖 Universal Smart Model")

        target_col = st.selectbox("Select Target Column", df.columns)

        if st.button("Train Model"):

            df_model = df.copy()

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
                    epochs=25,
                    validation_split=0.2,
                    verbose=0
                )

                loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

                st.success(f"Model Accuracy: {accuracy*100:.2f}%")

else:
    st.info("Upload a CSV file to start.")
