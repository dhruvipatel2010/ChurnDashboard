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

    # ======================================
    # TAB 1 → BASIC DIAGRAMS (3 PER ROW)
    # ======================================
    with tab1:

        # -------- ROW 1 --------
        col1, col2, col3 = st.columns(3)

        # PIE
        with col1:
            if categorical_cols:
                st.subheader("🥧 Pie Chart")
                col_pie = st.selectbox("Pie Column", categorical_cols, key="pie")
                values = df[col_pie].value_counts()
                fig1, ax1 = plt.subplots(figsize=(3,3))
                ax1.pie(values, labels=values.index, autopct='%1.1f%%')
                st.pyplot(fig1)

        # DONUT
        with col2:
            if categorical_cols:
                st.subheader("🍩 Donut Chart")
                col_donut = st.selectbox("Donut Column", categorical_cols, key="donut")
                values2 = df[col_donut].value_counts()
                fig2, ax2 = plt.subplots(figsize=(3,3))
                ax2.pie(values2, labels=values2.index, autopct='%1.1f%%')
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                fig2.gca().add_artist(centre_circle)
                st.pyplot(fig2)

        # 3D PIE
        with col3:
            if categorical_cols:
                st.subheader("🎨 3D Pie")
                col_3d = st.selectbox("3D Pie Column", categorical_cols, key="3d")
                values3d = df[col_3d].value_counts()
                explode = [0.05] * len(values3d)
                fig3d, ax3d = plt.subplots(figsize=(3,3))
                ax3d.pie(values3d, labels=values3d.index,
                         autopct='%1.1f%%', shadow=True, explode=explode)
                st.pyplot(fig3d)

        # -------- ROW 2 --------
        col4, col5, col6 = st.columns(3)

        # HISTOGRAM
        with col4:
            if numeric_cols:
                st.subheader("📊 Histogram")
                col_hist = st.selectbox("Histogram Column", numeric_cols, key="hist")
                fig4, ax4 = plt.subplots(figsize=(3,3))
                sns.histplot(df[col_hist], kde=True, ax=ax4)
                st.pyplot(fig4)

        # BOX
        with col5:
            if numeric_cols:
                st.subheader("📦 Box Plot")
                col_box = st.selectbox("Box Column", numeric_cols, key="box")
                fig5, ax5 = plt.subplots(figsize=(3,3))
                sns.boxplot(y=df[col_box], ax=ax5)
                st.pyplot(fig5)

        # LINE
        with col6:
            if numeric_cols:
                st.subheader("📈 Line Chart")
                col_line = st.selectbox("Line Column", numeric_cols, key="line")
                fig6, ax6 = plt.subplots(figsize=(3,3))
                ax6.plot(df[col_line])
                st.pyplot(fig6)

        # -------- ROW 3 --------
        col7, col8, col9 = st.columns(3)

        # CIRCULAR BAR
        with col7:
            if categorical_cols:
                st.subheader("🌀 Circular Bar")
                col_circ = st.selectbox("Circular Column", categorical_cols, key="circular")
                values4 = df[col_circ].value_counts()
                angles = np.linspace(0, 2*np.pi, len(values4), endpoint=False)
                fig7 = plt.figure(figsize=(3,3))
                ax7 = fig7.add_subplot(111, polar=True)
                ax7.bar(angles, values4.values)
                ax7.set_xticks(angles)
                ax7.set_xticklabels(values4.index)
                st.pyplot(fig7)

        # HEATMAP
        with col8:
            if len(numeric_cols) > 1:
                st.subheader("🔥 Heatmap")
                fig8, ax8 = plt.subplots(figsize=(3,3))
                sns.heatmap(df[numeric_cols].corr(),
                            annot=True, cmap="coolwarm", ax=ax8)
                st.pyplot(fig8)

        # EMPTY SPACE (kept for alignment)
        with col9:
            st.write("")

    # ======================================
    # TAB 2 → ADVANCED DIAGRAMS (UNCHANGED)
    # ======================================
    with tab2:

        col10, col11 = st.columns(2)

        with col10:
            st.subheader("📈 Line Chart (Streamlit)")
            if numeric_cols:
                col_line2 = st.selectbox("Line Column", numeric_cols, key="line2")
                st.line_chart(df[col_line2])

        with col11:
            st.subheader("🌀 Circular Bar Plot")
            if categorical_cols:
                col_circ2 = st.selectbox("Circular Column", categorical_cols, key="circular2")
                values5 = df[col_circ2].value_counts()
                angles = np.linspace(0, 2*np.pi, len(values5), endpoint=False)
                fig9 = plt.figure(figsize=(3,3))
                ax9 = fig9.add_subplot(111, polar=True)
                ax9.bar(angles, values5.values)
                ax9.set_xticks(angles)
                ax9.set_xticklabels(values5.index)
                st.pyplot(fig9)

    # ======================================
    # TAB 3 → MODEL (UNCHANGED)
    # ======================================
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
                    epochs=30,
                    validation_split=0.2,
                    verbose=0
                )

                loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

                st.success(f"Model Accuracy: {accuracy*100:.2f}%")

else:
    st.info("Upload a CSV file to start.")
