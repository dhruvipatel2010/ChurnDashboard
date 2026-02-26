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

st.set_page_config(page_title="Advanced Universal AI Dashboard", layout="wide")

st.title("🔥 Advanced Universal AI Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("File Uploaded Successfully ✅")
    st.dataframe(df.head())

    st.write("Dataset Shape:", df.shape)

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    st.subheader("📌 Column Types")
    st.write("Categorical:", categorical_cols)
    st.write("Numeric:", numeric_cols)

    # =========================
    # 🥧 PIE CHART
    # =========================
    st.subheader("🥧 Pie Chart")

    if len(categorical_cols) > 0:
        col1 = st.selectbox("Select Column", categorical_cols, key="pie")
        values = df[col1].value_counts()

        fig1, ax1 = plt.subplots()
        ax1.pie(values, labels=values.index, autopct='%1.1f%%')
        ax1.set_title(f"Distribution of {col1}")
        st.pyplot(fig1)

    # =========================
    # 🍩 DONUT CHART
    # =========================
    st.subheader("🍩 Donut Chart")

    if len(categorical_cols) > 0:
        col2 = st.selectbox("Select Column", categorical_cols, key="donut")
        values2 = df[col2].value_counts()

        fig2, ax2 = plt.subplots()
        ax2.pie(values2, labels=values2.index, autopct='%1.1f%%')
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig2.gca().add_artist(centre_circle)
        ax2.set_title(f"Donut Chart of {col2}")
        st.pyplot(fig2)

    # =========================
    # 📊 HISTOGRAM
    # =========================
    st.subheader("📊 Histogram")

    if len(numeric_cols) > 0:
        col3 = st.selectbox("Select Numeric Column", numeric_cols, key="hist")
        fig3, ax3 = plt.subplots()
        sns.histplot(df[col3], kde=True, ax=ax3)
        st.pyplot(fig3)

    # =========================
    # 🔥 CORRELATION HEATMAP
    # =========================
    st.subheader("🔥 Correlation Heatmap")

    if len(numeric_cols) > 1:
        fig4, ax4 = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

    # =========================
    # 📦 BOX PLOT
    # =========================
    st.subheader("📦 Box Plot")

    if len(numeric_cols) > 0:
        col4 = st.selectbox("Select Column", numeric_cols, key="box")
        fig5, ax5 = plt.subplots()
        sns.boxplot(y=df[col4], ax=ax5)
        st.pyplot(fig5)

    # =========================
    # 📈 LINE CHART
    # =========================
    st.subheader("📈 Line Chart")

    if len(numeric_cols) > 0:
        col5 = st.selectbox("Select Column", numeric_cols, key="line")
        st.line_chart(df[col5])

    # =========================
    # 🌀 CIRCULAR BAR PLOT
    # =========================
    st.subheader("🌀 Circular Bar Plot")

    if len(categorical_cols) > 0:
        col6 = st.selectbox("Select Column", categorical_cols, key="circular")
        values3 = df[col6].value_counts()

        angles = np.linspace(0, 2*np.pi, len(values3), endpoint=False)

        fig6 = plt.figure()
        ax6 = fig6.add_subplot(111, polar=True)

        ax6.bar(angles, values3.values)
        ax6.set_xticks(angles)
        ax6.set_xticklabels(values3.index)

        st.pyplot(fig6)

    # =========================
    # 🤖 MACHINE LEARNING MODEL (Random Forest)
    # =========================
    st.subheader("🤖 Random Forest Model")

    target_col = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Random Forest"):

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X = pd.get_dummies(X)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_rf = RandomForestClassifier()
        model_rf.fit(X_train, y_train)

        predictions = model_rf.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        st.success(f"Random Forest Accuracy: {acc*100:.2f}%")

        cm = confusion_matrix(y_test, predictions)
        fig7, ax7 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax7)
        st.pyplot(fig7)

    # =========================
    # 🧠 ADVANCED DEEP LEARNING MODEL
    # =========================
    st.subheader("🧠 Advanced Deep Learning Model")

    if st.button("Train Deep Learning Model"):

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X = pd.get_dummies(X)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = Sequential([
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation='relu'),
            Dropout(0.2),

            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        early_stop = EarlyStopping(patience=5, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            epochs=50,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        st.success(f"Deep Learning Accuracy: {accuracy*100:.2f}%")

else:
    st.info("Please upload a CSV file to begin.")
