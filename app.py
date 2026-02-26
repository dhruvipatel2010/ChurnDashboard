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

st.set_page_config(page_title="Ultimate AI Dashboard", layout="wide")

st.title("🔥 Ultimate Universal AI Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("File Uploaded Successfully ✅")
    st.dataframe(df.head())

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    st.write("Categorical:", categorical_cols)
    st.write("Numeric:", numeric_cols)

    # =============================
    # ROW 1 → PIE + DONUT
    # =============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥧 Pie Chart")
        if len(categorical_cols) > 0:
            col_pie = st.selectbox("Pie Column", categorical_cols, key="pie")
            values = df[col_pie].value_counts()
            fig1, ax1 = plt.subplots(figsize=(5,5))
            ax1.pie(values, labels=values.index, autopct='%1.1f%%')
            st.pyplot(fig1)

    with col2:
        st.subheader("🍩 Donut Chart")
        if len(categorical_cols) > 0:
            col_donut = st.selectbox("Donut Column", categorical_cols, key="donut")
            values2 = df[col_donut].value_counts()
            fig2, ax2 = plt.subplots(figsize=(5,5))
            ax2.pie(values2, labels=values2.index, autopct='%1.1f%%')
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig2.gca().add_artist(centre_circle)
            st.pyplot(fig2)

    # =============================
    # ROW 2 → HISTOGRAM + BOX
    # =============================
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📊 Histogram")
        if len(numeric_cols) > 0:
            col_hist = st.selectbox("Histogram Column", numeric_cols, key="hist")
            fig3, ax3 = plt.subplots(figsize=(6,4))
            sns.histplot(df[col_hist], kde=True, ax=ax3)
            st.pyplot(fig3)

    with col4:
        st.subheader("📦 Box Plot")
        if len(numeric_cols) > 0:
            col_box = st.selectbox("Box Column", numeric_cols, key="box")
            fig4, ax4 = plt.subplots(figsize=(6,4))
            sns.boxplot(y=df[col_box], ax=ax4)
            st.pyplot(fig4)

    # =============================
    # ROW 3 → LINE + CIRCULAR BAR
    # =============================
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("📈 Line Chart")
        if len(numeric_cols) > 0:
            col_line = st.selectbox("Line Column", numeric_cols, key="line")
            st.line_chart(df[col_line])

    with col6:
        st.subheader("🌀 Circular Bar Plot")
        if len(categorical_cols) > 0:
            col_circ = st.selectbox("Circular Column", categorical_cols, key="circular")
            values3 = df[col_circ].value_counts()
            angles = np.linspace(0, 2*np.pi, len(values3), endpoint=False)
            fig6 = plt.figure(figsize=(5,5))
            ax6 = fig6.add_subplot(111, polar=True)
            ax6.bar(angles, values3.values)
            ax6.set_xticks(angles)
            ax6.set_xticklabels(values3.index)
            st.pyplot(fig6)

    # =============================
    # HEATMAP (FULL WIDTH)
    # =============================
    st.subheader("🔥 Correlation Heatmap")
    if len(numeric_cols) > 1:
        fig7, ax7 = plt.subplots(figsize=(6,5))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax7)
        st.pyplot(fig7)

    # =============================
    # MACHINE LEARNING
    # =============================
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

        preds = model_rf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.success(f"Random Forest Accuracy: {acc*100:.2f}%")

        cm = confusion_matrix(y_test, preds)
        fig8, ax8 = plt.subplots(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax8)
        st.pyplot(fig8)

    # =============================
    # DEEP LEARNING
    # =============================
    st.subheader("🧠 Deep Learning Model")

    if st.button("Train Deep Learning"):

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

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        early_stop = EarlyStopping(patience=5, restore_best_weights=True)

        model.fit(X_train, y_train,
                  epochs=50,
                  validation_split=0.2,
                  callbacks=[early_stop],
                  verbose=0)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        st.success(f"Deep Learning Accuracy: {accuracy*100:.2f}%")

else:
    st.info("Please upload a CSV file to begin.")
