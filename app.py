import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

# -------------------------
# PREPROCESSING (FIXED FOR ANY CSV)
# -------------------------
def preprocess_data(data, target_column, feature_columns):
    try:
        X = data[feature_columns].copy()
        y = data[target_column].copy()

        # ✅ Remove ID-like columns automatically
        X = X[[col for col in X.columns if 'id' not in col.lower()]]

        # ✅ Handle missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna("Unknown")
            else:
                X[col] = X[col].fillna(X[col].median())

        # ✅ Convert categorical → numeric
        X = pd.get_dummies(X, drop_first=True)

        # ✅ Ensure numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # ✅ Encode target
        le_y = None
        if y.dtype == 'object':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))

        # ✅ Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X, y, X_scaled, le_y, scaler

    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None, None, None, None, None


# -------------------------
# ML MODELS
# -------------------------
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    for m in models.values():
        m.fit(X_train, y_train)

    return models


def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = (y_pred == y_test).mean()

        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr, tpr)
        except:
            y_prob, auc_score = None, None

        results[name] = {
            "accuracy": acc,
            "auc": auc_score,
            "predictions": y_pred,
            "probabilities": y_prob
        }

    return results


# -------------------------
# DEEP LEARNING (FIXED)
# -------------------------
def create_dl_model(hidden_layers):
    return MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        activation='relu',
        solver='adam',
        max_iter=300,
        early_stopping=True,
        random_state=42
    )


# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(layout="wide")
st.title("🤖 AI ML + Deep Learning Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(data.head())

    target = st.selectbox("Select Target", data.columns)

    features = st.multiselect(
        "Select Features",
        [col for col in data.columns if col != target],
        default=[col for col in data.columns if col != target]
    )

    X, y, X_scaled, le_y, scaler = preprocess_data(data, target, features)

    if X is not None:

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # ML
        models = train_models(X_train, y_train)
        results = evaluate_models(models, X_test, y_test)

        st.write("## ML Results")
        for name, res in results.items():
            st.write(f"{name}: Accuracy = {res['accuracy']:.4f}")

        # -------------------------
        # DEEP LEARNING
        # -------------------------
        st.write("## 🧠 Deep Learning")

        hidden_layers = st.multiselect(
            "Hidden Layers",
            [8, 16, 32, 64, 128],
            default=[64, 32]
        )

        if st.button("Train Deep Learning Model"):

            dl_model = create_dl_model(hidden_layers)
            dl_model.fit(X_train, y_train)

            y_pred = dl_model.predict(X_test)
            acc = (y_pred == y_test).mean()

            st.success(f"Deep Learning Accuracy: {acc:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_title("Confusion Matrix (DL)")
            st.pyplot(fig)

else:
    st.info("Upload a CSV file to start")
