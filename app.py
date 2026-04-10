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
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI ML Deep Learning Dashboard", 
    layout="wide",
    page_icon="🤖"
)

colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

sns.set_theme(style="whitegrid", palette="husl")

# -------------------------
# FIXED PREPROCESS FUNCTION 🔥
# -------------------------
def preprocess_data(data, target_column, feature_columns):
    try:
        X = data[feature_columns].copy()
        y = data[target_column].copy()

        # ✅ Remove ID-like columns
        for col in X.columns:
            if X[col].nunique() == len(X):
                X.drop(columns=[col], inplace=True)

        # ✅ Handle categorical & mixed data
        le_dict = {}
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    if X[col].isnull().mean() > 0.5:
                        raise Exception()
                except:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    le_dict[col] = le

        # ✅ Fill missing
        X = X.fillna(X.mean(numeric_only=True))

        # ✅ Encode target
        le_y = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))

        # ✅ Final safety
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X, y, X_scaled, le_y, scaler

    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None, None, None, None, None

# -------------------------
# LOAD DATA
# -------------------------
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file), None
    except Exception as e:
        return None, str(e)

# -------------------------
# MODEL TRAINING
# -------------------------
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models

# -------------------------
# MAIN APP
# -------------------------
def main():
    st.title("🤖 AI + ML Dashboard")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        data, err = load_data(uploaded_file)

        if err:
            st.error(err)
            return

        st.dataframe(data.head())

        target = st.selectbox("Select Target", data.columns)
        features = st.multiselect(
            "Select Features",
            [c for c in data.columns if c != target],
            default=[c for c in data.columns if c != target]
        )

        if not features:
            st.warning("Select features")
            return

        X, y, X_scaled, le_y, scaler = preprocess_data(data, target, features)

        if X is None:
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        models = train_models(X_train, y_train)

        st.subheader("📊 Model Results")

        for name, model in models.items():
            pred = model.predict(X_test)
            acc = (pred == y_test).mean()
            st.write(f"{name} Accuracy: {acc:.3f}")

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    main()
