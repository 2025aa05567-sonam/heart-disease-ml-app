import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Heart Disease Classification", layout="centered")

st.title("❤️ Heart Disease Classification App")

# -------------------------------
# Load feature columns safely
# -------------------------------
try:
    feature_columns = joblib.load("model/feature_columns.pkl")
except Exception as e:
    st.error("Error loading feature columns file.")
    st.stop()

# -------------------------------
# Model Names (Lazy Loading)
# -------------------------------
model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # -------------------------------
    # Check target column
    # -------------------------------
    if "HeartDisease" not in df.columns:
        st.error("'HeartDisease' column missing in uploaded file.")
        st.stop()

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # -------------------------------
    # Preprocessing
    # -------------------------------
    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)

    # -------------------------------
    # Model Selection
    # -------------------------------
    selected_model = st.selectbox("Select Model", list(model_files.keys()))

    try:
        model = joblib.load(model_files[selected_model])
    except Exception:
        st.error("Error loading selected model.")
        st.stop()

    # -------------------------------
    # Prediction
    # -------------------------------
    y_pred = model.predict(X)

    # -------------------------------
    # Evaluation Metrics
    # -------------------------------
    st.subheader(" Evaluation Metrics")

    st.write("Accuracy:", round(accuracy_score(y, y_pred), 4))
    st.write("Precision:", round(precision_score(y, y_pred), 4))
    st.write("Recall:", round(recall_score(y, y_pred), 4))
    st.write("F1 Score:", round(f1_score(y, y_pred), 4))
    st.write("MCC:", round(matthews_corrcoef(y, y_pred), 4))

    # AUC (safe handling)
    try:
        y_prob = model.predict_proba(X)[:, 1]
        st.write("AUC:", round(roc_auc_score(y, y_prob), 4))
    except:
        st.write("AUC: Not available for this model")

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.subheader(" Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # -------------------------------
    # Classification Report
    # -------------------------------
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))
