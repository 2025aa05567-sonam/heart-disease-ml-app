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

st.title("Heart Disease Classification App")

feature_columns = joblib.load("model/feature_columns.pkl")

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    if "HeartDisease" not in df.columns:
        st.error("HeartDisease column missing")
        st.stop()

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)

    selected_model = st.selectbox("Select Model", list(models.keys()))
    model = models[selected_model]

    y_pred = model.predict(X)

    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("Precision:", precision_score(y, y_pred))
    st.write("Recall:", recall_score(y, y_pred))
    st.write("F1 Score:", f1_score(y, y_pred))
    st.write("MCC:", matthews_corrcoef(y, y_pred))
    st.write("AUC:", roc_auc_score(y, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))