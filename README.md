# Heart Disease Classification using Machine Learning

## Problem Statement

The objective of this project is to predict whether a patient has heart disease based on clinical and medical attributes. This is formulated as a **binary classification problem**, where:

- 1 → Presence of Heart Disease  
- 0 → Absence of Heart Disease  

The goal is to compare multiple machine learning classification models and evaluate their performance using various metrics.

---

## Dataset Description

The dataset used is the **Heart Failure Prediction Dataset**.

- Total Instances: 918  
- Original Features: 11  
- Features after Encoding: > 12 (Requirement satisfied)  
- Target Column: `HeartDisease`  
- Missing Values: None  

The dataset contains both numerical and categorical features such as:

- Age  
- Sex  
- ChestPainType  
- RestingBP  
- Cholesterol  
- FastingBS  
- RestingECG  
- MaxHR  
- ExerciseAngina  
- Oldpeak  
- ST_Slope  

Categorical features were encoded using **One-Hot Encoding** before training the models.

---

## Machine Learning Models Implemented

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

---

## Evaluation Metrics Used

Each model was evaluated using the following metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|----------|------|-----------|--------|-----------|------|
| Logistic Regression | 0.853 | 0.853 | 0.86 | 0.88 | 0.869 | 0.704 |
| Decision Tree | 0.826 | 0.826 | 0.83 | 0.87 | 0.849 | 0.644 |
| KNN | 0.706 | 0.706 | 0.72 | 0.75 | 0.735 | 0.410 |
| Naive Bayes | 0.859 | 0.859 | 0.87 | 0.88 | 0.874 | 0.716 |
| Random Forest | 0.875 | 0.875 | 0.89 | 0.90 | 0.892 | 0.744 |
| XGBoost | 0.869 | 0.869 | 0.88 | 0.89 | 0.887 | 0.733 |

---

## Model Performance Observations

### Logistic Regression
Logistic Regression performed well due to linear relationships among several medical features. It provided stable and interpretable results.

### Decision Tree
Decision Tree showed moderate performance but slightly lower accuracy compared to ensemble methods. This may be due to overfitting caused by single-tree structure.

### KNN
KNN achieved the lowest performance. As a distance-based model, it is sensitive to high-dimensional data after encoding categorical variables.

### Naive Bayes
Naive Bayes performed strongly despite its independence assumption. It served as a solid probabilistic baseline model.

### Random Forest (Best Performing Model)
Random Forest achieved the highest Accuracy, F1 Score, and MCC. Being an ensemble of multiple decision trees, it effectively captured non-linear feature interactions and reduced overfitting.

### XGBoost
XGBoost also performed strongly and close to Random Forest. As a gradient boosting model, it sequentially improves prediction errors. Slightly lower performance may be due to default hyperparameters.

### Overall Conclusion
Ensemble models (Random Forest and XGBoost) outperformed individual classifiers, demonstrating the effectiveness of combining multiple learners for medical classification problems.

---

## Streamlit Application Features

The deployed Streamlit application includes:

- CSV dataset upload option  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix visualization  
- Classification report output  

---

## Project Structure

```
Assignment-2/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
    ├── model_training.ipynb
    ├── feature_columns.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
```

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

GitHub repository contains complete source code, saved models, and dependencies as required in the assignment.