# Heart-Disease-Classifier

## 1. Problem Statement

Given clinical features of a patient, the objective is to predict whether the patient has heart disease (binary classification).

This project focuses on building a reliable classification pipeline with proper validation, model comparison, and hyperparameter tuning.

---

## 2. Dataset

Dataset source:
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

The dataset contains 918 patient records with 11 clinical features, including:

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

Target variable:
- `HeartDisease` (1 = disease, 0 = normal)

Class distribution:
- 508 positive cases
- 410 negative cases

---

## 3. Methodology

### Data Preprocessing

- Train-test split (80/20) using stratification
- Numerical features scaled using StandardScaler
- Categorical features encoded using OneHotEncoder (drop="first")
- Preprocessing and modeling combined using sklearn Pipeline to prevent data leakage

---

### Model Selection

The following models were evaluated using 5-fold Stratified Cross-Validation:

- Logistic Regression
- Linear SVC
- K-Nearest Neighbors
- Random Forest Classifier

Evaluation metrics:
- Accuracy
- Recall
- Precision
- F1-score

---

### Hyperparameter Tuning

The top-performing models were further tuned using GridSearchCV:

- RandomForestClassifier
- LogisticRegression

Tuning was performed using 5-fold Stratified Cross-Validation with F1-score as the optimization metric.

---

## 4. Final Model

Logistic Regression was selected as the final model based on:

- Highest recall (0.93)
- Competitive precision (0.87)
- Best precision-recall AUC (~0.94)
- Simpler and more interpretable structure compared to Random Forest

Test set performance:

| Metric    | Score |
|-----------|--------|
| Accuracy  | 0.89   |
| Recall    | 0.93   |
| Precision | 0.87   |
| F1-score  | 0.90   |

---

## 5. Key Learnings

- Importance of stratified cross-validation in classification problems
- Preventing data leakage using sklearn Pipelines
- Trade-offs between recall and precision in medical classification
- Hyperparameter tuning does not always lead to large improvements
- Simpler models can generalize well on structured tabular data

---

## 6. Future Improvements

- Threshold optimization for improved recall/precision trade-off
- Feature importance and coefficient interpretation
- Deployment as a lightweight web application
- Testing on external validation dataset

---

## 7. Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
