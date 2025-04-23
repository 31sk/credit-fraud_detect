# 💳 Fraud Detection Using Machine Learning

This project is a comprehensive machine learning pipeline for detecting fraudulent credit card transactions. It uses an ensemble of models to achieve high accuracy while minimizing false positives — which is critical in fraud detection.

---

## 📁 Dataset

The dataset used is publicly available on Kaggle:  
🔗 [Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

It includes fields like:
- User and merchant locations (lat/lon)
- Transaction date and time
- Amount, category, gender, job, etc.
- `is_fraud` label indicating fraud (1) or not (0)

---

## 🎯 Project Goals

- Detect fraudulent transactions with high precision
- Reduce false positives using optimal thresholding
- Leverage **geospatial** and **temporal** features
- Use **ensemble learning** for better generalization

---

## 🧪 Techniques Used

- **Data Preprocessing**
  - Removed high-cardinality and non-informative features
  - Extracted datetime features (hour, day of week, month)
  - Engineered geospatial feature (Haversine distance between user and merchant)
  - Handled categorical features with one-hot encoding

- **Class Imbalance Handling**
  - Calculated `scale_pos_weight` for rare class weighting

- **Modeling**
  - Trained and optimized XGBoost using GridSearchCV
  - Combined XGBoost, Logistic Regression, and Random Forest in a `VotingClassifier`
  - Applied soft voting to improve performance and reduce variance

- **Threshold Tuning**
  - Used Precision-Recall curve to find the best decision threshold
  - Tuned to minimize false positives while maintaining high recall

- **Evaluation**
  - Classification Report (Precision, Recall, F1-Score)
  - Confusion Matrix
  - ROC-AUC Score

---

## 🧠 Why Use a Voting Classifier?

Fraud detection is complex — no single model works best in all situations.  
This ensemble combines:
- **XGBoost** (powerful tabular learner)
- **Random Forest** (robust to overfitting)
- **Logistic Regression** (strong baseline)

Together, these provide **more stable and accurate predictions**.

---

## 📊 Results

- Precision and recall were tuned to reduce false positives
- Ensemble outperformed individual models
- Geospatial and time-based features added clear predictive value

---
