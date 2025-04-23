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

## 📊 Project Steps

### 🔹 Step 1: Data Preprocessing
- Dropped unnecessary columns: `trans_num`, `cc_num`, `first`, `last`, `merchant`, etc.
- Engineered geospatial feature using Haversine distance
- One-hot encoded categorical variables like `gender`, `category`
- Scaled numerical features using `StandardScaler`

### 🔹 Step 2: Class Imbalance Handling
- Calculated `scale_pos_weight` to balance rare fraud class
- Ensured the model does not ignore minority class (frauds)

### 🔹 Step 3: Model Training
- Trained `XGBoost` classifier using `GridSearchCV` for hyperparameter tuning
- Trained additional models: `LogisticRegression`, `RandomForestClassifier`

### 🔹 Step 4: Ensemble with Voting Classifier
- Combined all three models into a `VotingClassifier` using soft voting
- Averaged prediction probabilities to make final decisions

### 🔹 Step 5: Threshold Optimization
- Used precision-recall curve to identify the optimal threshold
- Converted probabilities to binary predictions using this threshold to minimize false positives

### 🔹 Step 6: Evaluation
- Evaluated model using:
  - Confusion matrix
  - Classification report (precision, recall, F1-score)
  - ROC-AUC score

---

## 🧠 Why Use a Voting Classifier?

Fraud detection is complex — no single model works best in all situations.  
This ensemble combines:
- **XGBoost** (powerful tabular learner)
- **Random Forest** (robust to overfitting)
- **Logistic Regression** (strong baseline)

Together, these provide **more stable and accurate predictions**.

---
