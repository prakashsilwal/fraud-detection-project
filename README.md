# Credit Card Fraud Detection Using Machine Learning

## Overview
This project focuses on detecting fraudulent credit card transactions using advanced machine learning techniques. The dataset consists of over 1.8 million real-world transactions, with a highly imbalanced distribution where only ~0.4% are fraudulent. The objective is to accurately detect fraud while minimizing false positives.

## Problem Statement
Fraudulent transactions cost financial institutions billions annually. The task is to build a model that can identify these rare cases with high recall and precision. Given the severe class imbalance, standard modeling techniques may fail to detect meaningful fraud patterns without careful data preparation.

## Dataset Highlights
- Total Records: ~1.8 million
- Fraud Rate: ~0.4%
- Features: Transaction amount, merchant, category, location, timestamp, and personal identifiers

## Key Steps

### 1. Data Cleaning & Preprocessing
- Dropped uninformative columns and handled missing values
- Applied `log1p` transformation to normalize skewed `amt`
- Extracted time features: `hour`, `day`, `weekday`, `is_night`
- Engineered binary features: `is_online`, `is_high_amt`
- Binned transaction amounts into `amt_bin`

### 2. Handling Imbalanced Data
- Used SMOTE (Synthetic Minority Oversampling Technique) to balance class distribution

### 3. Exploratory Data Analysis (EDA)
- Fraud most likely during late-night hours (10 PM–1 AM)
- Categories like `shopping_net` and `misc_net` had highest fraud rates
- Fraudulent transactions had significantly higher monetary values

### 4. Modeling
**Models Trained:**
- Logistic Regression
- Random Forest
- XGBoost

**Best Performing Model: XGBoost**
- Recall: 92%
- ROC-AUC: 0.9975
- Precision: 44%
- F1-Score: 0.59

### 5. Feature Importance (Random Forest & XGBoost)
Top features included:
- `amt_bin`, `log_amt`, `category`, `hour`, `is_night`, `is_online`

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Curve
- Confusion Matrix

## Business Impact
- Proactively identifies high-risk transactions
- Reduces financial loss due to undetected fraud
- Enhances customer trust through better security mechanisms

## Technologies Used
- Python
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn, XGBoost, imbalanced-learn

## Project Status
✅ Completed and evaluated. Ready for deployment or integration with real-time systems.

## Author
Prakash Silwal— Data Scientist with 10+ years of experience in anomaly detection, machine learning, and financial analytics.
