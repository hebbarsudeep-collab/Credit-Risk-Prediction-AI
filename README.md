# 🏦 Credit Risk Prediction AI: Optimizing for Loan Defaults

## 📌 Project Overview
The goal of this project is to predict whether a borrower will default on a personal loan. Instead of just chasing a high overall accuracy score, this project focuses on the **Precision-Recall Trade-off**, specifically tuning the model to minimize **False Negatives** (costly missed defaults) while maintaining a healthy approval rate for safe customers.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, XGBoost, Joblib, Imbalanced-Learn (SMOTE)
* **Algorithms:** Random Forest, XGBoost Classifier

## 📊 The Data Pipeline
1. **Data Cleaning:** Handled missing values via median imputation to preserve critical financial context.
2. **Feature Engineering:** Used `pd.get_dummies` for One-Hot Encoding categorical text variables (`drop_first=True` to eliminate multicollinearity).
3. **Scaling:** Applied `RobustScaler` to handle extreme financial outliers (e.g., massive income disparities).
4. **Data Splitting:** Utilized a stratified 80/20 train-test split (`stratify=y`) to maintain exact class proportions and avoid data leakage.

## 🧠 Model Evolution & Business Impact

**Model 1: The Baseline (Random Forest)**
* *Overall Accuracy:* 92%
* *Class 1 Recall:* 72% (Missed 520 defaults)
* *Verdict:* Too naive. Missed too many high-risk loans, costing the hypothetical bank millions.

**Model 2: The Paranoid Model (Weighted Random Forest)**
* *Overall Accuracy:* 84%
* *Class 1 Recall:* 77% (Missed 417 defaults)
* *Verdict:* Caught 103 more bad loans (saving ~$1M in losses), but precision dropped to 60%. Rejected too many safe customers (False Positives).

**Model 3: The Champion (XGBoost Classifier)**
* *Overall Accuracy:* 88%
* *Class 1 Recall:* 78% (Missed 399 defaults)
* *Class 1 Precision:* 71%
* *Verdict:* **The Goldilocks Zone.** Using XGBoost with `scale_pos_weight=3.5`, the model learned from its mistakes. It caught the highest number of defaults (highest recall) while simultaneously recovering precision (rejecting fewer good customers). 

## 🚀 How to Run
The final trained AI is saved as `credit_risk_xgboost_model.pkl`. It can be easily loaded into any Python application using `joblib` for instant credit risk predictions on new loan applications.
