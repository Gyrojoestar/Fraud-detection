import joblib
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import mlflow
import mlflow.xgboost
from datetime import datetime
import os

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "sqlite:///D:/fraud-detection/mlflow.db")
)
mlflow.set_experiment("fraud_detection_xgboost")

# STEP 1: Load Kaggle Data
# TODO: Use pd.read_csv() to load 'creditcard.csv'.
df = pd.read_csv("creditcard.csv")
# TODO: Separate the target variable. The Kaggle dataset uses 'Class' as the target (1 = fraud, 0 = legit).
X = df.drop(columns=['Class', 'Time']).reset_index(drop=True)
y = df['Class'].reset_index(drop=True)

# STEP 2: Train/Test Split
# TODO: Split X and y into train and test sets (80/20 ratio).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Hint: Use stratify=y to maintain the extreme class imbalance in both sets.

# STEP 3: Handle Class Imbalance & Train Model
# TODO: Calculate 'scale_pos_weight' for XGBoost (number of negative class / number of positive class).

scale_pos_weight = (df['Class']==0).sum()/(df['Class']==1).sum()
# TODO: Instantiate and fit XGBClassifier using X_train and y_train.
params = {
    "n_estimators":100, 
    "max_depth":3,
    "learning_rate":0.1,
    "scale_pos_weight":scale_pos_weight,
    "objective":'binary:logistic'
}

with mlflow.start_run(run_name="xgboost"):
    mlflow.log_params(params)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_prob = pd.DataFrame(y_prob)
    print(y_prob.head(10))
    pr_auc  = float(average_precision_score(y_test, y_prob))
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.xgboost.log_model(model, name="model")
    
print("Training & MLflow logging complete.")
