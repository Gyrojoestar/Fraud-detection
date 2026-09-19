import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import mlflow
import mlflow.xgboost
from datetime import datetime
from sklearn.datasets import make_classification
import numpy as np
import os

data_path = Path("creditcard.csv")
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "sqlite:///D:/fraud-detection/mlflow.db")
)
mlflow.set_experiment("fraud_detection_xgboost")

# STEP 1: Load Kaggle Data
# TODO: Use pd.read_csv() to load 'creditcard.csv'.
if data_path.exists():
    print("Loading real dataset...")
    df = pd.read_csv(data_path)
else:
    print("Dataset CSV not found (CI Environment). Generating synthetic fallback data...")
    # Generate dummy data matching credit card dataset structure
    X_dummy, y_dummy = make_classification(
        n_samples=200, 
        n_features=28, 
        random_state=42
    )
    df = pd.DataFrame(X_dummy, columns=[f"V{i}" for i in range(1, 29)])
    df['Time'] = np.random.randint(0, 1000, size=200)
    df['Amount'] = np.random.uniform(1.0, 500.0, size=200)
    df['Class'] = y_dummy

# Now this drop operation is 100% safe in both Local and CI environments!
X = df.drop(columns=['Class', 'Time']).reset_index(drop=True)
y = df['Class']

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
