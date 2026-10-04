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
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("fraud_detection_xgboost")
MODEL_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "model.ubj"

def data_preprocessing(data_path, test_size=0.2, random_state=42):
    # load Kaggle Data
    if data_path.exists():
        print("Loading real dataset...")
        df = pd.read_csv(data_path)
    else:
        print("Dataset CSV not found (CI Environment). Generating synthetic fallback data...")
        # generate dummy data matching credit card dataset structure for ci workflow testing
        X_dummy, y_dummy = make_classification(
            n_samples=200, 
            n_features=28, 
            random_state=random_state
        )
        df = pd.DataFrame(X_dummy, columns=[f"V{i}" for i in range(1, 29)])
        df['Time'] = np.random.randint(0, 1000, size=200)
        df['Amount'] = np.random.uniform(1.0, 500.0, size=200)
        df['Class'] = y_dummy

    # remove time column as it's redundant (time = time lapsed after first transaction)
    # no indication of time of day

    X = df.drop(columns=['Class', 'Time']).reset_index(drop=True)
    y = df['Class'].reset_index(drop=True)

    # train test split 80/20 with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test, df

def export_best_model():
    experiment = mlflow.get_experiment_by_name("fraud_detection_xgboost")
    if experiment is None:
        raise RuntimeError("MLflow experiment not found.")

    best_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.pr_auc DESC"],
        max_results=1,
    )
    if best_runs.empty:
        raise RuntimeError("No MLflow runs with a PR-AUC score were found.")

    best_run = best_runs.iloc[0]
    best_model = mlflow.xgboost.load_model(f"runs:/{best_run['run_id']}/model")
    best_model.save_model(str(MODEL_OUTPUT_PATH))
    print(
        f"Exported best model (PR-AUC={best_run['metrics.pr_auc']:.6f}) "
        f"to {MODEL_OUTPUT_PATH}"
    )

def train_model(X_train, X_test, y_train, y_test, df):
    # handle use scale pos weight to address class imbalance in the dataset
    scale_pos_weight = (df['Class']==0).sum()/(df['Class']==1).sum()
    # define model parameters for XGBoost classifier
    params = {
        "n_estimators":100, 
        "max_depth":3,
        "learning_rate":0.1,
        "scale_pos_weight":scale_pos_weight,
        "objective":'binary:logistic'
    }

    # all training and evaluation logged to MLflow for experiment tracking
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
        
    export_best_model()
    print("Training & MLflow logging complete.")

if __name__ == "__main__":
    # execute the data preprocessing and model training steps
    train_model(*data_preprocessing(data_path))   