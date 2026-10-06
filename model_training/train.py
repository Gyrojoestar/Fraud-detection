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
    if isinstance(data_path, pd.DataFrame):
        df = data_path.copy()
    else:
        data_path = Path(data_path)
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
            df['is_fraud'] = y_dummy

    # strip duplicates across all combined parquet files before training
    if 'transaction_id' in df.columns:
        df = df.drop_duplicates(subset=['transaction_id'])
    else:
        df = df.drop_duplicates()
    # convert all column names to lowercase
    df.columns = [str(col).lower() for col in df.columns]
    
    # extract target variable
    if 'is_fraud' in df.columns:
        y = df['is_fraud'].reset_index(drop=True)
    else:
        raise KeyError("Target column 'is_fraud' not found in dataset.")

    # drop non-feature metadata columns
    cols_to_drop = [
        'is_fraud',
        'time', 'created_at', 
        'transaction_id', '_sa_instance_state'
    ]
    X = df.drop(columns=cols_to_drop, errors='ignore').reset_index(drop=True)

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
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
    # ensure lowercase columns when calculating fraud class balance
    df.columns = [str(col).lower() for col in df.columns]
    # account for class imbalance
    scale_pos_weight = (df['is_fraud'] == 0).sum() / max((df['is_fraud'] == 1).sum(), 1)
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