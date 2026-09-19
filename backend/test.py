import mlflow

mlflow.set_tracking_uri("sqlite:///D:/fraud-detection/mlflow.db")

exp = mlflow.get_experiment_by_name("fraud_detection_xgboost")
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["metrics.pr_auc DESC"])

top_run_id = runs.iloc[0]["run_id"]
print(f"Top Run ID: {top_run_id}")
print(f"Model URI: runs:/{top_run_id}")