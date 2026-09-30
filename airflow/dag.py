import os
import pandas as pd
from sklearn.model_selection import train_test_split

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import pendulum
from datetime import timedelta
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

import backend.database.conn as dblib
from model_training import train
from model_training.train import train_model


# 1. Define default arguments applied to all tasks and global variables for the DAG
default_args = {
    'owner': 'AndrewJWY',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['andrewjwy@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

LOCAL_LAKE_PATH = "./data/parquet_lake"

# 2. Instantiate the DAG using the @dag decorator
@dag(
    dag_id="fraud_detection_etl_pipeline",
    default_args=default_args,
    description="An ETL pipeline for fraud detection data processing and model retraining",
    # Set explicit timezone using pendulum to handle DST safely
    schedule="@daily",       # Runs at midnight daily (or use Cron strings like "0 2 * * *")
    catchup=False,           # Skips historical periods between start_date and today
    max_active_runs=1,       # Prevents multiple instances running concurrently
    tags=["production", "etl"],
)
def etl_pipeline():

    # 3. Define individual tasks using the @task decorator
    @task()
    def extract() -> dict:
        """Fetch new data from supabase."""
        print("Extracting data...")
        db = dblib.SessionLocal()
        # fetch all entries from 1 day ago
        data_payload = db.query(dblib.FraudDetection).filter(
            dblib.FraudDetection.created_at >= datetime.now() - timedelta(days=1)
        ).all()
        data_payload = [entry.__dict__ for entry in data_payload]
        return {"data_payload": data_payload}

    @task()
    def transform(raw_data: dict) -> list:
        """Process and clean data."""
        print(f"Transforming data: {raw_data}")
        # expect the new data be be the same as the old data since its a train test split of the larger dataset
        processed_data = raw_data["data_payload"]
        return processed_data

    @task()
    def load(processed_data: list):
        """Load data into a data warehouse or destination database."""
        #create parquet file with new data in the data lake
        os.makedirs(LOCAL_LAKE_PATH, exist_ok=True)
        df = pd.DataFrame(processed_data)
        execution_date = datetime.now().strftime("%Y-%m-%d")
        df.to_parquet(f"{LOCAL_LAKE_PATH}/batch_{execution_date}.parquet", index=False)
        print(f"Loading data into destination: {processed_data}")
        
    @task()
    def retrain_model():
        """Retrain the model using the new data."""
        print("Retraining model...")
        # Placeholder for model retraining logic
        df_full_dataset = pd.read_parquet("./data/parquet_lake/")
        X_train, X_test, y_train, y_test, df = train_model.data_preprocessing(df_full_dataset)
        # This could involve loading the new data, training a model, and saving it
        train_model(X_train, X_test, y_train, y_test, df)
        print("Model retraining complete.")

    # 4. Set downstream dependencies by chaining inputs and outputs
    raw_data = extract()
    cleaned_data = transform(raw_data)
    load(cleaned_data)
    retrain_model()
    
    
# 5. Call the function to register the DAG with the Airflow engine
etl_pipeline()