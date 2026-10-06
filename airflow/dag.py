import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import pendulum
from datetime import timedelta
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.exceptions import AirflowException

import backend.database.conn as dblib
from model_training.train import data_preprocessing, train_model


# define default arguments applied to all tasks and global variables for the DAG
default_args = {
    'owner': 'AndrewJWY',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['andrewjwy@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

LOCAL_LAKE_PATH = Path("/opt/airflow/data/parquet_lake")

# instantiate the DAG using the @dag decorator
@dag(
    dag_id="fraud_detection_etl_pipeline",
    default_args=default_args,
    description="An ETL pipeline for fraud detection data processing and model retraining",
    # set explicit timezone using pendulum to handle DST safely
    schedule="@daily",
    catchup=False,
    max_active_runs=1,       # prevents multiple instances running concurrently
    tags=["production", "etl"],
)
def etl_pipeline():

    # define tasks using the @task decorator
    @task()
    def extract() -> dict:
        """Fetch new data from supabase."""
        print("Extracting data...")
        db = dblib.SessionLocal()
        try:
            # fetch all entries from 1 day ago
            db_entries = db.query(dblib.RawTransaction).filter(
                dblib.RawTransaction.created_at >= datetime.now() - timedelta(days=1)
            ).all()
            
            # if no entries in the last 24h, fetch the latest 100 entries from the database
            if not db_entries:
                print("No transactions in last 24h. Fetching latest 100 entries from DB...")
                db_entries = db.query(dblib.RawTransaction).order_by(
                    dblib.RawTransaction.created_at.desc()
                ).limit(100).all()

            if not db_entries:
                raise AirflowException("Extraction Failed: Supabase database returned 0 records!")

            data_payload = []
            for entry in db_entries:
                # access via explicit Python attributes safely
                row_dict = {
                    "transaction_id": entry.transaction_id,
                    "amount": entry.amount,
                    "is_fraud": entry.is_fraud,
                    "created_at": str(entry.created_at) if entry.created_at else None,
                    **{f"v{i}": getattr(entry, f"v{i}") for i in range(1, 29)}
                }
                data_payload.append(row_dict)
        except Exception as e:
            print(f"Extraction error encountered: {e}")
            raise AirflowException(f"Extract task terminated due to error: {str(e)}")
        finally:
            db.close()
        return {"data_payload": data_payload}

    @task()
    def transform(raw_data: dict) -> list:
        """Process and clean data."""
        print(f"Transforming data: {raw_data}")
        # expect the new data formatting to be the same as the old data since its a train test split of the larger dataset
        processed_data = raw_data["data_payload"]
        return processed_data

    @task()
    def load(processed_data: list):
        """Load data into a data warehouse or destination database."""
        # create parquet file with new data in the data lake
        os.makedirs(LOCAL_LAKE_PATH, exist_ok=True)
        df = pd.DataFrame(processed_data)
        execution_date = datetime.now().strftime("%Y-%m-%d")
        df.to_parquet(f"{LOCAL_LAKE_PATH}/batch_{execution_date}.parquet", index=False)
        print(f"Loading data into destination: {processed_data}")
        
    @task()
    def retrain_model():
        """Retrain the model using the new data."""
        print("Retraining model...")
        # retrieve all the data from the lake do a train test split
        print("Retraining model...")
        parquet_files = list(LOCAL_LAKE_PATH.glob("*.parquet"))
        
        if not parquet_files:
            print("No parquet files found in lake. Skipping retraining.")
            return
        # load all parquet files into a single DataFrame
        df_full_dataset = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        X_train, X_test, y_train, y_test, df = data_preprocessing(df_full_dataset)
        # retrain the model and save the best model to model.ubj
        train_model(X_train, X_test, y_train, y_test, df)
        print("Model retraining complete.")

    # set downstream dependencies by chaining inputs and outputs
    raw_data = extract()
    cleaned_data = transform(raw_data)
    load_task = load(cleaned_data)
    retrain_task = retrain_model()
    load_task >> retrain_task
    
    
# call the function to register the DAG with the Airflow engine
etl_pipeline()