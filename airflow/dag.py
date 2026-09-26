from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import pendulum
from datetime import timedelta
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago


# 1. Define default arguments applied to all tasks
default_args = {
    'owner': 'AndrewJWY',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['andrewjwy@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

# # 2. Instantiate the DAG using the @dag decorator
# @dag(
#     dag_id="etl_boilerplate_v1",
#     default_args=default_args,
#     description="A production-ready boilerplate DAG using TaskFlow API",
#     # Set explicit timezone using pendulum to handle DST safely
#     start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
#     schedule="@daily",       # Runs at midnight daily (or use Cron strings like "0 2 * * *")
#     catchup=False,           # Skips historical periods between start_date and today
#     max_active_runs=1,       # Prevents multiple instances running concurrently
#     tags=["production", "etl"],
# )
# def etl_pipeline():

#     # 3. Define individual tasks using the @task decorator
#     @task()
#     def extract() -> dict:
#         """Fetch data from source."""
#         print("Extracting data...")
#         return {"data_payload":}

#     @task()
#     def transform(raw_data: dict) -> list:
#         """Process and clean data."""
#         print(f"Transforming data: {raw_data}")
#         processed_data = [x * 10 for x in raw_data["data_payload"]]
#         return processed_data

#     @task()
#     def load(processed_data: list):
#         """Load data into a data warehouse or destination database."""
#         print(f"Loading data into destination: {processed_data}")

#     # 4. Set downstream dependencies by chaining inputs and outputs
#     raw_data = extract()
#     cleaned_data = transform(raw_data)
#     load(cleaned_data)

# # 5. Call the function to register the DAG with the Airflow engine
# etl_pipeline()