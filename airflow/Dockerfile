FROM apache/airflow:2.8.1-python3.11

COPY airflow/requirements.txt /requirements.txt
USER airflow
RUN pip install --no-cache-dir \
    --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.11.txt \
    -r /requirements.txt
