# STEP 1: Base Image
# TODO: Start from official python:3.11-slim base image
FROM python:3.12-slim

# STEP 2: Set Working Directory
# TODO: Set working directory inside container to /app
WORKDIR /app

# STEP 3: Dependency Caching Step
# TODO: Copy requirements.txt into working directory first
COPY requirements.txt .
# TODO: Run pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# STEP 4: Copy Application Files
# TODO: Copy model.pkl and main.py into working directory
COPY main.py ./
COPY model/model.pkl ./model/

# STEP 5: Networking & Startup
# TODO: EXPOSE port 8000
EXPOSE 8000
# TODO: Define CMD to run uvicorn server on host "0.0.0.0" and port 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]