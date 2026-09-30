# Base Image
# TODO: Start from official python:3.11-slim base image
FROM python:3.12-slim

# Set Working Directory
# TODO: Set working directory inside container to /app
WORKDIR /app

# Dependency Caching Step
# TODO: Copy requirements.txt into working directory first
COPY requirements.txt .
# TODO: Run pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Files
# Copy the application module into the working directory used by Uvicorn.
COPY backend /app/backend
COPY model_training /app/model_training
COPY model.ubj /app/model.ubj

# Networking & Startup
# TODO: EXPOSE port 8000
EXPOSE 8000
ENV PYTHONPATH=/app
# TODO: Define CMD to run uvicorn server on host "0.0.0.0" and port 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]