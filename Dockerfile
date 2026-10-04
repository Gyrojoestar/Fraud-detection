# Dockerfile for the backend and model training of the project
FROM python:3.12-slim

# set the working directory inside the container to /app
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the backend and model_training directories into the working directory used by Uvicorn
# copy the best model file into the working directory used by Uvicorn
COPY backend /app/backend
COPY model_training /app/model_training
COPY model.ubj /app/model.ubj

# Networking & Startup
EXPOSE 8000
ENV PYTHONPATH=/app
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]