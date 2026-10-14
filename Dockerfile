FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml requirements.txt ./
COPY mlops_app ./mlops_app
COPY dashboard.py train.py ./
RUN pip install --no-cache-dir .
EXPOSE 8000 8501
