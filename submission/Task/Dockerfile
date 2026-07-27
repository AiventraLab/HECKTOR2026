FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/model
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY submission/Task/ .

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "inference.py"]
