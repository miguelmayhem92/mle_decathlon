FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install mlflow==3.1.*

EXPOSE 5000

CMD ["mlflow", "server","--host", "0.0.0.0","--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db"]
