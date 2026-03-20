import os

import loguru
import mlflow

DOWNLOAD_DIR=os.environ.get("DOWNLOAD_DIR")

def handler(event:list,context:None)->list[dict]:
    model = mlflow.sklearn.load_model(DOWNLOAD_DIR)
    print(model)
    return {"status":200}