import os

import loguru
import mlflow


def handler(event:list,context:None)->list[dict]:
    model = mlflow.sklearn.load_model("model")
    print(model)
    return {"status":200}