import os

import pandas as pd
from loguru import logger
import mlflow




def handler(event:dict,context:None)->list[dict]:
    """
    executable for inferences

    params:
    event: input data as dictionary e.g. {"col1":list(), "col2":list()}
    """
    logger.info("transforming incoming messages")
    input = pd.DataFrame(event)
    input["day_id"] = pd.to_datetime(input["day_id"])
    bu_features = pd.read_csv(os.path.join("dep_features","bu_feat.csv.gz"))
    merged_input = pd.merge(input, bu_features, how="left", on = "but_num_business_unit")

    logger.info("produccing prediction")
    model = mlflow.sklearn.load_model("model")
    predictions = model.predict(merged_input)
    predictions = list(predictions)

    logger.info("sending output")
    return {"status":200, "output":predictions}