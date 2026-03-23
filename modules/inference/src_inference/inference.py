import os

import pandas as pd
import mlflow
from loguru import logger

from src_inference.models.input import Message

class InferenceProduce:
    def __init__(self, input:dict):
        logger.info("transforming incoming messages")
        input = Message(**input)
        input_pd = pd.DataFrame(input.model_dump())
        input_pd["day_id"] = pd.to_datetime(input_pd["day_id"])
        self.input_pd = input_pd
        
    def get_features(self) -> pd.DataFrame:
        logger.info("tgetting features")
        self.bu_features = pd.read_csv(os.path.join("dep_features","bu_feat.csv.gz"))
        self.bus = list(self.bu_features["but_num_business_unit"].unique())
        merged_input = pd.merge(self.input_pd, self.bu_features, how="left", on = "but_num_business_unit")
        return merged_input
    
    def get_prediction(self, input:pd.DataFrame) -> list:
        logger.info("produccing prediction")
        model = mlflow.sklearn.load_model("model")
        predictions = model.predict(input)
        predictions = list(predictions)
        self.input_pd["prediction"] = predictions
        self.input_pd["day_id"] = self.input_pd["day_id"].dt.strftime('%Y-%m-%d')
        return self.input_pd.to_dict("list")
