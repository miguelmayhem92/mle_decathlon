import os

import pandas as pd
import mlflow
from loguru import logger

from modules.inference.src_inference.models.input import Message

DOWNLOAD_DIR=os.getenv("DOWNLOAD_DIR", "modules/model_training")

class InferenceProduce:
    def preprocess_input(self, input:dict)->pd.DataFrame:
        logger.info("transforming incoming messages")
        input = Message(**input)
        input_pd = pd.DataFrame(input.model_dump())
        input_pd["day_id"] = pd.to_datetime(input_pd["day_id"])
        return input_pd

    def instantiate_model(self):
        self.model = mlflow.sklearn.load_model(DOWNLOAD_DIR)

    def get_features(self,input_pd:pd.DataFrame) -> pd.DataFrame:
        logger.info("tgetting features")
        deps_data_folder = os.getenv("DEPS_DATA_FOLDER","dep_features")
        self.bu_features = pd.read_csv(os.path.join(deps_data_folder,"bu_feat.csv.gz"))
        self.bus = list(self.bu_features["but_num_business_unit"].unique())
        merged_input = pd.merge(input_pd, self.bu_features, how="left", on = "but_num_business_unit")
        return merged_input
    
    def get_prediction(self, input_pd:pd.DataFrame) -> dict:
        logger.info("produccing prediction")
        predictions = self.model.predict(input_pd)
        predictions = list(predictions)
        input_pd["prediction"] = predictions
        input_pd["day_id"] = input_pd["day_id"].dt.strftime('%Y-%m-%d')
        return input_pd.to_dict("list")
