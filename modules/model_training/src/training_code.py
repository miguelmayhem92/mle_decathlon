import os
from pathlib import Path

import yaml
import pandas as pd
from loguru import logger

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

from custom_transformers import CustomPreprocressing

DATA_FOLDER = "data"

class TrainerClient:
    """
    this class contains the main steps for ml traning
    """
    def __init__(self,model_name:str)->None:
        self.model_name = model_name
        configs = yaml.safe_load(Path(os.path.join("configs","training.yml")).read_text())
        self.preprocessing_configs = configs["preprosessing"]
        self.features_config = configs["features"]

    def run(self):
        self._extraction_job()
        self._data_processing()
        self._get_model_definition()
        self._fit_model()
        self._evaluation()
        self._log_model()


    def _extraction_job(self):
        """
        this method extracts the data from the warehouse
        """
        logger.info("Launching data extraction")
        self.df_bu_feat = pd.read_csv(os.path.join(DATA_FOLDER,"bu_feat.csv.gz")) 
        self.df_train = pd.read_csv(os.path.join(DATA_FOLDER, "train.csv.gz"))
        self.df_test = pd.read_csv(os.path.join(DATA_FOLDER, "test.csv.gz"))
        logger.info("Data is extracted")

    def _data_processing(self):
        """
        this method executes data preprocessing
        """
        logger.info("Data prerpocessing")
        year_split = self.preprocessing_configs["year_split"]
        # mergin the data
        df_train_feat = pd.merge(self.df_train, self.df_bu_feat, how="left", on = "but_num_business_unit")
        df_test_feat = pd.merge(self.df_test, self.df_bu_feat, how="left", on = "but_num_business_unit")

        # Train and val set
        df_train_feat["day_id"] = pd.to_datetime(df_train_feat["day_id"])
        df_train_feat["day_id_week"] = df_train_feat.day_id.dt.isocalendar().week
        df_train_feat["day_id_month"] = df_train_feat["day_id"].dt.month
        df_train_feat["day_id_year"] = df_train_feat["day_id"].dt.year

        # 2017 must a variable
        self.df_train = df_train_feat[(df_train_feat.day_id_year < year_split)]
        self.df_val = df_train_feat[(df_train_feat.day_id_year == year_split)]

        self.y_train = self.df_train.turnover
        self.y_val = self.df_val.turnover
        logger.info("Data preprosessing is finished")


    def _get_model_definition(self):
        """
        this method defines model architecture
        """
        logger.info("getting model definition")
        num_attrib = self.features_config["numeric_attributes"]
        cat_attrib = self.features_config["categorical_atributes"]
        logger.info("Definig model architecture")
        num_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
        ])
        cat_onehot_pipeline = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown="ignore")),
        ])
        preparation_pipeline = ColumnTransformer([
            ("num",num_pipeline, num_attrib),
            ("cat_onehot", cat_onehot_pipeline, cat_attrib)
        ])

        self.model_pipeline = Pipeline([
            ('preprocessing', CustomPreprocressing(cat_cols=cat_attrib )),
            ('preparation', preparation_pipeline),
            ('model', GradientBoostingRegressor())
        ])

    def _fit_model(self):
        logger.info("fiting model definition")
        self.model_pipelinefit(self.df_train, self.y_train)
    
    def _evaluation(self):
        logger.info("model evaluation")
        self.metric_mae = mean_absolute_error(self.y_val, self.y_predict_val)

    def _log_model(self):
        pass
