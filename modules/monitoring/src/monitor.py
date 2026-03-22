import os

import nannyml as nml
import pandas as pd
import mlflow
from loguru import logger


DATA_FOLDER = "data"
EXPERIMENT_NAME="forecasting"
RUN_NAME="run_0"
BASE_DIR = os.path.dirname("app")
TARGET = "turnover"

mlflow.set_tracking_uri("http://host.docker.internal:5000") # in docker
mlflow.set_tracking_uri("http://localhost:5000") # local
DOWNLOAD_DIR=os.getenv("DOWNLOAD_DIR", "model_deps")
YEAR_SPLIT = 2016

class Monitor:
    def __init__(self,application_name:str):
        self.application_name=application_name

    def run(self, experiment_name:str, run_name:str):
        self._extraction_job()
        self._data_processing()
        self._fetch_model(experiment_name, run_name)
        self._get_features()
        self._prediction_monitor()
        self._univariate_monitor()
        self._log_results()

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

        # mergin the data
        df_train_feat = pd.merge(self.df_train, self.df_bu_feat, how="left", on = "but_num_business_unit")
        df_test_feat = pd.merge(self.df_test, self.df_bu_feat, how="left", on = "but_num_business_unit")

        df_train_feat["day_id"] = pd.to_datetime(df_train_feat["day_id"])
        df_test_feat["day_id"] = pd.to_datetime(df_test_feat["day_id"])

        df_train_feat["day_id"] = df_train_feat["day_id"].dt.strftime('%Y-%m-%d')
        df_test_feat["day_id"] = df_test_feat["day_id"].dt.strftime('%Y-%m-%d')

        df_train_feat["day_id"] = pd.to_datetime(df_train_feat["day_id"])
        df_test_feat["day_id"] = pd.to_datetime(df_test_feat["day_id"])

        self.df_train_feat = df_train_feat
        self.df_test_feat = df_test_feat

    def _fetch_model(self, experiment_name, run_name):
        logger.info("fetching production model")
        os.makedirs("tmp", exist_ok=True)
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        client = mlflow.tracking.MlflowClient()

        logger.info(f"looking for model experiment: {experiment_name}")
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])

        logger.info(f"searching the run")
        for run in runs:
            if run.info.run_name == run_name and run.info.status == "FINISHED":
                run_id = run.info.run_id
                break
        
        logger.info(f"searching the model")
        models = client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id = '{run_id}'"
        )
        model_id = models[0].model_id

        logger.info(f"downloading model deps")
        logger.info(f"{os.getcwd()}")
        # download model and deps
        model_uri = f"models:/{model_id}/"
        _ = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=DOWNLOAD_DIR,
        )

        self.model = mlflow.sklearn.load_model(DOWNLOAD_DIR)
        self.run_id = run_id

    def _get_features(self):
        logger.info(f"finding feature names")
        transform = self.model[0]
        features = list(transform.transform(self.df_train_feat).columns)
        self.features = [x for x in features if x not in ["turnover","day_id"]]

    def _prediction_monitor(self):

        df_ref_feat = self.df_train_feat[(self.df_train_feat.day_id.dt.year > YEAR_SPLIT)].copy()
        df_ref_feat = df_ref_feat.reset_index(drop=True)
        logger.info(f"fitting dle")
        dle = nml.DLE(
            metrics=['mae'],
            y_true=TARGET,
            y_pred='y_pred',
            feature_column_names=self.features,
            timestamp_column_name='day_id',
            chunk_number=500,
            tune_hyperparameters=False
        )

        dle.fit(df_ref_feat)

        logger.info(f"dle results")
        y_pred = self.model.predict(self.df_test_feat)
        self.df_test_feat["y_pred"] = y_pred
        estimated_performance = dle.estimate(self.df_test_feat)
        estimated_performance.write_html("tmp/dle_monitor.html")
    
    def _univariate_monitor(self):
        logger.info(f"fitting univariate drift calculator")
        udc = nml.UnivariateDriftCalculator(
            column_names=self.features,
            timestamp_column_name='day_id',
            chunk_number=500,
        )

        udc.fit(self.df_ref_feat)
        univariate_data_drift = udc.calculate(self.df_test_feat)
        univariate_data_drift.write_html("tmp/univariate_monitor.html")

    def _log_results(self):
        logger.info(f"logging results")
        with mlflow.start_run(run_id=self.run_id) as run:
            mlflow.log_artifact("tmp/","nanny_monitoring")