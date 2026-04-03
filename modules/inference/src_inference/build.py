import os
import sys
import argparse
import shutil

sys.path.append(os.path.abspath("deps"))

import mlflow
from loguru import logger
import pandas as pd
import numpy as np

DOWNLOAD_DIR=os.getenv("DOWNLOAD_DIR", "modules/model_training")


class MyMlflowClient:
    def __init__(self,mlflow_url:str):
        mlflow.set_tracking_uri(mlflow_url)
        self.mlflow_client = mlflow.tracking.MlflowClient()

    def find_model_ids(self,experiment_name:str, run_name:str)->tuple[str,str]:
        experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
        runs = self.mlflow_client.search_runs(experiment_ids=[experiment.experiment_id])

        logger.info(f"searching the run")
        for run in runs:
            if run.info.run_name == run_name and run.info.status == "FINISHED":
                run_id = run.info.run_id
                break
        
        logger.info(f"searching the model")
        models = self.mlflow_client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id = '{run_id}'"
        )
        model_id = models[0].model_id
        model_uri = f"models:/{model_id}/"
        return model_uri, run_id
    
    def download_artifact(self,**kwargs):
        return mlflow.artifacts.download_artifacts(**kwargs)

        


class ModelBuilder:
    def __init__(self, mlflow_url):
        self.mlflow_client = MyMlflowClient(mlflow_url)

    def run(self, experiment_name:str, run_name:str):
        self._maker_dirs()
        model_uri, run_id = self._find_experiments(experiment_name, run_name)
        self._download_artifacts(model_uri, run_id)
        self._fix_code_deps_folder()
        df_test, df_bu_feat, model = self._get_test_data(run_id)
        self._test_model(df_test, df_bu_feat, model)

    def _maker_dirs(self):
        os.makedirs("tmp", exist_ok=True)
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    def _find_experiments(self, experiment_name:str, run_name:str)->tuple[str,str]:
        logger.info(f"looking for model experiment: {experiment_name}")
        model_uri, run_id = self.mlflow_client.find_model_ids(experiment_name, run_name)
        return model_uri, run_id

    def _download_artifacts(self, model_uri:str, run_id:str):
        logger.info(f"fetchin model: {model_uri}")
        logger.info(f"downloading model deps")
        logger.info(f"download model")
        self.mlflow_client.download_artifact(artifact_uri=model_uri,
            dst_path=DOWNLOAD_DIR)

        self.mlflow_client.download_artifact(run_id=run_id,
            artifact_path=f'data_version/data/bu_feat.csv.gz',
            dst_path="dep_features")
    
    def _fix_code_deps_folder(self):
        src_code_path = os.path.join(DOWNLOAD_DIR,"code","src")
        dest_code_path = os.path.join(DOWNLOAD_DIR,"src")
        shutil.copytree(src_code_path, dest_code_path)
        shutil.rmtree(src_code_path)
        
    def _get_test_data(self, run_id):
        logger.info(f"getting testing artifacts and data")
        self.mlflow_client.download_artifact(
            run_id=run_id,
            artifact_path=f'data_version/data/test.csv.gz',
            dst_path="tmp"
        )
        df_test = pd.read_csv("tmp/test.csv.gz")
        df_bu_feat = pd.read_csv("dep_features/bu_feat.csv.gz")
        model = mlflow.sklearn.load_model(DOWNLOAD_DIR)
        return df_test, df_bu_feat, model
    
    def _test_model(self, df_test, df_bu_feat, model):
        logger.info(f"testing model")
        df_test_feat = pd.merge(df_test, df_bu_feat, how="left", on = "but_num_business_unit")
        y_pred = model.predict(df_test_feat)
        assert float(np.mean(y_pred)) > 0
        logger.info(f"test passed, build done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-en", "--experiment_name",default="forecasting")
    parser.add_argument("-rn", "--run_name",default="run_0")
    parser.add_argument("-mu", "--mlflow_url")
    args = parser.parse_args()
    experiment_name = args.experiment_name
    run_name = args.run_name
    mlflow_url = args.mlflow_url
    mb=ModelBuilder(mlflow_url)
    mb.run(experiment_name, run_name)