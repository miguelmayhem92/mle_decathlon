import os
import sys
import argparse

sys.path.append(os.path.abspath("deps"))

import mlflow
from loguru import logger
import pandas as pd
import numpy as np

# mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_tracking_uri("http://host.docker.internal:5000") 
DOWNLOAD_DIR=os.getenv("DOWNLOAD_DIR", "dependencies_model")

def download_model(experiment_name, run_name):
    """
    this function search for model and model dependencies in mlflow server for build stage
    """
    os.makedirs("tmp", exist_ok=True)
    os.makedirs("dep_features", exist_ok=True)
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
    # download model and deps
    model_uri = f"models:/{model_id}/"
    _ = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=DOWNLOAD_DIR,
    )

    mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=f'feature_store/data/bu_feat.csv.gz',
        dst_path="dep_features")

    # testing
    logger.info(f"testing model")
    mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=f'feature_store/data/test.csv.gz',
        dst_path="tmp")
    df_bu_feat = pd.read_csv("dep_features/bu_feat.csv.gz")
    df_test = pd.read_csv("tmp/test.csv.gz")
    df_test_feat = pd.merge(df_test, df_bu_feat, how="left", on = "but_num_business_unit")

    model = mlflow.sklearn.load_model(DOWNLOAD_DIR)

    y_pred = model.predict(df_test_feat)
    assert float(np.mean(y_pred)) > 0
    logger.info(f"test passed, build done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-en", "--experiment_name",default="forecasting")
    parser.add_argument("-rn", "--run_name",default="run_0")
    args = parser.parse_args()
    experiment_name = args.experiment_name
    run_name = args.run_name
    download_model(experiment_name, run_name)