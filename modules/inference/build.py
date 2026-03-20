import os
import sys
import argparse
sys.path.append(os.path.abspath("asset-layer/modules"))

import mlflow
import loguru
import pandas as pd
import numpy as np

# mlflow.set_tracking_uri("http://localhost:5000/") # for local testing
DOWNLOAD_DIR=os.environ.get("DOWNLOAD_DIR")

def download_model(experiment_name, run_name):
    """
    this function serach for model and model dependencies for build stage
    """
    client = mlflow.tracking.MlflowClient()

    loguru.info(f"looking for model experiment: {experiment_name}")
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    loguru.info(f"searching the run")
    for run in runs:
        if run.info.run_name == run_name and run.info.status == "FINISHED":
            print(run.info.run_id, run.info.status, run.info.start_time)
            run_id = run.info.run_id
            break
    
    loguru.info(f"searching the model")
    models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id = '{run_id}'"
    )
    model_id = models[0].model_id

    loguru.info(f"downloading model deps")
    # download model and deps
    model_uri = f"models:/{model_id}/"
    _ = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=DOWNLOAD_DIR,
    )
    dep_features= f'feature_store/data/bu_feat.csv.gz'
    mlflow.artifacts.download_artifacts(dep_features, dst_path="dep_features")

    # testing
    loguru.info(f"testing model")
    dep_features= f'feature_store/data/test.csv.gz'
    mlflow.artifacts.download_artifacts(dep_features, dst_path="tmp")
    df_bu_feat = pd.read("dep_features/bu_feat.csv.gz")
    df_test = pd.read("tmp/test.csv.gz")
    df_test_feat = pd.merge(df_test, df_bu_feat, how="left", on = "but_num_business_unit")

    model = mlflow.sklearn.load_model(DOWNLOAD_DIR)

    y_pred = model.predict(df_test_feat)
    assert float(np.mean(y_pred)) > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("en", "--experiment_name")
    parser.add_argument("rn", "--run_name")
    args = parser.parse_args()
    experiment_name = args.experiment_name
    run_name = args.run_name
    download_model(experiment_name, run_name)