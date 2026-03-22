import argparse
from src.monitor import Monitor

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-en", "--experiment_name",default="forecasting")
#     parser.add_argument("-rn", "--run_name",default="run_0")
#     args = parser.parse_args()
#     experiment_name = args.experiment_name
#     run_name = args.run_name

#     trainer = Monitor(application_name="forecasting_usecase")
#     trainer.run(experiment_name, run_name)

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

print(os.getcwd())
mlflow.sklearn.load_model("model_deps")