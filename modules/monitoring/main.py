import argparse
import os

from loguru import logger
import mlflow

from src_monitor.monitor import Monitor

DOWNLOAD_DIR=os.getenv("DOWNLOAD_DIR", "model_deps")

mlflow.sklearn.load_model(DOWNLOAD_DIR)
logger.info("done!!!!!!!!!!!!!!!!!!")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-en", "--experiment_name",default="forecasting")
    parser.add_argument("-rn", "--run_name",default="run_0")
    args = parser.parse_args()
    experiment_name = args.experiment_name
    run_name = args.run_name

    trainer = Monitor(application_name="forecasting_usecase")
    trainer.run(experiment_name, run_name)

