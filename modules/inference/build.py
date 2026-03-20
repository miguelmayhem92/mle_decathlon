import mlflow
from  mlflow.tracking import MlflowClient

def download_model(model_uri):
    hmm_model = mlflow.pyfunc.load_model(f"runs:/{run_id_prod_model}/{rename_ticket_name}-hmm-model",suppress_warnings = True)