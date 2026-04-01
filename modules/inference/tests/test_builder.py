from unittest.mock import patch
from modules.inference.src_inference.build import ModelBuilder, MyMlflowClient

@patch.object(MyMlflowClient, "find_model_ids", return_value=("some_model_uri", "some_run_id"))

def test_experimetnt_finder(mock_find_experiments):
    mb=ModelBuilder("some_url")
    model_uri, run_id = mb._find_experiments("experiment_name", "run_name")
    assert model_uri == "some_model_uri"
    assert run_id == "some_run_id"
    mock_find_experiments.assert_called_once()


