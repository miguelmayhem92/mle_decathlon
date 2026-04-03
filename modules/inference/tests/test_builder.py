import pytest
from unittest.mock import patch
from modules.inference.src_inference.build import ModelBuilder, MyMlflowClient

@pytest.fixture(scope="session")
def model_builder():
    return ModelBuilder("some_url")




@patch.object(MyMlflowClient, "find_model_ids", return_value=("some_model_uri", "some_run_id"))
def test_experiment_finder(mock_find, model_builder):
    model_uri, run_id = model_builder._find_experiments("experiment_name", "run_name")
    assert model_uri == "some_model_uri"
    assert run_id == "some_run_id"



def test_download_artifacts(model_builder, monkeypatch):
    monkeypatch.setattr(MyMlflowClient, "download_artifact", lambda *args,**kwargs: None)
    result = model_builder._download_artifacts("some_model_uri", "some_run_id")
    assert result == None
    