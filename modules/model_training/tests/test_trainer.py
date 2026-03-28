from modules.model_training.src.training_code import TrainerClient


DATA_FOLDER = "data"

def test_trainer():
    tc=TrainerClient(model_name="test")
    tc._extraction_job()
    tc._data_processing()
    tc._data_test()
    tc._get_model_definition()
    tc._fit_model()
    tc._evaluation()
    assert tc.metric_mae > 0
