from modules.model_training.src.training_code import TrainerClient

def test_trainer():
    tc=TrainerClient(model_name="test")
    tc._extraction_job()
    tc._data_processing(sample=0.1)
    tc._data_test()
    tc._get_model_definition()
    tc._fit_model()
    tc._evaluation()
    assert tc.metric_mae > 0
