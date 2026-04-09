import os
import numpy as np

from modules.inference.src_inference.inference import InferenceProduce

class TestInference:
    def test_inference(self, mocker):

        MockModel = mocker.patch("modules.inference.src_inference.inference.load_model", autospec=True)
        MockModel.return_value.predict.side_effect = lambda X: np.random.rand(1,X.shape[0])

        event = {
            "day_id":["2017-05-01"],
            "but_num_business_unit":[64],
            "dpt_num_department":[125]
        }

        os.environ["DEPS_DATA_FOLDER"] = "modules/data"

        ip = InferenceProduce()
        input = ip.preprocess_input(event)
        merged_input = ip.get_features(input)
        ip.instantiate_model()
        predictions = ip.get_prediction(merged_input)

        assert len(predictions["prediction"]) > 0
        assert np.mean(predictions["prediction"]) > 0