import os
import pandas as pd
import numpy as np

from modules.inference.src_inference.inference import InferenceProduce

def test_inference():

    class MyFakemodel:
        def predict(self,X:pd.DataFrame):
            return np.random.rand(1,X.shape[0])

    event = {
        "day_id":["2017-05-01"],
        "but_num_business_unit":[64],
        "dpt_num_department":[125]
    }

    os.environ["DEPS_DATA_FOLDER"] = "modules/data"

    ip = InferenceProduce()
    input = ip.preprocess_input(event)
    merged_input = ip.get_features(input)
    ip.model = MyFakemodel()
    predictions = ip.get_prediction(merged_input)

    assert len(predictions["prediction"]) > 0
    assert np.mean(predictions["prediction"]) > 0