
from modules.inference.src_inference.inference import InferenceProduce


def handler(event:dict,context:None)->list[dict]:
    """
    executable for inferences

    params:
    event: input data as dictionary e.g. {"col1":list(), "col2":list()}
    """
    ip = InferenceProduce()
    input = ip.preprocess_input(event)
    merged_input = ip.get_features(input)
    ip.instantiate_model()
    predictions = ip.get_prediction(merged_input)
    return {"status":200, "output":predictions}