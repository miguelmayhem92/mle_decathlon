import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomPreprocressing(BaseEstimator, TransformerMixin):
    """
    This class includes all the steps for the preprocessing
    """
    def __init__(self, cat_cols):
        """
        Initialize the class / Can be empty
        """
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        """
        This method is only created so that the pipeline containing this transformer does not raise an error
        """
        return self

    def transform(self, data):
        """
        Inputs :
          -- data : DataFrame, DataFrame contening all the data needed for the model
        Outputs :
          -- DataFrame, DataFrame prepared for modeling

        """
        data["day_id"] = pd.to_datetime(data["day_id"]).copy() # these lines are problematic with warnings, using loc istead 
        data["day_id_week"] = data.day_id.dt.isocalendar().week
        data["day_id_month"] = data["day_id"].dt.month
        data["day_id_year"] = data["day_id"].dt.year
        data[self.cat_cols] = data[self.cat_cols].apply(lambda x: x.astype(str))
        return data