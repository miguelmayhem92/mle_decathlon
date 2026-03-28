import os
import pandas as pd

class DataLoader:
    def __init__(self, data_folder:str):
        self.data_folder = data_folder

    def get_data(self, data_name:str)->pd.DataFrame:
        return pd.read_csv(os.path.join(self.data_folder,data_name))