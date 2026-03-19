import os

import pandas as pd

DATA_FOLDER = "data"
YEAR_SPLIT = 2017 # this can be a dynamic config

class TrainerClient:
    """
    this class contains the main steps for ml traning
    """
    def __init__(self,model_name:str)->None:
        self.model_name = model_name
    
    def extraction_job(self):
        """
        this method extracts the data from the warehouse
        """
        self.df_bu_feat = pd.read_csv(os.path.join(DATA_FOLDER,"bu_feat.csv.gz")) 
        self.df_train = pd.read_csv(os.path.join(DATA_FOLDER, "train.csv.gz"))
        self.df_test = pd.read_csv(os.path.join(DATA_FOLDER, "test.csv.gz"))

    def data_processing(self):
        """
        this method executes data preprocessing
        """
        # mergin the data
        df_train_feat = pd.merge(self.df_train, self.df_bu_feat, how="left", on = "but_num_business_unit")
        df_test_feat = pd.merge(self.df_test, self.df_bu_feat, how="left", on = "but_num_business_unit")

        # Train and val set
        df_train_feat["day_id"] = pd.to_datetime(df_train_feat["day_id"])
        df_train_feat["day_id_week"] = df_train_feat.day_id.dt.isocalendar().week
        df_train_feat["day_id_month"] = df_train_feat["day_id"].dt.month
        df_train_feat["day_id_year"] = df_train_feat["day_id"].dt.year

        # 2017 must a variable
        df_train = df_train_feat[(df_train_feat.day_id_year < YEAR_SPLIT)]
        df_val = df_train_feat[(df_train_feat.day_id_year == YEAR_SPLIT)]

        y_train = df_train.turnover
        y_val = df_val.turnover