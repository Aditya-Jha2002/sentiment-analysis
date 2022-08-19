# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from src.utils import Utils
from sklearn.model_selection import train_test_split, StratifiedKFold


class CVFoldsDataset:
    """Split the dataset into train, dev and test"""

    def __init__(self, config_path):
        config = Utils().read_params(config_path)
        self.raw_data_path = config["data_source"]["raw_data_path"]
        self.folds_data_path = config["cv_folds_dataset"]["folds_data_path"]
        self.fold_num = config["cv_folds_dataset"]["fold_num"]
        self.test_data_path = config["cv_folds_dataset"]["test_data_path"]
        self.test_size = config["cv_folds_dataset"]["test_size"]
        self.random_state = config["base"]["random_state"]

    def cv_folds_dataset(self):
        """Runs scripts to load the raw data from (../raw) into
        fold datasets ready to be further cleaned on (saved in ../interim),
        and test dataset ready to be used for final test on (saved in ../test)
        """
        df = Utils().get_data(self.raw_data_path)

        df.drop(["Unnamed: 0"], axis=1, inplace=True)

        df, test = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )

        df_folds = self._create_folds(df)

        df_folds.to_csv(self.folds_data_path, index=False)
        test.to_csv(self.test_data_path, index=False)

    
    def _create_folds(self, data):
        """Create folds for cross-validation"""
        data["kfold"] = -1
    
        # the next step is to randomize the rows of the data
        data = data.sample(frac=1).reset_index(drop=True)

        # initiate the kfold class from model_selection module
        kf = StratifiedKFold(n_splits=self.fold_num)
        
        # fill the new kfold column
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data.airline_sentiment.values)):
            data.loc[v_, 'kfold'] = f

        return data

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    CVFoldsDataset(config_path=parsed_args.config).cv_folds_dataset()
