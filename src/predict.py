import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from dispatcher import Dispatcher
from src.utils import Utils
from src.clean_data import DataLoader
from src.feature_generator import BuildFeatures 
from src.metrics import Metrics
import argparse
import pickle
import joblib
import json
import os
import pprint 
pp = pprint.PrettyPrinter(indent=4)

class Predict:
    """ Evaluates the model on the test set"""
    def __init__(self, config_path):
        self.config_path = config_path
        config = Utils().read_params(config_path)

        self.estimator = config["base"]["estimator"]

        self.clean_folds_path = config["clean_dataset"]["clean_folds_path"]
        self.clean_test_path = config["clean_dataset"]["clean_test_path"]
        self.model_dir = config["model_dir"]
        self.vectorizer_path = config["feature_generator"]["artifact_path"]

        self.random_state = config["base"]["random_state"]

        self.scores_file = config["reports"]["scores_test"]

    def test(self):
        """Train the model and test the model performance"""
        # Load the vectors and labels
        xtrain_ft, ytrain = BuildFeatures(self.config_path).build_features_test("train")
        xtest_ft, ytest = BuildFeatures(self.config_path).build_features_test("test")

        # Load the model
        clf = Dispatcher(self.config_path).dispatch_model(self.estimator)

        clf.fit(xtrain_ft, ytrain)
        preds = clf.predict(xtest_ft)
        pred_proba = clf.predict_proba(xtest_ft)

        metrics_dict = Metrics(self.config_path)._eval_metrics(ytest, preds, pred_proba)
        
        print("-" * 50)
        pp.pprint(metrics_dict)
        print("-" * 50)

        # Log Parameters and Scores for the test set
        with open(self.scores_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        # Save the model
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "model.joblib")
        joblib.dump(clf, model_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    Predict(config_path=parsed_args.config).test()