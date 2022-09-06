import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from src.utils import Utils
from src.feature_generator import BuildFeatures 
from src.dispatcher import Dispatcher
from src.metrics import Metrics
import argparse
import joblib
import json
import pprint 
pp = pprint.PrettyPrinter(indent=4)

class Trainer:
    """Trains a model on the given dataset"""
    def __init__(self, config_path):
        self.config_path = config_path

        config = Utils().read_params(config_path)

        self.base = config["base"]
        self.estimator = config["estimator"]

        self.clean_data_path = config["clean_dataset"]["clean_folds_path"]
        self.model_dir = config["model_dir"]

        self.random_state = config["base"]["random_state"]

        self.scores_file = config["reports"]["scores_cv"]

    def train_and_evaluate(self):
        """Train the model and evaluate the model performance"""
        running_accuracy, running_f1, running_roc_auc, running_log_loss = [], [], [], []

        for i in range(1, 6):
            (accuracy, f1, roc_auc, log_loss_score) = self._train_one_fold(i - 1)

            running_accuracy.append(float(accuracy))
            running_f1.append(float(f1))
            running_roc_auc.append(float(roc_auc))
            running_log_loss.append(float(log_loss_score))

        scores = {
                "accuracy_score": sum(running_accuracy)/5,
                "f1_score": sum(running_f1)/5,
                "roc_auc_score": sum(running_roc_auc)/5,
                "log_loss_score": sum(running_log_loss)/5
            }

        print("-" * 50)
        print(f"  Model name: {self.estimator}")
        print(f"  Scores: \n")
        pp.pprint(scores)
        print("-" * 50)

        # Log Parameters and Scores for the cross validation
        with open(self.scores_file, "w") as f:
            json.dump(scores, f, indent=4)


    def _train_one_fold(self, fold_num):
        print(f"Training fold {fold_num} ...")
        xtrain_ft, ytrain, xvalid_ft, yvalid = BuildFeatures(self.config_path).build_features_train(fold_num)
        
        clf = Dispatcher(self.config_path).dispatch_model(self.estimator)

        clf.fit(xtrain_ft, ytrain)
        preds = clf.predict(xvalid_ft)
        pred_proba = clf.predict_proba(xvalid_ft)

        metrics_dict = Metrics(self.config_path).eval_metrics(yvalid, preds, pred_proba)
        
        print("-" * 50)
        print(f"  Fold {fold_num} score: \n")
        pp.pprint(metrics_dict)
        print("-" * 50)

        return metrics_dict["accuracy"], metrics_dict["f1"], metrics_dict["roc_auc"], metrics_dict["log_loss"]

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    Trainer(config_path=parsed_args.config).train_and_evaluate()