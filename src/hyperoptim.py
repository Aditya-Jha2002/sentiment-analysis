import os
import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from src.metrics import Metrics

from src import utils
from src.build_features import BuildFeatures

from functools import partial

import optuna
import mlflow

import pprint 
pp = pprint.PrettyPrinter(indent=4)

config_path = "config.yaml"

config = utils.read_params(config_path)

feature_dir = config["build_features"]["feature_dir"]
feature_train = config["build_features"]["train"]

estimator = config["base"]["estimator"]
params = config["hyperoptim"][estimator]["params"]

mlflow_config = config["mlflow_config"]
artifacts_dir = mlflow_config["artifacts_dir"] 
experiment_name = mlflow_config["experiment_name"]
tracking_uri = mlflow_config["remote_server_uri"]

mlflow.set_experiment(experiment_name)
mlflow.set_tracking_uri(tracking_uri)

def optimize(trial):
    """Optimize the model parameters"""
    df = pd.read_csv(os.path.join("./", feature_dir, feature_train))

    features = list(df.columns)
    features.remove("Target")
    features.remove("kfold")

    model = xgb.XGBRegressor(**params)

    running_accuracy, running_f1, running_roc_auc, running_log_loss = [], [], [], []

    for fold_num in range(5):
        xtrain = df[df["kfold"] != fold_num][features]
        ytrain = df[df["kfold"] != fold_num]["Target"]

        xvalid = df[df["kfold"] == fold_num][features]
        yvalid = df[df["kfold"] == fold_num]["Target"]

        xtrain_ft, xvalid_ft = BuildFeatures(config_path).build_features_for_model(xtrain, xvalid, None)
        
        model.fit(xtrain_ft, ytrain)
        preds = model.predict(xvalid_ft)
        pred_proba = model.predict_proba(xvalid_ft)
        (accuracy, f1, roc_auc, log_loss_score) = Metrics(config_path).eval_metrics(yvalid, preds, pred_proba)

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
    print(f"  Model name: {estimator}")
    print(f"  Scores: \n")
    pp.pprint(scores)
    print("-" * 50)
    
    mlflow.log_param(params)
    mlflow.log_metric(scores)

    return np.mean(running_f1)

if __name__ == "__main__":
    opimization_function = partial(optimize)

    study = optuna.create_study(study_name = experiment_name, direction="minimize")
    study.optimize(opimization_function, n_trials=50)