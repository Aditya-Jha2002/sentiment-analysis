from src.utils import Utils
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
import xgboost as xgb


class Trainer:
    """Trains a model on the given dataset"""
    def __init__(self, config_path):
        config = Utils().read_params(config_path)

        self.MODELS = {
            "logistic_regression": linear_model.LogisticRegression(**config["estimators"]["LogisticRegression"]["params"]),
            "naive_bayes": naive_bayes.MultinomialNB(**config["estimators"]["NaiveBayes"]["params"]),
            "random_forest": ensemble.RandomForestClassifier(**config["estimators"]["RandomForest"]["params"]),
            "xgboost": xgb.XGBClassifier(**config["estimators"]["XGBoost"]["params"]),
        }

    def dispatch_model(self, model_name):
        """Dispatches the model to train"""
        model = self.MODELS[model_name]
        return model
    