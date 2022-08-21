import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from src.utils import Utils
from src.clean_data import DataLoader
from src.build_features import BuildFeatures 
import argparse
import pickle

class Predict:
    """ Evaluates the model on the test set"""
    def __init__(self, config_path):
        self.config_path = config_path
        config = Utils().read_params(config_path)
        self.test_data_path = config["cv_folds_dataset"]["test_data_path"]
        self.clean_data_path = config["clean_dataset"]["clean_data_path"]
        self.random_state = config["base"]["random_state"]
        self.model_dir = config["model_dir"]
        self.tfv_artifact_path = config["build_features"]["tfv_artifact_path"]

        self.C = config["estimators"]["LogisticRegression"]["params"]["C"]
        self.l1_ratio = config["estimators"]["LogisticRegression"]["params"]["l1_ratio"]

    def test(self):
        """Train the model and test the model performance"""
        train_df = pd.read_csv(self.clean_data_path)
        test_df = pd.read_csv(self.test_data_path)
        test_df["text"] = test_df["text"].apply(DataLoader(self.config_path).preprocess_text)
        test_df["airline_sentiment"] = test_df["airline_sentiment"].map({"negative": 0, "positive": 1})
        
        train_df = train_df.fillna(" ", axis=1)
        test_df = test_df.fillna(" ", axis=1)

        # Load the tfidf vectorizer
        tfv = pickle.load(open(self.tfv_artifact_path, "rb"))

        # Transform the training and test data
        xtrain_tfv = tfv.transform(train_df["text"])
        ytrain = train_df["airline_sentiment"].values

        xtest_tfv = tfv.transform(test_df["text"])
        ytest = test_df["airline_sentiment"].values

        ytrain = ytrain.astype('int')
        ytest = ytest.astype('int')

        # Load the model
        clf = LogisticRegression(
            C=self.C,
            l1_ratio=self.l1_ratio,
            solver="liblinear",
            random_state=self.random_state)

        clf.fit(xtrain_tfv, ytrain)
        preds = clf.predict(xtest_tfv)
        pred_proba = clf.predict_proba(xtest_tfv)

        (accuracy, f1, roc_auc, log_loss_score) = self._eval_metrics(ytest, preds, pred_proba)
        
        print("-" * 50)
        print(f"  ACCURACY: {accuracy}")
        print(f"  F1: {f1}")
        print(f"  ROC AUC: {roc_auc}")
        print(f"  LOG LOSS: {log_loss_score}")
        print("-" * 50)

    def _eval_metrics(self, actual, pred, pred_proba):
        """ Takes in the ground truth labels, predictions labels, and prediction probabilities.
            Returns the accuracy, f1, auc_roc, log_loss scores.
        """
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred_proba[:, 1])
        log_loss_score = log_loss(actual, pred_proba)
        return accuracy, f1, roc_auc, log_loss_score

          

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    Predict(config_path=parsed_args.config).test()