# -*- coding: utf-8 -*-
from src.utils import Utils
from nltk.tokenize import word_tokenize
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from src.dispatcher import Dispatcher

nltk.download("punkt")


class BuildFeatures:
    """BuildFeatures class to take in train and validation features and perform feature engineering"""

    def __init__(self, config_path):
        self.config_path = config_path
        config = Utils().read_params(config_path)
        self.clean_folds_path = config["clean_dataset"]["clean_folds_path"]
        self.clean_test_path = config["clean_dataset"]["clean_test_path"]
        self.artifact_path = config["build_features"]["artifact_path"]

    def build_features_train(self, fold_num):
        """Performs feature engineering to the folds data from (../processed) into
        features ready to be trained by a model (returned in the function).
        """
        # Load the clean data
        df = Utils().get_data(self.clean_folds_path)

        df.fillna(" ", inplace=True)

        xtrain = df[df["kfold"] != fold_num]["text"]
        ytrain = df[df["kfold"] != fold_num]["airline_sentiment"]

        xvalid = df[df["kfold"] == fold_num]["text"]
        yvalid = df[df["kfold"] == fold_num]["airline_sentiment"]

        # Create a tfidf vectorizer
        vec = Dispatcher(self.config_path).dispatch_text_vectorizer("tfidf")

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        vec.fit(list(xtrain) + list(xvalid))

        if fold_num == 0:
            pickle.dump(vec, open(self.artifact_path, "wb"))

        xtrain_vec = vec.transform(xtrain)
        xvalid_vec = vec.transform(xvalid)

        return xtrain_vec, ytrain, xvalid_vec, yvalid

    def build_features_test(self):
        """Performs feature engineering to the test data from (../processed) into
        features ready to be trained by a model (returned in the function).
        """
        # Load the clean data
        df = Utils().get_data(self.clean_test_path)

        df.fillna(" ", inplace=True)

        xtest = df["text"]
        ytest = df["airline_sentiment"]

        # Load TF-IDF
        vec = pickle.load(open(self.artifact_path, "rb"))

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        xtest_vec = vec.transform(list(xtest))

        return xtest_vec, ytest

    def _build_features_text(self, text: str):
        """Runs feature engineering scripts to turn the text given as input,
        into features ready to be trained by a model (returned in the function).
        """
        vec = pickle.load(open(self.artifact_path, "rb"))

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        text_vec = vec.transform([text])
        return text_vec
