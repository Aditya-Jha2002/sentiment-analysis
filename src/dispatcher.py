import imp
from src.utils import Utils
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
from sklearn.feature_extraction import text

class Dispatcher:
    """
    Dispatcher class to map a given model to a sklearn model
    Example:
        model = Dispatcher(config_path).dispatch_model(model_name)

    Also maps a given vectorizer to a sklearn vectorizer
    Example:
        vectorizer = Dispatcher(config_path).dispatch_text_vectorizer(vectorizer_name)
    """
    def __init__(self, config_path):
        config = Utils().read_params(config_path)

        self.MODELS = {
            "naive_bayes": naive_bayes.MultinomialNB(**config["estimators"]["NaiveBayes"]["params"]),
            "logistic_regression": linear_model.LogisticRegression(**config["estimators"]["LogisticRegression"]["params"]),
            "decision_tree": tree.DecisionTreeClassifier(**config["estimators"]["DecisionTree"]["params"]),
            "random_forest": ensemble.RandomForestClassifier(**config["estimators"]["RandomForest"]["params"]),
            "xgboost": xgb.XGBClassifier(**config["estimators"]["XGBoost"]["params"]),
        }

        self.VECTORIZERS = {
            "tfidf": text.TfidfVectorizer(**config["vectorizers"]["TfidfVectorizer"]["params"]),
            "count": text.CountVectorizer(**config["vectorizers"]["CountVectorizer"]["params"]),
        }

    def dispatch_model(self, model_name):
        """Dispatches the model to train"""
        model = self.MODELS[model_name]
        return model
    
    def dispatch_text_vectorizer(self, vectorizer_name):
        """Dispatches the text vectorizer to use"""
        vectorizer = self.VECTORIZERS[vectorizer_name]
        return vectorizer