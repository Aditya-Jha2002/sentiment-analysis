# -*- coding: utf-8 -*-
from src.utils import Utils
from nltk.tokenize import word_tokenize
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download("punkt")

class BuildFeatures():
    """BuildFeatures class to take in train and validation features and perform feature engineering"""
    
    def __init__(self, df_path, fold_num, store_tfv = False):
        self.df_path = df_path
        self.fold_num = fold_num
        self.store_tfv = store_tfv

    def build_features(self):
        """ Performs feature engineering to the data from (../interm) into
            features ready to be trained by a model (returned in the function).
        """
        # Load the clean data
        df = Utils().get_data(self.df_path)

        xtrain = df[df['fold'] != self.fold_num]['text']
        ytrain = df[df['fold'] != self.fold_num]['airline_sentiment']

        xvalid = df[df['fold'] == self.fold_num]['text']
        yvalid = df[df['fold'] == self.fold_num]['airline_sentiment']

        # Create a tfidf vectorizer
        tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        tfv.fit(list(xtrain) + list(xvalid))

        if self.store_tfv:
            pickle.dump(tfv, open(self.tfv_artifact_path, 'wb'))

        xtrain_tfv =  tfv.transform(xtrain) 
        xvalid_tfv = tfv.transform(xvalid)

        return xtrain_tfv, ytrain, xvalid_tfv, yvalid, tfv

    def _build_features_test(self, text, tfv_artifact_path):
        """ Runs feature engineering scripts to turn the cleaned data from (../interm) into
            features ready to be trained by a model (saved in ../interim).
        """
        xtest = text

        tfv = pickle.load(open(tfv_artifact_path, 'rb'))

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        xtest_tfv =  tfv.transform(xtest) 
        return xtest_tfv