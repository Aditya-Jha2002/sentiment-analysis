# -*- coding: utf-8 -*-
from src.utils import Utils
from nltk.tokenize import word_tokenize
import argparse
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download("punkt")

class BuildFeatures():
    """BuildFeatures class to take in train and validation features and perform feature engineering"""
    
    def __init__(self, df_path, train_idx, val_idx):
        self.df_path = df_path
        self.train_idx = train_idx
        self.val_idx = val_idx

    def build_features(self):
        """ Runs feature engineering scripts to turn the cleaned data from (../interm) into
            features ready to be trained by a model (saved in ../interim).
        """
        # Load the clean data
        df = Utils().get_data(self.clean_data_path)

        df.fillna("", inplace=True)

        # Create a tfidf vectorizer
        tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

        # Fit the vectorizer to the text
        tfv.fit(df.loc[:, "text"])

        # Transform the text
        df.loc[:,"text"] = tfv.transform(df.loc[:, "text"])
                
        # Save the feature data
        df.to_csv(self.feature_data_path, sep=",", index=False)


    def _tokenize(self, text):
        """ Tokenize the text """
        return word_tokenize(str(text))