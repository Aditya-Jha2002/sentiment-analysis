# -*- coding: utf-8 -*-
import re
import argparse
from src import utils
import string
import nltk
import emoji

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

class DataLoader():
    """DataLoader class to load the data and preprocess it
    """
    def __init__(self, config_path):
        config = utils.Utils().read_params(config_path)
        self.cvfolds_data_path = config["cv_folds_dataset"]["folds_data_path"]
        self.clean_data_path = config["clean_dataset"]["clean_data_path"]

    def clean_dataset(self):
        """ Runs preprocessing scripts to turn folds data from (../interim) into
            cleaned and pre-processed data ready to be feature engineered on (saved in ../interim).
        """
        # Load the raw data
        df = utils.Utils().get_data(self.cvfolds_data_path)
        # Preprocess the text
        df['text'] = df['text'].apply(self.preprocess_text)
        # Label encode the labels
        df['airline_sentiment'] = df['airline_sentiment'].map({'negative': 0, 'positive': 1})

        # Save the clean data
        df.to_csv(self.clean_data_path, sep=",", index=False)

    def preprocess_text(self, text: str) -> str:
        """Preprocess text"""
        text = self._remove_space(text)
        text = text.lower()
        text = self._remove_contractions(text)
        text = self._remove_mentions(text)
        text = self._remove_emoji(text)
        text = self._remove_urls(text)
        text = self._remove_stopwords(text)
        text = self._remove_punctuation(text)
        text = self._stem_words(text)
        
        
        return text

    def _remove_space(self, text: str) -> str:
        """To remove weird spaces from text"""
        text = text.strip()
        text = text.split()
        return " ".join(text)

    def _remove_contractions(self, text: str) -> str:
        """To remove the contractions like shan't and convert them to shall not"""
        for key in utils.contractions.keys():
            text = text.replace(key, utils.contractions[key])
        return text

    def _remove_mentions(self, text: str) -> str:
        """To remove the mentions from text"""
        text = re.sub(r'@[^ ]+', '', text)
        return text

    def _remove_emoji(self, text: str) -> str:
        """To remove the emoji from text"""
        text = emoji.demojize(text)
        return text

    def _remove_urls(self, text: str) -> str:
        """To remove the urls from text"""
        text = re.sub(r'http\S+', '', text)
        return text
    
    def _remove_stopwords(self, text: str) -> str:
        """To remove the stopwords"""
        STOPWORDS = set(stopwords.words('english'))
        text = [word for word in str(text).split() if word not in STOPWORDS]
        return " ".join(text)

    def _remove_punctuation(self, text: str) -> str:
        """To remove the punctuations like !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text
    
    def _stem_words(self, text: str) -> str:
        """To stem the words"""
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text.split()]
        return " ".join(text)
 
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    DataLoader(config_path=parsed_args.config).clean_dataset()