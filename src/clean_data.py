# -*- coding: utf-8 -*-
import re
import argparse
from src import utils
import string
import nltk
import emoji

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download("stopwords")

class DataLoader:
    """DataLoader class to load the data, and preprocess it"""

    def __init__(self, config_path):
        config = utils.Utils().read_params(config_path)
        self.folds_data_path = config["split_dataset"]["folds_data_path"]
        self.test_data_path = config["split_dataset"]["test_data_path"]
        self.clean_train_path = config["clean_dataset"]["clean_folds_path"]
        self.clean_test_path = config["clean_dataset"]["clean_test_path"]

        self.clean_dict = config["clean_dataset"]["clean_dict"]

    def clean_dataset(self, df_type: str):
        """Runs preprocessing scripts to turn data given from (../interim) into
        cleaned and pre-processed data ready to be feature engineered on (saved in ../processed).
        """
        if df_type == "train":
            initial_data_path = self.folds_data_path
            clean_data_path = self.clean_train_path

        elif df_type == "test":
            initial_data_path = self.test_data_path
            clean_data_path = self.clean_test_path

        # Load the raw data
        df = utils.Utils().get_data(initial_data_path)
        # Preprocess the text
        df["text"] = df["text"].apply(self._preprocess_text)
        # Label encode the labels
        df["airline_sentiment"] = df["airline_sentiment"].map(
            {"negative": 0, "positive": 1}
        )

        # Save the clean data
        df.to_csv(clean_data_path, sep=",", index=False)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess and clean text given"""
        if self.clean_dict["space"]:
            text = self._remove_space(text)
        if self.clean_dict["lower"]:
            text = text.lower()
        if self.clean_dict["contractions"]:
            text = self._remove_contractions(text)
        if self.clean_dict["mentions"]:
            text = self._remove_mentions(text)
        if self.clean_dict["emoji"]:
            text = self._remove_emoji(text)
        if self.clean_dict["urls"]:
            text = self._remove_urls(text)
        if self.clean_dict["stopwords"]:
            text = self._remove_stopwords(text)
        if self.clean_dict["punctuation"]:
            text = self._remove_punctuation(text)
        if self.clean_dict["stem"]:
            text = self._stem_words(text)
        if self.clean_dict["lemmatize"]:
            text = self._lemmatize_words(text)
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
        text = re.sub(r"@[^ ]+", "", text)
        return text

    def _remove_emoji(self, text: str) -> str:
        """To remove the emoji from text"""
        text = emoji.demojize(text)
        return text

    def _remove_urls(self, text: str) -> str:
        """To remove the urls from text"""
        text = re.sub(r"http\S+", "", text)
        return text

    def _remove_stopwords(self, text: str) -> str:
        """To remove the stopwords"""
        STOPWORDS = set(stopwords.words("english"))
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

    def _lemmatize_words(self, text: str) -> str:
        """To lemmatize the words"""
        lemmatizer = nltk.stem.WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text.split()]
        return " ".join(text)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    DataLoader(config_path=parsed_args.config).clean_dataset("train")
    DataLoader(config_path=parsed_args.config).clean_dataset("test")