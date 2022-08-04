#!/usr/bin/env python3

## Classe modèle pour le tutoriel Jules Verne
# Copyright (C) <2018-2022>  <Agence Data Services, DSI Pôle Emploi>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
import logging
import os
import re
import math
import numpy as np
import pandas as pd
from typing import List, Union
from collections import Counter

from {{package_name}}.models_training.model_tfidf_svm import ModelTfidfSvm


class ModelAuthor(ModelTfidfSvm):

    _default_name = 'model_author'

    def __init__(self, nb_word_sentence: int = 50, **kwargs) -> None:
        '''Initialization of the class (see ModelTfidfSvm & ModelPipeline & ModelClass for more arguments)

        Kwargs:
            nb_word_sentence (int) : Number of words for each sentence
        Raises:
            ValueError: If multilabel task
        '''
        super().__init__(**kwargs)
        self.nb_word_sentence = nb_word_sentence  # Number of words per sentence
        # We do not manage multilabel task
        if self.multi_label:
            raise ValueError("On ne gère pas les cas multilabel")

    def fit(self, x_train, y_train, **kwargs) -> None:
        '''Trains the model
           **kwargs permits compatibility with Keras model

        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train (?): Array-like, shape = [n_samples, n_targets]
        Raises:
            RuntimeError: If the model is already fitted
        '''
        if self.trained:
            self.logger.error("We can't train again a pipeline sklearn model")
            self.logger.error("Please train a new model")
            raise RuntimeError("We can't train again a pipeline sklearn model")

        # Split texts into sentence / author pairs
        df_train_sentences = df_texts_to_df_sentences(x_train, y_train, nb_word_sentence=self.nb_word_sentence)
        new_x_train = df_train_sentences['sentence']
        new_y_train = df_train_sentences['author']
        # Fit
        super().fit(new_x_train, new_y_train)

    def predict(self, x_test, return_proba: bool = False, **kwargs) -> np.ndarray:
        '''Predictions on test set

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
        Kwargs:
            return_proba (bool): If the function should return the probabilities instead of the classes (Keras compatibility)
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        # Process texts one by one
        list_predictions = []
        for text in x_test:
            # Get predictions for each text' sentences
            sentences = text_to_sentences(text, nb_word_sentence=self.nb_word_sentence)
            predictions = super().predict(sentences)
            # Count nb of occurrences for each author
            counter_author_predictions = dict(Counter(list(predictions)))

            # If we do not return probabilites, just add predicted author (max count)
            if not return_proba:
                predicted_author = max(counter_author_predictions, key=counter_author_predictions.get)
                list_predictions.append(predicted_author)  # On ajoute l'auteur predit à la liste
            else:
                # We consider probabilities as the percentage of sentences predicted
                nb_sentences = len(sentences)
                list_probas = [counter_author_predictions.get(author, 0) / nb_sentences for author in self.list_classes]
                list_predictions.append(list_probas)

        # Return
        return np.array(list_predictions)

    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Probabilities predicted on the test set

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        return self.predict(x_test, return_proba=True, **kwargs)

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save model
        if json_data is None:
            json_data = {}
        json_data['nb_word_sentence'] = self.nb_word_sentence

        # Save
        super().save(json_data=json_data)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            sklearn_pipeline_path (str): Path to standalone pipeline
        Raises:
            ValueError: If configuration_path is None
            ValueError: If sklearn_pipeline_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object sklearn_pipeline_path is not an existing file
        '''
        super().reload_from_standalone(**kwargs)
        configuration_path = kwargs.get('configuration_path', None)
        with open(configuration_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.nb_word_sentence = configs.get('nb_word_sentence', self.nb_word_sentence)

### Fonctions utilitaires


def text_to_sentences(text: str, nb_word_sentence: int = 10) -> List[str]:
    '''Transforms a text in sentences.

    Args:
        text (str): The text to cut in sentences
    Kwargs:
        nb_word_sentence (int): The number of words in a sentence
    Returns:
        list: A list of sentences
    '''
    text = re.sub(r'\s', ' ', text)
    # Change some punctuations to a whitespace
    for punctuation in [r'\!', r'\?', r'\;', r'\.', r'\,', r'\:']:
        text = re.sub(punctuation, ' ', text)
    # Get rid of superfluous whitespaces
    text = re.sub(' +', ' ', text)
    words_list = text.split(' ')
    list_sentences = []
    # Cut the text in sentences
    for i in range(math.ceil(len(words_list) / nb_word_sentence)):
        list_sentences.append(' '.join(words_list[i * nb_word_sentence: (i + 1) * nb_word_sentence]))
    return list_sentences


def df_texts_to_df_sentences(texts: Union[list, pd.Series], authors: Union[list, pd.Series], nb_word_sentence: int = 10) -> pd.DataFrame:
    '''Creates a dataframe containaing 'sentences' from a collection of texts with their corresponding authors.

    Args:
        texts (list, pd.Series): a collection of texts
        authors (list, pd.Series): the corresponding authors
    Kwargs:
        nb_word_sentence (int): The number of words in a sentence
    Returns:
        pd.DataFrame: A dataframe containing sentences extracted from the initial texts with their corresponding author
    '''
    list_sentences = []
    for text, author in zip(list(texts), list(authors)):
        # This function transforms a text in 'sentences'
        sentences = text_to_sentences(text, nb_word_sentence)
        list_sentences = list_sentences + [(sentence, author) for sentence in sentences]
    df_sentences = pd.DataFrame(list_sentences, columns=['sentence', 'author'])
    return df_sentences


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
