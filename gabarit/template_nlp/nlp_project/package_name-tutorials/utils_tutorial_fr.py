#!/usr/bin/env python3

## Fichier utile pour formation NLP
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

import re
import math
import random
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Union

from {{package_name}} import utils
from {{package_name}}.models_training.model_tfidf_svm import ModelTfidfSvm


def text_to_sentences(text: str, nb_word_sentence: int) -> List[str]:
    '''Transforms a text in sentences.

    Args:
        text (str) : The text to cut in sentences
        nb_word_sentence (int) : The number of words in a sentence
    Returns:
        list : A list of sentences
    '''
    text = re.sub(r'\s',' ', text)
    # Changes some punctuations to a whitespace
    for punctuation in [r'\!', r'\?', r'\;', r'\.', r'\,']:
        text = re.sub(punctuation, ' ', text)
    # Get rid of superfluous whitespaces
    text = re.sub(' +', ' ', text)
    list_mots = text.split(' ')
    list_sentences = []
    # Cut the text in sentences
    for i in range(math.ceil(len(list_mots)/nb_word_sentence)):
        list_sentences.append(' '.join(list_mots[i*nb_word_sentence:(i+1)*nb_word_sentence]))
    return list_sentences


def df_texts_to_df_sentences(texts: Union[list, pd.Series], authors: Union[list, pd.Series], nb_word_sentence:int) -> pd.DataFrame:
    '''Creates a dataframe containaing 'sentences' from a collection of texts with their corresponding authors.

    Args:
        texts (list, pd.Series) : a collection of texts
        authors (list, pd.Series) : the corresponding authors
        nb_word_sentence (int) : The number of words in a sentence
    Returns:
        pd.DataFrame : A dataframe containing sentences extracted from the initial texts with their corresponding author
    '''
    list_phrases = []
    for text, author in zip(list(texts), list(authors)):
        # This function transforms a text in 'sentences'
        sentences = text_to_sentences(text, nb_word_sentence)
        list_phrases = list_phrases+[(sentence, author) for sentence in sentences]
    df_sentences = pd.DataFrame(list_phrases, columns=['sentence', 'author'])
    return df_sentences


class ModelAuthor(ModelTfidfSvm):

    _default_name = 'model_author'

    def __init__(self, nb_word_sentence: int=300, **kwargs):
        super().__init__(**kwargs)
        self.nb_word_sentence = nb_word_sentence

    def fit(self, x_train, y_train, **kwargs) -> None:
        df_train_sentences = df_texts_to_df_sentences(x_train, y_train, self.nb_word_sentence)
        super().fit(df_train_sentences['sentence'], df_train_sentences['author'])

    def get_nb_sentences_author(self, text:str) -> dict:
        '''Predicts the author of a text.
        
        Args:
            text (str) : The text whose author we want to predict
        Returns:
            dict : the number of sentences corresponding for each author
        '''
        # Cut the text in sentences
        sentences = text_to_sentences(text, self.nb_word_sentence)
        # For each sentence, predict an author. Gives the number of sentences predicted for each author
        counter = dict(Counter(list(super().predict(sentences))))
        return counter

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict(self, x_test, return_proba=False, **kwargs):
        list_predictions = []
        for text in x_test:
            # Get the number of sentences for each author
            nb_sentences_author = self.get_nb_sentences_author(text)
            if return_proba:
                # Calculates probability for each author : percentage of sentences attributed to this author
                nb_sentences = sum(nb_sentences_author.values())
                list_probas = [nb_sentences_author.get(author, 0)/nb_sentences for author in self.list_classes]
                list_predictions.append(list_probas)
            else:
                # Get the author with the highest number of sentences
                author = max(nb_sentences_author, key=nb_sentences_author.get)
                list_predictions.append(author)
        return np.array(list_predictions)


