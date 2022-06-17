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
import pandas as pd
from typing import List, Tuple
from collections import Counter

from {{package_name}}.models_training.model_tfidf_svm import ModelTfidfSvm


def text_to_sentence(text: str, nb_word_sentence: int) -> List[str]:
    '''Transforms a text in sentences.

    Args:
        text (str) : The text to cut in sentences
        nb_word_sentence (int) : The number of words in a sentence
    Returns:
        list : A list of sentence
    
    '''
    text = re.sub(r'\s',' ', text)
    # Changes some punctuations to a whitespace
    for punctuation in [r'\!', r'\?', r'\;', r'\.', r'\,']:
        text = re.sub(punctuation, ' ', text)
    # Get rid of superfluous whitespaces
    text = re.sub(' +', ' ', text)
    list_mots = text.split(' ')
    list_sentences = []
    for i in range(math.ceil(len(list_mots)/nb_word_sentence)):
        list_sentences.append(' '.join(list_mots[i*nb_word_sentence:(i+1)*nb_word_sentence]))
    return list_sentences

def df_texts_to_df_sentences(texts, author, nb_word_sentence):
    list_phrases = []
    for text, author in zip(list(texts), list(author)):
        # Cette fonction transforme un texte en phrases
        sentences = text_to_sentence(text, nb_word_sentence)
        list_phrases = list_phrases+[(sentence, author) for sentence in sentences]
    df_sentences = pd.DataFrame(list_phrases, columns=['sentence', 'author'])
    return df_sentences


class ModelAuthor(ModelTfidfSvm):

    def __init__(self, nb_word_sentence, **kwargs):
        super().__init__(**kwargs)
        self.nb_word_sentence = nb_word_sentence

    def fit(self, x_train, y_train, **kwargs) -> None:
        df_train_sentences = df_texts_to_df_sentences(x_train, y_train, self.nb_word_sentence)
        super().fit(df_train_sentences['sentence'], df_train_sentences['author'])

    def predict_author(self, text: str) -> Tuple[str, dict, int]:
        '''Predicts the author of a text.
        
        Args:
            text (str) : The text whose author we want to predict
        Returns:
            str : The predicted author
        '''
        # Cut the text in sentences
        sentences = text_to_sentence(text, self.nb_word_sentence)
        # For each sentence, predict an author. Gives the number of sentences predicted for each author
        counter = dict(Counter(list(super().predict(sentences))))
        # The author with the highest number of sentences
        author = max(counter, key=counter.get)
        return author

    def predict(self, x_test, **kwargs):
        return x_test.apply(self.predict_author)
