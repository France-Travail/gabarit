#!/usr/bin/env python3

## Fonctions utiles pour le tutoriel Jules Verne
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
import pandas as pd
from typing import List, Union


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
