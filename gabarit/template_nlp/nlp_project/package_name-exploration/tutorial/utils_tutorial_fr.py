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
import random
from typing import List, Tuple
from collections import Counter


def text_to_sentence(text: str, min_sentence_size: int, min_sentence_word: int) -> List[str]:
    '''Transforms a text in sentences.

    Args:
        text (str) : The text to cut in sentences
        min_sentence_size (int) : The minimal number of characters in a sentence for it to be considered a sentence
        min_sentence_word (int) : The minimal number of words in a sentence for it to be considered a sentence
    Returns:
        list : A list of sentence
    
    '''
    text = re.sub(r'\s',' ', text)
    # Changes all 'strong' punctuations to a period
    text = re.sub(r'\!', r'.', text)
    text = re.sub(r'\?', r'.', text)
    text = re.sub(r'\;', r'.', text)
    # Get rid of superfluous whitespaces
    text = re.sub(' +', ' ', text)
    list_sentences = text.split('.')
    list_sentences = [sentence for sentence in list_sentences if len(sentence) >= min_sentence_size]
    list_sentences = [sentence for sentence in list_sentences if len(sentence.split(' ')) >= min_sentence_word]
    return list_sentences


def predict_author(text: str, model, min_sentence_size: int, min_sentence_word: int, perc_sample: float=1.0) -> Tuple[str, dict, int]:
    '''Predicts the author of a text.
    
    Args:
        text (str) : The text whose author we want to predict
        min_sentence_size (int) : The minimal number of characters in a sentence for it to be considered a sentence
        min_sentence_word (int) : The minimal number of words in a sentence for it to be considered a sentence
        perc_sample (float): The percentage of sentence of the text we consider
    Returns:
        tuple :
            str : The predicted author
            dict : The percentage of sentences attributed to each author
            int : The number of sentences in the text
    '''
    # Cut the text in sentences
    sentences = text_to_sentence(text, min_sentence_size, min_sentence_word)
    sentences = random.sample(sentences, k=int(perc_sample*len(sentences)))
    # For each sentence, predict an author. Gives the number of sentences predicted for each author
    counter = dict(Counter(list(model.predict(sentences))))
    # The author with the highest number of sentences
    author = max(counter, key=counter.get)
    # Calculates a percentage of sentences instead of raw numbers
    count_sentences = [(key, round(value/len(sentences), 3)) for key, value in counter.items()]
    count_sentences = sorted(count_sentences, key=lambda x:x[1], reverse=True)
    return author, count_sentences, len(sentences)