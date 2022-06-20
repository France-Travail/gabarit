#!/usr/bin/env python3

## Preprocessing functions
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


import logging
import pandas as pd
from typing import Callable
from words_n_fun.preprocessing import api
from words_n_fun import utils as wnf_utils

# Get logger
logger = logging.getLogger(__name__)


def get_preprocessors_dict() -> dict:
    '''Gets a dictionary of available preprocessing

    Returns:
        dict: Dictionary of preprocessing
    '''
    preprocessors_dict = {
        'no_preprocess': lambda x: x,  # - /!\ DO NOT DELETE -> necessary for compatibility /!\ -
        'preprocess_P1': preprocess_sentence_P1,  # Example of a preprocessing
        #  'preprocess_P2': preprocess_sentence_P2 , ETC ...
    }
    return preprocessors_dict


def get_preprocessor(preprocess_str: str) -> Callable:
    '''Gets a preprocessing (function) from its name

    Args:
        preprocess_str (str): Name of the preprocess
    Raises:
        ValueError: If the name of the preprocess is not known
    Returns:
        Callable: Function to be used for the preprocessing
    '''
    # Process
    preprocessors_dict = get_preprocessors_dict()
    if preprocess_str not in preprocessors_dict.keys():
        raise ValueError(f"The preprocess {preprocess_str} is not known.")
    # Get preprocessor
    preprocessor = preprocessors_dict[preprocess_str]
    # Return
    return preprocessor


@wnf_utils.data_agnostic
@wnf_utils.regroup_data_series
def preprocess_sentence_P1(docs: pd.Series) -> pd.Series:
    '''Applies "default" preprocess to a list of documents (text)

    Args:
        docs (pd.Series): Documents to be preprocessed
    Returns:
        pd.Series: Preprocessed documents
    '''
    pipeline = ['remove_non_string', 'get_true_spaces', 'remove_punct', 'to_lower', 'trim_string',
                'remove_leading_and_ending_spaces']
    return api.preprocess_pipeline(docs, pipeline=pipeline, chunksize=100000)


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
