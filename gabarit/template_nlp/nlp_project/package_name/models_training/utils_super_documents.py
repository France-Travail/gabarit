#!/usr/bin/env python3

## Utils for tfidf super documents
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
#
# Classes :
# - TfidfTransformerSuperDocuments -> TfidfTransformer for super documents
# - TfidfVectorizerSuperDocuments -> TfidfVectorizer for super documents
#
# Super documents collects all documents and concatenate them by label.
# Unlike standard tfidf model fitting with [n_samples, n_terms],
# Super documents fits with [n_label, n_terms] and transforms with [n_samples, n_terms].

from __future__ import annotations

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Union

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

logger = logging.getLogger(__name__)

def get_super_documents(x_train, y_train) -> tuple[np.array, np.array]:
    '''Transform the documents to super documents

    Args:
        x_train (?): Array-like, shape = [n_samples, n_targets]
        y_train (?): Array-like, shape = [n_samples, n_targets]
    Returns:
        super_train(np.array): array, shape = [n_targets]
        y_train(np.array): array, shape = [n_targets]
    '''
    x_train = pd.Series(x_train, name='x_train')
    y_train = pd.Series(y_train, name='y_train')

    df_train = pd.concat([x_train, y_train], axis=1)
    super_train = df_train.groupby(y_train.name, sort=False).agg({x_train.name: lambda sentence: ' '.join((sentence))})

    return np.array(super_train[x_train.name]), np.array(super_train.index)

class TfidfTransformerSuperDocuments(TfidfTransformer):
    '''TfidfTransformer for super documents'''

    def get_super_documents_count_vectorizer(self, x_train: csr_matrix, y_train) -> tuple[np.array, np.array]:
        '''Transform the document to super document

        Args:
            x_train (csr_matrix): shape = [n_samples, n_term]
            y_train (?): Array-like, shape = [n_samples, n_targets]
        Returns:
            result(np.array): array, shape = [n_targets, n_term]
            y_train(np.array): array, shape = [n_targets]
        '''
        index_array = np.array([np.where(y_train == x)[0] for x in np.unique(y_train)], dtype=object)
        result = np.array([[sum(y) for y in x_train[x, :].transpose().toarray()] for x in index_array])
        return result, np.unique(y_train)

    def fit_transform(self, raw_documents, y=None) -> csr_matrix:
        '''Trains and transform the model with super documents

        Args:
            raw_documents (?): Array-like, shape = [n_samples, n_targets]
            y (?): Array-like, shape = [n_samples, n_targets]
        Returns:
            (csr_matrix): matrix, shape = [n_samples, n_term]
        '''
        raw_super_documents, target_super_documents = self.get_super_documents_count_vectorizer(raw_documents, y)
        self.fit(raw_super_documents, y)
        return self.transform(raw_documents)


class TfidfVectorizerSuperDocuments(TfidfVectorizer):
    '''TfidfVectorize for super documents'''

    def __init__(self, tfidf_super_documents: Union[np.array, None] = None, **kwargs) -> None:
        '''Initialization of the class

        Args:
            tfidf_super_documents (np.array): shape = [n_terme, n_label]
        '''
        # Init.
        super().__init__(**kwargs)
        self.tfidf_super_documents = tfidf_super_documents
        self.count_vec = CountVectorizer(**kwargs)


    def fit(self, raw_documents, y=None) -> TfidfVectorizerSuperDocuments:
        '''Trains the model with super documents

        Args:
            raw_documents (?): Array-like, shape = [n_samples, n_targets]
            y (?): Array-like, shape = [n_samples, n_targets]
        Returns:
            TfidfVectorizerSuperDocuments
        '''
        array_super_documents, _ = get_super_documents(raw_documents, y)
        super().fit(array_super_documents)
        self.tfidf_super_documents = super().transform(array_super_documents).toarray().T
        self.count_vec.fit(array_super_documents)
        return self

    def transform(self, raw_documents, y=None) -> csr_matrix:
        '''transform the model with super documents calculations

        Args:
            raw_documents (?): Array-like, shape = [n_samples, n_targets]
            y (?): Array-like, shape = [n_samples, n_targets]
        Returns:
            (csr_matrix): matrix, shape = [n_samples, n_term]
        '''
        count = self.count_vec.transform(raw_documents).toarray()
        return csr_matrix(np.dot(count, self.tfidf_super_documents))

    def fit_transform(self, raw_documents, y=None) -> csr_matrix:
        '''Trains and transform the model with super documents

        Args:
            raw_documents (?): Array-like, shape = [n_samples, n_targets]
            y (?): Array-like, shape = [n_samples, n_targets]
        Returns:
            (csr_matrix): matrix, shape = [n_samples, n_term]
        '''
        self.fit(raw_documents, y)
        return self.transform(raw_documents)

    def save(self, dir: dict, level_save: str) -> None:
        '''Saves the model

        Kwargs:
            dir (dict): Folder where to save
            level_save (str): Level of saving
        '''
        if not os.path.exists(dir):
            os.mkdir(dir)

        # Save CountVectorizer if wanted & self.count_vec is not None & level_save > 'LOW'
        if self.count_vec is not None and level_save in ['MEDIUM', 'HIGH']:
            pkl_path = os.path.join(dir, "count_vectorizer.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.count_vec, f)

        # Save array tfidf with super documents if wanted & self.tfidf_super_documents is not None & level_save > 'LOW'
        if self.tfidf_super_documents is not None and level_save in ['MEDIUM', 'HIGH']:
            pkl_path = os.path.join(dir, "tfidf_super_documents.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.tfidf_super_documents, f)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            count_vectorizer_path (str): Path to count vectorizer
            tfidf_super_documents_path (str): Path to tfidf super documents
        Raises:
            ValueError: If count_vectorizer_path is None
            ValueError: If tfidf_super_documents_path is None
            FileNotFoundError: If the object count_vectorizer_path is not an existing file
            FileNotFoundError: If the object tfidf_super_documents_path is not an existing file
        '''
        # Retrieve args
        count_vectorizer_path = kwargs.get('count_vectorizer_path', None)
        tfidf_super_documents_path = kwargs.get('tfidf_super_documents_path', None)

        # Checks
        if count_vectorizer_path is None:
            raise ValueError("The argument count_vectorizer_path can't be None")
        if tfidf_super_documents_path is None:
            raise ValueError("The argument tfidf_super_documents_path can't be None")
        if not os.path.exists(count_vectorizer_path):
            raise FileNotFoundError(f"The file {count_vectorizer_path} does not exist")
        if not os.path.exists(tfidf_super_documents_path):
            raise FileNotFoundError(f"The file {tfidf_super_documents_path} does not exist")

        with open(count_vectorizer_path, 'rb') as f:
            self.count_vec = pickle.load(f)
        with open(tfidf_super_documents_path, 'rb') as f:
            self.tfidf_super_documents = pickle.load(f)

if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")