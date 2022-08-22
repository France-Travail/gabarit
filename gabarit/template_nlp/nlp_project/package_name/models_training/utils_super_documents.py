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

import logging
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

logger = logging.getLogger(__name__)


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

    def get_super_documents(self, x_train, y_train) -> tuple[np.array, np.array]:
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
        super_train = df_train.groupby(y_train.name).agg({x_train.name: lambda sentence: ' '.join((sentence))})
        return np.array(super_train[x_train.name]), np.array(super_train.index)

    def fit(self, raw_documents, y=None) -> TfidfVectorizerSuperDocuments:
        '''Trains the model with super documents

        Args:
            raw_documents (?): Array-like, shape = [n_samples, n_targets]
            y (?): Array-like, shape = [n_samples, n_targets]
        Returns:
            TfidfVectorizerSuperDocuments
        '''
        raw_super_documents, target_super_documents = self.get_super_documents(raw_documents, y)
        super().fit(raw_super_documents)
        return self

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


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")