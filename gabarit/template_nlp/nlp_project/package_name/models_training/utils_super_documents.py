#!/usr/bin/env python3

## Generic model for sklearn pipeline
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
# - ModelPipeline -> Generic model for sklearn pipeline

import os
import numpy as np
import pandas as pd
from typing import Union

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer


class TfidfTransformerSuperDocuments(TfidfTransformer):

    def get_super_documents_count_vectorizer(self, x_train, y_train, **kwargs):
        '''Trains the model
            Transform the document to super document when with_super_documents = True

        **kwargs permits compatibility with Keras model
        Args:
            x_train (csr_matrix)
            y_train (?): Array-like, shape = [n_samples, n_targets]
        '''
        index_array = np.array([np.where(y_train==x)[0] for x in np.unique(y_train)], dtype=object)
        result = np.array([[sum(y) for y in x_train[x,:].transpose().toarray()] for x in index_array])
        return result, np.unique(y_train)

    def fit_transform(self, raw_documents, y=None):
        raw_super_documents, target_super_documents = self.get_super_documents_count_vectorizer(raw_documents, y)
        self.fit(raw_super_documents, y)
        return self.transform(raw_documents)

class TfidfVectorizerSuperDocuments(TfidfVectorizer):
    '''TfidfVectorize for super documents'''

    def get_super_documents(self, x_train, y_train, **kwargs):
        '''Trains the model
            Transform the document to super document when with_super_documents = True

        **kwargs permits compatibility with Keras model
        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train (?): Array-like, shape = [n_samples, n_targets]
        Raises:
            RuntimeError: If the model is already fitted
        '''
        x_train = pd.Series(x_train, name='x_train')
        y_train = pd.Series(y_train, name='y_train')

        df_train = pd.concat([x_train, y_train], axis=1)
        super_train = df_train.groupby(y_train.name).agg({x_train.name:lambda sentence: ' '.join((sentence))})
        return super_train[x_train.name], super_train.index

    def fit(self, raw_documents, y=None):
        raw_super_documents, target_super_documents = self.get_super_documents(raw_documents, y)
        X = super().fit(raw_super_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents, y)
        return self.transform(raw_documents)
