#!/usr/bin/env python3

## Modèle TFIDF Naive

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
# - ModelTfidfNaive -> Model for predictions TF-IDF naive


import os
import math
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime

from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from {{package_name}} import utils
from {{package_name}}.models_training import utils_models
from {{package_name}}.models_training.model_pipeline import ModelPipeline
from {{package_name}}.models_training.utils_super_documents import TfidfTransformerSuperDocuments


class ModelTfidfNaive(ModelPipeline):
    '''Model for predictions via TF-IDF + Naive'''

    _default_name = 'model_tfidf_naive'

    def __init__(self, tfidf_count_params: Union[dict, None] = None, tfidf_transformer_params: Union[dict, None] = None, 
                 multiclass_strategy: Union[str, None] = None, **kwargs):
        '''Initialization of the class (see ModelPipeline & ModelClass for more arguments)

        Kwargs:
            tfidf_transformer_params (dict): Parameters for the tfidf TfidfTransformer
            tfidf_count_params (dict): Parameters for the countVectorize
            multiclass_strategy (str): Multi-classes strategy, only can be None
            with_super_documents (bool): train model with super documents
        Raises:
            ValueError: If multiclass_strategy is not 'ovo', 'ovr' or None
        '''
        if multiclass_strategy is not None and multiclass_strategy not in ['ovo', 'ovr']:
            raise ValueError(f"The value of 'multiclass_strategy' must be 'ovo' or 'ovr' (not {multiclass_strategy})")
        # Init.
        super().__init__(**kwargs)
        self.with_super_documents = True

        if self.multi_label:
            raise ValueError("The TFIDF Naive does not support multi label")

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Manage model
        if tfidf_transformer_params is None:
            tfidf_transformer_params = {}
        self.tfidf = TfidfTransformerSuperDocuments(**tfidf_transformer_params)

        if tfidf_count_params is None:
            tfidf_count_params = {}
        self.tfidf_count = CountVectorizer(**tfidf_count_params)

        self.multiclass_strategy = multiclass_strategy
        self.matrix_train = csr_matrix((0,0))
        self.array_target = np.array([])

        # Can't do multi-labels / multi-classes
        if not self.multi_label:
            # If not multi-classes : no impact
            if multiclass_strategy == 'ovr':
                raise ValueError("The TFIDF Naive can't do ovr")
            elif multiclass_strategy == 'ovo':
                raise ValueError("The TFIDF Naive can't do ovo")
            else:
                self.pipeline = Pipeline([('tfidf_count', self.tfidf_count),('tfidf', self.tfidf)])


    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save model
        if json_data is None:
            json_data = {}

        # No need to save the parameters of the pipeline steps, it is already done in ModelPipeline
        json_data['multiclass_strategy'] = self.multiclass_strategy
        json_data['with_super_documents'] = self.with_super_documents
        json_data['classes_'] = self.tfidf.classes_ if hasattr(self.tfidf, 'classes_') else None

        np.savetxt(os.path.join(self.model_dir, 'matrix_train.csv'), self.matrix_train.toarray(), delimiter=";")
        np.savetxt(os.path.join(self.model_dir, 'array_target.csv'), self.array_target, delimiter=";", fmt="%s")

        # Save
        super().save(json_data=json_data)

    def reload_from_standalone(self, **kwargs):
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            sklearn_pipeline_path (str): Path to standalone pipeline
            matrix_train_path (str): Path to matrix_train file
            array_target_path (str): Path to array_target file
        Raises:
            ValueError: If configuration_path is None
            ValueError: If sklearn_pipeline_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object sklearn_pipeline_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        sklearn_pipeline_path = kwargs.get('sklearn_pipeline_path', None)
        matrix_train_path = kwargs.get('matrix_train_path', None)
        array_target_path = kwargs.get('array_target_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if sklearn_pipeline_path is None:
            raise ValueError("The argument sklearn_pipeline_path can't be None")
        if matrix_train_path is None:
            raise ValueError("The argument matrix_train_path can't be None")
        if array_target_path is None:
            raise ValueError("The argument array_target_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(sklearn_pipeline_path):
            raise FileNotFoundError(f"The file {sklearn_pipeline_path} does not exist")
        if not os.path.exists(matrix_train_path):
            raise FileNotFoundError(f"The file {matrix_train_path} does not exist")
        if not os.path.exists(array_target_path):
            raise FileNotFoundError(f"The file {array_target_path} does not exist")

        # Load confs
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        # Can't set int as keys in json, so need to cast it after reloading
        # dict_classes keys are always ints
        if 'dict_classes' in configs.keys():
            configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}
        elif 'list_classes' in configs.keys():
            configs['dict_classes'] = {i: col for i, col in enumerate(configs['list_classes'])}

        # Set class vars
        # self.model_name = # Keep the created name
        # self.model_dir = # Keep the created folder
        self.nb_fit = configs.get('nb_fit', 1)  # Consider one unique fit by default
        self.trained = configs.get('trained', True)  # Consider trained by default
        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['x_col', 'y_col',
                          'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'multiclass_strategy', 'with_super_documents']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload matrix_train and array_target
        self.matrix_train = csr_matrix(np.genfromtxt(matrix_train_path, delimiter=';'))
        self.array_target = np.genfromtxt(array_target_path, delimiter=';', dtype='str')

        # Reload pipeline
        with open(sklearn_pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)

        # Reload pipeline elements
        self.tfidf = self.pipeline['tfidf']
        self.tfidf_count = self.pipeline['tfidf_count']

    def fit(self, x_train, y_train, **kwargs):
        '''Trains the model
            Transform the document to super document when with_super_documents = True

           **kwargs permits compatibility with Keras model
        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train (?): Array-like, shape = [n_samples, n_targets]
        Raises:
            RuntimeError: If the model is already fitted
        '''
        self.tfidf.classes_ = list(np.unique(y_train))
        super().fit(x_train, y_train)
        x_count = self.pipeline['tfidf_count'].fit_transform(x_train)
        x_super, self.array_target = self.pipeline['tfidf'].get_super_documents_count_vectorizer(x_count, y_train)
        self.matrix_train = self.pipeline['tfidf'].transform(x_super)

    @utils.trained_needed
    def predict(self, x_test, return_proba: bool = False, **kwargs) -> np.ndarray:
        '''Predictions on test set

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
        Kwargs:
            return_proba (bool): If the function should return the probabilities instead of the classes (Keras compatibility)
        Raise:
            ValueError: if return_proba is True
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        x_test = np.array([x_test]) if isinstance(x_test, str) else x_test
        x_test = np.array(x_test) if isinstance(x_test, list) else x_test

        if return_proba:
            raise ValueError("The TFIDF Naive does not support return_proba")
        else:
            vec_counts = self.tfidf_count.transform(x_test)
            predicts = np.argmax(np.dot(vec_counts, self.matrix_train.transpose()).toarray(), axis=1)
            predicts = self.array_target[predicts]
            return predicts


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")