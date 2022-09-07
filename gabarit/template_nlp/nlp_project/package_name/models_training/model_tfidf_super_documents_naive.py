#!/usr/bin/env python3

## Model TFIDF Super Documents Naive

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
# - ModelTfidfSuperDocumentsNaive -> Model for predictions TF-IDF naive with super documents
#
# Super documents collects all documents and concatenate them by label.
# Unlike standard tfidf model fitting with [n_samples, n_terms],
# Super documents fits with [n_label, n_terms] and transforms with [n_samples, n_terms].
#
# Model_super_documents adds each term's tfidf*count for each document and returns the label with the highest value


import os
import json
import pickle
import logging
import numpy as np
from typing import Union

from sklearn.pipeline import Pipeline

from {{package_name}} import utils
from {{package_name}}.models_training.model_pipeline import ModelPipeline
from {{package_name}}.models_training.utils_super_documents import TfidfVectorizerSuperDocuments


class ModelTfidfSuperDocumentsNaive(ModelPipeline):
    '''Model for predictions via TF-IDF + Naive'''

    _default_name = 'model_tfidf_super_documents_naive'

    def __init__(self, tfidf_params: Union[dict, None] = None,
                 multiclass_strategy: Union[str, None] = None, **kwargs):
        '''Initialization of the class (see ModelPipeline & ModelClass for more arguments)

        Kwargs:
            tfidf_count_params (dict): Parameters for the countVectorize
            tfidf_params (dict): Parameters for the tfidf TfidfVectorizerSuperDocuments
            multiclass_strategy (str): Multi-classes strategy, only can be None
        Raises:
            ValueError: If multiclass_strategy is not 'ovo', 'ovr' or None
            ValueError: If multi_label is True
        '''
        if multiclass_strategy is not None and multiclass_strategy not in ['ovo', 'ovr']:
            raise ValueError(f"The value of 'multiclass_strategy' must be 'ovo' or 'ovr' (not {multiclass_strategy})")
        # Init.
        super().__init__(**kwargs)
        self.with_super_documents = True

        if self.multi_label:
            raise ValueError("Model_tfidf_super_documents_naive does not support multi-labels")

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Manage model
        if tfidf_params is None:
            tfidf_params = {}
        self.tfidf = TfidfVectorizerSuperDocuments(**tfidf_params)

        # Can't do multi-labels / multi-classes
        if not self.multi_label:
            # If not multi-classes : no impact
            if multiclass_strategy in ['ovr', 'ovo']:
                raise ValueError("The TFIDF naive super_documents can't do", self.multiclass_strategy)
            else:
                self.pipeline = Pipeline([('tfidf', self.tfidf)])

    @utils.trained_needed
    def predict(self, x_test, return_proba: bool = False, **kwargs) -> np.ndarray:
        '''Predictions on test set

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples]
        Kwargs:
            return_proba (bool): If the function should return the probabilities instead of the classes (Keras compatibility)
        Returns:
            (np.ndarray): Array, shape = [n_samples]
            return_proba (np.ndarray): Array, shape = [n_samples, n_train]
        '''
        if return_proba:
            return self.predict_proba(x_test)
        else:
            return self.compute_predict(x_test)

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Predicts the probabilities on the test set
        - /!\\ THE MODEL NAIVE DOES NOT RETURN PROBABILITIES, HERE WE NORMALIZE TFIDF /!\\ -

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples]
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        if not self.multi_label:
            # Normalize tfidf to "proba"
            trans = self.tfidf.transform(x_test).toarray()
            probas = np.array([col/sum(col) for col in trans])
        else:
            raise ValueError("The TFIDF Naive does not support multi label")
        return probas

    @utils.trained_needed
    def compute_predict(self, x_test) -> np.ndarray:
        '''Compute the scores for the prediction

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples]
        Returns:
            (np.ndarray): Array, shape = [n_samples]
        Raise:
            if self.matrix_train == None
        '''
        # if self.matrix_train is None:
        if self.tfidf.tfidf_super_documents is None:
            raise AttributeError('The tfidf not fitted')
        x_test = np.array([x_test]) if isinstance(x_test, str) else x_test
        x_test = np.array(x_test) if isinstance(x_test, list) else x_test

        trans = self.tfidf.transform(x_test).toarray().T
        index = np.argmax(trans, axis=0)
        predicts = self.tfidf.classes_[index]

        return predicts

    def reload_from_standalone(self, **kwargs):
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            sklearn_pipeline_path (str): Path to standalone pipeline
            count_vectorizer_path (str): Path to countVectorizer (only with_super_documents = True)
            tfidf_super_documents_path (str): Path to tfidf super documents (only with_super_documents = True)
        Raises:
            ValueError: If configuration_path is None
            ValueError: If sklearn_pipeline_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object sklearn_pipeline_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        sklearn_pipeline_path = kwargs.get('sklearn_pipeline_path', None)
        count_vectorizer_path = kwargs.get('count_vectorizer_path', None)
        tfidf_super_documents_path = kwargs.get('tfidf_super_documents_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if sklearn_pipeline_path is None:
            raise ValueError("The argument sklearn_pipeline_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(sklearn_pipeline_path):
            raise FileNotFoundError(f"The file {sklearn_pipeline_path} does not exist")

        # Get the path of the super document if it is not given
        dir = os.path.split(sklearn_pipeline_path)[:-1]
        if count_vectorizer_path == None:
            count_vectorizer_path = os.path.join(dir[0], f"count_vectorizer.pkl")
        if tfidf_super_documents_path == None:
            tfidf_super_documents_path = os.path.join(dir[0], "tfidf_super_documents.pkl")

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
                          'with_super_documents']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload pipeline
        with open(sklearn_pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)

        # Reload pipeline elements
        self.tfidf = self.pipeline['tfidf']

        # Reload utile super documents
        if self.with_super_documents:
            self.tfidf.reload_from_standalone(count_vectorizer_path=count_vectorizer_path, tfidf_super_documents_path=tfidf_super_documents_path)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")