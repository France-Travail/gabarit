#!/usr/bin/env python3

## Model TFIDF Cosine Similarity

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
# - ModelTfidfCos -> Model for predictions TF-IDF Cosine Similarity


import os
import math
import json
import pickle
import logging
import numpy as np
from typing import Union

from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from {{package_name}} import utils
from {{package_name}}.models_training.model_pipeline import ModelPipeline
from {{package_name}}.models_training.utils_super_documents import TfidfVectorizerSuperDocuments


class ModelTfidfCos(ModelPipeline):
    '''Model for predictions via TF-IDF + Cosine Similarity'''

    _default_name = 'model_tfidf_cos'

    def __init__(self, tfidf_params: Union[dict, None] = None, multiclass_strategy: Union[str, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelPipeline & ModelClass for more arguments)

        Kwargs:
            tfidf_params (dict): Parameters for the tfidf TfidfTransformer
            multiclass_strategy (str): Multi-classes strategy, only can be None
        Raises:
            ValueError: If multiclass_strategy is not 'ovo', 'ovr' or None
            ValueError: If with_super_documents and multi_label
        '''
        if multiclass_strategy is not None and multiclass_strategy not in ['ovo', 'ovr']:
            raise ValueError(f"The value of 'multiclass_strategy' must be 'ovo' or 'ovr' (not {multiclass_strategy})")
        # Init.
        super().__init__(**kwargs)

        if self.with_super_documents and self.multi_label:
            raise ValueError("The method with super documents does not support multi-labels")

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Manage model
        if tfidf_params is None:
            tfidf_params = {}
        self.tfidf = TfidfVectorizer(**tfidf_params) if not self.with_super_documents else TfidfVectorizerSuperDocuments(**tfidf_params)

        self.multiclass_strategy = multiclass_strategy
        self.matrix_train = None
        self.array_target = None

        # Can't do multi-labels / multi-classes
        if not self.multi_label:
            # If not multi-classes : no impact
            if multiclass_strategy in ['ovr', 'ovo']:
                raise ValueError("The TFIDF Cosine Similarity can't do", self.multiclass_strategy)
            else:
                self.pipeline = Pipeline([('tfidf', self.tfidf)])

    def fit(self, x_train, y_train, **kwargs) -> None:
        '''Trains the model

           **kwargs permits compatibility with Keras model
        Args:
            x_train (?): Array-like, shape = [n_samples]
            y_train (?): Array-like, shape = [n_samples]
        Raises:
            RuntimeError: If the model is already fitted
        '''
        self.tfidf.classes_ = list(np.unique(y_train))
        self.array_target = np.array(y_train)
        super().fit(x_train, y_train)
        self.matrix_train = self.pipeline.transform(x_train).astype(np.float16)

    @utils.trained_needed
    def predict(self, x_test, return_proba: bool = False, **kwargs) -> np.ndarray:
        '''Predictions
        - /!\\ THE MODEL COSINE SIMILARITY DOES NOT RETURN PROBABILITIES, HERE WE SIMULATE PROBABILITIES EQUAL COSINE SIMILARITY /!\\ -

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples]
        Kwargs:
            return_proba (bool): If the function should return the probabilities instead of the classes (Keras compatibility)
        Returns:
            (np.ndarray): Array, shape = [n_samples]
        '''
        if return_proba:
            return self.predict_proba(x_test)
        else:
            return self.predict_cosine_similarity(x_test)

    @utils.trained_needed
    def predict_cosine_similarity(self, x_test, return_cos: bool = False, **kwargs) -> np.ndarray:
        '''Predictions
        - /!\\ THE MODEL COSINE SIMILARITY DOES NOT RETURN PROBABILITIES, HERE WE SIMULATE PROBABILITIES EQUAL COSINE SIMILARITY /!\\ -

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples]
        Kwargs:
            return_proba (bool): If the function should return the probabilities instead of the classes (Keras compatibility)
        Returns:
            (np.ndarray): Array, shape = [n_samples]
            return_proba (np.ndarray): Array, shape = [n_samples, n_train]
        Raise:
            if self.matrix_train or self.array_target is None
        '''
        if self.matrix_train is None or self.array_target is None:
            raise AttributeError('your fit is not valid')

        x_test = np.array([x_test]) if isinstance(x_test, str) else x_test
        x_test = np.array(x_test) if isinstance(x_test, list) else x_test

        chunk_size = 5000
        vec = self.pipeline.transform(x_test).astype(np.float16)
        vec_size = math.ceil((vec.shape[0]) / chunk_size)
        train_size = math.ceil((self.matrix_train.shape[0]) / chunk_size)
        array_predicts = np.array([], dtype='int')
        for vec_row in range(vec_size):
            block_vec = vec[vec_row * chunk_size : (vec_row + 1) * chunk_size]
            list_cosine = []
            for train_row in range(train_size):
                block_train = self.matrix_train[train_row * chunk_size : (train_row + 1) * chunk_size]
                cosine = cosine_similarity(block_train, block_vec).astype(np.float16)
                list_cosine.append(cosine)
            concatenate_cosine = np.concatenate(list_cosine)
            array_predicts = np.append(array_predicts, np.argmax(concatenate_cosine, axis=0))

        if return_cos:
            return concatenate_cosine.T
        else:
            predicts = self.array_target[array_predicts]
            return predicts

    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Predictions
        - /!\\ THE MODEL COSINE SIMILARITY DOES NOT RETURN PROBABILITIES, HERE WE SIMULATE PROBABILITIES EQUAL COSINE SIMILARITY /!\\ -

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples]
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        Raise:
            if self.matrix_train or self.array_target is None
        '''
        if self.matrix_train is None or self.array_target is None:
            raise AttributeError('your fit is not valid')

        x_test = np.array([x_test]) if isinstance(x_test, str) else x_test
        x_test = np.array(x_test) if isinstance(x_test, list) else x_test
        preds = self.predict_cosine_similarity(x_test, return_cos=True)
        index_dict = {target: [i for i, t in enumerate(self.array_target) if t == target] for target in self.list_classes}

        probas_dict = {target:[] for target in set(self.array_target)}
        for col in preds:
            col_sum = col.sum()
            col_div = np.array([col[i] / col_sum for i in range(len(self.array_target))]) if col_sum != 0 else np.array([1 / len(col)] * len(self.array_target))
            for target in set(self.array_target):
                probas_dict[target] = probas_dict[target] + [sum(col_div[index_dict[target]])]

        probas = [[probas_dict[key][i] for key in self.list_classes] for i in range(len(x_test))]
        return np.array(probas, ndmin=2)

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
        json_data['classes_'] = self.tfidf.classes_ if hasattr(self.tfidf, 'classes_') else None

        # Save matrix_train if not None & level_save > LOW
        if (self.matrix_train is not None) and (self.level_save in ['MEDIUM', 'HIGH']):
            # Manage paths
            matrix_train_path = os.path.join(self.model_dir, "matrix_train.pkl")
            # Save as pickle
            with open(matrix_train_path, 'wb') as f:
                # TODO: use dill to get rid of  "can't pickle ..." errors
                pickle.dump(self.matrix_train, f)

        # Save array_target if not None & level_save > LOW
        if (self.array_target is not None) and (self.level_save in ['MEDIUM', 'HIGH']):
            # Manage paths
            array_target_path = os.path.join(self.model_dir, "array_target.pkl")
            # Save as pickle
            with open(array_target_path, 'wb') as f:
                # TODO: use dill to get rid of  "can't pickle ..." errors
                pickle.dump(self.array_target, f)

        # Save
        super().save(json_data=json_data)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            sklearn_pipeline_path (str): Path to standalone pipeline
            matrix_train_path (str): Path to matrix_train file
            array_target_path (str): Path to array_target file
            count_vectorizer_path (str): Path to countVectorizer (only with_super_documents = True)
            tfidf_super_documents_path (str): Path to tfidf super documents (only with_super_documents = True)
        Raises:
            ValueError: If configuration_path is None
            ValueError: If sklearn_pipeline_path is None
            ValueError: If matrix_train_path is None
            ValueError: If array_target_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object sklearn_pipeline_path is not an existing file
            FileNotFoundError: If the object matrix_train_path is not an existing file
            FileNotFoundError: If the object array_target_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        sklearn_pipeline_path = kwargs.get('sklearn_pipeline_path', None)
        matrix_train_path = kwargs.get('matrix_train_path', None)
        array_target_path = kwargs.get('array_target_path', None)
        count_vectorizer_path = kwargs.get('count_vectorizer_path', None)
        tfidf_super_documents_path = kwargs.get('tfidf_super_documents_path', None)

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
                          'multiclass_strategy', 'with_super_documents', 'with_super_documents']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload pipeline
        with open(sklearn_pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)

        # Reload pipeline elements
        self.tfidf = self.pipeline['tfidf']

        # Reload matrix_train and array_target
        with open(matrix_train_path, 'rb') as f:
            self.matrix_train = pickle.load(f)
        with open(array_target_path, 'rb') as f:
            self.array_target = pickle.load(f)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")