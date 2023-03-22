#!/usr/bin/env python3

## Model TFIDF SGDClassifier

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
# - ModelTfidfSgdc -> Model for predictions via TF-IDF + SGDClassier


import logging
import numpy as np
from typing import Union, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from ... import utils
from .model_pipeline import ModelPipeline


class ModelTfidfSgdc(ModelPipeline):
    '''Model for predictions via TF-IDF + SGDClassier'''

    _default_name = 'model_tfidf_sgdc'

    def __init__(self, tfidf_params: Union[dict, None] = None, sgdc_params: Union[dict, None] = None,
                 multiclass_strategy: Union[str, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelPipeline & ModelClass for more arguments)

        Kwargs:
            tfidf_params (dict) : Parameters for the tfidf
            sgdc_params (dict) : Parameters for the sgdc
            multiclass_strategy (str): Multi-classes strategy, 'ovr' (OneVsRest), or 'ovo' (OneVsOne). If None, use the default of the algorithm.
        Raises:
            ValueError: If multiclass_strategy is not 'ovo', 'ovr' or None
        '''
        if multiclass_strategy is not None and multiclass_strategy not in ['ovo', 'ovr']:
            raise ValueError(f"The value of 'multiclass_strategy' must be 'ovo' or 'ovr' (not {multiclass_strategy})")
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Manage model
        if tfidf_params is None:
            tfidf_params = {}
        self.tfidf = TfidfVectorizer(**tfidf_params)
        if sgdc_params is None:
            sgdc_params = {}
        self.sgdc = SGDClassifier(**sgdc_params)
        self.multiclass_strategy = multiclass_strategy

        # Can't do multi-labels / multi-classes
        if not self.multi_label:
            # If not multi-classes : no impact
            if multiclass_strategy == 'ovr':
                self.pipeline = Pipeline([('tfidf', self.tfidf), ('sgdc', OneVsRestClassifier(self.sgdc))])
            elif multiclass_strategy == 'ovo':
                self.pipeline = Pipeline([('tfidf', self.tfidf), ('sgdc', OneVsOneClassifier(self.sgdc))])
            else:
                self.pipeline = Pipeline([('tfidf', self.tfidf), ('sgdc', self.sgdc)])

        # Manage multi-labels -> add a MultiOutputClassifier
        # The SGDClassifier does not natively support multi-labels
        if self.multi_label:
            self.pipeline = Pipeline([('tfidf', self.tfidf), ('sgdc', MultiOutputClassifier(self.sgdc))])

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Predicts the probabilities on the test set
        - /!\\ THE SGDC DOES NOT RETURN PROBABILITIES UNDER SOME CONDITIONS HERE WE SIMULATE PROBABILITIES EQUAL TO 0 OR 1 /!\\ -

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        # Can't use probabilities if loss not in ['log', 'modified_huber'] or 'ovo' and not multi-labels
        if self.sgdc.loss not in ['log', 'modified_huber'] or (self.multiclass_strategy == 'ovo' and not self.multi_label):
            if not self.multi_label:
                preds = self.pipeline.predict(x_test)
                # Format ['a', 'b', 'c', 'a', ..., 'b']
                # Transform to "proba"
                transform_dict = {col: [0. if _ != i else 1. for _ in range(len(self.list_classes))] for i, col in enumerate(self.list_classes)}
                probas = np.array([transform_dict[x] for x in preds])
            else:
                preds = self.pipeline.predict(x_test)
                # Already right format, but in int !
                probas = np.array([[float(_) for _ in x] for x in preds])
        # Otherwise, classic probabilities
        else:
            probas = super().predict_proba(x_test, **kwargs)
        return probas

    def get_predict_position(self, x_test, y_true, **kwargs) -> np.ndarray:
        '''Gets the order of predictions of y_true.
        Positions start at 1 (not 0)

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
            y_true (?): Array-like, shape = [n_samples, n_features]
        Returns:
            np.ndarray: Array, shape = [n_samples]
        '''
        if self.sgdc.loss not in ['log', 'modified_huber']:
            self.logger.warning("Warning, the method get_predict_position is not suitable for a SGDC model (except for the losses 'log' and 'modified_huber')"
                                "(no probabilities, we use 1 or 0)")
        return super().get_predict_position(x_test, y_true)

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

        # Save
        super().save(json_data=json_data)

    @classmethod
    def _init_new_instance_from_configs(cls, configs):
        '''Inits a new instance from a set of configurations

        Args:
            configs: a set of configurations of a model to be reloaded
        Returns:
            ModelClass: the newly generated class
        '''
        # Call parent
        model = super()._init_new_instance_from_configs(configs)

        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['multiclass_strategy']:
            setattr(model, attribute, configs.get(attribute, getattr(model, attribute)))

        # Return the new model
        return model

    def _load_standalone_files(self, default_model_dir: Union[str, None] = None, *args, **kwargs):
        '''Loads standalone files for a newly created model via _init_new_instance_from_configs

        Kwargs:
            default_model_dir (str): a path to look for default file paths
                                     If None, standalone files path should all be provided
        '''
        # Call parent
        super()._load_standalone_files(default_model_dir=default_model_dir, **kwargs)

        # Reload pipeline elements
        self.tfidf = self.pipeline['tfidf']

        # Manage multi-labels or multi-classes
        if not self.multi_label and self.multiclass_strategy is None:
            self.sgdc = self.pipeline['sgdc']
        else:
            self.sgdc = self.pipeline['sgdc'].estimator


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
