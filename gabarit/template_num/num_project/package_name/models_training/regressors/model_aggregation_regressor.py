#!/usr/bin/env python3

## Model Aggregation Regressors

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
# - ModelAggregationRegressors -> model aggregation for regressor

import os
import json
import logging
import dill as pickle

import numpy as np
import pandas as pd
from typing import Callable, Union, List

from {{package_name}} import utils
from {{package_name}}.models_training import utils_models
from {{package_name}}.monitoring.model_logger import ModelLogger
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.regressors.model_regressor import ModelRegressorMixin  # type: ignore


class ModelAggregationRegressors(ModelRegressorMixin, ModelClass):
    '''Model for aggregating multiple ModelClasses'''
    _default_name = 'model_aggregation_regressor'

    def __init__(self, list_models: Union[list, None] = None, aggregation_function: Union[Callable, str] = 'median_predict', **kwargs) -> None:
        '''Initialization of the class (see ModelClass for more arguments)

        Kwargs:
            list_models (list) : The list of model to be aggregated
            aggregation_function (Callable or str) : The aggregation function used

        Raises:
            ValueError : If the object list_model has other model than model regressor (model_aggregation_regressor is only compatible with model regressor)
            ValueError : If the object aggregation_function is a str but not found in the dictionary dict_aggregation_function
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Get the aggregation function
        dict_aggregation_function = {'median_predict': {'function': self.median_predict, 'using_proba': False, 'multi_label': False},
                                     'mean_predict': {'function': self.mean_predict, 'using_proba': False, 'multi_label': False}}
        if isinstance(aggregation_function, str):
            if aggregation_function not in dict_aggregation_function.keys():
                raise ValueError(f"The aggregation_function object ({aggregation_function}) is not a valid option ({dict_aggregation_function.keys()})")
            aggregation_function = dict_aggregation_function[aggregation_function]['function'] # type: ignore

        # Manage model
        self.aggregation_function = aggregation_function
        self.list_real_models: list = None
        self.list_models: list = None
        if list_models is not None:
            self._sort_model_type(list_models)

        # Error: The classifier and regressor models cannot be combined in list_models
        if self.list_real_models is not None:
            if not self._all_sub_model_are_regressor():
                raise ValueError(f"model_aggregation_regressor is only compatible with model regressor")
            # set list_models_trained
            self.list_models_trained = [model.trained for model in self.list_real_models]

        self._check_trained()

    def _sort_model_type(self, list_models: list) -> None:
        '''Populate the self.list_real_models if it is None.
           Init list_real_models with each model and list_models with each model_name.

        Args:
            list_models (list): list of the models or of their name
        '''
        if self.list_real_models is None:
            list_real_models = []
            new_list_models = []
            # Get the real model and model name
            for model in list_models:
                if isinstance(model, str):
                    real_model, _ = utils_models.load_model(model)
                    new_list_models.append(model)
                else:
                    real_model = model
                    new_list_models.append(os.path.split(model.model_dir)[-1])
                list_real_models.append(real_model)
            self.list_real_models = list_real_models
            self.list_models = new_list_models

    def _all_sub_model_are_regressor(self) -> np.bool_:
        '''Checke all list_real_models are models regressor

        Args:
            (bool): all list_real_models are models regressor
        '''
        for model in self.list_real_models:
            if not isinstance(model, ModelRegressorMixin):
                return False
        return True

    def _check_trained(self):
        '''Check and sets various attributes related to the fitting of underlying models

        Raises more than one type of labels in list models
        '''
        # Check fitted
        if self.list_real_models is not None:
            models_trained = {model.trained for model in self.list_real_models}
            if False not in models_trained:
                self.trained = True
                self.nb_fit += 1

    def fit(self, x_train, y_train, **kwargs) -> None:
        '''Trains the model
           **kwargs enables Keras model compatibility.

        Args:
            x_train (?): Array-like, shape = [n_samples]
            y_train (?): Array-like, shape = [n_samples]
        '''

        # Fit each model
        for model in self.list_real_models:
            if not model.trained:
                model.fit(x_train, y_train, **kwargs)
        self._check_trained()

    @utils.trained_needed
    def predict(self, x_test, **kwargs) -> np.ndarray:
        '''Prediction

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.ndarray): array of shape = [n_samples]
        '''
        # We decide whether to rely on each model's probas or their prediction
        preds = self._get_predictions(x_test, **kwargs)
        return np.array([self.aggregation_function(array) for array in preds]) # type: ignore

    @utils.trained_needed
    def _get_predictions(self, x_test, **kwargs) -> np.ndarray:
        '''Recover the probability of each model being aggregated

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.ndarray): array of shape = [n_samples, nb_model]
        '''
        array_predict = np.array([model.predict(x_test) for model in self.list_real_models])
        array_predict = np.transpose(array_predict, (1, 0))
        return array_predict

    def median_predict(self, predictions: np.ndarray) -> np.float64:
        '''Returns the median predicted by predictions of each models

        Args:
            (np.ndarray) : shape (n_models) the array containing the predictions of each models
        Return:
            (np.float64) : predict
        '''
        return np.median(predictions)

    def mean_predict(self, predictions: np.ndarray) -> np.float64:
        '''Returns the mean predicted by predictions of each models

        Args:
            (np.ndarray) : shape (n_models) the array containing the predictions of each models
        Return:
            (np.float64) : predict
        '''
        return np.mean(predictions)

    def save(self, json_data: Union[dict, None] = {}) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save each trained and unsaved model
        for i, model in enumerate(self.list_real_models):
            if not self.list_models_trained[i] and model.trained:
                model.save()

        json_data['list_models'] = self.list_models.copy()

        aggregation_function = self.aggregation_function

        # Save aggregation_function if not None & level_save > LOW
        if (self.aggregation_function is not None) and (self.level_save in ['MEDIUM', 'HIGH']):
            # Manage paths
            aggregation_function_path = os.path.join(self.model_dir, "aggregation_function.pkl")
            # Save as pickle
            with open(aggregation_function_path, 'wb') as f:
                pickle.dump(self.aggregation_function, f)

        # Save
        list_real_models = self.list_real_models
        delattr(self, "list_real_models")
        delattr(self, "aggregation_function")
        super().save(json_data=json_data)
        setattr(self, "aggregation_function", aggregation_function)
        setattr(self, "list_real_models", list_real_models)

        # Add message in model_upload_instructions.md
        md_path = os.path.join(self.model_dir, f"model_upload_instructions.md")
        line = "/!\/!\/!\/!\/!\   The aggregation model is a special model, please ensure that all sub-models and the aggregation model are manually saved together in order to be able to load it .  /!\/!\/!\/!\/!\ "
        self.prepend_line(md_path, line)

    def prepend_line(self, file_name: str, line: str) -> None:
        ''' Insert given string as a new line at the beginning of a file 

        Kwargs:
            file_name (str): Path to file
            line (str): line to insert
        '''
        dummy_file = file_name + '.bak'
        with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
            write_obj.write(line + '\n')
            for line in read_obj:
                write_obj.write(line)
        os.remove(file_name)
        os.rename(dummy_file, file_name)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model aggregation from its configuration and "standalones" files
            Reloads list model from "list_models" files

        Kwargs:
            configuration_path (str): Path to configuration file
            aggregation_function_path (str): Path to aggregation_function_path
        Raises:
            ValueError: If configuration_path is None
            ValueError: If aggregation_function_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object aggregation_function_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        aggregation_function_path = kwargs.get('aggregation_function_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if aggregation_function_path is None:
            raise ValueError("The argument aggregation_function_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(aggregation_function_path):
            raise FileNotFoundError(f"The file {aggregation_function_path} does not exist")

        # Load confs
        with open(configuration_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)

        # Reload aggregation_function_path
        with open(aggregation_function_path, 'rb') as f:
            self.aggregation_function = pickle.load(f)

        # Set class vars
        # self.model_name = # Keep the created name
        # self.model_dir = # Keep the created folder
        self.nb_fit = configs.get('nb_fit', 1)  # Consider one unique fit by default
        self.trained = configs.get('trained', True)  # Consider trained by default
        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['x_col', 'y_col', 'level_save', 'list_models']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        self._sort_model_type(self.list_models)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")