#!/usr/bin/env python3

## Model Aggregation Regressor

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
# - ModelAggregationRegressor -> Model to aggregate several regressor models

import os
import json
import logging
import numpy as np
import dill as pickle
from typing import Callable, Union, List

from ... import utils
from .. import utils_models
from ..model_class import ModelClass
from ..regressors.model_regressor import ModelRegressorMixin  # type: ignore


def median_predict(predictions: np.ndarray) -> np.float64:
    '''Returns the median of the predictions of each model

    Args:
        predictions (np.ndarray) : The array containing the predictions of each models (shape (n_models))
    Return:
        (np.float64) : The median of the predictions
    '''
    return np.median(predictions)

def mean_predict(predictions: np.ndarray) -> np.float64:
    '''Returns the mean of predictions of each model

    Args:
        predictions (np.ndarray) : The array containing the predictions of each models (shape (n_models))
    Return:
        (np.float64) : The mean of the predictions
    '''
    return np.mean(predictions)


class ModelAggregationRegressor(ModelRegressorMixin, ModelClass):
    '''Model for aggregating several regressor models'''
    _default_name = 'model_aggregation_regressor'

    _dict_aggregation_function = {'median_predict': median_predict,
                                  'mean_predict': mean_predict}

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
        
        if isinstance(aggregation_function, str):
            if aggregation_function not in self._dict_aggregation_function.keys():
                raise ValueError(f"The aggregation_function ({aggregation_function}) is not a valid option ({self._dict_aggregation_function.keys()})")
            aggregation_function = self._dict_aggregation_function[aggregation_function] # type: ignore

        # Manage aggregated models
        self.aggregation_function = aggregation_function

        self.sub_models = list_models # Transform the list into a list of dictionnaries [{'name': xxx, 'model': xxx}, ...]

        # Error: The classifier and regressor models cannot be combined in list_models
        if False in [isinstance(sub_model['model'], ModelRegressorMixin) for sub_model in self.sub_models]:
            raise ValueError(f"model_aggregation_regressor only accepts regressor models")

        self.trained = self._check_trained()

        # Set nb_fit to 1 if already trained
        if self.trained:
            self.nb_fit = 1

    @property
    def aggregation_function(self):
        '''Getter for aggregation_function'''
        return self._aggregation_function

    @aggregation_function.setter
    def aggregation_function(self, agg_function: Union[Callable, str]):
        '''Setter for aggregation_function
        If a string, try to match a predefined function

        Raises:
            ValueError: If the object aggregation_function is a str but not found in the dictionary of predefined aggregation functions
        '''
        # Retrieve aggregation function from dict if a string
        if isinstance(agg_function, str):
            # Get infos
            if agg_function not in self._dict_aggregation_function.keys():
                raise ValueError(f"The aggregation_function ({agg_function}) is not a valid option (must be chosen in {self._dict_aggregation_function.keys()})")
            agg_function = self._dict_aggregation_function[agg_function]
            # Apply checks
        self._aggregation_function = agg_function

    @aggregation_function.deleter
    def aggregation_function(self):
        '''Deleter for aggregation_function'''
        self._aggregation_function = None

    @property
    def sub_models(self):
        '''Getter for sub_models'''
        return self._sub_models

    @sub_models.setter
    def sub_models(self, list_models: Union[list, None] = None):
        '''Setter for sub_models

        Kwargs:
            list_models (list) : The list of models to be aggregated
        '''
        list_models = [] if list_models is None else list_models
        sub_models = []  # Init list of models
        for model in list_models:
            # If a string (a model name), reload it
            if isinstance(model, str):
                real_model, _ = utils_models.load_model(model)
                dict_model = {'name': model, 'model': real_model}
            else:
                dict_model = {'name': os.path.split(model.model_dir)[-1], 'model': model}
            sub_models.append(dict_model.copy())
        self._sub_models = sub_models.copy()

    @sub_models.deleter
    def sub_models(self):
        '''Deleter for sub_models'''
        self._sub_models = None

    def _check_trained(self) -> bool:
        '''Checks various attributes related to the fitting of underlying models

        Returns:
            bool: is the aggregation model is considered fitted
        '''
        # Check fitted
        models_trained = {sub_model['model'].trained for sub_model in self.sub_models}
        if len(models_trained) > 0 and all(models_trained):
            # All models trained
            trained = True
        # No model or not fitted
        else:
            trained = False
        return trained

    def fit(self, x_train, y_train, **kwargs) -> None:
        '''Trains the model
           **kwargs enables Keras model compatibility.

        Args:
            x_train (?): Array-like, shape = [n_samples]
            y_train (?): Array-like, shape = [n_samples]
        '''
        # We check input format
        x_train, y_train = self._check_input_format(x_train, y_train, fit_function=True)

        # Fit each model
        for sub_model in self.sub_models:
            if not sub_model['model'].trained:
                sub_model['model'].fit(x_train, y_train, **kwargs)

        # Set nb_fit to 1 if not already trained
        if not self.trained:
            self.nb_fit = 1

        self.trained = self._check_trained()

    @utils.trained_needed
    def predict(self, x_test, return_proba: bool = False, **kwargs) -> np.ndarray:
        '''Prediction

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
            return_proba (bool): If the function should return the probabilities instead of the classes (Keras compatibility)
        Returns:
            (np.ndarray): Array of shape = [n_samples]
        Raises:
            ValueError: If return_proba=True
        '''
        if return_proba:
            raise ValueError(f"Models of the type {self.model_type} can't handle probabilities")
        preds = self._predict_sub_models(x_test, **kwargs)
        return np.array([self.aggregation_function(array) for array in preds]) # type: ignore

    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> None:
        '''Predicts the probabilities on the test set - raise ValueError

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Raises:
            ValueError: Models of type regressor do not implement the method predict_proba
        '''
        raise ValueError(f"Models of type regressor do not implement the method predict_proba")

    @utils.trained_needed
    def _predict_sub_models(self, x_test, **kwargs) -> np.ndarray:
        '''Recover the predictions of each model being aggregated

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.ndarray): array of shape = [n_samples, nb_model]
        '''
        array_predict = np.array([sub_model['model'].predict(x_test) for sub_model in self.sub_models])
        array_predict = np.transpose(array_predict, (1, 0))
        return array_predict

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        if json_data is None:
            json_data = {}
            
        # Specific aggregation - save some wanted entries
        train_keys = ['filename', 'filename_valid', 'preprocess_str']
        default_json_data = {key: json_data.get(key, None) for key in train_keys}
        default_json_data['aggregator_dir'] = self.model_dir
        # Save each trained and unsaved model
        for sub_model in self.sub_models:
            path_config = os.path.join(sub_model['model'].model_dir, 'configurations.json')
            if os.path.exists(path_config):
                with open(path_config, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                    trained = configs.get('trained', False)
                    if not trained:
                        sub_model['model'].save(default_json_data)
            else:
                sub_model['model'].save(default_json_data)

        json_data['list_models_name'] = [sub_model['name'] for sub_model in self.sub_models]

        aggregation_function = self.aggregation_function

        # Save aggregation_function if not None & level_save > LOW
        if (self.aggregation_function is not None) and (self.level_save in ['MEDIUM', 'HIGH']):
            # Manage paths
            aggregation_function_path = os.path.join(self.model_dir, "aggregation_function.pkl")
            # Save as pickle
            with open(aggregation_function_path, 'wb') as f:
                pickle.dump(self.aggregation_function, f)

        # Save
        models_list = [sub_model['name'] for sub_model in self.sub_models]
        delattr(self, "sub_models")
        delattr(self, "aggregation_function")
        super().save(json_data=json_data)
        setattr(self, "aggregation_function", aggregation_function)
        setattr(self, "sub_models", models_list)

        # Add message in model_upload_instructions.md
        md_path = os.path.join(self.model_dir, f"model_upload_instructions.md")
        line = "/!\\/!\\/!\\/!\\/!\\   The aggregation model is a special model, please ensure that all sub-models and the aggregation model are manually saved together in order to be able to load it  /!\\/!\\/!\\/!\\/!\\ \n"
        self.prepend_line(md_path, line)

    def prepend_line(self, file_name: str, line: str) -> None:
        ''' Insert given string as a new line at the beginning of a file

        Kwargs:
            file_name (str): Path to file
            line (str): line to insert
        '''
        with open(file_name, 'r+') as f:
            lines = f.readlines()
            lines.insert(0, line)
            f.seek(0)
            f.writelines(lines)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model aggregation from its configuration and "standalones" files
            Reloads list model from "list_models" files

        Kwargs:
            configuration_path (str): Path to configuration file
            preprocess_pipeline_path (str): Path to preprocess pipeline
            aggregation_function_path (str): Path to aggregation_function_path
        Raises:
            ValueError: If configuration_path is None
            ValueError: If preprocess_pipeline_path is None
            ValueError: If aggregation_function_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object preprocess_pipeline_path is not an existing file
            FileNotFoundError: If the object aggregation_function_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        preprocess_pipeline_path = kwargs.get('preprocess_pipeline_path', None)
        aggregation_function_path = kwargs.get('aggregation_function_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if preprocess_pipeline_path is None:
            raise ValueError("The argument preprocess_pipeline_path can't be None")
        if aggregation_function_path is None:
            raise ValueError("The argument aggregation_function_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(preprocess_pipeline_path):
            raise FileNotFoundError(f"The file {preprocess_pipeline_path} does not exist")
        if not os.path.exists(aggregation_function_path):
            raise FileNotFoundError(f"The file {aggregation_function_path} does not exist")

        # Load confs
        with open(configuration_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)

        # Reload pipeline preprocessing
        with open(preprocess_pipeline_path, 'rb') as f:
            self.preprocess_pipeline = pickle.load(f)

        # Reload aggregation_function_path
        with open(aggregation_function_path, 'rb') as f:
            self.aggregation_function = pickle.load(f)

        # Set class vars
        # self.model_name = # Keep the created name
        # self.model_dir = # Keep the created folder
        self.nb_fit = configs.get('nb_fit', 1)  # Consider one unique fit by default
        self.trained = configs.get('trained', True)  # Consider trained by default
        self.sub_models = configs.get('list_models_name', [])  # Transform the list into a list of dictionnaries [{'name': xxx, 'model': xxx}, ...]
        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['x_col', 'y_col', 'level_save']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")