#!/usr/bin/env python3

## Model Aggregation

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
# - ModelAggregation -> model aggregation with ModelClass

import os
import json
import logging
import dill as pickle

import numpy as np
import pandas as pd
from types import FunctionType, MethodType
from typing import Callable, Union, Dict

from {{package_name}} import utils
from {{package_name}}.models_training import utils_models
from {{package_name}}.models_training.model_class import ModelClass


class ModelAggregation(ModelClass):
    '''Model for aggregating multiple ModelClasses'''
    _default_name = 'model_aggregation'

    def __init__(self, list_models: Union[list, None] = None, aggregation_function: Union[Callable, str] = 'majority_vote', using_proba: Union[bool, None] = None, multi_label: Union[bool, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelClass for more arguments)

        Kwargs:
            list_models (list) : list of model to be aggregated
            aggregation_function (Callable or str) : aggregation function used
            using_proba (bool) : which object is being aggregated (the probas or the predictions).
            multi_label (bool): If the classification is multi-labels

        Raises:
            ValueError : if aggregation_function object is Callable and using_proba/multi_label is None
            ValueError : if the object aggregation_function is a str but not found in the dictionary dict_aggregation_function
            ValueError : if the object aggregation_function is not compatible with value using_proba
            ValueError : if the object aggregation_function is not compatible with value multi_label
            ValueError : The 'multi_label' parameters of the list models are inconsistent with the model_aggregation
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Get the aggregation function
        self.using_proba = using_proba
        self.multi_label = multi_label
        dict_aggregation_function = {'majority_vote': {'function': self.majority_vote, 'using_proba': False, 'multi_label': False},
                                                                        'proba_argmax': {'function': self.proba_argmax, 'using_proba': True, 'multi_label': False},
                                                                        'all_predictions': {'function': self.all_predictions, 'using_proba': False, 'multi_label': True},
                                                                        'vote_labels': {'function': self.vote_labels, 'using_proba': False, 'multi_label': True}}
        if isinstance(aggregation_function, (FunctionType, MethodType)):
            if using_proba is None or multi_label is None:
                raise ValueError(f"When aggregation_function is Callable, using_proba(bool) and multi_label(bool) cannot be None ")
        elif isinstance(aggregation_function, str):
            if aggregation_function not in dict_aggregation_function.keys():
                raise ValueError(f"The aggregation_function object ({aggregation_function}) is not a valid option ({dict_aggregation_function.keys()})")
            if using_proba is None:
                self.using_proba = dict_aggregation_function[aggregation_function]['using_proba'] # type: ignore
            elif using_proba != dict_aggregation_function[aggregation_function]['using_proba']:
                raise ValueError(f"The aggregation_function object ({aggregation_function}) is not compatible with using_proba=({using_proba})")
            if multi_label is None:
                self.multi_label = dict_aggregation_function[aggregation_function]['multi_label'] # type: ignore
            elif multi_label != dict_aggregation_function[aggregation_function]['multi_label']:
                raise ValueError(f"The aggregation_function object ({aggregation_function}) is not compatible with multi_label=({multi_label})")
            aggregation_function = dict_aggregation_function[aggregation_function]['function'] # type: ignore

        # Manage model
        self.aggregation_function = aggregation_function
        self.list_real_models: list = None
        self.list_models: list = None
        if list_models is not None:
            self._sort_model_type(list_models)

        # Error for multi label inconsistency
        if self.list_real_models is not None:
            set_multi_label = {model.multi_label for model in self.list_real_models}
            if True in set_multi_label and not self.multi_label:
                raise ValueError(f"The 'multi_label' parameters of the list models are inconsistent with the model_aggregation.")

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

                # Set list_classes
                self.list_classes = list({label for model in self.list_real_models for label in model.list_classes})
                list_label_str = [label for label in self.list_classes if isinstance(label, (str, np.str))]
                list_label_other = [int(label) for label in self.list_classes if label not in list_label_str]
                if len(list_label_str) > 0 and len(list_label_other) > 0:
                    raise TypeError('There are more than one type of labels in the list models.')
                self.list_classes.sort()

                # Set dict_classes based on list classes
                self.dict_classes = {i: col for i, col in enumerate(self.list_classes)}

    def fit(self, x_train, y_train, **kwargs) -> None:
        '''Trains the model
           **kwargs enables Keras model compatibility.

        Args:
            x_train (?): Array-like, shape = [n_samples]
            y_train (?): Array-like, shape = [n_samples]
        Raises:
            ValueError : if model needs mono_label but y_train is multi_label
            ValueError : if model needs multi_label but y_train is mono_label
        '''
        if isinstance(y_train, pd.DataFrame):
            bool_multi_label = True if len(y_train.iloc[0]) > 1 else False
        elif isinstance(y_train, np.ndarray):
            bool_multi_label = True if y_train.shape != (len(x_train),) else False
        elif isinstance(y_train, pd.Series):
            bool_multi_label = False
        else:
            bool_multi_label = False

        # Fit each model
        for model in self.list_real_models:
            if not model.trained:
                if bool_multi_label and not model.multi_label:
                    raise ValueError(f"Model {model} (model_name: {model.model_name}) needs y_train_mono_label to fit")
                if not bool_multi_label and model.multi_label:
                    raise ValueError(f"Model {model}(model_name: {model.model_name}) needs y_train_multi_label to fit")
                model.fit(x_train, y_train, **kwargs)

        self._check_trained()

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict(self, x_test, return_proba: Union[bool, None] = False, **kwargs) -> np.array:
        '''Prediction

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
            return_proba (bool): If the function should return the probabilities instead of the classes
        Returns:
            (np.array): array of shape = [n_samples]
        '''
        # We decide whether to rely on each model's probas or their prediction
        if return_proba:
            return self.predict_proba(x_test)
        elif self.using_proba:
            preds = self._get_probas(x_test, **kwargs)
        else:
            preds = self._get_predictions(x_test, **kwargs)
        return np.array([self.aggregation_function(array) for array in preds]) # type: ignore

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def _get_probas(self, x_test, **kwargs) -> np.ndarray:
        '''Recover the probability of each model being aggregated

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.ndarray): array of shape = [n_samples, nb_model, nb_classes]
        '''
        array_proba = np.array([self._predict_model_with_full_list_classes(model, x_test, return_proba=True) for model in self.list_real_models])
        array_proba = np.transpose(array_proba, (1, 0, 2))
        return array_proba

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def _get_predictions(self, x_test, **kwargs) -> np.ndarray:
        '''Recover the probability of each model being aggregated

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.ndarray): not multi-label : array of shape = [n_samples, nb_model]
                          multi-label : array of shape = [n_samples, nb_model, n_classes]
        '''
        if self.multi_label:
            array_predict = np.array([self._predict_model_with_full_list_classes(model, x_test, return_proba=False) for model in self.list_real_models])
            array_predict = np.transpose(array_predict, (1, 0, 2))
        else:
            array_predict = np.array([model.predict(x_test) for model in self.list_real_models])
            array_predict = np.transpose(array_predict, (1, 0))
        return array_predict

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.array:
        '''Predicts the probabilities on the test set

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.array): array of shape = [n_samples, n_classes]
        '''
        probas = self._get_probas(x_test, **kwargs)
        # The probas of all models are averaged.
        return np.sum(probas, axis=1) / probas.shape[1]

    def _predict_model_with_full_list_classes(self, model, x_test, return_proba: Union[bool, None] = False) -> np.array:
        '''For multi_label: Complete columns missing in the prediction of model (label missing in their list_classes)

        Args:
            model (?): model to predict
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
            return_proba (bool): If the function should return the probabilities instead of the classes
        Returns:
            (np.array): predict complete (0 for missing columns)
        '''
        pred = model.predict(x_test, return_proba=return_proba)

        if model.multi_label or return_proba:
            df_all = pd.DataFrame(np.zeros((len(pred), len(self.list_classes))), columns=self.list_classes)
            df_model = pd.DataFrame(pred, columns=model.list_classes)
            for col in model.list_classes:
                df_all[col] = df_model[col]
            return df_all.to_numpy()
        elif not self.multi_label and not return_proba:
            return pred
        else:
            return np.array([[1 if pred[n_test] == col else 0 for col in self.list_classes] for n_test in range(len(pred))])

    def proba_argmax(self, proba: np.ndarray):
        '''Aggregation_function: We take the argmax of the mean of the probabilities of the underlying models to provide a prediction

        Args:
            proba (np.ndarray): array of shape (nb_models, nb_classes) the probability of each model for each class
        Returns:
            the prediction
        '''
        proba_average = np.sum(proba, axis=0) / proba.shape[0]
        index_class = np.argmax(proba_average)
        return self.list_classes[index_class]

    def majority_vote(self, predictions: np.ndarray):
        '''Aggregation_function: A majority voting system of multiple predictions is used.
        In the case of a tie, we use the first model's prediction (even if it is not in the first votes)

        Args:
            (np.ndarray) : shape (n_models) the array containing the predictions of each models
        Returns:
            the prediction
        '''
        labels, counts = np.unique(predictions, return_counts=True)
        votes = [(label, count) for label, count in zip(labels, counts)]
        votes = sorted(votes, key=lambda x: x[1], reverse=True)
        if len(votes) > 1 and votes[0][1] == votes[1][1]:
            return predictions[0]
        else:
            return votes[0][0]

    def all_predictions(self, predictions: np.ndarray) -> np.ndarray:
        '''Returns all labels predicted by the list of models ie returns 1 if at least one model
        predicts this label  (multi_label only)

        Args:
            (np.ndarray) : array of shape : (n_models, n_classes)
        Return:
            (np.ndarray) : prediction
        '''
        return np.sum(predictions, axis=0, dtype=bool).astype(int)

    def vote_labels(self, predictions: np.ndarray) -> np.ndarray:
        '''Returns the labels predicted by majority_vote for each labels
        predicts this label (multi_label only)

        Args:
            (np.ndarray) : array of shape : (n_models, n_classes)
        Return:
            (np.ndarray) : prediction
        '''
        predictions = predictions.T
        return np.array([self.majority_vote(preds) for preds in predictions])

    def save(self, json_data: Union[dict, None] = {}) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save each model
        for model in self.list_real_models:
            model.save()

        json_data['list_models'] = self.list_models.copy()
        json_data['using_proba'] = self.using_proba

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
        # Can't set int as keys in json, so need to cast it after reloading
        # dict_classes keys are always ints
        if 'dict_classes' in configs.keys():
            configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}
        elif 'list_classes' in configs.keys():
            configs['dict_classes'] = {i: col for i, col in enumerate(configs['list_classes'])}

        # Reload aggregation_function_path
        with open(aggregation_function_path, 'rb') as f:
            self.aggregation_function = pickle.load(f)

        # Set class vars
        # self.model_name = # Keep the created name
        # self.model_dir = # Keep the created folder
        self.nb_fit = configs.get('nb_fit', 1)  # Consider one unique fit by default
        self.trained = configs.get('trained', True)  # Consider trained by default
        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['x_col', 'y_col',
                          'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'list_models', 'using_proba']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        self._sort_model_type(self.list_models)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")