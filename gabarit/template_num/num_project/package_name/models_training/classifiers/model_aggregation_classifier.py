#!/usr/bin/env python3

## Model Aggregation Classifier

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
# - ModelAggregationClassifier -> Model to aggregate several classifier models

import os
import json
import logging
import numpy as np
import pandas as pd
import dill as pickle
from typing import Callable, Union, Tuple, Type

from ... import utils
from .. import utils_models
from ..model_class import ModelClass
from .model_classifier import ModelClassifierMixin  # type: ignore


def proba_argmax(proba: np.ndarray, list_classes: list, **kwargs):
    '''Gives the class corresponding to the argmax of the average of the given probabilities

    Args:
        proba (np.ndarray): The probabilities of each model for each class, array of shape (nb_models, nb_classes)
        list_classes (list): List of classes
    Returns:
        The prediction
    '''
    proba_average = np.sum(proba, axis=0) / proba.shape[0]
    index_class = np.argmax(proba_average)
    return list_classes[index_class]


def majority_vote(predictions: np.ndarray, **kwargs):
    '''Gives the class corresponding to the most present prediction in the given predictions.
    In case of a tie, gives the prediction of the first model involved in the tie
    Args:
        predictions (np.ndarray): The array containing the predictions of each model (shape (n_models))
    Returns:
        The prediction
    '''
    labels, counts = np.unique(predictions, return_counts=True)
    votes = [(label, count) for label, count in zip(labels, counts)]
    votes = sorted(votes, key=lambda x: x[1], reverse=True)
    possible_classes = {vote[0] for vote in votes if vote[1]==votes[0][1]}
    return [prediction for prediction in predictions if prediction in possible_classes][0]


def all_predictions(predictions: np.ndarray, **kwargs) -> np.ndarray:
    '''Calculates the sum of the arrays along axis 0 casts it to bool and then to int.
    Expects a numpy array containing only zeroes and ones.
    When used as an aggregation function, keeps all the prediction of each model (multi-labels)

    Args:
        predictions (np.ndarray) : Array of shape : (n_models, n_classes)
    Return:
        np.ndarray: The prediction
    '''
    return np.sum(predictions, axis=0, dtype=bool).astype(int)


def vote_labels(predictions: np.ndarray, **kwargs) -> np.ndarray:
    '''Gives the result of majority_vote applied on the second axis.
    When used as an aggregation_function, for each class, performs a majority vote for the aggregated models.
    It gives a multi-labels result

    Args:
        predictions (np.ndarray): array of shape : (n_models, n_classes)
    Return:
        np.ndarray: prediction
    '''
    return np.apply_along_axis(majority_vote, 0, predictions)


class ModelAggregationClassifier(ModelClassifierMixin, ModelClass):
    '''Model for aggregating several classifier models'''
    _default_name = 'model_aggregation_classifier'

    _dict_aggregation_function = {'majority_vote': {'aggregation_function': majority_vote, 'using_proba': False, 'multi_label': False},
                                  'proba_argmax': {'aggregation_function': proba_argmax, 'using_proba': True, 'multi_label': False},
                                  'all_predictions': {'aggregation_function': all_predictions, 'using_proba': False, 'multi_label': True},
                                  'vote_labels': {'aggregation_function': vote_labels, 'using_proba': False, 'multi_label': True}}

    def __init__(self, list_models: Union[list, None] = None, aggregation_function: Union[Callable, str] = 'majority_vote',
                 using_proba: bool = False, **kwargs) -> None:
        '''Initialization of the class (see ModelClass for more arguments)
        This model will aggregate the predictions of several model. The user can choose an aggregation function (with **kwargs if not using a list_classes arg)
        from existing ones, or create its own. All models must be either mono label or multi label, we do not accept mixes.
        However, we accept models that do not have the same class / labels. We will consider a meta model with joined classes / labels.

        Kwargs:
            list_models (list) : The list of models to be aggregated (can be None if reloading from standalones)
            aggregation_function (Callable or str) : The aggregation function used (custom function must use **kwargs if not using a list_classes arg)
            using_proba (bool) : Which object is being aggregated (the probabilities or the predictions).
        Raises:
            ValueError: All the aggregated sub_models have not the same multi_label attributes
            ValueError: The multi_label attributes of the aggregated models are inconsistent with multi_label
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Set attributes
        self.using_proba = using_proba
        self.aggregation_function = aggregation_function
        
        # Manage submodels
        self.sub_models = list_models  # Transform the list into a list of dictionnaries [{'name': xxx, 'model': xxx}, ...]

        # Check if only classifiers are present
        if False in [isinstance(sub_model['model'], ModelClassifierMixin) for sub_model in self.sub_models]:
            raise ValueError(f"model_aggregation_classifier only accepts classifier models")

        # Check for multi-labels inconsistencies
        set_multi_label = {sub_model['model'].multi_label for sub_model in self.sub_models}
        if len(set_multi_label) > 1:
            raise ValueError(f"All the aggregated sub_models have not the same multi_label attribute")
        if len(set_multi_label.union({self.multi_label})) > 1:
            raise ValueError(f"The multi_label attributes of the aggregated models are inconsistent with self.multi_label = {self.multi_label}.")

        # Set trained & classes info from submodels
        self.trained, self.list_classes, self.dict_classes = self._check_trained()
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
            ValueError: If the object aggregation_function is incompatible with multi_label
        '''
        # Retrieve aggregation function from dict if a string
        if isinstance(agg_function, str):
            # Get infos
            if agg_function not in self._dict_aggregation_function.keys():
                raise ValueError(f"The aggregation_function ({agg_function}) is not a valid option (must be chosen in {self._dict_aggregation_function.keys()})")
            using_proba = self._dict_aggregation_function[agg_function]['using_proba']
            multi_label = self._dict_aggregation_function[agg_function]['multi_label']
            agg_function = self._dict_aggregation_function[agg_function]['aggregation_function']  # type: ignore
            # Apply checks
            if self.using_proba != using_proba:
                self.logger.warning(f"using_proba {self.using_proba} is incompatible with the selected aggregation function '{agg_function}'. We force using_proba to {using_proba}.")
                self.using_proba = using_proba  # type: ignore
            if self.multi_label != multi_label:
                raise ValueError(f"multi_label {self.multi_label} is incompatible with the selected aggregation function '{agg_function}'.")
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

    def _check_trained(self) -> Tuple[bool, list, dict]:
        '''Checks and sets various attributes related to the fitting of underlying models

        Returns:
            bool: is the aggregation model is considered fitted
            list: list of classes
            dict: dict of classes
        '''
        # Check fitted
        models_trained = {sub_model['model'].trained for sub_model in self.sub_models}
        if len(models_trained) > 0 and all(models_trained):
            # All models trained
            trained = True
            # Set list_classes
            list_classes = list({label for sub_model in self.sub_models for label in sub_model['model'].list_classes})
            list_classes.sort()
            # Set dict_classes based on self.list_classes
            dict_classes = {i: col for i, col in enumerate(list_classes)}
        # No model or not fitted
        else:
            trained, list_classes, dict_classes = False, [], {}
        return trained, list_classes, dict_classes

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, with_shuffle: bool = True, **kwargs) -> None:
        '''Fits the model

        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train (?): Array-like, shape = [n_samples, n_targets]
        Kwargs:
            x_valid (?): Array-like, shape = [n_samples, n_features] - not used by sklearn models
            y_valid (?): Array-like, shape = [n_samples, n_targets] - not used by sklearn models
            with_shuffle (bool): If x, y must be shuffled before fitting - not used by sklearn models
        '''
        # Fit each model
        for sub_model in self.sub_models:
            model = sub_model['model']
            if not model.trained:
                model.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid, with_shuffle=True, **kwargs)

        # Set nb_fit to 1 if not already trained
        if not self.trained:
            self.nb_fit = 1

        # Update attributes
        self.trained, self.list_classes, self.dict_classes = self._check_trained()

    @utils.trained_needed
    def predict(self, x_test, return_proba: bool = False, **kwargs) -> np.ndarray:
        '''Prediction

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
            return_proba (bool): If the function should return the probabilities instead of the classes
        Returns:
            np.ndarray: array of shape = [n_samples]
        '''
        # We decide whether to rely on each model's probas or their predictions
        if return_proba:
            return self.predict_proba(x_test)
        else:
            # Get what we want (probas or preds) and use the aggregation function
            if self.using_proba:
                preds_or_probas = self._predict_probas_sub_models(x_test, **kwargs)
            else:
                preds_or_probas = self._predict_sub_models(x_test, **kwargs)
            return np.array([self.aggregation_function(array, list_classes=self.list_classes) for array in preds_or_probas])  # type: ignore

    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Predicts the probabilities on the test set

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            np.ndarray: array of shape = [n_samples, n_classes]
        '''
        probas_sub_models = self._predict_probas_sub_models(x_test, **kwargs)
        # The probas of all models are averaged
        return np.sum(probas_sub_models, axis=1) / probas_sub_models.shape[1]
    
    @utils.trained_needed
    def _predict_sub_models(self, x_test, **kwargs) -> np.ndarray:
        '''Recover the predictions of each model being aggregated

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            np.ndarray: not multi_label : array of shape = [n_samples, nb_model]
                        multi_label : array of shape = [n_samples, nb_model, n_classes]
        '''
        if self.multi_label:
            array_predict = np.array([self._predict_full_list_classes(sub_model['model'], x_test, return_proba=False) for sub_model in self.sub_models])
            array_predict = np.transpose(array_predict, (1, 0, 2))
        else:
            array_predict = np.array([sub_model['model'].predict(x_test) for sub_model in self.sub_models])
            array_predict = np.transpose(array_predict, (1, 0))
        return array_predict

    @utils.trained_needed
    def _predict_probas_sub_models(self, x_test, **kwargs) -> np.ndarray:
        '''Recover the probabilities of each model being aggregated

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            np.ndarray: array of shape = [n_samples, nb_model, nb_classes]
        '''
        array_probas = np.array([self._predict_full_list_classes(sub_model['model'], x_test, return_proba=True) for sub_model in self.sub_models])
        array_probas = np.transpose(array_probas, (1, 0, 2))
        return array_probas

    def _predict_full_list_classes(self, model: Type[ModelClass], x_test, return_proba: bool = False) -> np.ndarray:
        '''For multi_label: adds missing columns in the prediction of model (class missing in their list_classes)
        Or, if return_proba, adds a proba of zero to the missing classes in their list_classes

        Args:
            model (ModelClass): Model to use
            x_test (?): Array-like or sparse matrix of shape = [n_samples, n_features]
            return_proba (bool): If the function should return the probabilities instead of the classes
        Returns:
            np.ndarray: The array with the missing columns added
        '''
        # Get predictions or probas
        preds_or_probas = model.predict(x_test, return_proba=return_proba) # type: ignore

        # Manage each cases. Reorder predictions or probas according to aggregation model list_classes
        # Multi label, proba = True
        # Multi label, proba = False
        # Mono label, proba = True
        if model.multi_label or return_proba:
            df_all = pd.DataFrame(np.zeros((len(preds_or_probas), len(self.list_classes))), columns=self.list_classes)  # type: ignore
            df_model = pd.DataFrame(preds_or_probas, columns=model.list_classes)
            for col in model.list_classes:
                df_all[col] = df_model[col]
            return df_all.to_numpy()
        # Mono label, proba = False
        else:
            return preds_or_probas

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

        # Add some specific information
        json_data['list_models_name'] = [sub_model['name'] for sub_model in self.sub_models]
        json_data['using_proba'] = self.using_proba

        # Save aggregation_function if not None & level_save > LOW
        if (self.aggregation_function is not None) and (self.level_save in ['MEDIUM', 'HIGH']):
            # Manage paths
            aggregation_function_path = os.path.join(self.model_dir, "aggregation_function.pkl")
            # Save as pickle
            with open(aggregation_function_path, 'wb') as f:
                pickle.dump(self.aggregation_function, f)

        # Save
        models_list = [sub_model['name'] for sub_model in self.sub_models]
        aggregation_function = self.aggregation_function
        delattr(self, "sub_models")
        delattr(self, "aggregation_function")
        super().save(json_data=json_data)
        setattr(self, "aggregation_function", aggregation_function)
        setattr(self, "sub_models", models_list)  # Setter needs list of models, not sub_models itself

        # Add message in model_upload_instructions.md
        md_path = os.path.join(self.model_dir, f"model_upload_instructions.md")
        line = "/!\\/!\\/!\\/!\\/!\\   The aggregation model is a special model, please ensure that all sub-models and the aggregation model are manually saved together in order to be able to load it .  /!\\/!\\/!\\/!\\/!\\ \n"
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
           Reloads the sub_models from their files
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
        # Can't set int as keys in json, so need to cast it after reloading
        # dict_classes keys are always ints
        if 'dict_classes' in configs.keys():
            configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}
        elif 'list_classes' in configs.keys():
            configs['dict_classes'] = {i: col for i, col in enumerate(configs['list_classes'])}

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
        # Try to read the following attributes from configs and, if absent, keep the current one
        self.sub_models = configs.get('list_models_name', [])  # Transform the list into a list of dictionnaries [{'name': xxx, 'model': xxx}, ...]
        for attribute in ['x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'using_proba']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")