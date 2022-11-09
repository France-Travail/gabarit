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
from typing import Callable, Union, List
from types import FunctionType, MethodType

from {{package_name}} import utils
from {{package_name}}.models_training import utils_models
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.classifiers.model_classifier import ModelClassifierMixin  # type: ignore


class ModelAggregationClassifier(ModelClassifierMixin, ModelClass):
    '''Model for aggregating several classifier models'''
    _default_name = 'model_aggregation_classifier'

    def __init__(self, list_models: Union[list, None] = None, aggregation_function: Union[Callable, str] = 'majority_vote', 
                 using_proba: bool = False, **kwargs) -> None:
        '''Initialization of the class (see ModelClass & ModelClassifierMixin for more arguments)

        Kwargs:
            list_models (list) : The list of model to be aggregated
            aggregation_function (Callable or str) : The aggregation function used
            using_proba (bool) : Which object is being aggregated (the probabilities or the predictions).
        Raises:
            ValueError : If the object aggregation_function is a str but not found in the dictionary dict_aggregation_function
            ValueError : If the object list_model has other model than model classifier (model_aggregation_classifier is only compatible with model classifier)
            ValueError : The multi_label attributes of the aggregated models are inconsistent with multi_label
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Get the aggregation function
        self.using_proba = using_proba
        dict_aggregation_function = {'majority_vote': {'function': self.majority_vote, 'using_proba': False, 'multi_label': False},
                                     'proba_argmax': {'function': self.proba_argmax, 'using_proba': True, 'multi_label': False},
                                     'all_predictions': {'function': self.all_predictions, 'using_proba': False, 'multi_label': True},
                                     'vote_labels': {'function': self.vote_labels, 'using_proba': False, 'multi_label': True}}

        if isinstance(aggregation_function, str):
            if aggregation_function not in dict_aggregation_function.keys():
                raise ValueError(f"The aggregation_function ({aggregation_function}) is not a valid option (must be chosen in {dict_aggregation_function.keys()})")
            using_proba_from_str: bool = dict_aggregation_function[aggregation_function]['using_proba'] # type: ignore
            multi_label_from_str:bool = dict_aggregation_function[aggregation_function]['multi_label'] # type: ignore
            if self.using_proba != using_proba_from_str:
                self.logger.warning(f"using_proba {self.using_proba} is incompatible with the selected aggregation function '{aggregation_function}'. We force using_proba to {using_proba_from_str}")
            if self.multi_label != multi_label_from_str:
                self.logger.warning(f"multi_label {self.multi_label} is incompatible with the selected aggregation function '{aggregation_function}'. We force multi_label to {multi_label_from_str}")
            self.using_proba = using_proba_from_str
            self.multi_label = multi_label_from_str
            aggregation_function = dict_aggregation_function[aggregation_function]['function'] # type: ignore

        # Manage aggregated models
        self.aggregation_function = aggregation_function
        
        self._manage_sub_models(list_models)

        if False in [isinstance(sub_model['model'], ModelClassifierMixin) for sub_model in self.sub_models]:
            raise ValueError(f"model_aggregation_classifier only accepts classifier models")

        # Check for multi-labels inconsistencies
        set_multi_label = {sub_model['model'].multi_label for sub_model in self.sub_models}
        if True in set_multi_label and not self.multi_label:
            raise ValueError(f"The multi_label attributes of the aggregated models are inconsistent with self.multi_label = {self.multi_label}.")

        self._check_trained()

    def _manage_sub_models(self, list_models: list) -> None:
        '''Populates the self.sub_models list

        Args:
            list_models (list): List of models or name of models
        '''
        sub_models = []
        if list_models is None:
            list_models = []
        for model in list_models:
            if isinstance(model, str):
                real_model, _ = utils_models.load_model(model)
                dict_model = {'name': model, 'model': real_model, 'init_trained': real_model.trained}
            else:
                dict_model = {'name': os.path.split(model.model_dir)[-1], 'model': model, 'init_trained': model.trained}
            sub_models.append(dict_model.copy())
        self.sub_models = sub_models.copy()

    def _check_trained(self):
        '''Checks and sets various attributes related to the fitting of underlying models

        Raises:
            TypeError : The classes of all the aggregated models are not of the same type
        '''
        # Check fitted
        models_trained = {sub_model['model'].trained for sub_model in self.sub_models}
        if len(models_trained) and False not in models_trained:
            self.trained = True
            self.nb_fit += 1

            # Set list_classes
            self.list_classes = list({label for sub_model in self.sub_models for label in sub_model['model'].list_classes})
            list_label_str = [label for label in self.list_classes if isinstance(label, str)]
            list_label_other = [int(label) for label in self.list_classes if label not in list_label_str]
            if len(list_label_str) > 0 and len(list_label_other) > 0:
                raise TypeError('The classes of all the aggregated models are not of the same type.')
            self.list_classes.sort()

            # Set dict_classes based on self.list_classes
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
            y_train_multi_label = True if len(y_train.iloc[0]) > 1 else False
        elif isinstance(y_train, np.ndarray):
            y_train_multi_label = True if y_train.shape != (len(x_train),) else False
        else:
            y_train_multi_label = False

        # We check input format
        x_train, y_train = self._check_input_format(x_train, y_train, fit_function=True)

        # Fit each model
        for sub_model in self.sub_models:
            model = sub_model['model']
            if not model.trained:
                if y_train_multi_label and not model.multi_label:
                    raise ValueError(f"Model {model} (model_name: {model.model_name}) needs y_train to be mono-label when fitting")
                if not y_train_multi_label and model.multi_label:
                    raise ValueError(f"Model {model}(model_name: {model.model_name}) needs y_train to be multi-labels when fitting")
                model.fit(x_train, y_train, **kwargs)

        self._check_trained()

    @utils.trained_needed
    def predict(self, x_test, return_proba: bool = False, **kwargs) -> np.ndarray:
        '''Prediction

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
            return_proba (bool): If the function should return the probabilities instead of the classes
        Returns:
            (np.ndarray): array of shape = [n_samples]
        '''
        # We decide whether to rely on each model's probas or their prediction
        if return_proba:
            return self.predict_proba(x_test)
        else:
            if self.using_proba:
                preds = self._get_probas_sub_models(x_test, **kwargs)
            else:
                preds = self._get_predictions_sub_models(x_test, **kwargs)
            return np.array([self.aggregation_function(array) for array in preds]) # type: ignore

    @utils.trained_needed
    def _get_probas_sub_models(self, x_test, **kwargs) -> np.ndarray:
        '''Recover the probabilities of each model being aggregated

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.ndarray): array of shape = [n_samples, nb_model, nb_classes]
        '''
        array_proba = np.array([self._predict_model_with_full_list_classes(sub_model['model'], x_test, return_proba=True) for sub_model in self.sub_models])
        array_proba = np.transpose(array_proba, (1, 0, 2))
        return array_proba

    @utils.trained_needed
    def _get_predictions_sub_models(self, x_test, **kwargs) -> np.ndarray:
        '''Recover the predictions of each model being aggregated

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.ndarray): not multi_label : array of shape = [n_samples, nb_model]
                          multi_label : array of shape = [n_samples, nb_model, n_classes]
        '''
        if self.multi_label:
            array_predict = np.array([self._predict_model_with_full_list_classes(sub_model['model'], x_test, return_proba=False) for sub_model in self.sub_models])
            array_predict = np.transpose(array_predict, (1, 0, 2))
        else:
            array_predict = np.array([sub_model['model'].predict(x_test) for sub_model in self.sub_models])
            array_predict = np.transpose(array_predict, (1, 0))
        return array_predict

    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Predicts the probabilities on the test set

        Args:
            x_test (?): array-like or sparse matrix of shape = [n_samples, n_features]
        Returns:
            (np.ndarray): array of shape = [n_samples, n_classes]
        '''
        probas = self._get_probas_sub_models(x_test, **kwargs)
        # The probas of all models are averaged
        return np.sum(probas, axis=1) / probas.shape[1]

    def _predict_model_with_full_list_classes(self, model, x_test, return_proba: bool = False) -> np.ndarray:
        '''For multi_label: Adds missing columns in the prediction of model (class missing in their list_classes)
        Or, if return_proba, adds a proba of zero to the missing classes in their list_classes

        Args:
            model (?): Model to use
            x_test (?): Array-like or sparse matrix of shape = [n_samples, n_features]
            return_proba (bool): If the function should return the probabilities instead of the classes
        Returns:
            (np.ndarray): The array with the missing columns added
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
            return np.array([[1 if one_pred == col else 0 for col in self.list_classes] for one_pred in pred])

    def proba_argmax(self, proba: np.ndarray):
        '''Gives the class corresponding to the argmax of the average of the given probabilities.

        Args:
            proba (np.ndarray): The probabilities of each model for each class, array of shape (nb_models, nb_classes) 
        Returns:
            The prediction
        '''
        proba_average = np.sum(proba, axis=0) / proba.shape[0]
        index_class = np.argmax(proba_average)
        return self.list_classes[index_class]

    def majority_vote(self, predictions: np.ndarray):
        '''Gives the class corresponding to the most present prediction in the given predictions. 
        In case of a tie, gives the first prediction (even if not the most present) 

        Args:
            predictions (np.ndarray) : The array containing the predictions of each model (shape (n_models)) 
        Returns:
            The prediction
        '''
        labels, counts = np.unique(predictions, return_counts=True)
        votes = [(label, count) for label, count in zip(labels, counts)]
        votes = sorted(votes, key=lambda x: x[1], reverse=True)
        if len(votes) > 1 and votes[0][1] == votes[1][1]:
            return predictions[0]
        else:
            return votes[0][0]

    def all_predictions(self, predictions: np.ndarray) -> np.ndarray:
        '''Calculates the sum of the arrays along axis 0 casts it to bool and then to int. 
        Expects a numpy array containing only zeroes and ones. 
        When used as an aggregation function, keeps all the prediction of each model (multi-labels)

        Args:
            predictions (np.ndarray) : Array of shape : (n_models, n_classes)
        Return:
            (np.ndarray) : The prediction
        '''
        return np.sum(predictions, axis=0, dtype=bool).astype(int)

    def vote_labels(self, predictions: np.ndarray) -> np.ndarray:
        '''Gives the result of majority_vote applied on the second axis. 
        When used as an aggregation_function, for each class, performs a majority vote for the aggregated models. 
        It gives a multi-labels result

        Args:
            (np.ndarray) : array of shape : (n_models, n_classes)
        Return:
            (np.ndarray) : prediction
        '''
        return np.apply_along_axis(self.majority_vote, 0, predictions)

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        if json_data is None:
            json_data = {}
         # Save each trained and unsaved model
        for sub_model in self.sub_models:
            if not sub_model['init_trained'] and sub_model['model'].trained:
                sub_model['model'].save()

        json_data['list_models_name'] = [sub_model['name'] for sub_model in self.sub_models]
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
        sub_models = self.sub_models
        delattr(self, "sub_models")
        delattr(self, "aggregation_function")
        super().save(json_data=json_data)
        setattr(self, "aggregation_function", aggregation_function)
        setattr(self, "sub_models", sub_models)

        # Add message in model_upload_instructions.md
        md_path = os.path.join(self.model_dir, f"model_upload_instructions.md")
        line = r"/!\/!\/!\/!\/!\   The aggregation model is a special model, please ensure that all sub-models and the aggregation model are manually saved together in order to be able to load it  /!\/!\/!\/!\/!\ "
        self.prepend_line(md_path, line)

    def prepend_line(self, file_name: str, line: str) -> None:
        ''' Insert given string as a new line at the beginning of a file

        Kwargs:
            file_name (str): Path to file
            line (str): line to insert
        '''
        with open(file_name, 'r') as original:
            data = original.read()
        with open(file_name, 'w') as modified:
            modified.write(line + "\n" + data)

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
        for attribute in ['x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'using_proba']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        list_models_name = configs.get('list_models_name', [])
        self._manage_sub_models(list_models_name)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")