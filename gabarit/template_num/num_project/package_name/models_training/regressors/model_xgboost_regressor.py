#!/usr/bin/env python3

## Xgboost model
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
# - ModelXgboostRegressor -> Xgboost model for regression


import os
import json
import logging
import numpy as np
import pandas as pd
import dill as pickle
from typing import Union
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from {{package_name}} import utils
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.regressors.model_regressor import ModelRegressorMixin  # type: ignore


class ModelXgboostRegressor(ModelRegressorMixin, ModelClass):
    '''Xgboost model for regression'''

    _default_name = 'model_xgboost_regressor'

    def __init__(self, xgboost_params: Union[dict, None] = None, early_stopping_rounds: int = 5, validation_split: float = 0.2, **kwargs) -> None:
        '''Initialization of the class  (see ModelClass & ModelRegressorMixin for more arguments)

        Kwargs:
            xgboost_params (dict): Parameters for the Xgboost
                -> https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
            early_stopping_rounds (int): Number of rounds for early stopping
            validation_split (float): Validation split fraction.
                Only used if not validation dataset in the fit input
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Set parameters
        if xgboost_params is None:
            xgboost_params = {}
        self.xgboost_params = xgboost_params
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_split = validation_split

        # Set objective (if not in params) & init. model
        if 'objective' not in self.xgboost_params.keys():
            self.xgboost_params['objective'] = 'reg:squarederror'
            #  List of objectives https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
        self.model = XGBRegressor(**self.xgboost_params)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, with_shuffle: bool = True, **kwargs) -> None:
        '''Trains the model
           **kwargs permits compatibility with Keras model

        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train (?): Array-like, shape = [n_samples,]
            x_valid (?): Array-like, shape = [n_samples, n_features]
            y_valid (?): Array-like, shape = [n_samples,]
        Kwargs:
            with_shuffle (bool): If x, y must be shuffled before fitting
        Raises:
            RuntimeError: If the model is already fitted
        '''
        # TODO: Check if we can continue the training of a xgboost
        if self.trained:
            self.logger.error("We can't train again a xgboost model")
            self.logger.error("Please train a new model")
            raise RuntimeError("We can't train again a xgboost model")

        # We check input format
        x_train, y_train = self._check_input_format(x_train, y_train, fit_function=True)
        # If there is a validation set, we also check the format (but fit_function to False)
        if y_valid is not None and x_valid is not None:
            x_valid, y_valid = self._check_input_format(x_valid, y_valid, fit_function=False)
        # Otherwise, we do a random split
        else:
            self.logger.warning(f"Warning, no validation dataset. We split the training set (fraction valid = {self.validation_split})")
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=self.validation_split)

        # Shuffle x, y if wanted
        if with_shuffle:
            p = np.random.permutation(len(x_train))
            x_train = np.array(x_train)[p]
            y_train = np.array(y_train)[p]
        # Else still transform to numpy array
        else:
            x_train = np.array(x_train)
            y_train = np.array(y_train)

        # Also get x_valid & y_valid as numpy
        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        # Set eval set and train
        eval_set = [(x_train, y_train), (x_valid, y_valid)]  # If there’s more than one item in eval_set, the last entry will be used for early stopping.
        prior_objective = self.model.objective
        self.model.fit(x_train, y_train, eval_set=eval_set, early_stopping_rounds=self.early_stopping_rounds, verbose=True)
        post_objective = self.model.objective
        if prior_objective != post_objective:
            self.logger.warning("Warning: the objective function was automatically changed by XGBOOST")
            self.logger.warning(f"Before: {prior_objective}")
            self.logger.warning(f"After: {post_objective}")

        # Set trained
        self.trained = True
        self.nb_fit += 1

    @utils.trained_needed
    def predict(self, x_test: pd.DataFrame, return_proba: bool = False, **kwargs) -> np.ndarray:
        '''Predictions on test set

        Args:
            x_test (pd.DataFrame): DataFrame with the test data to be predicted
        Kwargs:
            return_proba (bool): Present for compatibility with other models. Raises an error if True
        Raises:
            ValueError: If return_proba is True
        Returns:
            (np.ndarray): Array, shape = [n_samples,]
        '''
        # Manage errors
        if return_proba:
            raise ValueError("Models of type model_xgboost_regressor can't handle probabilities")
        # We check input format
        x_test, _ = self._check_input_format(x_test)
        # Warning, "The method returns the model from the last iteration"
        # But : "Predict with X. If the model is trained with early stopping, then best_iteration is used automatically."
        y_pred = self.model.predict(x_test)
        return y_pred

    @utils.trained_needed
    def predict_proba(self, x_test: pd.DataFrame, **kwargs) -> np.ndarray:
        '''Predicts the probabilities on the test set. Here for compatibility

        Args:
            x_test (pd.DataFrame): Array-like, shape = [n_samples]
        Raises:
            ValueError: Model_xgboost_regressor does not implement predict_proba
        Returns:
            (np.ndarray): Array, shape = [n_samples,]
        '''
        # For compatibility
        raise ValueError("Models of type model_xgboost_regressor do not implement the method predict_proba")

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save model
        if json_data is None:
            json_data = {}

        json_data['librairie'] = 'xgboost'
        json_data['xgboost_params'] = self.xgboost_params
        json_data['early_stopping_rounds'] = self.early_stopping_rounds
        json_data['validation_split'] = self.validation_split

        # Save xgboost standalone
        if self.level_save in ['MEDIUM', 'HIGH']:
            if self.trained:
                save_path = os.path.join(self.model_dir, 'xbgoost_standalone.model')
                self.model.save_model(save_path)
            else:
                self.logger.warning("Can't save the XGboost in standalone because it hasn't been already fitted")

        # Save
        super().save(json_data=json_data)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            xgboost_path (str): Path to standalone xgboost
            preprocess_pipeline_path (str): Path to preprocess pipeline
        Raises:
            ValueError: If configuration_path is None
            ValueError: If xgboost_path is None
            ValueError: If preprocess_pipeline_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object xgboost_path is not an existing file
            FileNotFoundError: If the object preprocess_pipeline_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        xgboost_path = kwargs.get('xgboost_path', None)
        preprocess_pipeline_path = kwargs.get('preprocess_pipeline_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if xgboost_path is None:
            raise ValueError("The argument xgboost_path can't be None")
        if preprocess_pipeline_path is None:
            raise ValueError("The argument preprocess_pipeline_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(xgboost_path):
            raise FileNotFoundError(f"The file {xgboost_path} does not exist")
        if not os.path.exists(preprocess_pipeline_path):
            raise FileNotFoundError(f"The file {preprocess_pipeline_path} does not exist")

        # Load confs
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        # Set class vars
        # self.model_name = # Keep the created name
        # self.model_dir = # Keep the created folder
        self.nb_fit = configs.get('nb_fit', 1)  # Consider one unique fit by default
        self.trained = configs.get('trained', True)  # Consider trained by default
        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['model_type', 'x_col', 'y_col', 'columns_in', 'mandatory_columns',
                          'level_save', 'xgboost_params', 'early_stopping_rounds', 'validation_split']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload xgboost model
        self.model.load_model(xgboost_path)  # load data

        # Reload pipeline preprocessing
        with open(preprocess_pipeline_path, 'rb') as f:
            self.preprocess_pipeline = pickle.load(f)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
