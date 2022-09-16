#!/usr/bin/env python3

## Definition of the parent class for the models
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
# - ModelClass -> Parent class for the models


import os
import time
import json
import logging
import numpy as np
import pandas as pd
import dill as pickle
from datetime import datetime
from typing import List, Union, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training import utils_models
from {{package_name}}.monitoring.model_logger import ModelLogger


class ModelClass:
    '''Parent class for the models'''

    _default_name = 'none'
    # Variable annotation : https://www.python.org/dev/peps/pep-0526/
    # Solves lots of typing errors, cf mypy
    multi_label: Union[bool, None]
    list_classes: Union[list, None]
    dict_classes: Union[dict, None]

    # Not implemented :
    # -> fit
    # -> predict
    # -> predict_proba
    # -> inverse_transform
    # -> get_and_save_metrics

    def __init__(self, model_dir: Union[str, None] = None, model_name: Union[str, None] = None,
                 x_col: Union[list, None] = None, y_col: Union[str, int, list, None] = None,
                 preprocess_pipeline: Union[ColumnTransformer, None] = None, level_save: str = 'HIGH', **kwargs) -> None:
        '''Initialization of the parent class.

        Kwargs:
            model_dir (str): Folder where to save the model
                If None, creates a directory based on the model's name and the date (most common usage)
            model_name (str): The name of the model
            x_col (list): Names of the columns used for the training - x
            y_col (str or int or list if multi-labels): Name of the model's target column(s) - y
            preprocess_pipeline (ColumnTransformer): The pipeline used for preprocessing. If None -> no preprocessing !
            level_save (str): Level of saving
                LOW: stats + configurations + logger keras - /!\\ The model can't be reused /!\\ -
                MEDIUM: LOW + hdf5 + pkl + plots
                HIGH: MEDIUM + predictions
        Raises:
            ValueError: If the object level_save is not a valid option (['LOW', 'MEDIUM', 'HIGH'])
            NotADirectoryError: If a provided model directory is not a directory (i.e. it's a file)
        '''
        if level_save not in ['LOW', 'MEDIUM', 'HIGH']:
            raise ValueError(f"The object level_save ({level_save}) is not a valid option (['LOW', 'MEDIUM', 'HIGH'])")

        # Get logger
        self.logger = logging.getLogger(__name__)

        # Model type -> 'classifier' or 'regressor' depending on the model
        self.model_type = None

        # Model name
        if model_name is None:
            self.model_name = self._default_name
        else:
            self.model_name = model_name

        # Names of the columns used
        self.x_col = x_col
        self.y_col = y_col
        # Can be None if reloading a model
        if x_col is None:
            self.logger.warning("Warning, the attribute x_col is not given! The model might not work as intended.")
        if y_col is None:
            self.logger.warning("Warning, the attribute y_col is not given! The model might not work as intended.")

        # Model folder
        if model_dir is None:
            self.model_dir = self._get_model_dir()
        else:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if not os.path.isdir(model_dir):
                raise NotADirectoryError(f"{model_dir} is not a valid directory")
            self.model_dir = os.path.abspath(model_dir)

        # Preprocessing pipeline
        self.preprocess_pipeline = preprocess_pipeline
        if self.preprocess_pipeline is not None:
            try:
                check_is_fitted(self.preprocess_pipeline)
            except NotFittedError as e:
                self.logger.error("The preprocessing pipeline hasn't been fitted !")
                self.logger.error(repr(e))
                raise NotFittedError()
            # We get the associated columns (and a check if there has been a fit is done)
            self.columns_in, self.mandatory_columns = utils_models.get_columns_pipeline(self.preprocess_pipeline)
        else:
            # We can't define a "no_preprocess" pipeline since we should fit it
            # So we take care of that at the first fit
            self.logger.warning("Warning, no preprocessing pipeline given !")
            self.columns_in, self.mandatory_columns = None, None

        # Other options
        self.level_save = level_save

        # is trained ?
        self.trained = False
        self.nb_fit = 0

    def fit(self, x_train, y_train, **kwargs) -> None:
        '''Trains the model

        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train (?): Array-like, shape = [n_samples, n_targets]
        '''
        raise NotImplementedError("'fit' needs to be overridden")

    def predict(self, x_test: pd.DataFrame, **kwargs) -> np.ndarray:
        '''Predictions on the test set

        Args:
            x_test (pd.DataFrame): DataFrame with the test data to be predicted
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        raise NotImplementedError("'predict' needs to be overridden")

    def predict_proba(self, x_test: pd.DataFrame, **kwargs) -> np.ndarray:
        '''Probabilities predicted on the test set

        Args:
            x_test (pd.DataFrame): DataFrame with the test data to be predicted
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        raise NotImplementedError("'predict_proba' needs to be overridden")

    def inverse_transform(self, y: Union[list, np.ndarray]) -> Union[list, tuple]:
        '''Gets the final format of prediction
            - Classification : classes from predictions
            - Regression : values (identity function)

        Args:
            y (list | np.ndarray): Array-like, shape = [n_samples,]
                   OR 1D array shape = [n_classes] (only one prediction)
        Returns:
            (?): Array, shape = [n_samples, ?]
        '''
        raise NotImplementedError("'inverse_transform' needs to be overridden")

    def get_and_save_metrics(self, y_true, y_pred, df_x: Union[pd.DataFrame, None] = None,
                             series_to_add: Union[List[pd.Series], None] = None, type_data: str = '',
                             model_logger: Union[ModelLogger, None] = None) -> pd.DataFrame:
        '''Gets and saves the metrics of a model

        Args:
            y_true (?): Array-like, shape = [n_samples, n_targets]
            y_pred (?): Array-like, shape = [n_samples, n_targets]
        Kwargs:
            df_x (pd.DataFrame or None): Input dataFrame used for the prediction
            series_to_add (list): List of pd.Series to add to the dataframe
            type_data (str): Type of dataset (validation, test, ...)
            model_logger (ModelLogger): Custom class to log the metrics with MLflow
        Returns:
            pd.DataFrame: The dataframe containing the statistics
        '''
        raise NotImplementedError("'get_and_save_metrics' needs to be overridden")

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''

        # Manage paths
        pkl_path = os.path.join(self.model_dir, f"{self.model_name}.pkl")
        preprocess_pipeline_path = os.path.join(self.model_dir, "preprocess_pipeline.pkl")
        conf_path = os.path.join(self.model_dir, "configurations.json")

        # Save the model & preprocessing pipeline if level_save > 'LOW'
        if self.level_save in ['MEDIUM', 'HIGH']:
            with open(pkl_path, 'wb') as f:
                pickle.dump(self, f)
            # Useful for reload_from_standalone, otherwise, saved as a class attribute
            with open(preprocess_pipeline_path, 'wb') as f:
                pickle.dump(self.preprocess_pipeline, f)

        # Saving JSON configuration
        json_dict = {
            'mainteners': 'Agence DataServices',
            'date': datetime.now().strftime("%d/%m/%Y - %H:%M:%S"),  # Not the same as the folder's name
            'package_version': utils.get_package_version(),
            'model_name': self.model_name,
            'model_dir': self.model_dir,
            'model_type': self.model_type,
            'trained': self.trained,
            'nb_fit': self.nb_fit,
            'x_col': self.x_col,
            'y_col': self.y_col,
            'columns_in': self.columns_in,
            'mandatory_columns': self.mandatory_columns,
            'level_save': self.level_save,
            'librairie': None,
        }
        # Merge json_data if not None
        if json_data is not None:
            # Priority given to json_data !
            json_dict = {**json_dict, **json_data}

        # Save conf
        with open(conf_path, 'w', encoding='{{default_encoding}}') as json_file:
            json.dump(json_dict, json_file, indent=4, cls=utils.NpEncoder)

        # Now, save a proprietes file for the model upload
        self._save_upload_properties(json_dict)

    def _save_upload_properties(self, json_dict: Union[dict, None] = None) -> None:
        '''Prepares a configuration file for a future export (e.g on an artifactory)

        Kwargs:
            json_dict: Configurations to save
        '''
        if json_dict is None:
            json_dict = {}

        # Manage paths
        proprietes_path = os.path.join(self.model_dir, "proprietes.json")
        vanilla_model_upload_instructions = os.path.join(utils.get_ressources_path(), 'model_upload_instructions.md')
        specific_model_upload_instructions = os.path.join(self.model_dir, "model_upload_instructions.md")

        # First, we define a list of "allowed" properties
        allowed_properties = ["mainteners", "date", "package_version", "model_name", "list_classes",
                              "librairie", "fit_time"]
        # Now we filter these properties
        final_dict = {k: v for k, v in json_dict.items() if k in allowed_properties}
        # Save
        with open(proprietes_path, 'w', encoding='{{default_encoding}}') as f:
            json.dump(final_dict, f, indent=4, cls=utils.NpEncoder)

        # Add instructions to upload a model to a storage solution (e.g. Artifactory)
        with open(vanilla_model_upload_instructions, 'r', encoding='{{default_encoding}}') as f:
            content = f.read()
        # TODO: to be improved
        new_content = content.replace('model_dir_path_identifier', os.path.abspath(self.model_dir))
        with open(specific_model_upload_instructions, 'w', encoding='{{default_encoding}}') as f:
            f.write(new_content)

    def _get_model_dir(self) -> str:
        '''Gets a folder where to save the model

        Returns:
            str: Path to the folder
        '''
        models_dir = utils.get_models_path()
        subfolder = os.path.join(models_dir, self.model_name)
        folder_name = datetime.now().strftime(f"{self.model_name}_%Y_%m_%d-%H_%M_%S")
        model_dir = os.path.join(subfolder, folder_name)
        if os.path.isdir(model_dir):
            time.sleep(1)  # Wait 1 second so that the 'date' changes...
            return self._get_model_dir()  # Get new directory name
        else:
            os.makedirs(model_dir)
        return model_dir

    def _check_input_format(self, x_input: Union[pd.DataFrame, np.ndarray], y_input: Union[pd.DataFrame, pd.Series, np.ndarray, None] = None,
                            fit_function: bool = False) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, pd.Series, np.ndarray, None]]:
        '''Checks the inputs of a function. We check the number of columns and the ordering.
        Warnings if :
            - Not the right columns
            - Columns not in the right order
        If fit and x_col and/or y_col is not defined -> warning + use the input columns
        We also set the pipeline, columns_in and mandatory_columns if equal to None

        Args:
            x_input (pd.DataFrame, np.ndarray): Array-like, shape = [n_samples, n_features]
        Kwargs:
            y_input (pd.DataFrame, pd.Series, np.ndarray): Array-like, shape = [n_samples, n_target]
                Mandatory if fit_function
            fit_function (bool): If it is a fit function
        Raises:
            AttributeError: If fit_function == True, but y_input is None
            ValueError: If one of the inputs hasn't the right number of columns
        Returns:
            (pd.DataFrame, np.ndarray): x_input, may be reordered if needed
            (pd.DataFrame, pd.Series, np.ndarray): y_input, may be reordered if needed
        '''
        # Getting some info first
        x_input_shape = x_input.shape[-1] if len(x_input.shape) > 1 else 1
        if y_input is not None:
            y_input_shape = y_input.shape[-1] if len(y_input.shape) > 1 else 1
        else:
            y_input_shape = 0  # not used

        # Manage fit_function = True
        if fit_function:
            if y_input is None:
                raise AttributeError("The argument y_input is mandatory if fit_function == True")
            if self.x_col is None:
                self.logger.warning("Warning, the attribute x_col was not given when creating the model")
                self.logger.warning("We set it now with the input data of the fit function")
                if hasattr(x_input, 'columns'):
                    self.x_col = list(x_input.columns)
                else:
                    self.x_col = [_ for _ in range(x_input_shape)]
            # Same thing for y_col
            if self.y_col is None:
                self.logger.warning("Warning, the attribute y_col was not given when creating the model")
                self.logger.warning("We set it now with the input data of the fit function")
                if hasattr(y_input, 'columns'):
                    self.y_col = list(y_input.columns)
                else:
                    self.y_col = [_ for _ in range(y_input_shape)]
                # If there is only one element, we get rid of the list
                if y_input_shape == 1:
                    self.y_col = self.y_col[0]
            # If pipeline, columns_in or mandatory_columns is None, sets it
            if self.preprocess_pipeline is None:  # ie no pipeline given when initializing the class
                preprocess_str = "no_preprocess"
                preprocess_pipeline = preprocess.get_pipeline(preprocess_str)  # Warning, the pipeline must be fitted
                preprocess_pipeline.fit(x_input)  # We fit the pipeline to set the necessary columns for the pipeline
                self.preprocess_pipeline = preprocess_pipeline
                self.columns_in, self.mandatory_columns = utils_models.get_columns_pipeline(self.preprocess_pipeline)

        # Checking x_input
        if self.x_col is None:
            self.logger.warning("Can't check the input format (x) because x_col is not set...")
        else:
            # Checking x_input format
            x_col_len = len(self.x_col)
            if x_input_shape != x_col_len:
                raise ValueError(f"Input data (x) is not in the right format ({x_input_shape} != {x_col_len})")
            # We check the presence of the columns
            if hasattr(x_input, 'columns'):
                can_reorder = True
                for col in self.x_col:
                    if col not in x_input.columns:
                        can_reorder = False
                        self.logger.warning(f"The column {col} is missing from the input (x)")
                # If we can't reorder, we write a warning message, otherwise we check if it is needed
                if not can_reorder:
                    self.logger.warning("The names of the columns do not match. The process continues since there is the right number of columns")
                else:
                    if list(x_input.columns) != self.x_col:
                        self.logger.warning("The input columns (x) are not in the right order -> automatic reordering !")
                        x_input = x_input[self.x_col]
            else:
                self.logger.warning("The input (x) does not have the 'columns' attribute -> can't check the ordering of the columns")

        # Checking y_input
        if y_input is not None:
            if self.y_col is None:
                self.logger.warning("Can't check the input format (y) because y_col is not set...")
            else:
                # Checking y_input format
                y_col_len = len(self.y_col) if type(self.y_col) == list else 1
                if y_input_shape != y_col_len:
                    raise ValueError(f"Input data (y) is not in the right format ({y_input_shape} != {y_col_len})")
                # We check the presence of the columns
                if hasattr(y_input, 'columns'):
                    can_reorder = True
                    target_cols = self.y_col if type(self.y_col) == list else [self.y_col]
                    for col in target_cols:
                        if col not in y_input.columns:
                            can_reorder = False
                            self.logger.warning(f"The column {col} is missing from the input (y)")
                    # If can't reorder we write a warning message, otherwise we check if it is needed
                    if not can_reorder:
                        self.logger.warning("The names of the columns do not match. The process continues since there is the right number of columns")
                    else:
                        if list(y_input.columns) != target_cols:
                            self.logger.warning("The input columns (y) are not in the right order -> automatic reordering !")
                            y_input = y_input[target_cols]
                else:
                    self.logger.warning("The input (y) does not have the 'columns' attribute -> can't check the ordering of the columns")

        # Return
        return x_input, y_input

    def display_if_gpu_activated(self) -> None:
        '''Displays if a GPU is being used'''
        if self._is_gpu_activated():
            ascii_art = '''
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*         (=========)                                                                                                            (=========)         *
*         |=========|                                                                                                            |=========|         *
*         |====_====|                                                                                                            |====_====|         *
*         |== / \ ==|                                                                                                            |== / \ ==|         *
*         |= / _ \ =|                                                                                                            |= / _ \ =|         *
*      _  |=| ( ) |=|                                                                                                         _  |=| ( ) |=|         *
*     /=\ |=|     |=| /=\                                                                                                    /=\ |=|     |=| /=\     *
*     |=| |=| GPU |=| |=|        _____ _____  _    _            _____ _______ _______      __  _______ ______ _____          |=| |=| GPU |=| |=|     *
*     |=| |=|  _  |=| |=|       / ____|  __ \| |  | |     /\   / ____|__   __|_   _\ \    / /\|__   __|  ____|  __ \         |=| |=|  _  |=| |=|     *
*     |=| |=| | | |=| |=|      | |  __| |__) | |  | |    /  \ | |       | |    | |  \ \  / /  \  | |  | |__  | |  | |        |=| |=| | | |=| |=|     *
*     |=| |=| | | |=| |=|      | | |_ |  ___/| |  | |   / /\ \| |       | |    | |   \ \/ / /\ \ | |  |  __| | |  | |        |=| |=| | | |=| |=|     *
*     |=| |=| | | |=| |=|      | |__| | |    | |__| |  / ____ \ |____   | |   _| |_   \  / ____ \| |  | |____| |__| |        |=| |=| | | |=| |=|     *
*     |=| |/  | |  \| |=|       \_____|_|     \____/  /_/    \_\_____|  |_|  |_____|   \/_/    \_\_|  |______|_____/         |=| |/  | |  \| |=|     *
*     |=|/    | |    \|=|                                                                                                    |=|/    | |    \|=|     *
*     |=/ ADS |_| ADS \=|                                                                                                    |=/ ADS |_| ADS \=|     *
*     |(_______________)|                                                                                                    |(_______________)|     *
*     |=| |_|__|__|_| |=|                                                                                                    |=| |_|__|__|_| |=|     *
*     |=|   ( ) ( )   |=|                                                                                                    |=|   ( ) ( )   |=|     *
*    /===\           /===\                                                                                                  /===\           /===\    *
*   |||||||         |||||||                                                                                                |||||||         |||||||   *
*   -------         -------                                                                                                -------         -------   *
*    (~~~)           (~~~)                                                                                                  (~~~)           (~~~)    *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            '''
        else:
            ascii_art = ''
        print(ascii_art)

    def _is_gpu_activated(self) -> bool:
        '''Checks if we use a GPU

        Returns:
            bool: whether GPU is available or not
        '''
        # By default, no GPU
        return False


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
