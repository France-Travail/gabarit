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
from typing import Union
from datetime import datetime

from {{package_name}} import utils
from {{package_name}}.monitoring.model_logger import ModelLogger


class ModelClass:
    '''Parent class for the models'''

    _default_name = 'none'
    # Variable annotation : https://www.python.org/dev/peps/pep-0526/
    # Solves lots of typing errors, cf mypy
    list_classes: list
    dict_classes: dict

    # Not implemented :
    # -> fit
    # -> predict
    # -> predict_proba
    # -> inverse_transform
    # -> get_and_save_metrics

    def __init__(self, model_dir: Union[str, None] = None, model_name: Union[str, None] = None,
                 level_save: str = 'HIGH', **kwargs) -> None:
        '''Initialization of the parent class.

        Kwargs:
            model_dir (str): Folder where to save the model
                If None, creates a directory based on the model's name and the date (most common usage)
            model_name (str): The name of the model
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

        # Model type -> 'classifier' or 'object_detector' depending on the model
        self.model_type = None

        # Model name
        if model_name is None:
            self.model_name = self._default_name
        else:
            self.model_name = model_name

        # Model folder
        if model_dir is None:
            self.model_dir = self._get_model_dir()
        else:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if not os.path.isdir(model_dir):
                raise NotADirectoryError(f"{model_dir} is not a valid directory")
            self.model_dir = os.path.abspath(model_dir)

        # Other options
        self.level_save = level_save

        # is trained ?
        self.trained = False
        self.nb_fit = 0

    def fit(self, df_train, **kwargs) -> dict:
        '''Trains the model

        Args:
            df_train (pd.DataFrame): Train dataset
                Must contain file_path & file_class columns if classifier
                Must contain file_path & bboxes columns if object detector
        Returns:
            dict: Fit arguments, to be used with transfer learning fine-tuning
        '''
        raise NotImplementedError("'fit' needs to be overridden")

    def predict(self, df_test: pd.DataFrame, **kwargs) -> Union[np.ndarray, list]:
        '''Predictions on test set

        Args:
            df_test (pd.DataFrame): DataFrame to be predicted, with column file_path
        Returns:
            (np.ndarray | list): Array, shape = [n_samples, n_classes] or List of n_samples elements
        '''
        raise NotImplementedError("'predict' needs to be overridden")

    def predict_proba(self, df_test: pd.DataFrame, **kwargs) -> np.ndarray:
        '''Probabilities predicted on the test set

        Args:
            df_test (pd.DataFrame): DataFrame to be predicted, with column file_path
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        raise NotImplementedError("'predict_proba' needs to be overridden")

    def inverse_transform(self, y: Union[list, np.ndarray]) -> Union[list, tuple]:
        '''Gets the final format of prediction
            - Classification : classes from predictions
            - Object detections : list of bboxes per image

        Args:
            y (list | np.ndarray): Array-like
        Returns:
            List of classes if classifier
            List of bboxes if object detector
        '''
        raise NotImplementedError("'inverse_transform' needs to be overridden")

    def get_and_save_metrics(self, y_true, y_pred, list_files_x: Union[list, None] = None, type_data: str = '',
                             model_logger: Union[ModelLogger, None] = None) -> pd.DataFrame:
        '''Gets and saves the metrics of a model

        Args:
            y_true (?): Array-like [n_samples, 1] if classifier
                # If classifier, class of each image
                # If object detector, list of list of bboxes per image
                    bbox format : {'class': ..., 'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}
            y_pred (?): Array-like [n_samples, 1] if classifier
                # If classifier, class of each image
                # If object detector, list of list of bboxes per image
                    bbox format : {'class': ..., 'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}
        Kwargs:
            list_files_x (list): Input images file paths
            type_data (str): Type of dataset (validation, test, ...)
            model_logger (ModelLogger): Custom class to log the metrics with MLflow
        Returns:
            pd.DataFrame: The dataframe containing statistics
        '''
        raise NotImplementedError("'get_and_save_metrics' needs to be overridden")

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''

        # Manage paths
        pkl_path = os.path.join(self.model_dir, f"{self.model_name}.pkl")
        conf_path = os.path.join(self.model_dir, "configurations.json")

        # Save model & pipeline preprocessing si level_save > 'LOW'
        if self.level_save in ['MEDIUM', 'HIGH']:
            # TODO: use dill to get rid of  "can't pickle ..." errors
            with open(pkl_path, 'wb') as f:
                pickle.dump(self, f)

        # Save configuration JSON
        json_dict = {
            'mainteners': 'Agence DataServices',
            'date': datetime.now().strftime("%d/%m/%Y - %H:%M:%S"),  # Not the same as the folder's name
            'package_version': utils.get_package_version(),
            'model_name': self.model_name,
            'model_dir': self.model_dir,
            'model_type': self.model_type,
            'trained': self.trained,
            'nb_fit': self.nb_fit,
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
