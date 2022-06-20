#!/usr/bin/env python3

## K-nearest Neighbors model
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
# - ModelKNNRegressor -> K-nearest Neighbors model for regression


import os
import json
import logging
import dill as pickle
from typing import Union

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

from {{package_name}}.models_training.model_pipeline import ModelPipeline
from {{package_name}}.models_training.regressors.model_regressor import ModelRegressorMixin  # type: ignore


class ModelKNNRegressor(ModelRegressorMixin, ModelPipeline):
    '''K-nearest Neighbors model for regression'''

    _default_name = 'model_knn_regressor'

    def __init__(self, knn_params: Union[dict, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelPipeline, ModelClass & ModelRegressorMixin for more arguments)

        Kwargs:
            knn_params (dict) : Parameters for the K-nearest Neighbors
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Manage model
        if knn_params is None:
            knn_params = {}
        self.knn = KNeighborsRegressor(**knn_params)
        # We define a pipeline in order to be compatible with other models
        self.pipeline = Pipeline([('knn', self.knn)])

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            sklearn_pipeline_path (str): Path to standalone pipeline
            preprocess_pipeline_path (str): Path to preprocess pipeline
        Raises:
            ValueError: If configuration_path is None
            ValueError: If sklearn_pipeline_path is None
            ValueError: If preprocess_pipeline_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object sklearn_pipeline_path is not an existing file
            FileNotFoundError: If the object preprocess_pipeline_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        sklearn_pipeline_path = kwargs.get('sklearn_pipeline_path', None)
        preprocess_pipeline_path = kwargs.get('preprocess_pipeline_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if sklearn_pipeline_path is None:
            raise ValueError("The argument sklearn_pipeline_path can't be None")
        if preprocess_pipeline_path is None:
            raise ValueError("The argument preprocess_pipeline_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(sklearn_pipeline_path):
            raise FileNotFoundError(f"The file {sklearn_pipeline_path} does not exist")
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
                          'level_save']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload pipeline model
        with open(sklearn_pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)

        # Reload pipeline elements
        self.knn = self.pipeline['knn']

        # Reload pipeline preprocessing
        with open(preprocess_pipeline_path, 'rb') as f:
            self.preprocess_pipeline = pickle.load(f)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
