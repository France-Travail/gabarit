#!/usr/bin/env python3

## Generic model for sklearn pipeline
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
# - ModelPipeline -> Generic model for sklearn pipeline

import os
import dill as pickle
import logging
import numpy as np
import pandas as pd
from typing import Union, Any

from sklearn.pipeline import Pipeline

from ... import utils
from ..model_class import ModelClass


class ModelPipeline(ModelClass):
    '''Generic model for sklearn pipeline'''

    _default_name = 'model_pipeline'

    # Probably need to be overridden, depending on your model :
    # -> predict_proba (predict on new content - returns probas) -> some pipelines do not provide proba, or may have specificities
    # -> save (specific save instructions)
    # -> _init_new_instance_from_configs (loads model attributes - for a newly created model)
    # -> _load_standalone_files (loads standalone files - for a newly created model) -> add pipeline elements

    def __init__(self, pipeline: Union[Pipeline, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelClass for more arguments)

        Kwargs:
            pipeline (Pipeline): Pipeline to use
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Manage model (to implement for children class)
        self.pipeline = pipeline

    def fit(self, x_train, y_train, **kwargs) -> None:
        '''Trains the model
           **kwargs permits compatibility with Keras model
        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train (?): Array-like, shape = [n_samples, n_targets]
        Raises:
            RuntimeError: If the model is already fitted
        '''
        if self.trained:
            self.logger.error("We can't train again a pipeline sklearn model")
            self.logger.error("Please train a new model")
            raise RuntimeError("We can't train again a pipeline sklearn model")

        # We "only" check if no multi-classes multi-labels (which can't be managed by most SKLEARN pipelines)
        if self.multi_label:
            df_tmp = pd.DataFrame(y_train)
            for col in df_tmp:
                uniques = df_tmp[col].unique()
                if len(uniques) > 2:
                    self.logger.warning(' - /!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\ - ')
                    self.logger.warning("Most sklearn pipelines can't manage multi-classes/multi-labels")
                    self.logger.warning(' - /!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\/!\\ - ')
                    # We "let" the process crash by itself
                    break

        # Fit pipeline
        self.pipeline.fit(x_train, y_train)

        # Set list classes
        if not self.multi_label:
            self.list_classes = list(self.pipeline.classes_)
        # TODO : check pipeline.classes_ for multi-labels
        else:
            if hasattr(y_train, 'columns'):
                self.list_classes = list(y_train.columns)
            else:
                self.logger.warning(
                    "Can't read the name of the columns of y_train -> inverse transformation won't be possible"
                )
                # We still create a list of classes in order to be compatible with other functions
                self.list_classes = [str(_) for _ in range(pd.DataFrame(y_train).shape[1])]

        # Set dict_classes based on list classes
        self.dict_classes = {i: col for i, col in enumerate(self.list_classes)}

        # Set trained
        self.trained = True
        self.nb_fit += 1

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict(self, x_test, return_proba: bool = False, **kwargs) -> np.ndarray:
        '''Predictions on test set

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
        Kwargs:
            return_proba (bool): If the function should return the probabilities instead of the classes (Keras compatibility)
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        if not return_proba:
            return np.array(self.pipeline.predict(x_test))
        else:
            return self.predict_proba(x_test)

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Predicts probabilities on the test dataset

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        probas = np.array(self.pipeline.predict_proba(x_test))
        # Very specific fix: in some cases, with OvR, strategy, all estimators return 0, which generates a division per 0 when normalizing
        # Hence, we replace NaNs with 1 / nb_classes
        if not self.multi_label:
            probas = np.nan_to_num(probas, nan=1/len(self.list_classes))
        # If use of MultiOutputClassifier ->  returns probabilities of 0 and 1 for all elements and all classes
        # Same thing for some base models
        # Correction in case where we detect a shape of length > 2 (ie. equals to 3)
        # Reminder : we do not manage multi-labels/multi-classes
        if len(probas.shape) > 2:
            probas = np.swapaxes(probas[:, :, 1], 0, 1)
        return probas

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save model
        if json_data is None:
            json_data = {}

        json_data['librairie'] = 'scikit-learn'

        # Add each pipeline steps' conf
        if self.pipeline is not None:
            for step in self.pipeline.steps:
                name = step[0]
                confs = step[1].get_params()
                # Get rid of some non serializable conf
                for special_conf in ['dtype', 'base_estimator', 'estimator', 'estimator__base_estimator',
                                     'estimator__estimator', 'estimator__estimator__base_estimator']:
                    if special_conf in confs.keys():
                        confs[special_conf] = str(confs[special_conf])
                json_data[f'{name}_confs'] = confs

        # Save
        super().save(json_data=json_data)

        # Save model standalone if wanted & pipeline is not None & level_save > 'LOW'
        if self.pipeline is not None and self.level_save in ['MEDIUM', 'HIGH']:
            pkl_path = os.path.join(self.model_dir, "sklearn_pipeline_standalone.pkl")
            # Save model
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.pipeline, f)

    def _load_standalone_files(self, default_model_dir: Union[str, None] = None,
                               sklearn_pipeline_path: Union[str, None] = None, *args, **kwargs):
        '''Loads standalone files for a newly created model via _init_new_instance_from_configs

        Kwargs:
            default_model_dir (str): a path to look for default file paths
                                     If None, standalone files path should all be provided
            sklearn_pipeline_path (str): Path to the sklearn pipeline
                                         If None, we'll use the default path if default_model_dir is not None
        Raises:
            ValueError: If the sklearn pipeline is not specified and can't be inferred
            FileNotFoundError: If the sklearn pipeline path does not exist
        '''
        # Check if we are able to get all needed paths
        if default_model_dir is None and sklearn_pipeline_path is None:
            raise ValueError("Sklearn pipeline path is not specified and can't be inferred")

        # Retrieve file paths
        if sklearn_pipeline_path is None:
            sklearn_pipeline_path = os.path.join(default_model_dir, "sklearn_pipeline_standalone.pkl")

        # Check paths exists
        if not os.path.isfile(sklearn_pipeline_path):
            raise FileNotFoundError(f"Can't find sklearn pipeline path ({sklearn_pipeline_path})")

        # Reload sklearn pipeline
        with open(sklearn_pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
