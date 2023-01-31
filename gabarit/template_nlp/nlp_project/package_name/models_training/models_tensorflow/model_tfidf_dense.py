#!/usr/bin/env python3

## Model TF-IDF + Dense
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
# - ModelTfidfDense -> Model for predictions via TF-IDF + Dense

import os
import logging
import numpy as np
import dill as pickle
import seaborn as sns
from typing import Union, List, Callable, Any
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model as load_model_keras
from tensorflow.keras.layers import ELU, BatchNormalization, Dense, Dropout

from . import utils_deep_keras
from .model_keras import ModelKeras


sns.set(style="darkgrid")


class ModelTfidfDense(ModelKeras):
    '''Model for predictions via TF-IDF + Dense'''

    _default_name = 'model_tfidf_dense'

    def __init__(self, tfidf_params: Union[dict, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelClass & ModelKeras for more arguments).

        Kwargs:
            tfidf_params (dict) : Parameters for the tfidf
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        if tfidf_params is None:
            tfidf_params = {}
        self.tfidf = TfidfVectorizer(**tfidf_params)

    def _prepare_x_train(self, x_train) -> np.ndarray:
        '''Prepares the input data for the model. Called when fitting the model

        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
        Returns:
            (np.ndarray): Prepared data
        '''
        # Fit tfidf & return x transformed
        self.tfidf.fit(x_train)
        # TODO: Use of todense because tensorflow 2.3 does not support sparse data anymore
        return self.tfidf.transform(x_train).todense()

    def _prepare_x_test(self, x_test) -> np.ndarray:
        '''Prepares the input data for the model. Called when fitting the model

        Args:
            x_test (?): Array-like, shape = [n_samples, n_features]
        Returns:
            (np.ndarray): Prepared data
        '''
        # Get tf-idf & fit on train
        # TODO: Use of todense because tensorflow 2.3 does not support sparse data anymore
        return self.tfidf.transform(x_test).todense()

    def _get_model(self) -> Model:
        '''Gets a model structure - returns the instance model instead if already defined

        Returns:
            (Model): a Keras model
        '''
        # Return model if already set
        if self.model is not None:
            return self.model

        # Get input/output dimensions
        input_dim = len(self.tfidf.get_feature_names())
        num_classes = len(self.list_classes)

        # Process
        model = Sequential()

        model.add(Dense(128, activation=None, kernel_initializer='he_uniform', input_shape=(input_dim,)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(ELU(alpha=1.0))
        model.add(Dropout(0.5))

        model.add(Dense(64, activation=None, kernel_initializer='he_uniform'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(ELU(alpha=1.0))
        model.add(Dropout(0.5))

        model.add(Dense(32, activation=None, kernel_initializer='he_uniform'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(ELU(alpha=1.0))
        model.add(Dropout(0.5))

        # Last layer
        activation = 'sigmoid' if self.multi_label else 'softmax'
        model.add(Dense(num_classes, activation=activation, kernel_initializer='glorot_uniform'))

        # Compile model
        lr = self.keras_params.get('learning_rate', 0.002)
        decay = self.keras_params.get('decay', 0.0)
        self.logger.info(f"Learning rate: {lr}")
        self.logger.info(f"Decay: {decay}")
        optimizer = Adam(lr=lr, decay=decay)
        # loss = utils_deep_keras.f1_loss if self.multi_label else 'categorical_crossentropy'
        loss = 'binary_crossentropy' if self.multi_label else 'categorical_crossentropy'  # utils_deep_keras.f1_loss also possible if multi-labels
        metrics: List[Union[str, Callable]] = ['accuracy'] if not self.multi_label else ['categorical_accuracy', utils_deep_keras.f1, utils_deep_keras.precision, utils_deep_keras.recall]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        if self.logger.getEffectiveLevel() < logging.ERROR:
            model.summary()

        # Try to save model as png if level_save > 'LOW'
        if self.level_save in ['MEDIUM', 'HIGH']:
            self._save_model_png(model)

        # Return
        return model

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save configuration JSON
        if json_data is None:
            json_data = {}

        # Add tfidf params
        confs = self.tfidf.get_params()
        # Get rid of some non serializable conf
        for special_conf in ['dtype', 'base_estimator']:
            if special_conf in confs.keys():
                confs[special_conf] = str(confs[special_conf])
        json_data['tfidf_confs'] = confs

        # Save tfidf if not None & level_save > LOW
        if (self.tfidf is not None) and (self.level_save in ['MEDIUM', 'HIGH']):
            # Manage paths
            tfidf_path = os.path.join(self.model_dir, "tfidf_standalone.pkl")
            # Save as pickle
            with open(tfidf_path, 'wb') as f:
                pickle.dump(self.tfidf, f)

        # Save
        super().save(json_data=json_data)

    def _load_standalone_files(self, default_model_dir: Union[str, None] = None,  # type: ignore
                               tfidf_path: Union[str, None] = None, *args, **kwargs):
        '''Loads standalone files for a newly created model via _init_new_instance_from_configs

        Kwargs:
            default_model_dir (str): a path to look for default file paths
                                     If None, standalone files path should all be provided
            tfidf_path (str): Path to the TFIDF file
        Raises:
            ValueError: If the TFIDF file is not specified and can't be inferred
            FileNotFoundError: If the TFIDF file does not exist
        '''
        # Check if we are able to get all needed paths
        if default_model_dir is None and tfidf_path is None:
            raise ValueError("The TFIDF file is not specified and can't be inferred")

        # Call parent
        super()._load_standalone_files(default_model_dir=default_model_dir, **kwargs)

        # Retrieve file paths
        if tfidf_path is None:
            tfidf_path = os.path.join(default_model_dir, "tfidf_standalone.pkl")

        # Check paths exists
        if not os.path.isfile(tfidf_path):
            raise FileNotFoundError(f"Can't find the TFIDF file ({tfidf_path})")

        # Reload tfidf
        with open(tfidf_path, 'rb') as f:
            self.tfidf = pickle.load(f)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
