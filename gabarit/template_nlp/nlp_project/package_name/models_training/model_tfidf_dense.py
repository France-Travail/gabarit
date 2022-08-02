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
import json
import pickle
import shutil
import logging
import numpy as np
import seaborn as sns
from typing import Union, List, Callable
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model as load_model_keras
from tensorflow.keras.layers import ELU, BatchNormalization, Dense, Dropout

from {{package_name}} import utils
from {{package_name}}.models_training import utils_deep_keras
from {{package_name}}.models_training.model_keras import ModelKeras


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

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, experimental_version: bool = False, **kwargs) -> np.ndarray:
        '''Probabilies predictions on the test dataset.

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples]
        Kwargs:
            experimental_version (bool): If an experimental (but faster) version must be used
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        # Prepare input
        x_test = self._prepare_x_test(x_test)
        # Process
        if experimental_version:
            return self.experimental_predict_proba(x_test)
        else:
            return self.model.predict(x_test, batch_size=128, verbose=1)  # type: ignore

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
        '''Gets a model structure

        Returns:
            (Model): a Keras model
        '''
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
                # TODO: use dill to get rid of  "can't pickle ..." errors
                pickle.dump(self.tfidf, f)

        # Save
        super().save(json_data=json_data)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            hdf5_path (str): Path to hdf5 file
            tf_idf_path (str): Path to tfidf file
        Raises:
            ValueError: If configuration_path is None
            ValueError: If hdf5_path is None
            ValueError: If tf_idf_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object hdf5_path is not an existing file
            FileNotFoundError: If the object tf_idf_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        hdf5_path = kwargs.get('hdf5_path', None)
        tfidf_path = kwargs.get('tfidf_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if hdf5_path is None:
            raise ValueError("The argument hdf5_path can't be None")
        if tfidf_path is None:
            raise ValueError("The argument tfidf_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"The file {hdf5_path} does not exist")
        if not os.path.exists(tfidf_path):
            raise FileNotFoundError(f"The file {tfidf_path} does not exist")

        # Load confs
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        # Can't set int as keys in json, so need to cast it after reloading
        # dict_classes keys are always ints
        if 'dict_classes' in configs.keys():
            configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}
        elif 'list_classes' in configs.keys():
            configs['dict_classes'] = {i: col for i, col in enumerate(configs['list_classes'])}

        # Set class vars
        # self.model_name = # Keep the created name
        # self.model_dir = # Keep the created folder
        self.nb_fit = configs.get('nb_fit', 1)  # Consider one unique fit by default
        self.trained = configs.get('trained', True)  # Consider trained by default
        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['x_col', 'y_col',
                          'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'batch_size', 'epochs', 'validation_split', 'patience',
                          'nb_iter_keras', 'keras_params', 'embedding_name']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload model
        self.model = load_model_keras(hdf5_path, custom_objects=self.custom_objects)

        # Save best hdf5 in new folder
        new_hdf5_path = os.path.join(self.model_dir, 'best.hdf5')
        shutil.copyfile(hdf5_path, new_hdf5_path)

        # Reload tfidf
        with open(tfidf_path, 'rb') as f:
            self.tfidf = pickle.load(f)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
