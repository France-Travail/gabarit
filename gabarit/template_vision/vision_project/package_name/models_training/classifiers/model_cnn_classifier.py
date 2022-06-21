#!/usr/bin/env python3

## CNN model - Classification
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
# - ModelCnnClassifier -> CNN model for classifiction


import os
import json
import shutil
import logging
import dill as pickle
from typing import Union, List, Callable

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model as load_model_keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.layers import (ELU, BatchNormalization, Dense, Dropout,
                                     Input, Conv2D, Flatten, AveragePooling2D)

from {{package_name}}.models_training.model_keras import ModelKeras
from {{package_name}}.models_training.classifiers.model_classifier import ModelClassifierMixin  # type: ignore


class ModelCnnClassifier(ModelClassifierMixin, ModelKeras):
    '''CNN model for classification'''

    _default_name = 'model_cnn_classifier'

    def __init__(self, **kwargs) -> None:
        '''Initialization of the class (see ModelClass, ModelKeras & ModelClassifierMixin for more arguments)'''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

    def _get_model(self) -> Model:
        '''Gets a model structure

        Returns:
            (Model): a Keras model
        '''
        # Get input/output dimensions
        input_shape = (self.width, self.height, self.depth)
        num_classes = len(self.list_classes)

        # Process
        input_layer = Input(shape=input_shape)

        # Feature extraction

        x = Conv2D(16, 3, padding='same', activation=None, kernel_initializer="he_uniform")(input_layer)
        x = BatchNormalization(momentum=0.99)(x)
        x = ELU(alpha=1.0)(x)
        x = AveragePooling2D(2, strides=2, padding='valid')(x)

        x = Conv2D(32, 3, padding='same', activation=None, kernel_initializer="he_uniform")(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = ELU(alpha=1.0)(x)
        x = AveragePooling2D(2, strides=2, padding='valid')(x)

        x = Conv2D(48, 3, padding='same', activation=None, kernel_initializer="he_uniform")(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = ELU(alpha=1.0)(x)
        x = AveragePooling2D(2, strides=2, padding='valid')(x)

        x = Conv2D(32, 3, padding='same', activation=None, kernel_initializer="he_uniform")(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = ELU(alpha=1.0)(x)
        x = AveragePooling2D(2, strides=2, padding='valid')(x)

        # Flatten

        x = Flatten()(x)

        # Classification

        x = Dense(64, activation=None, kernel_initializer="he_uniform")(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = ELU(alpha=1.0)(x)
        x = Dropout(0.2)(x)

        x = Dense(128, activation=None, kernel_initializer="he_uniform")(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = ELU(alpha=1.0)(x)
        x = Dropout(0.2)(x)

        x = Dense(128, activation=None, kernel_initializer="he_uniform")(x)
        x = BatchNormalization(momentum=0.99)(x)
        x = ELU(alpha=1.0)(x)
        x = Dropout(0.2)(x)

        # Last layer
        activation = 'softmax'
        out = Dense(num_classes, activation=activation, kernel_initializer='glorot_uniform')(x)

        # Set model
        model = Model(inputs=input_layer, outputs=[out])

        # Set optimizer
        lr = self.keras_params['learning_rate'] if 'learning_rate' in self.keras_params.keys() else 0.001
        decay = self.keras_params['decay'] if 'decay' in self.keras_params.keys() else 0.0
        self.logger.info(f"Learning rate: {lr}")
        self.logger.info(f"Decay: {decay}")
        optimizer = Adam(lr=lr, decay=decay)

        # Set loss & metrics
        loss = 'categorical_crossentropy'
        metrics: List[Union[str, Callable]] = ['accuracy']

        # Compile model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        if self.logger.getEffectiveLevel() < logging.ERROR:
            model.summary()

        # Try to save model as png if level_save > 'LOW'
        if self.level_save in ['MEDIUM', 'HIGH']:
            self._save_model_png(model)

        # Return
        return model

    def _get_preprocess_input(self) -> Union[Callable, None]:
        '''Gets the preprocessing to be used before feeding images to the NN

        Returns:
            (Callable | None): Preprocessing function
        '''
        return preprocess_input

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save configuration JSON
        if json_data is None:
            json_data = {}

        # Save
        super().save(json_data=json_data)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            hdf5_path (str): Path to hdf5 file
            preprocess_input_path (str): Path to preprocess input file
        Raises:
            ValueError: If configuration_path is None
            ValueError: If hdf5_path is None
            ValueError: If preprocess_input_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object hdf5_path is not an existing file
            FileNotFoundError: If the object preprocess_input_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        hdf5_path = kwargs.get('hdf5_path', None)
        preprocess_input_path = kwargs.get('preprocess_input_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if hdf5_path is None:
            raise ValueError("The argument hdf5_path can't be None")
        if preprocess_input_path is None:
            raise ValueError("The argument preprocess_input_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"The file {hdf5_path} does not exist")
        if not os.path.exists(preprocess_input_path):
            raise FileNotFoundError(f"The file {preprocess_input_path} does not exist")

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
        for attribute in ['model_type', 'list_classes', 'dict_classes', 'level_save',
                          'batch_size', 'epochs', 'validation_split', 'patience',
                          'width', 'height', 'depth', 'color_mode', 'in_memory',
                          'data_augmentation_params', 'nb_train_generator_images_to_save',
                          'keras_params']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload model
        self.model = load_model_keras(hdf5_path, custom_objects=self.custom_objects)

        # Reload preprocess_input
        with open(preprocess_input_path, 'rb') as f:
            self.preprocess_input = pickle.load(f)

        # Save best hdf5 in new folder
        new_hdf5_path = os.path.join(self.model_dir, 'best.hdf5')
        shutil.copyfile(hdf5_path, new_hdf5_path)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
