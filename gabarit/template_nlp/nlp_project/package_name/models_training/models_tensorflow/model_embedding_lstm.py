#!/usr/bin/env python3

## Model embedding + LSTM
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
# - ModelEmbeddingLstm -> Model for predictions via embedding + LSTM


import os
import logging
import numpy as np
import dill as pickle
import seaborn as sns
from typing import Union, Any, List, Callable

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model as load_model_keras
from tensorflow.keras.layers import (LSTM, BatchNormalization, Bidirectional, Dense, Embedding,
                                     GlobalAveragePooling1D, GlobalMaxPooling1D, Input,
                                     SpatialDropout1D, add, concatenate)

from . import utils_deep_keras
from .model_keras import ModelKeras

sns.set(style="darkgrid")


class ModelEmbeddingLstm(ModelKeras):
    '''Model for prediction via embedding + LSTM'''

    _default_name = 'model_embedding_lstm'

    def __init__(self, max_sequence_length: int = 200, max_words: int = 100000,
                 padding: str = 'pre', truncating: str = 'post',
                 tokenizer_filters="’!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r\'\"", **kwargs) -> None:
        '''Initialization of the class (see ModelClass & ModelKeras for more arguments)

        Kwargs:
            max_sequence_length (int): Maximum number of words per sequence (ie. sentences)
            max_words (int): Maximum number of words for tokenization
            padding (str): Padding (add zeros) at the beginning ('pre') or at the end ('post') of the sequences
            truncating (str): Truncating the beginning ('pre') or the end ('post') of the sequences (if superior to max_sequence_length)
            tokenizer_filters (str): Filter to use by the tokenizer
        Raises:
            ValueError: If the object padding is not a valid choice (['pre', 'post'])
            ValueError: If the object truncating is not a valid choice (['pre', 'post'])
        '''
        if padding not in ['pre', 'post']:
            raise ValueError(f"The object padding ({padding}) is not a valid choice (['pre', 'post'])")
        if truncating not in ['pre', 'post']:
            raise ValueError(f"The object truncating ({truncating}) is not a valid choice (['pre', 'post'])")

        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        self.max_sequence_length = max_sequence_length
        self.max_words = max_words
        self.padding = padding
        self.truncating = truncating

        # Tokenizer set on fit
        self.tokenizer: Any = None
        self.tokenizer_filters = tokenizer_filters

    def _prepare_x_train(self, x_train) -> np.ndarray:
        '''Prepares the input data for the model. Called when fitting the model

        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
        Returns:
            (np.ndarray): Prepared data
        '''
        # Get tokenizer & fit on train
        self.tokenizer = Tokenizer(num_words=self.max_words, filters=self.tokenizer_filters)
        self.logger.info('Fitting the tokenizer')
        self.tokenizer.fit_on_texts(x_train)
        return self._get_sequence(x_train, self.tokenizer, self.max_sequence_length, padding=self.padding, truncating=self.truncating)

    def _prepare_x_test(self, x_test) -> np.ndarray:
        '''Prepares the input data for the model. Called when fitting the model

        Args:
            x_test (?): Array-like, shape = [n_samples, n_features]
        Returns:
            (np.ndarray): Prepared data
        '''
        # Get sequences on test (already fitted on train)
        return self._get_sequence(x_test, self.tokenizer, self.max_sequence_length, padding=self.padding, truncating=self.truncating)

    def _get_model(self, custom_tokenizer=None) -> Any:
        '''Gets a model structure - returns the instance model instead if already defined

        Kwargs:
            custom_tokenizer (?): Tokenizer (if different from the one of the class). Permits to manage "new embeddings"
        Returns:
            (Model): a Keras model
        '''
        # Return model if already set
        if self.model is not None:
            return self.model

        # Start by getting embedding matrix
        if custom_tokenizer is not None:
            embedding_matrix, embedding_size = self._get_embedding_matrix(custom_tokenizer)
        else:
            embedding_matrix, embedding_size = self._get_embedding_matrix(self.tokenizer)

        # Get input dim
        input_dim = embedding_matrix.shape[0]

        # Get model
        num_classes = len(self.list_classes)
        # Process
        LSTM_UNITS = 100
        DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
        words = Input(shape=(self.max_sequence_length,))
        x = Embedding(input_dim, embedding_size, weights=[embedding_matrix], trainable=False)(words)
        x = BatchNormalization(momentum=0.9)(x)
        x = SpatialDropout1D(0.5)(x)
        x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
        x = SpatialDropout1D(0.5)(x)
        hidden = concatenate([
            GlobalMaxPooling1D()(x),
            GlobalAveragePooling1D()(x),
        ])
        hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
        # Last layer
        activation = 'sigmoid' if self.multi_label else 'softmax'
        out = Dense(num_classes, activation=activation, kernel_initializer='glorot_uniform')(hidden)

        # Compile model
        model = Model(inputs=words, outputs=[out])
        lr = self.keras_params['learning_rate'] if 'learning_rate' in self.keras_params.keys() else 0.01
        decay = self.keras_params['decay'] if 'decay' in self.keras_params.keys() else 0.004
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

        # Add specific data
        json_data['max_sequence_length'] = self.max_sequence_length
        json_data['max_words'] = self.max_words
        json_data['padding'] = self.padding
        json_data['truncating'] = self.truncating
        json_data['tokenizer_filters'] = self.tokenizer_filters

        # Save tokenizer if not None & level_save > LOW
        if (self.tokenizer is not None) and (self.level_save in ['MEDIUM', 'HIGH']):
            # Manage paths
            tokenizer_path = os.path.join(self.model_dir, "embedding_tokenizer.pkl")
            # Save as pickle
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)

        # Save
        super().save(json_data=json_data)

    @classmethod
    def _init_new_instance_from_configs(cls, configs):
        '''Inits a new instance from a set of configurations

        Args:
            configs: a set of configurations of a model to be reloaded
        Returns:
            ModelClass: the newly generated class
        '''
        # Call parent
        model = super()._init_new_instance_from_configs(configs)

        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['max_sequence_length', 'max_words', 'padding', 'truncating', 'tokenizer_filters']:
            setattr(model, attribute, configs.get(attribute, getattr(model, attribute)))

        # Return the new model
        return model

    def _load_standalone_files(self, default_model_dir: Union[str, None] = None,  # type: ignore
                               tokenizer_path: Union[str, None] = None, *args, **kwargs):
        '''Loads standalone files for a newly created model via _init_new_instance_from_configs

        Kwargs:
            default_model_dir (str): a path to look for default file paths
                                     If None, standalone files path should all be provided
            tokenizer_path (str): Path to the tokenizer file
        Raises:
            ValueError: If the tokenizer file is not specified and can't be inferred
            FileNotFoundError: If the tokenizer file does not exist
        '''
        # Check if we are able to get all needed paths
        if default_model_dir is None and tokenizer_path is None:
            raise ValueError("The tokenizer file is not specified and can't be inferred")

        # Call parent
        super()._load_standalone_files(default_model_dir=default_model_dir, **kwargs)

        # Retrieve file paths
        if tokenizer_path is None:
            tokenizer_path = os.path.join(default_model_dir, "embedding_tokenizer.pkl")

        # Check paths exists
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"Can't find tokenizer file ({tokenizer_path})")

        # Reload tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
