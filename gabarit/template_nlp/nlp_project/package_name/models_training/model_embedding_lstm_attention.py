#!/usr/bin/env python3

## Model embedding + LSTM + Attention
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


# ** EXPERIMENTAL **
# ** EXPERIMENTAL **
# ** EXPERIMENTAL **


import os
import json
import pickle
import logging
import shutil
import numpy as np
import seaborn as sns
from typing import Union, Any, List, Callable

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model as load_model_keras
from tensorflow.keras.layers import (Dense, Input, Embedding, GlobalMaxPooling1D,
                                     GlobalAveragePooling1D, BatchNormalization, LSTM,
                                     GRU, SpatialDropout1D, Bidirectional, concatenate)

from {{package_name}} import utils
from {{package_name}}.models_training import utils_deep_keras
from {{package_name}}.models_training.model_keras import ModelKeras
from {{package_name}}.models_training.utils_deep_keras import AttentionWithContext

sns.set(style="darkgrid")


class ModelEmbeddingLstmAttention(ModelKeras):
    '''Model for predictions via embedding + LSTM + Attention'''

    _default_name = 'model_embedding_lstm_attention'

    def __init__(self, max_sequence_length: int = 200, max_words: int = 100000,
                 padding: str = 'pre', truncating: str = 'post',
                 tokenizer_filters: str = "’!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r\'\"", **kwargs) -> None:
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

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, with_new_embedding: bool = False, experimental_version: bool = False, **kwargs) -> np.ndarray:
        '''Predicts probabilities on the test dataset

        Warning, this provides probabilities for a single model. If we use nb_iter > 1, we must use predict(return_proba=True)

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
        Kwargs:
            with_new_embedding (bool): If we use a new embedding matrix
            experimental_version (bool): If an experimental (but faster) version must be used
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        # If not with_new_embedding, just get classic predictions
        if not with_new_embedding:
            # Prepare input
            x_test = self._prepare_x_test(x_test)
            # Process
            if experimental_version:
                return self.experimental_predict_proba(x_test)
            else:
                return self.model.predict(x_test, batch_size=128, verbose=1)  # type: ignore
        # Else, get new tokenizer/embedding matrix
        else:
            # Get tokenizer & fit on test
            tokenizer = Tokenizer(num_words=self.max_words, filters=self.tokenizer_filters)
            self.logger.info('Fitting the tokenizer')
            tokenizer.fit_on_texts(x_test)

            # Create new model with new embedding
            tmp_model = self._get_model(custom_tokenizer=tokenizer)
            # Add 'old model' weights
            for i, layer in enumerate(self.model.layers[2:]):  # type: ignore
                tmp_model.layers[i + 2].set_weights(layer.get_weights())

            # Prepare input
            x_test = self._get_sequence(x_test, tokenizer, self.max_sequence_length, padding=self.padding, truncating=self.truncating)
            # Get predictions
            if experimental_version:
                return self.experimental_predict_proba(x_test)
            else:
                return tmp_model.predict(x_test, batch_size=128, verbose=1)

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
        '''Gets a model structure

        Kwargs:
            custom_tokenizer (?): Tokenizer (if different from the one of the class). Permits to manage "new embeddings"
        Returns:
            (Model): a Keras model
        '''
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
        words = Input(shape=(self.max_sequence_length,))
        # trainable=True to finetune the model
        # words = Input(shape=(None,))
        # x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
        x = Embedding(input_dim, embedding_size, weights=[embedding_matrix], trainable=False)(words)
        x = BatchNormalization(momentum=0.9)(x)
        x = SpatialDropout1D(0.5)(x)
        # LSTM and GRU will default to CuDNNLSTM and CuDNNGRU if all conditions are met:
        # - activation = 'tanh'
        # - recurrent_activation = 'sigmoid'
        # - recurrent_dropout = 0
        # - unroll = False
        # - use_bias = True
        # - Inputs, if masked, are strictly right-padded
        # - reset_after = True (GRU only)
        # /!\ https://stackoverflow.com/questions/60468385/is-there-cudnnlstm-or-cudnngru-alternative-in-tensorflow-2-0
        x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)  # returns a sequence of vectors of dimension 32
        x = Bidirectional(GRU(LSTM_UNITS, return_sequences=True))(x)  # returns a sequence of vectors of dimension 32

        att = AttentionWithContext()(x)
        avg_pool1 = GlobalAveragePooling1D()(x)
        max_pool1 = GlobalMaxPooling1D()(x)

        x = concatenate([att, avg_pool1, max_pool1])
        # Last layer
        activation = 'sigmoid' if self.multi_label else 'softmax'
        out = Dense(num_classes, activation=activation, kernel_initializer='glorot_uniform')(x)

        # Compile model
        model = Model(inputs=words, outputs=[out])
        lr = self.keras_params['learning_rate'] if 'learning_rate' in self.keras_params.keys() else 0.001
        decay = self.keras_params['decay'] if 'decay' in self.keras_params.keys() else 0.0
        self.logger.info(f"Learning rate: {lr}")
        self.logger.info(f"Decay: {decay}")
        optimizer = Adam(lr=lr, decay=decay)
        loss = utils_deep_keras.f1_loss if self.multi_label else 'categorical_crossentropy'
        # loss = 'binary_crossentropy' if self.multi_label else 'categorical_crossentropy'  # utils_deep_keras.f1_loss also possible if multi-labels
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
                # TODO: use dill to get rid of  "can't pickle ..." errors
                pickle.dump(self.tokenizer, f)

        # Save
        super().save(json_data=json_data)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            hdf5_path (str): Path to hdf5 file
            tokenizer_path (str): Path to tokenizer file
        Raises:
            ValueError: If configuration_path is None
            ValueError: If hdf5_path is None
            ValueError: If tokenizer_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object hdf5_path is not an existing file
            FileNotFoundError: If the object tokenizer_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        hdf5_path = kwargs.get('hdf5_path', None)
        tokenizer_path = kwargs.get('tokenizer_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if hdf5_path is None:
            raise ValueError("The argument hdf5_path can't be None")
        if tokenizer_path is None:
            raise ValueError("The argument tokenizer_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"The file {hdf5_path} does not exist")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"The file {tokenizer_path} does not exist")

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
        for attribute in ['x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'batch_size', 'epochs', 'validation_split', 'patience',
                          'embedding_name', 'max_sequence_length', 'max_words', 'padding',
                          'truncating', 'tokenizer_filters', 'nb_iter_keras', 'keras_params']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload model
        self.model = load_model_keras(hdf5_path, custom_objects=self.custom_objects)

        # Save best hdf5 in new folder
        new_hdf5_path = os.path.join(self.model_dir, 'best.hdf5')
        shutil.copyfile(hdf5_path, new_hdf5_path)

        # Reload tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
