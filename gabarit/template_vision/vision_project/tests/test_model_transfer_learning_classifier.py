#!/usr/bin/env python3
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

# Libs unittest
import unittest
from unittest.mock import Mock
from unittest.mock import patch

# Utils libs
import os
import json
import shutil
import keras
import numpy as np
import pandas as pd
from typing import Any

import tensorflow
import tensorflow as tf

from {{package_name}} import utils
from {{package_name}}.models_training import utils_deep_keras
from {{package_name}}.models_training.classifiers.model_transfer_learning_classifier import ModelTransferLearningClassifier


# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)
    tensorflow.keras.backend.clear_session()

def download_url_crash(x, y):
    raise ConnectionError("error")

# If there are no access to a base model (eg. VGG16) for the transfer learning, we can mock it with this class
# Here, we use it to speed up the tests
class ModelMockTransferLearningClassifier(ModelTransferLearningClassifier):
    '''We mock _get_model in order not to depend on a transfer learning model (which needs internet access)'''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    def _get_model(self) -> Any:
        '''Gets a model structure'''
        input_shape = (self.width, self.height, self.depth)
        num_classes = len(self.list_classes)
        input_layer = tf.keras.layers.Input(shape=input_shape)
        base_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, 2, padding='same', activation='relu', kernel_initializer="he_uniform"),
            tf.keras.layers.MaxPooling2D(pool_size=8),
            tf.keras.layers.Conv2D(8, 2, padding='same', activation='relu', kernel_initializer="he_uniform"),
            tf.keras.layers.MaxPooling2D(pool_size=8),
        ])
        base_model.trainable = False  # We disable the first layers
        x = base_model(input_layer, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        out = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')(x)
        model = tf.keras.models.Model(inputs=input_layer, outputs=[out])
        lr = self.keras_params['learning_rate'] if 'learning_rate' in self.keras_params.keys() else 0.001
        decay = self.keras_params['decay'] if 'decay' in self.keras_params.keys() else 0.0
        optimizer = tf.keras.optimizers.Adam(lr=lr, decay=decay)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        if self.logger.getEffectiveLevel() < logging.ERROR:
            model.summary()
        if self.level_save in ['MEDIUM', 'HIGH']:
            self._save_model_png(model)
        return model


class ModelTransferLearningClassifierTests(unittest.TestCase):
    '''Main class to test model_transfer_learning_classifier'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_transfer_learning_classifier_init(self):
        '''Test of {{package_name}}.models_training.classifiers.model_transfer_learning_classifier.ModelTransferLearningClassifier.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        model = ModelTransferLearningClassifier(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.model_type, 'classifier')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model, None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, batch_size=8)
        self.assertEqual(model.batch_size, 8)
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, epochs=42)
        self.assertEqual(model.epochs, 42)
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, validation_split=0.3)
        self.assertEqual(model.validation_split, 0.3)
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, patience=65)
        self.assertEqual(model.patience, 65)
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, width=22)
        self.assertEqual(model.width, 22)
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, height=56)
        self.assertEqual(model.height, 56)
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, depth=4)
        self.assertEqual(model.depth, 4)
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, color_mode='rgba')
        self.assertEqual(model.color_mode, 'rgba')
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, in_memory=True)
        self.assertEqual(model.in_memory, True)
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, with_fine_tune=False)
        self.assertEqual(model.with_fine_tune, False)
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, second_epochs=2)
        self.assertEqual(model.second_epochs, 2)
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, second_lr=0.1)
        self.assertEqual(model.second_lr, 0.1)
        remove_dir(model_dir)

        #
        model = ModelTransferLearningClassifier(model_dir=model_dir, second_patience=33)
        self.assertEqual(model.second_patience, 33)
        remove_dir(model_dir)

        # keras_params must accept anything !
        model = ModelTransferLearningClassifier(model_dir=model_dir, keras_params={'toto': 5})
        self.assertEqual(model.keras_params, {'toto': 5})
        remove_dir(model_dir)

    def test02_model_transfer_learning_classifier_fit(self):
        '''Test of the method fit of {{package_name}}.models_training.model_transfer_learning_classifier.ModelTransferLearningClassifier'''
        # First test with the real model, we mock the rest in order to speed up the process

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'test_data_1')
        filenames = ['Birman_1.jpg', 'Birman_2.jpg', 'Birman_4.jpg', 'Bombay_1.png', 'Bombay_10.jpg',
                     'Bombay_3.jpg', 'shiba_inu_15.jpg', 'shiba_inu_30.jpg', 'shiba_inu_31.jpg',
                     'shiba_inu_34.jpg', 'subfolder/Birman_36.jpg', 'subfolder/Bombay_19.jpg']
        df_train_mono = pd.DataFrame({
            'file_class': ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'cat'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        df_train_multi = pd.DataFrame({
            'file_class': ['birman', 'birman', 'birman', 'bombay', 'bombay', 'bombay', 'shiba', 'shiba', 'shiba', 'shiba', 'birman', 'bombay'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        df_valid_multi_missing = df_train_multi.copy()
        df_valid_multi_missing['file_class'] = df_valid_multi_missing['file_class'].apply(lambda x: 'birman' if x == 'shiba' else x)
        # For the "val" datasets, we reuse the train, not important here
        fit_arguments_keys = ['x', 'y', 'batch_size', 'steps_per_epoch', 'validation_data', 'validation_split', 'validation_steps']  # wanted keys in fit_arguments

        #
        lr = 0.123456

        ## Classification - Mono-label - Mono-Class - WITHOUT FINETUNING
        model = ModelTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With valid
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With shuffle to False
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With different image formats (we use the mock because efficient net must have a depth equal to 3)
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    width=64, height=64, depth=4, color_mode='rgba', with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # In memory
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    in_memory=True, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNotNone(fit_arguments['y'])
        self.assertIsNotNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])  # We have got a valid
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])  # We have got a valid
        self.assertIsNone(fit_arguments['steps_per_epoch'])
        self.assertIsNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # In memory - without valid
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                in_memory=True, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNotNone(fit_arguments['y'])
        self.assertIsNotNone(fit_arguments['batch_size'])
        self.assertIsNotNone(fit_arguments['validation_split'])  # we have no valid
        self.assertEqual(fit_arguments['validation_split'], model.validation_split)
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNone(fit_arguments['validation_data'])  # we have no valid
        self.assertIsNone(fit_arguments['steps_per_epoch'])
        self.assertIsNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With data augmentation
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                data_augmentation_params={'rotation_range':30}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # Test continue training
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # 2nd fit
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)  # We fit again with the same data, not important
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        # We do not save on purpose
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # 3rd fit
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)  # We fit again with the same data, not important
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        # Test data errors
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        df_train_mono_fake = pd.DataFrame({
            'file_class': ['toto', 'toto', 'tata', 'tata', 'tata', 'toto', 'toto', 'toto', 'tata', 'toto', 'toto', 'toto'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        with self.assertRaises(AssertionError):
            fit_arguments = model.fit(df_train_mono_fake, df_valid=df_train_mono, with_shuffle=True)
        remove_dir(model_dir)

        ## Classification - Mono-label - Mono-Class - WITH FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=True, second_epochs=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With valid
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=True, second_epochs=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With shuffle to False
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=True, second_epochs=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With different image formats (we use the mock because efficient net must have a depth equal to 3)
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    width=64, height=64, depth=4, color_mode='rgba', with_fine_tune=True, second_epochs=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # In memory
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    in_memory=True, with_fine_tune=True, second_epochs=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNotNone(fit_arguments['y'])
        self.assertIsNotNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])  # We have got a valid
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])  # We have got a valid
        self.assertIsNone(fit_arguments['steps_per_epoch'])
        self.assertIsNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # In memory - without valid
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    in_memory=True, with_fine_tune=True, second_epochs=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNotNone(fit_arguments['y'])
        self.assertIsNotNone(fit_arguments['batch_size'])
        self.assertIsNotNone(fit_arguments['validation_split'])  # we have no valid
        self.assertEqual(fit_arguments['validation_split'], model.validation_split)
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNone(fit_arguments['validation_data'])  # we have no valid
        self.assertIsNone(fit_arguments['steps_per_epoch'])
        self.assertIsNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With data augmentation
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    data_augmentation_params={'rotation_range':30}, with_fine_tune=True, second_epochs=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # Test continue training
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=True, second_epochs=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # 2nd fit
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)  # We fit again with the same data, not important
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        # We do not save on purpose
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # 3rd fit
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)  # We fit again with the same data, not important
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        # Test data errors
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=True, second_epochs=2)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        df_train_mono_fake = pd.DataFrame({
            'file_class': ['toto', 'toto', 'tata', 'tata', 'tata', 'toto', 'toto', 'toto', 'tata', 'toto', 'toto', 'toto'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        with self.assertRaises(AssertionError):
            fit_arguments = model.fit(df_train_mono_fake, df_valid=df_train_mono, with_shuffle=True)
        remove_dir(model_dir)

        ## Classification - Mono-label - Multi-Classes - WITHOUT FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With valid
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With shuffle to False
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With different image formats (we use the mock because efficient net must have a depth equal to 3)
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    width=64, height=64, depth=4, color_mode='rgba', with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # In memory
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    in_memory=True, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNotNone(fit_arguments['y'])
        self.assertIsNotNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])  # We have got a valid
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])  # We have got a valid
        self.assertIsNone(fit_arguments['steps_per_epoch'])
        self.assertIsNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # In memory - without valid
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    in_memory=True, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNotNone(fit_arguments['y'])
        self.assertIsNotNone(fit_arguments['batch_size'])
        self.assertIsNotNone(fit_arguments['validation_split'])  # we have no valid
        self.assertEqual(fit_arguments['validation_split'], model.validation_split)
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNone(fit_arguments['validation_data'])  # we have no valid
        self.assertIsNone(fit_arguments['steps_per_epoch'])
        self.assertIsNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With data augmentation
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    data_augmentation_params={'rotation_range':30}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # Test continue training
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # 2nd fit
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)  # We fit again with the same data, not important
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        # We do not save on purpose
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # 3rd fit
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)  # We fit again with the same data, not important
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        # Test data errors
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        df_train_multi_fake = pd.DataFrame({
            'file_class': ['toto', 'toto', 'tata', 'tata', 'tata', 'toto', 'toto', 'toto', 'tata', 'toto', 'toto', 'toto'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        with self.assertRaises(AssertionError):
            fit_arguments = model.fit(df_train_multi_fake, df_valid=df_train_multi, with_shuffle=True)
        remove_dir(model_dir)
        # Missing targets in df_valid
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_valid_multi_missing, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)

        ## Classification - Mono-label - Multi-Classes - WITH FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With valid
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With shuffle to False
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With different image formats (we use the mock because efficient net must have a depth equal to 3)
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    width=64, height=64, depth=4, color_mode='rgba', with_fine_tune=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # In memory
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    in_memory=True, with_fine_tune=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNotNone(fit_arguments['y'])
        self.assertIsNotNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])  # We have got a valid
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])  # We have got a valid
        self.assertIsNone(fit_arguments['steps_per_epoch'])
        self.assertIsNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # In memory - without valid
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    in_memory=True, with_fine_tune=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNotNone(fit_arguments['y'])
        self.assertIsNotNone(fit_arguments['batch_size'])
        self.assertIsNotNone(fit_arguments['validation_split'])  # we have no valid
        self.assertEqual(fit_arguments['validation_split'], model.validation_split)
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNone(fit_arguments['validation_data'])  # we have no valid
        self.assertIsNone(fit_arguments['steps_per_epoch'])
        self.assertIsNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # With data augmentation
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                                    data_augmentation_params={'rotation_range':30}, with_fine_tune=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)
        # Test continue training
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # 2nd fit
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)  # We fit again with the same data, not important
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        # We do not save on purpose
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # 3rd fit
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)  # We fit again with the same data, not important
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        # Test data errors
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        df_train_multi_fake = pd.DataFrame({
            'file_class': ['toto', 'toto', 'tata', 'tata', 'tata', 'toto', 'toto', 'toto', 'tata', 'toto', 'toto', 'toto'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        with self.assertRaises(AssertionError):
            fit_arguments = model.fit(df_train_multi_fake, df_valid=df_train_multi, with_shuffle=True)
        remove_dir(model_dir)
        # Missing targets in df_valid
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, with_fine_tune=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_valid_multi_missing, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_initial_fit.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'logger_initial_fit.csv')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'plots_initial_fit')))
        self.assertEqual(model.second_lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))
        self.assertEqual(type(fit_arguments), dict)
        self.assertTrue(all([_ in fit_arguments.keys() for _ in fit_arguments_keys]))
        self.assertTrue(all([_ in fit_arguments_keys for _ in fit_arguments.keys()]))
        self.assertIsNone(fit_arguments['y'])
        self.assertIsNone(fit_arguments['batch_size'])
        self.assertIsNone(fit_arguments['validation_split'])
        self.assertIsNotNone(fit_arguments['x'])
        self.assertIsNotNone(fit_arguments['validation_data'])
        self.assertIsNotNone(fit_arguments['steps_per_epoch'])
        self.assertIsNotNone(fit_arguments['validation_steps'])
        remove_dir(model_dir)

    def test03_model_transfer_learning_classifier_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.classifiers.model_transfer_learning_classifier.ModelTransferLearningClassifier'''
        # First test with the real model, we mock the rest in order to speed up the process

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'test_data_1')
        filenames = ['Birman_1.jpg', 'Birman_2.jpg', 'Birman_4.jpg', 'Bombay_1.png', 'Bombay_10.jpg',
                     'Bombay_3.jpg', 'shiba_inu_15.jpg', 'shiba_inu_30.jpg', 'shiba_inu_31.jpg',
                     'shiba_inu_34.jpg', 'subfolder/Birman_36.jpg', 'subfolder/Bombay_19.jpg']
        df_train_mono = pd.DataFrame({
            'file_class': ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'cat'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        df_train_multi = pd.DataFrame({
            'file_class': ['birman', 'birman', 'birman', 'bombay', 'bombay', 'bombay', 'shiba', 'shiba', 'shiba', 'shiba', 'birman', 'bombay'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })

        # Classification - Mono-label - Mono-Class - WITHOUT FINETUNING
        model = ModelTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=False)
        model.fit(df_train_mono, df_valid=df_train_mono)
        preds = model.predict(df_train_mono, return_proba=False)
        self.assertEqual(preds.shape, (df_train_mono.shape[0],))
        proba = model.predict(df_train_mono, return_proba=True)
        self.assertEqual(proba.shape, (df_train_mono.shape[0], 2)) # 2 classes
        remove_dir(model_dir)
        # in_memory
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, in_memory=True, with_fine_tune=False)
        model.fit(df_train_mono, df_valid=df_train_mono)
        preds = model.predict(df_train_mono, return_proba=False)
        self.assertEqual(preds.shape, (df_train_mono.shape[0],))
        proba = model.predict(df_train_mono, return_proba=True)
        self.assertEqual(proba.shape, (df_train_mono.shape[0], 2)) # 2 classes
        remove_dir(model_dir)

        # Classification - Mono-label - Mono-Class - WITH FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=True, second_epochs=2)
        model.fit(df_train_mono, df_valid=df_train_mono)
        preds = model.predict(df_train_mono, return_proba=False)
        self.assertEqual(preds.shape, (df_train_mono.shape[0],))
        proba = model.predict(df_train_mono, return_proba=True)
        self.assertEqual(proba.shape, (df_train_mono.shape[0], 2)) # 2 classes
        remove_dir(model_dir)
        # in_memory
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, in_memory=True, with_fine_tune=True, second_epochs=2)
        model.fit(df_train_mono, df_valid=df_train_mono)
        preds = model.predict(df_train_mono, return_proba=False)
        self.assertEqual(preds.shape, (df_train_mono.shape[0],))
        proba = model.predict(df_train_mono, return_proba=True)
        self.assertEqual(proba.shape, (df_train_mono.shape[0], 2)) # 2 classes
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes - WITHOUT FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=False)
        model.fit(df_train_multi, df_valid=df_train_multi)
        preds = model.predict(df_train_multi, return_proba=False)
        self.assertEqual(preds.shape, (df_train_multi.shape[0],))
        proba = model.predict(df_train_multi, return_proba=True)
        self.assertEqual(proba.shape, (df_train_multi.shape[0], 3)) # 3 classes
        remove_dir(model_dir)
        # in_memory
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, in_memory=True, with_fine_tune=False)
        model.fit(df_train_multi, df_valid=df_train_multi)
        preds = model.predict(df_train_multi, return_proba=False)
        self.assertEqual(preds.shape, (df_train_multi.shape[0],))
        proba = model.predict(df_train_multi, return_proba=True)
        self.assertEqual(proba.shape, (df_train_multi.shape[0], 3)) # 3 classes
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes - WITH FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=True, second_epochs=2)
        model.fit(df_train_multi, df_valid=df_train_multi)
        preds = model.predict(df_train_multi, return_proba=False)
        self.assertEqual(preds.shape, (df_train_multi.shape[0],))
        proba = model.predict(df_train_multi, return_proba=True)
        self.assertEqual(proba.shape, (df_train_multi.shape[0], 3)) # 3 classes
        remove_dir(model_dir)
        # in_memory
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, in_memory=True, with_fine_tune=True, second_epochs=2)
        model.fit(df_train_multi, df_valid=df_train_multi)
        preds = model.predict(df_train_multi, return_proba=False)
        self.assertEqual(preds.shape, (df_train_multi.shape[0],))
        proba = model.predict(df_train_multi, return_proba=True)
        self.assertEqual(proba.shape, (df_train_multi.shape[0], 3)) # 3 classes
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2)
            model.predict(df_train_multi)
        remove_dir(model_dir)

    def test04_model_transfer_learning_classifier_predict_proba(self):
        '''Test of the method predict_proba of {{package_name}}.models_training.classifiers.model_transfer_learning_classifier.ModelTransferLearningClassifier'''
        # First test with the real model, we mock the rest in order to speed up the process

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'test_data_1')
        filenames = ['Birman_1.jpg', 'Birman_2.jpg', 'Birman_4.jpg', 'Bombay_1.png', 'Bombay_10.jpg',
                     'Bombay_3.jpg', 'shiba_inu_15.jpg', 'shiba_inu_30.jpg', 'shiba_inu_31.jpg',
                     'shiba_inu_34.jpg', 'subfolder/Birman_36.jpg', 'subfolder/Bombay_19.jpg']
        df_train_mono = pd.DataFrame({
            'file_class': ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'cat'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        df_train_multi = pd.DataFrame({
            'file_class': ['birman', 'birman', 'birman', 'bombay', 'bombay', 'bombay', 'shiba', 'shiba', 'shiba', 'shiba', 'birman', 'bombay'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })

        # Classification - Mono-label - Mono-Class - WITHOUT FINETUNING
        model = ModelTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=False)
        model.fit(df_train_mono, df_valid=df_train_mono)
        preds = model.predict_proba(df_train_mono)
        self.assertEqual(preds.shape, (df_train_mono.shape[0], 2)) # 2 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Classification - Mono-label - Mono-Class - WITH FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=True, second_epochs=2)
        model.fit(df_train_mono, df_valid=df_train_mono)
        preds = model.predict_proba(df_train_mono)
        self.assertEqual(preds.shape, (df_train_mono.shape[0], 2)) # 2 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes - WITHOUT FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=False)
        model.fit(df_train_multi, df_valid=df_train_multi)
        preds = model.predict_proba(df_train_multi)
        self.assertEqual(preds.shape, (df_train_multi.shape[0], 3)) # 3 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes - WITHOUT FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=True, second_epochs=2)
        model.fit(df_train_multi, df_valid=df_train_multi)
        preds = model.predict_proba(df_train_multi)
        self.assertEqual(preds.shape, (df_train_multi.shape[0], 3)) # 3 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2)
            model.predict_proba(df_train_multi)
        remove_dir(model_dir)

    def test05_model_transfer_learning_classifier_get_predict_position(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_transfer_learning_classifier.ModelTransferLearningClassifier.get_predict_position'''
        # First test with the real model, we mock the rest in order to speed up the process

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'test_data_1')
        filenames = ['Birman_1.jpg', 'Birman_2.jpg', 'Birman_4.jpg', 'Bombay_1.png', 'Bombay_10.jpg',
                     'Bombay_3.jpg', 'shiba_inu_15.jpg', 'shiba_inu_30.jpg', 'shiba_inu_31.jpg',
                     'shiba_inu_34.jpg', 'subfolder/Birman_36.jpg', 'subfolder/Bombay_19.jpg']
        df_train_mono = pd.DataFrame({
            'file_class': ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'cat'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        y_true_mono = list(df_train_mono['file_class'].values)
        df_train_multi = pd.DataFrame({
            'file_class': ['birman', 'birman', 'birman', 'bombay', 'bombay', 'bombay', 'shiba', 'shiba', 'shiba', 'shiba', 'birman', 'bombay'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        y_true_multi = list(df_train_multi['file_class'].values)

        # Classification - Mono-label - Mono-Class - WITHOUT FINETUNING
        model = ModelTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=False)
        model.fit(df_train_mono)
        predict_positions = model.get_predict_position(df_train_mono, y_true_mono)
        self.assertEqual(predict_positions.shape, (df_train_mono.shape[0],))
        remove_dir(model_dir)

        # Classification - Mono-label - Mono-Class - WITH FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=True, second_epochs=2)
        model.fit(df_train_mono)
        predict_positions = model.get_predict_position(df_train_mono, y_true_mono)
        self.assertEqual(predict_positions.shape, (df_train_mono.shape[0],))
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes - WITHOUT FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=False)
        model.fit(df_train_multi)
        predict_positions = model.get_predict_position(df_train_multi, y_true_multi)
        self.assertEqual(predict_positions.shape, (df_train_multi.shape[0],))
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes - WITH FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=True, second_epochs=2)
        model.fit(df_train_multi)
        predict_positions = model.get_predict_position(df_train_multi, y_true_multi)
        self.assertEqual(predict_positions.shape, (df_train_multi.shape[0],))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2)
            model.get_predict_position(df_train_mono, y_true_mono)
        remove_dir(model_dir)

    def test06_model_transfer_learning_classifier_get_model(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_transfer_learning_classifier.ModelTransferLearningClassifier._get_model'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelTransferLearningClassifier(model_dir=model_dir, epochs=2, batch_size=2)

        # Nominal case
        model.list_classes = ['a', 'b']  # We force the creation of a list of classes
        model_res = model._get_model()
        self.assertTrue(isinstance(model_res, keras.Model))

        # Clean
        remove_dir(model_dir)

    @patch('keras.utils.data_utils.urlretrieve', side_effect=download_url_crash)
    @patch('{{package_name}}.utils.download_url', side_effect=download_url_crash)
    def test07_model_transfer_learning_classifier_get_model_offline(self, mock_download_url, mock_urlretrieve):
        '''Test of the method {{package_name}}.models_training.classifiers.model_transfer_learning_classifier.ModelTransferLearningClassifier._get_model
        - No access to a base model
        '''

        # Clean cache path if exists
        cache_path = os.path.join(utils.get_data_path(), 'cache_keras')
        if os.path.isdir(cache_path):
            shutil.rmtree(cache_path)

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelTransferLearningClassifier(model_dir=model_dir, epochs=2, batch_size=2)

        # Nominal case
        model.list_classes = ['a', 'b']  # We force the creation of a list of classes
        with self.assertRaises(Exception):
            model_res = model._get_model()

        # Clean
        remove_dir(model_dir)

    def test08_model_transfer_learning_classifier_get_preprocess_input(self):
        '''Test of the method _get_preprocess_input of {{package_name}}.models_training.classifiers.model_transfer_learning_classifier.ModelTransferLearningClassifier'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelTransferLearningClassifier(model_dir=model_dir)
        preprocess_input = model._get_preprocess_input()
        self.assertIsNotNone(preprocess_input)
        remove_dir(model_dir)

    def test09_model_transfer_learning_classifier_get_second_callbacks(self):
        '''Test of the method _get_second_callbacks of {{package_name}}.models_training.classifiers.model_transfer_learning_classifier.ModelTransferLearningClassifier'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelTransferLearningClassifier(model_dir=model_dir)

        # Nominal case
        callbacks = model._get_second_callbacks()
        callbacks_types = [type(_) for _ in callbacks]
        self.assertTrue(tensorflow.keras.callbacks.EarlyStopping in callbacks_types)
        self.assertTrue(tensorflow.keras.callbacks.ModelCheckpoint in callbacks_types)
        self.assertTrue(tensorflow.keras.callbacks.CSVLogger in callbacks_types)
        self.assertTrue(tensorflow.keras.callbacks.TerminateOnNaN in callbacks_types)
        checkpoint = callbacks[callbacks_types.index(tensorflow.keras.callbacks.ModelCheckpoint)]
        csv_logger = callbacks[callbacks_types.index(tensorflow.keras.callbacks.CSVLogger)]
        self.assertEqual(checkpoint.filepath, os.path.join(model.model_dir, 'best.hdf5'))
        self.assertEqual(csv_logger.filename, os.path.join(model.model_dir, 'logger.csv'))

        # level save 'LOW'
        model.level_save = 'LOW'
        callbacks = model._get_second_callbacks()
        callbacks_types = [type(_) for _ in callbacks]
        self.assertTrue(tensorflow.keras.callbacks.EarlyStopping in callbacks_types)
        self.assertFalse(tensorflow.keras.callbacks.ModelCheckpoint in callbacks_types)
        self.assertTrue(tensorflow.keras.callbacks.CSVLogger in callbacks_types)
        self.assertTrue(tensorflow.keras.callbacks.TerminateOnNaN in callbacks_types)
        csv_logger = callbacks[callbacks_types.index(tensorflow.keras.callbacks.CSVLogger)]
        self.assertEqual(csv_logger.filename, os.path.join(model.model_dir, 'logger.csv'))

        # Clean
        remove_dir(model_dir)

    def test10_model_transfer_learning_classifier_get_second_learning_rate_scheduler(self):
        '''Test of the method _get_second_learning_rate_scheduler of {{package_name}}.models_training.classifiers.model_transfer_learning_classifier.ModelTransferLearningClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelTransferLearningClassifier(model_dir=model_dir)
        self.assertEqual(model._get_second_learning_rate_scheduler(), None)

        # Clean
        remove_dir(model_dir)

    def test11_model_transfer_learning_classifier_save(self):
        '''Test of the method save of {{package_name}}.models_training.classifiers.model_transfer_learning_classifier.ModelTransferLearningClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelTransferLearningClassifier(model_dir=model_dir)
        model.model_type = 'classifier' # We test classifier
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"preprocess_input.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertTrue('mainteners' in configs.keys())
        self.assertTrue('date' in configs.keys())
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('model_type' in configs.keys())
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'keras')
        self.assertTrue('batch_size' in configs.keys())
        self.assertTrue('epochs' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('patience' in configs.keys())
        self.assertTrue('width' in configs.keys())
        self.assertTrue('height' in configs.keys())
        self.assertTrue('depth' in configs.keys())
        self.assertTrue('color_mode' in configs.keys())
        self.assertTrue('in_memory' in configs.keys())
        self.assertTrue('data_augmentation_params' in configs.keys())
        self.assertTrue('nb_train_generator_images_to_save' in configs.keys())
        self.assertTrue('keras_params' in configs.keys())
        self.assertTrue('_get_model' in configs.keys())
        self.assertTrue('_get_learning_rate_scheduler' in configs.keys())
        self.assertTrue('custom_objects' in configs.keys())
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('with_fine_tune' in configs.keys())
        self.assertTrue('second_epochs' in configs.keys())
        self.assertTrue('second_lr' in configs.keys())
        self.assertTrue('second_patience' in configs.keys())
        self.assertTrue('_get_second_learning_rate_scheduler' in configs.keys())
        remove_dir(model_dir)

        # Use custom_objects containing a "partial" function
        model = ModelTransferLearningClassifier(model_dir=model_dir)
        model.model_type = 'classifier' # We do not test 'regressor', it is the same thing
        custom_objects = utils_deep_keras.custom_objects
        custom_objects['fb_loss'] = utils_deep_keras.get_fb_loss(0.5)
        model.custom_objects = custom_objects
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"preprocess_input.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertTrue('mainteners' in configs.keys())
        self.assertTrue('date' in configs.keys())
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('model_type' in configs.keys())
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'keras')
        self.assertTrue('batch_size' in configs.keys())
        self.assertTrue('epochs' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('patience' in configs.keys())
        self.assertTrue('width' in configs.keys())
        self.assertTrue('height' in configs.keys())
        self.assertTrue('depth' in configs.keys())
        self.assertTrue('color_mode' in configs.keys())
        self.assertTrue('in_memory' in configs.keys())
        self.assertTrue('data_augmentation_params' in configs.keys())
        self.assertTrue('nb_train_generator_images_to_save' in configs.keys())
        self.assertTrue('keras_params' in configs.keys())
        self.assertTrue('_get_model' in configs.keys())
        self.assertTrue('_get_learning_rate_scheduler' in configs.keys())
        self.assertTrue('custom_objects' in configs.keys())
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('with_fine_tune' in configs.keys())
        self.assertTrue('second_epochs' in configs.keys())
        self.assertTrue('second_lr' in configs.keys())
        self.assertTrue('second_patience' in configs.keys())
        self.assertTrue('_get_second_learning_rate_scheduler' in configs.keys())
        remove_dir(model_dir)

    def test12_model_transfer_learning_classifier_reload_model(self):
        '''Test of the method reload_model of {{package_name}}.models_training.classifiers.model_transfer_learning_classifier.ModelTransferLearningClassifier'''
        # First test with the real model, we mock the rest in order to speed up the process

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'test_data_1')
        filenames = ['Birman_1.jpg', 'Birman_2.jpg', 'Birman_4.jpg', 'Bombay_1.png', 'Bombay_10.jpg',
                     'Bombay_3.jpg', 'shiba_inu_15.jpg', 'shiba_inu_30.jpg', 'shiba_inu_31.jpg',
                     'shiba_inu_34.jpg', 'subfolder/Birman_36.jpg', 'subfolder/Bombay_19.jpg']
        df_train_mono = pd.DataFrame({
            'file_class': ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'cat'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        df_train_multi = pd.DataFrame({
            'file_class': ['birman', 'birman', 'birman', 'bombay', 'bombay', 'bombay', 'shiba', 'shiba', 'shiba', 'shiba', 'birman', 'bombay'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })

        # Classification - Mono-label - Mono-Class - WITHOUT FINETUNING
        model = ModelTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=False)
        fit_params = model.fit(df_train_mono, df_valid=None)
        x_batch = fit_params['x'].next()
        model.save()
        # Reload keras
        hdf5_path = os.path.join(model.model_dir, 'best.hdf5')
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_batch[0])], [list(_) for _ in model.model.predict(x_batch[0])], 3)
        # Test without custom_objects
        model.custom_objects = None
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_batch[0])], [list(_) for _ in model.model.predict(x_batch[0])], 3)
        # Clean
        remove_dir(model_dir)

        # Classification - Mono-label - Mono-Class - WITH FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=True, second_epochs=2)
        fit_params = model.fit(df_train_mono, df_valid=None)
        x_batch = fit_params['x'].next()
        model.save()
        # Reload keras
        hdf5_path = os.path.join(model.model_dir, 'best.hdf5')
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_batch[0])], [list(_) for _ in model.model.predict(x_batch[0])], 3)
        # Test without custom_objects
        model.custom_objects = None
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_batch[0])], [list(_) for _ in model.model.predict(x_batch[0])], 3)
        # Clean
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes - WITHOUT FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=False)
        fit_params = model.fit(df_train_multi, df_valid=None)
        x_batch = fit_params['x'].next()
        model.save()
        # Reload keras
        hdf5_path = os.path.join(model.model_dir, 'best.hdf5')
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_batch[0])], [list(_) for _ in model.model.predict(x_batch[0])], 3)
        # Test without custom_objects
        model.custom_objects = None
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_batch[0])], [list(_) for _ in model.model.predict(x_batch[0])], 3)
        # Clean
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes - WITH FINETUNING
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=True, second_epochs=2)
        fit_params = model.fit(df_train_multi, df_valid=None)
        x_batch = fit_params['x'].next()
        model.save()
        # Reload keras
        hdf5_path = os.path.join(model.model_dir, 'best.hdf5')
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_batch[0])], [list(_) for _ in model.model.predict(x_batch[0])], 3)
        # Test without custom_objects
        model.custom_objects = None
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_batch[0])], [list(_) for _ in model.model.predict(x_batch[0])], 3)
        # Clean
        remove_dir(model_dir)

    def test13_model_transfer_learning_classifier_reload_from_standalone(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_transfer_learning_classifier.ModelTransferLearningClassifier.reload_from_standalone'''
        # First test with the real model, we mock the rest in order to speed up the process

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'test_data_1')
        filenames = ['Birman_1.jpg', 'Birman_2.jpg', 'Birman_4.jpg', 'Bombay_1.png', 'Bombay_10.jpg',
                     'Bombay_3.jpg', 'shiba_inu_15.jpg', 'shiba_inu_30.jpg', 'shiba_inu_31.jpg',
                     'shiba_inu_34.jpg', 'subfolder/Birman_36.jpg', 'subfolder/Bombay_19.jpg']
        df_train_mono = pd.DataFrame({
            'file_class': ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'cat'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        df_train_multi = pd.DataFrame({
            'file_class': ['birman', 'birman', 'birman', 'bombay', 'bombay', 'bombay', 'shiba', 'shiba', 'shiba', 'shiba', 'birman', 'bombay'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })

        ############################################
        # Classification - Mono class - WITHOUT FINETUNING
        ############################################

        # Create model
        model = ModelTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=False)
        fit_params = model.fit(df_train_mono)
        model.save()
        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        preprocess_input_path = os.path.join(model.model_dir, "preprocess_input.pkl")
        new_model = ModelTransferLearningClassifier()
        new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
                                         preprocess_input_path=preprocess_input_path)
        # Tests
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.model_type, new_model.model_type)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.batch_size, new_model.batch_size)
        self.assertEqual(model.epochs, new_model.epochs)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.patience, new_model.patience)
        self.assertEqual(model.width, new_model.width)
        self.assertEqual(model.height, new_model.height)
        self.assertEqual(model.depth, new_model.depth)
        self.assertEqual(model.color_mode, new_model.color_mode)
        self.assertEqual(model.in_memory, new_model.in_memory)
        self.assertEqual(model.data_augmentation_params, new_model.data_augmentation_params)
        self.assertEqual(model.nb_train_generator_images_to_save, new_model.nb_train_generator_images_to_save)
        self.assertEqual(model.keras_params, new_model.keras_params)
        self.assertEqual(model.custom_objects, new_model.custom_objects)
        self.assertEqual(model.with_fine_tune, new_model.with_fine_tune)
        self.assertEqual(model.second_epochs, new_model.second_epochs)
        self.assertEqual(model.second_lr, new_model.second_lr)
        self.assertEqual(model.second_patience, new_model.second_patience)
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(df_train_mono)], [list(_) for _ in new_model.predict_proba(df_train_mono)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # Classification - Mono class - WITH FINETUNING
        ############################################

        # Create model
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=True, second_epochs=2)
        fit_params = model.fit(df_train_mono)
        model.save()
        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        preprocess_input_path = os.path.join(model.model_dir, "preprocess_input.pkl")
        new_model = ModelMockTransferLearningClassifier()
        new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
                                         preprocess_input_path=preprocess_input_path)
        # Tests
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.model_type, new_model.model_type)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.batch_size, new_model.batch_size)
        self.assertEqual(model.epochs, new_model.epochs)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.patience, new_model.patience)
        self.assertEqual(model.width, new_model.width)
        self.assertEqual(model.height, new_model.height)
        self.assertEqual(model.depth, new_model.depth)
        self.assertEqual(model.color_mode, new_model.color_mode)
        self.assertEqual(model.in_memory, new_model.in_memory)
        self.assertEqual(model.data_augmentation_params, new_model.data_augmentation_params)
        self.assertEqual(model.nb_train_generator_images_to_save, new_model.nb_train_generator_images_to_save)
        self.assertEqual(model.keras_params, new_model.keras_params)
        self.assertEqual(model.custom_objects, new_model.custom_objects)
        self.assertEqual(model.custom_objects, new_model.custom_objects)
        self.assertEqual(model.with_fine_tune, new_model.with_fine_tune)
        self.assertEqual(model.second_epochs, new_model.second_epochs)
        self.assertEqual(model.second_lr, new_model.second_lr)
        self.assertEqual(model.second_patience, new_model.second_patience)
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(df_train_mono)], [list(_) for _ in new_model.predict_proba(df_train_mono)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # Classification - Multi class - WITHOUT FINETUNING
        ############################################

        # Create model
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=False)
        fit_params = model.fit(df_train_multi)
        model.save()
        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        preprocess_input_path = os.path.join(model.model_dir, "preprocess_input.pkl")
        new_model = ModelMockTransferLearningClassifier()
        new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
                                         preprocess_input_path=preprocess_input_path)
        # Tests
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.model_type, new_model.model_type)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.batch_size, new_model.batch_size)
        self.assertEqual(model.epochs, new_model.epochs)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.patience, new_model.patience)
        self.assertEqual(model.width, new_model.width)
        self.assertEqual(model.height, new_model.height)
        self.assertEqual(model.depth, new_model.depth)
        self.assertEqual(model.color_mode, new_model.color_mode)
        self.assertEqual(model.in_memory, new_model.in_memory)
        self.assertEqual(model.data_augmentation_params, new_model.data_augmentation_params)
        self.assertEqual(model.nb_train_generator_images_to_save, new_model.nb_train_generator_images_to_save)
        self.assertEqual(model.keras_params, new_model.keras_params)
        self.assertEqual(model.custom_objects, new_model.custom_objects)
        self.assertEqual(model.custom_objects, new_model.custom_objects)
        self.assertEqual(model.with_fine_tune, new_model.with_fine_tune)
        self.assertEqual(model.second_epochs, new_model.second_epochs)
        self.assertEqual(model.second_lr, new_model.second_lr)
        self.assertEqual(model.second_patience, new_model.second_patience)
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(df_train_multi)], [list(_) for _ in new_model.predict_proba(df_train_multi)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # Classification - Multi class - WITH FINETUNING
        ############################################

        # Create model
        model = ModelMockTransferLearningClassifier(model_dir=model_dir, batch_size=2, epochs=2, with_fine_tune=True, second_epochs=2)
        fit_params = model.fit(df_train_multi)
        model.save()
        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        preprocess_input_path = os.path.join(model.model_dir, "preprocess_input.pkl")
        new_model = ModelMockTransferLearningClassifier()
        new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
                                         preprocess_input_path=preprocess_input_path)
        # Tests
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.model_type, new_model.model_type)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.batch_size, new_model.batch_size)
        self.assertEqual(model.epochs, new_model.epochs)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.patience, new_model.patience)
        self.assertEqual(model.width, new_model.width)
        self.assertEqual(model.height, new_model.height)
        self.assertEqual(model.depth, new_model.depth)
        self.assertEqual(model.color_mode, new_model.color_mode)
        self.assertEqual(model.in_memory, new_model.in_memory)
        self.assertEqual(model.data_augmentation_params, new_model.data_augmentation_params)
        self.assertEqual(model.nb_train_generator_images_to_save, new_model.nb_train_generator_images_to_save)
        self.assertEqual(model.keras_params, new_model.keras_params)
        self.assertEqual(model.custom_objects, new_model.custom_objects)
        self.assertEqual(model.custom_objects, new_model.custom_objects)
        self.assertEqual(model.with_fine_tune, new_model.with_fine_tune)
        self.assertEqual(model.second_epochs, new_model.second_epochs)
        self.assertEqual(model.second_lr, new_model.second_lr)
        self.assertEqual(model.second_patience, new_model.second_patience)
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(df_train_multi)], [list(_) for _ in new_model.predict_proba(df_train_multi)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)


        ############################################
        # Errors
        ############################################

        with self.assertRaises(FileNotFoundError):
            new_model = ModelTransferLearningClassifier()
            new_model.reload_from_standalone(configuration_path='toto.json', hdf5_path=hdf5_path,
                                             preprocess_input_path=preprocess_input_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelTransferLearningClassifier()
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path='toto.pkl',
                                             preprocess_input_path=preprocess_input_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelTransferLearningClassifier()
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
                                             preprocess_input_path='toto.pkl')

        # Clean
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
