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
import time
import shutil
import numpy as np
import pandas as pd
import dill as pickle

import tensorflow

from {{package_name}} import utils
from {{package_name}}.models_training import utils_deep_keras
from {{package_name}}.models_training.model_keras import ModelKeras
from {{package_name}}.models_training.classifiers.model_cnn_classifier import ModelCnnClassifier

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelKerasTests(unittest.TestCase):
    '''Main class to test model_keras'''

    def setUp(self):
        '''Setup fonction -> we create a mock embedding'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_keras_init(self):
        '''Test of the initialization of {{package_name}}.models_training.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        model = ModelKeras(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.model_type, None)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model, None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, batch_size=8)
        self.assertEqual(model.batch_size, 8)
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, epochs=42)
        self.assertEqual(model.epochs, 42)
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, validation_split=0.3)
        self.assertEqual(model.validation_split, 0.3)
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, patience=65)
        self.assertEqual(model.patience, 65)
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, width=22)
        self.assertEqual(model.width, 22)
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, height=56)
        self.assertEqual(model.height, 56)
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, depth=4)
        self.assertEqual(model.depth, 4)
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, color_mode='rgba')
        self.assertEqual(model.color_mode, 'rgba')
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, in_memory=True)
        self.assertEqual(model.in_memory, True)
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, data_augmentation_params={'toto': 'titi'})
        self.assertEqual(model.data_augmentation_params, {'toto': 'titi'})
        remove_dir(model_dir)

        #
        model = ModelKeras(model_dir=model_dir, nb_train_generator_images_to_save=10)
        self.assertEqual(model.nb_train_generator_images_to_save, 10)
        remove_dir(model_dir)

        # keras_params must accept anything !
        model = ModelKeras(model_dir=model_dir, keras_params={'toto': 5})
        self.assertEqual(model.keras_params, {'toto': 5})
        remove_dir(model_dir)

        # Check errors
        with self.assertRaises(ValueError):
            ModelKeras(model_dir=model_dir, data_augmentation_params={'toto': 'titi'}, in_memory=True)
        remove_dir(model_dir)

    def test02_model_keras_fit(self):
        '''Test of the method fit of {{package_name}}.models_training.model_keras.ModelKeras'''
        # /!\ We test with model_cnn_classifier /!\

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

        ## Classification - Mono-label - Mono-Class
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        # With different image format
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                   width=64, height=64, depth=4, color_mode='rgba')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, in_memory=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, in_memory=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, data_augmentation_params={'rotation_range':30})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['cat', 'dog'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # 2nd fit
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)  # We fit again with the same data, not important
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        # Test data errors
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        df_train_mono_fake = pd.DataFrame({
            'file_class': ['toto', 'toto', 'tata', 'tata', 'tata', 'toto', 'toto', 'toto', 'tata', 'toto', 'toto', 'toto'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        with self.assertRaises(AssertionError):
            fit_arguments = model.fit(df_train_mono_fake, df_valid=df_train_mono, with_shuffle=True)
        remove_dir(model_dir)

        ## Classification - Mono-label - Multi-Classes
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        # With different image format
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0},
                                   width=64, height=64, depth=4, color_mode='rgba')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, in_memory=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, in_memory=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0}, data_augmentation_params={'rotation_range':30})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # 2nd fit
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)  # We fit again with the same data, not important
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        # Test data errors
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        df_train_multi_fake = pd.DataFrame({
            'file_class': ['toto', 'toto', 'tata', 'tata', 'tata', 'toto', 'toto', 'toto', 'tata', 'toto', 'toto', 'toto'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        with self.assertRaises(AssertionError):
            fit_arguments = model.fit(df_train_multi_fake, df_valid=df_train_multi, with_shuffle=True)
        remove_dir(model_dir)
        # Missing targets in df_valid
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        fit_arguments = model.fit(df_train_multi, df_valid=df_valid_multi_missing, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['birman', 'bombay', 'shiba'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
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

    def test03_model_keras_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.model_keras.ModelKeras'''

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

        # Classification - Mono-label - Mono-Class
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
        model.fit(df_train_mono, df_valid=df_train_mono)
        preds = model.predict(df_train_mono, return_proba=False)
        self.assertEqual(preds.shape, (df_train_mono.shape[0],))
        proba = model.predict(df_train_mono, return_proba=True)
        self.assertEqual(proba.shape, (df_train_mono.shape[0], 2)) # 2 classes
        remove_dir(model_dir)
        # in_memory
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, in_memory=True)
        model.fit(df_train_mono, df_valid=df_train_mono)
        preds = model.predict(df_train_mono, return_proba=False)
        self.assertEqual(preds.shape, (df_train_mono.shape[0],))
        proba = model.predict(df_train_mono, return_proba=True)
        self.assertEqual(proba.shape, (df_train_mono.shape[0], 2)) # 2 classes
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
        model.fit(df_train_multi, df_valid=df_train_multi)
        preds = model.predict(df_train_multi, return_proba=False)
        self.assertEqual(preds.shape, (df_train_multi.shape[0],))
        proba = model.predict(df_train_multi, return_proba=True)
        self.assertEqual(proba.shape, (df_train_multi.shape[0], 3)) # 3 classes
        remove_dir(model_dir)
        # in_memory
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2, in_memory=True)
        model.fit(df_train_multi, df_valid=df_train_multi)
        preds = model.predict(df_train_multi, return_proba=False)
        self.assertEqual(preds.shape, (df_train_multi.shape[0],))
        proba = model.predict(df_train_multi, return_proba=True)
        self.assertEqual(proba.shape, (df_train_multi.shape[0], 3)) # 3 classes
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
            model.predict(df_train_multi)
        remove_dir(model_dir)

    def test04_model_keras_predict_proba(self):
        '''Test of the method predict_proba of {{package_name}}.models_training.model_keras.ModelKeras'''

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

        # Classification - Mono-label - Mono-Class
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
        model.fit(df_train_mono, df_valid=df_train_mono)
        preds = model.predict_proba(df_train_mono)
        self.assertEqual(preds.shape, (df_train_mono.shape[0], 2)) # 2 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
        model.fit(df_train_multi, df_valid=df_train_multi)
        preds = model.predict_proba(df_train_multi)
        self.assertEqual(preds.shape, (df_train_multi.shape[0], 3)) # 3 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
            model.predict_proba(df_train_multi)
        remove_dir(model_dir)

    def test05_model_keras_get_generator(self):
        '''Test of the method predict_proba of {{package_name}}.models_training.model_keras.ModelKeras'''

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

        # Classification - Mono-label - Mono-Class
        # train
        model = ModelKeras(model_dir=model_dir, batch_size=2, epochs=2, data_augmentation_params={'vertical_flip': True})
        model.model_type = 'classifier'  # We set classifier for the tests
        model.list_classes = ['cat', 'dog']
        generator = model._get_generator(df_train_mono, data_type='train', batch_size=4)
        self.assertTrue(hasattr(generator, 'next'))
        self.assertEqual(generator.batch_size, 4)
        self.assertEqual(len(generator.filepaths), df_train_mono.shape[0])
        self.assertEqual(generator.target_size, (model.width, model.height))
        self.assertTrue(generator.shuffle)
        self.assertEqual(len(generator.class_indices), 2)  # 2 classes
        self.assertEqual(sorted(list(generator.class_indices.keys())), sorted(model.list_classes))
        self.assertTrue(generator.image_data_generator.vertical_flip)
        batch_1 = generator.next()
        self.assertEqual(len(batch_1), 2)  # [images, classes]
        self.assertEqual(len(batch_1[0]), 4)
        self.assertEqual(len(batch_1[1]), 4)
        remove_dir(model_dir)
        # valid & different sizes
        model = ModelKeras(model_dir=model_dir, batch_size=2, epochs=2, data_augmentation_params={'vertical_flip': True},
                                   width=60, height=120)
        model.model_type = 'classifier'  # We set classifier for the tests
        model.list_classes = ['cat', 'dog']
        generator = model._get_generator(df_train_mono, data_type='valid', batch_size=4)
        self.assertTrue(hasattr(generator, 'next'))
        self.assertEqual(generator.batch_size, 4)
        self.assertEqual(len(generator.filepaths), df_train_mono.shape[0])
        self.assertEqual(generator.target_size, (model.width, model.height))
        self.assertFalse(generator.shuffle)
        self.assertEqual(len(generator.class_indices), 2)  # 2 classes
        self.assertEqual(sorted(list(generator.class_indices.keys())), sorted(model.list_classes))
        self.assertFalse(generator.image_data_generator.vertical_flip)
        batch_1 = generator.next()
        self.assertEqual(len(batch_1), 2)  # [images, classes]
        self.assertEqual(len(batch_1[0]), 4)
        self.assertEqual(len(batch_1[1]), 4)
        remove_dir(model_dir)
        # test
        model = ModelKeras(model_dir=model_dir, batch_size=2, epochs=2, data_augmentation_params={'vertical_flip': True})
        model.model_type = 'classifier'  # We set classifier for the tests
        model.list_classes = ['cat', 'dog']
        generator = model._get_generator(df_train_mono, data_type='test', batch_size=4)
        self.assertTrue(hasattr(generator, 'next'))
        self.assertEqual(generator.batch_size, 4)
        self.assertEqual(len(generator.filepaths), df_train_mono.shape[0])
        self.assertEqual(generator.target_size, (model.width, model.height))
        self.assertFalse(generator.shuffle)
        self.assertEqual(len(generator.class_indices), 1)  # 1 fake class
        self.assertEqual(sorted(list(generator.class_indices.keys())), ['all_classes'])
        self.assertFalse(generator.image_data_generator.vertical_flip)
        batch_1 = generator.next()
        self.assertEqual(len(batch_1), 2)  # [images, classes]
        self.assertEqual(len(batch_1[0]), 4)
        self.assertEqual(len(batch_1[1]), 4)
        remove_dir(model_dir)
        # error type
        model = ModelKeras(model_dir=model_dir, batch_size=2, epochs=2, data_augmentation_params={'vertical_flip': True})
        model.model_type = 'classifier'  # We set classifier for the tests
        model.list_classes = ['cat', 'dog']
        with self.assertRaises(ValueError):
            generator = model._get_generator(df_train_mono, data_type='toto', batch_size=4)
        remove_dir(model_dir)
        # error list_classes not set
        model = ModelKeras(model_dir=model_dir, batch_size=2, epochs=2, data_augmentation_params={'vertical_flip': True})
        model.model_type = 'classifier'  # We set classifier for the tests
        with self.assertRaises(AttributeError):
            generator = model._get_generator(df_train_mono, data_type='train', batch_size=4)
        remove_dir(model_dir)

    def test06_model_keras_get_preprocess_input(self):
        '''Test of the method _get_preprocess_input of {{package_name}}.models_training.model_keras.ModelKeras'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelKeras(model_dir=model_dir)
        preprocess_input = model._get_preprocess_input()
        self.assertIsNone(preprocess_input)
        remove_dir(model_dir)

    def test07_model_keras_get_callbacks(self):
        '''Test of the method _get_callbacks of {{package_name}}.models_training.model_keras.ModelKeras'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelKeras(model_dir=model_dir)

        # Nominal case
        callbacks = model._get_callbacks()
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
        callbacks = model._get_callbacks()
        callbacks_types = [type(_) for _ in callbacks]
        self.assertTrue(tensorflow.keras.callbacks.EarlyStopping in callbacks_types)
        self.assertFalse(tensorflow.keras.callbacks.ModelCheckpoint in callbacks_types)
        self.assertTrue(tensorflow.keras.callbacks.CSVLogger in callbacks_types)
        self.assertTrue(tensorflow.keras.callbacks.TerminateOnNaN in callbacks_types)
        csv_logger = callbacks[callbacks_types.index(tensorflow.keras.callbacks.CSVLogger)]
        self.assertEqual(csv_logger.filename, os.path.join(model.model_dir, 'logger.csv'))

        # Clean
        remove_dir(model_dir)

    def test08_model_keras_get_learning_rate_scheduler(self):
        '''Test of the method _get_learning_rate_scheduler of {{package_name}}.models_training.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelKeras(model_dir=model_dir)
        self.assertEqual(model._get_learning_rate_scheduler(), None)

        # Clean
        remove_dir(model_dir)

    def test09_model_keras_save(self):
        '''Test of the method save of {{package_name}}.models_training.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelKeras(model_dir=model_dir)
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
        self.assertTrue('list_classes' not in configs.keys()) # not in because we do not use the Classifier mixin
        self.assertTrue('dict_classes' not in configs.keys()) # not in because we do not use the Classifier mixin
        remove_dir(model_dir)

        # Use custom_objects containing a "partial" function
        model = ModelKeras(model_dir=model_dir)
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
        self.assertTrue('list_classes' not in configs.keys()) # not in because we do not use the Classifier mixin
        self.assertTrue('dict_classes' not in configs.keys()) # not in because we do not use the Classifier mixin
        remove_dir(model_dir)

    def test10_model_keras_reload_model(self):
        '''Test of the method reload_model of {{package_name}}.models_training.model_keras.ModelKeras'''

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

        # Classification - Mono-label - Mono-Class
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
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

        # Classification - Mono-label - Multi-Classes
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
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


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
