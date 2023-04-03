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
import dill as pickle
import numpy as np
import pandas as pd

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from {{package_name}} import utils
from {{package_name}}.models_training.models_tensorflow import utils_deep_keras
from {{package_name}}.models_training.models_tensorflow.model_keras import ModelKeras
from {{package_name}}.models_training.models_tensorflow.model_embedding_cnn import ModelEmbeddingCnn
from {{package_name}}.models_training.models_tensorflow.model_embedding_lstm import ModelEmbeddingLstm

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
        # Check if data folder exists
        data_path = utils.get_data_path()
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        # Create a mock embedding
        fake_embedding = {'toto': [0.25, 0.90, 0.12], 'titi': [0.85, 0.12, 0.8], 'test': [0.5, 0.6, 0.1],
                          'pas': [0.82, 0.90, 0.13], 'ici': [0.65, 0.01, 0.01], 'rien': [0.25, 0.02, 0.98],
                          'cela': [0.5, 0.5, 0.5]}
        fake_path = os.path.join(data_path, 'fake_embedding.pkl')
        if os.path.exists(fake_path):
            os.remove(fake_path)
        with open(fake_path, 'wb') as f:
            pickle.dump(fake_embedding, f, pickle.HIGHEST_PROTOCOL)

    def tearDown(self):
        '''Cleaning fonction -> we delete the mock embedding'''
        data_path = utils.get_data_path()
        fake_path = os.path.join(data_path, 'fake_embedding.pkl')
        if os.path.exists(fake_path):
            os.remove(fake_path)

    def check_weights_equality(self, model_1, model_2):
        self.assertEqual(len(model_1.model.weights), len(model_2.model.weights))
        for layer_nb in range(len(model_1.model.weights)):
            self.assertEqual(model_1.model.weights[layer_nb].numpy().shape, model_2.model.weights[layer_nb].numpy().shape)
        for layer_nb, x1, x2, x3 in [(1, 0, 0, 0), (7, 1, 2, 3)]:
            self.assertAlmostEqual(model_1.model.weights[layer_nb].numpy()[x1, x2, x3], model_2.model.weights[layer_nb].numpy()[x1, x2, x3])

    def test01_model_keras_init(self):
        '''Test of the initialization of {{package_name}}.models_training.models_tensorflow.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        model = ModelKeras(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
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
        model = ModelKeras(model_dir=model_dir, embedding_name='toto')
        self.assertEqual(model.embedding_name, 'toto')
        remove_dir(model_dir)

        # keras_params must accept anything !
        model = ModelKeras(model_dir=model_dir, keras_params={'toto': 5})
        self.assertEqual(model.keras_params, {'toto': 5})
        remove_dir(model_dir)

    def test02_model_keras_fit(self):
        '''Test of the method fit of {{package_name}}.models_training.models_tensorflow.model_keras.ModelKeras'''
        # /!\ We test with model_embedding_lstm /!\

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"] * 100)
        x_valid = np.array(["cela est un test", "ni cela", "non plus", "ici test", "là, rien de rien!"] * 100)
        y_train_mono = np.array([0, 1, 0, 1, 2] * 100)
        y_valid_mono = y_train_mono.copy()
        y_valid_mono_missing = y_train_mono.copy()
        y_valid_mono_missing[y_valid_mono_missing == 2] = 0
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0] * 100, 'test2': [1, 0, 0, 0, 0] * 100, 'test3': [0, 0, 0, 1, 0] * 100})
        y_valid_multi = y_train_multi.copy()
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        # We must set a decay to 0 to validate the lr value
        lr = 0.123456
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl', keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))  # We must round (almost_equal is fine too)
        remove_dir(model_dir)

        #
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=x_valid, y_valid=y_valid_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        #
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=x_valid, y_valid=y_valid_mono, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        # Validation with y_train & y_valid of shape 2
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, np.expand_dims(y_train_mono, 1), x_valid=x_valid, y_valid=np.expand_dims(y_valid_mono, 1), with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        # Missing targets in y_valid
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=x_valid, y_valid=y_valid_mono_missing, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        # Multi-labels
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['test1', 'test2', 'test3'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        # We must set a decay to 0 to validate the lr value
        lr = 0.123456
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl', keras_params={'learning_rate': lr, 'decay': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=x_valid, y_valid=y_valid_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['test1', 'test2', 'test3'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertEqual(lr, round(float(model.model.optimizer._decayed_lr(tensorflow.float32).numpy()), 6))  # We must round (almost_equal is fine too)
        remove_dir(model_dir)

        #
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=x_valid, y_valid=y_valid_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['test1', 'test2', 'test3'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        #
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=x_valid, y_valid=y_valid_multi, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['test1', 'test2', 'test3'])
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)


        ###########
        # Test continue training

        # Test mono-label nominal case
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # Second fit
        y_train_different_order = np.array([1, 0, 0, 1, 2] * 100)
        model.fit(x_train[:50], y_train_different_order[:50], x_valid=None, y_valid=None, with_shuffle=True)
        # We do not save on purpose
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # third fit
        model.fit(x_train[50:], y_train_mono[50:], x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        # Fourth fit
        model.fit(x_train[50:], y_train_mono[50:], x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 4)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_3.json')))
        model_dir_4 = model.model_dir
        self.assertNotEqual(model_dir_3, model_dir_4)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        remove_dir(model_dir_4)

        # Test data errors mono-label
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # Second fit
        y_train_mono_fake = np.array([3, 1, 0, 1, 2] * 100)
        with self.assertRaises(ValueError):
            model.fit(x_train[:50], y_train_mono_fake[:50], x_valid=None, y_valid=None, with_shuffle=True)
        remove_dir(model_dir)

        # Test multi-labels nominal case
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # Second fit
        model.fit(x_train[:50], y_train_multi[:50], x_valid=None, y_valid=None, with_shuffle=True)
        # We do not save on purpose
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # third fit
        model.fit(x_train[50:], y_train_multi[50:], x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        # Fourth fit
        model.fit(x_train[50:], y_train_multi[50:], x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 4)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_3.json')))
        model_dir_4 = model.model_dir
        self.assertNotEqual(model_dir_3, model_dir_4)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        remove_dir(model_dir_4)

        # Test data errors multi-labels
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # Second fit
        y_train_multi_fake = pd.DataFrame({'test3': [0, 0, 0, 1, 0] * 100, 'test2': [1, 0, 0, 0, 0] * 100, 'test1': [0, 0, 0, 1, 0] * 100})
        with self.assertRaises(ValueError):
            model.fit(x_train[:50], y_train_multi_fake[:50], x_valid=None, y_valid=None, with_shuffle=True)
        remove_dir(model_dir)

    def test03_model_keras_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.models_tensorflow.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"] * 100)
        x_valid = np.array(["cela est un test", "ni cela", "non plus", "ici test", "là, rien de rien!"] * 100)
        y_train_mono = np.array([0, 1, 0, 1, 2] * 100)
        y_valid_mono = y_train_mono.copy()
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0] * 100, 'test2': [1, 0, 0, 0, 0] * 100, 'test3': [0, 0, 0, 1, 0] * 100})
        y_valid_multi = y_train_multi.copy()
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        model.fit(x_train, y_train_mono)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        preds = model.predict('test', return_proba=False)
        self.assertEqual(preds, model.predict(['test'], return_proba=False)[0])
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 3))
        proba = model.predict('test', return_proba=True)
        self.assertEqual([elem for elem in proba], [elem for elem in model.predict(['test'], return_proba=True)[0]])
        remove_dir(model_dir)

        # Multi-labels
        model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        model.fit(x_train, y_train_multi)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict('test', return_proba=False)
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict(['test'], return_proba=False)[0]])
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), len(cols)))
        proba = model.predict('test', return_proba=True)
        self.assertEqual([elem for elem in proba], [elem for elem in model.predict(['test'], return_proba=True)[0]])
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelEmbeddingLstm(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                       max_sequence_length=10, max_words=100,
                                       embedding_name='fake_embedding.pkl')
            model.predict('test')
        remove_dir(model_dir)

    def test04_model_keras_get_embedding_matrix(self):
        '''Test of the method _get_embedding_matrix of {{package_name}}.models_training.models_tensorflow.model_keras.ModelKeras'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelKeras(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        embedding_null = [0., 0., 0.]
        embedding_toto = [0.25, 0.90, 0.12]
        embedding_test = [0.5, 0.6, 0.1]
        expected_result = [embedding_null,  # token null
                           embedding_toto,  # 'toto'
                           embedding_test]  # 'test'

        # Nominal case
        tokenizer = Tokenizer(num_words=5)
        tokenizer.fit_on_texts(['toto', 'test'])
        embedding_matrix, embedding_size = model._get_embedding_matrix(tokenizer)
        self.assertEqual([list(_) for _ in embedding_matrix], expected_result)
        self.assertEqual(embedding_size, 3)

        # Without num_words
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(['toto', 'test'])
        embedding_matrix, embedding_size = model._get_embedding_matrix(tokenizer)
        self.assertEqual([list(_) for _ in embedding_matrix], expected_result)
        self.assertEqual(embedding_size, 3)

        # Test eager execution that the embedding work as intended
        max_sequence_length = 2
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)
        x = ['test titi toto', 'toto', 'titi test test toto']
        sequences = tokenizer.texts_to_sequences(x)
        expected_result_padding = [[2, 1], [0, 1], [2, 2]]
        expected_result_embedding = [[embedding_test, embedding_toto],
                                     [embedding_null, embedding_toto],
                                     [embedding_test, embedding_test]]
        # Get results
        padded_sequences = pad_sequences(sequences, maxlen=2, padding='pre', truncating='post')
        embedding_out = embedding_layer(padded_sequences)
        self.assertEqual([list(_) for _ in padded_sequences], expected_result_padding)
        self.assertEqual([[list(_) for _ in emb_out] for emb_out in embedding_out], expected_result_embedding)

        # Clean
        remove_dir(model_dir)

    def test05_model_keras_get_sequence(self):
        '''Test of the method _get_sequence of {{package_name}}.models_training.models_tensorflow.model_keras.ModelKeras'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelKeras(model_dir=model_dir, embedding_name='fake_embedding.pkl')

        # Nominal case
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(['toto', 'test'])
        x = ['test titi toto', 'toto', 'titi test test toto']
        # def: padding='pre', truncating='post'
        expected_result = [[2, 1], [0, 1], [2, 2]]
        # Get results
        padded_sequences = model._get_sequence(x, tokenizer, maxlen=2)
        self.assertEqual([list(_) for _ in padded_sequences], expected_result)

        # Test padding = 'post'
        expected_result_padding = [[2, 1], [1, 0], [2, 2]]
        padded_sequences = model._get_sequence(x, tokenizer, maxlen=2, padding='post')
        self.assertEqual([list(_) for _ in padded_sequences], expected_result_padding)

        # Test truncating = 'pre'
        expected_result_truncating = [[2, 1], [0, 1], [2, 1]]
        padded_sequences = model._get_sequence(x, tokenizer, maxlen=2, truncating='pre')
        self.assertEqual([list(_) for _ in padded_sequences], expected_result_truncating)

        # Check errors
        with self.assertRaises(ValueError):
            model._get_sequence(x, tokenizer, maxlen=2, padding='toto')
        with self.assertRaises(ValueError):
            model._get_sequence(x, tokenizer, maxlen=2, truncating='toto')

        # Clean
        remove_dir(model_dir)

    def test06_model_keras_get_callbacks(self):
        '''Test of the method _get_callbacks of {{package_name}}.models_training.models_tensorflow.model_keras.ModelKeras'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelKeras(model_dir=model_dir, embedding_name='fake_embedding.pkl')

        # Nominal case
        callbacks = model._get_callbacks()
        callbacks_types = [type(_) for _ in callbacks]
        self.assertTrue(keras.callbacks.EarlyStopping in callbacks_types)
        self.assertTrue(keras.callbacks.ModelCheckpoint in callbacks_types)
        self.assertTrue(keras.callbacks.CSVLogger in callbacks_types)
        self.assertTrue(keras.callbacks.TerminateOnNaN in callbacks_types)
        checkpoint = callbacks[callbacks_types.index(keras.callbacks.ModelCheckpoint)]
        csv_logger = callbacks[callbacks_types.index(keras.callbacks.CSVLogger)]
        self.assertEqual(checkpoint.filepath, os.path.join(model.model_dir, 'best.hdf5'))
        self.assertEqual(csv_logger.filename, os.path.join(model.model_dir, 'logger.csv'))

        # level save 'LOW'
        model.level_save = 'LOW'
        callbacks = model._get_callbacks()
        callbacks_types = [type(_) for _ in callbacks]
        self.assertTrue(keras.callbacks.EarlyStopping in callbacks_types)
        self.assertFalse(keras.callbacks.ModelCheckpoint in callbacks_types)
        self.assertTrue(keras.callbacks.CSVLogger in callbacks_types)
        self.assertTrue(keras.callbacks.TerminateOnNaN in callbacks_types)
        csv_logger = callbacks[callbacks_types.index(keras.callbacks.CSVLogger)]
        self.assertEqual(csv_logger.filename, os.path.join(model.model_dir, 'logger.csv'))

        # Clean
        remove_dir(model_dir)

    def test07_model_keras_get_learning_rate_scheduler(self):
        '''Test of the method _get_learning_rate_scheduler of {{package_name}}.models_training.models_tensorflow.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelKeras(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        self.assertEqual(model._get_learning_rate_scheduler(), None)

        # Clean
        remove_dir(model_dir)

    def test08_model_keras_save(self):
        '''Test of the method save of {{package_name}}.models_training.models_tensorflow.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelKeras(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_keras.pkl')))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'keras')
        self.assertTrue('batch_size' in configs.keys())
        self.assertTrue('epochs' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('patience' in configs.keys())
        self.assertTrue('embedding_name' in configs.keys())
        self.assertTrue('keras_params' in configs.keys())
        self.assertTrue('_get_model' in configs.keys())
        self.assertTrue('_get_learning_rate_scheduler' in configs.keys())
        self.assertTrue('custom_objects' in configs.keys())

        # Use custom_objects containing a "partial" function
        model = ModelKeras(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        custom_objects = utils_deep_keras.custom_objects
        custom_objects['fb_loss'] = utils_deep_keras.get_fb_loss(0.5)
        model.custom_objects = custom_objects
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_keras.pkl')))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'keras')
        self.assertTrue('batch_size' in configs.keys())
        self.assertTrue('epochs' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('patience' in configs.keys())
        self.assertTrue('embedding_name' in configs.keys())
        self.assertTrue('keras_params' in configs.keys())
        self.assertTrue('_get_model' in configs.keys())
        self.assertTrue('_get_learning_rate_scheduler' in configs.keys())
        self.assertTrue('custom_objects' in configs.keys())

        # Clean
        remove_dir(model_dir)

    def test09_model_keras_hook_post_load_model_pkl(self):
        '''Test of the method _hook_post_load_model_pkl of {{package_name}}.models_training.models_tensorflow.model_keras.ModelKeras'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        new_model_dir = os.path.join(os.getcwd(), 'model_test_987654321')
        remove_dir(new_model_dir)

        old_hdf5_path = os.path.join(model_dir, 'best.hdf5')
        new_hdf5_path = os.path.join(new_model_dir, 'best.hdf5')

        # Nominal case
        model = ModelEmbeddingCnn(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        model.tokenizer = Tokenizer(num_words=model.max_words, filters=model.tokenizer_filters)
        model.list_classes = ['class_1', 'class_2']
        model.model = model._get_model()
        model.model.save(old_hdf5_path)

        new_model = ModelKeras(model_dir=new_model_dir, embedding_name='fake_embedding.pkl')
        shutil.copyfile(old_hdf5_path, new_hdf5_path)
        self.assertTrue(new_model.model is None)
        new_model._hook_post_load_model_pkl()
        self.check_weights_equality(model, new_model)

        remove_dir(new_model_dir)

        # Error
        new_model = ModelKeras(model_dir=new_model_dir, embedding_name='fake_embedding.pkl')
        self.assertTrue(new_model.model is None)
        with self.assertRaises(FileNotFoundError):
            new_model._hook_post_load_model_pkl()

        remove_dir(new_model_dir)
        remove_dir(model_dir)

    def test10_model_keras_init_new_instance_from_configs(self):
        '''Test of the method _init_new_instance_from_configs of {{package_name}}.models_training.models_tensorflow.model_keras.ModelKeras'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)


        # Nominal case
        model = ModelKeras(model_dir=model_dir)
        model.save(json_data={'test': 8})
        configs = model.load_configs(model_dir=model_dir)

        new_model = ModelKeras._init_new_instance_from_configs(configs=configs)
        self.assertTrue(isinstance(new_model, ModelKeras))
        self.assertEqual(new_model.nb_fit, 0)
        self.assertFalse(new_model.trained)
        for attribute in ['x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save', 'batch_size', 'epochs',
                          'patience', 'embedding_name']:
            self.assertEqual(getattr(model, attribute), getattr(new_model, attribute))
        for attribute in ['validation_split']:
            self.assertAlmostEqual(getattr(model, attribute), getattr(new_model, attribute))
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Check by changing some attributes
        model = ModelKeras(model_dir=model_dir)
        model.nb_fit = 2
        model.trained = True
        model.x_col = 'coucou'
        model.y_col = 'coucou_2'
        model.list_classes = ['class_1', 'class_2', 'class_3']
        model.dict_classes = {0: 'class_1', 1: 'class_2', 2: 'class_3'}
        model.multi_label = True
        model.level_save = 'MEDIUM'
        model.batch_size = 13
        model.epochs = 42
        model.validation_split = 0.3
        model.patience = 15
        model.embedding_name = 'coucou_embedding'
        model.keras_params = {'coucou':1, 'coucou2': 0.3, 'coucou3':'coucou4'}
        model.save(json_data={'test': 8})
        configs = model.load_configs(model_dir=model_dir)

        new_model = ModelKeras._init_new_instance_from_configs(configs=configs)
        self.assertTrue(isinstance(new_model, ModelKeras))
        self.assertEqual(new_model.nb_fit, 2)
        self.assertTrue(new_model.trained)
        for attribute in ['x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save', 'batch_size', 'epochs',
                          'patience', 'embedding_name']:
            self.assertEqual(getattr(model, attribute), getattr(new_model, attribute))
        for attribute in ['validation_split']:
            self.assertAlmostEqual(getattr(model, attribute), getattr(new_model, attribute))
        self.assertEqual(set(model.keras_params), set(new_model.keras_params))
        self.assertEqual(model.keras_params['coucou'], new_model.keras_params['coucou'])
        self.assertAlmostEqual(model.keras_params['coucou2'], new_model.keras_params['coucou2'])
        self.assertEqual(model.keras_params['coucou3'], new_model.keras_params['coucou3'])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)
    
    def test11_model_keras_load_standalone_files(self):
        '''Test of the method _load_standalone_files of {{package_name}}.models_training.models_tensorflow.model_keras.ModelKeras'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        new_model_dir = os.path.join(os.getcwd(), 'model_test_987654321')
        remove_dir(new_model_dir)

        old_hdf5_path = os.path.join(model_dir, 'best.hdf5')
        

        # Nominal case with default_model_dir
        model = ModelEmbeddingCnn(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        model.tokenizer = Tokenizer(num_words=model.max_words, filters=model.tokenizer_filters)
        model.list_classes = ['class_1', 'class_2']
        model.model = model._get_model()
        model.model.save(old_hdf5_path)
        model.save(json_data={'test': 8})

        configs = model.load_configs(model_dir=model_dir)
        new_model = ModelKeras._init_new_instance_from_configs(configs=configs)
        new_hdf5_path = os.path.join(new_model.model_dir, 'best.hdf5')
        new_model._load_standalone_files(default_model_dir=model_dir)
        self.assertTrue(os.path.exists(new_hdf5_path))
        self.check_weights_equality(model, new_model)

        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Nominal case with hdf5_path
        model = ModelEmbeddingCnn(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        model.tokenizer = Tokenizer(num_words=model.max_words, filters=model.tokenizer_filters)
        model.list_classes = ['class_1', 'class_2']
        model.model = model._get_model()
        model.model.save(old_hdf5_path)
        model.save(json_data={'test': 8})

        configs = model.load_configs(model_dir=model_dir)
        new_model = ModelKeras._init_new_instance_from_configs(configs=configs)
        new_hdf5_path = os.path.join(new_model.model_dir, 'best.hdf5')
        new_model._load_standalone_files(hdf5_path=old_hdf5_path)
        self.assertTrue(os.path.exists(new_hdf5_path))
        self.check_weights_equality(model, new_model)

        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Errors
        model = ModelEmbeddingCnn(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        model.tokenizer = Tokenizer(num_words=model.max_words, filters=model.tokenizer_filters)
        model.list_classes = ['class_1', 'class_2']
        model.model = model._get_model()
        model.save(json_data={'test': 8})

        configs = model.load_configs(model_dir=model_dir)
        new_model = ModelKeras._init_new_instance_from_configs(configs=configs)
        with self.assertRaises(ValueError):
            new_model._load_standalone_files()
        with self.assertRaises(FileNotFoundError):
            new_model._load_standalone_files(hdf5_path=old_hdf5_path)

        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

    def test12_model_keras_reload_weights(self):
        '''Test of the method _reload_weights of {{package_name}}.models_training.models_tensorflow.model_keras.ModelKeras'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        new_model_dir = os.path.join(os.getcwd(), 'model_test_987654321')
        remove_dir(new_model_dir)
        old_hdf5_path = os.path.join(model_dir, 'best.hdf5')

        model = ModelEmbeddingCnn(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        model.tokenizer = Tokenizer(num_words=model.max_words, filters=model.tokenizer_filters)
        model.list_classes = ['class_1', 'class_2']
        model.model = model._get_model()
        model.model.save(old_hdf5_path)
        model.save(json_data={'test': 8})

        # Nominal case
        new_model = ModelKeras(model_dir=new_model_dir)
        new_model.model = new_model._reload_weights(old_hdf5_path)
        self.check_weights_equality(model, new_model)
        remove_dir(new_model_dir)

        # with custom_objects at None
        new_model = ModelKeras(model_dir=new_model_dir)
        new_model.custom_objects = None
        new_model.model = new_model._reload_weights(old_hdf5_path)
        self.check_weights_equality(model, new_model)
        remove_dir(new_model_dir)

        # without custom_objects at None
        new_model = ModelKeras(model_dir=new_model_dir)
        del new_model.__dict__['custom_objects']
        new_model.model = new_model._reload_weights(old_hdf5_path)
        self.check_weights_equality(model, new_model)
        remove_dir(new_model_dir)

        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
