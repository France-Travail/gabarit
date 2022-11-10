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
import pickle
import numpy as np
import pandas as pd

from {{package_name}} import utils
from {{package_name}}.models_training.model_huggingface import ModelHuggingFace

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelHuggingFaceTests(unittest.TestCase):
    '''Main class to test model_huggingface'''

    def test01_model_huggingface_init(self):
        '''Test of the initialization of {{package_name}}.models_training.model_huggingface.ModelHuggingFace'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        model = ModelHuggingFace(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model, None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8)
        self.assertEqual(model.batch_size, 8)
        remove_dir(model_dir)

        #
        model = ModelHuggingFace(model_dir=model_dir, epochs=42)
        self.assertEqual(model.epochs, 42)
        remove_dir(model_dir)

        #
        model = ModelHuggingFace(model_dir=model_dir, validation_split=0.3)
        self.assertEqual(model.validation_split, 0.3)
        remove_dir(model_dir)

        #
        model = ModelHuggingFace(model_dir=model_dir, transformer_name='toto')
        self.assertEqual(model.transformer_name, 'toto')
        remove_dir(model_dir)

        # transformer_params must accept anything !
        model = ModelHuggingFace(model_dir=model_dir, transformer_params={'toto': 5})
        self.assertEqual(model.transformer_params, {'toto': 5})
        remove_dir(model_dir)

    def test02_model_huggingface_fit(self):
        '''Test of the method fit of {{package_name}}.models_training.model_huggingface.ModelHuggingFace'''
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
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        #self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        #
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=x_valid, y_valid=y_valid_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        #self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        #
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=x_valid, y_valid=y_valid_mono, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        #self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        # Validation with y_train & y_valid of shape 2
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, np.expand_dims(y_train_mono, 1), x_valid=x_valid, y_valid=np.expand_dims(y_valid_mono, 1), with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        #self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        # Missing targets in y_valid
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=x_valid, y_valid=y_valid_mono_missing, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        #self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        # Multi-labels
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['test1', 'test2', 'test3'])
        #self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)
   
        #
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=x_valid, y_valid=y_valid_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['test1', 'test2', 'test3'])
        #self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)

        #
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=x_valid, y_valid=y_valid_multi, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['test1', 'test2', 'test3'])
        #self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        remove_dir(model_dir)


        ###########
        # Test continue training

        # Test mono-label nominal case
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
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
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
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
        with self.assertRaises(AssertionError):
            model.fit(x_train[:50], y_train_mono_fake[:50], x_valid=None, y_valid=None, with_shuffle=True)
        remove_dir(model_dir)

        # Test multi-labels nominal case
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
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
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
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
        with self.assertRaises(AssertionError):
            model.fit(x_train[:50], y_train_multi_fake[:50], x_valid=None, y_valid=None, with_shuffle=True)
        remove_dir(model_dir)

    def test03_model_huggingface_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.model_huggingface.ModelHuggingFace'''

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
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
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
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
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
            model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
            model.predict('test')
        remove_dir(model_dir)

        
    
    def test04_model_huggingface_save(self):
        '''Test of the method save of {{package_name}}.models_training.model_huggingface.ModelHuggingFace'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelHuggingFace(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_huggingface.pkl')))
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
        self.assertEqual(configs['librairie'], 'huggingface')
        self.assertTrue('batch_size' in configs.keys())
        self.assertTrue('epochs' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('patience' in configs.keys())
        self.assertTrue('transformer_name' in configs.keys())
        self.assertTrue('trainer_params' in configs.keys())
        self.assertTrue('_get_model' in configs.keys())

        # Use custom_objects containing a "partial" function
        model = ModelHuggingFace(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        custom_objects = utils_deep_keras.custom_objects
        custom_objects['fb_loss'] = utils_deep_keras.get_fb_loss(0.5)
        model.custom_objects = custom_objects
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_huggingface.pkl')))
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

    def test05_model_huggingface_reload_model(self):
        '''Test of the method reload_model of {{package_name}}.models_training.model_huggingface.ModelHuggingFace'''

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

        # We test with a model embedding LSTM
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                   max_sequence_length=10, max_words=100,
                                   embedding_name='fake_embedding.pkl')
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload keras
        hdf5_path = os.path.join(model.model_dir, 'best.hdf5')
        reloaded_model = model.reload_model(hdf5_path)
        self.assertEqual([list(_) for _ in reloaded_model.predict(model._prepare_x_test(['test', 'toto', 'titi']))], [list(_) for _ in model.predict_proba(['test', 'toto', 'titi'])])

        # Test without custom_objects
        model.custom_objects = None
        reloaded_model = model.reload_model(hdf5_path)
        self.assertEqual([list(_) for _ in reloaded_model.predict(model._prepare_x_test(['test', 'toto', 'titi']))], [list(_) for _ in model.predict_proba(['test', 'toto', 'titi'])])

        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
