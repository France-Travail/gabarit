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
import torch
import shutil
import pickle
import numpy as np
import pandas as pd

import tensorflow
from transformers import CONFIG_NAME, WEIGHTS_NAME
from tensorflow.keras.preprocessing.text import Tokenizer

from {{package_name}} import utils
from {{package_name}}.models_training import utils_models
from {{package_name}}.models_training.model_pytorch_light import ModelPyTorchTransformersLight
from {{package_name}}.models_training.model_pytorch_transformers import ModelPyTorchTransformers, TaskClass

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelPyTorchTransformersLightTests01(unittest.TestCase):
    '''Main class to test model_pytorch_light - Without specific setUp'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)
        # Check if data folder exists
        data_path = utils.get_data_path()
        if not os.path.exists(data_path):
            os.mkdir(data_path)

    def test01_model_pytorch_light_init(self):
        '''Test of {{package_name}}.models_training.model_pytorch_light.ModelPyTorchTransformersLight.__init__'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        # Init., test all parameters
        model = ModelPyTorchTransformersLight(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformersLight(model_dir=model_dir, max_sequence_length=20)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.max_sequence_length, 20)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformersLight(model_dir=model_dir, tokenizer_special_tokens=('totototototototototo'))
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.tokenizer_special_tokens, ('totototototototototo'))
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformersLight(model_dir=model_dir, padding='longest')
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.padding, 'longest')
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformersLight(model_dir=model_dir, truncation=False)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.truncation, False)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformersLight(model_dir=model_dir, batch_size=9)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.batch_size, 9)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformersLight(model_dir=model_dir, epochs=9)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.epochs, 9)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformersLight(model_dir=model_dir, validation_split=0.3)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.validation_split, 0.3)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformersLight(model_dir=model_dir, patience=18)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.patience, 18)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformersLight(model_dir=model_dir, pytorch_params={'titi': 'toto'})
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.pytorch_params, {'titi': 'toto'})
        remove_dir(model_dir)


class ModelPyTorchTransformersLightTests02(unittest.TestCase):
    '''Main class to test model_pytorch_light - With specific setUp'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)
        # Check if data folder exists
        data_path = utils.get_data_path()
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        # Create models
        model_dir_mono = os.path.join(os.getcwd(), 'model_test_mono')
        model_dir_mono_light = os.path.join(os.getcwd(), 'model_test_mono_light')
        model_dir_multi = os.path.join(os.getcwd(), 'model_test_multi')
        model_dir_multi_light = os.path.join(os.getcwd(), 'model_test_multi_light')
        remove_dir(model_dir_mono)
        remove_dir(model_dir_mono_light)
        remove_dir(model_dir_multi)
        remove_dir(model_dir_multi_light)
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"] * 100)
        y_train_mono = np.array([0, 1, 0, 1, 2] * 100)
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0] * 100, 'test2': [1, 0, 0, 0, 0] * 100, 'test3': [0, 0, 0, 1, 0] * 100})

        # Mono-label
        model = ModelPyTorchTransformers(model_dir=model_dir_mono, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=10,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=False)
        model.fit(x_train, y_train_mono)
        model.save()
        model, model_conf = utils_models.load_model(model_dir_mono, is_path=True)
        light_conf = {}
        conf_keys = ["multi_label", "max_sequence_length",
                     "tokenizer_special_tokens", "padding", "truncation"]
        for k in conf_keys:
            if k in model_conf.keys():
                light_conf[k] = model_conf[k]
        new_model = ModelPyTorchTransformersLight(model_dir=model_dir_mono_light, **light_conf)
        output_model_file = os.path.join(model_dir_mono, WEIGHTS_NAME)
        output_config_file = os.path.join(model_dir_mono, CONFIG_NAME)
        torch.save(model.model.model.state_dict(), output_model_file)
        model.model.model.config.to_json_file(output_config_file)
        model.tokenizer.save_vocabulary(model_dir_mono)
        new_model.trained = model.trained
        new_model.nb_fit = model.nb_fit
        new_model.list_classes = model.list_classes
        new_model.dict_classes = model.dict_classes
        new_model.x_col = model.x_col
        new_model.y_col = model.y_col
        new_model.multi_label = model.multi_label
        new_model.pytorch_params = model.pytorch_params
        new_model.model = new_model.reload_model(model_dir_mono)
        new_model.tokenizer = new_model.reload_tokenizer(model_dir_mono)
        new_model.save()

        # Multi-labels
        model = ModelPyTorchTransformers(model_dir=model_dir_multi, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=10,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=True)
        model.fit(x_train, y_train_multi)
        model.save()
        model, model_conf = utils_models.load_model(model_dir_multi, is_path=True)
        light_conf = {}
        conf_keys = ["multi_label", "max_sequence_length",
                     "tokenizer_special_tokens", "padding", "truncation"]
        for k in conf_keys:
            if k in model_conf.keys():
                light_conf[k] = model_conf[k]
        new_model = ModelPyTorchTransformersLight(model_dir=model_dir_multi_light, **light_conf)
        output_model_file = os.path.join(model_dir_multi, WEIGHTS_NAME)
        output_config_file = os.path.join(model_dir_multi, CONFIG_NAME)
        torch.save(model.model.model.state_dict(), output_model_file)
        model.model.model.config.to_json_file(output_config_file)
        model.tokenizer.save_vocabulary(model_dir_multi)
        new_model.trained = model.trained
        new_model.nb_fit = model.nb_fit
        new_model.list_classes = model.list_classes
        new_model.dict_classes = model.dict_classes
        new_model.x_col = model.x_col
        new_model.y_col = model.y_col
        new_model.multi_label = model.multi_label
        new_model.pytorch_params = model.pytorch_params
        new_model.model = new_model.reload_model(model_dir_multi)
        new_model.tokenizer = new_model.reload_tokenizer(model_dir_multi)
        new_model.save()

    def tearDown(self):
        '''Cleaning fonction -> on supprimer les temp models'''
        model_dir_mono = os.path.join(os.getcwd(), 'model_test_mono')
        model_dir_mono_light = os.path.join(os.getcwd(), 'model_test_mono_light')
        model_dir_multi = os.path.join(os.getcwd(), 'model_test_multi')
        model_dir_multi_light = os.path.join(os.getcwd(), 'model_test_multi_light')
        remove_dir(model_dir_mono)
        remove_dir(model_dir_multi)
        remove_dir(model_dir_mono_light)
        remove_dir(model_dir_multi_light)

    def test01_model_pytorch_light_predict_proba(self):
        '''Test of {{package_name}}.models_training.model_pytorch_light.ModelPyTorchTransformersLight.predict_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_dir_mono = os.path.join(os.getcwd(), 'model_test_mono')
        model_dir_mono_light = os.path.join(os.getcwd(), 'model_test_mono_light')
        model_dir_multi = os.path.join(os.getcwd(), 'model_test_multi')
        model_dir_multi_light = os.path.join(os.getcwd(), 'model_test_multi_light')

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"] * 100)
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        model, _ = utils_models.load_model(model_dir_mono_light, is_path=True)
        model_2, _ = utils_models.load_model(model_dir_mono, is_path=True)
        preds = model.predict_proba(x_train)
        preds_2 = model_2.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3))
        # np.tolist breaks precision, so we can't use np.around ...
        np.testing.assert_array_almost_equal(preds, preds_2, decimal=5)
        # self.assertEqual([[round(_, 4) for _ in nested] for nested in preds], [[round(_, 4) for _ in nested] for nested in preds_2])
        preds = model.predict_proba('test')
        preds_2 = model_2.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        np.testing.assert_array_almost_equal(preds, preds_2, decimal=5)
        # self.assertEqual([round(_, 4) for _ in preds], [round(_, 4) for _ in preds_2])

        # Multi-labels
        model, _ = utils_models.load_model(model_dir_multi_light, is_path=True)
        model_2, _ = utils_models.load_model(model_dir_multi, is_path=True)
        preds = model.predict_proba(x_train)
        preds_2 = model_2.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        # np.tolist breaks precision, so we can't use np.around ...
        np.testing.assert_array_almost_equal(preds, preds_2, decimal=5)
        # self.assertEqual([[round(_, 4) for _ in nested] for nested in preds], [[round(_, 4) for _ in nested] for nested in preds_2])
        preds = model.predict_proba('test')
        preds_2 = model_2.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        np.testing.assert_array_almost_equal(preds, preds_2, decimal=5)
        # self.assertEqual([round(_, 4) for _ in preds], [round(_, 4) for _ in preds_2])

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelPyTorchTransformersLight(model_dir=model_dir)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test02_model_pytorch_light_save(self):
        '''Test of the method save of {{package_name}}.models_training.model_pytorch_light.ModelPyTorchTransformersLight'''
        model_dir_mono_light = os.path.join(os.getcwd(), 'model_test_mono_light')
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Reload a model & save
        model = ModelPyTorchTransformersLight(model_dir=model_dir)
        configuration_path = os.path.join(model_dir_mono_light, 'configurations.json')
        model.reload_from_standalone(configuration_path=configuration_path, torch_dir=model_dir_mono_light)
        model.save(json_data={'test': 8})

        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_pytorch_light.pkl')))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertEqual(configs['max_sequence_length'], model.max_sequence_length)
        self.assertEqual(configs['tokenizer_special_tokens'], list(model.tokenizer_special_tokens)) # casted in list when saved
        self.assertEqual(configs['padding'], model.padding)
        self.assertEqual(configs['truncation'], model.truncation)
        self.assertEqual(configs['pytorch_params'], model.pytorch_params)
        remove_dir(model_dir)

    def test03_model_pytorch_light_reload_model(self):
        '''Test of the method reload_model of {{package_name}}.models_training.model_pytorch_light.ModelPyTorchTransformersLight'''
        model_dir_mono_light = os.path.join(os.getcwd(), 'model_test_mono_light')

        # Load model
        model, _ = utils_models.load_model(model_dir_mono_light, is_path=True)

        # Reload pytorch
        torch_dir = model.model_dir
        reloaded_model = model.reload_model(torch_dir)

        # Check same preds
        x_val = ['test', 'toto', 'titi']
        initial_probas = model.predict_proba(x_val)
        model.model = reloaded_model
        model.freeze()
        new_probas = model.predict_proba(x_val)
        self.assertEqual([list(_) for _ in new_probas], [list(_) for _ in initial_probas])

    def test04_model_pytorch_light_reload_tokenizer(self):
        '''Test of the method reload_tokenizer of {{package_name}}.models_training.model_pytorch_light.ModelPyTorchTransformersLight'''
        model_dir_mono_light = os.path.join(os.getcwd(), 'model_test_mono_light')

        # Load model
        model, _ = utils_models.load_model(model_dir_mono_light, is_path=True)

        # Reload pytorch
        torch_dir = model.model_dir
        reloaded_tokenizer = model.reload_tokenizer(torch_dir)

        # Check same preds
        x_val = ['test', 'toto', 'titi']
        initial_probas = model.predict_proba(x_val)
        model.tokenizer = reloaded_tokenizer
        new_probas = model.predict_proba(x_val)
        self.assertEqual([list(_) for _ in new_probas], [list(_) for _ in initial_probas])

    def test05_test_model_pytorch_light_reload_from_standalone(self):
        '''Test of the method {{package_name}}.models_training.model_pytorch_light.ModelPyTorchTransformersLight.reload'''

        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])

        # Model already saved
        model_dir_mono_light = os.path.join(os.getcwd(), 'model_test_mono_light')
        model, _ = utils_models.load_model(model_dir_mono_light, is_path=True)

        # Reload
        conf_path = os.path.join(model_dir_mono_light, 'configurations.json')
        pytorch_path = model_dir_mono_light
        new_model = ModelPyTorchTransformersLight()
        new_model.reload_from_standalone(configuration_path=conf_path, torch_dir=pytorch_path)

        # Test
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.x_col, new_model.x_col)
        self.assertEqual(model.y_col, new_model.y_col)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.batch_size, new_model.batch_size)
        self.assertEqual(model.epochs, new_model.epochs)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.patience, new_model.patience)
        self.assertEqual(model.embedding_name, new_model.embedding_name)
        self.assertEqual(model.max_sequence_length, new_model.max_sequence_length)
        self.assertEqual(list(model.tokenizer_special_tokens), list(new_model.tokenizer_special_tokens))
        self.assertEqual(model.padding, new_model.padding)
        self.assertEqual(model.truncation, new_model.truncation)
        self.assertEqual(model.pytorch_params, new_model.pytorch_params)
        self.assertEqual([list(_) for _ in model.predict_proba(x_test)], [list(_) for _ in new_model.predict_proba(x_test)])
        remove_dir(new_model.model_dir)

        # Check errors
        with self.assertRaises(FileNotFoundError):
            new_model = ModelPyTorchTransformersLight()
            new_model.reload_from_standalone(configuration_path='toto.json', torch_dir=pytorch_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelPyTorchTransformersLight()
            new_model.reload_from_standalone(configuration_path=conf_path, torch_dir='toto_dir')


# Perform tests
if __name__ == '__main__':
    # Run only if transformer loaded locally
    transformer_path = os.path.join(utils.get_transformers_path(), 'flaubert', 'flaubert_small_cased')
    if os.path.exists(transformer_path):
        # Start tests
        unittest.main()
    else:
        print(f"{transformer_path} does not exists. We won't start the tests for model_pytorch_light.py")
        print("Remainder : this is still experimental")
