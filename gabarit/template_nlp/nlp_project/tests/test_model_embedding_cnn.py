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
import dill as pickle
import numpy as np
import pandas as pd

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer

from {{package_name}} import utils
from {{package_name}}.models_training.models_tensorflow.model_embedding_cnn import ModelEmbeddingCnn

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelEmbeddingCnnTests(unittest.TestCase):
    '''Main class to test model_embedding_cnn'''


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

    def test01_model_embedding_cnn_init(self):
        '''Test of {{package_name}}.models_training.models_tensorflow.model_embedding_cnn.ModelEmbeddingCnn.__init__'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        # Init., test all parameters
        model = ModelEmbeddingCnn(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertTrue(model.tokenizer is None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelEmbeddingCnn(model_dir=model_dir, max_sequence_length=20)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.max_sequence_length, 20)
        self.assertTrue(model.tokenizer is None)
        remove_dir(model_dir)

        #
        model = ModelEmbeddingCnn(model_dir=model_dir, max_words=100)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.max_words, 100)
        self.assertTrue(model.tokenizer is None)
        remove_dir(model_dir)

        #
        model = ModelEmbeddingCnn(model_dir=model_dir, padding='post')
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.padding, 'post')
        self.assertTrue(model.tokenizer is None)
        remove_dir(model_dir)

        #
        model = ModelEmbeddingCnn(model_dir=model_dir, truncating='pre')
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.truncating, 'pre')
        self.assertTrue(model.tokenizer is None)
        remove_dir(model_dir)

        #
        model = ModelEmbeddingCnn(model_dir=model_dir, tokenizer_filters="!;/")
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.tokenizer_filters, "!;/")
        self.assertTrue(model.tokenizer is None)
        # We also test if tokenizer_filters works correctly
        x_train = ['test titi toto', 'toto', 'titi test test toto']
        x_train_prepared = model._prepare_x_train(x_train)
        self.assertFalse(model.tokenizer is None)
        self.assertEqual(model.tokenizer.filters, "!;/")
        remove_dir(model_dir)

        # Check errors
        with self.assertRaises(ValueError):
            model = ModelEmbeddingCnn(model_dir=model_dir, padding='toto')
        with self.assertRaises(ValueError):
            model = ModelEmbeddingCnn(model_dir=model_dir, truncating='toto')

    def test02_model_embedding_cnn_predict_proba(self):
        '''Test of {{package_name}}.models_training.models_tensorflow.model_embedding_cnn.ModelEmbeddingCnn.predict_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"] * 100)
        x_test = np.array(["cela est un test", "ni cela", "non plus", "ici test", "là, rien de rien!"] * 100)
        y_train_mono = np.array([0, 1, 0, 1, 2] * 100)
        y_test_mono = y_train_mono.copy()
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0] * 100, 'test2': [1, 0, 0, 0, 0] * 100, 'test3': [0, 0, 0, 1, 0] * 100})
        y_test_multi = y_train_multi.copy()
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        model = ModelEmbeddingCnn(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                  max_sequence_length=10, max_words=100,
                                  padding='pre', truncating='post',
                                  embedding_name='fake_embedding.pkl')
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        #
        model = ModelEmbeddingCnn(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                  max_sequence_length=10, max_words=100,
                                  padding='pre', truncating='post',
                                  embedding_name='fake_embedding.pkl')
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_train, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train), 3))
        preds = model.predict_proba('test', experimental_version=True)
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'], experimental_version=True)[0]])
        remove_dir(model_dir)

        # Multi-labels
        model = ModelEmbeddingCnn(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True,
                                  max_sequence_length=10, max_words=100,
                                  padding='pre', truncating='post',
                                  embedding_name='fake_embedding.pkl')
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        #
        model = ModelEmbeddingCnn(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True,
                                  max_sequence_length=10, max_words=100,
                                  padding='pre', truncating='post',
                                  embedding_name='fake_embedding.pkl')
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test', experimental_version=True)
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'], experimental_version=True)[0]])
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelEmbeddingCnn(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                      max_sequence_length=10, max_words=100,
                                      padding='pre', truncating='post',
                                      embedding_name='fake_embedding.pkl')
            model.predict_proba('test')
        remove_dir(model_dir)

    def test03_model_embedding_cnn_prepare_x_train(self):
        '''Test of {{package_name}}.models_training.models_tensorflow.model_embedding_cnn.ModelEmbeddingCnn._prepare_x_train'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        max_sequence_length = 10
        model = ModelEmbeddingCnn(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                  max_sequence_length=max_sequence_length, max_words=100,
                                  padding='pre', truncating='post',
                                  embedding_name='fake_embedding.pkl')

        # Nominal case
        x_train = ['test titi toto', 'toto', 'titi test test toto']
        x_train_prepared = model._prepare_x_train(x_train)
        # We can't easily test the results, too many dependences
        self.assertEqual(x_train_prepared.shape[0], len(x_train))
        self.assertEqual(x_train_prepared.shape[1], max_sequence_length)

        # Clean
        remove_dir(model_dir)

    def test04_model_embedding_cnn_prepare_x_test(self):
        '''Test of {{package_name}}.models_training.models_tensorflow.model_embedding_cnn.ModelEmbeddingCnn._prepare_x_test'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        max_sequence_length = 10
        model = ModelEmbeddingCnn(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                  max_sequence_length=max_sequence_length, max_words=100,
                                  padding='pre', truncating='post',
                                  embedding_name='fake_embedding.pkl')

        # Nominal case
        x_test = ['test titi toto', 'toto', 'titi test test toto']
        model._prepare_x_train(x_test)  # We force the creation of the tokenizer
        x_test_prepared = model._prepare_x_test(x_test)
        # We can't easily test the results, too many dependences
        self.assertEqual(x_test_prepared.shape[0], len(x_test))
        self.assertEqual(x_test_prepared.shape[1], max_sequence_length)

        # Clean
        remove_dir(model_dir)

    def test05_model_embedding_cnn_get_model(self):
        '''Test of {{package_name}}.models_training.models_tensorflow.model_embedding_cnn.ModelEmbeddingCnn._get_model'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelEmbeddingCnn(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                  max_sequence_length=10, max_words=100,
                                  padding='pre', truncating='post',
                                  embedding_name='fake_embedding.pkl')

        # Nominal case
        x_train = ['test titi toto', 'toto', 'titi test test toto']
        model._prepare_x_train(x_train)  # We force the creation of the tokenizer
        model.list_classes = ['a', 'b']  # We force the creation of a list of classes
        model_res = model._get_model()
        self.assertTrue(isinstance(model_res, keras.Model))

        # With custom Tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(['toto', 'test', 'tata'])
        model_res = model._get_model(custom_tokenizer=tokenizer)
        self.assertTrue(isinstance(model_res, keras.Model))

        # Clean
        remove_dir(model_dir)

    def test06_model_embedding_cnn_save(self):
        '''Test of the method save of {{package_name}}.models_training.models_tensorflow.model_embedding_cnn.ModelEmbeddingCnn'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        max_sequence_length = 10
        max_words = 100
        padding = 'post'
        truncating = 'pre'
        tokenizer_filters = "!;/"

        # Nominal case - without tokenizer
        model = ModelEmbeddingCnn(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                  max_sequence_length=max_sequence_length, max_words=100,
                                  padding=padding, truncating=truncating, tokenizer_filters=tokenizer_filters,
                                  embedding_name='fake_embedding.pkl')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_embedding_cnn.pkl')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'embedding_tokenizer.pkl')))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertEqual(configs['max_sequence_length'], max_sequence_length)
        self.assertEqual(configs['max_words'], max_words)
        self.assertEqual(configs['padding'], padding)
        self.assertEqual(configs['truncating'], truncating)
        self.assertEqual(configs['tokenizer_filters'], tokenizer_filters)
        remove_dir(model_dir)

        # Nominal case - with tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(['toto', 'test', 'tata'])
        model = ModelEmbeddingCnn(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                  max_sequence_length=max_sequence_length, max_words=100,
                                  padding=padding, truncating=truncating, tokenizer_filters=tokenizer_filters,
                                  embedding_name='fake_embedding.pkl')
        model.tokenizer = tokenizer
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_embedding_cnn.pkl')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'embedding_tokenizer.pkl')))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertEqual(configs['max_sequence_length'], max_sequence_length)
        self.assertEqual(configs['max_words'], max_words)
        self.assertEqual(configs['padding'], padding)
        self.assertEqual(configs['truncating'], truncating)
        self.assertEqual(configs['tokenizer_filters'], tokenizer_filters)
        remove_dir(model_dir)

        # Nominal case - with tokenizer, but level_save = 'LOW'
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(['toto', 'test', 'tata'])
        model = ModelEmbeddingCnn(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                  max_sequence_length=max_sequence_length, max_words=100,
                                  padding=padding, truncating=truncating, tokenizer_filters=tokenizer_filters,
                                  embedding_name='fake_embedding.pkl', level_save='LOW')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'model_embedding_cnn.pkl')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'embedding_tokenizer.pkl')))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertEqual(configs['max_sequence_length'], max_sequence_length)
        self.assertEqual(configs['max_words'], max_words)
        self.assertEqual(configs['padding'], padding)
        self.assertEqual(configs['truncating'], truncating)
        self.assertEqual(configs['tokenizer_filters'], tokenizer_filters)
        remove_dir(model_dir)

    def test07_model_embedding_cnn_init_new_instance_from_configs(self):
        '''Test of the method _init_new_instance_from_configs of {{package_name}}.models_training.models_tensorflow.model_embedding_cnn.ModelEmbeddingCnn'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelEmbeddingCnn(model_dir=model_dir)
        model.save(json_data={'test': 8})
        configs = model.load_configs(model_dir=model_dir)

        new_model = ModelEmbeddingCnn._init_new_instance_from_configs(configs=configs)
        self.assertTrue(isinstance(new_model, ModelEmbeddingCnn))
        self.assertEqual(new_model.nb_fit, 0)
        self.assertFalse(new_model.trained)
        for attribute in ['x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save', 'batch_size', 'epochs',
                          'patience', 'embedding_name', 'max_sequence_length', 'max_words', 'padding', 'truncating', 'tokenizer_filters']:
            self.assertEqual(getattr(model, attribute), getattr(new_model, attribute))
        for attribute in ['validation_split']:
            self.assertAlmostEqual(getattr(model, attribute), getattr(new_model, attribute))
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Check by changing some attributes
        model = ModelEmbeddingCnn(model_dir=model_dir)
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
        model.max_sequence_length = 10
        model.max_words = 232
        model.padding = 'post'
        model.truncating = 'pre'
        model.tokenizer_filters = 'coucou'
        model.save(json_data={'test': 8})
        configs = model.load_configs(model_dir=model_dir)

        new_model = ModelEmbeddingCnn._init_new_instance_from_configs(configs=configs)
        self.assertTrue(isinstance(new_model, ModelEmbeddingCnn))
        self.assertEqual(new_model.nb_fit, 2)
        self.assertTrue(new_model.trained)
        for attribute in ['x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save', 'batch_size', 'epochs',
                          'patience', 'embedding_name', 'max_sequence_length', 'max_words', 'padding', 'truncating', 'tokenizer_filters']:
            self.assertEqual(getattr(model, attribute), getattr(new_model, attribute))
        for attribute in ['validation_split']:
            self.assertAlmostEqual(getattr(model, attribute), getattr(new_model, attribute))
        self.assertEqual(set(model.keras_params), set(new_model.keras_params))
        self.assertEqual(model.keras_params['coucou'], new_model.keras_params['coucou'])
        self.assertAlmostEqual(model.keras_params['coucou2'], new_model.keras_params['coucou2'])
        self.assertEqual(model.keras_params['coucou3'], new_model.keras_params['coucou3'])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

    def test08_model_embedding_cnn_load_standalone_files(self):
        '''Test of the method _load_standalone_files of {{package_name}}.models_training.models_tensorflow.model_embedding_cnn.ModelEmbeddingCnn'''
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
        new_model = ModelEmbeddingCnn._init_new_instance_from_configs(configs=configs)
        new_hdf5_path = os.path.join(new_model.model_dir, 'best.hdf5')
        new_model._load_standalone_files(default_model_dir=model_dir)
        self.assertTrue(os.path.exists(new_hdf5_path))
        self.check_weights_equality(model, new_model)
        self.assertTrue(isinstance(new_model.tokenizer, Tokenizer))

        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Nominal case with explicit paths
        model = ModelEmbeddingCnn(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        model.tokenizer = Tokenizer(num_words=model.max_words, filters=model.tokenizer_filters)
        model.list_classes = ['class_1', 'class_2']
        model.model = model._get_model()
        model.model.save(old_hdf5_path)
        model.save(json_data={'test': 8})

        configs = model.load_configs(model_dir=model_dir)
        new_model = ModelEmbeddingCnn._init_new_instance_from_configs(configs=configs)
        new_hdf5_path = os.path.join(new_model.model_dir, 'best.hdf5')
        new_model._load_standalone_files(tokenizer_path=os.path.join(model_dir, 'embedding_tokenizer.pkl'),
                                         hdf5_path=os.path.join(model_dir, 'best.hdf5'))
        self.assertTrue(os.path.exists(new_hdf5_path))
        self.check_weights_equality(model, new_model)
        self.assertTrue(isinstance(new_model.tokenizer, Tokenizer))

        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Errors
        model = ModelEmbeddingCnn(model_dir=model_dir, embedding_name='fake_embedding.pkl')
        model.tokenizer = Tokenizer(num_words=model.max_words, filters=model.tokenizer_filters)
        model.list_classes = ['class_1', 'class_2']
        model.model = model._get_model()
        model.model.save(old_hdf5_path)
        model.save(json_data={'test': 8})
        os.remove(os.path.join(model_dir, 'embedding_tokenizer.pkl'))

        configs = model.load_configs(model_dir=model_dir)
        new_model = ModelEmbeddingCnn._init_new_instance_from_configs(configs=configs)
        with self.assertRaises(ValueError):
            new_model._load_standalone_files()
        with self.assertRaises(FileNotFoundError):
            new_model._load_standalone_files(default_model_dir=model_dir)

        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
