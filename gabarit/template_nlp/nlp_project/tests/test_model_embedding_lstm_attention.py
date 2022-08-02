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
import pickle
import numpy as np
import pandas as pd

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer

from {{package_name}} import utils
from {{package_name}}.models_training.model_embedding_lstm_attention import ModelEmbeddingLstmAttention

# Disable logging
import logging
logging.disable(logging.CRITICAL)

def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelEmbeddingLstmAttentionTests(unittest.TestCase):
    '''Main class to test model_embedding_lstm_attention'''

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


    def test01_model_embedding_lstm_attention_init(self):
        '''Test of {{package_name}}.models_training.model_embedding_lstm_attention.ModelEmbeddingLstmAttention.__init__'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        # Init., test all parameters
        model = ModelEmbeddingLstmAttention(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertTrue(model.tokenizer is None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, max_sequence_length=20)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.max_sequence_length, 20)
        self.assertTrue(model.tokenizer is None)
        remove_dir(model_dir)

        #
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, max_words=100)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.max_words, 100)
        self.assertTrue(model.tokenizer is None)
        remove_dir(model_dir)

        #
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, padding='post')
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.padding, 'post')
        self.assertTrue(model.tokenizer is None)
        remove_dir(model_dir)

        #
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, truncating='pre')
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.truncating, 'pre')
        self.assertTrue(model.tokenizer is None)
        remove_dir(model_dir)

        #
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, tokenizer_filters="!;/")
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
            model = ModelEmbeddingLstmAttention(model_dir=model_dir, padding='toto')
        with self.assertRaises(ValueError):
            model = ModelEmbeddingLstmAttention(model_dir=model_dir, truncating='toto')

    def test02_model_embedding_lstm_attention_predict_proba(self):
        '''Test of {{package_name}}.models_training.model_embedding_lstm_attention.ModelEmbeddingLstmAttention.predict_proba'''

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
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
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
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                            max_sequence_length=10, max_words=100,
                                            padding='pre', truncating='post',
                                            embedding_name='fake_embedding.pkl')
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_train, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train), 3))
        preds = model.predict_proba('test', experimental_version=True)
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'], experimental_version=True)[0]])
        remove_dir(model_dir)

        #
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                            max_sequence_length=10, max_words=100,
                                            padding='pre', truncating='post',
                                            embedding_name='fake_embedding.pkl', nb_iter_keras=3)
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Multi-labels
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True,
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
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True,
                                            max_sequence_length=10, max_words=100,
                                            padding='pre', truncating='post',
                                            embedding_name='fake_embedding.pkl')
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test', experimental_version=True)
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'], experimental_version=True)[0]])
        remove_dir(model_dir)

        #
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True,
                                            max_sequence_length=10, max_words=100,
                                            padding='pre', truncating='post',
                                            embedding_name='fake_embedding.pkl', nb_iter_keras=3)
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                                max_sequence_length=10, max_words=100,
                                                padding='pre', truncating='post',
                                                embedding_name='fake_embedding.pkl')
            model.predict_proba('test')
        remove_dir(model_dir)

    def test03_model_embedding_lstm_attention_prepare_x_train(self):
        '''Test of {{package_name}}.models_training.model_embedding_lstm_attention.ModelEmbeddingLstmAttention._prepare_x_train'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        max_sequence_length = 10
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
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

    def test04_model_embedding_lstm_attention_prepare_x_test(self):
        '''Test of {{package_name}}.models_training.model_embedding_lstm_attention.ModelEmbeddingLstmAttention._prepare_x_test'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        max_sequence_length = 10
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
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

    def test05_model_embedding_lstm_attention_get_model(self):
        '''Test of {{package_name}}.models_training.model_embedding_lstm_attention.ModelEmbeddingLstmAttention._get_model'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
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

    def test06_model_embedding_lstm_attention_save(self):
        '''Test of the method save of {{package_name}}.models_training.model_embedding_lstm_attention.ModelEmbeddingLstmAttention'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        max_sequence_length = 10
        max_words = 100
        padding = 'post'
        truncating = 'pre'
        tokenizer_filters = "!;/"

        # Nominal case - without tokenizer
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                            max_sequence_length=max_sequence_length, max_words=100,
                                            padding=padding, truncating=truncating, tokenizer_filters=tokenizer_filters,
                                            embedding_name='fake_embedding.pkl')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_embedding_lstm_attention.pkl')))
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
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                            max_sequence_length=max_sequence_length, max_words=100,
                                            padding=padding, truncating=truncating, tokenizer_filters=tokenizer_filters,
                                            embedding_name='fake_embedding.pkl')
        model.tokenizer = tokenizer
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_embedding_lstm_attention.pkl')))
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
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                            max_sequence_length=max_sequence_length, max_words=100,
                                            padding=padding, truncating=truncating, tokenizer_filters=tokenizer_filters,
                                            embedding_name='fake_embedding.pkl', level_save='LOW')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'model_embedding_lstm_attention.pkl')))
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

    def test07_model_embedding_lstm_attention_reload_model(self):
        '''Test of the method reload_model of {{package_name}}.models_training.model_embedding_lstm_attention.ModelEmbeddingLstmAttention'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])
        y_train_mono = np.array(['non', 'oui', 'non', 'oui', 'non'])
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                            max_sequence_length=10, max_words=100,
                                            padding='post', truncating='pre',
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

    def test08_test_model_embedding_lstm_attention_reload_from_standalone(self):
        '''Test of the method {{package_name}}.models_training.model_embedding_lstm_attention.ModelEmbeddingLstmAttention.reload'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])
        y_train_mono = np.array(['non', 'oui', 'non', 'oui', 'non'])
        model = ModelEmbeddingLstmAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                            max_sequence_length=10, max_words=100,
                                            padding='post', truncating='pre',
                                            embedding_name='fake_embedding.pkl')
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        tokenizer_path = os.path.join(model.model_dir, 'embedding_tokenizer.pkl')
        new_model = ModelEmbeddingLstmAttention()
        new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path, tokenizer_path=tokenizer_path)

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
        self.assertEqual(model.max_words, new_model.max_words)
        self.assertEqual(model.padding, new_model.padding)
        self.assertEqual(model.truncating, new_model.truncating)
        self.assertEqual(model.tokenizer_filters, new_model.tokenizer_filters)
        self.assertEqual([list(_) for _ in model.predict_proba(x_test)], [list(_) for _ in new_model.predict_proba(x_test)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Check errors
        with self.assertRaises(FileNotFoundError):
            new_model = ModelEmbeddingLstmAttention()
            new_model.reload_from_standalone(configuration_path='toto.json', hdf5_path=hdf5_path, tokenizer_path=tokenizer_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelEmbeddingLstmAttention()
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path='toto.hdf5', tokenizer_path=tokenizer_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelEmbeddingLstmAttention()
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path, tokenizer_path='toto.pkl')


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
