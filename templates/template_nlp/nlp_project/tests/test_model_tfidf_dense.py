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

from {{package_name}} import utils
from {{package_name}}.models_training.model_tfidf_dense import ModelTfidfDense

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelTfidfDenseTests(unittest.TestCase):
    '''Main class to test model_tfidf_dense'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_tfidf_dense_init(self):
        '''Test of {{package_name}}.models_training.test_model_tfidf_dense.ModelTfidfDense.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        model = ModelTfidfDense(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        # Check TFIDF params
        model = ModelTfidfDense(model_dir=model_dir, tfidf_params={'analyzer': 'char', 'binary': True})
        self.assertEqual(model.tfidf.analyzer, 'char')
        self.assertEqual(model.tfidf.binary, True)
        remove_dir(model_dir)

    def test02_model_tfidf_dense_predict_proba(self):
        '''Test of {{package_name}}.models_training.test_model_tfidf_dense.ModelTfidfDense.predict_proba'''

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
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        #
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_train, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train), 3))
        preds = model.predict_proba('test', experimental_version=True)
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'], experimental_version=True)[0]])
        remove_dir(model_dir)

        #
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False, nb_iter_keras=3)
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Multi-labels
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        #
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test', experimental_version=True)
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'], experimental_version=True)[0]])
        remove_dir(model_dir)

        #
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True, nb_iter_keras=3)
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test03_model_tfidf_dense_get_predict_position(self):
        '''Test of the method {{package_name}}.models_training.test_model_tfidf_dense.ModelTfidfDense.get_predict_position'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!", "coucou"])
        y_train_mono = np.array([0, 1, 0, 1, 0, 2])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0, 0], 'test2': [1, 0, 0, 0, 0, 1], 'test3': [0, 0, 0, 1, 0, 1]})
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono)
        predict_positions = model.get_predict_position(x_train, y_train_mono)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)

        # Multi-labels
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        # Unavailable in multi-labels
        with self.assertRaises(ValueError):
            model.get_predict_position(x_train, y_train_mono)
        remove_dir(model_dir)

    def test04_model_tfidf_dense_prepare_x_train(self):
        '''Test of {{package_name}}.models_training.model_tfidf_dense.ModelTfidfDense._prepare_x_train'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)

        # Nominal case
        x_train = ['test titi toto', 'toto', 'titi test test toto', 'titi']
        x_train_prepared = model._prepare_x_train(x_train)
        # Hard to easily test the results. We "only" check shapes
        size_vocab = len(set([word for elem in x_train for word in elem.split()]))
        nb_elems = len(x_train_prepared)
        self.assertEqual(x_train_prepared.shape[0], nb_elems)
        self.assertEqual(x_train_prepared.shape[1], size_vocab)

        # Clean
        remove_dir(model_dir)

    def test05_model_tfidf_dense_prepare_x_test(self):
        '''Test of {{package_name}}.models_training.model_tfidf_dense.ModelTfidfDense._prepare_x_test'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)

        # Nominal case
        x_test = ['test titi toto', 'toto', 'titi test test toto', 'titi']
        model._prepare_x_train(x_test)  # We ensure the creation of the tfidf
        x_test_prepared = model._prepare_x_test(x_test)
        # Hard to easily test the results. We "only" check shapes
        size_vocab = len(set([word for elem in x_test for word in elem.split()]))
        nb_elems = len(x_test_prepared)
        self.assertEqual(x_test_prepared.shape[0], nb_elems)
        self.assertEqual(x_test_prepared.shape[1], size_vocab)

        # Clean
        remove_dir(model_dir)

    def test06_model_tfidf_dense_get_model(self):
        '''Test of {{package_name}}.models_training.model_tfidf_dense.ModelTfidfDense._get_model'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)

        # Nominal case
        x_train = ['test titi toto', 'toto', 'titi test test toto']
        model._prepare_x_train(x_train)  # We force the creation of the tokenizer
        model.list_classes = ['a', 'b']  # We force the creation of a list of classes
        model_res = model._get_model()
        self.assertTrue(isinstance(model_res, keras.Model))

        # Clean
        remove_dir(model_dir)

    def test07_model_tfidf_dense_save(self):
        '''Test of the method save of {{package_name}}.models_training.model_tfidf_dense.ModelTfidfDense'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_tfidf_dense.pkl')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"tfidf_standalone.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        remove_dir(model_dir)

        # Nominal case, but level_save = 'LOW'
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False, level_save='LOW')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'model_tfidf_dense.pkl')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f"tfidf_standalone.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        remove_dir(model_dir)


    def test08_model_tfidf_dense_reload_model(self):
        '''Test of the method reload_model of {{package_name}}.models_training.model_tfidf_dense.ModelTfidfDense'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])
        y_train_mono = np.array(['non', 'oui', 'non', 'oui', 'non'])
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
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

    def test09_test_model_tfidf_dense_reload_from_standalone(self):
        '''Test of the method {{package_name}}.models_training.model_tfidf_dense.ModelTfidfDense.reload'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])
        y_train_mono = np.array(['non', 'oui', 'non', 'oui', 'non'])
        model = ModelTfidfDense(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        tfidf_path = os.path.join(model.model_dir, f"tfidf_standalone.pkl")
        new_model = ModelTfidfDense()
        new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path, tfidf_path=tfidf_path)

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
        self.assertEqual([list(_) for _ in model.predict_proba(x_test)], [list(_) for _ in new_model.predict_proba(x_test)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Check errors
        with self.assertRaises(FileNotFoundError):
            new_model = ModelTfidfDense()
            new_model.reload_from_standalone(configuration_path='toto.json', hdf5_path=hdf5_path, tfidf_path=tfidf_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelTfidfDense()
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path='toto.hdf5', tfidf_path=tfidf_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelTfidfDense()
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path, tfidf_path='toto.pkl')


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
