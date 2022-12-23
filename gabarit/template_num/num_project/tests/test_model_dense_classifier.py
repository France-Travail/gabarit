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
import tensorflow
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from {{package_name}} import utils
from {{package_name}}.models_training import utils_deep_keras
from {{package_name}}.models_training.classifiers.models_tensorflow.model_dense_classifier import ModelDenseClassifier

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelDenseClassifierTests(unittest.TestCase):
    '''Main class to test model_dense_classifier'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_dense_classifier_init(self):
        '''Test of {{package_name}}.models_training.classifiers.models_tensorflow.model_dense_classifier.ModelDenseClassifier.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)


        # Init., test all parameters
        model = ModelDenseClassifier(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.model_type, 'classifier')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model, None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8)
        self.assertEqual(model.batch_size, 8)
        remove_dir(model_dir)

        #
        model = ModelDenseClassifier(model_dir=model_dir, epochs=42)
        self.assertEqual(model.epochs, 42)
        remove_dir(model_dir)

        #
        model = ModelDenseClassifier(model_dir=model_dir, validation_split=0.3)
        self.assertEqual(model.validation_split, 0.3)
        remove_dir(model_dir)

        #
        model = ModelDenseClassifier(model_dir=model_dir, patience=65)
        self.assertEqual(model.patience, 65)
        remove_dir(model_dir)

        # keras_params must accept anything !
        model = ModelDenseClassifier(model_dir=model_dir, keras_params={'toto': 5})
        self.assertEqual(model.keras_params, {'toto': 5})
        remove_dir(model_dir)

    # We do not test the fit method. It is already done in test_model_keras.py

    def test02_model_dense_classifier_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.classifiers.models_tensorflow.model_dense_classifier.ModelDenseClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        x_train_inv = pd.DataFrame({'col_2': [2, -1, -8, 2, 3, 12, 2] * 10, 'fake_col': [0.5, -3, 5, 5, 2, 0, 8] * 10, 'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Classification - Mono-label - Mono-Class
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono_2)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        probas = model.predict(x_train, return_proba=True)
        self.assertEqual(probas.shape, (len(x_train), 2))  # 2 classes
        preds = model.predict(x_train, return_proba=False, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train),))
        probas = model.predict(x_train, return_proba=True, experimental_version=True)
        self.assertEqual(probas.shape, (len(x_train), 2))  # 2 classes
        # Test inversed columns order
        preds_inv = model.predict(x_train_inv, return_proba=False)
        np.testing.assert_almost_equal(preds, preds_inv, decimal=5)
        probas_inv = model.predict(x_train_inv, return_proba=True)
        np.testing.assert_almost_equal(probas, probas_inv, decimal=5)
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono_3)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        probas = model.predict(x_train, return_proba=True)
        self.assertEqual(probas.shape, (len(x_train), 3))  # 3 classes
        preds = model.predict(x_train, return_proba=False, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train),))
        probas = model.predict(x_train, return_proba=True, experimental_version=True)
        self.assertEqual(probas.shape, (len(x_train), 3))  # 3 classes
        # Test inversed columns order
        preds_inv = model.predict(x_train_inv, return_proba=False)
        np.testing.assert_almost_equal(preds, preds_inv, decimal=5)
        probas_inv = model.predict(x_train_inv, return_proba=True)
        np.testing.assert_almost_equal(probas, probas_inv, decimal=5)
        remove_dir(model_dir)

        # Classification - Multi-labels
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        model.fit(x_train, y_train_multi)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi)))
        probas = model.predict(x_train, return_proba=True)
        self.assertEqual(probas.shape, (len(x_train), len(y_col_multi)))
        preds = model.predict(x_train, return_proba=False, experimental_version=True)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi)))
        probas = model.predict(x_train, return_proba=True, experimental_version=True)
        self.assertEqual(probas.shape, (len(x_train), len(y_col_multi)))
        # Test inversed columns order
        preds_inv = model.predict(x_train_inv, return_proba=False)
        np.testing.assert_almost_equal(preds, preds_inv, decimal=5)
        probas_inv = model.predict(x_train_inv, return_proba=True)
        np.testing.assert_almost_equal(probas, probas_inv, decimal=5)
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
            model.predict(x_train)
        remove_dir(model_dir)

    def test03_model_dense_classifier_predict_proba(self):
        '''Test of the method predict_proba of {{package_name}}.models_training.classifiers.models_tensorflow.model_dense_classifier.ModelDenseClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        x_train_inv = pd.DataFrame({'col_2': [2, -1, -8, 2, 3, 12, 2] * 10, 'fake_col': [0.5, -3, 5, 5, 2, 0, 8] * 10, 'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Classification - Mono-label - Mono-Class
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono_2)
        probas = model.predict_proba(x_train)
        self.assertEqual(probas.shape, (len(x_train), 2))  # 2 classes
        self.assertTrue(isinstance(probas[0][0], (np.floating, float)))
        # Test inversed columns order
        probas_inv = model.predict_proba(x_train_inv)
        np.testing.assert_almost_equal(probas, probas_inv, decimal=5)
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono_3)
        probas = model.predict_proba(x_train)
        self.assertEqual(probas.shape, (len(x_train), 3))  # 3 classes
        self.assertTrue(isinstance(probas[0][0], (np.floating, float)))
        # Test inversed columns order
        probas_inv = model.predict_proba(x_train_inv)
        np.testing.assert_almost_equal(probas, probas_inv, decimal=5)
        remove_dir(model_dir)

        # Classification - Multi-labels
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        model.fit(x_train, y_train_multi)
        probas = model.predict_proba(x_train)
        self.assertEqual(probas.shape, (len(x_train), len(y_col_multi)))  # 3 labels
        self.assertTrue(isinstance(probas[0][0], (np.floating, float)))
        # Test inversed columns order
        probas_inv = model.predict_proba(x_train_inv)
        np.testing.assert_almost_equal(probas, probas_inv, decimal=5)
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test04_model_dense_classifier_get_predict_position(self):
        '''Test of the method {{package_name}}.models_training.classifiers.models_tensorflow.model_dense_classifier.ModelDenseClassifier.get_predict_position'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        x_train_inv = pd.DataFrame({'col_2': [2, -1, -8, 2, 3, 12, 2] * 10, 'fake_col': [0.5, -3, 5, 5, 2, 0, 8] * 10, 'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Classification - Mono-label - Mono-Class
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)
        model.fit(x_train, y_train_mono_2)
        predict_positions = model.get_predict_position(x_train, y_train_mono_2)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        # Test inversed columns order
        predict_positions_inv = model.get_predict_position(x_train_inv, y_train_mono_2)
        np.testing.assert_almost_equal(predict_positions, predict_positions_inv, decimal=5)
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)
        model.fit(x_train, y_train_mono_3)
        predict_positions = model.get_predict_position(x_train, y_train_mono_2)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        # Test inversed columns order
        predict_positions_inv = model.get_predict_position(x_train_inv, y_train_mono_2)
        np.testing.assert_almost_equal(predict_positions, predict_positions_inv, decimal=5)
        remove_dir(model_dir)

        # Classification - Multi-labels
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True)
        model.fit(x_train, y_train_multi)  # Unavailable in multi-labels
        with self.assertRaises(ValueError):
            model.get_predict_position(x_train, y_train_multi)
        remove_dir(model_dir)

    def test05_model_dense_classifier_get_model(self):
        '''Test of the method {{package_name}}.models_training.classifiers.models_tensorflow.model_dense_classifier.ModelDenseClassifier._get_model'''

        # Set vars
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)

        # Nominal case
        model.list_classes = ['a', 'b']  # We force the creation of a list of classes
        model_res = model._get_model()
        self.assertTrue(isinstance(model_res, keras.Model))

        # Clean
        remove_dir(model_dir)

        #######
        # Same thing with multi-labels

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True)

        # Nominal case
        model.list_classes = ['a', 'b', 'c']  # We force the creation of a list of classes
        model_res = model._get_model()
        self.assertTrue(isinstance(model_res, keras.Model))

        # Clean
        remove_dir(model_dir)

    def test06_model_dense_classifier_save(self):
        '''Test of the method save of {{package_name}}.models_training.classifiers.models_tensorflow.model_dense_classifier.ModelDenseClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']

        # Nominal case
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'preprocess_pipeline.pkl')))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertTrue('maintainers' in configs.keys())
        self.assertTrue('date' in configs.keys())
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('model_type' in configs.keys())
        self.assertEqual(configs['model_type'], 'classifier')
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('columns_in' in configs.keys())
        self.assertTrue('mandatory_columns' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'keras')
        self.assertTrue('batch_size' in configs.keys())
        self.assertTrue('epochs' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('patience' in configs.keys())
        self.assertTrue('keras_params' in configs.keys())
        self.assertTrue('_get_model' in configs.keys())
        self.assertTrue('_get_learning_rate_scheduler' in configs.keys())
        self.assertTrue('custom_objects' in configs.keys())
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        remove_dir(model_dir)

        # Use custom_objects containing a "partial" function
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)
        custom_objects = utils_deep_keras.custom_objects
        custom_objects['fb_loss'] = utils_deep_keras.get_fb_loss(0.5)
        model.custom_objects = custom_objects
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'preprocess_pipeline.pkl')))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertTrue('maintainers' in configs.keys())
        self.assertTrue('date' in configs.keys())
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('model_type' in configs.keys())
        self.assertEqual(configs['model_type'], 'classifier')
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('columns_in' in configs.keys())
        self.assertTrue('mandatory_columns' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'keras')
        self.assertTrue('batch_size' in configs.keys())
        self.assertTrue('epochs' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('patience' in configs.keys())
        self.assertTrue('keras_params' in configs.keys())
        self.assertTrue('_get_model' in configs.keys())
        self.assertTrue('_get_learning_rate_scheduler' in configs.keys())
        self.assertTrue('custom_objects' in configs.keys())
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        remove_dir(model_dir)

    def test07_model_dense_classifier_reload_model(self):
        '''Test of the method reload_model of {{package_name}}.models_training.classifiers.models_tensorflow.model_dense_classifier.ModelDenseClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})

        # Classification - Mono-label - Mono-Class
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono_2)
        model.save()
        # Reload keras
        hdf5_path = os.path.join(model.model_dir, 'best.hdf5')
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_train)], [list(_) for _ in model.model.predict(x_train)], decimal=3)
        # Test without custom_objects
        model.custom_objects = None
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_train)], [list(_) for _ in model.model.predict(x_train)], decimal=3)
        # Clean
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono_3)
        model.save()
        # Reload keras
        hdf5_path = os.path.join(model.model_dir, 'best.hdf5')
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_train)], [list(_) for _ in model.model.predict(x_train)], decimal=3)
        # Test without custom_objects
        model.custom_objects = None
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_train)], [list(_) for _ in model.model.predict(x_train)], decimal=3)
        # Clean
        remove_dir(model_dir)

        # Classification - Multi-labels
        model = ModelDenseClassifier(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        model.fit(x_train, y_train_multi)
        model.save()
        # Reload keras
        hdf5_path = os.path.join(model.model_dir, 'best.hdf5')
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_train)], [list(_) for _ in model.model.predict(x_train)], decimal=3)
        # Test without custom_objects
        model.custom_objects = None
        reloaded_model = model.reload_model(hdf5_path)
        np.testing.assert_almost_equal([list(_) for _ in reloaded_model.predict(x_train)], [list(_) for _ in model.model.predict(x_train)], decimal=3)
        # Clean
        remove_dir(model_dir)

    def test08_model_dense_classifier_reload_from_standalone(self):
        '''Test of the method {{package_name}}.models_training.classifiers.models_tensorflow.model_dense_classifier.ModelDenseClassifier.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        model_dir_2 = os.path.join(os.getcwd(), 'model_test_123456789_2')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        ############################################
        # Classification - Mono label
        ############################################

        # Create model
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, epochs=2)
        model.fit(x_train, y_train_mono_2)
        model.save()
        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelDenseClassifier(model_dir=model_dir_2)
        self.assertTrue(new_model.preprocess_pipeline is None)
        new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
                                         preprocess_pipeline_path=preprocess_pipeline_path)
        # Tests
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.model_type, new_model.model_type)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.x_col, new_model.x_col)
        self.assertEqual(model.y_col, new_model.y_col)
        self.assertEqual(model.columns_in, new_model.columns_in)
        self.assertEqual(model.mandatory_columns, new_model.mandatory_columns)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.batch_size, new_model.batch_size)
        self.assertEqual(model.epochs, new_model.epochs)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.patience, new_model.patience)
        self.assertEqual(model.keras_params, new_model.keras_params)
        self.assertEqual(model.custom_objects, new_model.custom_objects)
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_train)], [list(_) for _ in new_model.predict_proba(x_train)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # Classification - Multi-labels
        ############################################

        # Create model
        model = ModelDenseClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True, epochs=2)
        model.fit(x_train, y_train_multi)
        model.save()
        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelDenseClassifier(model_dir=model_dir_2)
        self.assertTrue(new_model.preprocess_pipeline is None)
        new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
                                         preprocess_pipeline_path=preprocess_pipeline_path)
        # Tests
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.model_type, new_model.model_type)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.x_col, new_model.x_col)
        self.assertEqual(model.y_col, new_model.y_col)
        self.assertEqual(model.columns_in, new_model.columns_in)
        self.assertEqual(model.mandatory_columns, new_model.mandatory_columns)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.batch_size, new_model.batch_size)
        self.assertEqual(model.epochs, new_model.epochs)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.patience, new_model.patience)
        self.assertEqual(model.keras_params, new_model.keras_params)
        self.assertEqual(model.custom_objects, new_model.custom_objects)
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_train)], [list(_) for _ in new_model.predict_proba(x_train)])
        remove_dir(new_model.model_dir)
        # We do not remove model_dir to test the errors


        ############################################
        # Errors
        ############################################

        with self.assertRaises(FileNotFoundError):
            new_model = ModelDenseClassifier(model_dir=model_dir_2)
            new_model.reload_from_standalone(configuration_path='toto.json', hdf5_path=hdf5_path,
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelDenseClassifier(model_dir=model_dir_2)
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path='toto.pkl',
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelDenseClassifier(model_dir=model_dir_2)
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
                                             preprocess_pipeline_path='toto.pkl')

        # Clean
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
