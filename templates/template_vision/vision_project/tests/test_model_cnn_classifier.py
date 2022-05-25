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
import keras
import shutil
import tensorflow
import numpy as np
import pandas as pd

from {{package_name}} import utils
from {{package_name}}.models_training import utils_deep_keras
from {{package_name}}.models_training.classifiers.model_cnn_classifier import ModelCnnClassifier

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)
    tensorflow.keras.backend.clear_session()

class ModelCnnClassifierTests(unittest.TestCase):
    '''Main class to test model_cnn_classifier'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_cnn_classifier_init(self):
        '''Test of {{package_name}}.models_training.classifiers.model_cnn_classifier.ModelCnnClassifier.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        model = ModelCnnClassifier(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.model_type, 'classifier')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model, None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=8)
        self.assertEqual(model.batch_size, 8)
        remove_dir(model_dir)

        #
        model = ModelCnnClassifier(model_dir=model_dir, epochs=42)
        self.assertEqual(model.epochs, 42)
        remove_dir(model_dir)

        #
        model = ModelCnnClassifier(model_dir=model_dir, validation_split=0.3)
        self.assertEqual(model.validation_split, 0.3)
        remove_dir(model_dir)

        #
        model = ModelCnnClassifier(model_dir=model_dir, patience=65)
        self.assertEqual(model.patience, 65)
        remove_dir(model_dir)

        #
        model = ModelCnnClassifier(model_dir=model_dir, width=22)
        self.assertEqual(model.width, 22)
        remove_dir(model_dir)

        #
        model = ModelCnnClassifier(model_dir=model_dir, height=56)
        self.assertEqual(model.height, 56)
        remove_dir(model_dir)

        #
        model = ModelCnnClassifier(model_dir=model_dir, depth=4)
        self.assertEqual(model.depth, 4)
        remove_dir(model_dir)

        #
        model = ModelCnnClassifier(model_dir=model_dir, color_mode='rgba')
        self.assertEqual(model.color_mode, 'rgba')
        remove_dir(model_dir)

        #
        model = ModelCnnClassifier(model_dir=model_dir, in_memory=True)
        self.assertEqual(model.in_memory, True)
        remove_dir(model_dir)

        # keras_params must accept anything !
        model = ModelCnnClassifier(model_dir=model_dir, keras_params={'toto': 5})
        self.assertEqual(model.keras_params, {'toto': 5})
        remove_dir(model_dir)

    # We do not test the fit method. It is already done in test_model_keras.py

    def test02_model_cnn_classifier_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.classifiers.model_cnn_classifier.ModelCnnClassifier'''

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

    def test03_model_cnn_classifier_predict_proba(self):
        '''Test of the method predict_proba of {{package_name}}.models_training.classifiers.model_cnn_classifier.ModelCnnClassifier'''

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

    def test04_model_cnn_classifier_get_predict_position(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_cnn_classifier.ModelCnnClassifier.get_predict_position'''

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

        # Classification - Mono-label - Mono-Class
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
        model.fit(df_train_mono)
        predict_positions = model.get_predict_position(df_train_mono, y_true_mono)
        self.assertEqual(predict_positions.shape, (df_train_mono.shape[0],))
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
        model.fit(df_train_multi)
        predict_positions = model.get_predict_position(df_train_multi, y_true_multi)
        self.assertEqual(predict_positions.shape, (df_train_multi.shape[0],))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
            model.get_predict_position(df_train_mono, y_true_mono)
        remove_dir(model_dir)

    def test05_model_cnn_classifier_get_model(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_cnn_classifier.ModelCnnClassifier._get_model'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelCnnClassifier(model_dir=model_dir, epochs=2, batch_size=2)

        # Nominal case
        model.list_classes = ['a', 'b']  # We force the creation of a list of classes
        model_res = model._get_model()
        self.assertTrue(isinstance(model_res, keras.Model))

        # Clean
        remove_dir(model_dir)

    def test06_model_cnn_classifier_save(self):
        '''Test of the method save of {{package_name}}.models_training.classifiers.model_cnn_classifier.ModelCnnClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelCnnClassifier(model_dir=model_dir)
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
        remove_dir(model_dir)

        # Use custom_objects containing a "partial" function
        model = ModelCnnClassifier(model_dir=model_dir)
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
        remove_dir(model_dir)

    def test07_model_cnn_classifier_reload_model(self):
        '''Test of the method reload_model of {{package_name}}.models_training.classifiers.model_cnn_classifier.ModelCnnClassifier'''

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

    def test08_model_cnn_classifier_reload_from_standalone(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_cnn_classifier.ModelCnnClassifier.reload_from_standalone'''

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
        # Classification - Mono class
        ############################################

        # Create model
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
        fit_params = model.fit(df_train_mono)
        model.save()
        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        preprocess_input_path = os.path.join(model.model_dir, "preprocess_input.pkl")
        new_model = ModelCnnClassifier()
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
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(df_train_mono)], [list(_) for _ in new_model.predict_proba(df_train_mono)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # Classification - Multi class
        ############################################

        # Create model
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
        fit_params = model.fit(df_train_multi)
        model.save()
        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        preprocess_input_path = os.path.join(model.model_dir, "preprocess_input.pkl")
        new_model = ModelCnnClassifier()
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
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(df_train_multi)], [list(_) for _ in new_model.predict_proba(df_train_multi)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)


        ############################################
        # Errors
        ############################################

        with self.assertRaises(FileNotFoundError):
            new_model = ModelCnnClassifier()
            new_model.reload_from_standalone(configuration_path='toto.json', hdf5_path=hdf5_path,
                                             preprocess_input_path=preprocess_input_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelCnnClassifier()
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path='toto.pkl',
                                             preprocess_input_path=preprocess_input_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelCnnClassifier()
            new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
                                             preprocess_input_path='toto.pkl')

        # Clean
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
