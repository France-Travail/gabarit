#!/usr/bin/env python3
# Copyright (C) <2018-2021>  <Agence Data Services, DSI Pôle Emploi>
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
import numpy as np
import pandas as pd

from {{package_name}} import utils
from {{package_name}}.models_training.classifiers.model_xgboost_classifier import ModelXgboostClassifier

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelXgboostClassifierTests(unittest.TestCase):
    '''Main class to test model_xgboost_classifier'''


    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_model_xgboost_classifier_init(self):
        '''Test of the initialization of {{package_name}}.models_training.classifiers.model_xgboost_classifier.ModelXgboostClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        model = ModelXgboostClassifier(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.model_type, 'classifier')
        self.assertTrue(model.model is not None)
        self.assertTrue(os.path.isdir(model_dir))
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelXgboostClassifier(model_dir=model_dir, xgboost_params={'toto': 5})
        # Add auto objective when not given
        self.assertEqual(model.xgboost_params, {'objective': 'binary:logistic', 'toto': 5})
        remove_dir(model_dir)

        #
        model = ModelXgboostClassifier(model_dir=model_dir, early_stopping_rounds=8)
        self.assertEqual(model.early_stopping_rounds, 8)
        remove_dir(model_dir)

        #
        model = ModelXgboostClassifier(model_dir=model_dir, validation_split=0.3)
        self.assertEqual(model.validation_split, 0.3)
        remove_dir(model_dir)


    def test02_model_xgboost_classifier_fit(self):
        '''Test of the method fit of {{package_name}}.models_training.classifiers.model_xgboost_classifier.ModelXgboostClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_valid_mono_missing = y_train_mono_3.copy()
        y_valid_mono_missing[y_valid_mono_missing == 2] = 0
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Classification - Mono-label - Mono-Class
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        model.fit(x_train, y_train_mono_2, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertEqual(model.list_classes, [0, 1])
        self.assertEqual(model.dict_classes, {0: 0, 1: 1})
        remove_dir(model_dir)
        # With valid
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        model.fit(x_train, y_train_mono_2, x_valid=x_train, y_valid=y_train_mono_2, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertEqual(model.list_classes, [0, 1])
        self.assertEqual(model.dict_classes, {0: 0, 1: 1})
        remove_dir(model_dir)
        # With shuffle to False
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        model.fit(x_train, y_train_mono_2, x_valid=x_train, y_valid=y_train_mono_2, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertEqual(model.list_classes, [0, 1])
        self.assertEqual(model.dict_classes, {0: 0, 1: 1})
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        model.fit(x_train, y_train_mono_3)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertEqual(model.list_classes, [0, 1, 2])
        self.assertEqual(model.dict_classes, {0: 0, 1: 1, 2: 2})
        remove_dir(model_dir)
        # With valid
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        model.fit(x_train, y_train_mono_3, x_valid=x_train, y_valid=y_train_mono_3, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertEqual(model.list_classes, [0, 1, 2])
        self.assertEqual(model.dict_classes, {0: 0, 1: 1, 2: 2})
        remove_dir(model_dir)
        # With shuffle to False
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        model.fit(x_train, y_train_mono_3, x_valid=x_train, y_valid=y_train_mono_3, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertEqual(model.list_classes, [0, 1, 2])
        self.assertEqual(model.dict_classes, {0: 0, 1: 1, 2: 2})
        remove_dir(model_dir)
        # Missing targets in y_valid
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        model.fit(x_train, y_train_mono_3, x_valid=x_train, y_valid=y_valid_mono_missing, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_mono)
        self.assertEqual(model.list_classes, [0, 1, 2])
        self.assertEqual(model.dict_classes, {0: 0, 1: 1, 2: 2})
        remove_dir(model_dir)

        # Classification - Multi-labels
        # We also check without x_col & y_col
        model = ModelXgboostClassifier(model_dir=model_dir, multi_label=True, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, None)
        self.assertEqual(model.y_col, None)
        model.fit(x_train, y_train_multi)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.list_classes, y_col_multi)
        self.assertEqual(model.dict_classes, {0: 'y1', 1: 'y2', 2: 'y3'})
        remove_dir(model_dir)
        # With valid
        model = ModelXgboostClassifier(model_dir=model_dir, multi_label=True, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, None)
        self.assertEqual(model.y_col, None)
        model.fit(x_train, y_train_multi, x_valid=x_train, y_valid=y_train_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.list_classes, y_col_multi)
        self.assertEqual(model.dict_classes, {0: 'y1', 1: 'y2', 2: 'y3'})
        remove_dir(model_dir)
        # With shuffle to False
        model = ModelXgboostClassifier(model_dir=model_dir, multi_label=True, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.x_col, None)
        self.assertEqual(model.y_col, None)
        model.fit(x_train, y_train_multi, x_valid=x_train, y_valid=y_train_multi, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.list_classes, y_col_multi)
        self.assertEqual(model.dict_classes, {0: 'y1', 1: 'y2', 2: 'y3'})
        remove_dir(model_dir)

        #
        ############
        # Test continue training
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono_2)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        # Second fit
        with self.assertRaises(RuntimeError):
            model.fit(x_train[:50], y_train_mono_2[:50])
        self.assertEqual(model_dir, model.model_dir)
        remove_dir(model_dir)


    def test03_model_xgboost_classifier_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.classifiers.model_xgboost_classifier.ModelXgboostClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Classification - Mono-label - Mono-Class
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        model.fit(x_train, y_train_mono_2)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 2)) # 2 classes
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        model.fit(x_train, y_train_mono_3)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 3)) # 3 classes
        remove_dir(model_dir)

        # Classification - Multi-labels
        # We also check without x_col & y_col
        model = ModelXgboostClassifier(model_dir=model_dir, multi_label=True, xgboost_params={'n_estimators': 5})
        model.fit(x_train, y_train_multi)
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi)))
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), len(y_col_multi)))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
            model.predict(pd.Series([-2, 3]))
        remove_dir(model_dir)


    def test04_model_xgboost_classifier_predict_proba(self):
        '''Test of the method predict_proba of {{package_name}}.models_training.classifiers.model_xgboost_classifier.ModelXgboostClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Classification - Mono-label - Mono-Class
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        model.fit(x_train, y_train_mono_2)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 2)) # 2 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Classification - Mono-label - Multi-Classes
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        model.fit(x_train, y_train_mono_3)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3)) # 3 classes
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Classification - Multi-labels
        # We also check without x_col & y_col
        model = ModelXgboostClassifier(model_dir=model_dir, multi_label=True, xgboost_params={'n_estimators': 5})
        model.fit(x_train, y_train_multi)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(y_col_multi))) # 3 labels
        self.assertTrue(isinstance(preds[0][0], (np.floating, float)))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
            model.predict_proba('test')
        remove_dir(model_dir)


    def test05_model_xgboost_classifier_save(self):
        '''Test of the method save of {{package_name}}.models_training.classifiers.model_xgboost_classifier.ModelXgboostClassifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']


        # Nominal case - without fit
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f'xbgoost_standalone.model')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'preprocess_pipeline.pkl')))
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
        self.assertEqual(configs['model_type'], 'classifier')
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('columns_in' in configs.keys())
        self.assertTrue('mandatory_columns' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'xgboost')
        self.assertTrue('xgboost_params' in configs.keys())
        self.assertTrue('early_stopping_rounds' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        remove_dir(model_dir)

        # Nominal case - with fit
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        model.fit(x_train, y_train_mono_2)
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f'xbgoost_standalone.model')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'preprocess_pipeline.pkl')))
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
        self.assertEqual(configs['model_type'], 'classifier')
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('columns_in' in configs.keys())
        self.assertTrue('mandatory_columns' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'xgboost')
        self.assertTrue('xgboost_params' in configs.keys())
        self.assertTrue('early_stopping_rounds' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        remove_dir(model_dir)

        # WITH level_save = 'LOW' & fit
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5}, level_save='LOW')
        model.fit(x_train, y_train_mono_2)
        # Save
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f'xbgoost_standalone.model')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'preprocess_pipeline.pkl')))
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
        self.assertEqual(configs['model_type'], 'classifier')
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('columns_in' in configs.keys())
        self.assertTrue('mandatory_columns' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'xgboost')
        self.assertTrue('xgboost_params' in configs.keys())
        self.assertTrue('early_stopping_rounds' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        remove_dir(model_dir)


    def test06_model_xgboost_classifier_reload_from_standalone(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_xgboost_classifier.ModelXgboostClassifier.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
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
        model = ModelXgboostClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir, xgboost_params={'n_estimators': 5})
        model.fit(x_train, y_train_mono_2)
        xgb_model = model.model
        model.save()
        # Reload
        xgboost_path = os.path.join(model.model_dir, f"xbgoost_standalone.model")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelXgboostClassifier()
        self.assertTrue(new_model.preprocess_pipeline is None)
        new_model.reload_from_standalone(configuration_path=conf_path, xgboost_path=xgboost_path,
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
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.xgboost_params, new_model.xgboost_params)
        self.assertEqual(model.early_stopping_rounds, new_model.early_stopping_rounds)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.model.get_params(), xgb_model.get_params())
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_train)], [list(_) for _ in new_model.predict_proba(x_train)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # Classification - Multi-labels
        ############################################

        # Create model
        model = ModelXgboostClassifier(model_dir=model_dir, multi_label=True, xgboost_params={'n_estimators': 5})
        model.fit(x_train, y_train_multi)
        xgb_model = model.model
        model.save()
        # Reload
        xgboost_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        new_model = ModelXgboostClassifier()
        self.assertTrue(new_model.preprocess_pipeline is None)
        new_model.reload_from_standalone(configuration_path=conf_path, xgboost_path=xgboost_path,
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
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.xgboost_params, new_model.xgboost_params)
        self.assertEqual(model.early_stopping_rounds, new_model.early_stopping_rounds)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.model.get_params(), xgb_model.get_params())
        self.assertTrue(new_model.preprocess_pipeline is not None)
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_train)], [list(_) for _ in new_model.predict_proba(x_train)])
        remove_dir(new_model.model_dir)
        # We do not remove model_dir to test the errors

        ############################################
        # Errors
        ############################################

        with self.assertRaises(FileNotFoundError):
            new_model = ModelXgboostClassifier()
            new_model.reload_from_standalone(configuration_path='toto.json', xgboost_path=xgboost_path,
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelXgboostClassifier()
            new_model.reload_from_standalone(configuration_path=conf_path, xgboost_path='toto.pkl',
                                             preprocess_pipeline_path=preprocess_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelXgboostClassifier()
            new_model.reload_from_standalone(configuration_path=conf_path, xgboost_path=xgboost_path,
                                             preprocess_pipeline_path='toto.pkl')

        # Clean
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
