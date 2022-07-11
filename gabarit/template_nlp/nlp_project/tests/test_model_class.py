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
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from {{package_name}} import utils
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.model_pipeline import ModelPipeline
from {{package_name}}.monitoring.model_logger import ModelLogger

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelClassTests(unittest.TestCase):
    '''Main class to test model_class'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_class_init(self):
        '''Test of the initialization of {{package_name}}.models_training.model_class.ModelClass'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        x_col = 'test_x'
        y_col = 'test_y'
        multi_label = True

        # Init., test all parameters
        model = ModelClass(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.multi_label, False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        with self.assertRaises(NotImplementedError):
            model.fit('test', 'test')
        with self.assertRaises(NotImplementedError):
            model.predict('test')
        with self.assertRaises(NotImplementedError):
            model.predict_proba('test')
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        model = ModelClass(model_dir=model_dir, model_name=model_name)
        self.assertEqual(model.model_name, model_name)
        remove_dir(model_dir)

        model = ModelClass(model_dir=model_dir, x_col=x_col)
        self.assertEqual(model.x_col, x_col)
        remove_dir(model_dir)

        model = ModelClass(model_dir=model_dir, y_col=y_col)
        self.assertEqual(model.y_col, y_col)
        remove_dir(model_dir)

        model = ModelClass(model_dir=model_dir, multi_label=multi_label)
        self.assertEqual(model.multi_label, multi_label)
        remove_dir(model_dir)

        model = ModelClass(model_dir=model_dir, level_save='HIGH')
        self.assertEqual(model.level_save, 'HIGH')
        remove_dir(model_dir)

        model = ModelClass(model_dir=model_dir, level_save='MEDIUM')
        self.assertEqual(model.level_save, 'MEDIUM')
        remove_dir(model_dir)

        model = ModelClass(model_dir=model_dir, level_save='LOW')
        self.assertEqual(model.level_save, 'LOW')
        remove_dir(model_dir)

    def test02_model_class_get_classes_from_proba(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass.get_classes_from_proba'''
        # TODO: same as test05 ?

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        x_col = 'test_x'

        # Test mono-label
        multi_label = False
        y_col = 'test_y'
        model = ModelClass(model_dir=model_dir, model_name=model_name, x_col=x_col, y_col=y_col, multi_label=multi_label)
        model.list_classes = ['a', 'b']
        model.dict_classes = {0: 'a', 1: 'b'}
        probas = np.array([[0.2, 0.8], [0.6, 0.4]])
        self.assertEqual(list(model.get_classes_from_proba(probas)), ['b', 'a'])
        remove_dir(model_dir)

        # Test multi-labels
        multi_label = True
        y_col = ['test_y1', 'test_y2']
        model = ModelClass(model_dir=model_dir, model_name=model_name, x_col=x_col, y_col=y_col, multi_label=multi_label)
        model.list_classes = ['a', 'b', 'c']
        model.dict_classes = {0: 'a', 1: 'b', 2: 'c'}
        probas = np.array([[0.2, 0.8, 0.6], [0.6, 0.4, 0.8], [0.6, 0.9, 0.8], [0.3, 0.4, 0.4]])
        self.assertEqual([list(_) for _ in model.get_classes_from_proba(probas)], [[0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0]])
        remove_dir(model_dir)


    def test03_model_class_predict_with_proba(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass.predict_with_proba'''
        # /!\ We must use a sub-class for the tests because the class ModelClass does not implement predict / predict_proba

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 0])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        tfidf = TfidfVectorizer()
        rf = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('tfidf', tfidf), ('rf', rf)])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=False)
        model.fit(x_train, y_train_mono)
        preds, probas = model.predict_with_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train),))
        self.assertEqual(probas.shape, (len(x_train), 2))
        remove_dir(model_dir)

        # Multi-labels
        tfidf = TfidfVectorizer()
        rf = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('tfidf', tfidf), ('rf', OneVsRestClassifier(rf))])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        preds, probas = model.predict_with_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        self.assertEqual(probas.shape, (len(x_train), len(cols)))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=False)
            model.predict_with_proba(x_train)
        remove_dir(model_dir)


    def test04_model_class_get_predict_position(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass.get_predict_position'''
        # /!\ We must use a sub-class for the tests because the class ModelClass does not implement predict / predict_proba

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!", "coucou"])
        y_train_mono = np.array([0, 1, 0, 1, 0, 2])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0, 0], 'test2': [1, 0, 0, 0, 0, 1], 'test3': [0, 0, 0, 1, 0, 1]})
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        tfidf = TfidfVectorizer()
        rf = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('tfidf', tfidf), ('rf', rf)])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=False)
        model.fit(x_train, y_train_mono)
        predict_positions = model.get_predict_position(x_train, y_train_mono)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)

        # Multi-labels
        tfidf = TfidfVectorizer()
        rf = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('tfidf', tfidf), ('rf', OneVsRestClassifier(rf))])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        # Unavailable in multi-labels
        with self.assertRaises(ValueError):
            model.get_predict_position(x_train, y_train_mono)
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=False)
            model.get_predict_position(x_train, y_train_mono)
        remove_dir(model_dir)


    def test05_model_class_get_classes_from_proba(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass.get_classes_from_proba'''
        # /!\ We must use a sub-class for the tests because the class ModelClass does not implement predict / predict_proba

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array(['non', 'oui', 'non', 'oui', 'non'])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0]})
        cols = ['test1', 'test2']

        # Mono-label
        tfidf = TfidfVectorizer()
        rf = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('tfidf', tfidf), ('rf', rf)])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=False)
        model.fit(x_train, y_train_mono)
        index_non = model.list_classes.index('non')
        index_oui = model.list_classes.index('oui')
        predicted_proba = np.array([[0.8, 0.2], [0.1, 0.9]])
        predicted_classes = model.get_classes_from_proba(predicted_proba)
        if index_non == 0:
            self.assertEqual(list(predicted_classes), ['non', 'oui'])
        else:
            self.assertEqual(list(predicted_classes), ['oui', 'non'])
        remove_dir(model_dir)

        # Multi-labels
        tfidf = TfidfVectorizer()
        rf = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('tfidf', tfidf), ('rf', OneVsRestClassifier(rf))])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=True)
        model.fit(x_train, y_train_multi)
        index_col_1 = model.list_classes.index('test1')
        index_col_2 = model.list_classes.index('test2')
        pred_none = [0, 0]
        pred_col_1 = [0, 0]
        pred_col_1[index_col_1] = 1
        pred_col_2 = [0, 0]
        pred_col_2[index_col_2] = 1
        pred_all = [1, 1]
        predicted_proba = np.array([[0.1, 0.2], [0.8, 0.9], [0.1, 0.9], [0.7, 0.4]])
        predicted_classes = model.get_classes_from_proba(predicted_proba)
        if index_col_1 == 0:
            self.assertEqual([list(_) for _ in predicted_classes], [[0, 0], [1, 1], pred_col_2, pred_col_1])
        else:
            self.assertEqual([list(_) for _ in predicted_classes], [[0, 0], [1, 1], pred_col_1, pred_col_2])
        remove_dir(model_dir)

    def test06_model_class_get_top_n_from_proba(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass.get_top_n_from_proba'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        x_col = 'test_x'

        # Test mono-label
        multi_label = False
        y_col = 'test_y'
        model = ModelClass(model_dir=model_dir, model_name=model_name, x_col=x_col, y_col=y_col, multi_label=multi_label)
        model.list_classes = ['a', 'b', 'c']
        model.dict_classes = {0: 'a', 1: 'b', 2: 'c'}
        probas = np.array([[0.1, 0.6, 0.3], [0.7, 0.2, 0.1]])
        top_n, top_n_proba = model.get_top_n_from_proba(probas, n=2)
        self.assertEqual([list(_) for _ in top_n], [['b', 'c'], ['a', 'b']])
        self.assertEqual([list(_) for _ in top_n_proba], [[0.6, 0.3], [0.7, 0.2]])
        with self.assertRaises(ValueError):
            model.get_top_n_from_proba(probas, n=5)
        remove_dir(model_dir)

        # Test multi-labels
        multi_label = True
        y_col = ['test_y1', 'test_y2']
        model = ModelClass(model_dir=model_dir, model_name=model_name, x_col=x_col, y_col=y_col, multi_label=multi_label)
        with self.assertRaises(ValueError):
            model.get_top_n_from_proba(probas)
        remove_dir(model_dir)

    def test07_model_class_inverse_transform(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass.inverse_transform'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        x_col = 'test_x'
        y_col = 'test_y'
        multi_label = True

        # inverse_transform - mono-label
        model = ModelClass(model_dir=model_dir, y_col=y_col, multi_label=False)
        model.list_classes = ['toto', 'titi', 'tata']
        y1 = np.array(['toto', 'titi', 'tata', 'toto'])
        expected_result1 = ['toto', 'titi', 'tata', 'toto']
        y2 = 'toto'
        expected_result2 = 'toto'
        self.assertEqual(model.inverse_transform(y1), expected_result1)
        self.assertEqual(model.inverse_transform(y2), expected_result2)
        remove_dir(model_dir)

        # inverse_transform - multi-labels
        model = ModelClass(model_dir=model_dir, y_col=y_col, multi_label=True)
        model.list_classes = ['test1', 'test2', 'test3']
        y_bad = np.array([[1, 2], [4, 5]])
        with self.assertRaises(ValueError):
            model.inverse_transform(y_bad)
        y1 = np.array([[0, 0, 1], [1, 1, 0], [0, 0, 0]])
        expected_result1 = [('test3',), ('test1', 'test2'), ()]
        y2 = np.array([0, 0, 1])
        expected_result2 = ('test3',)
        self.assertEqual(model.inverse_transform(y1), expected_result1)
        self.assertEqual(model.inverse_transform(y2), expected_result2)
        remove_dir(model_dir)

    def test08_model_class_get_and_save_metrics(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass.get_and_save_metrics'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # get_and_save_metrics - mono-label
        model = ModelClass(model_dir=model_dir, model_name=model_name, multi_label=False)
        model.list_classes = [0, 1]
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        df_metrics = model.get_and_save_metrics(y_true, y_pred)
        self.assertEqual(df_metrics.shape[0], 3)  # 2 classes + All
        self.assertEqual(df_metrics.loc[2, :]['Label'], 'All')
        self.assertEqual(df_metrics.loc[2, :]['Accuracy'], 0.5)
        plots_path = os.path.join(model.model_dir, 'plots')
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'predictions.csv')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'confusion_matrix_normalized.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'confusion_matrix.png')))
        remove_dir(model_dir)

        # get_and_save_metrics - multi-labels
        model = ModelClass(model_dir=model_dir, model_name=model_name, multi_label=True)
        model.list_classes = ['test1', 'test2', 'test3']
        y_true = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
        y_pred = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]])
        df_metrics = model.get_and_save_metrics(y_true, y_pred)
        self.assertEqual(df_metrics.shape[0], 4) # 3 classes + All
        self.assertEqual(df_metrics.loc[3, :]['Label'], 'All')
        self.assertEqual(df_metrics.loc[0, :]['Accuracy'], 1.0)
        plots_path = os.path.join(model.model_dir, 'plots')
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'predictions.csv')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'test1__confusion_matrix_normalized.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'test1__confusion_matrix.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'test2__confusion_matrix_normalized.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'test2__confusion_matrix.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'test3__confusion_matrix_normalized.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, 'test3__confusion_matrix.png')))
        remove_dir(model_dir)

        # get_and_save_metrics, with the other parameters
        model = ModelClass(model_dir=model_dir, model_name=model_name, multi_label=False)
        model.list_classes = [0, 1]
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        x = np.array([8, 5, -1, 12])
        series_to_add = [pd.Series(['a', 'b', 'c', 'd'], name='test')]
        type_data = 'toto'
        model_logger = ModelLogger(
            tracking_uri="http://toto.titi.tata.test",
            experiment_name="test"
        )
        df_metrics = model.get_and_save_metrics(y_true, y_pred, x=x, series_to_add=series_to_add, type_data=type_data, model_logger=model_logger)
        self.assertEqual(df_metrics.shape[0], 3)  # 2 classes + All
        self.assertEqual(df_metrics.loc[2, :]['Label'], 'All')
        self.assertEqual(df_metrics.loc[2, :]['Accuracy'], 0.5)
        plots_path = os.path.join(model.model_dir, 'plots')
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f'predictions_{type_data}.csv')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, f'{type_data}_confusion_matrix_normalized.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, f'{type_data}_confusion_matrix.png')))
        df_preds = pd.read_csv(os.path.join(model.model_dir, f'predictions_{type_data}.csv'), sep='{{default_sep}}', encoding='{{default_encoding}}')
        self.assertTrue('x' in df_preds.columns)
        self.assertTrue('y_true' in df_preds.columns)
        self.assertTrue('y_pred' in df_preds.columns)
        self.assertTrue('matched' in df_preds.columns)
        self.assertTrue('test' in df_preds.columns)
        remove_dir(model_dir)

    def test09_model_class_get_metrics_simple_monolabel(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass.get_metrics_simple_monolabel'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # get_and_save_metrics - mono-label
        model = ModelClass(model_dir=model_dir, model_name=model_name, multi_label=False)
        model.list_classes = [0, 1]
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        df_metrics = model.get_metrics_simple_monolabel(y_true, y_pred)
        self.assertEqual(df_metrics.shape[0], 3)  # 2 classes + All
        self.assertEqual(df_metrics.loc[2, :]['Label'], 'All')
        self.assertEqual(df_metrics.loc[2, :]['Accuracy'], 0.5)
        remove_dir(model_dir)

        # Test multi-labels
        model = ModelClass(model_dir=model_dir, model_name=model_name, multi_label=True)
        with self.assertRaises(ValueError):
            model.get_metrics_simple_monolabel(y_true, y_pred)
        remove_dir(model_dir)

    def test10_model_class_get_metrics_simple_multilabel(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass.get_metrics_simple_multilabel'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # get_and_save_metrics - multi-label
        model = ModelClass(model_dir=model_dir, model_name=model_name, multi_label=True)
        model.list_classes = ['test1', 'test2', 'test3']
        y_true = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
        y_pred = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]])
        df_metrics = model.get_metrics_simple_multilabel(y_true, y_pred)
        self.assertEqual(df_metrics.shape[0], 4) # 3 classes + All
        self.assertEqual(df_metrics.loc[3, :]['Label'], 'All')
        self.assertEqual(df_metrics.loc[0, :]['Accuracy'], 1.0)
        remove_dir(model_dir)

        # Test mono label
        model = ModelClass(model_dir=model_dir, model_name=model_name, multi_label=False)
        with self.assertRaises(ValueError):
            model.get_metrics_simple_multilabel(y_true, y_pred)
        remove_dir(model_dir)

    def test11_model_class_update_info_from_c_mat(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass._update_info_from_c_mat'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        model = ModelClass(model_dir=model_dir)
        c_mat = [[10, 5], [8, 1]]
        expected_result = {'Accuracy': (10 + 1) / (10 + 5 + 8 + 1),
                           'Condition negative': 10 + 5,
                           'Condition positive': 8 + 1,
                           'F1-Score': 2 / ((1/(1 / (5 + 1))) + (1/(1 / (8 + 1)))),
                           'False negative': 8,
                           'False positive': 5,
                           'Falses': 5 + 8,
                           'Label': 'toto',
                           'Precision': 1 / (5 + 1),
                           'Predicted negative': 10 + 8,
                           'Predicted positive': 5 + 1,
                           'Recall': 1 / (8 + 1),
                           'True negative': 10,
                           'True positive': 1,
                           'Trues': 10 + 1}

        # Nominal case
        info_dict = model._update_info_from_c_mat(c_mat, label='toto', log_info=False)
        self.assertEqual(info_dict, expected_result)

    def test12_model_class_save(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass.save'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        x_col = 'test_x'
        y_col = 'test_y'
        multi_label = True

        # test save
        model = ModelClass(model_dir=model_dir, model_name=model_name)
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertTrue('mainteners' in configs.keys())
        self.assertTrue('date' in configs.keys())
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
        self.assertEqual(configs['librairie'], None)
        remove_dir(model_dir)

        # test save, level_save = 'LOW'
        model = ModelClass(model_dir=model_dir, model_name=model_name, level_save='LOW')
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        model.save(json_data={'test': 8})
        self.assertFalse(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        remove_dir(model_dir)

    def test13_model_class_save_upload_properties(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass._save_upload_properties'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Nominal case
        json_dict = {
            "mainteners": "c'est nous",
            "date": "01/01/1970 - 00:00:00",
            "bruit": "toto",
            "package_version": "0.0.8",
            "model_name": "hello_model",
            "list_classes": ["c1", "c2", np.int32(9), "c3", 3],
            "autre_bruit": "titi",
            "librairie": "ma_lib",
            "fit_time": "7895s",
        }
        model = ModelClass(model_dir=model_dir, model_name=model_name)
        model._save_upload_properties(json_dict)
        # Checks the presence of a file model_upload_instructions.md
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_upload_instructions.md')))
        with open(os.path.join(model.model_dir, 'model_upload_instructions.md'), 'r', encoding='{{default_encoding}}') as f:
            instructions = f.read()
        self.assertTrue(os.path.abspath(model.model_dir) in instructions)
        # Checks the presence of a file proprietes.json
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'proprietes.json')))
        with open(os.path.join(model.model_dir, 'proprietes.json'), 'r', encoding='{{default_encoding}}') as f:
            proprietes = json.load(f)
        self.assertTrue('mainteners' in proprietes.keys())
        self.assertEqual(proprietes['mainteners'], "c'est nous")
        self.assertTrue('date' in proprietes.keys())
        self.assertEqual(proprietes['date'], "01/01/1970 - 00:00:00")
        self.assertTrue('package_version' in proprietes.keys())
        self.assertEqual(proprietes['package_version'], "0.0.8")
        self.assertTrue('model_name' in proprietes.keys())
        self.assertEqual(proprietes['model_name'], "hello_model")
        self.assertTrue('list_classes' in proprietes.keys())
        self.assertEqual(proprietes['list_classes'], ["c1", "c2", 9, "c3", 3])
        self.assertTrue('librairie' in proprietes.keys())
        self.assertEqual(proprietes['librairie'], "ma_lib")
        self.assertTrue('fit_time' in proprietes.keys())
        self.assertEqual(proprietes['fit_time'], "7895s")
        self.assertFalse('bruit' in proprietes.keys())
        self.assertFalse('autre_bruit' in proprietes.keys())
        remove_dir(model_dir)

        # Same, mais via la fonction save
        json_dict = {
            "mainteners": "c'est nous",
            "date": "01/01/1970 - 00:00:00",
            "bruit": "toto",
            "package_version": "0.0.8",
            "model_name": "hello_model",
            "list_classes": ["c1", "c2", "c8", "c3"],
            "autre_bruit": "titi",
            "librairie": "ma_lib",
            "fit_time": "7895s",
        }
        model = ModelClass(model_dir=model_dir, model_name=model_name)
        model.save(json_dict)
        # Checks the presence of a file model_upload_instructions.md
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_upload_instructions.md')))
        with open(os.path.join(model.model_dir, 'model_upload_instructions.md'), 'r', encoding='{{default_encoding}}') as f:
            instructions = f.read()
        self.assertTrue(os.path.abspath(model.model_dir) in instructions)
        # Checks the presence of a file proprietes.json
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'proprietes.json')))
        with open(os.path.join(model.model_dir, 'proprietes.json'), 'r', encoding='{{default_encoding}}') as f:
            proprietes = json.load(f)
        self.assertTrue('mainteners' in proprietes.keys())
        self.assertEqual(proprietes['mainteners'], "c'est nous")
        self.assertTrue('date' in proprietes.keys())
        self.assertEqual(proprietes['date'], "01/01/1970 - 00:00:00")
        self.assertTrue('package_version' in proprietes.keys())
        self.assertEqual(proprietes['package_version'], "0.0.8")
        self.assertTrue('model_name' in proprietes.keys())
        self.assertEqual(proprietes['model_name'], "hello_model")
        self.assertTrue('list_classes' in proprietes.keys())
        self.assertEqual(proprietes['list_classes'], ["c1", "c2", "c8", "c3"])
        self.assertTrue('librairie' in proprietes.keys())
        self.assertEqual(proprietes['librairie'], "ma_lib")
        self.assertTrue('fit_time' in proprietes.keys())
        self.assertEqual(proprietes['fit_time'], "7895s")
        self.assertFalse('bruit' in proprietes.keys())
        self.assertFalse('autre_bruit' in proprietes.keys())
        remove_dir(model_dir)

        # Empty case
        json_dict = {}
        model = ModelClass(model_dir=model_dir, model_name=model_name)
        model._save_upload_properties(json_dict)
        # Checks the presence of a file model_upload_instructions.md
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_upload_instructions.md')))
        with open(os.path.join(model.model_dir, 'model_upload_instructions.md'), 'r', encoding='{{default_encoding}}') as f:
            instructions = f.read()
        self.assertTrue(os.path.abspath(model.model_dir) in instructions)
        # Checks the presence of a file proprietes.json
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'proprietes.json')))
        with open(os.path.join(model.model_dir, 'proprietes.json'), 'r', encoding='{{default_encoding}}') as f:
            proprietes = json.load(f)
        self.assertFalse('mainteners' in proprietes.keys())
        self.assertFalse('date' in proprietes.keys())
        self.assertFalse('package_version' in proprietes.keys())
        self.assertFalse('model_name' in proprietes.keys())
        self.assertFalse('list_classes' in proprietes.keys())
        self.assertFalse('librairie' in proprietes.keys())
        self.assertFalse('fit_time' in proprietes.keys())
        self.assertFalse('bruit' in proprietes.keys())
        self.assertFalse('autre_bruit' in proprietes.keys())
        remove_dir(model_dir)

    def test14_model_class_get_model_dir(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass._get_model_dir'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Nominal case
        model = ModelClass(model_dir=model_dir, model_name=model_name)
        expected_dir = os.path.join(utils.get_models_path(), model_name, f"{model_name}_")
        res_dir = model._get_model_dir()
        self.assertTrue(res_dir.startswith(expected_dir))
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
