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

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.monitoring.model_logger import ModelLogger
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.classifiers.model_classifier import ModelClassifierMixin

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelMockClassifier(ModelClassifierMixin, ModelClass):
    '''We need a mock implementation of the Mixin class'''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pipeline = Pipeline([('rf', RandomForestClassifier())])
    def fit(self, x_train, y_train, **kwargs):
        '''Simplified version of fit'''
        x_train, y_train = self._check_input_format(x_train, y_train, fit_function=True)
        self.pipeline.fit(x_train, y_train)
        if not self.multi_label:
            self.list_classes = list(self.pipeline.classes_)
        else:
            self.list_classes = list(y_train.columns)
        self.dict_classes = {i: col for i, col in enumerate(self.list_classes)}
        self.trained = True
        self.nb_fit += 1
    def predict(self, x_test: pd.DataFrame, return_proba: bool = False, **kwargs):
        '''Simplified version of predict'''
        x_test, _ = self._check_input_format(x_test)
        if not return_proba:
            return np.array(self.pipeline.predict(x_test))
        else:
            return self.predict_proba(x_test)
    def predict_proba(self, x_test: pd.DataFrame, **kwargs):
        '''Simplified version of predict_proba'''
        x_test, _ = self._check_input_format(x_test)
        probas = np.array(self.pipeline.predict_proba(x_test))
        if not np.isnan(probas).any():
            probas = np.nan_to_num(probas, nan=1/len(self.list_classes))
        if len(probas.shape) > 2:
            probas = np.swapaxes(probas[:, :, 1], 0, 1)
        return probas


class ModelClassifierMixinTests(unittest.TestCase):
    '''Main class to test model_classifier'''


    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_model_classifier_init(self):
        '''Test of initialization of {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        x_col = ['test_x1', 'test_x2']
        y_col = 'test_y'

        # Nominal case
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.list_classes, None)
        self.assertEqual(model.dict_classes, None)
        remove_dir(model_dir)

        # Test level_save
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, level_save='HIGH')
        self.assertEqual(model.level_save, 'HIGH')
        remove_dir(model_dir)
        #
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, level_save='MEDIUM')
        self.assertEqual(model.level_save, 'MEDIUM')
        remove_dir(model_dir)
        #
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, level_save='LOW')
        self.assertEqual(model.level_save, 'LOW')
        remove_dir(model_dir)

        # Test multi-labels
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=False)
        self.assertEqual(model.multi_label, False)
        remove_dir(model_dir)
        #
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
        self.assertEqual(model.multi_label, True)
        remove_dir(model_dir)

        # Manage errors
        with self.assertRaises(ValueError):
            ModelMockClassifier(model_dir=model_dir, model_name=model_name, level_save='toto')
        remove_dir(model_dir)


    def test02_model_classifier_predict_with_proba(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.predict_with_proba'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 3, 12, 2] * 10})
        y_train_mono = pd.Series([0, 0, 0, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1] * 10})
        cols = list(y_train_multi.columns)

        # Mono-label
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=False)
        model.fit(x_train, y_train_mono)
        preds, probas = model.predict_with_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train),))
        self.assertEqual(probas.shape, (len(x_train), 2))
        remove_dir(model_dir)

        # Multi-labels
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
        model.fit(x_train, y_train_multi)
        preds, probas = model.predict_with_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        self.assertEqual(probas.shape, (len(x_train), len(cols)))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=False)
            model.predict_with_proba(x_train)
        remove_dir(model_dir)


    def test03_model_classifier_get_predict_position(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.get_predict_position'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 3, 12, 2] * 10})
        y_train_mono = pd.Series([0, 0, 0, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1] * 10})
        cols = list(y_train_multi.columns)

        # Mono-label
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=False)
        model.fit(x_train, y_train_mono)
        predict_positions = model.get_predict_position(x_train, y_train_mono)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)

        # Multi-labels
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
        model.fit(x_train, y_train_multi)
        # Unavailable in multi-labels
        with self.assertRaises(ValueError):
            model.get_predict_position(x_train, y_train_mono)
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=False)
            model.get_predict_position(x_train, y_train_mono)
        remove_dir(model_dir)


    def test04_model_classifier_get_classes_from_proba(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.get_classes_from_proba'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 3, 12, 2] * 10})
        y_train_mono = pd.Series(['non', 'non', 'non', 'oui', 'oui', 'oui'] * 10)
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 1, 1] * 10, 'test2': [1, 0, 0, 1, 1, 1] * 10})
        cols = list(y_train_multi.columns)

        # Mono-label
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=False)
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
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
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


    def test05_model_classifier_get_top_n_from_proba(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.get_top_n_from_proba'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Test mono-label
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=False)
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
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
        with self.assertRaises(ValueError):
            model.get_top_n_from_proba(probas)
        remove_dir(model_dir)


    def test06_model_classifier_inverse_transform(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.inverse_transform'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # inverse_transform - mono-label
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=False)
        model.list_classes = ['toto', 'titi', 'tata']
        y1 = np.array(['toto', 'titi', 'tata', 'toto'])
        expected_result1 = ['toto', 'titi', 'tata', 'toto']
        y2 = 'toto'
        expected_result2 = 'toto'
        self.assertEqual(model.inverse_transform(y1), expected_result1)
        self.assertEqual(model.inverse_transform(y2), expected_result2)
        remove_dir(model_dir)

        # inverse_transform - multi-labels
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
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


    def test07_model_classifier_get_and_save_metrics(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.get_and_save_metrics'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # get_and_save_metrics - mono-label
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=False)
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
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
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
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=False)
        model.list_classes = [0, 1]
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        df_x = pd.DataFrame({'col_1': [-5, -1, 0, 2], 'col_2': [2, -1, -8, 3]})
        series_to_add = [pd.Series(['a', 'b', 'c', 'd'], name='test')]
        type_data = 'toto'
        model_logger = ModelLogger(
            tracking_uri="http://toto.titi.tata.test",
            experiment_name="test"
        )
        df_metrics = model.get_and_save_metrics(y_true, y_pred, df_x=df_x, series_to_add=series_to_add, type_data=type_data, model_logger=model_logger)
        self.assertEqual(df_metrics.shape[0], 3)  # 2 classes + All
        self.assertEqual(df_metrics.loc[2, :]['Label'], 'All')
        self.assertEqual(df_metrics.loc[2, :]['Accuracy'], 0.5)
        plots_path = os.path.join(model.model_dir, 'plots')
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f'predictions_{type_data}.csv')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, f'{type_data}_confusion_matrix_normalized.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, f'{type_data}_confusion_matrix.png')))
        df_preds = pd.read_csv(os.path.join(model.model_dir, f'predictions_{type_data}.csv'), sep='{{default_sep}}', encoding='{{default_encoding}}')
        self.assertTrue('col_1' in df_preds.columns)
        self.assertTrue('col_2' in df_preds.columns)
        self.assertTrue('y_true' in df_preds.columns)
        self.assertTrue('y_pred' in df_preds.columns)
        self.assertTrue('matched' in df_preds.columns)
        self.assertTrue('test' in df_preds.columns)
        remove_dir(model_dir)


    def test08_model_classifier_get_metrics_simple_monolabel(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.get_metrics_simple_monolabel'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # get_and_save_metrics - mono-label
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=False)
        model.list_classes = [0, 1]
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        df_metrics = model.get_metrics_simple_monolabel(y_true, y_pred)
        self.assertEqual(df_metrics.shape[0], 3)  # 2 classes + All
        self.assertEqual(df_metrics.loc[2, :]['Label'], 'All')
        self.assertEqual(df_metrics.loc[2, :]['Accuracy'], 0.5)
        remove_dir(model_dir)

        # Test multi-labels
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
        with self.assertRaises(ValueError):
            model.get_metrics_simple_monolabel(y_true, y_pred)
        remove_dir(model_dir)


    def test09_model_classifier_get_metrics_simple_multilabel(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.get_metrics_simple_multilabel'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # get_and_save_metrics - mono-label
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
        model.list_classes = ['test1', 'test2', 'test3']
        y_true = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
        y_pred = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]])
        df_metrics = model.get_metrics_simple_multilabel(y_true, y_pred)
        self.assertEqual(df_metrics.shape[0], 4) # 3 classes + All
        self.assertEqual(df_metrics.loc[3, :]['Label'], 'All')
        self.assertEqual(df_metrics.loc[0, :]['Accuracy'], 1.0)
        remove_dir(model_dir)

        # Test mono label
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, multi_label=False)
        with self.assertRaises(ValueError):
            model.get_metrics_simple_multilabel(y_true, y_pred)
        remove_dir(model_dir)


    def test10_model_classifier_update_info_from_c_mat(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin._update_info_from_c_mat'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
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


    def test11_model_classifier_save(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.save'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        preprocess_pipeline = preprocess.get_pipeline("no_preprocess") # Warning, needs to be fitted
        preprocess_pipeline.fit(pd.DataFrame({'test_x1': [1, 2, 3], 'test_x2': [4, 5, 6]}))

        # test save
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline)
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        preprocess_pipeline_path = os.path.join(model.model_dir, 'preprocess_pipeline.pkl')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        self.assertTrue(os.path.exists(preprocess_pipeline_path))
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
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
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('columns_in' in configs.keys())
        self.assertTrue('mandatory_columns' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], None)
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        remove_dir(model_dir)

        # test save, level_save = 'LOW'
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline, level_save='LOW')
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        preprocess_pipeline_path = os.path.join(model.model_dir, 'preprocess_pipeline.pkl')
        model.save(json_data={'test': 8})
        self.assertFalse(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        self.assertFalse(os.path.exists(preprocess_pipeline_path))
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
