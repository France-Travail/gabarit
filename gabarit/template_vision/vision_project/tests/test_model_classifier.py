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
    def fit(self, df_train: pd.DataFrame, df_valid: pd.DataFrame = None, with_shuffle: bool = True, **kwargs):
        '''Simplified version of fit'''
        self.list_classes = sorted(list(df_train['file_class'].unique()))
        self.dict_classes = {i: col for i, col in enumerate(self.list_classes)}
        self.trained = True
        self.nb_fit += 1
    def predict(self, df_test, return_proba: bool = False, batch_size: int = None):
        '''Simplified version of predict'''
        predicted_proba = []
        for i, row in df_test.iterrows():
            tmp_probas = self.predict_on_name(row['file_path'])
            predicted_proba.append(tmp_probas)
        predicted_proba = np.array(predicted_proba)
        if return_proba:
            return predicted_proba
        else:
            return self.get_classes_from_proba(predicted_proba)
    def predict_proba(self, df_test, batch_size: int = None):
        '''Simplified version of predict_proba'''
        return self.predict(df_test, return_proba=True, batch_size=batch_size)
    def predict_on_name(self, name: str):
        if 'toto' in name:
            return [0.8, 0.1, 0.1]
        elif 'titi' in name:
            return [0.7, 0.1, 0.2]
        elif 'tata' in name:
            return [0.2, 0.6, 0.2]
        else:
            return [0.0, 0.2, 0.8]


class ModelClassifierMixinTests(unittest.TestCase):
    '''Main class to test model_classifier'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_classifier_init(self):
        '''Test of the initialization of {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Nominal case
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.list_classes, None)
        self.assertEqual(model.dict_classes, None)
        self.assertEqual(model.model_type, 'classifier')
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
        df_train = pd.DataFrame({
            'file_class': ['cl1', 'cl1', 'cl2', 'cl3', 'cl1'],
            'file_path': ['toto.png', 'titi.png', 'tata.png', 'tutu.png', 'toto.png'],
        })

        # Nominal case
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
        model.fit(df_train)
        preds, probas = model.predict_with_proba(df_train)
        self.assertEqual(preds.shape, (df_train.shape[0],))
        self.assertEqual(probas.shape, (df_train.shape[0], 3))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
            model.predict_with_proba(df_train)
        remove_dir(model_dir)

    def test03_model_classifier_get_predict_position(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.get_predict_position'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Set vars
        df_train = pd.DataFrame({
            'file_class': ['cl1', 'cl1', 'cl2', 'cl3', 'cl1'],
            'file_path': ['toto.png', 'titi.png', 'tata.png', 'tutu.png', 'toto.png'],
        })
        y_true = ['cl1', 'cl1', 'cl1', 'cl3', 'cl2']  # Add some errors

        # Nominal case
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
        model.fit(df_train)
        predict_positions = model.get_predict_position(df_train, y_true)
        self.assertEqual(predict_positions.shape, (df_train.shape[0],))
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
            model.get_predict_position(df_train, y_true)
        remove_dir(model_dir)

    def test04_model_classifier_get_classes_from_proba(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.get_classes_from_proba'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Set vars
        df_train = pd.DataFrame({
            'file_class': ['cl1', 'cl1', 'cl2', 'cl3', 'cl1'],
            'file_path': ['toto.png', 'titi.png', 'tata.png', 'tutu.png', 'toto.png'],
        })

        # Nominal case
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
        model.fit(df_train)
        index_cl1 = model.list_classes.index('cl1')
        index_cl2 = model.list_classes.index('cl2')
        index_cl3 = model.list_classes.index('cl3')
        inv_cl = [0, 0, 0]
        inv_cl[index_cl1] = 'cl1'
        inv_cl[index_cl2] = 'cl2'
        inv_cl[index_cl3] = 'cl3'
        predicted_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.9, 0.0], [0.1, 0.2, 0.7]])
        predicted_classes = model.get_classes_from_proba(predicted_proba)
        self.assertEqual(predicted_classes.shape, (predicted_proba.shape[0], ))
        self.assertEqual(predicted_classes[0], inv_cl[0])
        self.assertEqual(predicted_classes[1], inv_cl[1])
        self.assertEqual(predicted_classes[2], inv_cl[2])
        remove_dir(model_dir)

    def test05_model_classifier_get_top_n_from_proba(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.get_top_n_from_proba'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Nominal case
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
        model.list_classes = ['a', 'b', 'c']
        model.dict_classes = {0: 'a', 1: 'b', 2: 'c'}
        probas = np.array([[0.1, 0.6, 0.3], [0.7, 0.2, 0.1]])
        top_n, top_n_proba = model.get_top_n_from_proba(probas, n=2)
        self.assertEqual([list(_) for _ in top_n], [['b', 'c'], ['a', 'b']])
        self.assertEqual([list(_) for _ in top_n_proba], [[0.6, 0.3], [0.7, 0.2]])
        with self.assertRaises(ValueError):
            model.get_top_n_from_proba(probas, n=5)
        remove_dir(model_dir)

    def test06_model_classifier_inverse_transform(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.inverse_transform'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Nominal case
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
        model.list_classes = ['toto', 'titi', 'tata']
        y1 = np.array(['toto', 'titi', 'tata', 'toto'])
        expected_result1 = ['toto', 'titi', 'tata', 'toto']
        y2 = 'toto'
        expected_result2 = 'toto'
        self.assertEqual(model.inverse_transform(y1), expected_result1)
        self.assertEqual(model.inverse_transform(y2), expected_result2)
        remove_dir(model_dir)

    def test07_model_classifier_get_and_save_metrics(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.get_and_save_metrics'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Nominal case
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
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

        # With the other parameters
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
        model.list_classes = [0, 1]
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        list_files_x = ['toto.png', 'titi.png', 'tata.png', 'tutu.png']
        type_data = 'toto'
        model_logger = ModelLogger(
            tracking_uri="http://toto.titi.tata.test",
            experiment_name="test"
        )
        df_metrics = model.get_and_save_metrics(y_true, y_pred, list_files_x=list_files_x, type_data=type_data, model_logger=model_logger)
        self.assertEqual(df_metrics.shape[0], 3)  # 2 classes + All
        self.assertEqual(df_metrics.loc[2, :]['Label'], 'All')
        self.assertEqual(df_metrics.loc[2, :]['Accuracy'], 0.5)
        plots_path = os.path.join(model.model_dir, 'plots')
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f'predictions_{type_data}.csv')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, f'{type_data}_confusion_matrix_normalized.png')))
        self.assertTrue(os.path.exists(os.path.join(plots_path, f'{type_data}_confusion_matrix.png')))
        df_preds = pd.read_csv(os.path.join(model.model_dir, f'predictions_{type_data}.csv'), sep='{{default_sep}}', encoding='{{default_encoding}}')
        self.assertTrue('y_true' in df_preds.columns)
        self.assertTrue('y_pred' in df_preds.columns)
        self.assertTrue('file_path' in df_preds.columns)
        self.assertTrue('matched' in df_preds.columns)
        remove_dir(model_dir)

    def test08_model_classifier_get_metrics_simple_monolabel(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.get_metrics_simple_monolabel'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Nominal case
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
        model.list_classes = [0, 1]
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        df_metrics = model.get_metrics_simple_monolabel(y_true, y_pred)
        self.assertEqual(df_metrics.shape[0], 3)  # 2 classes + All
        self.assertEqual(df_metrics.loc[2, :]['Label'], 'All')
        self.assertEqual(df_metrics.loc[2, :]['Accuracy'], 0.5)
        remove_dir(model_dir)

    def test09_model_classifier_update_info_from_c_mat(self):
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

    def test10_model_classifier_save(self):
        '''Test of the method {{package_name}}.models_training.classifiers.model_classifier.ModelClassifierMixin.save'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # test save
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name)
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
        self.assertTrue('model_type' in configs.keys())
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], None)
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        remove_dir(model_dir)

        # test save, level_save = 'LOW'
        model = ModelMockClassifier(model_dir=model_dir, model_name=model_name, level_save='LOW')
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        model.save(json_data={'test': 8})
        self.assertFalse(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
