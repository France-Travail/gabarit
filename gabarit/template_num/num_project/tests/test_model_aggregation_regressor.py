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

# Utils libs
import os
import json
import shutil
import numpy as np
import pandas as pd

from {{package_name}} import utils
from {{package_name}}.models_training.regressors.model_regressor import ModelRegressorMixin 
from {{package_name}}.models_training.regressors.models_sklearn.model_gbt_regressor import ModelGBTRegressor
from {{package_name}}.models_training.regressors.models_sklearn.model_sgd_regressor import ModelSGDRegressor
from {{package_name}}.models_training.classifiers.models_sklearn.model_sgd_classifier import ModelSGDClassifier
from {{package_name}}.models_training.regressors.model_aggregation_regressor import ModelAggregationRegressor, median_predict, mean_predict

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)

models_path = utils.get_models_path()

def remove_dir_model(model, model_dir):
    for sub_model in model.sub_models:
        remove_dir(os.path.join(models_path, os.path.split(sub_model['model'].model_dir)[-1]))
    remove_dir(model_dir)

class MockModel(ModelRegressorMixin):

    def __init__(self, dict_predictions, model_name):
        self.dict_predictions = dict_predictions.copy()
        self.trained = True
        self.nb_fit = 1
        self.model_name = model_name
        self.model_dir = os.path.join('false_path',f'{model_name}')
        self.model_type = 'regressor'

    def predict(self, x_test, **kwargs):
        return np.array([self.dict_predictions[tuple(x)] for x in x_test])

dict_predictions_1 = {(1, 2): 1, (1, 3): 2, (1, 4): 3, (2,1): 2, (2,2): 5}
dict_predictions_2 = {(1, 2): -3, (1, 3): 3, (1, 4): 4, (2,1): 4, (2,2): 3}
dict_predictions_3 = {(1, 2): 4.1, (1, 3): 4.4, (1, 4): 5.2, (2,1): 3.1, (2,2): -23}
dict_predictions_4 = {(1, 2): 2, (1, 3): 1, (1, 4): -2, (2,1): 2, (2,2): 1}
dict_predictions_5 = {(1, 2): 6, (1, 3): -1, (1, 4): 0, (2,1): 1, (2,2): 2}
list_dict_predictions = [dict_predictions_1, dict_predictions_2, dict_predictions_3, dict_predictions_4, dict_predictions_5]

x_test = np.array([[1, 2], [1, 3], [1, 4], [2, 1], [2, 2]])

mock_target_mean = np.array([np.mean([dict_prediction[tuple(x)] for dict_prediction in list_dict_predictions]) for x in x_test])
mock_target_median = np.array([np.median([dict_prediction[tuple(x)] for dict_prediction in list_dict_predictions]) for x in x_test])

mock_model_1 = MockModel(dict_predictions_1, 'model_1')
mock_model_2 = MockModel(dict_predictions_2, 'model_2')
mock_model_3 = MockModel(dict_predictions_3, 'model_3')
mock_model_4 = MockModel(dict_predictions_4, 'model_4')
mock_model_5 = MockModel(dict_predictions_5, 'model_5')
list_mock_model = [mock_model_1, mock_model_2, mock_model_3, mock_model_4, mock_model_5]

target_get_predictions = np.array([[model.predict(x_test)[i] for model in list_mock_model] for i, x in enumerate(x_test)])


class ModelAggregationRegressorTests(unittest.TestCase):
    '''Main class to test model_aggregation_regressor'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    # Create and save a ModelGBTRegressor and a ModelSGDRegressor models
    def create_models(self, gbt_param: dict = None, sgd_param: dict = None):
        model_path = utils.get_models_path()
        model_dir_gbt = os.path.join(model_path, 'model_test_123456789_gbt')
        model_dir_sgd = os.path.join(model_path, 'model_test_123456789_sgd')
        remove_dir(model_dir_gbt)
        remove_dir(model_dir_sgd)

        gbt_param = {} if gbt_param is None else gbt_param
        sgd_param = {} if sgd_param is None else sgd_param

        gbt = ModelGBTRegressor(model_dir=model_dir_gbt, **gbt_param)
        sgd = ModelSGDRegressor(model_dir=model_dir_sgd, **sgd_param)

        gbt.save()
        sgd.save()
        gbt_name = os.path.split(gbt.model_dir)[-1]
        sgd_name = os.path.split(sgd.model_dir)[-1]
        return gbt, sgd, gbt_name, sgd_name

    def test01_model_aggregation_regressor_init(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model_name = 'test_model_name'

        x_train = pd.DataFrame.from_dict({'0':{'1': 1.2, '2':3.2}, '1':{'1': 1.1, '2':-3.2}}, orient='index')
        y_train = pd.Series([0.1, 0.2])

        def test_init_partial(model, model_name, model_dir, list_model_names, trained):
            self.assertTrue(os.path.isdir(model_dir))
            self.assertEqual(model.model_name, model_name)
            self.assertTrue(isinstance(model.sub_models, list))
            self.assertEqual([sub_model['name'] for sub_model in model.sub_models], list_model_names)
            self.assertTrue(isinstance(model.sub_models[0]['model'], ModelGBTRegressor))
            self.assertTrue(isinstance(model.sub_models[1]['model'], ModelSGDRegressor))
            self.assertEqual(model.trained, trained)

            # We test display_if_gpu_activated and _is_gpu_activated just by calling them
            model.display_if_gpu_activated()
            self.assertTrue(isinstance(model._is_gpu_activated(), bool))

        ############################################
        # Init., test all parameters
        ############################################

        # list_models = [model, model]
        # aggregation_function: median_predict
        # using_proba
        # not multi_label
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        gbt.fit(x_train, y_train)
        list_models = [gbt, sgd]
        list_model_names = [gbt_name, sgd_name]
        model = ModelAggregationRegressor(model_dir=model_dir, model_name=model_name, list_models=list_models, aggregation_function='median_predict')
        test_init_partial(model, model_name, model_dir, list_model_names, trained=False)
        self.assertEqual(median_predict.__code__.co_code, model.aggregation_function.__code__.co_code)
        remove_dir_model(model, model_dir)

        # list_models = [model_name, model_name]
        # aggregation_function: mean_predict
        # not using_proba
        # not multi_label
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt_name, sgd_name]
        list_model_names = [gbt_name, sgd_name]
        model = ModelAggregationRegressor(model_dir=model_dir, model_name=model_name, list_models=list_models, aggregation_function='mean_predict')
        test_init_partial(model, model_name, model_dir, list_model_names, trained=False)
        self.assertEqual(mean_predict.__code__.co_code, model.aggregation_function.__code__.co_code)
        remove_dir_model(model, model_dir)

        # list_models = [model_name, model]
        # aggregation_function: median_predict
        # not using_proba
        # multi_label
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt_name, sgd]
        list_model_names = [gbt_name, sgd_name]
        model = ModelAggregationRegressor(model_dir=model_dir, model_name=model_name, list_models=list_models, aggregation_function='median_predict')
        test_init_partial(model, model_name, model_dir, list_model_names, trained=False)
        self.assertEqual(median_predict.__code__.co_code, model.aggregation_function.__code__.co_code)
        remove_dir_model(model, model_dir)

        # list_models = [model_name, model]
        # aggregation_function: Callable
        # not using_proba
        # not multi_label

        # This function is a copy of median_predict function
        def function_test(predictions):
            return np.median(predictions)

        gbt, sgd, gbt_name, sgd_name = self.create_models()
        gbt.fit(x_train, y_train)
        gbt.save()
        sgd.fit(x_train, y_train)
        list_models = [gbt_name, sgd]
        list_model_names = [gbt_name, sgd_name]
        model = ModelAggregationRegressor(model_dir=model_dir, model_name=model_name, list_models=list_models, aggregation_function=function_test)
        test_init_partial(model, model_name, model_dir, list_model_names, trained=True)
        self.assertEqual(function_test.__code__.co_code, model.aggregation_function.__code__.co_code)
        remove_dir_model(model, model_dir)

        ############################################
        # Error
        ############################################

        # if the object aggregation_function is a str but not found in the dictionary dict_aggregation_function
        with self.assertRaises(ValueError):
            model = ModelAggregationRegressor(model_dir=model_dir, aggregation_function='1234')
        remove_dir(model_dir)

        # The classifier and regressor models cannot be combined in list models
        gbt, sgd, _, _ = self.create_models()
        model_dir_classifier = os.path.join(utils.get_models_path(), 'model_test_123456789_classifier')
        sgd_classifier = ModelSGDClassifier(model_dir=model_dir_classifier)
        list_models = [gbt, sgd_classifier]
        with self.assertRaises(ValueError):
            model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_models)
        remove_dir_model(model, model_dir)
        remove_dir(sgd.model_dir)
        remove_dir(sgd_classifier.model_dir)

    def test02_model_aggregation_setter_aggregation_function(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.aggregation_function'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model = ModelAggregationRegressor(model_dir=model_dir, multi_label=False)
        model.aggregation_function = 'mean_predict'
        self.assertEqual(model.aggregation_function.__code__.co_code, mean_predict.__code__.co_code)
        model.aggregation_function = 'median_predict'
        self.assertEqual(model.aggregation_function.__code__.co_code, median_predict.__code__.co_code)
        # str not in the dictionary
        with self.assertRaises(ValueError):
            model.aggregation_function = 'coucou'
        remove_dir(model_dir)

        # local aggregation function
        # This function is a copy of median_predict function
        def function_test(predictions: np.ndarray) -> np.float64:
            '''Returns the mean of the predictions of each model

            Args:
                predictions (np.ndarray) : The array containing the predictions of each models (shape (n_models))
            Return:
                (np.float64) : The mean of the predictions
            '''
            return np.mean(predictions)

        model = ModelAggregationRegressor(model_dir=model_dir)
        model.aggregation_function = function_test
        self.assertEqual(model.aggregation_function.__code__.co_code, function_test.__code__.co_code)
        remove_dir(model_dir)

    def test03_model_aggregation_setter_sub_models(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.sub_models'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        def check_sub_models(sub_models, list_models_name):
            self.assertTrue(isinstance(sub_models[0]['model'], ModelGBTRegressor))
            self.assertTrue(isinstance(sub_models[1]['model'], ModelSGDRegressor))
            self.assertEqual(len(sub_models), len(list_models))
            self.assertEqual([sub_model['name'] for sub_model in sub_models], list_models_name)

        # list_models = [model, model]
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt, sgd]
        list_models_name = [gbt_name, sgd_name]
        model = ModelAggregationRegressor(model_dir=model_dir)
        model.sub_models = list_models
        check_sub_models(model.sub_models, list_models_name)
        remove_dir_model(model, model_dir)

        # list_models = [model_name, model_name]
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt_name, sgd_name]
        list_models_name = [gbt_name, sgd_name]
        model = ModelAggregationRegressor(model_dir=model_dir)
        model.sub_models = list_models
        check_sub_models(model.sub_models, list_models_name)
        remove_dir_model(model, model_dir)

        # list_models = [model_name, model]
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt_name, sgd]
        list_models_name = [gbt_name, sgd_name]
        model = ModelAggregationRegressor(model_dir=model_dir)
        model.sub_models = list_models
        check_sub_models(model.sub_models, list_models_name)
        remove_dir_model(model, model_dir)

    def test04_model_aggregation_regressor_check_trained(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor._check_trained'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_int = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)

        # int
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir)
        self.assertFalse(model._check_trained())
        gbt.fit(x_train, y_train_int)
        sgd.fit(x_train, y_train_int)
        model.sub_models = [gbt, sgd]
        self.assertTrue(model._check_trained())
        remove_dir_model(model, model_dir)

        # not trained
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir)
        self.assertFalse(model._check_trained())
        model.sub_models = [gbt, sgd]
        self.assertFalse(model._check_trained())
        remove_dir_model(model, model_dir)

        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir)
        self.assertFalse(model._check_trained())
        sgd.fit(x_train, y_train_int)
        model.sub_models = [gbt, sgd]
        self.assertFalse(model._check_trained())
        remove_dir_model(model, model_dir)
        remove_dir(gbt.model_dir)
        remove_dir(sgd.model_dir)

    def test05_model_aggregation_regressor_fit(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.fit'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        def check_not_trained(model):
            self.assertFalse(model.trained)
            self.assertEqual(model.nb_fit, 0)

        def check_trained(model):
            self.assertTrue(model.trained)
            self.assertEqual(model.nb_fit, 1)
            for sub_model in model.sub_models:
                self.assertTrue(sub_model['model'].trained)
                self.assertEqual(sub_model['model'].nb_fit, 1)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train = pd.Series([1, 1, 1, 2, 3, 3, 3] * 10)

        # Not trained
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        check_not_trained(model)
        model.fit(x_train, y_train)
        check_trained(model)
        remove_dir_model(model, model_dir)

        # Some model trained
        gbt, sgd, _, _ = self.create_models()
        gbt.fit(x_train, y_train)
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        check_not_trained(model)
        model.fit(x_train, y_train)
        check_trained(model)
        remove_dir_model(model, model_dir)

    def test06_model_aggregation_regressor_predict(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.predict'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        def test_method(model, x_test, target_predict):
            preds = model.predict(x_test)
            self.assertEqual(preds.shape, target_predict.shape)
            for i in range(len(preds)):
                self.assertAlmostEqual(preds[i], target_predict[i], places=4)

        # median_predict
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_mock_model, aggregation_function='median_predict')
        test_method(model, x_test, target_predict=mock_target_median)
        remove_dir(model_dir)

        # mean_predict
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_mock_model, aggregation_function='mean_predict')
        test_method(model, x_test, target_predict=mock_target_mean)
        remove_dir(model_dir)

        # This function is a copy of median_predict function
        def function_test(predictions):
            return np.median(predictions)

        # callable
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_mock_model, aggregation_function=function_test)
        test_method(model, x_test, target_predict=mock_target_median)
        remove_dir(model_dir)

        ############################################
        # Error
        ############################################

        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train = pd.Series([1, 1, 1, 2, 3, 3, 3] * 10)

        # Model needs to be fitted
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        with self.assertRaises(AttributeError):
            model.predict(x_train)
        remove_dir_model(model, model_dir)

        # No return_proba
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train)
        with self.assertRaises(ValueError):
            model.predict(x_train, return_proba=True)
        remove_dir_model(model, model_dir)

    def test07_model_aggregation_regressor_predict_proba(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.predict_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train = pd.Series([1, 1, 1, 2, 3, 3, 3] * 10)

        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train)
        with self.assertRaises(ValueError):
            model.predict_proba(x_train)
        remove_dir_model(model, model_dir)

    def test08_model_aggregation_regressor_get_predictions(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor._predict_sub_models'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_mock_model)
        preds = model._predict_sub_models(x_test)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(target_get_predictions.shape, preds.shape)
        for i in range(target_get_predictions.shape[0]):
            for j in range(target_get_predictions.shape[1]):
                self.assertAlmostEqual(target_get_predictions[i][j], preds[i][j])
        remove_dir(model_dir)

        # Model needs to be fitted
        model = ModelAggregationRegressor(model_dir=model_dir)
        with self.assertRaises(AttributeError):
            model._predict_sub_models('test')
        remove_dir(model_dir)

    def test09_median_predict(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.median_predict'''

        # 3 models int
        preds = np.array([3, 4, 4])
        self.assertAlmostEqual(median_predict(preds), np.median(preds))
        # 3 models float
        preds = np.array([3, 3.1, 4])
        self.assertAlmostEqual(median_predict(preds), np.median(preds))
        # 1 model int
        preds = np.array([5])
        self.assertAlmostEqual(median_predict(preds), np.median(preds))
        # 1 model float
        preds = np.array([5.2])
        self.assertAlmostEqual(median_predict(preds), np.median(preds))


    def test10_mean_predict(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.mean_predict'''

        # 3 models int
        preds = np.array([3, 4, 4])
        self.assertAlmostEqual(mean_predict(preds), np.mean(preds))
        # 3 models float
        preds = np.array([3, 3.1, 4])
        self.assertAlmostEqual(mean_predict(preds), np.mean(preds))
        # 1 model int
        preds = np.array([5])
        self.assertAlmostEqual(mean_predict(preds), np.mean(preds))
        # 1 model float
        preds = np.array([5.2])
        self.assertAlmostEqual(mean_predict(preds), np.mean(preds))

    def test11_model_aggregation_regressor_save(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.save'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train = pd.Series([0.1, 0.5, 0.2, -2.9, 1.1, 3.2, 1.0] * 10)

        gbt, sgd, gbt_name, sgd_name = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train)
        model.save(json_data={'test': 10})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))

        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 10)
        set_attributes_in_config = {'package_version', 'model_name', 'model_dir', 'trained', 'nb_fit', 
                                    'x_col', 'y_col', 'level_save', 'librairie'}
        set_attributes_in_config_tot = set_attributes_in_config.union({'list_models_name'})
        self.assertTrue(set_attributes_in_config_tot.issubset(set(configs.keys())))
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertEqual(configs['librairie'], None)

        for sub_model in model.sub_models:
            self.assertTrue(os.path.exists(os.path.join(sub_model['model'].model_dir, 'configurations.json')))
            self.assertTrue(os.path.exists(os.path.join(sub_model['model'].model_dir, f"{sub_model['model'].model_name}.pkl")))
            with open(os.path.join(sub_model['model'].model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
                configs = json.load(f)
            self.assertTrue(set_attributes_in_config.issubset(set(configs.keys())))
            self.assertEqual(configs['package_version'], utils.get_package_version())

        self.assertTrue(os.path.exists(os.path.join(model_dir, "model_upload_instructions.md")))
        with open(os.path.join(model_dir, "model_upload_instructions.md"), 'r') as read_obj:
            text = read_obj.read()
            self.assertEqual(text[0:20], "/!\\/!\/!\\/!\/!\\   The aggregation model is a special model, please ensure that"[0:20])
        remove_dir_model(model, model_dir)

        # Save each trained and unsaved model
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        gbt.fit(x_train, y_train)
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train)
        model.save(json_data={'test': 10})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))

        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 10)
        set_attributes_in_config = {'package_version', 'model_name', 'model_dir', 'trained', 'nb_fit', 
                                    'x_col', 'y_col', 'level_save', 'librairie'}
        set_attributes_in_config_tot = set_attributes_in_config.union({'list_models_name'})
        self.assertTrue(set_attributes_in_config_tot.issubset(set(configs.keys())))
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertEqual(configs['librairie'], None)

        for sub_model in model.sub_models:
            self.assertTrue(os.path.exists(os.path.join(sub_model['model'].model_dir, 'configurations.json')))
            self.assertTrue(os.path.exists(os.path.join(sub_model['model'].model_dir, f"{sub_model['model'].model_name}.pkl")))
            with open(os.path.join(sub_model['model'].model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
                configs = json.load(f)
            self.assertTrue(set_attributes_in_config.issubset(set(configs.keys())))
            self.assertEqual(configs['package_version'], utils.get_package_version())

        self.assertTrue(os.path.exists(os.path.join(model_dir, "model_upload_instructions.md")))
        with open(os.path.join(model_dir, "model_upload_instructions.md"), 'r') as read_obj:
            text = read_obj.read()
            self.assertEqual(text[0:20], "/!\\/!\/!\\/!\/!\\   The aggregation model is a special model, please ensure that"[0:20])
        remove_dir_model(model, model_dir)

        # Same thing with a local function
        # This function is a copy of median_predict function
        def function_test(predictions):
            return np.median(predictions)

        gbt, sgd, gbt_name, sgd_name = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd], aggregation_function=function_test)
        model.fit(x_train, y_train)
        model.save(json_data={'test': 10})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))

        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 10)
        set_attributes_in_config = {'package_version', 'model_name', 'model_dir', 'trained', 'nb_fit', 
                                    'x_col', 'y_col', 'level_save', 'librairie'}
        set_attributes_in_config_tot = set_attributes_in_config.union({'list_models_name'})
        self.assertTrue(set_attributes_in_config_tot.issubset(set(configs.keys())))
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertEqual(configs['librairie'], None)

        for sub_model in model.sub_models:
            self.assertTrue(os.path.exists(os.path.join(sub_model['model'].model_dir, 'configurations.json')))
            self.assertTrue(os.path.exists(os.path.join(sub_model['model'].model_dir, f"{sub_model['model'].model_name}.pkl")))
            with open(os.path.join(sub_model['model'].model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
                configs = json.load(f)
            self.assertTrue(set_attributes_in_config.issubset(set(configs.keys())))
            self.assertEqual(configs['package_version'], utils.get_package_version())

        self.assertTrue(os.path.exists(os.path.join(model_dir, "model_upload_instructions.md")))
        with open(os.path.join(model_dir, "model_upload_instructions.md"), 'r') as read_obj:
            text = read_obj.read()
            self.assertEqual(text[0:20], "/!\\/!\/!\\/!\/!\\   The aggregation model is a special model, please ensure that"[0:20])
        remove_dir_model(model, model_dir)

    def test12_model_aggregation_prepend_line(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.prepend_line'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model = ModelAggregationRegressor(model_dir=model_dir)
        path = os.path.join(model_dir, 'test.md')
        with open(path, 'w') as f:
            f.write('toto')
        with open(path, 'r') as f:
            self.assertTrue(f.read() == 'toto')
        model.prepend_line(path, 'titi\n')
        with open(path, 'r') as f:
            self.assertTrue(f.read() == 'titi\ntoto')
        remove_dir(model_dir)

    def test13_model_aggregation_regressor_reload_from_standalone(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        model_new_dir = os.path.join(os.getcwd(), 'model_new_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, -1, 0, 4.2, 6, 3] * 10, 'col_2': [-8.1, -1.3, -1, 0, 4, 12, 2] * 10})
        x_test = pd.DataFrame({'col_1': [-5], 'col_2': [-8.1]})
        y_train = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)

        # Create model
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregationRegressor(model_dir=model_new_dir)
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path, preprocess_pipeline_path=preprocess_pipeline_path)

        # Test
        self.assertEqual(model.trained, model_new.trained)
        self.assertEqual(model.nb_fit, model_new.nb_fit)
        self.assertEqual(model.x_col, model_new.x_col)
        self.assertEqual(model.y_col, model_new.y_col)
        self.assertEqual(model.level_save, model_new.level_save)
        self.assertEqual([sub_model['name'] for sub_model in model.sub_models], [sub_model['name'] for sub_model in model_new.sub_models])
        self.assertTrue(isinstance(model.sub_models[0]['model'], type(model_new.sub_models[0]['model'])))
        self.assertTrue(isinstance(model.sub_models[1]['model'], type(model_new.sub_models[1]['model'])))
        self.assertEqual(model.aggregation_function.__code__.co_code, model_new.aggregation_function.__code__.co_code)
        preds = model.predict(x_test)
        new_preds = model_new.predict(x_test)
        for pred, new_pred in zip(preds, new_preds):
            self.assertAlmostEqual(pred, new_pred)
        remove_dir_model(model, model_dir)
        remove_dir_model(model_new, model_new_dir)

        ############################################
        # Errors
        ############################################

        model_new = ModelAggregationRegressor(model_dir=model_new_dir)
        with self.assertRaises(FileNotFoundError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path='toto.json', aggregation_function_path=aggregation_function_path, preprocess_pipeline_path=preprocess_pipeline_path)
        remove_dir(model_new.model_dir)
        remove_dir(model_new_dir)
        model_new = ModelAggregationRegressor(model_dir=model_new_dir)
        with self.assertRaises(FileNotFoundError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path='toto.pkl', preprocess_pipeline_path=preprocess_pipeline_path)
        remove_dir(model_new.model_dir)
        remove_dir(model_new_dir)
        model_new = ModelAggregationRegressor(model_dir=model_new_dir)
        with self.assertRaises(FileNotFoundError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path, preprocess_pipeline_path='toto.pkl')
        remove_dir(model_new.model_dir)
        remove_dir(model_new_dir)

        model_new = ModelAggregationRegressor(model_dir=model_new_dir)
        with self.assertRaises(ValueError):
            model_new.reload_from_standalone(model_dir=model_dir, aggregation_function_path=aggregation_function_path, preprocess_pipeline_path=preprocess_pipeline_path)
        remove_dir(model_new.model_dir)
        remove_dir(model_new_dir)
        model_new = ModelAggregationRegressor(model_dir=model_new_dir)
        with self.assertRaises(ValueError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, preprocess_pipeline_path=preprocess_pipeline_path)
        remove_dir(model_new.model_dir)
        remove_dir(model_new_dir)
        model_new = ModelAggregationRegressor(model_dir=model_new_dir)
        with self.assertRaises(ValueError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)
        remove_dir(model_new.model_dir)
        remove_dir(model_new_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()