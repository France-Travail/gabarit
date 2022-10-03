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
from {{package_name}}.models_training.classifiers.model_sgd_classifier import ModelSGDClassifier
from {{package_name}}.models_training.regressors.model_gbt_regressor import ModelGBTRegressor
from {{package_name}}.models_training.regressors.model_sgd_regressor import ModelSGDRegressor
from {{package_name}}.models_training.regressors.model_aggregation_regressor import ModelAggregationRegressor

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


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

        ############################################
        # Init., test all parameters
        ############################################

        # list_models = [model, model]
        # aggregation_function: median_predict
        # using_proba
        # not multi_label
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt, sgd]
        model = ModelAggregationRegressor(model_dir=model_dir, model_name=model_name, list_models=list_models, aggregation_function='median_predict')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertTrue(isinstance(model.list_models, list))
        self.assertEqual(model.list_models, [gbt_name, sgd_name])
        self.assertEqual(model.list_models_trained, [False, False])
        self.assertTrue(isinstance(model.list_real_models[0], type(gbt)))
        self.assertTrue(isinstance(model.list_real_models[1], type(sgd)))
        self.assertTrue(isinstance(model._is_gpu_activated(), bool))
        self.assertEqual(model.median_predict.__code__.co_code, model.aggregation_function.__code__.co_code)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model_name]
        # aggregation_function: mean_predict
        # not using_proba
        # not multi_label
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt_name, sgd_name]
        model = ModelAggregationRegressor(model_dir=model_dir, model_name=model_name, list_models=list_models, aggregation_function='mean_predict')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model_name, model_name)
        self.assertTrue(isinstance(model.list_models, list))
        self.assertEqual(model.list_models, list_models)
        self.assertEqual(model.list_models_trained, [False, False])
        self.assertTrue(isinstance(model.list_real_models[0], type(gbt)))
        self.assertTrue(isinstance(model.list_real_models[1], type(sgd)))
        self.assertTrue(isinstance(model._is_gpu_activated(), bool))
        self.assertEqual(model.mean_predict.__code__.co_code, model.aggregation_function.__code__.co_code)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model]
        # aggregation_function: median_predict
        # not using_proba
        # multi_label
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt_name, sgd]
        model = ModelAggregationRegressor(model_dir=model_dir, model_name=model_name, list_models=list_models, aggregation_function='median_predict')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model_name, model_name)
        self.assertTrue(isinstance(model.list_models, list))
        self.assertEqual(model.list_models, [gbt_name, sgd_name])
        self.assertEqual(model.list_models_trained, [False, False])
        self.assertTrue(isinstance(model.list_real_models[0], type(gbt)))
        self.assertTrue(isinstance(model.list_real_models[1], type(sgd)))
        self.assertTrue(isinstance(model._is_gpu_activated(), bool))
        self.assertEqual(model.median_predict.__code__.co_code, model.aggregation_function.__code__.co_code)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model]
        # aggregation_function: Callable
        # not using_proba
        # not multi_label

        # This function is a copy of median_predict function
        def function_test(predictions):
            return np.median(predictions)

        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt_name, sgd]
        model = ModelAggregationRegressor(model_dir=model_dir, model_name=model_name, list_models=list_models, aggregation_function=function_test)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model_name, model_name)
        self.assertTrue(isinstance(model.list_models, list))
        self.assertEqual(model.list_models, [gbt_name, sgd_name])
        self.assertTrue(isinstance(model.list_real_models[0], type(gbt)))
        self.assertTrue(isinstance(model.list_real_models[1], type(sgd)))
        self.assertTrue(isinstance(model._is_gpu_activated(), bool))
        self.assertEqual(function_test.__code__.co_code, model.aggregation_function.__code__.co_code)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        model.display_if_gpu_activated()
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

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
        sgd_regressor = ModelSGDClassifier(model_dir=model_dir_classifier)
        list_models = [gbt, sgd_regressor]
        with self.assertRaises(ValueError):
            model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_models)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(sgd_regressor.model_dir)
        remove_dir(gbt.model_dir)
        remove_dir(sgd.model_dir)

    def test02_model_aggregation_regressor_sort_model_type(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor._sort_model_type'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # list_models = [model, model]
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt, sgd]
        model = ModelAggregationRegressor(model_dir=model_dir)
        model._sort_model_type(list_models)
        self.assertTrue(isinstance(model.list_real_models[0], type(gbt)))
        self.assertTrue(isinstance(model.list_real_models[1], type(sgd)))
        self.assertEqual(len(model.list_models), len(list_models))
        self.assertEqual(model.list_models, [gbt_name, sgd_name])
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model_name]
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt_name, sgd_name]
        model = ModelAggregationRegressor(model_dir=model_dir)
        model._sort_model_type(list_models)
        self.assertTrue(isinstance(model.list_real_models[0], type(gbt)))
        self.assertTrue(isinstance(model.list_real_models[1], type(sgd)))
        self.assertEqual(len(model.list_models), len(list_models))
        self.assertEqual(model.list_models, [gbt_name, sgd_name])
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # list_models = [model_name, model]
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt_name, sgd]
        model = ModelAggregationRegressor(model_dir=model_dir)
        model._sort_model_type(list_models)
        self.assertTrue(isinstance(model.list_real_models[0], type(gbt)))
        self.assertTrue(isinstance(model.list_real_models[1], type(sgd)))
        self.assertEqual(len(model.list_models), len(list_models))
        self.assertEqual(model.list_models, [gbt_name, sgd_name])
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(gbt.model_dir)
        remove_dir(sgd.model_dir)

    def test03_model_aggregation_regressor_check_trained(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor._check_trained'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_int = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)

        # int
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir)
        self.assertFalse(model.trained)
        gbt.fit(x_train, y_train_int)
        sgd.fit(x_train, y_train_int)
        model._sort_model_type([gbt, sgd])
        model._check_trained()
        self.assertTrue(model.trained)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # not trained
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir)
        self.assertFalse(model.trained)
        model._sort_model_type([gbt, sgd])
        model._check_trained()
        self.assertFalse(model.trained)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir)
        self.assertFalse(model.trained)
        sgd.fit(x_train, y_train_int)
        model._sort_model_type([gbt, sgd])
        model._check_trained()
        self.assertFalse(model.trained)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(gbt.model_dir)
        remove_dir(sgd.model_dir)

    def test04_model_aggregation_regressor_fit(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.fit'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono = pd.Series([1, 1, 1, 2, 3, 3, 3] * 10)

        # Not trained
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for submodel in model.list_real_models:
            self.assertTrue(submodel.trained)
            self.assertEqual(submodel.nb_fit, 1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # Some model trained
        gbt, sgd, _, _ = self.create_models()
        gbt.fit(x_train, y_train_mono)
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        for submodel in model.list_real_models:
            self.assertTrue(submodel.trained)
            self.assertEqual(submodel.nb_fit, 1)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(gbt.model_dir)
        remove_dir(sgd.model_dir)

    def test05_model_aggregation_regressor_predict(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.predict'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        #################################################
        # Set vars mono_label
        #################################################

        dic_mono = {'x_train': pd.DataFrame({'col_1': [-5, -1, -1, 0, 4, 6, 3] * 10, 'col_2': [-2, -1, -8, 0, 4, 12, 2] * 10}),
                    'x_test': pd.DataFrame({'col_1': [-5, 2], 'col_2': [-2, 3]}),
                    'y_train_1': pd.Series([0, 0, 0, 2, 1, 1, 1] * 10),
                    'y_train_2': pd.Series([3, 3, 3, 2, 4, 4, 4] * 10),
                    'y_train_9': pd.Series([9, 9, 9, 1, 9, 9, 9] * 10),
                    'target_1': np.array([0, 1]),
                    'target_2': np.array([3, 4]),
                    'target_9': np.array([np.median(np.array([9, 0, 0, 3, 3])), np.median(np.array([9, 1, 1, 4, 4]))]),
                    'target_median': np.array([3, 4]),
                    'target_mean': np.array([np.mean(np.array([0, 3, 3])), np.mean(np.array([1, 4, 4]))])}

        model_path = utils.get_models_path()

        list_model_mono = [ModelGBTRegressor(model_dir=os.path.join(model_path, 'model_test_123456789_gbt_regressor_1')),
                           ModelGBTRegressor(model_dir=os.path.join(model_path, 'model_test_123456789_gbt_regressor_2')),
                           ModelGBTRegressor(model_dir=os.path.join(model_path, 'model_test_123456789_gbt_regressor_3')),
                           ModelGBTRegressor(model_dir=os.path.join(model_path, 'model_test_123456789_gbt_regressor_4')),
                           ModelGBTRegressor(model_dir=os.path.join(model_path, 'model_test_123456789_gbt_regressor_5'))]

        for i in range(2):
            list_model_mono[i].fit(dic_mono['x_train'], dic_mono['y_train_1'])
            list_model_mono[i + 2].fit(dic_mono['x_train'], dic_mono['y_train_2'])
        list_model_mono[4].fit(dic_mono['x_train'], dic_mono['y_train_9'])

        def test_method(model, x_test, target_predict):
            preds = model.predict(x_test)
            self.assertEqual(preds.shape, target_predict.shape)
            for i in range(len(preds)):
                self.assertAlmostEqual(preds[i], target_predict[i], places=4)

        #################################################
        # aggregation_function: median_predict
        #################################################

        # All models have the same labels
        list_models = [list_model_mono[0], list_model_mono[1]]
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_models, aggregation_function='median_predict')
        test_method(model, dic_mono['x_test'], target_predict=dic_mono['target_1'])
        remove_dir(model_dir)

        # The models have different labels
        list_models = [list_model_mono[0], list_model_mono[2], list_model_mono[3]]
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_models, aggregation_function='median_predict')
        test_method(model, dic_mono['x_test'], target_predict=dic_mono['target_median'])
        remove_dir(model_dir)

        #################################################
        # aggregation_function: mean_predict
        #################################################

        # All models have the same labels
        list_models = [list_model_mono[0], list_model_mono[1]]
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_models, aggregation_function='mean_predict')
        test_method(model, dic_mono['x_test'], target_predict=dic_mono['target_1'])
        remove_dir(model_dir)

        # The models have different labels
        list_models = [list_model_mono[0], list_model_mono[2], list_model_mono[3]]
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_models, aggregation_function='mean_predict')
        test_method(model, dic_mono['x_test'], target_predict=dic_mono['target_mean'])
        remove_dir(model_dir)

        #################################################
        # aggregation_function: Callable
        #################################################

        # This function is a copy of median_predict function
        def function_test(predictions):
            return np.median(predictions)

        # All models have the same labels
        list_models = [list_model_mono[0], list_model_mono[1]]
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_models, using_proba=False, multi_label=False, aggregation_function=function_test)
        test_method(model, dic_mono['x_test'], target_predict=dic_mono['target_1'])
        remove_dir(model_dir)

        # The models have different labels
        list_models = [list_model_mono[0], list_model_mono[2], list_model_mono[3]]
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_models, using_proba=False, multi_label=False, aggregation_function=function_test)
        test_method(model, dic_mono['x_test'], target_predict=dic_mono['target_2'])
        remove_dir(model_dir)

        # Equality case
        list_models = [list_model_mono[4], list_model_mono[0], list_model_mono[1], list_model_mono[2], list_model_mono[3]]
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=list_models, aggregation_function='median_predict')
        test_method(model, dic_mono['x_test'], target_predict=dic_mono['target_9'])
        remove_dir(model_dir)

        for i in range(len(list_model_mono)):
            remove_dir(list_model_mono[i].model_dir)

        ############################################
        # Error
        ############################################

        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono = pd.Series([1, 1, 1, 2, 3, 3, 3] * 10)

        # Model needs to be fitted
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        with self.assertRaises(AttributeError):
            model.predict(x_train)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train_mono)
        with self.assertRaises(ValueError):
            model.predict(x_train, return_proba=True)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(gbt.model_dir)
        remove_dir(sgd.model_dir)

    def test06_model_aggregation_regressor_predict_proba(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.predict_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono = pd.Series([1, 1, 1, 2, 3, 3, 3] * 10)

        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train_mono)
        with self.assertRaises(ValueError):
            model.predict_proba(x_train)
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(gbt.model_dir)
        remove_dir(sgd.model_dir)

    def test07_model_aggregation_regressor_get_predictions(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor._get_predictions'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)

        # mono_label
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train_mono)
        preds = model._get_predictions(x_train)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(len(preds), len(x_train))
        self.assertTrue(([preds[i][0] for i in range(len(x_train))] == gbt.predict(x_train)).all())
        self.assertTrue(([preds[i][1] for i in range(len(x_train))] == sgd.predict(x_train)).all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train_mono)
        preds = model._get_predictions(x_train)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(len(preds), len(x_train))
        self.assertTrue(([preds[i][0] for i in range(len(x_train))] == gbt.predict(x_train)).all())
        self.assertTrue(([preds[i][1] for i in range(len(x_train))] == sgd.predict(x_train)).all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)

        # Model needs to be fitted
        model = ModelAggregationRegressor(model_dir=model_dir)
        with self.assertRaises(AttributeError):
            model._get_predictions('test')
        remove_dir(model_dir)
        remove_dir(gbt.model_dir)
        remove_dir(sgd.model_dir)

    def test08_model_aggregation_regressor_median_predict(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.median_predict'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model = ModelAggregationRegressor(model_dir=model_dir)

        # normal case (3 models)
        preds = np.array([3, 4, 4])
        self.assertEqual(model.median_predict(preds), 4)
        # normal case (1 model)
        preds = np.array([5])
        self.assertEqual(model.median_predict(preds), 5)
        # same predict (5 models)
        preds = np.array([4, 3, 4, 2, 2])
        self.assertEqual(model.median_predict(preds), 3)

        remove_dir(model_dir)

    def test09_model_aggregation_regressor_mean_predict(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.mean_predict'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model = ModelAggregationRegressor(model_dir=model_dir)

        # normal case (3 models)
        preds = np.array([3, 2, 4])
        self.assertEqual(model.mean_predict(preds), 3)
        # normal case (1 model)
        preds = np.array([5])
        self.assertEqual(model.mean_predict(preds), 5)
        # same predict (5 models)
        preds = np.array([4, 3, 4, 2, 2])
        self.assertEqual(model.mean_predict(preds), 3)

        remove_dir(model_dir)

    def test10_model_aggregation_regressor_save(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAggregationRegressor.save'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)

        model_path = utils.get_models_path()
        gbt = ModelGBTRegressor(model_dir=os.path.join(model_path, 'model_test_123456789_gbt'))
        sgd = ModelSGDRegressor(model_dir=os.path.join(model_path, 'model_test_123456789_sgd'))
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train_mono)
        model.save(json_data={'test': 10})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 10)
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertTrue('list_models' in configs.keys())
        self.assertEqual(configs['librairie'], None)
        for submodel in model.list_real_models:
            self.assertTrue(os.path.exists(os.path.join(submodel.model_dir, 'configurations.json')))
            self.assertTrue(os.path.exists(os.path.join(submodel.model_dir, f"{submodel.model_name}.pkl")))
            with open(os.path.join(submodel.model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
                configs = json.load(f)
            self.assertTrue('package_version' in configs.keys())
            self.assertEqual(configs['package_version'], utils.get_package_version())
            self.assertTrue('model_name' in configs.keys())
            self.assertTrue('model_dir' in configs.keys())
            self.assertTrue('trained' in configs.keys())
            self.assertTrue('nb_fit' in configs.keys())
            self.assertTrue('x_col' in configs.keys())
            self.assertTrue('y_col' in configs.keys())
            self.assertTrue('level_save' in configs.keys())
            self.assertTrue('librairie' in configs.keys())
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(gbt.model_dir)
        remove_dir(sgd.model_dir)

    def test11_model_aggregation_regressor_reload_from_standalone(self):
        '''Test of {{package_name}}.models_training.model_aggregation_regressor.ModelAaggregation.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        model_new_dir = os.path.join(os.getcwd(), 'model_new_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, -1, 0, 4, 6, 3] * 10, 'col_2': [-8, -1, -1, 0, 4, 12, 2] * 10})
        x_test = pd.DataFrame({'col_1': [-5], 'col_2': [-8]})
        y_train_mono = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)

        # Create model
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationRegressor(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train_mono)
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
        self.assertEqual(model.list_models, model_new.list_models)
        self.assertTrue(isinstance(model.list_real_models[0], type(model_new.list_real_models[0])))
        self.assertTrue(isinstance(model.list_real_models[1], type(model_new.list_real_models[1])))
        self.assertEqual(model.aggregation_function.__code__.co_code, model_new.aggregation_function.__code__.co_code)
        self.assertTrue((model.predict(x_test) == model_new.predict(x_test)).all())
        for submodel in model.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        for submodel in model_new.list_real_models:
            remove_dir(os.path.split(submodel.model_dir)[-1])
        remove_dir(model_dir)
        remove_dir(model_new_dir)

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