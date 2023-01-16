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
from {{package_name}}.models_training.regressors.models_sklearn.model_sgd_regressor import ModelSGDRegressor
from {{package_name}}.models_training.classifiers.model_classifier import ModelClassifierMixin
from {{package_name}}.models_training.classifiers.models_sklearn.model_gbt_classifier import ModelGBTClassifier
from {{package_name}}.models_training.classifiers.models_sklearn.model_sgd_classifier import ModelSGDClassifier
from {{package_name}}.models_training.classifiers.model_aggregation_classifier import ModelAggregationClassifier, proba_argmax, majority_vote, all_predictions, vote_labels

# Disable logging
import logging
logging.disable(logging.CRITICAL)

models_path = utils.get_models_path()

def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)

def remove_dir_model(model, model_dir):
    for sub_model in model.sub_models:
        remove_dir(os.path.join(models_path, os.path.split(sub_model['model'].model_dir)[-1]))
    remove_dir(model_dir)

# The class to mock the submodels
class MockModel(ModelClassifierMixin, object):

    def __init__(self, dict_predictions, dict_predictions_proba, model_name, multi_label, list_classes):
        self.dict_predictions = dict_predictions.copy()
        self.dict_predictions_proba = dict_predictions_proba.copy()
        self.trained = True
        self.nb_fit = 1
        self.list_classes = list_classes.copy()
        self.dict_classes = {i: col for i, col in enumerate(self.list_classes)}
        self.model_name = model_name
        self.model_dir = os.path.join('false_path',f'{model_name}')
        self.multi_label = multi_label

    def predict(self, x_test, return_proba = False, **kwargs):
        if return_proba:
            return self.predict_proba(x_test, **kwargs)
        else:
            return np.array([self.dict_predictions[tuple(x)] for x in x_test])

    def predict_proba(self, x_test, **kwargs):
        return np.array([self.dict_predictions_proba[tuple(x)] for x in x_test])

# Predictions for the mock mono_label models
dict_predictions_1 = {(1, 2): '0', (1, 3): '0', (2, 1): '1', (2, 2): '0', (3, 1): '1'}
dict_predictions_2 = {(1, 2): '1', (1, 3): '1', (2, 1): '0', (2, 2): '0', (3, 1): '1'}
dict_predictions_3 = {(1, 2): '1', (1, 3): '1', (2, 1): '2', (2, 2): '1', (3, 1): '4'}
dict_predictions_4 = {(1, 2): '2', (1, 3): '2', (2, 1): '3', (2, 2): '3', (3, 1): '1'}
dict_predictions_5 = {(1, 2): '3', (1, 3): '2', (2, 1): '4', (2, 2): '3', (3, 1): '1'}
list_dict_predictions = [dict_predictions_1, dict_predictions_2, dict_predictions_3, dict_predictions_4, dict_predictions_5] 

# Predictions for the mock multi_label models
dict_predictions_multi_1 = {(1, 2): np.array([0, 1]), (1, 3): np.array([0, 0]), 
                            (2, 1): np.array([0, 0]), (2, 2): np.array([1, 0]), (3, 1): np.array([0, 0])}
dict_predictions_multi_2 = {(1, 2): np.array([1, 0]), (1, 3): np.array([1, 0]), (2, 1): np.array([0, 0]), 
                            (2, 2): np.array([0, 1]), (3, 1): np.array([0, 0])}
dict_predictions_multi_3 = {(1, 2): np.array([1, 1, 0]), (1, 3): np.array([1, 0, 0]), 
                            (2, 1): np.array([0, 0, 0]), (2, 2): np.array([1, 0, 0]), (3, 1): np.array([0, 0, 0])}
dict_predictions_multi_4 = {(1, 2): np.array([1, 1, 1]), (1, 3): np.array([0, 1, 0]), 
                            (2, 1): np.array([0, 0, 0]), (2, 2): np.array([0, 1, 0]), (3, 1): np.array([0, 0, 0])}

# Probabilities for the mock models
dict_predictions_proba_1 = {(1, 2): np.array([0.7, 0.3]), (1, 3): np.array([0.6, 0.4]), 
                            (2, 1): np.array([0.2, 0.8]), (2, 2): np.array([0.9, 0.1]), (3, 1): np.array([0.1, 0.9])}
dict_predictions_proba_2 = {(1, 2): np.array([0.7, 0.3]), (1, 3): np.array([0.9, 0.1]), 
                            (2, 1): np.array([0.4, 0.6]), (2, 2): np.array([0.1, 0.9]), (3, 1): np.array([0.9, 0.1])}
dict_predictions_proba_3 = {(1, 2): np.array([0.4, 0.25, 0.35]), (1, 3): np.array([0.4, 0.25, 0.35]), 
                            (2, 1): np.array([0.25, 0.4, 0.35]), (2, 2): np.array([0.4, 0.25, 0.35]), (3, 1): np.array([0.3, 0.25, 0.45])}
dict_predictions_proba_4 = {(1, 2): np.array([0.5, 0.1, 0.4]), (1, 3): np.array([0.4, 0.25, 0.35]), 
                            (2, 1): np.array([0.3, 0.25, 0.45]), (2, 2): np.array([0.4, 0.1, 0.5]), (3, 1): np.array([0.25, 0.4, 0.35])}
dict_predictions_proba_5 = {(1, 2): np.array([0.1, 0.3, 0.5, 0.1]), (1, 3): np.array([0.1, 0.5, 0.3, 0.1]), 
                            (2, 1): np.array([0.1, 0.3, 0.1, 0.5]), (2, 2): np.array([0.1, 0.3, 0.5, 0.1]), (3, 1): np.array([0.5, 0.3, 0.1, 0.1])}

x_test = np.array([(1, 2), (1, 3), (2, 1), (2, 2), (3, 1)])

# Definition of the targets for the mono_label cases
target_predict_mono_majority_vote_dict = {(1, 2):'1', (1, 3):'1', (2, 1):'1', (2, 2):'0', (3, 1):'1'}
target_get_predictions_mono = np.array([[list_dict_predictions[i][tuple(key)] for i in range(len(list_dict_predictions))] for key in x_test])
target_get_proba_mono = np.array([[[0.7, 0.3, 0.0, 0.0, 0.0],
  [0.3, 0.7, 0.0, 0.0, 0.0],
  [0.0, 0.4, 0.25, 0.0, 0.35],
  [0.0, 0.1, 0.5, 0.4, 0.0],
  [0.0, 0.1, 0.3, 0.5, 0.1]],
 [[0.6, 0.4, 0.0, 0.0, 0.0],
  [0.1, 0.9, 0.0, 0.0, 0.0],
  [0.0, 0.4, 0.25, 0.0, 0.35],
  [0.0, 0.25, 0.4, 0.35, 0.0],
  [0.0, 0.1, 0.5, 0.3, 0.1]],
 [[0.2, 0.8, 0.0, 0.0, 0.0],
  [0.6, 0.4, 0.0, 0.0, 0.0],
  [0.0, 0.25, 0.4, 0.0, 0.35],
  [0.0, 0.25, 0.3, 0.45, 0.0],
  [0.0, 0.1, 0.3, 0.1, 0.5]],
 [[0.9, 0.1, 0.0, 0.0, 0.0],
  [0.9, 0.1, 0.0, 0.0, 0.0],
  [0.0, 0.4, 0.25, 0.0, 0.35],
  [0.0, 0.1, 0.4, 0.5, 0.0],
  [0.0, 0.1, 0.3, 0.5, 0.1]],
 [[0.1, 0.9, 0.0, 0.0, 0.0],
  [0.1, 0.9, 0.0, 0.0, 0.0],
  [0.0, 0.3, 0.25, 0.0, 0.45],
  [0.0, 0.4, 0.25, 0.35, 0.0],
  [0.0, 0.5, 0.3, 0.1, 0.1]]])
target_predict_mono_proba_dict = {(1, 2): [0.2 , 0.32, 0.21, 0.18, 0.09],
                                (1, 3): [0.14, 0.41, 0.23, 0.13, 0.09], 
                                (2, 1): [0.16, 0.36, 0.2 , 0.11, 0.17], 
                                (2, 2): [0.36, 0.16, 0.19, 0.2 , 0.09], 
                                (3, 1): [0.04, 0.6 , 0.16, 0.09, 0.11]}
target_predict_mono_majority_vote = np.array([target_predict_mono_majority_vote_dict[tuple(x)] for x in x_test])
target_predict_mono_proba = np.array([target_predict_mono_proba_dict[tuple(x)] for x in x_test])
target_predict_mono_proba_argmax = np.array([str(np.argmax(target_predict_mono_proba_dict[tuple(x)])) for x in x_test])

# Definition of the targets for the multi_label cases
target_predict_multi_all_predictions_dict = {(1, 2): [0, 1, 1, 1, 0], (1, 3): [0, 1, 0, 0, 0,], (2, 1): [0, 0, 0, 0, 0], 
                                                (2, 2): [1, 1, 0, 0, 0], (3, 1): [0, 0, 0, 0, 0]}
target_predict_multi_vote_labels_dict = {(1, 2): [0, 1, 0, 0, 0], (1, 3): [0, 1, 0, 0, 0], (2, 1): [0, 0, 0, 0, 0], 
                                            (2, 2): [1, 0, 0, 0, 0], (3, 1): [0, 0, 0, 0, 0]}
target_predict_multi_proba_dict = {(1, 2):[0.25  , 0.375 , 0.1875, 0.1   , 0.0875],
                                (1, 3):[0.175, 0.4875, 0.1625, 0.0875, 0.0875], 
                                (2, 1):[0.2, 0.425, 0.175, 0.1125, 0.0875], 
                                (2, 2):[0.45, 0.175, 0.1625, 0.125 , 0.0875], 
                                (3, 1):[0.05, 0.625, 0.125, 0.0875, 0.1125]}
target_predict_multi_proba = np.array([target_predict_multi_proba_dict[tuple(x)] for x in x_test])
target_predict_multi_all_predictions = np.array([target_predict_multi_all_predictions_dict[tuple(x)] for x in x_test])
target_predict_multi_vote_labels = np.array([target_predict_multi_vote_labels_dict[tuple(x)] for x in x_test])
target_get_predictions_multi = np.array([[[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 1, 0]],
                                         [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]],
                                         [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                                         [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]],
                                         [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])

# Definition of the targets for the multi_label cases where the submodels are mono_label
target_predict_mono_all_predictions_dict = {(1, 2): [1, 1, 1, 1, 0], (1, 3): [1, 1, 1, 0, 0,], (2, 1): [1, 1, 1, 1, 1], 
                                                (2, 2): [1, 1, 0, 1, 0], (3, 1): [0, 1, 0, 0, 1]}
target_predict_mono_vote_labels_dict = {(1, 2): [0, 0, 0, 0, 0], (1, 3): [0, 0, 0, 0, 0,], (2, 1): [0, 0, 0, 0, 0], 
                                                (2, 2): [0, 0, 0, 0, 0], (3, 1): [0, 1, 0, 0, 0]}
target_predict_mono_all_predictions = np.array([target_predict_mono_all_predictions_dict[tuple(x)] for x in x_test])
target_predict_mono_vote_labels = np.array([target_predict_mono_vote_labels_dict[tuple(x)] for x in x_test])

# Instanciation of the mock mono_label models
mock_model_mono_1 = MockModel(dict_predictions_1, dict_predictions_proba_1, 'model_mono_1', False, ['0', '1'])
mock_model_mono_2 = MockModel(dict_predictions_2, dict_predictions_proba_2, 'model_mono_2', False, ['1', '0'])
mock_model_mono_3 = MockModel(dict_predictions_3, dict_predictions_proba_3, 'model_mono_3', False, ['1', '2', '4'])
mock_model_mono_4 = MockModel(dict_predictions_4, dict_predictions_proba_4, 'model_mono_4', False, ['2', '1', '3'])
mock_model_mono_5 = MockModel(dict_predictions_5, dict_predictions_proba_5, 'model_mono_5', False, ['1', '2', '3', '4'])
list_models_mono = [mock_model_mono_1, mock_model_mono_2, mock_model_mono_3, mock_model_mono_4, mock_model_mono_5]

# Instanciation of the mock multi_label models
mock_model_multi_1 = MockModel(dict_predictions_multi_1, dict_predictions_proba_1, 'model_multi_1', True, ['0', '1'])
mock_model_multi_2 = MockModel(dict_predictions_multi_2, dict_predictions_proba_2, 'model_multi_2', True, ['1', '0'])
mock_model_multi_3 = MockModel(dict_predictions_multi_3, dict_predictions_proba_3, 'model_multi_3', True, ['1', '2', '4'])
mock_model_multi_4 = MockModel(dict_predictions_multi_4, dict_predictions_proba_4, 'model_multi_4', True, ['2', '1', '3'])
list_models_multi = [mock_model_multi_1, mock_model_multi_2, mock_model_multi_3, mock_model_multi_4]

# Definitions for a mixture of mono/multi-labels models
list_model_mono_multi = [mock_model_mono_5, mock_model_multi_1, mock_model_multi_2, mock_model_mono_3]
target_predict_mono_multi_proba_dict = {(1, 2):[0.25, 0.375, 0.1375, 0.125, 0.1125],
                                (1, 3):[0.175, 0.45, 0.1875, 0.075, 0.1125], 
                                (2, 1):[0.2, 0.3875, 0.175, 0.025, 0.2125], 
                                (2, 2):[0.45, 0.175, 0.1375, 0.125, 0.1125], 
                                (3, 1):[0.05, 0.65, 0.1375 , 0.025, 0.1375]}
target_predict_mono_multi_all_predictions_dict = {(1, 2): [0, 1, 0, 1, 0], (1, 3): [0, 1, 1, 0, 0,], (2, 1): [0, 0, 1, 0, 1], 
                                                (2, 2): [1, 1, 0, 1, 0], (3, 1): [0, 1, 0, 0, 1]}
target_predict_mono_multi_vote_labels_dict = {(1, 2): [0, 1, 0, 0, 0], (1, 3): [0, 0, 0, 0, 0,], (2, 1): [0, 0, 0, 0, 0], 
                                                (2, 2): [0, 0, 0, 0, 0], (3, 1): [0, 0, 0, 0, 0]}
target_predict_mono_multi_all_predictions = np.array([target_predict_mono_multi_all_predictions_dict[tuple(x)] for x in x_test])
target_predict_mono_multi_vote_labels = np.array([target_predict_mono_multi_vote_labels_dict[tuple(x)] for x in x_test])
target_predict_mono_multi_proba = np.array([target_predict_mono_multi_proba_dict[tuple(x)] for x in x_test])

class ModelAggregationClassifierTests(unittest.TestCase):
    '''Main class to test model_aggregation_classifier'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    # Create and save a ModelGBTClassifier and a ModelSGDClassifier models
    def create_models(self, gbt_param: dict = None, sgd_param: dict = None):
        model_path = utils.get_models_path()
        model_dir_gbt = os.path.join(model_path, 'model_test_123456789_gbt')
        model_dir_sgd = os.path.join(model_path, 'model_test_123456789_sgd')
        remove_dir(model_dir_gbt)
        remove_dir(model_dir_sgd)

        gbt_param = {} if gbt_param is None else gbt_param
        sgd_param = {} if sgd_param is None else sgd_param

        gbt = ModelGBTClassifier(model_dir=model_dir_gbt, **gbt_param)
        sgd = ModelSGDClassifier(model_dir=model_dir_sgd, **sgd_param)

        gbt.save()
        sgd.save()
        gbt_name = os.path.split(gbt.model_dir)[-1]
        sgd_name = os.path.split(sgd.model_dir)[-1]
        return gbt, sgd, gbt_name, sgd_name

    def test01_model_aggregation_classifier_init(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.ModelAggregationClassifier.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model_name = 'test_model_name'

        x_train = pd.DataFrame.from_dict({'0':{'1': 1.2, '2':3.2}, '1':{'1': 1.1, '2':-3.2}}, orient='index')
        y_train = pd.Series(['coucou', 'coucou2'])
        def test_init_partial(model, model_name, model_dir, multi_label, using_proba, list_model_names, trained):
            self.assertTrue(os.path.isdir(model_dir))
            self.assertEqual(model.model_name, model_name)
            if multi_label:
                self.assertTrue(model.multi_label)
            else:
                self.assertFalse(model.multi_label)
            if using_proba:
                self.assertTrue(model.using_proba)
            else:
                self.assertFalse(model.using_proba)
            self.assertTrue(isinstance(model.sub_models, list))
            self.assertEqual([sub_model['name'] for sub_model in model.sub_models], list_model_names)
            self.assertTrue(isinstance(model.sub_models[0]['model'], ModelGBTClassifier))
            self.assertTrue(isinstance(model.sub_models[1]['model'], ModelSGDClassifier))
            self.assertEqual(model.trained, trained)

            # We test display_if_gpu_activated and _is_gpu_activated just by calling them
            model.display_if_gpu_activated()
            self.assertTrue(isinstance(model._is_gpu_activated(), bool))

        ############################################
        # Init., test all parameters
        ############################################

        # list_models = [model, model]
        # aggregation_function: proba_argmax
        # using_proba
        # not multi_label
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        gbt.fit(x_train, y_train)
        list_models = [gbt, sgd]
        list_model_names = [gbt_name, sgd_name]
        model = ModelAggregationClassifier(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=True, multi_label=False, aggregation_function='proba_argmax')
        test_init_partial(model, model_name, model_dir, False, True, list_model_names, trained=False)
        self.assertEqual(proba_argmax.__code__.co_code, model.aggregation_function.__code__.co_code)
        remove_dir_model(model, model_dir)

        # list_models = [model_name, model_name]
        # aggregation_function: majority_vote
        # not using_proba
        # not multi_label
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt_name, sgd_name]
        list_model_names = [gbt_name, sgd_name]
        model = ModelAggregationClassifier(model_dir=model_dir, model_name=model_name, list_models=list_models, aggregation_function='majority_vote')
        test_init_partial(model, model_name, model_dir, False, False, list_model_names, trained=False)
        self.assertEqual(majority_vote.__code__.co_code, model.aggregation_function.__code__.co_code)
        remove_dir_model(model, model_dir)

        # list_models = [model_name, model]
        # aggregation_function: all_predictions
        # not using_proba
        # multi_label
        gbt, sgd, gbt_name, sgd_name = self.create_models(gbt_param={'multi_label':True}, sgd_param={'multi_label':True})
        list_models = [gbt_name, sgd]
        list_model_names = [gbt_name, sgd_name]
        model = ModelAggregationClassifier(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=False, multi_label=True, aggregation_function='all_predictions')
        test_init_partial(model, model_name, model_dir, True, False, list_model_names, trained=False)
        self.assertEqual(all_predictions.__code__.co_code, model.aggregation_function.__code__.co_code)
        remove_dir_model(model, model_dir)

        # list_models = [model_name, model]
        # aggregation_function: vote_labels
        # not using_proba
        # multi_label
        gbt, sgd, gbt_name, sgd_name = self.create_models(gbt_param={'multi_label': True}, sgd_param={'multi_label': True})
        list_models = [gbt_name, sgd]
        list_model_names = [gbt_name, sgd_name]
        model = ModelAggregationClassifier(model_dir=model_dir, model_name=model_name, list_models=list_models, aggregation_function='vote_labels', multi_label=True)
        test_init_partial(model, model_name, model_dir, True, False, list_model_names, trained=False)
        self.assertEqual(vote_labels.__code__.co_code, model.aggregation_function.__code__.co_code)
        remove_dir_model(model, model_dir)

        # list_models = [model_name, model]
        # aggregation_function: Callable
        # not using_proba
        # not multi_label

        # This function is a copy of majority_vote function
        def function_test(predictions, **kwargs):
            '''Gives the class corresponding to the most present prediction in the given predictions. 
            In case of a tie, gives the prediction of the first model involved in the tie
            Args:
                predictions (np.ndarray) : The array containing the predictions of each model (shape (n_models)) 
            Returns:
                The prediction
            '''
            labels, counts = np.unique(predictions, return_counts=True)
            votes = [(label, count) for label, count in zip(labels, counts)]
            votes = sorted(votes, key=lambda x: x[1], reverse=True)
            possible_classes = {vote[0] for vote in votes if vote[1]==votes[0][1]}
            return [prediction for prediction in predictions if prediction in possible_classes][0]

        gbt, sgd, gbt_name, sgd_name = self.create_models()
        gbt.fit(x_train, y_train)
        gbt.save()
        sgd.fit(x_train, y_train)
        list_models = [gbt_name, sgd]
        list_model_names = [gbt_name, sgd_name]
        model = ModelAggregationClassifier(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=False, multi_label=False, aggregation_function=function_test)
        test_init_partial(model, model_name, model_dir, False, False, list_model_names, trained=True)
        self.assertEqual(function_test.__code__.co_code, model.aggregation_function.__code__.co_code)
        remove_dir_model(model, model_dir)

        ############################################
        # Error
        ############################################

        # if the object aggregation_function is a str but not found in the dictionary dict_aggregation_function
        with self.assertRaises(ValueError):
            model = ModelAggregationClassifier(model_dir=model_dir, aggregation_function='1234')
        remove_dir(model_dir)

        # if the object list_model has other model than model classifier (model_aggregation_classifier is only compatible with model classifier)
        gbt, _, _, _ = self.create_models()
        model_dir_regressor = os.path.join(utils.get_models_path(), 'model_test_123456789_regressor')
        sgd_regressor = ModelSGDRegressor(model_dir=model_dir_regressor)
        list_models = [gbt, sgd_regressor]
        with self.assertRaises(ValueError):
            model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models)
        remove_dir_model(model, model_dir)
        remove_dir(sgd_regressor.model_dir)
        
        # if 'multi_label' inconsistent with sub_models
        gbt, sgd, gbt_name, sgd_name = self.create_models(gbt_param={'multi_label':True}, sgd_param={'multi_label':False})
        list_models = [gbt, sgd]
        with self.assertRaises(ValueError):
            model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models, multi_label=False)
        remove_dir_model(model, model_dir)

        gbt, sgd, gbt_name, sgd_name = self.create_models(gbt_param={'multi_label':True}, sgd_param={'multi_label':True})
        list_models = [gbt, sgd]
        with self.assertRaises(ValueError):
            model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models, multi_label=False)
        remove_dir_model(model, model_dir)

        gbt, sgd, gbt_name, sgd_name = self.create_models(gbt_param={'multi_label':False}, sgd_param={'multi_label':False})
        list_models = [gbt, sgd]
        with self.assertRaises(ValueError):
            model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')
        remove_dir_model(model, model_dir)

        # if 'multi_label' inconsistent with aggregation_function
        gbt, sgd, gbt_name, sgd_name = self.create_models(gbt_param={'multi_label':True}, sgd_param={'multi_label':True})
        list_models = [gbt, sgd]
        with self.assertRaises(ValueError):
            model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models, multi_label=False, aggregation_function='all_predictions')
        remove_dir_model(model, model_dir)

        gbt, sgd, gbt_name, sgd_name = self.create_models(gbt_param={'multi_label':True}, sgd_param={'multi_label':True})
        list_models = [gbt, sgd]
        with self.assertRaises(ValueError):
            model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models, multi_label=False, aggregation_function='vote_label')
        remove_dir_model(model, model_dir)

        gbt, sgd, gbt_name, sgd_name = self.create_models(gbt_param={'multi_label':False}, sgd_param={'multi_label':False})
        list_models = [gbt, sgd]
        with self.assertRaises(ValueError):
            model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='majority_vote')
        remove_dir_model(model, model_dir)

        gbt, sgd, gbt_name, sgd_name = self.create_models(gbt_param={'multi_label':False}, sgd_param={'multi_label':False})
        list_models = [gbt, sgd]
        with self.assertRaises(ValueError):
            model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='proba_argmax')
        remove_dir_model(model, model_dir)

    def test02_model_aggregation_setter_aggregation_function(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.ModelAggregationClassifier.aggregation_function'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        def test_setter_aggregation_function(model, aggregation_function, using_proba, multi_label):
            self.assertEqual(model.aggregation_function.__code__.co_code, aggregation_function.__code__.co_code)
            self.assertEqual(model.using_proba, using_proba)
            self.assertEqual(model.multi_label, multi_label)

        # Mono label aggregation functions
        model = ModelAggregationClassifier(model_dir=model_dir, multi_label=False)
        model.aggregation_function = 'proba_argmax'
        test_setter_aggregation_function(model, proba_argmax, using_proba=True, multi_label=False)
        model.aggregation_function = 'majority_vote'
        test_setter_aggregation_function(model, majority_vote, using_proba=False, multi_label=False)
        with self.assertRaises(ValueError):
            model.aggregation_function = 'all_predictions'
        remove_dir(model_dir)

        # Multi labels aggregation functions
        model = ModelAggregationClassifier(model_dir=model_dir, aggregation_function='all_predictions', multi_label=True)
        model.aggregation_function = 'vote_labels'
        test_setter_aggregation_function(model, vote_labels, using_proba=False, multi_label=True)
        model.aggregation_function = 'all_predictions'
        test_setter_aggregation_function(model, all_predictions, using_proba=False, multi_label=True)
        with self.assertRaises(ValueError):
            model.aggregation_function = 'majority_vote'
        remove_dir(model_dir)

        # local aggregation function
        # This function is a copy of all_predictions function
        def function_test(predictions: np.ndarray, **kwargs) -> np.ndarray:
            '''Calculates the sum of the arrays along axis 0 casts it to bool and then to int.
            Expects a numpy array containing only zeroes and ones.
            When used as an aggregation function, keeps all the prediction of each model (multi-labels)

            Args:
                predictions (np.ndarray) : Array of shape : (n_models, n_classes)
            Return:
                np.ndarray: The prediction
            '''
            return np.sum(predictions, axis=0, dtype=bool).astype(int)

        model = ModelAggregationClassifier(model_dir=model_dir, aggregation_function='vote_labels', multi_label=True)
        model.aggregation_function = function_test
        test_setter_aggregation_function(model, function_test, using_proba=False, multi_label=True)
        remove_dir(model_dir)

        # error
        with self.assertRaises(ValueError):
            model = ModelAggregationClassifier(model_dir=model_dir, aggregation_function='toto', multi_label=True)
        remove_dir(model_dir)

    def test03_model_aggregation_classifier_sub_models(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.ModelAggregationClassifier.sub_models'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        def check_sub_models(sub_models, list_models_name):
            self.assertTrue(isinstance(sub_models[0]['model'], ModelGBTClassifier))
            self.assertTrue(isinstance(sub_models[1]['model'], ModelSGDClassifier))
            self.assertEqual(len(sub_models), len(list_models))
            self.assertEqual([sub_model['name'] for sub_model in sub_models], list_models_name)

        # list_models = [model, model]
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt, sgd]
        list_models_name = [gbt_name, sgd_name]
        model = ModelAggregationClassifier(model_dir=model_dir)
        model.sub_models = list_models
        check_sub_models(model.sub_models, list_models_name)
        remove_dir_model(model, model_dir)

        # list_models = [model_name, model_name]
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt_name, sgd_name]
        list_models_name = [gbt_name, sgd_name]
        model = ModelAggregationClassifier(model_dir=model_dir)
        model.sub_models = list_models
        check_sub_models(model.sub_models, list_models_name)
        remove_dir_model(model, model_dir)

        # list_models = [model_name, model]
        gbt, sgd, gbt_name, sgd_name = self.create_models()
        list_models = [gbt_name, sgd]
        list_models_name = [gbt_name, sgd_name]
        model = ModelAggregationClassifier(model_dir=model_dir)
        model.sub_models = list_models
        check_sub_models(model.sub_models, list_models_name)
        remove_dir_model(model, model_dir)

    def test04_model_aggregation_classifier_check_trained(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.ModelAggregationClassifier._check_trained'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        def check_empty(model):
            self.assertFalse(model.trained)
            self.assertTrue(model.list_classes == [])
            self.assertTrue(model.dict_classes == {})

        def check_explicit_empty(trained, list_classes, dict_classes):
            self.assertFalse(trained)
            self.assertTrue(list_classes == [])
            self.assertTrue(dict_classes == {})
        
        def check_not_empty(trained, list_classes, dict_classes, n_classes, list_class_target, dict_classes_target):
            self.assertTrue(trained)
            self.assertTrue(len(list_classes), n_classes)
            self.assertEqual(list_classes, list_class_target)
            self.assertEqual(dict_classes, dict_classes_target)

        # set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_str = pd.Series(['non', 'non', 'non', 'non', 'oui', 'oui', 'oui'] * 10)
        n_classes_str = 2
        list_classes_str = ['non', 'oui']
        dict_classes_str = {0: 'non', 1: 'oui'}

        # str
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationClassifier(model_dir=model_dir)
        check_empty(model)
        gbt.fit(x_train, y_train_str)
        sgd.fit(x_train, y_train_str)
        model.sub_models = [gbt, sgd]
        trained, list_classes, dict_classes = model._check_trained()
        check_not_empty(trained, list_classes, dict_classes, n_classes_str, list_classes_str, dict_classes_str)
        remove_dir_model(model, model_dir)

        # not trained
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationClassifier(model_dir=model_dir)
        check_empty(model)
        model.sub_models =[gbt, sgd]
        trained, list_classes, dict_classes = model._check_trained()
        check_explicit_empty(trained, list_classes, dict_classes)
        remove_dir_model(model, model_dir)

        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationClassifier(model_dir=model_dir)
        check_empty(model)
        sgd.fit(x_train, y_train_str)
        model.sub_models = [gbt, sgd]
        trained, list_classes, dict_classes = model._check_trained()
        check_explicit_empty(trained, list_classes, dict_classes)
        remove_dir_model(model, model_dir)

    def test05_model_aggregation_classifier_fit(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.ModelAggregationClassifier.fit'''

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

        ############################################
        # mono_label
        ############################################

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train_mono = pd.Series(['y1', 'y1', 'y1', 'y2', 'y3', 'y3', 'y3'] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})

        # Not trained
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=[gbt, sgd])
        check_not_trained(model)
        model.fit(x_train, y_train_mono)
        check_trained(model)
        remove_dir_model(model, model_dir)

        # Some model trained
        gbt, sgd, _, _ = self.create_models()
        gbt.fit(x_train, y_train_mono)
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=[gbt, sgd])
        check_not_trained(model)
        model.fit(x_train, y_train_mono)
        check_trained(model)
        remove_dir_model(model, model_dir)

        ############################################
        # multi_label
        ############################################

        gbt, sgd, _, _ = self.create_models(gbt_param={'multi_label': True}, sgd_param={'multi_label': True})

        model = ModelAggregationClassifier(model_dir=model_dir, list_models=[gbt, sgd], multi_label=True, aggregation_function='all_predictions')
        check_not_trained(model)
        model.fit(x_train, y_train_multi)
        check_trained(model)
        remove_dir_model(model, model_dir)

    def test06_model_aggregation_classifier_predict(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.ModelAggregationClassifier.predict'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # This function is a copy of majority_vote function
        def function_test(predictions, **kwargs):
            '''Gives the class corresponding to the most present prediction in the given predictions. 
            In case of a tie, gives the prediction of the first model involved in the tie
            Args:
                predictions (np.ndarray) : The array containing the predictions of each model (shape (n_models)) 
            Returns:
                The prediction
            '''
            labels, counts = np.unique(predictions, return_counts=True)
            votes = [(label, count) for label, count in zip(labels, counts)]
            votes = sorted(votes, key=lambda x: x[1], reverse=True)
            possible_classes = {vote[0] for vote in votes if vote[1]==votes[0][1]}
            return [prediction for prediction in predictions if prediction in possible_classes][0]

        # Function including all the checks we have to perform
        def test_predict(model, x_test, target_predict, target_probas):
            preds = model.predict(x_test)
            probas = model.predict(x_test, return_proba=True)
            self.assertEqual(preds.shape, target_predict.shape)
            self.assertEqual(probas.shape, target_probas.shape)
            # Check the predictions
            if model.multi_label:
                for truth, pred in zip(target_predict, preds):
                    for value_true, value_pred in zip(truth, pred):
                        self.assertEqual(value_true, value_pred)
            else:
                for truth, pred in zip(target_predict, preds):
                    self.assertEqual(truth, pred)
            # Check the probabilities
            for truth, pred in zip(target_probas, probas):
                for value_true, value_pred in zip(truth, pred):
                    self.assertAlmostEqual(value_true, value_pred)
        
        # majority_vote predictions
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models_mono, aggregation_function='majority_vote')
        test_predict(model, x_test, target_predict=target_predict_mono_majority_vote, target_probas=target_predict_mono_proba)
        remove_dir(model_dir)

        # proba_argmax predictions
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models_mono, aggregation_function='proba_argmax')
        test_predict(model, x_test, target_predict=target_predict_mono_proba_argmax, target_probas=target_predict_mono_proba)
        remove_dir(model_dir)

        # aggregation_function: Callable
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models_mono, aggregation_function=function_test, using_proba=False, multi_label=False)
        test_predict(model, x_test, target_predict=target_predict_mono_majority_vote, target_probas=target_predict_mono_proba)
        remove_dir(model_dir)

        # all_predictions predictions
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models_multi, aggregation_function='all_predictions', multi_label=True)
        test_predict(model, x_test, target_predict=target_predict_multi_all_predictions, target_probas=target_predict_multi_proba)
        remove_dir(model_dir)

        # vote_labels predictions
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models_multi, aggregation_function='vote_labels', multi_label=True)
        test_predict(model, x_test, target_predict=target_predict_multi_vote_labels, target_probas=target_predict_multi_proba)
        remove_dir(model_dir)

        ############################################
        # Error
        ############################################

        # Model needs to be fitted
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=[gbt, sgd])
        with self.assertRaises(AttributeError):
            model.predict('test')
        remove_dir_model(model, model_dir)

    def test07_model_aggregation_classifier_predict_probas_sub_models(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.ModelAggregationClassifier._predict_probas_sub_models'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models_mono)
        probas = model._predict_probas_sub_models(x_test)
        self.assertTrue(isinstance(probas, np.ndarray))
        self.assertEqual(target_get_proba_mono.shape, probas.shape)
        for i in range(len(x_test)):
            for truth, pred in zip(target_get_proba_mono[i], probas[i]):
                for truth_value, pred_value in zip(truth, pred):
                    self.assertAlmostEqual(truth_value, pred_value)
        remove_dir_model(model, model_dir)

        # Model needs to be fitted
        model = ModelAggregationClassifier(model_dir=model_dir)
        with self.assertRaises(AttributeError):
            model._predict_probas_sub_models('test')
        remove_dir(model_dir)

    def test08_model_aggregation_classifier__predict_sub_models(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.ModelAggregationClassifier._predict_sub_models'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # mono_label
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models_mono)
        preds = model._predict_sub_models(x_test)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(target_get_predictions_mono.shape, preds.shape)
        for i in range(len(x_test)):
            for truth, pred in zip(target_get_predictions_mono[i], preds[i]):
                self.assertEqual(truth, pred)
        remove_dir_model(model, model_dir)

        # multi_label
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models_multi, aggregation_function='all_predictions', multi_label=True)
        preds = model._predict_sub_models(x_test)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(target_get_predictions_multi.shape, preds.shape)
        for i in range(len(x_test)):
            for truth, pred in zip(target_get_predictions_multi[i], preds[i]):
                for truth_value, pred_value in zip(truth, pred):
                    self.assertEqual(truth_value, pred_value)
        remove_dir_model(model, model_dir)

        # Model needs to be fitted
        model = ModelAggregationClassifier(model_dir=model_dir)
        with self.assertRaises(AttributeError):
            model._predict_sub_models('test')
        remove_dir(model_dir)

    def test09_model_aggregation_classifier_predict_proba(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.ModelAggregationClassifier.predict_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models_mono)
        probas = model.predict_proba(x_test)
        self.assertEqual(target_predict_mono_proba.shape, probas.shape)
        for i in range(len(x_test)):
            for truth, pred in zip(target_predict_mono_proba[i], probas[i]):
                self.assertAlmostEqual(truth, pred)
        remove_dir_model(model, model_dir)

        # Model needs to be fitted
        model = ModelAggregationClassifier(model_dir=model_dir)
        with self.assertRaises(AttributeError):
            model.predict_proba('test')
        remove_dir(model_dir)

    def test10_model_aggregation_classifier_predict_full_list_classes(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.ModelAggregationClassifier._predict_full_list_classes'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        ############################################
        # mono_label
        ############################################
        model = ModelAggregationClassifier(model_dir=model_dir)
        model.list_classes = ['0', '1', '2', '3', '4', '5']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}

        mock_model_mono = MockModel({(1, 2):'2', (1, 3):'1', (2, 1):'4'}, 
                                    {(1, 2):np.array([0.2, 0.8, 0.0]), (1, 3):np.array([0.9, 0.0, 0.1]), (2, 1):np.array([0.2, 0.1, 0.7])}, 
                                    'model_mono', 
                                    False, 
                                    ['1', '2', '4'])

        # mono_label, no return_proba
        preds = model._predict_full_list_classes(mock_model_mono, np.array([(1, 2), (1, 3), (2, 1)]), return_proba=False)
        target_mono = np.array(['2', '1', '4'])
        self.assertEqual(target_mono.shape, preds.shape)
        for truth, pred in zip(target_mono, preds):
            self.assertEqual(truth, pred)

        # mono_label, return_proba
        preds = model._predict_full_list_classes(mock_model_mono, np.array([(1, 2), (1, 3), (2, 1)]), return_proba=True)
        target_mono_return_proba = np.array([[0.0, 0.2, 0.8, 0.0, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0, 0.1, 0.0], [0.0, 0.2, 0.1, 0.0, 0.7, 0.0]])
        self.assertEqual(target_mono_return_proba.shape, preds.shape)
        for truth, pred in zip(target_mono_return_proba, preds):
            for truth_value, pred_value in zip(truth, pred):
                self.assertAlmostEqual(truth_value, pred_value)

        remove_dir(model_dir)

        ############################################
        # multi_label
        ############################################

        model = ModelAggregationClassifier(model_dir=model_dir, aggregation_function='all_predictions', multi_label=True)
        model.list_classes = ['0', '1', '2', '3', '4', '5']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}

        mock_model_multi = MockModel({(1, 2):np.array([1, 1, 0]), (1, 3):np.array([1, 0, 1]), (2, 1):np.array([0, 0, 0])}, 
                                    {(1, 2):np.array([0.2, 0.8, 0.0]), (1, 3):np.array([0.9, 0.0, 0.1]), (2, 1):np.array([0.2, 0.1, 0.7])}, 
                                    'model_mono', 
                                    True, 
                                    ['1', '2', '4'])

        # multi_label, no return_proba
        preds = model._predict_full_list_classes(mock_model_multi, np.array([(1, 2), (1, 3), (2, 1)]), return_proba=False)
        target_multi = np.array([[0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]])
        self.assertEqual(target_multi.shape, preds.shape)
        for truth, pred in zip(target_multi, preds):
            for truth_value, pred_value in zip(truth, pred):
                self.assertEqual(truth_value, pred_value)

        # multi_label, return_proba
        preds = model._predict_full_list_classes(mock_model_multi, np.array([(1, 2), (1, 3), (2, 1)]), return_proba=True)
        target_mono_return_proba = np.array([[0.0, 0.2, 0.8, 0.0, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0, 0.1, 0.0], [0.0, 0.2, 0.1, 0.0, 0.7, 0.0]])
        self.assertEqual(target_mono_return_proba.shape, preds.shape)
        for truth, pred in zip(target_mono_return_proba, preds):
            for truth_value, pred_value in zip(truth, pred):
                self.assertAlmostEqual(truth_value, pred_value)

        remove_dir(model_dir)

    def test11_proba_argmax(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.proba_argmax'''

        list_classes = ['0', '1', '2', '3']

        # shape (3 models, 4 classes)
        probas = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.3, 0.4], [0.5, 0.1, 0.3, 0.1]])
        self.assertEqual(proba_argmax(probas, list_classes), '2')

        probas = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.3, 0.4], [0.5, 0.05, 0.3, 0.15]])
        self.assertEqual(proba_argmax(probas, list_classes), '3')

        probas = np.array([[0.1, 0.2, 0.3, 0.4], [0.7, 0.0, 0.3, 0.0], [0.5, 0.05, 0.3, 0.15]])
        self.assertEqual(proba_argmax(probas, list_classes), '0')

        probas = np.array([[0.1, 0.2, 0.3, 0.4], [0.0, 0.7, 0.3, 0.0], [0.5, 0.3, 0.05, 0.15]])
        self.assertEqual(proba_argmax(probas, list_classes), '1')

    def test12_majority_vote(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.majority_vote'''

        # normal case (4 models)
        preds = np.array(['a', 'b', 'b', 'c'])
        self.assertEqual(majority_vote(preds), 'b')
        # normal case (1 model)
        preds = np.array(['5'])
        self.assertEqual(majority_vote(preds), '5')
        # same predict (5 models)
        preds = np.array(['a', 'b', 'c', 'b', 'c'])
        self.assertEqual(majority_vote(preds), 'b')
        # same predict (5 models)
        preds = np.array(['b', 'a', 'c', 'b', 'c'])
        self.assertEqual(majority_vote(preds), 'b')

    def test13_all_predictions(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.all_predictions'''

        def test_all_predictions(preds, target_pred):
            actual_pred = all_predictions(preds)
            for truth, pred in zip(target_pred, actual_pred):
                self.assertEqual(truth, pred)

        # normal case
        # shape (3 models, 4 classes)
        preds = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]])
        target_pred = np.array([1, 1, 0, 1])
        test_all_predictions(preds, target_pred)

        # shape (3 models, 2 classes)
        preds = np.array([[0, 1], [1, 0], [0, 1]])
        target_pred = np.array([1, 1])
        test_all_predictions(preds, target_pred)

        # shape (1 model, 3 classes)
        preds = np.array([[1, 1, 0]])
        target_pred = np.array([1, 1, 0])
        test_all_predictions(preds, target_pred)

        # shape (3 models, 1 class)
        preds = np.array([[0], [0], [1]])
        target_pred = np.array([1])
        test_all_predictions(preds, target_pred)
        
        # shape (3 models, 1 class)
        preds = np.array([[0], [0], [0]])
        target_pred = np.array([0])
        test_all_predictions(preds, target_pred)

    def test14_vote_labels(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.vote_labels'''

        def test_vote_labels(preds, target_pred):
            actual_pred = vote_labels(preds)
            for truth, pred in zip(target_pred, actual_pred):
                self.assertEqual(truth, pred)

        # normal case
        # shape (3 models, 4 labels)
        preds = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1]])
        target_pred = np.array([1, 0, 0, 1])
        test_vote_labels(preds, target_pred)

        # shape (3 models, 2 labels)
        preds = np.array([[0, 1], [1, 0], [0, 1]])
        target_pred = np.array([0, 1])
        test_vote_labels(preds, target_pred)

        # shape (1 model, 3 labels)
        preds = np.array([[1, 1, 0]])
        target_pred = np.array([1, 1, 0])
        test_vote_labels(preds, target_pred)

        # shape (3 models, 1 label)
        preds = np.array([[0], [1], [1]])
        target_pred = np.array([1])
        test_vote_labels(preds, target_pred)

        # shape (3 models, 1 label)
        preds = np.array([[0], [1], [0]])
        target_pred = np.array([0])
        test_vote_labels(preds, target_pred)

        # same predict
        preds = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
        target_pred = np.array([0, 1])
        test_vote_labels(preds, target_pred)

    def test15_model_aggregation_classifier_save(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.ModelAggregationClassifier.save'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)

        gbt, sgd, gbt_name, sgd_name = self.create_models()
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train)
        model.save(json_data={'test': 10})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 10)
        set_attributes_in_config = {'package_version', 'model_name', 'model_dir', 'trained', 'nb_fit', 
                                    'list_classes', 'dict_classes', 'x_col', 'y_col', 'multi_label', 
                                    'level_save', 'librairie'}
        set_attributes_in_config_tot = set_attributes_in_config.union({'list_models_name', 'using_proba'})
        self.assertTrue(set_attributes_in_config_tot.issubset(set(configs.keys())))
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertEqual(configs['librairie'], None)
        for sub_model in model.sub_models:
            self.assertTrue(os.path.exists(os.path.join(sub_model['model'].model_dir, 'configurations.json')))
            self.assertTrue(os.path.exists(os.path.join(sub_model['model'].model_dir, f"{sub_model['model'].model_name}.pkl")))
            with open(os.path.join(sub_model['model'].model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
                configs = json.load(f)
            self.assertTrue(set_attributes_in_config.issubset(set(configs.keys())))
            self.assertTrue(configs['trained'])
            self.assertEqual(configs['package_version'], utils.get_package_version())
            remove_dir(os.path.join(models_path, os.path.split(sub_model['model'].model_dir)[-1]))

        self.assertTrue(os.path.exists(os.path.join(model_dir, "model_upload_instructions.md")))
        with open(os.path.join(model_dir, "model_upload_instructions.md"), 'r') as read_obj:
            text = read_obj.read()
            self.assertEqual(text[0:20], "/!\\/!\/!\\/!\/!\\   The aggregation model is a special model, please ensure that"[0:20])
        remove_dir(model_dir)

        # Same thing with a fitted model which is not saved before

        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)

        gbt, sgd, gbt_name, sgd_name = self.create_models()
        gbt.fit(x_train, y_train)
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train)
        model.save(json_data={'test': 10})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 10)
        set_attributes_in_config = {'package_version', 'model_name', 'model_dir', 'trained', 'nb_fit', 
                                    'list_classes', 'dict_classes', 'x_col', 'y_col', 'multi_label', 
                                    'level_save', 'librairie'}
        set_attributes_in_config_tot = set_attributes_in_config.union({'list_models_name', 'using_proba'})
        self.assertTrue(set_attributes_in_config_tot.issubset(set(configs.keys())))
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertEqual(configs['librairie'], None)
        for sub_model in model.sub_models:
            self.assertTrue(os.path.exists(os.path.join(sub_model['model'].model_dir, 'configurations.json')))
            self.assertTrue(os.path.exists(os.path.join(sub_model['model'].model_dir, f"{sub_model['model'].model_name}.pkl")))
            with open(os.path.join(sub_model['model'].model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
                configs = json.load(f)
            self.assertTrue(set_attributes_in_config.issubset(set(configs.keys())))
            self.assertTrue(configs['trained'])
            self.assertEqual(configs['package_version'], utils.get_package_version())
            remove_dir(os.path.join(models_path, os.path.split(sub_model['model'].model_dir)[-1]))

        self.assertTrue(os.path.exists(os.path.join(model_dir, "model_upload_instructions.md")))
        with open(os.path.join(model_dir, "model_upload_instructions.md"), 'r') as read_obj:
            text = read_obj.read()
            self.assertEqual(text[0:20], "/!\\/!\/!\\/!\/!\\   The aggregation model is a special model, please ensure that"[0:20])
        remove_dir(model_dir)


        # Same thing with a local function
        # This function is a copy of majority_vote function
        def function_test(predictions, **kwargs):
            '''Gives the class corresponding to the most present prediction in the given predictions. 
            In case of a tie, gives the prediction of the first model involved in the tie
            Args:
                predictions (np.ndarray) : The array containing the predictions of each model (shape (n_models)) 
            Returns:
                The prediction
            '''
            labels, counts = np.unique(predictions, return_counts=True)
            votes = [(label, count) for label, count in zip(labels, counts)]
            votes = sorted(votes, key=lambda x: x[1], reverse=True)
            possible_classes = {vote[0] for vote in votes if vote[1]==votes[0][1]}
            return [prediction for prediction in predictions if prediction in possible_classes][0]

        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        y_train = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)

        gbt, sgd, gbt_name, sgd_name = self.create_models()
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=[gbt, sgd], aggregation_function = function_test)
        model.fit(x_train, y_train)
        model.save(json_data={'test': 10})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 10)
        set_attributes_in_config = {'package_version', 'model_name', 'model_dir', 'trained', 'nb_fit', 
                                    'list_classes', 'dict_classes', 'x_col', 'y_col', 'multi_label', 
                                    'level_save', 'librairie'}
        set_attributes_in_config_sub_models = set_attributes_in_config.union({'aggregator_dir'})
        set_attributes_in_config_tot = set_attributes_in_config.union({'list_models_name', 'using_proba'})
        self.assertTrue(set_attributes_in_config_tot.issubset(set(configs.keys())))
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertEqual(configs['librairie'], None)
        for sub_model in model.sub_models:
            self.assertTrue(os.path.exists(os.path.join(sub_model['model'].model_dir, 'configurations.json')))
            self.assertTrue(os.path.exists(os.path.join(sub_model['model'].model_dir, f"{sub_model['model'].model_name}.pkl")))
            with open(os.path.join(sub_model['model'].model_dir, 'configurations.json'), 'r', encoding='utf-8') as f:
                configs = json.load(f)
            self.assertTrue(set_attributes_in_config_sub_models.issubset(set(configs.keys())))
            self.assertTrue(configs['trained'])
            self.assertEqual(configs['package_version'], utils.get_package_version())
            remove_dir(os.path.join(models_path, os.path.split(sub_model['model'].model_dir)[-1]))

        self.assertTrue(os.path.exists(os.path.join(model_dir, "model_upload_instructions.md")))
        with open(os.path.join(model_dir, "model_upload_instructions.md"), 'r') as read_obj:
            text = read_obj.read()
            self.assertEqual(text[0:20], "/!\\/!\/!\\/!\/!\\   The aggregation model is a special model, please ensure that"[0:20])
        remove_dir(model_dir)

    def test16_model_aggregation_classifier_prepend_line(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregationClassifier.prepend_line'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model = ModelAggregationClassifier(model_dir=model_dir)
        path = os.path.join(model_dir, 'test.md')
        with open(path, 'w') as f:
            f.write('toto')
        with open(path, 'r') as f:
            self.assertTrue(f.read() == 'toto')
        model.prepend_line(path, 'titi\n')
        with open(path, 'r') as f:
            self.assertTrue(f.read() == 'titi\ntoto')
        remove_dir(model_dir)

    def test17_model_aggregation_classifier_reload_from_standalone(self):
        '''Test of {{package_name}}.models_training.model_aggregation_classifier.ModelAggregationClassifier.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        model_new_dir = os.path.join(os.getcwd(), 'model_new_test_123456789')
        remove_dir(model_dir)

        #######################
        #  mono_label
        #######################

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, -1, 0, 4, 6, 3] * 10, 'col_2': [-8, -1, -1, 0, 4, 12, 2] * 10})
        x_test = pd.DataFrame({'col_1': [-5], 'col_2': [-8]})
        y_train_mono = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)

        # Create model
        gbt, sgd, _, _ = self.create_models()
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=[gbt, sgd])
        model.fit(x_train, y_train_mono)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"aggregation_function.pkl")))

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregationClassifier(model_dir=model_new_dir)
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path, preprocess_pipeline_path=preprocess_pipeline_path)

        # Test
        for attribute in ['trained', 'nb_fit', 'x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'using_proba']:
            self.assertEqual(getattr(model, attribute), getattr(model_new, attribute))

        for sub_model, new_sub_model in zip(model.sub_models, model_new.sub_models):
            self.assertTrue(isinstance(sub_model['model'], type(new_sub_model['model'])))
            self.assertEqual(sub_model['name'], new_sub_model['name'])
        self.assertEqual(model.aggregation_function.__code__.co_code, model_new.aggregation_function.__code__.co_code)

        preds = model.predict(x_test)
        preds_proba = model.predict_proba(x_test)
        new_preds = model_new.predict(x_test)
        new_preds_proba = model_new.predict_proba(x_test)
        for pred, new_pred in zip(preds, new_preds):
            self.assertEqual(pred, new_pred)
        for pred_proba, new_pred_proba in zip(preds_proba, new_preds_proba):
            for proba_value, new_proba_value in zip(pred_proba, new_pred_proba):
                self.assertAlmostEqual(proba_value, new_proba_value)

        for sub_model in model.sub_models:
            remove_dir(os.path.join(models_path, os.path.split(sub_model['model'].model_dir)[-1]))
        for sub_model in model_new.sub_models:
            remove_dir(os.path.join(models_path, os.path.split(sub_model['model'].model_dir)[-1]))
        remove_dir(model_dir)
        remove_dir(model_new_dir)

        #######################
        #  multi_label
        #######################

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, -1, 0, 4, 6, 3] * 10, 'col_2': [-8, -1, -1, 0, 4, 12, 2] * 10})
        x_test = pd.DataFrame({'col_1': [-5], 'col_2': [-8]})
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 1, 1, 1] * 10, 'test2': [0, 0, 0, 0, 1, 1, 1] * 10, 'test4': [1, 1, 1, 0, 0, 0, 0] * 10})
        y_train_mono = pd.Series(['test1', 'test1', 'test1', 'test3', 'test4', 'test0', 'test5'] * 10)

        # Create model
        gbt, sgd, gbt_name, sgd_name = self.create_models(gbt_param={'multi_label': True}, sgd_param={'multi_label': True})
        list_models = [gbt_name, sgd_name]
        model = ModelAggregationClassifier(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')
        model.fit(x_train, y_train_multi)
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        preprocess_pipeline_path = os.path.join(model.model_dir, "preprocess_pipeline.pkl")
        aggregation_function_path = os.path.join(model.model_dir, "aggregation_function.pkl")
        model_new = ModelAggregationClassifier(model_dir=model_new_dir)
        model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path, preprocess_pipeline_path=preprocess_pipeline_path)

        # Test
        for attribute in ['trained', 'nb_fit', 'x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'using_proba']:
            self.assertEqual(getattr(model, attribute), getattr(model_new, attribute))

        for sub_model, new_sub_model in zip(model.sub_models, model_new.sub_models):
            self.assertTrue(isinstance(sub_model['model'], type(new_sub_model['model'])))
            self.assertEqual(sub_model['name'], new_sub_model['name'])
        self.assertEqual(model.aggregation_function.__code__.co_code, model_new.aggregation_function.__code__.co_code)

        preds = model.predict(x_test)
        preds_proba = model.predict_proba(x_test)
        new_preds = model_new.predict(x_test)
        new_preds_proba = model_new.predict_proba(x_test)
        for pred, new_pred in zip(preds, new_preds):
            for pred_value, new_pred_value in zip(pred, new_pred):
                self.assertEqual(pred_value, new_pred_value)
        for pred_proba, new_pred_proba in zip(preds_proba, new_preds_proba):
            for proba_value, new_proba_value in zip(pred_proba, new_pred_proba):
                self.assertAlmostEqual(proba_value, new_proba_value)

        for sub_model in model.sub_models:
            remove_dir(os.path.join(models_path, os.path.split(sub_model['model'].model_dir)[-1]))
        for sub_model in model_new.sub_models:
            remove_dir(os.path.join(models_path, os.path.split(sub_model['model'].model_dir)[-1]))
        remove_dir(model_dir)
        remove_dir(model_new_dir)

        ############################################
        # Errors
        ############################################

        model_new = ModelAggregationClassifier(model_dir=model_new_dir)
        with self.assertRaises(FileNotFoundError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path='toto.json', aggregation_function_path=aggregation_function_path, preprocess_pipeline_path=preprocess_pipeline_path)
        remove_dir(model_new.model_dir)
        remove_dir(model_new_dir)
        model_new = ModelAggregationClassifier(model_dir=model_new_dir)
        with self.assertRaises(FileNotFoundError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path='toto.pkl', preprocess_pipeline_path=preprocess_pipeline_path)
        remove_dir(model_new.model_dir)
        remove_dir(model_new_dir)
        model_new = ModelAggregationClassifier(model_dir=model_new_dir)
        with self.assertRaises(FileNotFoundError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path, preprocess_pipeline_path='toto.pkl')
        remove_dir(model_new.model_dir)
        remove_dir(model_new_dir)
        model_new = ModelAggregationClassifier(model_dir=model_new_dir)
        with self.assertRaises(ValueError):
            model_new.reload_from_standalone(model_dir=model_dir, aggregation_function_path=aggregation_function_path, preprocess_pipeline_path=preprocess_pipeline_path)
        remove_dir(model_new.model_dir)
        remove_dir(model_new_dir)
        model_new = ModelAggregationClassifier(model_dir=model_new_dir)
        with self.assertRaises(ValueError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, preprocess_pipeline_path=preprocess_pipeline_path)
        remove_dir(model_new.model_dir)
        remove_dir(model_new_dir)
        model_new = ModelAggregationClassifier(model_dir=model_new_dir)
        with self.assertRaises(ValueError):
            model_new.reload_from_standalone(model_dir=model_dir, configuration_path=conf_path, aggregation_function_path=aggregation_function_path)
        remove_dir(model_new.model_dir)
        remove_dir(model_new_dir)

# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()