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
import dill as pickle

from {{package_name}} import utils
from {{package_name}}.models_training.models_sklearn.model_tfidf_svm import ModelTfidfSvm
from {{package_name}}.models_training.models_sklearn.model_tfidf_gbt import ModelTfidfGbt
from {{package_name}}.models_training.model_aggregation import ModelAggregation, proba_argmax, majority_vote, all_predictions, vote_labels
# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)

# The class to mock the submodels
class MockModel(object):

    def __init__(self, dict_predictions, dict_predictions_proba, model_name, multi_label, list_classes):
        self.dict_predictions = dict_predictions.copy()
        self.dict_predictions_proba = dict_predictions_proba.copy()
        self.trained = True
        self.nb_fit = 1
        self.list_classes = list_classes.copy()
        self.dict_classes = {i: col for i, col in enumerate(self.list_classes)}
        self.model_name = model_name
        self.model_dir = os.path.join('false_path', f'{model_name}')
        self.multi_label = multi_label

    def predict(self, x_test, return_proba = False, **kwargs):
        if return_proba:
            return self.predict_proba(x_test, **kwargs)
        else:
            return np.array([self.dict_predictions[x] for x in x_test])

    def predict_proba(self, x_test, **kwargs):
        return np.array([self.dict_predictions_proba[x] for x in x_test])

# Predictions for the mock mono_label models
dict_predictions_1 = {"ceci est un test": '0', "pas cela": '0', "cela non plus": '1', "ici test": '0', "là, rien!": '1'}
dict_predictions_2 = {"ceci est un test": '1', "pas cela": '1', "cela non plus": '0', "ici test": '0', "là, rien!": '1'}
dict_predictions_3 = {"ceci est un test": '1', "pas cela": '1', "cela non plus": '2', "ici test": '1', "là, rien!": '4'}
dict_predictions_4 = {"ceci est un test": '2', "pas cela": '2', "cela non plus": '3', "ici test": '3', "là, rien!": '1'}
dict_predictions_5 = {"ceci est un test": '3', "pas cela": '2', "cela non plus": '4', "ici test": '3', "là, rien!": '1'}
list_dict_predictions = [dict_predictions_1, dict_predictions_2, dict_predictions_3, dict_predictions_4, dict_predictions_5]

# Predictions for the mock multi_label models
dict_predictions_multi_1 = {"ceci est un test": np.array([0, 1]), "pas cela": np.array([0, 0]),
                            "cela non plus": np.array([0, 0]), "ici test": np.array([1, 0]), "là, rien!": np.array([0, 0])}
dict_predictions_multi_2 = {"ceci est un test": np.array([1, 0]), "pas cela": np.array([1, 0]), "cela non plus": np.array([0, 0]),
                            "ici test": np.array([0, 1]), "là, rien!": np.array([0, 0])}
dict_predictions_multi_3 = {"ceci est un test": np.array([1, 1, 0]), "pas cela": np.array([1, 0, 0]),
                            "cela non plus": np.array([0, 0, 0]), "ici test": np.array([1, 0, 0]), "là, rien!": np.array([0, 0, 0])}
dict_predictions_multi_4 = {"ceci est un test": np.array([1, 1, 1]), "pas cela": np.array([0, 1, 0]),
                            "cela non plus": np.array([0, 0, 0]), "ici test": np.array([0, 1, 0]), "là, rien!": np.array([0, 0, 0])}

# Probabilities for the mock models
dict_predictions_proba_1 = {"ceci est un test": np.array([0.7, 0.3]), "pas cela": np.array([0.6, 0.4]),
                            "cela non plus": np.array([0.2, 0.8]), "ici test": np.array([0.9, 0.1]), "là, rien!": np.array([0.1, 0.9])}
dict_predictions_proba_2 = {"ceci est un test": np.array([0.7, 0.3]), "pas cela": np.array([0.9, 0.1]),
                            "cela non plus": np.array([0.4, 0.6]), "ici test": np.array([0.1, 0.9]), "là, rien!": np.array([0.9, 0.1])}
dict_predictions_proba_3 = {"ceci est un test": np.array([0.4, 0.25, 0.35]), "pas cela": np.array([0.4, 0.25, 0.35]),
                            "cela non plus": np.array([0.25, 0.4, 0.35]), "ici test": np.array([0.4, 0.25, 0.35]), "là, rien!": np.array([0.3, 0.25, 0.45])}
dict_predictions_proba_4 = {"ceci est un test": np.array([0.5, 0.1, 0.4]), "pas cela": np.array([0.4, 0.25, 0.35]),
                            "cela non plus": np.array([0.3, 0.25, 0.45]), "ici test": np.array([0.4, 0.1, 0.5]), "là, rien!": np.array([0.25, 0.4, 0.35])}
dict_predictions_proba_5 = {"ceci est un test": np.array([0.1, 0.3, 0.5, 0.1]), "pas cela": np.array([0.1, 0.5, 0.3, 0.1]),
                            "cela non plus": np.array([0.1, 0.3, 0.1, 0.5]), "ici test": np.array([0.1, 0.3, 0.5, 0.1]), "là, rien!": np.array([0.5, 0.3, 0.1, 0.1])}

x_test = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])

# Definition of the targets for the mono_label cases
target_predict_mono_majority_vote_dict = {"ceci est un test":'1', "pas cela":'1', "cela non plus":'1', "ici test":'0', "là, rien!":'1'}
target_get_predictions_mono = np.array([[list_dict_predictions[i][key] for i in range(len(list_dict_predictions))] for key in x_test])
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
target_predict_mono_proba_dict = {"ceci est un test": [0.2 , 0.32, 0.21, 0.18, 0.09],
                                "pas cela": [0.14, 0.41, 0.23, 0.13, 0.09],
                                "cela non plus": [0.16, 0.36, 0.2 , 0.11, 0.17],
                                "ici test": [0.36, 0.16, 0.19, 0.2 , 0.09],
                                "là, rien!": [0.04, 0.6 , 0.16, 0.09, 0.11]}
target_predict_mono_majority_vote = np.array([target_predict_mono_majority_vote_dict[x] for x in x_test])
target_predict_mono_proba = np.array([target_predict_mono_proba_dict[x] for x in x_test])
target_predict_mono_proba_argmax = np.array([str(np.argmax(target_predict_mono_proba_dict[x])) for x in x_test])

# Definition of the targets for the multi_label cases
target_predict_multi_all_predictions_dict = {"ceci est un test": [0, 1, 1, 1, 0], "pas cela": [0, 1, 0, 0, 0,], "cela non plus": [0, 0, 0, 0, 0],
                                                "ici test": [1, 1, 0, 0, 0], "là, rien!": [0, 0, 0, 0, 0]}
target_predict_multi_vote_labels_dict = {"ceci est un test": [0, 1, 0, 0, 0], "pas cela": [0, 1, 0, 0, 0], "cela non plus": [0, 0, 0, 0, 0],
                                            "ici test": [1, 0, 0, 0, 0], "là, rien!": [0, 0, 0, 0, 0]}
target_predict_multi_proba_dict = {"ceci est un test":[0.25  , 0.375 , 0.1875, 0.1   , 0.0875],
                                "pas cela":[0.175, 0.4875, 0.1625, 0.0875, 0.0875],
                                "cela non plus":[0.2, 0.425, 0.175, 0.1125, 0.0875],
                                "ici test":[0.45, 0.175, 0.1625, 0.125 , 0.0875],
                                "là, rien!":[0.05, 0.625, 0.125, 0.0875, 0.1125]}
target_predict_multi_proba = np.array([target_predict_multi_proba_dict[x] for x in x_test])
target_predict_multi_all_predictions = np.array([target_predict_multi_all_predictions_dict[x] for x in x_test])
target_predict_multi_vote_labels = np.array([target_predict_multi_vote_labels_dict[x] for x in x_test])
target_get_predictions_multi = np.array([[[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 1, 0]],
                                         [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]],
                                         [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                                         [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]],
                                         [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])

# Definition of the targets for the multi_label cases where the submodels are mono_label
target_predict_mono_all_predictions_dict = {"ceci est un test": [1, 1, 1, 1, 0], "pas cela": [1, 1, 1, 0, 0,], "cela non plus": [1, 1, 1, 1, 1],
                                                "ici test": [1, 1, 0, 1, 0], "là, rien!": [0, 1, 0, 0, 1]}
target_predict_mono_vote_labels_dict = {"ceci est un test": [0, 0, 0, 0, 0], "pas cela": [0, 0, 0, 0, 0,], "cela non plus": [0, 0, 0, 0, 0],
                                                "ici test": [0, 0, 0, 0, 0], "là, rien!": [0, 1, 0, 0, 0]}
target_predict_mono_all_predictions = np.array([target_predict_mono_all_predictions_dict[x] for x in x_test])
target_predict_mono_vote_labels = np.array([target_predict_mono_vote_labels_dict[x] for x in x_test])

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
target_predict_mono_multi_proba_dict = {"ceci est un test":[0.25, 0.375, 0.1375, 0.125, 0.1125],
                                "pas cela":[0.175, 0.45, 0.1875, 0.075, 0.1125],
                                "cela non plus":[0.2, 0.3875, 0.175, 0.025, 0.2125],
                                "ici test":[0.45, 0.175, 0.1375, 0.125, 0.1125],
                                "là, rien!":[0.05, 0.65, 0.1375 , 0.025, 0.1375]}
target_predict_mono_multi_all_predictions_dict = {"ceci est un test": [0, 1, 0, 1, 0], "pas cela": [0, 1, 1, 0, 0,], "cela non plus": [0, 0, 1, 0, 1],
                                                "ici test": [1, 1, 0, 1, 0], "là, rien!": [0, 1, 0, 0, 1]}
target_predict_mono_multi_vote_labels_dict = {"ceci est un test": [0, 1, 0, 0, 0], "pas cela": [0, 0, 0, 0, 0,], "cela non plus": [0, 0, 0, 0, 0],
                                                "ici test": [0, 0, 0, 0, 0], "là, rien!": [0, 0, 0, 0, 0]}
target_predict_mono_multi_all_predictions = np.array([target_predict_mono_multi_all_predictions_dict[x] for x in x_test])
target_predict_mono_multi_vote_labels = np.array([target_predict_mono_multi_vote_labels_dict[x] for x in x_test])
target_predict_mono_multi_proba = np.array([target_predict_mono_multi_proba_dict[x] for x in x_test])


models_path = utils.get_models_path()


def remove_dir_model(model, model_dir):
    for sub_model in model.sub_models:
        remove_dir(os.path.join(models_path, os.path.split(sub_model['model'].model_dir)[-1]))
    remove_dir(model_dir)


class ModelAggregationTests(unittest.TestCase):
    '''Main class to test model_aggregation'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    # Create and save a ModelTfidfSvm model and a ModelTfidfGbt model
    def create_svm_gbt(self, svm_param=None, gbt_param=None):
        model_path = utils.get_models_path()
        model_dir_svm = os.path.join(model_path, 'model_test_123456789_svm')
        model_dir_gbt = os.path.join(model_path, 'model_test_123456789_gbt')
        remove_dir(model_dir_svm)
        remove_dir(model_dir_gbt)

        if svm_param is None:
            svm_param = {}
        if gbt_param is None:
            gbt_param = {}
        svm = ModelTfidfSvm(model_dir=model_dir_svm, **svm_param)
        gbt = ModelTfidfGbt(model_dir=model_dir_gbt, **gbt_param)

        svm.save()
        gbt.save()
        svm_name = os.path.split(svm.model_dir)[-1]
        gbt_name = os.path.split(gbt.model_dir)[-1]
        return svm, gbt, svm_name, gbt_name

    def test01_model_aggregation_init(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.__init__'''

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
            self.assertTrue(isinstance(model.sub_models[0]['model'], ModelTfidfSvm))
            self.assertTrue(isinstance(model.sub_models[1]['model'], ModelTfidfGbt))
            self.assertEqual(model.trained, trained)

            # We test display_if_gpu_activated and _is_gpu_activated just by calling them
            model.display_if_gpu_activated()
            self.assertTrue(isinstance(model._is_gpu_activated(), bool))

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model_name = 'test_model_name'

        ############################################
        # Init., test all parameters
        ############################################

        # list_models = [model, model]
        # aggregation_function: proba_argmax
        # using_proba
        # not multi_label
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm, gbt]
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=True, multi_label=False, aggregation_function='proba_argmax')
        test_init_partial(model=model, model_name=model_name, model_dir=model_dir, multi_label=False, using_proba=True,
                          list_model_names=[svm_name, gbt_name], trained=False)
        self.assertEqual(proba_argmax.__code__.co_code, model.aggregation_function.__code__.co_code)

        remove_dir_model(model, model_dir)

        # list_models = [model_name, model_name]
        # aggregation_function: majority_vote
        # not using_proba
        # not multi_label
        # Trained model
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt_name]
        svm.fit(np.array(['ma phrase', 'ta phrase']), np.array(['coucou', 'coucou2']))
        svm.save()
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=False, multi_label=False, aggregation_function='majority_vote')
        test_init_partial(model=model, model_name=model_name, model_dir=model_dir, multi_label=False, using_proba=False,
                               list_model_names=[svm_name, gbt_name], trained=False)
        self.assertEqual(majority_vote.__code__.co_code, model.aggregation_function.__code__.co_code)

        remove_dir_model(model, model_dir)

        # list_models = [model_name, model]
        # aggregation_function: all_predictions
        # not using_proba
        # multi_label
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt(svm_param={'multi_label':True}, gbt_param={'multi_label':True})
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=False, multi_label=True, aggregation_function='all_predictions')
        test_init_partial(model=model, model_name=model_name, model_dir=model_dir, multi_label=True, using_proba=False,
                               list_model_names=[svm_name, gbt_name], trained=False)
        self.assertEqual(all_predictions.__code__.co_code, model.aggregation_function.__code__.co_code)

        remove_dir_model(model, model_dir)

        # list_models = [model_name, model]
        # aggregation_function: vote_labels
        # not using_proba
        # multi_label
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt(svm_param={'multi_label':True}, gbt_param={'multi_label':True})
        list_models = [svm_name, gbt]
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, aggregation_function='vote_labels', multi_label=True)
        test_init_partial(model=model, model_name=model_name, model_dir=model_dir, multi_label=True, using_proba=False,
                               list_model_names=[svm_name, gbt_name], trained=False)
        self.assertEqual(vote_labels.__code__.co_code, model.aggregation_function.__code__.co_code)

        remove_dir_model(model, model_dir)

        # list_models = [model_name, model_name]
        # aggregation_function: Callable
        # not using_proba
        # not multi_label
        # Trained models

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

        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt_name]
        svm.fit(np.array(['ma phrase', 'ta phrase']), np.array(['coucou', 'coucou2']))
        svm.save()
        gbt.fit(np.array(['ma phrase', 'ta phrase']), np.array(['coucou', 'coucou2']))
        gbt.save()
        model = ModelAggregation(model_dir=model_dir, model_name=model_name, list_models=list_models, using_proba=False, multi_label=False, aggregation_function=function_test)
        test_init_partial(model=model, model_name=model_name, model_dir=model_dir, multi_label=False, using_proba=False,
                               list_model_names=[svm_name, gbt_name], trained=True)
        self.assertEqual(function_test.__code__.co_code, model.aggregation_function.__code__.co_code)

        remove_dir_model(model, model_dir)

        ############################################
        # Errors
        ############################################

        # if the object aggregation_function is a str but not found in the dictionary dict_aggregation_function
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, aggregation_function='1234')
        remove_dir(model_dir)

        # if 'multi_label' inconsistent with sub_models
        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label':True}, gbt_param={'multi_label':False})
        list_models = [svm, gbt]
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=False)
        remove_dir_model(model, model_dir)

        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label':True}, gbt_param={'multi_label':True})
        list_models = [svm, gbt]
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=False)
        remove_dir_model(model, model_dir)

        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label':False}, gbt_param={'multi_label':False})
        list_models = [svm, gbt]
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='all_predictions')
        remove_dir_model(model, model_dir)

        # if 'multi_label' inconsistent with aggregation_function
        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label':False}, gbt_param={'multi_label':False})
        list_models = [svm, gbt]
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=False, aggregation_function='all_predictions')
        remove_dir_model(model, model_dir)

        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label':False}, gbt_param={'multi_label':False})
        list_models = [svm, gbt]
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=False, aggregation_function='vote_label')
        remove_dir_model(model, model_dir)

        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label':True}, gbt_param={'multi_label':True})
        list_models = [svm, gbt]
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='majority_vote')
        remove_dir_model(model, model_dir)

        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label':True}, gbt_param={'multi_label':True})
        list_models = [svm, gbt]
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, list_models=list_models, multi_label=True, aggregation_function='proba_argmax')
        remove_dir_model(model, model_dir)

    def test02_model_aggregation_setter_aggregation_function(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.aggregation_function'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        def test_setter_aggregation_function(model, aggregation_function, using_proba, multi_label):
            self.assertEqual(model.aggregation_function.__code__.co_code, aggregation_function.__code__.co_code)
            self.assertEqual(model.using_proba, using_proba)
            self.assertEqual(model.multi_label, multi_label)

        # Mono label aggregation functions
        model = ModelAggregation(model_dir=model_dir, multi_label=False)
        model.aggregation_function = 'proba_argmax'
        test_setter_aggregation_function(model, proba_argmax, using_proba=True, multi_label=False)
        model.aggregation_function = 'majority_vote'
        test_setter_aggregation_function(model, majority_vote, using_proba=False, multi_label=False)
        with self.assertRaises(ValueError):
            model.aggregation_function = 'all_predictions'
        remove_dir(model_dir)

        # Multi labels aggregation functions
        model = ModelAggregation(model_dir=model_dir, aggregation_function='all_predictions', multi_label=True)
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

        model = ModelAggregation(model_dir=model_dir, aggregation_function='vote_labels', multi_label=True)
        model.aggregation_function = function_test
        test_setter_aggregation_function(model, function_test, using_proba=False, multi_label=True)
        remove_dir(model_dir)

        # error
        with self.assertRaises(ValueError):
            model = ModelAggregation(model_dir=model_dir, aggregation_function='toto', multi_label=True)
        remove_dir(model_dir)

    def test03_model_aggregation_setter_sub_models(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.sub_models'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        def check_sub_models(sub_models, list_models_name):
            self.assertTrue(isinstance(sub_models[0]['model'], ModelTfidfSvm))
            self.assertTrue(isinstance(sub_models[1]['model'], ModelTfidfGbt))
            self.assertEqual(len(sub_models), len(list_models))
            self.assertEqual([sub_model['name'] for sub_model in sub_models], list_models_name)

        # list_models = [model, model]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm, gbt]
        list_models_name = [svm_name, gbt_name]
        model = ModelAggregation(model_dir=model_dir)
        model.sub_models = list_models
        check_sub_models(model.sub_models, list_models_name)

        remove_dir_model(model, model_dir)

        # list_models = [model_name, model_name]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt_name]
        list_models_name = [svm_name, gbt_name]
        model = ModelAggregation(model_dir=model_dir)
        model.sub_models = list_models
        check_sub_models(model.sub_models, list_models_name)

        remove_dir_model(model, model_dir)

        # list_models = [model_name, model]
        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        list_models = [svm_name, gbt]
        list_models_name = [svm_name, gbt_name]
        model = ModelAggregation(model_dir=model_dir)
        model.sub_models = list_models
        check_sub_models(model.sub_models, list_models_name)

        remove_dir_model(model, model_dir)

    def test04_model_aggregation_check_trained(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._check_trained'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_str = np.array(['oui', 'non', 'oui', 'non', 'oui'])
        n_classes_str = 2
        list_classes_str = ['non', 'oui']
        dict_classes_str = {0: 'non', 1: 'oui'}

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

        # str
        svm, gbt, _, _ = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir)
        check_empty(model)
        svm.fit(x_train, y_train_str)
        gbt.fit(x_train, y_train_str)
        model.sub_models = [svm, gbt]
        trained, list_classes, dict_classes = model._check_trained()
        check_not_empty(trained, list_classes, dict_classes, n_classes_str, list_classes_str, dict_classes_str)
        remove_dir_model(model, model_dir)

        # not trained
        svm, gbt, _, _ = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir)
        check_empty(model)
        model.sub_models = [svm, gbt]
        trained, list_classes, dict_classes = model._check_trained()
        check_explicit_empty(trained, list_classes, dict_classes)
        remove_dir_model(model, model_dir)

        svm, gbt, _, _ = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir)
        check_empty(model)
        gbt.fit(x_train, y_train_str)
        model.sub_models = [svm, gbt]
        trained, list_classes, dict_classes = model._check_trained()
        check_explicit_empty(trained, list_classes, dict_classes)
        remove_dir_model(model, model_dir)

    def test05_model_aggregation_fit(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.fit'''

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
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array(['oui', 'non', 'oui', 'non', 'none'])
        y_train_multi = pd.DataFrame({'test1': [0, 1, 0, 1, 0], 'test2': [1, 0, 0, 0, 1], 'test3': [1, 0, 1, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        # not trained
        svm, gbt, _, _ = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir, list_models=[svm, gbt])
        check_not_trained(model)
        model.fit(x_train, y_train_mono)
        check_trained(model)
        remove_dir_model(model, model_dir)

        # some model trained
        svm, gbt, _, _ = self.create_svm_gbt()
        svm.fit(x_train, y_train_mono)
        model = ModelAggregation(model_dir=model_dir, list_models=[svm, gbt])
        check_not_trained(model)
        model.fit(x_train, y_train_mono)
        check_trained(model)
        remove_dir_model(model, model_dir)

        ############################################
        # multi_label
        ############################################

        svm, gbt, _, _ = self.create_svm_gbt(svm_param={'multi_label': True}, gbt_param={'multi_label': True})
        model = ModelAggregation(model_dir=model_dir, list_models=[svm, gbt], multi_label=True, aggregation_function='all_predictions')
        check_not_trained(model)
        model.fit(x_train, y_train_multi[cols])
        check_trained(model)
        remove_dir_model(model, model_dir)

    def test05_model_aggregation_predict(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.predict'''

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
        model = ModelAggregation(model_dir=model_dir, list_models=list_models_mono, aggregation_function='majority_vote')
        test_predict(model, x_test, target_predict=target_predict_mono_majority_vote, target_probas=target_predict_mono_proba)
        remove_dir(model_dir)

        # proba_argmax predictions
        model = ModelAggregation(model_dir=model_dir, list_models=list_models_mono, aggregation_function='proba_argmax')
        test_predict(model, x_test, target_predict=target_predict_mono_proba_argmax, target_probas=target_predict_mono_proba)
        remove_dir(model_dir)

        # aggregation_function: Callable
        model = ModelAggregation(model_dir=model_dir, list_models=list_models_mono, aggregation_function=function_test, using_proba=False, multi_label=False)
        test_predict(model, x_test, target_predict=target_predict_mono_majority_vote, target_probas=target_predict_mono_proba)
        remove_dir(model_dir)

        # all_predictions predictions (multi_label models only)
        model = ModelAggregation(model_dir=model_dir, list_models=list_models_multi, aggregation_function='all_predictions', multi_label=True)
        test_predict(model, x_test, target_predict=target_predict_multi_all_predictions, target_probas=target_predict_multi_proba)
        remove_dir(model_dir)

        # vote_labels predictions (multi_label models only)
        model = ModelAggregation(model_dir=model_dir, list_models=list_models_multi, aggregation_function='vote_labels', multi_label=True)
        test_predict(model, x_test, target_predict=target_predict_multi_vote_labels, target_probas=target_predict_multi_proba)
        remove_dir(model_dir)

        ############################################
        # Errors
        ############################################

        # Model needs to be fitted
        svm, gbt, _, _ = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir, list_models=[svm, gbt])
        with self.assertRaises(AttributeError):
            model.predict('test')
        remove_dir_model(model, model_dir)

    def test07_model_aggregation_predict_probas_sub_models(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._predict_probas_sub_models'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model = ModelAggregation(model_dir=model_dir, list_models=list_models_mono)
        probas = model._predict_probas_sub_models(x_test)
        self.assertTrue(isinstance(probas, np.ndarray))
        self.assertEqual(target_get_proba_mono.shape, probas.shape)
        for i in range(len(x_test)):
            for truth, pred in zip(target_get_proba_mono[i], probas[i]):
                for truth_value, pred_value in zip(truth, pred):
                    self.assertAlmostEqual(truth_value, pred_value)
        remove_dir_model(model, model_dir)

        # Model needs to be fitted
        model = ModelAggregation(model_dir=model_dir)
        with self.assertRaises(AttributeError):
            model._predict_probas_sub_models('test')
        remove_dir(model_dir)

    def test08_model_aggregation_predict_sub_models(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._predict_sub_models'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # mono_label
        model = ModelAggregation(model_dir=model_dir, list_models=list_models_mono)
        preds = model._predict_sub_models(x_test)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(target_get_predictions_mono.shape, preds.shape)
        for i in range(len(x_test)):
            for truth, pred in zip(target_get_predictions_mono[i], preds[i]):
                self.assertEqual(truth, pred)
        remove_dir_model(model, model_dir)

        # multi_label
        model = ModelAggregation(model_dir=model_dir, list_models=list_models_multi, aggregation_function='all_predictions', multi_label=True)
        preds = model._predict_sub_models(x_test)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(target_get_predictions_multi.shape, preds.shape)
        for i in range(len(x_test)):
            for truth, pred in zip(target_get_predictions_multi[i], preds[i]):
                for truth_value, pred_value in zip(truth, pred):
                    self.assertEqual(truth_value, pred_value)
        remove_dir_model(model, model_dir)

        # Model needs to be fitted
        model = ModelAggregation(model_dir=model_dir)
        with self.assertRaises(AttributeError):
            model._predict_sub_models('test')
        remove_dir(model_dir)

    def test09_model_aggregation_predict_proba(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.predict_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model = ModelAggregation(model_dir=model_dir, list_models=list_models_mono)
        probas = model.predict_proba(x_test)
        self.assertEqual(target_predict_mono_proba.shape, probas.shape)
        for i in range(len(x_test)):
            for truth, pred in zip(target_predict_mono_proba[i], probas[i]):
                self.assertAlmostEqual(truth, pred)
        remove_dir_model(model, model_dir)

        # Model needs to be fitted
        model = ModelAggregation(model_dir=model_dir)
        with self.assertRaises(AttributeError):
            model.predict_proba('test')
        remove_dir(model_dir)

    def test10_model_aggregation_predict_full_list_classes(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._predict_full_list_classes'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        ############################################
        # mono_label
        ############################################
        model = ModelAggregation(model_dir=model_dir)
        model.list_classes = ['0', '1', '2', '3', '4', '5']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}

        mock_model_mono = MockModel({'test_1':'2', 'test_2':'1', 'test_3':'4'},
                                    {'test_1':np.array([0.2, 0.8, 0.0]), 'test_2':np.array([0.9, 0.0, 0.1]), 'test_3':np.array([0.2, 0.1, 0.7])},
                                    'model_mono',
                                    False,
                                    ['1', '2', '4'])

        # mono_label, no return_proba
        preds = model._predict_full_list_classes(mock_model_mono, np.array(['test_1', 'test_2', 'test_3']), return_proba=False)
        target_mono = np.array(['2', '1', '4'])
        self.assertEqual(target_mono.shape, preds.shape)
        for truth, pred in zip(target_mono, preds):
            self.assertEqual(truth, pred)

        # mono_label, return_proba
        preds = model._predict_full_list_classes(mock_model_mono, np.array(['test_1', 'test_2', 'test_3']), return_proba=True)
        target_mono_return_proba = np.array([[0.0, 0.2, 0.8, 0.0, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0, 0.1, 0.0], [0.0, 0.2, 0.1, 0.0, 0.7, 0.0]])
        self.assertEqual(target_mono_return_proba.shape, preds.shape)
        for truth, pred in zip(target_mono_return_proba, preds):
            for truth_value, pred_value in zip(truth, pred):
                self.assertAlmostEqual(truth_value, pred_value)

        remove_dir(model_dir)

        ############################################
        # multi_label
        ############################################

        model = ModelAggregation(model_dir=model_dir, aggregation_function='all_predictions', multi_label=True)
        model.list_classes = ['0', '1', '2', '3', '4', '5']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}

        mock_model_multi = MockModel({'test_1':np.array([1, 1, 0]), 'test_2':np.array([1, 0, 1]), 'test_3':np.array([0, 0, 0])},
                                    {'test_1':np.array([0.2, 0.8, 0.0]), 'test_2':np.array([0.9, 0.0, 0.1]), 'test_3':np.array([0.2, 0.1, 0.7])},
                                    'model_mono',
                                    True,
                                    ['1', '2', '4'])

        # multi_label, no return_proba
        preds = model._predict_full_list_classes(mock_model_multi, np.array(['test_1', 'test_2', 'test_3']), return_proba=False)
        target_multi = np.array([[0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]])
        self.assertEqual(target_multi.shape, preds.shape)
        for truth, pred in zip(target_multi, preds):
            for truth_value, pred_value in zip(truth, pred):
                self.assertEqual(truth_value, pred_value)

        # multi_label, return_proba
        preds = model._predict_full_list_classes(mock_model_multi, np.array(['test_1', 'test_2', 'test_3']), return_proba=True)
        target_mono_return_proba = np.array([[0.0, 0.2, 0.8, 0.0, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0, 0.1, 0.0], [0.0, 0.2, 0.1, 0.0, 0.7, 0.0]])
        self.assertEqual(target_mono_return_proba.shape, preds.shape)
        for truth, pred in zip(target_mono_return_proba, preds):
            for truth_value, pred_value in zip(truth, pred):
                self.assertAlmostEqual(truth_value, pred_value)

        remove_dir(model_dir)

    def test11_proba_argmax(self):
        '''Test of {{package_name}}.models_training.model_aggregation.proba_argmax'''

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
        '''Test of {{package_name}}.models_training.model_aggregation.majority_vote'''

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
        '''Test of {{package_name}}.models_training.model_aggregation.all_predictions'''

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
        '''Test of {{package_name}}.models_training.model_aggregation.vote_labels'''

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

    def test15_model_aggregation_save(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation.save'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test"])
        y_train = ['test1', 'test2', 'test4', 'test1']

        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir, list_models=[svm, gbt])
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
            self.assertEqual(text[0:20], "/!\\/!\\/!\\/!\\/!\\   The aggregation model is a special model, please ensure that"[0:20])
        remove_dir(model_dir)

        # Same thing with a fitted model which is not saved before

        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test"])
        y_train = ['test1', 'test2', 'test4', 'test1']

        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        svm.fit(x_train, y_train)
        model = ModelAggregation(model_dir=model_dir, list_models=[svm, gbt])
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
            self.assertEqual(text[0:20], "/!\\/!\\/!\\/!\\/!\\   The aggregation model is a special model, please ensure that"[0:20])
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

        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test"])
        y_train = ['test1', 'test2', 'test4', 'test1']

        svm, gbt, svm_name, gbt_name = self.create_svm_gbt()
        model = ModelAggregation(model_dir=model_dir, list_models=[svm, gbt], aggregation_function = function_test)
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
            self.assertEqual(text[0:20], "/!\\/!\\/!\\/!\\/!\\   The aggregation model is a special model, please ensure that"[0:20])
        remove_dir(model_dir)

    def test16_model_aggregation_prepend_line(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._prepend_line'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        model = ModelAggregation(model_dir=model_dir)
        path = os.path.join(model_dir, 'test.md')
        with open(path, 'w') as f:
            f.write('toto')
        with open(path, 'r') as f:
            self.assertTrue(f.read() == 'toto')
        model._prepend_line(path, 'titi\n')
        with open(path, 'r') as f:
            self.assertTrue(f.read() == 'titi\ntoto')
        remove_dir(model_dir)

    def test17_model_aggregation_hook_post_load_model_pkl(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._hook_post_load_model_pkl'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Basic case with several different aggregation function
        for multi_label, aggregation_function in [(False, 'majority_vote'), (False, 'proba_argmax'),
                                                  (True, 'all_predictions'), (True, 'vote_labels')]:
            sub_model_1 = ModelTfidfSvm(multi_label=multi_label)
            sub_model_2 = ModelTfidfSvm(multi_label=multi_label)
            model = ModelAggregation(list_models=[sub_model_1, sub_model_2], model_dir=model_dir,
                                     multi_label=multi_label, aggregation_function=aggregation_function)
            model.save()
            with open(os.path.join(model_dir, f'{model.model_name}.pkl'), 'rb') as f:
                new_model = pickle.load(f)
            self.assertTrue(new_model.aggregation_function is None)
            self.assertTrue(new_model.sub_models is None)
            new_model._hook_post_load_model_pkl()
            self.assertEqual(new_model.aggregation_function.__name__, model.aggregation_function.__name__)
            self.assertEqual([sub_model['name'] for sub_model in new_model.sub_models],
                             [os.path.split(sub_model_1.model_dir)[-1], os.path.split(sub_model_2.model_dir)[-1]])
            remove_dir(model_dir)
            remove_dir(sub_model_1.model_dir)
            remove_dir(sub_model_2.model_dir)

        # Basic case with a custom aggregation function
        def test_aggregation_function(predictions, **kwargs):
            return 'coucou'
        sub_model_1 = ModelTfidfSvm()
        sub_model_2 = ModelTfidfSvm()
        model = ModelAggregation(list_models=[sub_model_1, sub_model_2], model_dir=model_dir,
                                 aggregation_function=test_aggregation_function)
        model.save()
        with open(os.path.join(model_dir, f'{model.model_name}.pkl'), 'rb') as f:
            new_model = pickle.load(f)
        self.assertTrue(new_model.aggregation_function is None)
        self.assertTrue(new_model.sub_models is None)
        new_model._hook_post_load_model_pkl()
        self.assertEqual(new_model.aggregation_function(''), 'coucou')
        self.assertEqual([sub_model['name'] for sub_model in new_model.sub_models],
                             [os.path.split(sub_model_1.model_dir)[-1], os.path.split(sub_model_2.model_dir)[-1]])
        remove_dir(model_dir)
        remove_dir(sub_model_1.model_dir)
        remove_dir(sub_model_2.model_dir)

        # Errors
        model = ModelAggregation(model_dir=model_dir)
        model.save()
        os.remove(os.path.join(model_dir, 'aggregation_function.pkl'))
        with open(os.path.join(model_dir, f'{model.model_name}.pkl'), 'rb') as f:
            new_model = pickle.load(f)
        with self.assertRaises(FileNotFoundError):
            new_model._hook_post_load_model_pkl()
        remove_dir(model_dir)  

        model = ModelAggregation(model_dir=model_dir)
        model.save()
        os.remove(os.path.join(model_dir, 'configurations.json'))
        with open(os.path.join(model_dir, f'{model.model_name}.pkl'), 'rb') as f:
            new_model = pickle.load(f)
        with self.assertRaises(FileNotFoundError):
            new_model._hook_post_load_model_pkl()
        remove_dir(model_dir)  


    def test18_model_aggregation_init_new_instance_from_configs(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._init_new_instance_from_configs'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        sub_model_1 = ModelTfidfSvm()
        sub_model_2 = ModelTfidfSvm()
        model = ModelAggregation( list_models=[sub_model_1, sub_model_2], model_dir=model_dir)
        model.save(json_data={'test': 8})
        configs = model.load_configs(model_dir=model_dir)
        new_model = ModelAggregation._init_new_instance_from_configs(configs=configs)
        self.assertTrue(isinstance(new_model, ModelAggregation))
        self.assertEqual(new_model.nb_fit, 0)
        self.assertFalse(new_model.trained)
        for attribute in ['x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save', 'using_proba']:
            self.assertEqual(getattr(model, attribute), getattr(new_model, attribute))
        self.assertEqual([sub_model['name'] for sub_model in new_model.sub_models],
                             [os.path.split(sub_model_1.model_dir)[-1], os.path.split(sub_model_2.model_dir)[-1]])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)
        remove_dir(sub_model_1.model_dir)
        remove_dir(sub_model_2.model_dir)

        # Check by changing some attributes
        sub_model_1 = ModelTfidfSvm()
        sub_model_1.save()
        sub_model_2 = ModelTfidfSvm()
        sub_model_2.save()
        model = ModelAggregation(list_models=[os.path.split(sub_model_1.model_dir)[-1], os.path.split(sub_model_2.model_dir)[-1]],
                                 model_dir=model_dir, using_proba=True, aggregation_function='proba_argmax')
        model.nb_fit = 2
        model.trained = True
        model.x_col = 'coucou'
        model.y_col = 'coucou_2'
        model.list_classes = ['class_1', 'class_2', 'class_3']
        model.dict_classes = {0: 'class_1', 1: 'class_2', 2: 'class_3'}
        model.multi_label = True
        model.level_save = 'MEDIUM'
        model.save(json_data={'test': 8})
        configs = model.load_configs(model_dir=model_dir)
        new_model = ModelAggregation._init_new_instance_from_configs(configs=configs)
        self.assertTrue(isinstance(new_model, ModelAggregation))
        self.assertEqual(new_model.nb_fit, 2)
        self.assertTrue(new_model.trained)
        for attribute in ['x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save', 'using_proba']:
            self.assertEqual(getattr(model, attribute), getattr(new_model, attribute))
        self.assertEqual([sub_model['name'] for sub_model in new_model.sub_models],
                             [os.path.split(sub_model_1.model_dir)[-1], os.path.split(sub_model_2.model_dir)[-1]])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)
        remove_dir(sub_model_1.model_dir)
        remove_dir(sub_model_2.model_dir)

    def test19_model_aggregation_load_standalone_files(self):
        '''Test of {{package_name}}.models_training.model_aggregation.ModelAggregation._load_standalone_files'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Test for a registered aggregation_function with default_model_dir
        for multi_label, aggregation_function in [(False, 'majority_vote'), (False, 'proba_argmax'),
                                                  (True, 'all_predictions'), (True, 'vote_labels')]:
            model = ModelAggregation(model_dir=model_dir, multi_label=multi_label,
                                     aggregation_function=aggregation_function)
            model.save()

            configs = ModelAggregation.load_configs(model_dir=model_dir)
            new_model = ModelAggregation._init_new_instance_from_configs(configs)
            self.assertEqual(new_model.aggregation_function.__name__, 'majority_vote')
            new_model._load_standalone_files(default_model_dir=model_dir)
            self.assertEqual(new_model.aggregation_function.__name__, aggregation_function)
            remove_dir(model_dir)
            remove_dir(new_model.model_dir)

        # Test for a custom aggregation_function with default_model_dir
        def test_aggregation_function(predictions, **kwargs):
            return 'coucou'
        model = ModelAggregation(model_dir=model_dir, aggregation_function=test_aggregation_function)
        model.save()

        configs = ModelAggregation.load_configs(model_dir=model_dir)
        new_model = ModelAggregation._init_new_instance_from_configs(configs)
        self.assertEqual(new_model.aggregation_function.__name__, 'majority_vote')
        new_model._load_standalone_files(default_model_dir=model_dir)
        self.assertEqual(new_model.aggregation_function([]), 'coucou')
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)


        aggregation_function_path = os.path.join(model_dir, 'aggregation_function.pkl')
        # Test for a registered aggregation_function with aggregation_function path
        for multi_label, aggregation_function in [(False, 'majority_vote'), (False, 'proba_argmax'),
                                                  (True, 'all_predictions'), (True, 'vote_labels')]:
            model = ModelAggregation(model_dir=model_dir, multi_label=multi_label,
                                     aggregation_function=aggregation_function)
            model.save()

            configs = ModelAggregation.load_configs(model_dir=model_dir)
            new_model = ModelAggregation._init_new_instance_from_configs(configs)
            self.assertEqual(new_model.aggregation_function.__name__, 'majority_vote')
            new_model._load_standalone_files(aggregation_function_path=aggregation_function_path)
            self.assertEqual(new_model.aggregation_function.__name__, aggregation_function)
            remove_dir(model_dir)
            remove_dir(new_model.model_dir)

        # Test for a custom aggregation_function with aggregation_function path
        def test_aggregation_function(predictions, **kwargs):
            return 'coucou'
        model = ModelAggregation(model_dir=model_dir, aggregation_function=test_aggregation_function)
        model.save()

        configs = ModelAggregation.load_configs(model_dir=model_dir)
        new_model = ModelAggregation._init_new_instance_from_configs(configs)
        self.assertEqual(new_model.aggregation_function.__name__, 'majority_vote')
        new_model._load_standalone_files(aggregation_function_path=aggregation_function_path)
        self.assertEqual(new_model.aggregation_function([]), 'coucou')
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Errors
        model = ModelAggregation(model_dir=model_dir, aggregation_function=test_aggregation_function)
        model.save()
        configs = ModelAggregation.load_configs(model_dir=model_dir)
        new_model = ModelAggregation._init_new_instance_from_configs(configs)
        with self.assertRaises(ValueError):
            new_model._load_standalone_files()
        with self.assertRaises(FileNotFoundError):
            new_model._load_standalone_files(aggregation_function_path=os.path.join(model_dir, 'aggregation_function_2.pkl'))
        os.remove(os.path.join(model_dir, 'aggregation_function.pkl'))
        with self.assertRaises(FileNotFoundError):
            new_model._load_standalone_files(default_model_dir=model_dir)
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
