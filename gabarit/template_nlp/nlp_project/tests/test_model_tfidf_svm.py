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
from {{package_name}}.models_training.model_tfidf_svm import ModelTfidfSvm

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelTfidfSvmTests(unittest.TestCase):
    '''Main class to test model_tfidf_svm'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_tfidf_svm_init(self):
        '''Test of {{package_name}}.models_training.model_tfidf_svm.ModelTfidfSvm.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        model = ModelTfidfSvm(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertFalse(model.pipeline is None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        # Check TFIDF params
        model = ModelTfidfSvm(model_dir=model_dir, tfidf_params={'analyzer': 'char', 'binary': True})
        self.assertEqual(model.pipeline['tfidf'].analyzer, 'char')
        self.assertEqual(model.pipeline['tfidf'].binary, True)
        remove_dir(model_dir)

        # Check SVC params - mono-label
        model = ModelTfidfSvm(model_dir=model_dir, svc_params={'penalty': 'l1', 'fit_intercept': False}, multi_label=False, multiclass_strategy=None)
        self.assertEqual(model.pipeline['svc'].penalty, 'l1')
        self.assertEqual(model.pipeline['svc'].fit_intercept, False)
        remove_dir(model_dir)
        model = ModelTfidfSvm(model_dir=model_dir, svc_params={'penalty': 'l1', 'fit_intercept': False}, multi_label=False, multiclass_strategy='ovr')
        self.assertEqual(model.pipeline['svc'].estimator.penalty, 'l1')
        self.assertEqual(model.pipeline['svc'].estimator.fit_intercept, False)
        remove_dir(model_dir)
        model = ModelTfidfSvm(model_dir=model_dir, svc_params={'penalty': 'l1', 'fit_intercept': False}, multi_label=False, multiclass_strategy='ovo')
        self.assertEqual(model.pipeline['svc'].estimator.penalty, 'l1')
        self.assertEqual(model.pipeline['svc'].estimator.fit_intercept, False)
        remove_dir(model_dir)

        # Check SVC params - multi-labels
        model = ModelTfidfSvm(model_dir=model_dir, svc_params={'penalty': 'l1', 'fit_intercept': False}, multi_label=True, multiclass_strategy=None)
        self.assertEqual(model.pipeline['svc'].estimator.penalty, 'l1')
        self.assertEqual(model.pipeline['svc'].estimator.fit_intercept, False)
        self.assertEqual(model.multi_label, True)
        remove_dir(model_dir)
        model = ModelTfidfSvm(model_dir=model_dir, svc_params={'penalty': 'l1', 'fit_intercept': False}, multi_label=True, multiclass_strategy='ovr')
        self.assertEqual(model.pipeline['svc'].estimator.penalty, 'l1')
        self.assertEqual(model.pipeline['svc'].estimator.fit_intercept, False)
        self.assertEqual(model.multi_label, True)
        remove_dir(model_dir)
        model = ModelTfidfSvm(model_dir=model_dir, svc_params={'penalty': 'l1', 'fit_intercept': False}, multi_label=True, multiclass_strategy='ovo')
        self.assertEqual(model.pipeline['svc'].estimator.penalty, 'l1')
        self.assertEqual(model.pipeline['svc'].estimator.fit_intercept, False)
        self.assertEqual(model.multi_label, True)
        remove_dir(model_dir)

        # Error
        with self.assertRaises(ValueError):
            model = ModelTfidfSvm(model_dir=model_dir, svc_params={'penalty': 'l1', 'fit_intercept': False}, multi_label=False, multiclass_strategy='toto')
        remove_dir(model_dir)

    def test02_model_tfidf_svm_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.model_tfidf_svm.ModelTfidfSvm'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        n_classes = 3
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        # Mono-label - no strategy
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
        model.fit(x_train, y_train_mono)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        preds = model.predict('test', return_proba=False)
        self.assertEqual(preds, model.predict(['test'], return_proba=False)[0])
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), n_classes))
        proba = model.predict('test', return_proba=True)
        self.assertEqual([elem for elem in proba], [elem for elem in model.predict(['test'], return_proba=True)[0]])
        remove_dir(model_dir)

        # Mono-label - with strategy 'ovr'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy='ovr')
        model.fit(x_train, y_train_mono)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        preds = model.predict('test', return_proba=False)
        self.assertEqual(preds, model.predict(['test'], return_proba=False)[0])
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), n_classes))
        proba = model.predict('test', return_proba=True)
        self.assertEqual([elem for elem in proba], [elem for elem in model.predict(['test'], return_proba=True)[0]])
        remove_dir(model_dir)

        # Mono-label - with strategy 'ovo'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy='ovo')
        model.fit(x_train, y_train_mono)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        preds = model.predict('test', return_proba=False)
        self.assertEqual(preds, model.predict(['test'], return_proba=False)[0])
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), n_classes))
        proba = model.predict('test', return_proba=True)
        self.assertEqual([elem for elem in proba], [elem for elem in model.predict(['test'], return_proba=True)[0]])
        remove_dir(model_dir)

        # Multi-labels - no strategy
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True, multiclass_strategy=None)
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict('test', return_proba=False)
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict(['test'], return_proba=False)[0]])
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), len(cols)))
        proba = model.predict('test', return_proba=True)
        self.assertEqual([elem for elem in proba], [elem for elem in model.predict(['test'], return_proba=True)[0]])
        remove_dir(model_dir)

        # Multi-labels - with strategy 'ovr'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True, multiclass_strategy='ovr')
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict('test', return_proba=False)
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict(['test'], return_proba=False)[0]])
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), len(cols)))
        proba = model.predict('test', return_proba=True)
        self.assertEqual([elem for elem in proba], [elem for elem in model.predict(['test'], return_proba=True)[0]])
        remove_dir(model_dir)

        # Multi-labels - with strategy 'ovo'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True, multiclass_strategy='ovo')
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict('test', return_proba=False)
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict(['test'], return_proba=False)[0]])
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), len(cols)))
        proba = model.predict('test', return_proba=True)
        self.assertEqual([elem for elem in proba], [elem for elem in model.predict(['test'], return_proba=True)[0]])
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test03_model_tfidf_svm_predict_proba(self):
        '''Test of {{package_name}}.models_training.model_tfidf_svm.ModelTfidfSvm.predict_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        n_classes = 3
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        # Mono-label - no strategy
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), n_classes))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Mono-label - with strategy 'ovr'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy='ovr')
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), n_classes))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Mono-label - with strategy 'ovo'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy='ovo')
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), n_classes))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Multi-labels - no strategy
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True, multiclass_strategy=None)
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Multi-labels - with strategy 'ovr'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True, multiclass_strategy='ovr')
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Multi-labels - with strategy 'ovo'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True, multiclass_strategy='ovo')
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

    def test04_model_tfidf_svm_decision_function(self):
        '''Test of {{package_name}}.models_training.model_tfidf_svm.ModelTfidfSvm.decision_function'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        n_classes = 3
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        # Mono-label - no strategy
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
        model.fit(x_train, y_train_mono)
        preds = model.decision_function(x_train)
        self.assertEqual(preds.shape, (len(x_train), n_classes))
        preds = model.decision_function('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.decision_function(['test'])[0]])
        remove_dir(model_dir)

        # Mono-label - with strategy 'ovr'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy='ovr')
        model.fit(x_train, y_train_mono)
        preds = model.decision_function(x_train)
        self.assertEqual(preds.shape, (len(x_train), n_classes))
        preds = model.decision_function('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.decision_function(['test'])[0]])
        remove_dir(model_dir)

        # Mono-label - with strategy 'ovo'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy='ovo')
        model.fit(x_train, y_train_mono)
        preds = model.decision_function(x_train)
        self.assertEqual(preds.shape, (len(x_train), n_classes))
        preds = model.decision_function('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.decision_function(['test'])[0]])
        remove_dir(model_dir)

        # Multi-labels - do not support decision_function
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True, multiclass_strategy=None)
        model.fit(x_train, y_train_multi[cols])
        with self.assertRaises(ValueError):
            model.decision_function(x_train)
        remove_dir(model_dir)

    def test05_model_tfidf_svm_get_predict_position(self):
        '''Test of the method {{package_name}}.models_training.model_tfidf_svm.ModelTfidfSvm.get_predict_position'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!", "coucou"])
        y_train_mono = np.array([0, 1, 0, 1, 0, 2])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0, 0], 'test2': [1, 0, 0, 0, 0, 1], 'test3': [0, 0, 0, 1, 0, 1]})
        cols = ['test1', 'test2', 'test3']

        # Mono-label - no strategy
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
        model.fit(x_train, y_train_mono)
        predict_positions = model.get_predict_position(x_train, y_train_mono)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)

        # Mono-label - with strategy 'ovr'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy='ovr')
        model.fit(x_train, y_train_mono)
        predict_positions = model.get_predict_position(x_train, y_train_mono)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)

        # Mono-label - with strategy 'ovo'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy='ovo')
        model.fit(x_train, y_train_mono)
        predict_positions = model.get_predict_position(x_train, y_train_mono)
        self.assertEqual(predict_positions.shape, (len(x_train),))
        remove_dir(model_dir)

        # Multi-labels - no strategy
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True, multiclass_strategy=None)
        model.fit(x_train, y_train_multi[cols])
        # Unavailable in multi-labels
        with self.assertRaises(ValueError):
            model.get_predict_position(x_train, y_train_multi[cols])
        remove_dir(model_dir)

        # Multi-labels - with strategy 'ovr'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True, multiclass_strategy='ovr')
        model.fit(x_train, y_train_multi[cols])
        # Unavailable in multi-labels
        with self.assertRaises(ValueError):
            model.get_predict_position(x_train, y_train_multi[cols])
        remove_dir(model_dir)

        # Multi-labels - with strategy 'ovo'
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True, multiclass_strategy='ovo')
        model.fit(x_train, y_train_multi[cols])
        # Unavailable in multi-labels
        with self.assertRaises(ValueError):
            model.get_predict_position(x_train, y_train_multi[cols])
        remove_dir(model_dir)

    def test06_model_tfidf_svm_save(self):
        '''Test of the method save of {{package_name}}.models_training.model_pipeline.ModelPipeline'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"sklearn_pipeline_standalone.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
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
        self.assertEqual(configs['librairie'], 'scikit-learn')
        self.assertEqual(configs['multiclass_strategy'], None)
        # Specific model used
        self.assertTrue('tfidf_confs' in configs.keys())
        self.assertTrue('svc_confs' in configs.keys())
        remove_dir(model_dir)

        # With multiclass_strategy != None
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy='ovr')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"sklearn_pipeline_standalone.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
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
        self.assertEqual(configs['librairie'], 'scikit-learn')
        self.assertEqual(configs['multiclass_strategy'], 'ovr')
        # Specific model used
        self.assertTrue('tfidf_confs' in configs.keys())
        self.assertTrue('svc_confs' in configs.keys())
        remove_dir(model_dir)

    def test07_model_tfidf_svm_reload_from_standalone(self):
        '''Test of the method {{package_name}}.models_training.model_tfidf_svm.ModelTfidfSvm.reload'''

        ############################################
        # mono_label & without multi-classes strategy
        ############################################

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])
        y_train_mono = np.array(['non', 'oui', 'non', 'oui', 'non'])
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
        tfidf = model.tfidf
        svc = model.svc
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        pkl_path = os.path.join(model.model_dir, f"sklearn_pipeline_standalone.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        new_model = ModelTfidfSvm()
        new_model.reload_from_standalone(configuration_path=conf_path, sklearn_pipeline_path=pkl_path)

        # Test
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.x_col, new_model.x_col)
        self.assertEqual(model.y_col, new_model.y_col)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.multiclass_strategy, new_model.multiclass_strategy)
        self.assertEqual(model.tfidf.get_params(), tfidf.get_params())
        self.assertEqual(model.svc.get_params(), svc.get_params())
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_test)], [list(_) for _ in new_model.predict_proba(x_test)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # mono_label & with multi-classes strategy
        ############################################

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])
        y_train_mono = np.array(['non', 'oui', 'non', 'oui', 'non'])
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False, multiclass_strategy='ovr')
        tfidf = model.tfidf
        svc = model.svc
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        pkl_path = os.path.join(model.model_dir, f"sklearn_pipeline_standalone.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        new_model = ModelTfidfSvm()
        new_model.reload_from_standalone(configuration_path=conf_path, sklearn_pipeline_path=pkl_path)

        # Test
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.x_col, new_model.x_col)
        self.assertEqual(model.y_col, new_model.y_col)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.multiclass_strategy, new_model.multiclass_strategy)
        self.assertEqual(model.tfidf.get_params(), tfidf.get_params())
        self.assertEqual(model.svc.get_params(), svc.get_params())
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_test)], [list(_) for _ in new_model.predict_proba(x_test)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # multi_labels & without multi-classes strategy
        ############################################

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True, multiclass_strategy=None)
        tfidf = model.tfidf
        svc = model.svc
        model.fit(x_train, y_train_multi[cols])
        model.save()

        # Reload
        pkl_path = os.path.join(model.model_dir, f"sklearn_pipeline_standalone.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        new_model = ModelTfidfSvm()
        new_model.reload_from_standalone(configuration_path=conf_path, sklearn_pipeline_path=pkl_path)

        # Test
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.x_col, new_model.x_col)
        self.assertEqual(model.y_col, new_model.y_col)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.multiclass_strategy, new_model.multiclass_strategy)
        self.assertEqual(model.tfidf.get_params(), tfidf.get_params())
        self.assertEqual(model.svc.get_params(), svc.get_params())
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_test)], [list(_) for _ in new_model.predict_proba(x_test)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # multi_labels & with multi-classes strategy
        ############################################

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True, multiclass_strategy='ovr')
        tfidf = model.tfidf
        svc = model.svc
        model.fit(x_train, y_train_multi[cols])
        model.save()

        # Reload
        pkl_path = os.path.join(model.model_dir, f"sklearn_pipeline_standalone.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        new_model = ModelTfidfSvm()
        new_model.reload_from_standalone(configuration_path=conf_path, sklearn_pipeline_path=pkl_path)

        # Test
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.x_col, new_model.x_col)
        self.assertEqual(model.y_col, new_model.y_col)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.multiclass_strategy, new_model.multiclass_strategy)
        self.assertEqual(model.tfidf.get_params(), tfidf.get_params())
        self.assertEqual(model.svc.get_params(), svc.get_params())
        # We can't really test the pipeline so we test predictions
        self.assertEqual([list(_) for _ in model.predict_proba(x_test)], [list(_) for _ in new_model.predict_proba(x_test)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # Errors
        ############################################

        with self.assertRaises(FileNotFoundError):
            new_model = ModelTfidfSvm()
            new_model.reload_from_standalone(configuration_path='toto.json', sklearn_pipeline_path=pkl_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelTfidfSvm()
            new_model.reload_from_standalone(configuration_path=conf_path, sklearn_pipeline_path='toto.pkl')


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
