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
from {{package_name}}.models_training.model_pipeline import ModelPipeline

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelPipelineTests(unittest.TestCase):
    '''Main class to test model_pipeline'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_pipeline_init(self):
        '''Test of the initialization of {{package_name}}.models_training.model_pipeline.ModelPipeline'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        model = ModelPipeline(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.pipeline, None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        tfidf = TfidfVectorizer()
        svc = LinearSVC()
        pipeline = Pipeline([('tfidf', tfidf), ('svc', svc)])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline)
        model.save()
        self.assertEqual(model.pipeline, pipeline)
        remove_dir(model_dir)

    def test02_model_pipeline_fit(self):
        '''Test of the method fit of {{package_name}}.models_training.model_pipeline.ModelPipeline'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 0])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        tfidf = TfidfVectorizer()
        svc = LinearSVC()
        pipeline = Pipeline([('tfidf', tfidf), ('svc', svc)])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertFalse(hasattr(model.pipeline['svc'], "classes_"))
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(hasattr(model.pipeline['svc'], "classes_"))
        remove_dir(model_dir)

        # We also test that the presence of arguments such as x_valid has no impact
        tfidf = TfidfVectorizer()
        svc = LinearSVC()
        pipeline = Pipeline([('tfidf', tfidf), ('svc', svc)])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertFalse(hasattr(model.pipeline['svc'], "classes_"))
        model.fit(x_train, y_train_mono, x_valid='toto', y_valid='titi', test=5)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(hasattr(model.pipeline['svc'], "classes_"))
        remove_dir(model_dir)

        # Multi-labels
        tfidf = TfidfVectorizer()
        svc = LinearSVC()
        pipeline = Pipeline([('tfidf', tfidf), ('svc', OneVsRestClassifier(svc))])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertFalse(hasattr(model.pipeline, "classes_"))
        model.fit(x_train, y_train_multi[cols])
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(hasattr(model.pipeline, "classes_"))
        remove_dir(model_dir)


        ############
        # Test continue training
        tfidf = TfidfVectorizer()
        svc = LinearSVC()
        pipeline = Pipeline([('tfidf', tfidf), ('svc', svc)])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertFalse(hasattr(model.pipeline['svc'], "classes_"))
        model.fit(x_train, y_train_mono)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(hasattr(model.pipeline['svc'], "classes_"))
        # Second fit
        with self.assertRaises(RuntimeError):
            model.fit(x_train[:50], y_train_mono[:50])
        self.assertEqual(model_dir, model.model_dir)
        remove_dir(model_dir)

    def test03_model_pipeline_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.model_pipeline.ModelPipeline'''

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
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        preds = model.predict('test', return_proba=False)
        self.assertEqual(preds, model.predict(['test'], return_proba=False)[0])
        proba = model.predict(x_train, return_proba=True)
        self.assertEqual(proba.shape, (len(x_train), 2))
        proba = model.predict('test', return_proba=True)
        self.assertEqual([elem for elem in proba], [elem for elem in model.predict(['test'], return_proba=True)[0]])
        remove_dir(model_dir)

        # Multi-labels
        tfidf = TfidfVectorizer()
        rf = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('tfidf', tfidf), ('rf', OneVsRestClassifier(rf))])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=True)
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
            model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=False)
            model.predict('test')
        remove_dir(model_dir)

    def test04_model_pipeline_predict_proba(self):
        '''Test of the method predict_proba of {{package_name}}.models_training.model_pipeline.ModelPipeline'''

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
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 2))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Multi-labels
        tfidf = TfidfVectorizer()
        rf = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('tfidf', tfidf), ('rf', OneVsRestClassifier(rf))])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, multi_label=False)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test05_model_pipeline_save(self):
        '''Test of the method save of {{package_name}}.models_training.model_pipeline.ModelPipeline'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # With Pipeline
        tfidf = TfidfVectorizer()
        rf = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('tfidf', tfidf), ('rf', rf)])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline)
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
        # Specific model used
        self.assertTrue('tfidf_confs' in configs.keys())
        self.assertTrue('rf_confs' in configs.keys())
        remove_dir(model_dir)

        # Without Pipeline
        model = ModelPipeline(model_dir=model_dir, pipeline=None)
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f"sklearn_pipeline_standalone.pkl")))
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
        # Specific model used
        self.assertTrue('tfidf_confs' not in configs.keys())
        self.assertTrue('rf_confs' not in configs.keys())
        remove_dir(model_dir)

        # WITH level_save = 'LOW'
        tfidf = TfidfVectorizer()
        rf = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('tfidf', tfidf), ('rf', rf)])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline, level_save='LOW')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, f"sklearn_pipeline_standalone.pkl")))
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
        # Specific model used
        self.assertTrue('tfidf_confs' in configs.keys())
        self.assertTrue('rf_confs' in configs.keys())
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
