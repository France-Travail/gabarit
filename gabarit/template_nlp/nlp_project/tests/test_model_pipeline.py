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
import dill as pickle

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from {{package_name}} import utils
from {{package_name}}.models_training.models_sklearn.model_pipeline import ModelPipeline
from {{package_name}}.models_training.models_sklearn.model_tfidf_gbt import ModelTfidfGbt


# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


def compare_trees(tree1, tree2):
    '''Checks if two DecisionTreeClassifiers are equal
    Args:
        tree1 (DecisionTreeClassifier): First tree to consider
        tree2 (DecisionTreeClassifier): Second tree to consider
    Results:
        bool: True if all trees nodes and values are equal, else False
    '''
    state1 = tree1.tree_.__getstate__()
    state2 = tree2.tree_.__getstate__()
    if not np.array_equal(state1["nodes"], state2["nodes"]):
        return False
    if not np.array_equal(state1["values"], state2["values"]):
        return False
    return True 


class ModelPipelineTests(unittest.TestCase):
    '''Main class to test model_pipeline'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_pipeline_init(self):
        '''Test of the initialization of {{package_name}}.models_training.models_sklearn.model_pipeline.ModelPipeline'''

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
        '''Test of the method fit of {{package_name}}.models_training.models_sklearn.model_pipeline.ModelPipeline'''

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

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_dir2 = os.path.join(os.getcwd(), 'model_test_123456789_2')
        remove_dir(model_dir2)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0], 'test2': [1, 0, 0, 0, 0], 'test3': [0, 0, 0, 1, 0]})
        cols = ['test1', 'test2', 'test3']

        # Mono-label - no strategy - same random_seed
        model1 = ModelTfidfGbt(model_dir=model_dir, multi_label=False, multiclass_strategy=None, random_seed=42)
        model1.fit(x_train, y_train_mono)
        model2 = ModelTfidfGbt(model_dir=model_dir2, multi_label=False, multiclass_strategy=None, random_seed=42)
        model2.fit(x_train, y_train_mono)
        self.assertEqual(model1.gbt.get_params(),  model2.gbt.get_params())
        self.assertTrue(all(compare_trees(tree1, tree2) for tree1, tree2 in zip(model1.gbt.estimators_.flatten(), model2.gbt.estimators_.flatten())))
        remove_dir(model_dir), remove_dir(model_dir2)

        # Mono-label - ovr strategy - same random_seed
        model1 = ModelTfidfGbt(model_dir=model_dir, multi_label=False, multiclass_strategy='ovr', random_seed=42)
        model1.fit(x_train, y_train_mono)
        model2 = ModelTfidfGbt(model_dir=model_dir2, multi_label=False, multiclass_strategy='ovr', random_seed=42)
        model2.fit(x_train, y_train_mono)
        models1, models2 = model1.pipeline['gbt'].estimators_, model2.pipeline['gbt'].estimators_
        self.assertEqual(model1.gbt.get_params(),  model2.gbt.get_params())
        self.assertTrue(all(all(compare_trees(tree1, tree2) 
                                for tree1, tree2 in zip(m1.estimators_.flatten(), m2.estimators_.flatten()))
                             for m1, m2 in zip(models1, models2)))
        remove_dir(model_dir), remove_dir(model_dir2)

        # Mono-label - ovo strategy - same random_seed
        model1 = ModelTfidfGbt(model_dir=model_dir, multi_label=False, multiclass_strategy='ovo', random_seed=42)
        model1.fit(x_train, y_train_mono)
        model2 = ModelTfidfGbt(model_dir=model_dir2, multi_label=False, multiclass_strategy='ovo', random_seed=42)
        model2.fit(x_train, y_train_mono)
        models1, models2 = model1.pipeline['gbt'].estimators_, model2.pipeline['gbt'].estimators_
        self.assertEqual(model1.gbt.get_params(),  model2.gbt.get_params())
        self.assertTrue(all(all(compare_trees(tree1, tree2) 
                                for tree1, tree2 in zip(m1.estimators_.flatten(), m2.estimators_.flatten()))
                             for m1, m2 in zip(models1, models2)))
        remove_dir(model_dir), remove_dir(model_dir2)

        # Mono-label - no strategy - different random_seed
        model1 = ModelTfidfGbt(model_dir=model_dir, multi_label=False, multiclass_strategy=None, random_seed=42)
        model1.fit(x_train, y_train_mono)
        model2 = ModelTfidfGbt(model_dir=model_dir2, multi_label=False, multiclass_strategy=None, random_seed=41)
        model2.fit(x_train, y_train_mono)
        self.assertNotEqual(model1.gbt.get_params(),  model2.gbt.get_params())
        self.assertFalse(all(compare_trees(tree1, tree2) for tree1, tree2 in zip(model1.gbt.estimators_.flatten(), model2.gbt.estimators_.flatten())))
        remove_dir(model_dir), remove_dir(model_dir2)

        # Mono-label - ovr strategy - different random_seed
        model1 = ModelTfidfGbt(model_dir=model_dir, multi_label=False, multiclass_strategy='ovr', random_seed=42)
        model1.fit(x_train, y_train_mono)
        model2 = ModelTfidfGbt(model_dir=model_dir2, multi_label=False, multiclass_strategy='ovr', random_seed=41)
        model2.fit(x_train, y_train_mono)
        models1, models2 = model1.pipeline['gbt'].estimators_, model2.pipeline['gbt'].estimators_
        self.assertNotEqual(model1.gbt.get_params(),  model2.gbt.get_params())
        self.assertFalse(all(all(compare_trees(tree1, tree2) 
                                for tree1, tree2 in zip(m1.estimators_.flatten(), m2.estimators_.flatten()))
                             for m1, m2 in zip(models1, models2)))
        remove_dir(model_dir), remove_dir(model_dir2)

        # Mono-label - ovo strategy - different random_seed
        model1 = ModelTfidfGbt(model_dir=model_dir, multi_label=False, multiclass_strategy='ovo', random_seed=42)
        model1.fit(x_train, y_train_mono)
        model2 = ModelTfidfGbt(model_dir=model_dir2, multi_label=False, multiclass_strategy='ovo', random_seed=41)
        model2.fit(x_train, y_train_mono)
        models1, models2 = model1.pipeline['gbt'].estimators_, model2.pipeline['gbt'].estimators_
        self.assertNotEqual(model1.gbt.get_params(),  model2.gbt.get_params())
        self.assertFalse(all(all(compare_trees(tree1, tree2) 
                                for tree1, tree2 in zip(m1.estimators_.flatten(), m2.estimators_.flatten()))
                             for m1, m2 in zip(models1, models2)))
        remove_dir(model_dir), remove_dir(model_dir2)

        # Multi-label - no strategy - same random_seed
        model1 = ModelTfidfGbt(model_dir=model_dir, multi_label=True, multiclass_strategy=None, random_seed=42)
        model1.fit(x_train, y_train_multi)
        model2 = ModelTfidfGbt(model_dir=model_dir2, multi_label=True, multiclass_strategy=None, random_seed=42)
        model2.fit(x_train, y_train_multi)
        models1, models2 = model1.pipeline['gbt'].estimators_, model2.pipeline['gbt'].estimators_
        self.assertEqual(model1.gbt.get_params(),  model2.gbt.get_params())
        self.assertTrue(all(all(compare_trees(tree1, tree2) 
                                for tree1, tree2 in zip(m1.estimators_.flatten(), m2.estimators_.flatten()))
                             for m1, m2 in zip(models1, models2)))
        remove_dir(model_dir), remove_dir(model_dir2)

        # Mutli-label - ovr strategy - same random_seed
        model1 = ModelTfidfGbt(model_dir=model_dir, multi_label=True, multiclass_strategy='ovr', random_seed=42)
        model1.fit(x_train, y_train_multi)
        model2 = ModelTfidfGbt(model_dir=model_dir2, multi_label=True, multiclass_strategy='ovr', random_seed=42)
        model2.fit(x_train, y_train_multi)
        models1, models2 = model1.pipeline['gbt'].estimators_, model2.pipeline['gbt'].estimators_
        self.assertEqual(model1.gbt.get_params(),  model2.gbt.get_params())
        self.assertTrue(all(all(compare_trees(tree1, tree2) 
                                for tree1, tree2 in zip(m1.estimators_.flatten(), m2.estimators_.flatten()))
                             for m1, m2 in zip(models1, models2)))
        remove_dir(model_dir), remove_dir(model_dir2)

        # Mutli-label - ovo strategy - same random_seed
        model1 = ModelTfidfGbt(model_dir=model_dir, multi_label=True, multiclass_strategy='ovo', random_seed=42)
        model1.fit(x_train, y_train_multi)
        model2 = ModelTfidfGbt(model_dir=model_dir2, multi_label=True, multiclass_strategy='ovo', random_seed=42)
        model2.fit(x_train, y_train_multi)
        models1, models2 = model1.pipeline['gbt'].estimators_, model2.pipeline['gbt'].estimators_
        self.assertEqual(model1.gbt.get_params(),  model2.gbt.get_params())
        self.assertTrue(all(all(compare_trees(tree1, tree2) 
                                for tree1, tree2 in zip(m1.estimators_.flatten(), m2.estimators_.flatten()))
                             for m1, m2 in zip(models1, models2)))
        remove_dir(model_dir), remove_dir(model_dir2)

        # Multi-label - no strategy - different random_seed
        model1 = ModelTfidfGbt(model_dir=model_dir, multi_label=True, multiclass_strategy=None, random_seed=42)
        model1.fit(x_train, y_train_multi)
        model2 = ModelTfidfGbt(model_dir=model_dir2, multi_label=True, multiclass_strategy=None, random_seed=41)
        model2.fit(x_train, y_train_multi)
        models1, models2 = model1.pipeline['gbt'].estimators_, model2.pipeline['gbt'].estimators_
        self.assertNotEqual(model1.gbt.get_params(),  model2.gbt.get_params())
        self.assertFalse(all(all(compare_trees(tree1, tree2) 
                                for tree1, tree2 in zip(m1.estimators_.flatten(), m2.estimators_.flatten()))
                             for m1, m2 in zip(models1, models2)))
        remove_dir(model_dir), remove_dir(model_dir2)

        # Mutli-label - ovr strategy - different random_seed
        model1 = ModelTfidfGbt(model_dir=model_dir, multi_label=True, multiclass_strategy='ovr', random_seed=42)
        model1.fit(x_train, y_train_multi)
        model2 = ModelTfidfGbt(model_dir=model_dir2, multi_label=True, multiclass_strategy='ovr', random_seed=41)
        model2.fit(x_train, y_train_multi)
        models1, models2 = model1.pipeline['gbt'].estimators_, model2.pipeline['gbt'].estimators_
        self.assertNotEqual(model1.gbt.get_params(),  model2.gbt.get_params())
        self.assertFalse(all(all(compare_trees(tree1, tree2) 
                                for tree1, tree2 in zip(m1.estimators_.flatten(), m2.estimators_.flatten()))
                             for m1, m2 in zip(models1, models2)))
        remove_dir(model_dir), remove_dir(model_dir2)

        # Mutli-label - ovo strategy - different random_seed
        model1 = ModelTfidfGbt(model_dir=model_dir, multi_label=True, multiclass_strategy='ovo', random_seed=42)
        model1.fit(x_train, y_train_multi)
        model2 = ModelTfidfGbt(model_dir=model_dir2, multi_label=True, multiclass_strategy='ovo', random_seed=41)
        model2.fit(x_train, y_train_multi)
        models1, models2 = model1.pipeline['gbt'].estimators_, model2.pipeline['gbt'].estimators_
        self.assertNotEqual(model1.gbt.get_params(),  model2.gbt.get_params())
        self.assertFalse(all(all(compare_trees(tree1, tree2) 
                                for tree1, tree2 in zip(m1.estimators_.flatten(), m2.estimators_.flatten()))
                             for m1, m2 in zip(models1, models2)))
        remove_dir(model_dir), remove_dir(model_dir2)
        

    def test03_model_pipeline_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.models_sklearn.model_pipeline.ModelPipeline'''

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
        '''Test of the method predict_proba of {{package_name}}.models_training.models_sklearn.model_pipeline.ModelPipeline'''

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
        '''Test of the method save of {{package_name}}.models_training.models_sklearn.model_pipeline.ModelPipeline'''

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
        with open(os.path.join(model.model_dir, f"sklearn_pipeline_standalone.pkl"), 'rb') as pickle_file:
            loaded_pipeline = pickle.load(pickle_file)
        self.assertTrue(isinstance(loaded_pipeline[0], TfidfVectorizer))
        self.assertTrue(isinstance(loaded_pipeline[1], RandomForestClassifier))
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

    def test06_model_pipeline_load_standalone_files(self):
        '''Test of the method _load_standalone_files of {{package_name}}.models_training.models_sklearn.model_pipeline.ModelPipeline'''
        
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        new_model_dir = os.path.join(os.getcwd(), 'model_test_987654321')
        remove_dir(new_model_dir)

        # With Pipeline
        tfidf = TfidfVectorizer()
        rf = RandomForestClassifier(n_estimators=10)
        pipeline = Pipeline([('tfidf', tfidf), ('rf', rf)])
        model = ModelPipeline(model_dir=model_dir, pipeline=pipeline)
        sklearn_pipeline_path = os.path.join(model_dir, "sklearn_pipeline_standalone.pkl")
        
        # Check errors
        with self.assertRaises(ValueError):
            model._load_standalone_files()
        with self.assertRaises(FileNotFoundError):
            model._load_standalone_files(sklearn_pipeline_path=sklearn_pipeline_path)
        with self.assertRaises(FileNotFoundError):
            model._load_standalone_files(sklearn_pipeline_path=model_dir)

        # Save model
        model.save(json_data={'test': 8})

        # Reload it with a pipeline path
        new_model = ModelPipeline(model_dir=new_model_dir)
        self.assertTrue(new_model.pipeline is None)
        new_model._load_standalone_files(sklearn_pipeline_path=sklearn_pipeline_path)
        self.assertEqual(len(new_model.pipeline), 2)
        self.assertTrue(isinstance(new_model.pipeline[0], TfidfVectorizer))
        self.assertTrue(isinstance(new_model.pipeline[1], RandomForestClassifier))
        remove_dir(new_model_dir)

        # Reload it from a default_model_dir
        new_model = ModelPipeline(model_dir=new_model_dir)
        self.assertTrue(new_model.pipeline is None)
        new_model._load_standalone_files(default_model_dir=model_dir)
        self.assertEqual(len(new_model.pipeline), 2)
        self.assertTrue(isinstance(new_model.pipeline[0], TfidfVectorizer))
        self.assertTrue(isinstance(new_model.pipeline[1], RandomForestClassifier))
        remove_dir(new_model_dir)

        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
