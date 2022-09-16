#!/usr/bin/env python3
# Copyright (C) <2018-2021>  <Agence Data Services, DSI Pôle Emploi>
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
import shutil
import pickle
import numpy as np
import pandas as pd

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training import utils_models
from {{package_name}}.models_training import model_tfidf_svm, model_embedding_lstm

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class UtilsModelsTests(unittest.TestCase):
    '''Main class to test all functions in utils_models.py'''

    # We avoid tqdm prints
    pd.Series.progress_apply = pd.Series.apply

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def tearDown(self):
        '''Cleaning fonction -> we delete the mock embedding'''
        data_path = utils.get_data_path()
        fake_path = os.path.join(data_path, 'fake_embedding.pkl')
        if os.path.exists(fake_path):
            os.remove(fake_path)

    def test01_get_embedding(self):
        '''Test of the function utils_models.get_embedding'''
        # Create a mock embedding
        data_path = utils.get_data_path()
        # Check if data folder exists
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        fake_embedding = {'toto': [0.25, 0.30], 'titi': [0.85, 0.12]}
        fake_path = os.path.join(data_path, 'fake_embedding.pkl')
        if os.path.exists(fake_path):
            os.remove(fake_path)
        with open(fake_path, 'wb') as f:
            pickle.dump(fake_embedding, f, pickle.HIGHEST_PROTOCOL)

        # Nominal case
        reloaded_embedding = utils_models.get_embedding('fake_embedding.pkl')
        self.assertEqual(reloaded_embedding, fake_embedding)

    def test02_normal_split(self):
        '''Test of the function {{package_name}}.models_training.utils_models.normal_split'''
        # Valids to test
        input_test = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']})
        test_size = 0.2
        train, test = utils_models.normal_split(input_test, test_size=test_size)
        self.assertEqual(train.shape[0], input_test.shape[0] * (1 - test_size))
        self.assertEqual(test.shape[0], input_test.shape[0] * test_size)

        # Check inputs
        with self.assertRaises(ValueError):
            utils_models.normal_split(input_test, test_size=1.2)
        with self.assertRaises(ValueError):
            utils_models.normal_split(input_test, test_size=-0.2)

    def test03_stratified_split(self):
        '''Test of the function {{package_name}}.models_training.utils_models.stratified_split'''
        # Valids to test
        input_test = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'],
                                   'col2': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]})
        test_size = 0.5
        col = 'col2'
        train, test = utils_models.stratified_split(input_test, col, test_size=test_size)
        # -1 because the '2' is removed (not enough values)
        self.assertEqual(train.shape[0], (input_test.shape[0] - 1) * (1 - test_size))
        self.assertEqual(test.shape[0], (input_test.shape[0] - 1) * test_size)
        self.assertEqual(train[train[col] == 0].shape[0], input_test[input_test[col] == 0].shape[0] * (1 - test_size))
        self.assertEqual(test[test[col] == 0].shape[0], input_test[input_test[col] == 0].shape[0] * test_size)
        self.assertEqual(train[train[col] == 2].shape[0], 0)
        self.assertEqual(test[test[col] == 2].shape[0], 0)

        # Check the input(s) type(s)
        with self.assertRaises(ValueError):
            utils_models.stratified_split(input_test, col, test_size=1.2)
        with self.assertRaises(ValueError):
            utils_models.stratified_split(input_test, col, test_size=-0.2)

    def test04_hierarchical_split(self):
        '''Test of the function {{package_name}}.models_training.utils_models.hierarchical_split'''
        # Valids to test
        duplicate_val = 'test1'
        input_test = pd.DataFrame({'col1': ['x', 'a', 'b', 'c', 'x', 'd', 'e', 'f', 'g', 'h', 'x', 'x', 'x', 'x', 'i'],
                                   'col2': [duplicate_val, 'a', 'b', 'c', duplicate_val, 'd', 'e', 'f', 'g', 'h', duplicate_val, duplicate_val, duplicate_val, duplicate_val, 'i']})
        test_size = 0.2
        col = 'col2'
        train, test = utils_models.hierarchical_split(input_test, col, test_size=test_size)
        # We can't really test the dataframe size here (depends on where the duplicate value is sent)
        # Instead, we test the nb of unique values
        self.assertEqual(len(train[col].unique()), len(input_test[col].unique()) * (1 - test_size))
        self.assertEqual(len(test[col].unique()), len(input_test[col].unique()) * test_size)
        # Check all 6 duplicates value in the same dataframe
        self.assertTrue((train[train[col] == duplicate_val].shape[0] == 6 and test[test[col] == duplicate_val].shape[0] == 0)
                        or (train[train[col] == duplicate_val].shape[0] == 0 and test[test[col] == duplicate_val].shape[0] == 6))

        # Check the input(s) type(s)
        with self.assertRaises(ValueError):
            utils_models.hierarchical_split(input_test, col, test_size=1.2)
        with self.assertRaises(ValueError):
            utils_models.hierarchical_split(input_test, col, test_size=-0.2)

    def test05_remove_small_classes(self):
        '''Test of the function {{package_name}}.models_training.utils_models.remove_small_classes'''
        # Valids to test
        input_test = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'],
                                   'col2': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]})
        test_size = 0.2
        col = 'col2'

        result_df = utils_models.remove_small_classes(input_test, col, min_rows=2)
        self.assertEqual(result_df[result_df[col] == 0].shape[0], input_test[input_test[col] == 0].shape[0])
        self.assertEqual(result_df[result_df[col] == 1].shape[0], input_test[input_test[col] == 1].shape[0])
        self.assertEqual(result_df[result_df[col] == 2].shape[0], 0)

        result_df = utils_models.remove_small_classes(input_test, col, min_rows=5)
        self.assertEqual(result_df[result_df[col] == 0].shape[0], 0)
        self.assertEqual(result_df[result_df[col] == 1].shape[0], input_test[input_test[col] == 1].shape[0])
        self.assertEqual(result_df[result_df[col] == 2].shape[0], 0)

        # Check the input(s) type(s)
        with self.assertRaises(ValueError):
            utils_models.remove_small_classes(input_test, col, min_rows=0)

    def test06_display_train_test_shape(self):
        '''Test of the function {{package_name}}.models_training.utils_models.display_train_test_shape'''
        # Valids to test
        df = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'],
                           'col2': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]})

        # Nominal case
        utils_models.display_train_test_shape(df, df)
        utils_models.display_train_test_shape(df, df, df_shape=10)

    def test07_preprocess_model_multilabel(self):
        '''Test of the function {{package_name}}.models_training.utils_models.preprocess_model_multilabel'''
        # Creation of a dataset
        df = pd.DataFrame({'x_col': ['test', 'toto', 'titi'], 'y_col': [(), ('x1', 'x2'), ('x3', 'x4', 'x1')]})
        df_expected = pd.DataFrame({'x_col': ['test', 'toto', 'titi'], 'y_col': [(), ('x1', 'x2'), ('x3', 'x4', 'x1')],
                                    'x1': [0, 1, 1], 'x2': [0, 1, 0], 'x3': [0, 0, 1], 'x4': [0, 0, 1]})
        subset_classes = ['x2', 'x4']
        df_subset_expected = pd.DataFrame({'x_col': ['test', 'toto', 'titi'], 'y_col': [(), ('x1', 'x2'), ('x3', 'x4', 'x1')],
                                           'x2': [0, 1, 0], 'x4': [0, 0, 1]})

        # Nominal case
        df_mlb, classes = utils_models.preprocess_model_multilabel(df, 'y_col')
        self.assertEqual(sorted(classes), ['x1', 'x2', 'x3', 'x4'])
        pd.testing.assert_frame_equal(df_mlb, df_expected, check_dtype=False)

        # Test argument classes
        df_mlb, classes = utils_models.preprocess_model_multilabel(df, 'y_col', classes=subset_classes)
        self.assertEqual(sorted(classes), sorted(subset_classes))
        pd.testing.assert_frame_equal(df_mlb, df_subset_expected, check_dtype=False)

    def test08_load_model(self):
        '''Test of the function {{package_name}}.models_training.utils_models.load_model'''
        # Creation fake model
        x_train = ['test', 'test', 'test', 'toto', 'toto', 'toto']
        y_train = [0, 0, 0, 1, 1, 1]
        model_dir = os.path.join(utils.get_models_path(), 'test_model')
        remove_dir(model_dir)
        model_name = 'test_model_name'
        model = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir, model_name=model_name)
        model.fit(x_train, y_train)
        model.save()

        # Reload
        new_model, new_config = utils_models.load_model(model_dir='test_model')
        # We perform some tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(['test', 'toto', 'a'])), list(model.predict(['test', 'toto', 'a'])))

        # Same thing but with a path
        new_model, new_config = utils_models.load_model(model_dir=model_dir, is_path=True)
        # We perform some tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(['test', 'toto', 'a'])), list(model.predict(['test', 'toto', 'a'])))
        remove_dir(model_dir)

        # We do the same thing on a keras model
        data_path = utils.get_data_path()
        # Check if data folder exists
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        fake_embedding = {'test': [0.8, 0.2], 'toto': [0.1, 0.4]}
        fake_path = os.path.join(data_path, 'fake_embedding.pkl')
        if os.path.exists(fake_path):
            os.remove(fake_path)
        with open(fake_path, 'wb') as f:
            pickle.dump(fake_embedding, f, pickle.HIGHEST_PROTOCOL)
        x_train = ['test', 'test', 'test', 'toto', 'toto', 'toto'] * 10
        y_train = [0, 0, 0, 1, 1, 1] * 10
        model_dir = os.path.join(utils.get_models_path(), 'test_model_dl')
        remove_dir(model_dir)
        model_name = 'test_model_dl_name'
        epochs = 2
        batch_size = 8
        embedding_name = 'fake_embedding.pkl'
        model = model_embedding_lstm.ModelEmbeddingLstm(model_dir=model_dir, model_name=model_name,
                                                        epochs=epochs, batch_size=batch_size,
                                                        embedding_name=embedding_name)
        model.fit(x_train, y_train)
        model.save()

        # Reload
        new_model, new_config = utils_models.load_model(model_dir='test_model_dl')
        # We perform some tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_config['epochs'], epochs)
        self.assertEqual(new_config['batch_size'], batch_size)
        self.assertEqual(new_config['embedding_name'], embedding_name)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(new_model.epochs, epochs)
        self.assertEqual(new_model.batch_size, batch_size)
        self.assertEqual(new_model.embedding_name, embedding_name)
        self.assertEqual(list(new_model.predict(['test', 'toto', 'a'])), list(model.predict(['test', 'toto', 'a'])))
        self.assertEqual([list(_) for _ in new_model.predict_proba(['test', 'toto', 'a'])], [list(_) for _ in model.predict_proba(['test', 'toto', 'a'])])
        remove_dir(model_dir)

        # TODO: test pytorch models ?

        # Check errors
        with self.assertRaises(FileNotFoundError):
            utils_models.load_model(model_dir='tototo')
        with self.assertRaises(FileNotFoundError):
            utils_models.load_model(model_dir='./tototo', is_path=True)

    def test09_predict(self):
        '''Test of the function {{package_name}}.models_training.utils_models.predict'''
        # Data
        x_train_base = ['Ceci est un test', 'Ceci est un test', 'Ceci est un test', 'Test "deux" !', 'Toto "deux" !', 'Test "deux" !', 'deux'] * 10
        y_train_mono = ['test', 'test', 'test', 'deux', 'deux', 'deux', 'deux'] * 10
        y_train_multi = pd.DataFrame({'test': [1, 1, 1, 1, 0, 1, 0] * 10, 'toto': [0, 0, 0, 0, 1, 0, 0] * 10})
        preprocessor = preprocess.get_preprocessor('preprocess_P1')
        x_train = preprocessor(x_train_base)

        # Nominal case - mono-label
        model_dir = os.path.join(utils.get_models_path(), 'test_model3')
        remove_dir(model_dir)
        model_name = 'test_model3_name'
        model = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir, model_name=model_name)
        model.fit(x_train, y_train_mono)
        model.save()
        model, model_conf = utils_models.load_model('test_model3')
        self.assertEqual(utils_models.predict('Ceci est un test', model, model_conf), 'test')
        self.assertEqual(utils_models.predict('Test "deux" !', model, model_conf), 'deux')
        remove_dir(model_dir)

        # Nominal case - multi-labels
        model_dir = os.path.join(utils.get_models_path(), 'test_model3')
        remove_dir(model_dir)
        model_name = 'test_model3_name'
        model = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir, model_name=model_name, multi_label=True)
        model.fit(x_train, y_train_multi)
        model.save()
        model, model_conf = utils_models.load_model('test_model3')
        self.assertEqual(utils_models.predict('Ceci est un test', model, model_conf), ('test',))
        self.assertEqual(utils_models.predict('Toto "deux" !', model, model_conf), ('toto',))
        remove_dir(model_dir)

    def test10_predict_with_proba(self):
        '''Test of the function {{package_name}}.models_training.utils_models.predict_with_proba'''
        # Data
        x_train_base = ['Ceci est un test', 'Ceci est un test', 'Ceci est un test', 'Test "deux" !', 'Toto "deux" !', 'Test "deux" !', 'deux'] * 10
        y_train_mono = ['test', 'test', 'test', 'deux', 'deux', 'deux', 'deux'] * 10
        y_train_multi = pd.DataFrame({'test': [1, 1, 1, 1, 0, 1, 0] * 10, 'toto': [0, 0, 0, 0, 1, 0, 0] * 10})
        preprocessor = preprocess.get_preprocessor('preprocess_P1')
        x_train = preprocessor(x_train_base)

        # Nominal case - mono-label
        model_dir = os.path.join(utils.get_models_path(), 'test_model4')
        remove_dir(model_dir)
        model_name = 'test_model4_name'
        model = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir, model_name=model_name)
        model.fit(x_train, y_train_mono)
        model.save()
        model, model_conf = utils_models.load_model('test_model4')
        pred, proba = utils_models.predict_with_proba('Ceci est un test', model, model_conf)
        self.assertEqual(pred, 'test')
        self.assertEqual(proba, 1.0) # 1.0 because svm
        pred, proba = utils_models.predict_with_proba('Test "deux" !', model, model_conf)
        self.assertEqual(pred, 'deux')
        self.assertEqual(proba, 1.0) # 1.0 because svm
        remove_dir(model_dir)

        # Nominal case - multi-labels
        model_dir = os.path.join(utils.get_models_path(), 'test_model4')
        remove_dir(model_dir)
        model_name = 'test_model4_name'
        model = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir, model_name=model_name, multi_label=True)
        model.fit(x_train, y_train_multi)
        model.save()
        model, model_conf = utils_models.load_model('test_model4')
        pred, proba = utils_models.predict_with_proba('Ceci est un test', model, model_conf)
        self.assertEqual(pred, ('test',))
        self.assertEqual(proba, (1.0,)) # 1.0 because svm
        pred, proba = utils_models.predict_with_proba('Toto "deux" !', model, model_conf)
        self.assertEqual(pred, ('toto',))
        self.assertEqual(proba, (1.0,)) # 1.0 because svm
        remove_dir(model_dir)

    def test11_search_hp_cv(self):
        '''Test of the function {{package_name}}.models_training.utils_models.search_hp_cv'''
        # Definition of the variables for the nominal case
        x_train = ['Ceci est un test', 'Ceci est un test', 'Ceci est un test', 'Test "deux" !', 'Toto "deux" !', 'Test "deux" !', 'deux'] * 10
        y_train_mono = ['test', 'test', 'test', 'deux', 'deux', 'deux', 'deux'] * 10
        y_train_multi = pd.DataFrame({'test': [1, 1, 1, 1, 0, 1, 0] * 10, 'toto': [0, 0, 0, 0, 1, 0, 0] * 10})
        model_cls = model_tfidf_svm.ModelTfidfSvm
        model_params_mono = {'multi_label': False}
        model_params_multi = {'multi_label': True}
        hp_params = {'tfidf_params': [{'analyzer': 'word', 'ngram_range': (1, 2), "max_df": 1.0, 'min_df': 0.0}, {'analyzer': 'word', 'ngram_range': (1, 3), "max_df": 1.0, 'min_df': 0.0}], 'svc_params': [{'C': 1.0}, {'C': 2.0}]}
        kwargs_fit_mono = {'x_train':x_train, 'y_train': y_train_mono}
        kwargs_fit_multi = {'x_train':x_train, 'y_train': y_train_multi}

        # Nominal case
        n_splits = 5
        model = utils_models.search_hp_cv(model_cls, model_params_mono, hp_params, "accuracy", kwargs_fit_mono, n_splits=n_splits)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        csv_path = os.path.join(model.model_dir, f"hyper_params_results.csv")
        json_path = os.path.join(model.model_dir, f"hyper_params_tested.json")
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        with open(csv_path, 'r', encoding="{{default_encoding}}") as f:
            self.assertEqual(sum(1 for line in f), n_splits * len(hp_params) + 1)
        remove_dir(model.model_dir)

        # Multi-labels case
        n_splits = 3
        model = utils_models.search_hp_cv(model_cls, model_params_multi, hp_params, "f1", kwargs_fit_multi, n_splits=n_splits)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        csv_path = os.path.join(model.model_dir, f"hyper_params_results.csv")
        json_path = os.path.join(model.model_dir, f"hyper_params_tested.json")
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        with open(csv_path, 'r', encoding="{{default_encoding}}") as f:
            self.assertEqual(sum(1 for line in f), n_splits * len(hp_params) + 1)
        remove_dir(model.model_dir)

        # Check the various scoring functions
        n_splits = 2
        model = utils_models.search_hp_cv(model_cls, model_params_mono, hp_params, "precision", kwargs_fit_mono, n_splits=n_splits)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        csv_path = os.path.join(model.model_dir, f"hyper_params_results.csv")
        json_path = os.path.join(model.model_dir, f"hyper_params_tested.json")
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        with open(csv_path, 'r', encoding="{{default_encoding}}") as f:
            self.assertEqual(sum(1 for line in f), n_splits * len(hp_params) + 1)
        remove_dir(model.model_dir)

        n_splits = 2
        model = utils_models.search_hp_cv(model_cls, model_params_mono, hp_params, "recall", kwargs_fit_mono, n_splits=n_splits)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        csv_path = os.path.join(model.model_dir, f"hyper_params_results.csv")
        json_path = os.path.join(model.model_dir, f"hyper_params_tested.json")
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        with open(csv_path, 'r', encoding="{{default_encoding}}") as f:
            self.assertEqual(sum(1 for line in f), n_splits * len(hp_params) + 1)
        remove_dir(model.model_dir)

        # Custom scoring function
        def custom_func(test_dict: dict):
            return (test_dict['Precision'] + test_dict['Recall']) / 2
        n_splits = 2
        model = utils_models.search_hp_cv(model_cls, model_params_mono, hp_params, custom_func, kwargs_fit_mono, n_splits=n_splits)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        csv_path = os.path.join(model.model_dir, f"hyper_params_results.csv")
        json_path = os.path.join(model.model_dir, f"hyper_params_tested.json")
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        with open(csv_path, 'r', encoding="{{default_encoding}}") as f:
            self.assertEqual(sum(1 for line in f), n_splits * len(hp_params) + 1)
        remove_dir(model.model_dir)

        # Check errors
        with self.assertRaises(TypeError):
            model = utils_models.search_hp_cv(model_cls, model_params_mono, hp_params, 5, kwargs_fit_mono, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv(model_cls, model_params_mono, hp_params, 'toto', kwargs_fit_mono, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv(model_cls, {'toto': True}, hp_params, 'accuracy', kwargs_fit_mono, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv(model_cls, model_params_mono, hp_params, 'accuracy', {'toto': True, 'y_train': y_train_mono}, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv(model_cls, model_params_mono, hp_params, 'accuracy', {'toto': True, 'x_train': x_train}, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv(model_cls, model_params_mono, hp_params, 'accuracy', {'toto': True, 'x_train': x_train}, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv(model_cls, {**model_params_mono, **{'toto': True}}, {**hp_params, **{'toto': [False]}}, 'accuracy', kwargs_fit_mono, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv(model_cls, model_params_mono, {'toto': [1, 2], 'titi': [3]}, 'accuracy', kwargs_fit_mono, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv(model_cls, model_params_mono, hp_params, 'accuracy', kwargs_fit_mono, n_splits=1)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
