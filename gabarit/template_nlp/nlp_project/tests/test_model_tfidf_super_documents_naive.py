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

from {{package_name}} import utils
from {{package_name}}.models_training.model_tfidf_super_documents_naive import ModelTfidfSuperDocumentsNaive
from {{package_name}}.models_training.utils_super_documents import TfidfVectorizerSuperDocuments

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelTfidfSuperDocumentsNaiveTests(unittest.TestCase):
    '''Main class to test model_tfidf_super_documents_naive'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_tfidf_super_documents_naive_init(self):
        '''Test of {{package_name}}.models_training.model_tfidf_super_documents_naive.ModelTfidfSuperDocumentsNaive.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all params
        model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertFalse(model.pipeline is None)
        self.assertTrue(type(model.with_super_documents) == bool)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        # Check TFIDF params
        model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, tfidf_params={'norm': 'l1', 'sublinear_tf': True})
        self.assertEqual(model.tfidf.norm, 'l1')
        self.assertEqual(model.tfidf.sublinear_tf, True)
        remove_dir(model_dir)

        # Check with super documents
        model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, with_super_documents=True)
        self.assertEqual(model.with_super_documents, True)
        remove_dir(model_dir)

        # Error
        with self.assertRaises(ValueError):
            model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, multi_label=False, multiclass_strategy='toto')
        remove_dir(model_dir)

        with self.assertRaises(ValueError):
            model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, multi_label=True)
        remove_dir(model_dir)

    def test02_model_tfidf_super_documents_naive_predict(self):
        '''Test of {{package_name}}.models_training.model_tfidf_super_documents_naive.ModelTfidfSuperDocumentsNaive.predict'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        y_train_str = np.array(['a', 'b', 'a', 'b', 'c'])
        x_test = np.array(['test', 'test2'])

        # Mono label - no strategy
        model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, multi_label=False, multiclass_strategy=None, with_super_documents=True)
        model.fit(x_train, y_train_mono)
        self.assertTrue(isinstance(model.tfidf, TfidfVectorizerSuperDocuments))
        preds = model.predict(x_test, return_proba=False)
        self.assertEqual(preds.shape, (len(x_test),))
        self.assertEqual(preds.all(), model.predict(x_test, return_proba=False).all())
        remove_dir(model_dir)

        model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, multi_label=False, multiclass_strategy=None, with_super_documents=True)
        model.fit(x_train, y_train_str)
        preds = model.predict(x_train, return_proba=False)
        self.assertEqual(preds.shape, (len(x_train),))
        self.assertTrue((preds == y_train_str).all())
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
            model.predict('test')
        remove_dir(model_dir)

    def test03_model_tfidf_super_documents_naive_predict_proba(self):
        '''Test of {{package_name}}.models_training.model_tfidf_super_documents_naive.ModelTfidfSuperDocumentsNaive.predict_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        n_classes = 3

        # Mono-label - no strategy
        model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), n_classes))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test04_model_tfidf_super_documents_naive_compute_predict(self):
        '''Test of {{package_name}}.models_training.model_tfidf_super_documents_naive.ModelTfidfSuperDocumentsNaive.compute_predict'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        y_train_mono = np.array([0, 1, 0, 1, 2])
        y_train_str = np.array(['a', 'b', 'a', 'b', 'c'])

        # Mono label - no strategy
        model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, multi_label=False, multiclass_strategy=None, with_super_documents=True)
        model.fit(x_train, y_train_mono)
        preds = model.compute_predict(x_train)
        self.assertEqual(preds.shape, (len(x_train),))
        remove_dir(model_dir)

        model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, multi_label=False, multiclass_strategy=None, with_super_documents=True)
        model.fit(x_train, y_train_str)
        preds = model.compute_predict(x_train)
        self.assertEqual(preds.shape, (len(x_train),))
        self.assertTrue((preds == y_train_str).all())
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
            model.compute_predict(x_train)
        remove_dir(model_dir)

        # tfidf not fitted
        model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, multi_label=False, multiclass_strategy=None)
        with self.assertRaises(AttributeError):
            model.compute_predict('test')
        remove_dir(model_dir)

    def test05_model_tfidf_super_documents_naive_reload_from_standalone(self):
        '''Test of {{package_name}}.models_training.model_tfidf_super_documents_naive.ModelTfidfSuperDocumentsNaive.reload_from_standalone'''

        ############################################
        # mono_label & without multi-classes strategy
        ############################################

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])
        y_train_mono = np.array(['non', 'oui', 'non', 'oui', 'non'])
        model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir, multi_label=False, multiclass_strategy=None, with_super_documents=True)
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        pkl_path = os.path.join(model.model_dir, f"sklearn_pipeline_standalone.pkl")
        conf_path = os.path.join(model.model_dir, "configurations.json")
        count_vectorizer_path = os.path.join(model_dir, f"count_vectorizer.pkl")
        tfidf_super_documents_path = os.path.join(model_dir, "tfidf_super_documents.pkl")
        new_model = ModelTfidfSuperDocumentsNaive()
        new_model.reload_from_standalone(configuration_path=conf_path, sklearn_pipeline_path=pkl_path, count_vectorizer_path=count_vectorizer_path, tfidf_super_documents_path=tfidf_super_documents_path)

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
        self.assertTrue((model.tfidf.tfidf_super_documents == new_model.tfidf.tfidf_super_documents).all())
        self.assertEqual(model.tfidf.count_vec.get_params(), new_model.tfidf.count_vec.get_params())
        self.assertEqual(model.with_super_documents, new_model.with_super_documents)
        self.assertTrue((model.tfidf.classes_ == new_model.tfidf.classes_).all())
        # We can't really test the pipeline so we test predictions
        self.assertTrue(len(np.setdiff1d(model.predict(x_test), new_model.predict(x_test))) == 0)
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        ############################################
        # Errors
        ############################################
        new_model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir)
        with self.assertRaises(FileNotFoundError):
            new_model.reload_from_standalone(configuration_path='toto.json', sklearn_pipeline_path=pkl_path)
        remove_dir(model_dir)
        new_model = ModelTfidfSuperDocumentsNaive(model_dir=model_dir)
        with self.assertRaises(FileNotFoundError):
            new_model.reload_from_standalone(configuration_path=conf_path, sklearn_pipeline_path='toto.pkl')
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()