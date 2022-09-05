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
import shutil
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from {{package_name}}.models_training.utils_super_documents import TfidfVectorizerSuperDocuments, TfidfTransformerSuperDocuments
from {{package_name}}.models_training import utils_super_documents

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class tfidfSuperDocumentsTests(unittest.TestCase):
    '''Main class to test utils_super_documents'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_get_super_documents(self):
        '''Test the fit and fit_transform of {{package_name}}.models_training.utils_super_documents.get_super_documents'''

        x_train = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023",
                           "Covid - le point sur des chiffres qui s'envolent en France",
                           "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])
        y_train = np.array(['a', 'a', 'b'])

        x_train_s = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023 Covid - le point sur des chiffres qui s'envolent en France",
                             "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])
        y_train_s = np.array(['a', 'b'])

        super_documents, y_super_documents = utils_super_documents.get_super_documents(x_train, y_train)
        self.assertTrue((super_documents == x_train_s).all())
        self.assertTrue((y_super_documents == y_train_s).all())

    def test02_TfidfVectorizerSuperDocuments_init(self):
        '''Test the fit and fit_transform of {{package_name}}.models_training.utils_super_documents.TfidfVectorizerSuperDocuments.init'''

        x_train_s = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023 Covid - le point sur des chiffres qui s'envolent en France",
                             "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])
        param = {'ngram_range': [2, 3], 'min_df': 0.02, 'max_df': 0.8, 'binary': False}

        vec_trans = TfidfVectorizer().fit_transform(x_train_s).toarray().T
        vec = TfidfVectorizerSuperDocuments(tfidf_super_documents=vec_trans, **param)
        self.assertTrue((vec.tfidf_super_documents == vec_trans).all())
        self.assertEqual(vec.ngram_range, param['ngram_range'])
        self.assertEqual(vec.min_df, param['min_df'])
        self.assertEqual(vec.max_df, param['max_df'])
        self.assertEqual(vec.binary, param['binary'])
        self.assertTrue(isinstance(vec.count_vec, CountVectorizer))
        self.assertEqual(vec.count_vec.ngram_range, param['ngram_range'])
        self.assertEqual(vec.count_vec.min_df, param['min_df'])
        self.assertEqual(vec.count_vec.max_df, param['max_df'])
        self.assertEqual(vec.count_vec.binary, param['binary'])

    def test03_TfidfVectorizerSuperDocuments_fit(self):
        '''Test the fit of {{package_name}}.models_training.utils_super_documents.TfidfVectorizerSuperDocuments.fit'''

        x_train = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023",
                           "Covid - le point sur des chiffres qui s'envolent en France",
                           "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])
        y_train = np.array(['s', 's', 'p'])

        x_train_s = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023 Covid - le point sur des chiffres qui s'envolent en France",
                             "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])

        param = {'ngram_range': [2, 3], 'min_df': 0.02, 'max_df': 0.8, 'binary': False}

        vec = TfidfVectorizerSuperDocuments()
        vec.fit(x_train, y_train)
        vec_trans = TfidfVectorizer().fit_transform(x_train_s).toarray().T
        self.assertTrue((vec.tfidf_super_documents == vec_trans).all())
        count = CountVectorizer().fit(x_train_s)
        self.assertTrue(isinstance(vec.count_vec, CountVectorizer))
        self.assertTrue((vec.count_vec.transform(x_train).toarray() == count.transform(x_train).toarray()).all())

        vec = TfidfVectorizerSuperDocuments(**param)
        vec.fit(x_train, y_train)
        vec_trans = TfidfVectorizer(**param).fit_transform(x_train_s).toarray().T
        self.assertTrue((vec.tfidf_super_documents == vec_trans).all())
        count = CountVectorizer(**param).fit(x_train_s)
        self.assertTrue(isinstance(vec.count_vec, CountVectorizer))
        self.assertTrue((vec.count_vec.transform(x_train).toarray() == count.transform(x_train).toarray()).all())


    def test04_TfidfVectorizerSuperDocuments_transform(self):
        '''Test the transform of {{package_name}}.models_training.utils_super_documents.TfidfVectorizerSuperDocuments.transform'''

        # Set vars
        x_train = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023",
                           "Covid - le point sur des chiffres qui s'envolent en France",
                           "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])
        y_train = np.array(['a', 'a', 'b'])

        x_train_s = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023 Covid - le point sur des chiffres qui s'envolent en France",
                             "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])

        x_test = np.array(["Covid-19 : Le certificat Covid numérique de l'Union Européenne est prolongé d'un an"])
        param = {'ngram_range': [2, 3], 'min_df': 0.02, 'max_df': 0.8, 'binary': False}

        vec = TfidfVectorizerSuperDocuments().fit(x_train, y_train)
        vec_trans = TfidfVectorizer().fit_transform(x_train_s).toarray().T
        count_vec = CountVectorizer().fit(x_train_s)
        count = count_vec.transform(x_test).toarray()
        vec_trans_s = np.dot(count, vec_trans)
        self.assertTrue((vec.transform(x_test).toarray() == vec_trans_s).all())

        vec = TfidfVectorizerSuperDocuments(**param).fit(x_train, y_train)
        vec_trans = TfidfVectorizer(**param).fit_transform(x_train_s).toarray().T
        count_vec = CountVectorizer(**param).fit(x_train_s)
        count = count_vec.transform(x_train).toarray()
        vec_trans_s = np.dot(count, vec_trans)
        self.assertTrue((vec.transform(x_train).toarray() == vec_trans_s).all())

    def test05_TfidfVectorizerSuperDocuments_fit_transform(self):
        '''Test the fit_transform of {{package_name}}.models_training.utils_super_documents.TfidfVectorizerSuperDocuments.fit_transform'''

        x_train = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023",
                           "Covid - le point sur des chiffres qui s'envolent en France",
                           "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])
        y_train = np.array(['s', 's', 'p'])

        x_train_s = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023 Covid - le point sur des chiffres qui s'envolent en France",
                             "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])

        param = {'ngram_range': [2, 3], 'min_df': 0.02, 'max_df': 0.8, 'binary': False}

        vec = TfidfVectorizerSuperDocuments()
        vec_trans = TfidfVectorizer().fit_transform(x_train_s).toarray().T
        count_vec = CountVectorizer().fit(x_train_s)
        count = count_vec.transform(x_train).toarray()
        vec_trans_s = np.dot(count, vec_trans)
        self.assertTrue((vec.fit_transform(x_train, y_train).toarray() == vec_trans_s).all())

        vec = TfidfVectorizerSuperDocuments(**param)
        vec_trans = TfidfVectorizer(**param).fit_transform(x_train_s).toarray().T
        count_vec = CountVectorizer(**param).fit(x_train_s)
        count = count_vec.transform(x_train).toarray()
        vec_trans_s = np.dot(count, vec_trans)
        self.assertTrue((vec.fit_transform(x_train, y_train).toarray() == vec_trans_s).all())

    def test06_TfidfVectorizerSuperDocuments_save(self):
        '''Test the save of {{package_name}}.models_training.utils_super_documents.TfidfVectorizerSuperDocuments.save'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        x_train = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023",
                           "Covid - le point sur des chiffres qui s'envolent en France",
                           "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])
        y_train = np.array(['a', 'a', 'b'])
        param = {'ngram_range': [2, 3], 'min_df': 0.02, 'max_df': 0.8, 'binary': False}

        vec = TfidfVectorizerSuperDocuments(**param).fit(x_train, y_train)
        vec.save(model_dir, level_save='HIGH')
        self.assertTrue(os.path.exists(os.path.join(model_dir, 'count_vectorizer.pkl')))
        self.assertTrue(os.path.exists(os.path.join(model_dir, 'tfidf_super_documents.pkl')))
        remove_dir(model_dir)

        vec = TfidfVectorizerSuperDocuments(**param)
        vec.save(model_dir, level_save='HIGH')
        self.assertTrue(os.path.exists(os.path.join(model_dir, 'count_vectorizer.pkl')))
        self.assertFalse(os.path.exists(os.path.join(model_dir, 'tfidf_super_documents.pkl')))
        remove_dir(model_dir)

    def test07_TfidfVectorizerSuperDocuments_reload_from_standalone(self):
        '''Test the reload_from_standalone of {{package_name}}.models_training.utils_super_documents.TfidfVectorizerSuperDocuments.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Create model
        x_train = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023",
                           "Covid - le point sur des chiffres qui s'envolent en France",
                           "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])
        y_train = np.array(['a', 'a', 'b'])
        param = {'ngram_range': [2, 3], 'min_df': 0.02, 'max_df': 0.8, 'binary': False}
        vec = TfidfVectorizerSuperDocuments(**param).fit(x_train, y_train)
        vec.save(model_dir, level_save='HIGH')
        self.assertTrue(os.path.exists(os.path.join(model_dir, 'count_vectorizer.pkl')))
        self.assertTrue(os.path.exists(os.path.join(model_dir, 'tfidf_super_documents.pkl')))

        # Reload
        count_vectorizer_path = os.path.join(model_dir, f"count_vectorizer.pkl")
        tfidf_super_documents_path = os.path.join(model_dir, "tfidf_super_documents.pkl")
        new_vec = TfidfVectorizerSuperDocuments()
        new_vec.reload_from_standalone(count_vectorizer_path=count_vectorizer_path, tfidf_super_documents_path=tfidf_super_documents_path)

        # Test
        self.assertTrue((vec.tfidf_super_documents == new_vec.tfidf_super_documents).all())
        self.assertTrue(isinstance(vec.count_vec, CountVectorizer))
        self.assertEqual(vec.count_vec.ngram_range, param['ngram_range'])
        self.assertEqual(vec.count_vec.min_df, param['min_df'])
        self.assertEqual(vec.count_vec.max_df, param['max_df'])
        self.assertEqual(vec.count_vec.binary, param['binary'])


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
