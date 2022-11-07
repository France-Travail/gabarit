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

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from {{package_name}}.models_training.utils_super_documents import TfidfVectorizerSuperDocuments

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

    def test01_TfidfVectorizerSuperDocuments_init(self):
        '''Test the init of {{package_name}}.models_training.utils_super_documents.TfidfVectorizerSuperDocuments.init'''

        param = {'ngram_range': [2, 3], 'min_df': 0.02, 'max_df': 0.8, 'binary': False}

        vec = TfidfVectorizerSuperDocuments(**param)
        self.assertTrue(vec.tfidf_super_documents is None)
        self.assertEqual(vec.ngram_range, param['ngram_range'])
        self.assertEqual(vec.min_df, param['min_df'])
        self.assertEqual(vec.max_df, param['max_df'])
        self.assertEqual(vec.binary, param['binary'])
        self.assertTrue(isinstance(vec.classes_, np.ndarray))
        self.assertTrue(vec.classes_.shape, (0))

    def test02_get_super_documents(self):
        '''Test the get_super_documents of {{package_name}}.models_training.utils_super_documents.TfidfVectorizerSuperDocuments.get_super_documents'''

        x_train = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023",
                           "Covid - le point sur des chiffres qui s'envolent en France",
                           "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])
        y_train = np.array(['a', 'a', 'b'])

        x_train_s = np.array(["Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023 Covid - le point sur des chiffres qui s'envolent en France",
                             "Carte des résultats des législatives : les qualifiés circonscription par circonscription"])
        y_train_s = np.array(['a', 'b'])

        super_documents, y_super_documents = TfidfVectorizerSuperDocuments().get_super_documents(x_train, y_train)
        self.assertTrue((super_documents == x_train_s).all())
        self.assertTrue((y_super_documents == y_train_s).all())

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
        self.assertTrue((vec.classes_ == np.array(['s', 'p'])).all())

        vec = TfidfVectorizerSuperDocuments(**param)
        vec.fit(x_train, y_train)
        vec_trans = TfidfVectorizer(**param).fit_transform(x_train_s).toarray().T
        self.assertTrue((vec.tfidf_super_documents == vec_trans).all())
        self.assertTrue((vec.classes_ == np.array(['s', 'p'])).all())

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
        self.assertTrue((vec.classes_ == np.array(['s', 'p'])).all())

        vec = TfidfVectorizerSuperDocuments(**param)
        vec_trans = TfidfVectorizer(**param).fit_transform(x_train_s).toarray().T
        count_vec = CountVectorizer(**param).fit(x_train_s)
        count = count_vec.transform(x_train).toarray()
        vec_trans_s = np.dot(count, vec_trans)
        self.assertTrue((vec.fit_transform(x_train, y_train).toarray() == vec_trans_s).all())
        self.assertTrue((vec.classes_ == np.array(['s', 'p'])).all())


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
