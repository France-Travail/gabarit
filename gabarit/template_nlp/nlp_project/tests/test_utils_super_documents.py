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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from {{package_name}} import utils
from {{package_name}}.models_training.utils_super_documents import TfidfVectorizerSuperDocuments, TfidfTransformerSuperDocuments


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


    def test01_model_pipeline_TfidfVectorizerSuperDocuments(self):
        '''Test of the fit and fit_transform tfidfDemo.models_training.utils_super_documents.TfidfVectorizerSuperDocuments'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        corpus = np.array([
                        "Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023",
                        "Covid - le point sur des chiffres qui s'envolent en France",
                        "Carte des résultats des législatives : les qualifiés circonscription par circonscription",
                            ])
        target = np.array(['s','s','p'])

        corpus_s = np.array([
                        "Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023 Covid - le point sur des chiffres qui s'envolent en France",
                        "Carte des résultats des législatives : les qualifiés circonscription par circonscription",
                            ])
        target_s = np.array(['s', 'p'])

        param = {'ngram_range': [2,3], 'min_df': 0.02, 'max_df': 0.8, 'binary': False}

        # test fit
        tfidf = TfidfVectorizer().fit(corpus_s, target_s)
        tfidf_s = TfidfVectorizerSuperDocuments().fit(corpus, target)
        self.assertEqual(tfidf.transform(corpus).toarray().all(), tfidf_s.transform(corpus).toarray().all())
        remove_dir(model_dir)

        # test fit with parameters
        tfidf = TfidfVectorizer(**param).fit(corpus_s, target_s)
        tfidf_s = TfidfVectorizerSuperDocuments(**param).fit(corpus, target)
        self.assertEqual(tfidf.transform(corpus).toarray().all(), tfidf_s.transform(corpus).toarray().all())
        remove_dir(model_dir)

        # test fit_transform
        tfidf = TfidfVectorizer().fit(corpus_s, target_s)
        tfidf_s = TfidfVectorizerSuperDocuments()
        self.assertEqual(tfidf.transform(corpus).toarray().all(), tfidf_s.fit_transform(corpus, target).toarray().all())
        remove_dir(model_dir)

    
    def test02_model_pipeline_get_super_documents_count_vectorizer(self):
        '''Test of the method get_super_documents_count_vectorizer of tfidfDemo.models_training.utils_super_documents'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        corpus = np.array([
                        "Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023",
                        "Covid - le point sur des chiffres qui s'envolent en France",
                        "Carte des résultats des législatives : les qualifiés circonscription par circonscription",
                            ])
        target = np.array(['a','a','b'])

        corpus_s = np.array([
                        "Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023 Covid - le point sur des chiffres qui s'envolent en France",
                        "Carte des résultats des législatives : les qualifiés circonscription par circonscription",
                            ])
        target_s = np.array(['a', 'b'])

        count_vec = CountVectorizer().fit_transform(corpus, target)
        super_documents, _ = TfidfTransformerSuperDocuments().get_super_documents_count_vectorizer(count_vec, target)
        self.assertEqual(super_documents.all(), CountVectorizer().fit_transform(corpus_s, target_s).toarray().all())
        remove_dir(model_dir)


    def test03_model_pipeline_TfidfTransformerSuperDocuments(self):
        '''Test of the fit and fit_transform tfidfDemo.models_training.utils_super_documents.TfidfTransformerSuperDocuments'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        corpus = np.array([
                        "Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023",
                        "Covid - le point sur des chiffres qui s'envolent en France",
                        "Carte des résultats des législatives : les qualifiés circonscription par circonscription",
                            ])
        target = np.array(['s','s','p'])

        corpus_s = np.array([
                        "Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023 Covid - le point sur des chiffres qui s'envolent en France",
                        "Carte des résultats des législatives : les qualifiés circonscription par circonscription",
                            ])
        target_s = np.array(['s', 'p'])

        param = {'ngram_range': [2,3], 'min_df': 0.02, 'max_df': 0.8, 'binary': False}

        # test fit
        pipe = Pipeline([('count', CountVectorizer()),
                            ('tfidf', TfidfTransformer())]).fit(corpus_s, target_s)
        pipe_s = Pipeline([('count', CountVectorizer()),
                            ('tfidf', TfidfTransformerSuperDocuments())]).fit(corpus, target)
        self.assertEqual(pipe.transform(corpus).toarray().all(), pipe_s.transform(corpus).toarray().all())
        remove_dir(model_dir)

        # test fit with parameters
        pipe = Pipeline([('count', CountVectorizer(**param)),
                            ('tfidf', TfidfTransformer())]).fit(corpus_s, target_s)
        pipe_s = Pipeline([('count', CountVectorizer(**param)),
                            ('tfidf', TfidfTransformerSuperDocuments())]).fit(corpus, target)
        self.assertEqual(pipe.transform(corpus).toarray().all(), pipe_s.fit_transform(corpus).toarray().all())
        remove_dir(model_dir)

        # test fit_transform
        pipe = Pipeline([('count', CountVectorizer()),
                            ('tfidf', TfidfTransformer())]).fit(corpus_s, target_s)
        pipe_s = Pipeline([('count', CountVectorizer()),
                            ('tfidf', TfidfTransformerSuperDocuments())])
        self.assertEqual(pipe.transform(corpus).toarray().all(), pipe_s.fit_transform(corpus, target).toarray().all())
        remove_dir(model_dir)


    def test04_model_pipeline_get_super_documents(self):
        '''Test of the method get_super_documents of tfidfDemo.models_training.utils_super_documents'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        corpus = np.array([
                        "Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023",
                        "Covid - le point sur des chiffres qui s'envolent en France",
                        "Carte des résultats des législatives : les qualifiés circonscription par circonscription",
                            ])
        target = np.array(['s','s','p'])

        corpus_s = np.array([
                        "Covid - Omicron : l'Europe veut prolonger le certificat Covid jusqu'en 2023 Covid - le point sur des chiffres qui s'envolent en France",
                        "Carte des résultats des législatives : les qualifiés circonscription par circonscription",
                            ])
        target_s = np.array(['s', 'p'])

        super_documents, _ = TfidfVectorizerSuperDocuments().get_super_documents(corpus, target)
        self.assertEqual(super_documents['s'], corpus_s[0])
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
