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
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class PreprocessTests(unittest.TestCase):
    '''Main class to test all functions in {{package_name}}.preprocessing.preprocess'''


    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_get_pipelines_dict(self):
        '''Test of the function preprocess.get_pipelines_dict'''
        # Valids to test
        # TODO: to modify depending on your project !
        content = pd.DataFrame({'col_1': [-5, -1, 0, 2, -6, 3], 'col_2': [2, -1, -8, 3, 12, 2],
                                'text': ['toto', 'titi', 'tata', 'tutu', 'tyty', 'tete']})
        y = pd.Series([0, 1, 1, 1, 0, 0])

        # Nominal case
        pipelines_dict = preprocess.get_pipelines_dict()
        self.assertEqual(type(pipelines_dict), dict)
        self.assertTrue('no_preprocess' in pipelines_dict.keys())

        # We test each returned function
        for p in pipelines_dict.values():
            p.fit(content, y)
            self.assertEqual(type(p.transform(content)), np.ndarray)


    def test02_get_pipeline(self):
        '''Test of the function preprocess.get_pipeline'''
        # Valids to test
        # We take a preprocessing 'at random'
        pipeline_str = list(preprocess.get_pipelines_dict().keys())[0]

        # Nominal case
        pipeline = preprocess.get_pipeline(pipeline_str)
        # We just test if wa have a pipeline
        self.assertEqual(type(pipeline), ColumnTransformer)

        # Check the input(s) type(s)
        with self.assertRaises(ValueError):
            preprocess.get_pipeline('NOT A VALID PREPROCESS')


    def test03_retrieve_columns_from_pipeline(self):
        '''Test of the function preprocess.retrieve_columns_from_pipeline'''
        # Pipeline
        col_1_3_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        col_2_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
        text_pipeline = make_pipeline(CountVectorizer(), SelectKBest(k=2))
        transformers = [
            ('col_1_3', col_1_3_pipeline, ['col_1', 'col_3']),
            ('col_2', col_2_pipeline, ['col_2']),
            ('text', text_pipeline, 'text'),
        ]
        pipeline = ColumnTransformer(transformers, remainder='drop')
        # DataFrame
        df = pd.DataFrame({'col_1': [1, 5, 8, 4], 'col_2': [0.0, None, 1.0, 1.0], 'col_3': [-5, 6, 8, 6],
                           'toto': [4, 8, 9, 4],
                           'text': ['ceci est un test', 'un autre test', 'et un troisième test', 'et un dernier']})
        # Target
        y = pd.Series([1, 1, 1, 0])
        # Fit
        pipeline.fit(df, y)
        # transform
        transformed_df = pd.DataFrame(pipeline.transform(df))

        # Nominal case
        new_transformed_df = preprocess.retrieve_columns_from_pipeline(transformed_df, pipeline)
        self.assertEqual(list(new_transformed_df.columns), ['col_1', 'col_3', 'col_2_0.0', 'col_2_1.0', 'vec_dernier', 'vec_test'])

        # If unfitted pipeline, no modifications
        tmp_pipeline = ColumnTransformer(transformers, remainder='drop')
        new_transformed_df = preprocess.retrieve_columns_from_pipeline(transformed_df, tmp_pipeline)
        pd.testing.assert_frame_equal(new_transformed_df, transformed_df)

        # If there is not the right number of columns, no modifications
        new_transformed_df = preprocess.retrieve_columns_from_pipeline(df, pipeline)
        pd.testing.assert_frame_equal(new_transformed_df, df)


    def test04_get_feature_out(self):
        '''Test of the function preprocess.get_feature_out'''

        # Nominal case - non _VectorizerMixin - non SelectorMixin
        estimator = SimpleImputer(strategy='median')
        estimator.fit(pd.DataFrame({'col': [1, 0, 1, 1, None]}))
        feature_out = preprocess.get_feature_out(estimator, 'toto')
        self.assertEqual(feature_out, 'toto')
        feature_out = preprocess.get_feature_out(estimator, ['toto', 'tata'])
        self.assertEqual(feature_out, ['toto', 'tata'])

        # Nominal case - _VectorizerMixin
        estimator = OneHotEncoder(handle_unknown='ignore')
        estimator.fit(pd.DataFrame({'col': [0, 0, 1, 1, 0]}))
        feature_out = preprocess.get_feature_out(estimator, ['toto'])
        self.assertEqual(list(feature_out), ['toto_0', 'toto_1'])

        # Nominal case - SelectorMixin
        estimator = SelectKBest(k=2)
        estimator.fit(pd.DataFrame({'col_1': [0, 0, 1, 1, 1], 'col_2': [1, 1, 0, 0, 0], 'col_3': [0, 0, 0, 0, 0]}), pd.Series([-1, -1, 1, 1, 1]))
        feature_out = preprocess.get_feature_out(estimator, ['col_1', 'col_2', 'col_3'])
        self.assertEqual(list(feature_out), ['col_1', 'col_2'])


    def test05_get_ct_feature_names(self):
        '''Test of the function preprocess.get_ct_feature_names'''
        # Pipeline
        col_1_3_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        col_2_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
        text_pipeline = make_pipeline(CountVectorizer(), SelectKBest(k=2))
        transformers = [
            ('col_1_3', col_1_3_pipeline, ['col_1', 'col_3']),
            ('col_2', col_2_pipeline, ['col_2']),
            ('text', text_pipeline, 'text'),
        ]
        pipeline = ColumnTransformer(transformers, remainder='drop')
        # DataFrame
        df = pd.DataFrame({'col_1': [1, 5, 8, 4], 'col_2': [0.0, None, 1.0, 1.0], 'col_3': [-5, 6, 8, 6],
                           'toto': [4, 8, 9, 4],
                           'text': ['ceci est un test', 'un autre test', 'et un troisième test', 'et un dernier']})
        # Target
        y = pd.Series([1, 1, 1, 0])
        # Fit
        pipeline.fit(df, y)

        # Nominal case
        output_features = preprocess.get_ct_feature_names(pipeline)
        self.assertEqual(output_features, ['col_1', 'col_3', 'col_2_0.0', 'col_2_1.0', 'vec_dernier', 'vec_test'])

        # remainder == 'passthrough'
        pipeline = ColumnTransformer(transformers, remainder='passthrough')
        pipeline.fit(df, y)
        output_features = preprocess.get_ct_feature_names(pipeline)
        self.assertEqual(output_features, ['col_1', 'col_3', 'col_2_0.0', 'col_2_1.0', 'vec_dernier', 'vec_test', 'toto'])


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
