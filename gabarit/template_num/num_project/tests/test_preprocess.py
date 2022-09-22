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
            ('tr1', col_1_3_pipeline, ['col_1', 'col_3']),
            ('tr2', col_2_pipeline, ['col_2']),
            ('tr3', text_pipeline, 'text'),
        ]
        pipeline = ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)
        pipeline_verbose = ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=True)
        pipeline_passthrough = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False)
        pipeline_passthrough_verbose = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=True)
        # DataFrame
        df = pd.DataFrame({'col_1': [1, 5, 8, 4], 'col_2': [0.0, None, 1.0, 1.0], 'col_3': [-5, 6, 8, 6],
                           'toto': [4, 8, 9, 4],
                           'text': ['ceci est un test', 'un autre test', 'et un troisième test', 'et un dernier']})
        # Target
        y = pd.Series([1, 1, 1, 0])
        # Fit
        pipeline.fit(df, y)
        pipeline_verbose.fit(df, y)
        pipeline_passthrough.fit(df, y)
        pipeline_passthrough_verbose.fit(df, y)
        # transform
        transformed_df = pd.DataFrame(pipeline.transform(df))
        transformed_df_verbose = pd.DataFrame(pipeline_verbose.transform(df))
        transformed_df_passthrough = pd.DataFrame(pipeline_passthrough.transform(df))
        transformed_df_passthrough_verbose = pd.DataFrame(pipeline_passthrough_verbose.transform(df))

        # Nominal case
        new_transformed_df = preprocess.retrieve_columns_from_pipeline(transformed_df, pipeline)
        new_transformed_df_verbose = preprocess.retrieve_columns_from_pipeline(transformed_df_verbose, pipeline_verbose)
        new_transformed_df_passthrough = preprocess.retrieve_columns_from_pipeline(transformed_df_passthrough, pipeline_passthrough)
        new_transformed_df_passthrough_verbose = preprocess.retrieve_columns_from_pipeline(transformed_df_passthrough_verbose, pipeline_passthrough_verbose)
        self.assertEqual(list(new_transformed_df.columns), ['col_1', 'col_3', 'col_2_0.0', 'col_2_1.0', 'dernier', 'test'])
        self.assertEqual(list(new_transformed_df_verbose.columns), ['tr1__col_1', 'tr1__col_3', 'tr2__col_2_0.0', 'tr2__col_2_1.0', 'tr3__dernier', 'tr3__test'])
        self.assertEqual(list(new_transformed_df_passthrough.columns), ['col_1', 'col_3', 'col_2_0.0', 'col_2_1.0', 'dernier', 'test', 'toto'])
        self.assertEqual(list(new_transformed_df_passthrough_verbose.columns), ['tr1__col_1', 'tr1__col_3', 'tr2__col_2_0.0', 'tr2__col_2_1.0', 'tr3__dernier', 'tr3__test', 'remainder__toto'])

        # If there isn't the right number of columns, backup solution
        # We consider df, but the idea is to make it work for any dataframe
        new_transformed_df = preprocess.retrieve_columns_from_pipeline(df, pipeline)
        new_transformed_df_verbose = preprocess.retrieve_columns_from_pipeline(df, pipeline_verbose)
        new_transformed_df_passthrough = preprocess.retrieve_columns_from_pipeline(df, pipeline_passthrough)
        new_transformed_df_passthrough_verbose = preprocess.retrieve_columns_from_pipeline(df, pipeline_passthrough_verbose)
        target_df = df.copy()
        target_df.columns = ['x0', 'x1', 'x2', 'x3', 'x4']
        pd.testing.assert_frame_equal(new_transformed_df, target_df)
        pd.testing.assert_frame_equal(new_transformed_df_verbose, target_df)
        pd.testing.assert_frame_equal(new_transformed_df_passthrough, target_df)
        pd.testing.assert_frame_equal(new_transformed_df_passthrough_verbose, target_df)

        # If unfitted pipeline, raise an error
        tmp_pipeline = ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)
        tmp_pipeline_verbose = ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=True)
        tmp_pipeline_passthrough = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False)
        tmp_pipeline_passthrough_verbose = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=True)
        with self.assertRaises(AttributeError):
            new_transformed_df = preprocess.retrieve_columns_from_pipeline(df, tmp_pipeline)
        with self.assertRaises(AttributeError):
            new_transformed_df_verbose = preprocess.retrieve_columns_from_pipeline(df, tmp_pipeline_verbose)
        with self.assertRaises(AttributeError):
            new_transformed_df_passthrough = preprocess.retrieve_columns_from_pipeline(df, tmp_pipeline_passthrough)
        with self.assertRaises(AttributeError):
            new_transformed_df_passthrough_verbose = preprocess.retrieve_columns_from_pipeline(df, tmp_pipeline_passthrough_verbose)


    def test04_get_ct_feature_names(self):
        '''Test of the function preprocess.get_ct_feature_names'''
        # Pipeline
        col_1_3_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        col_2_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
        text_pipeline = make_pipeline(CountVectorizer(), SelectKBest(k=2))
        transformers = [
            ('tr1', col_1_3_pipeline, ['col_1', 'col_3']),
            ('tr2', col_2_pipeline, ['col_2']),
            ('tr3', text_pipeline, 'text'),
        ]
        pipeline = ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)
        pipeline_verbose = ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=True)
        # DataFrame
        df = pd.DataFrame({'col_1': [1, 5, 8, 4], 'col_2': [0.0, None, 1.0, 1.0], 'col_3': [-5, 6, 8, 6],
                           'toto': [4, 8, 9, 4],
                           'text': ['ceci est un test', 'un autre test', 'et un troisième test', 'et un dernier']})
        # Target
        y = pd.Series([1, 1, 1, 0])
        # Fit
        pipeline.fit(df, y)
        pipeline_verbose.fit(df, y)

        # Nominal case
        output_features = preprocess.get_ct_feature_names(pipeline)
        output_features_verbose = preprocess.get_ct_feature_names(pipeline_verbose)
        self.assertEqual(output_features, ['col_1', 'col_3', 'col_2_0.0', 'col_2_1.0', 'dernier', 'test'])
        self.assertEqual(output_features_verbose, ['tr1__col_1', 'tr1__col_3', 'tr2__col_2_0.0', 'tr2__col_2_1.0', 'tr3__dernier', 'tr3__test'])

        # remainder == 'passthrough'
        pipeline = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False)
        pipeline_verbose = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=True)
        pipeline.fit(df, y)
        pipeline_verbose.fit(df, y)
        output_features = preprocess.get_ct_feature_names(pipeline)
        output_features_verbose = preprocess.get_ct_feature_names(pipeline_verbose)
        self.assertEqual(output_features, ['col_1', 'col_3', 'col_2_0.0', 'col_2_1.0', 'dernier', 'test', 'toto'])
        self.assertEqual(output_features_verbose, ['tr1__col_1', 'tr1__col_3', 'tr2__col_2_0.0', 'tr2__col_2_1.0', 'tr3__dernier', 'tr3__test', 'remainder__toto'])


    def test05_get_feature_out(self):
        '''Test of the function preprocess.get_feature_out'''

        # Nominal case - non _VectorizerMixin - non SelectorMixin
        estimator = SimpleImputer(strategy='median')
        estimator.fit(pd.DataFrame({'col': [1, 0, 1, 1, None]}))
        feature_out = preprocess.get_feature_out(estimator, 'toto')
        self.assertEqual(feature_out, 'toto')
        feature_out = preprocess.get_feature_out(estimator, ['toto', 'tata'])  # Uses inputed cols
        self.assertEqual(feature_out, ['toto', 'tata'])

        # Nominal case 2
        estimator = OneHotEncoder(handle_unknown='ignore')
        estimator.fit(pd.DataFrame({'col': [0, 0, 1, 1, 0]}))
        feature_out = preprocess.get_feature_out(estimator, ['toto'])  # Uses inputed cols
        self.assertEqual(list(feature_out), ['toto_0', 'toto_1'])

        # Nominal case - _VectorizerMixin
        estimator = CountVectorizer()
        estimator.fit(['ceci test', 'cela aussi', 'ceci cela'])
        feature_out = preprocess.get_feature_out(estimator, ['toto'])  # Does not use inputed cols
        self.assertEqual(list(feature_out), ['aussi', 'ceci', 'cela', 'test'])

        # Nominal case - SelectorMixin
        estimator = SelectKBest(k=2)
        estimator.fit(pd.DataFrame({'col_1': [0, 0, 1, 1, 1], 'col_2': [1, 1, 0, 0, 0], 'col_3': [0, 0, 0, 0, 0]}), pd.Series([-1, -1, 1, 1, 1]))
        feature_out = preprocess.get_feature_out(estimator, ['toto_1', 'toto_2', 'toto_3'])  # Uses inputed cols
        self.assertEqual(list(feature_out), ['toto_1', 'toto_2'])


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
