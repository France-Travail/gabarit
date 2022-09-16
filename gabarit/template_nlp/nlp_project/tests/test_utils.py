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

from {{package_name}} import utils

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class UtilsTests(unittest.TestCase):
    '''Main class to test all functions in utils.py'''
    # We avoid tqdm prints
    pd.Series.progress_apply = pd.Series.apply

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    @patch('logging.Logger._log')
    def test01_read_csv(self, PrintMockLog):
        '''Test of the function utils.read_csv'''

        # Arguments df1
        path1 = './test_dataset.csv'
        sep1 = ','
        df1_shape_expected = (96, 5)
        first_line1_expected = None

        # Arguments df2
        path2 = './test_dataset2.csv'
        sep2 = ';'
        df2_expected = pd.DataFrame([["Ceci est un test", '1', "ceci est un test"],
                                     ["Ceci est un autre test; avec un point-virgule", '0', "ceci est un autre test avec un point virgule"]] * 15,
                                    columns=["x_col", "y_col", "preprocessed_text"])
        first_line2_expected = '#preprocess_P1'

        # Nominal case
        df1, first_line1 = utils.read_csv(path1, sep=sep1, encoding='utf-8', dtype=str)
        df2, first_line2 = utils.read_csv(path2, sep=sep2, encoding='utf-8', dtype=str)
        self.assertEqual(df1.shape, df1_shape_expected)
        self.assertEqual(first_line1, first_line1_expected)
        pd.testing.assert_frame_equal(df2, df2_expected)
        self.assertEqual(first_line2, first_line2_expected)

        # Testing kwargs
        df3_expected = pd.DataFrame([["Ceci est un test", '1', "ceci est un test"],
                                     ["Ceci est un autre test; avec un point-virgule", '0', "ceci est un autre test avec un point virgule"]] * 2,
                                    columns=["x_col", "y_col", "preprocessed_text"])
        df3, first_line3 = utils.read_csv(path2, sep=sep2, encoding='utf-8', dtype=str, nrows=4)
        pd.testing.assert_frame_equal(df3, df3_expected)

        # Check errors
        with self.assertRaises(ValueError):
            utils.read_csv('test_utils.py')
        with self.assertRaises(FileNotFoundError):
            utils.read_csv('toto.csv')

    @patch('logging.Logger._log')
    def test02_to_csv(self, PrintMockLog):
        '''Test of the function utils.to_csv'''
        # Data
        df = pd.DataFrame([['test', 'test'], ['toto', 'titi'], ['tata', 'tutu']], columns=['col1', 'col2'])
        fake_filepath = 'fake_csv.csv'
        # Clear
        if os.path.exists(fake_filepath):
            os.remove(fake_filepath)

        # Nominal case
        utils.to_csv(df, fake_filepath, first_line=None, sep='{{default_sep}}', encoding='{{default_encoding}}')
        self.assertTrue(os.path.exists(fake_filepath))
        reloaded_df = pd.read_csv(fake_filepath, sep='{{default_sep}}', encoding='{{default_encoding}}')
        pd.testing.assert_frame_equal(df, reloaded_df)

        # With first_line
        os.remove(fake_filepath)
        first_line = 'ligne1'
        utils.to_csv(df, fake_filepath, first_line=first_line, sep='{{default_sep}}', encoding='{{default_encoding}}')
        self.assertTrue(os.path.exists(fake_filepath))
        with open(fake_filepath, 'r', encoding='utf-8') as f:
            first_line_realoaded = f.readline().replace('\n', '').replace('\r', '')
        self.assertEqual(first_line, first_line_realoaded)
        reloaded_df2 = pd.read_csv(fake_filepath, sep='{{default_sep}}', encoding='{{default_encoding}}', skiprows=1)
        pd.testing.assert_frame_equal(df, reloaded_df2)

        # Clear
        if os.path.exists(fake_filepath):
            os.remove(fake_filepath)

    @patch('logging.Logger._log')
    def test03_display_shape(self, PrintMockLog):
        '''Test of the function utils.display_shape'''
        # Enable the logger again
        logging.disable(logging.NOTSET)

        # Valids to test
        input_test = pd.DataFrame({'col1': ['A', 'A', 'D', np.nan, 'C', 'B'],
                                   'col2': [0, 0, 1, 2, 3, 1],
                                   'col3': [0, 0, 3, np.nan, np.nan, 3]})

        # Assert info called 1 time
        utils.display_shape(input_test)
        self.assertEqual(len(PrintMockLog.mock_calls), 1)

        # RESET DEFAULT
        logging.disable(logging.CRITICAL)

    def test04_get_new_column_name(self):
        '''Test of the function utils.get_new_column_name'''
        # Valids to test
        column_list = ['toto', 'titi']

        # Nominal cases
        self.assertEqual(utils.get_new_column_name(column_list, 'tata'), 'tata')
        #
        new_col = utils.get_new_column_name(column_list, 'toto')
        self.assertTrue(new_col.startswith('toto_'))
        self.assertEqual(len(new_col), len('toto_') + 8)

    def test05_get_chunk_limits(self):
        '''Test of the function utils.get_chunk_limits'''
        # Valids to test
        input_test_1 = pd.DataFrame()
        expected_result_1 = [(0, 0)]
        input_test_2 = pd.DataFrame([0])
        expected_result_2 = [(0, 1)]
        input_test_3 = pd.DataFrame([0] * 100000)
        expected_result_3 = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000),
                             (40000, 50000), (50000, 60000), (60000, 70000), (70000, 80000),
                             (80000, 90000), (90000, 100000)]
        input_test_4 = pd.DataFrame([0] * 100001)
        expected_result_4 = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000),
                             (40000, 50000), (50000, 60000), (60000, 70000), (70000, 80000),
                             (80000, 90000), (90000, 100000), (100000, 100001)]
        input_test_5 = pd.DataFrame([0] * 100)
        expected_result_5 = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60),
                             (60, 70), (70, 80), (80, 90), (90, 100)]

        # Nominal case
        self.assertEqual(utils.get_chunk_limits(input_test_1), expected_result_1)
        self.assertEqual(utils.get_chunk_limits(input_test_2), expected_result_2)
        self.assertEqual(utils.get_chunk_limits(input_test_3), expected_result_3)
        self.assertEqual(utils.get_chunk_limits(input_test_4), expected_result_4)
        self.assertEqual(utils.get_chunk_limits(input_test_5, chunksize=10), expected_result_5)

        # Check the input(s) type(s)
        with self.assertRaises(ValueError):
            utils.get_chunk_limits(input_test_1, chunksize=-1)

    def test06_get_data_path(self):
        '''Test of the function utils.get_data_path'''
        # Nominal case
        path = utils.get_data_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path.endswith('{{package_name}}-data'), True)

        # With a DIR_PATH != None
        current_dir = os.path.abspath(os.getcwd())
        utils.DIR_PATH = current_dir
        path = utils.get_data_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path, os.path.join(current_dir, '{{package_name}}-data'))
        utils.DIR_PATH = None

    def test07_get_models_path(self):
        '''Test of the function utils.get_models_path'''
        # Nominal case
        path = utils.get_models_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path.endswith('{{package_name}}-models'), True)

        # With a DIR_PATH != None
        current_dir = os.path.abspath(os.getcwd())
        utils.DIR_PATH = current_dir
        path = utils.get_models_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path, os.path.join(current_dir, '{{package_name}}-models'))
        utils.DIR_PATH = None

    def test08_get_ressources_path(self):
        '''Test of the function utils.get_ressources_path'''
        # Nominal case
        path = utils.get_ressources_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path.endswith('{{package_name}}-ressources'), True)

    def test09_get_package_version(self):
        '''Test of the function utils.get_package_version'''
        # Nominal case
        version = utils.get_package_version()
        self.assertEqual(type(version), str)


# TODO: test trained_needed & data_agnostic_str_to_list


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
