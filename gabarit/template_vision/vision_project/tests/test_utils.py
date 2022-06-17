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

    def test01_read_folder_classification(self):
        '''Test of the function utils.read_folder_classification'''
        # We set DIR_PATH equal to the test directory in order to use test data
        current_dir = os.path.abspath(os.getcwd())
        test_data_path = os.path.join(current_dir, 'test_data')
        utils.DIR_PATH = test_data_path

        #######################################
        # Format 1 : metadata file
        #######################################
        sorted_file_list = ['Birman_1.jpg', 'Birman_2.jpg', 'Birman_4.jpg', 'Bombay_1.png', 'Bombay_10.jpg',
                            'Bombay_3.jpg', 'shiba_inu_15.jpg', 'shiba_inu_30.jpg', 'shiba_inu_31.jpg',
                            'shiba_inu_34.jpg', 'subfolder/Birman_36.jpg', 'subfolder/Bombay_19.jpg']
        sorted_classes_list = ['birman', 'birman', 'birman', 'bombay', 'bombay', 'bombay', 'shiba', 'shiba', 'shiba', 'shiba', 'birman', 'bombay']

        # Nominal case
        directory_path = os.path.join(utils.get_data_path(), 'test_data_1')
        path_list, classes_list, preprocess_str = utils.read_folder_classification(directory_path)
        wanted_file_list = [os.path.join(test_data_path, '{{package_name}}-data', 'test_data_1', f) for f in sorted_file_list]
        self.assertEqual(sorted(path_list), wanted_file_list)
        self.assertEqual([cl for p, cl in sorted(zip(path_list, classes_list), key=lambda pair: pair[0])], sorted_classes_list)
        self.assertEqual(preprocess_str, 'no_preprocess')

        # With different sep
        directory_path = os.path.join(utils.get_data_path(), 'test_data_1_diff_sep')
        path_list, classes_list, preprocess_str = utils.read_folder_classification(directory_path, sep=',')
        wanted_file_list = [os.path.join(test_data_path, '{{package_name}}-data', 'test_data_1_diff_sep', f) for f in sorted_file_list]
        self.assertEqual(sorted(path_list), wanted_file_list)
        self.assertEqual([cl for p, cl in sorted(zip(path_list, classes_list), key=lambda pair: pair[0])], sorted_classes_list)
        self.assertEqual(preprocess_str, 'no_preprocess')

        # With preprocessing
        directory_path = os.path.join(utils.get_data_path(), 'test_data_1_with_preprocess')
        path_list, classes_list, preprocess_str = utils.read_folder_classification(directory_path)
        wanted_file_list = [os.path.join(test_data_path, '{{package_name}}-data', 'test_data_1_with_preprocess', f) for f in sorted_file_list]
        self.assertEqual(sorted(path_list), wanted_file_list)
        self.assertEqual([cl for p, cl in sorted(zip(path_list, classes_list), key=lambda pair: pair[0])], sorted_classes_list)
        self.assertEqual(preprocess_str, 'test_preprocess')

        # Without classes
        directory_path = os.path.join(utils.get_data_path(), 'test_data_1_no_class')
        path_list, classes_list, preprocess_str = utils.read_folder_classification(directory_path, accept_no_metadata=True)
        wanted_file_list = [os.path.join(test_data_path, '{{package_name}}-data', 'test_data_1_no_class', f) for f in sorted_file_list]
        self.assertEqual(sorted(path_list), wanted_file_list)
        self.assertEqual(classes_list, None)
        self.assertEqual(preprocess_str, 'no_preprocess')

        # Check errors
        directory_path = os.path.join(utils.get_data_path(), 'bad_test_data_1')
        with self.assertRaises(ValueError):
            utils.read_folder_classification(directory_path)  # No filename column
        directory_path = os.path.join(utils.get_data_path(), 'test_data_1_no_class')
        with self.assertRaises(ValueError):
            path_list, classes_list, preprocess_str = utils.read_folder_classification(directory_path, accept_no_metadata=False)

        #######################################
        # Format 2 : prefix class
        #######################################
        sorted_file_list = ['birman_Birman_1.jpg', 'birman_Birman_2.jpg', 'birman_Birman_36.jpg', 'birman_Birman_4.jpg',
                            'bombay_Bombay_1.png', 'bombay_Bombay_10.jpg', 'bombay_Bombay_19.jpg', 'bombay_Bombay_3.jpg',
                            'shiba_shiba_inu_15.jpg', 'shiba_shiba_inu_30.jpg', 'shiba_shiba_inu_31.jpg', 'shiba_shiba_inu_34.jpg']
        sorted_file_list_no_class = ['Birman1.jpg', 'Birman2.jpg', 'Birman36.jpg', 'Birman4.jpg', 'Bombay1.png', 'Bombay10.jpg',
                                     'Bombay19.jpg', 'Bombay3.jpg', 'shiba15.jpg', 'shiba30.jpg', 'shiba31.jpg', 'shiba34.jpg']
        sorted_classes_list = ['birman', 'birman', 'birman', 'birman', 'bombay', 'bombay', 'bombay', 'bombay', 'shiba', 'shiba', 'shiba', 'shiba']

        # Nominal case
        directory_path = os.path.join(utils.get_data_path(), 'test_data_2')
        path_list, classes_list, preprocess_str = utils.read_folder_classification(directory_path)
        wanted_file_list = [os.path.join(test_data_path, '{{package_name}}-data', 'test_data_2', f) for f in sorted_file_list]
        self.assertEqual(sorted(path_list), wanted_file_list)
        self.assertEqual([cl for p, cl in sorted(zip(path_list, classes_list), key=lambda pair: pair[0])], sorted_classes_list)
        self.assertEqual(preprocess_str, 'no_preprocess')

        # With preprocessing
        directory_path = os.path.join(utils.get_data_path(), 'test_data_2_with_preprocess')
        path_list, classes_list, preprocess_str = utils.read_folder_classification(directory_path)
        wanted_file_list = [os.path.join(test_data_path, '{{package_name}}-data', 'test_data_2_with_preprocess', f) for f in sorted_file_list]
        self.assertEqual(sorted(path_list), wanted_file_list)
        self.assertEqual([cl for p, cl in sorted(zip(path_list, classes_list), key=lambda pair: pair[0])], sorted_classes_list)
        self.assertEqual(preprocess_str, 'test_preprocess')

        # Without classes
        directory_path = os.path.join(utils.get_data_path(), 'test_data_2_no_class')
        path_list, classes_list, preprocess_str = utils.read_folder_classification(directory_path)
        wanted_file_list = [os.path.join(test_data_path, '{{package_name}}-data', 'test_data_2_no_class', f) for f in sorted_file_list_no_class]
        self.assertEqual(sorted(path_list), wanted_file_list)
        self.assertEqual(classes_list, None)
        self.assertEqual(preprocess_str, 'no_preprocess')

        # Check errors
        directory_path = os.path.join(utils.get_data_path(), 'bad_test_data_2')
        with self.assertRaises(RuntimeError):
            utils.read_folder_classification(directory_path)  # Not only images

        #######################################
        # Format 3 : one sub-folder per class
        #######################################
        sorted_file_list = [os.path.join('birman', 'Birman_1.jpg'), os.path.join('birman', 'Birman_2.jpg'), os.path.join('birman', 'Birman_36.jpg'),
                            os.path.join('birman', 'Birman_4.jpg'), os.path.join('bombay', 'Bombay_1.png'), os.path.join('bombay', 'Bombay_10.jpg'),
                            os.path.join('bombay', 'Bombay_19.jpg'), os.path.join('bombay', 'Bombay_3.jpg'), os.path.join('shiba', 'shiba_inu_15.jpg'),
                            os.path.join('shiba', 'shiba_inu_30.jpg'), os.path.join('shiba', 'shiba_inu_31.jpg'), os.path.join('shiba', 'shiba_inu_34.jpg')]
        sorted_classes_list = ['birman', 'birman', 'birman', 'birman', 'bombay', 'bombay', 'bombay', 'bombay', 'shiba', 'shiba', 'shiba', 'shiba']

        # Nominal case
        directory_path = os.path.join(utils.get_data_path(), 'test_data_3')
        path_list, classes_list, preprocess_str = utils.read_folder_classification(directory_path)
        wanted_file_list = [os.path.join(test_data_path, '{{package_name}}-data', 'test_data_3', f) for f in sorted_file_list]
        self.assertEqual(sorted(path_list), wanted_file_list)
        self.assertEqual([cl for p, cl in sorted(zip(path_list, classes_list), key=lambda pair: pair[0])], sorted_classes_list)
        self.assertEqual(preprocess_str, 'no_preprocess')

        # With preprocessing
        directory_path = os.path.join(utils.get_data_path(), 'test_data_3_with_preprocess')
        path_list, classes_list, preprocess_str = utils.read_folder_classification(directory_path)
        wanted_file_list = [os.path.join(test_data_path, '{{package_name}}-data', 'test_data_3_with_preprocess', f) for f in sorted_file_list]
        self.assertEqual(sorted(path_list), wanted_file_list)
        self.assertEqual([cl for p, cl in sorted(zip(path_list, classes_list), key=lambda pair: pair[0])], sorted_classes_list)
        self.assertEqual(preprocess_str, 'test_preprocess')

        # Check errors
        directory_path = os.path.join(utils.get_data_path(), 'bad_test_data_3')
        with self.assertRaises(RuntimeError):
            utils.read_folder_classification(directory_path)  # Not only sub-folders
        directory_path = os.path.join(utils.get_data_path(), 'bad_test_data_3_v2')
        with self.assertRaises(RuntimeError):
            utils.read_folder_classification(directory_path)  # Not only images in sub-folders

        # Clean
        utils.DIR_PATH = None

    @patch('logging.Logger._log')
    def test02_display_shape(self, PrintMockLog):
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

    def test03_get_chunk_limits(self):
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

    def test04_get_data_path(self):
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

    def test05_get_models_path(self):
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

    def test06_get_ressources_path(self):
        '''Test of the function utils.get_ressources_path'''
        # Nominal case
        path = utils.get_ressources_path()
        self.assertEqual(os.path.isdir(path), True)
        self.assertEqual(path.endswith('{{package_name}}-ressources'), True)

    def test07_get_package_version(self):
        '''Test of the function utils.get_package_version'''
        # Nominal case
        version = utils.get_package_version()
        self.assertEqual(type(version), str)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
