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
from {{package_name}}.preprocessing import preprocess

# Disable logging
import logging
logging.disable(logging.CRITICAL)

# We won't test the different pipeline function of preprocess but rather check them
# quickly through the tests of get_preprocessors_dict

class PreprocessTests(unittest.TestCase):
    '''Main class to test all functions in {{package_name}}.preprocessing.preprocess'''

    # We avoid tqdm prints
    pd.Series.progress_apply = pd.Series.apply

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_get_preprocessors_dict(self):
        '''Test of the method preprocess.get_preprocessors_dict'''
        # Valids to test
        content = 'CEci est une phrase à tester. cool :)'

        # Nominal case
        preprocessors_dict = preprocess.get_preprocessors_dict()
        self.assertEqual(type(preprocessors_dict), dict)
        self.assertTrue('no_preprocess' in preprocessors_dict.keys())

        # We test each returned function
        for f in preprocessors_dict.values():
            self.assertEqual(type(f(content)), str)
            self.assertEqual(type(f([content])), list)
            self.assertEqual(type(f(pd.Series(content))), pd.Series)
            self.assertEqual(type(f(pd.DataFrame([content]))), pd.DataFrame)

    def test02_get_preprocessor(self):
        '''Test of the method preprocess.get_preprocessor'''
        # Valids to test
        # We take a preprocessing 'at random'
        preprocessor_str = list(preprocess.get_preprocessors_dict().keys())[0]
        preprocessor_val = list(preprocess.get_preprocessors_dict().values())[0]

        # Nominal case
        # To check if 2 functions are "equal" : https://stackoverflow.com/questions/20059011/check-if-two-python-functions-are-equal
        preprocessor_res = preprocess.get_preprocessor(preprocessor_str)
        self.assertEqual(preprocessor_res.__code__.co_code, preprocessor_val.__code__.co_code)

        # Check the input(s) type(s)
        with self.assertRaises(ValueError):
            preprocess.get_preprocessor('NOT A VALID PREPROCESS')

    # We do not test preprocess_sentence_P1 -> the function is just given as an example here


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
