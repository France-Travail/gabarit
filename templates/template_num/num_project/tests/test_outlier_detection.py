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
import json
import numpy as np
import pandas as pd
from {{package_name}} import utils
from {{package_name}}.preprocessing import outlier_detection

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class OutlierDetectionTests(unittest.TestCase):
    '''Main class to test all functions in {{package_name}}.preprocessing.outlier_detection'''


    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_check_for_outliers(self):
        '''Test of the method outlier_detection.check_for_outliers'''

        # Valids to test
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'y': [1, 2, 3, 4, 5, 6, 7, 100000, 100000000, 100000000],
            'z': [-10000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        })
        df_copy = df.copy(deep=True)
        arr = df.to_numpy()
        arr_copy = arr.copy()
        expected_outliers = np.array([-1,  1,  1,  1,  1,  1,  1, -1, -1, -1])

        # Nominal case (we just set n_neighbors to a value < nb elements)
        outliers = outlier_detection.check_for_outliers(df, n_neighbors=3)
        outliers_bis = outlier_detection.check_for_outliers(arr, n_neighbors=3)
        # We can't test returned values as IsolationForest has a random part

        # Only IsolationForest
        outliers = outlier_detection.check_for_outliers(df, n_neighbors=0)
        outliers_bis = outlier_detection.check_for_outliers(arr, n_neighbors=0)
        # We can't test returned values as IsolationForest has a random part

        # Only LocalOutlierFactor
        outliers = outlier_detection.check_for_outliers(df, n_estimators=0, n_neighbors=3)
        outliers_bis = outlier_detection.check_for_outliers(arr, n_estimators=0, n_neighbors=3)
        # We can test returned values as LocalOutlierFactor does not have a random part
        np.testing.assert_array_equal(outliers, expected_outliers)
        np.testing.assert_array_equal(outliers_bis, expected_outliers)
        pd.testing.assert_frame_equal(df, df_copy)
        np.testing.assert_array_equal(arr, arr_copy)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
