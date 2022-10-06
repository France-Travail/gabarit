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
import mlflow
import numpy as np
import pandas as pd

from {{package_name}}.monitoring.mlflow_logger import MLflowLogger

# Disable logging
import logging
logging.disable(logging.CRITICAL)

# TMP directory for mlruns
MLRUNS_TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_experiments', 'mlruns')
LOCAL_TRACKING_URI = f"file://{MLRUNS_TMP_DIR}"


class MLflowLoggerTests(unittest.TestCase):
    '''Main class to test mlflow_logger'''

    @classmethod
    def setUpClass(cls):
        # Set mlruns directory
        if os.path.exists(MLRUNS_TMP_DIR):
            shutil.rmtree(MLRUNS_TMP_DIR)

    @classmethod
    def tearDownClass(cls):
        # Remove mlruns directory
        if os.path.exists(MLRUNS_TMP_DIR):
            shutil.rmtree(MLRUNS_TMP_DIR)

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_mlflow_logger_init(self):
        '''Test of the initialization of {{package_name}}.monitoring.mlflow_logger.MLflowLogger'''
        experiment_name = 'test_mlflow_logger_init'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        self.assertEqual(mlflow_logger.tracking_uri, LOCAL_TRACKING_URI)
        self.assertEqual(mlflow.get_tracking_uri(), LOCAL_TRACKING_URI)
        self.assertEqual(mlflow_logger.experiment_name, experiment_name)
        # No need to call end_run

    def test02_mlflow_logger_end_run(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.end_run'''
        experiment_name = 'test_mlflow_logger_end_run'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # We activate a run via a log
        mlflow_logger.log_param('stop', 'toto')
        # Use of end_run & check
        mlflow_logger.end_run()
        self.assertEqual(mlflow.active_run(), None)

    def test03_mlflow_logger_log_metric(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_metric'''
        experiment_name = 'test_mlflow_logger_log_metric'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        mlflow_logger.log_metric('test', 5)
        mlflow_logger.log_metric('test', 5, step=2)
        # Clear
        mlflow_logger.end_run()

    def test04_mlflow_logger_log_metrics(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_metrics'''
        experiment_name = 'test_mlflow_logger_log_metrics'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        mlflow_logger.log_metrics({'test': 5})
        mlflow_logger.log_metrics({'test': 5}, step=2)
        # Clear
        mlflow_logger.end_run()

    def test05_mlflow_logger_log_param(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_param'''
        experiment_name = 'test_mlflow_logger_log_param'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        mlflow_logger.log_param('test', 5)
        # Clear
        mlflow_logger.end_run()

    def test06_mlflow_logger_log_params(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_params'''
        experiment_name = 'test_mlflow_logger_log_params'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        mlflow_logger.log_params({'test': 5})
        # Clear
        mlflow_logger.end_run()

    def test07_mlflow_logger_set_tag(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.set_tag'''
        experiment_name = 'test_mlflow_logger_set_tag'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        mlflow_logger.set_tag('test', 5)
        # Clear
        mlflow_logger.end_run()

    def test08_mlflow_logger_set_tags(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.set_tags'''
        experiment_name = 'test_mlflow_logger_set_tags'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        mlflow_logger.set_tags({'test': 5})
        # Clear
        mlflow_logger.end_run()

    def test09_mlflow_logger_valid_name(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.valid_name'''
        experiment_name = 'test_mlflow_logger_valid_name'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Valid case
        self.assertTrue(mlflow_logger.valid_name('valid_name'))
        self.assertFalse(mlflow_logger.valid_name('not a valid_name!'))
        # No need to call end_run

    def test10_mlflow_logger_log_df_stats(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_df_stats'''
        experiment_name = 'test_mlflow_logger_log_df_stats'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)

        # Nominal case
        df_stats = pd.DataFrame({
            'Label': ['label1', 'label2', 'label3', 'label4'],
            'metric1': [0.5, 0.2, 1.0, -1.5],
            'metric2!!!': [None, 0.2, 'test', np.NaN]
        })
        mlflow.log_df_stats(df_stats)

        # Clear
        mlflow_logger.end_run()

# TODO: log_dict & log_text
# TODO: check local writing ?

# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
