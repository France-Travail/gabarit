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

from {{package_name}}.monitoring.mlflow_logger import MLflowLogger

# Disable logging
import logging
logging.disable(logging.CRITICAL)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

class MLflowLoggerTests(unittest.TestCase):
    '''Main class to test mlflow_logger'''
    
    @classmethod
    def setUpClass(cls):
        mlruns = os.path.join(TEST_DIR, "mlruns")

        if os.path.exists(mlruns):
            shutil.rmtree(mlruns)

        mlflow.set_tracking_uri(f"file://{mlruns}")
        

    def test_mlflow_logger_init(self):
        '''Test of the initialization of {{package_name}}.monitoring.mlflow_logger.MLflowLogger'''
        experiment_name = 'test_mlflow_logger_init'

        model = MLflowLogger(experiment_name=experiment_name)
        self.assertEqual(model.experiment_name, experiment_name)

    def test_mlflow_logger_end_run(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.end_run'''
        experiment_name = 'test_mlflow_logger_end_run'
        model = MLflowLogger(experiment_name=experiment_name)

        # We activate a run via a log
        model.log_param('stop', 'toto')

        # Use of end_run
        model.end_run()

        # Check
        self.assertEqual(mlflow.active_run(), None)

    def test_mlflow_logger_log_metric(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_metric'''
        experiment_name = 'test_mlflow_logger_log_metric'
        model = MLflowLogger(experiment_name=experiment_name)

        # Nominal case
        model.log_metric('test', 5)
        model.log_metric('test', 5, step=2)

        # Clear
        model.end_run()

    def test_mlflow_logger_log_metrics(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_metrics'''
        experiment_name = 'test_mlflow_logger_log_metrics'
        model = MLflowLogger(experiment_name=experiment_name)

        # Nominal case
        model.log_metrics({'test': 5})
        model.log_metrics({'test': 5}, step=2)

    def test_mlflow_logger_log_param(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_param'''
        experiment_name = 'test_mlflow_logger_log_param'
        model = MLflowLogger(experiment_name=experiment_name)

        # Nominal case
        model.log_param('test', 5)

    def test_mlflow_logger_log_params(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_params'''    
        experiment_name = 'test_mlflow_logger_log_params'
        model = MLflowLogger(experiment_name=experiment_name)

        # Nominal case
        model.log_params({'test': 5})

    def test_mlflow_logger_set_tag(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.set_tag'''
        experiment_name = 'test_mlflow_logger_set_tag'
        model = MLflowLogger(experiment_name=experiment_name)

        # Nominal case
        model.set_tag('test', 5)

    def test_mlflow_logger_set_tags(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.set_tags'''
        experiment_name = 'test_mlflow_logger_set_tags'
        model = MLflowLogger(experiment_name=experiment_name)

        # Nominal case
        model.set_tags({'test': 5})


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()