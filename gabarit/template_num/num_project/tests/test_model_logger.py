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
import mlflow
import numpy as np
import pandas as pd

from {{package_name}} import utils
from {{package_name}}.monitoring.model_logger import ModelLogger, is_running, is_local, is_mlflow_up

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class ModelLoggerTests(unittest.TestCase):
    '''Main class to test model_logger'''


    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_is_running(self):
        '''Test of the function {{package_name}}.monitoring.model_logger.is_running'''
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir)) # We just want the logger
        bad_host = 'http://toto.titi.tata.test'
        bad_port = 80
        self.assertFalse(is_running(bad_host, bad_port, model.logger))
        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test02_is_local(self):
        '''Test of the method {{package_name}}.monitoring.model_logger.is_local'''
        local = 'ceci/est/un/test'
        distant = 'http://ceci.est.un.faux.site.com'
        self.assertTrue(is_local(local))
        self.assertFalse(is_local(distant))

    def test03_model_logger_init(self):
        '''Test of the initialization of {{package_name}}.monitoring.model_logger.ModelLogger'''
        # We test with a fake host
        host = 'http://toto.titi.tata.test'
        name = 'test'
        model = ModelLogger(tracking_uri=host, experiment_name=name)
        self.assertEqual(model.tracking_uri, host)
        self.assertEqual(model.experiment_name, name)
        self.assertFalse(model.running)
        # Clear
        model.stop_run()

        # We test with nothing (by default, mlflow saves in a folder 'ml_runs')
        save_dir = os.path.join(os.getcwd(), 'mlruns')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        self.assertFalse(os.path.exists(save_dir))
        model = ModelLogger()
        self.assertEqual(model.tracking_uri, '')
        self.assertTrue(model.running)
        self.assertTrue(os.path.exists(save_dir))
        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test04_model_logger_stop_run(self):
        '''Test of {{package_name}}.monitoring.model_logger.ModelLogger.stop_run'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))
        # We activate a run via a log
        model.log_param('test', 'toto')
        # Use of stop_run
        model.stop_run()
        # Check
        self.assertEqual(mlflow.active_run(), None)
        # Clear
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test05_model_logger_log_metric(self):
        '''Test of {{package_name}}.monitoring.model_logger.ModelLogger.log_metric'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))

        # Nominal case
        model.log_metric('test', 5)
        model.log_metric('test', 5, step=2)

        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test06_model_logger_log_metrics(self):
        '''Test of {{package_name}}.monitoring.model_logger.ModelLogger.log_metrics'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))

        # Nominal case
        model.log_metrics({'test': 5})
        model.log_metrics({'test': 5}, step=2)

        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test07_model_logger_log_param(self):
        '''Test of {{package_name}}.monitoring.model_logger.ModelLogger.log_param'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))

        # Nominal case
        model.log_param('test', 5)

        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test08_model_logger_log_params(self):
        '''Test of {{package_name}}.monitoring.model_logger.ModelLogger.log_params'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))

        # Nominal case
        model.log_params({'test': 5})

        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test09_model_logger_set_tag(self):
        '''Test of {{package_name}}.monitoring.model_logger.ModelLogger.set_tag'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))

        # Nominal case
        model.set_tag('test', 5)

        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    def test10_model_logger_set_tags(self):
        '''Test of {{package_name}}.monitoring.model_logger.ModelLogger.set_tags'''
        # Init. logger
        save_dir = os.path.join(os.getcwd(), 'ml_flow_test')
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        model = ModelLogger(os.path.relpath(save_dir))

        # Nominal case
        model.set_tags({'test': 5})

        # Clear
        model.stop_run()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
