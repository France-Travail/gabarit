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
import json
import shutil
import mlflow
import urllib
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from {{package_name}}.monitoring.mlflow_logger import MLflowLogger

# Disable logging
import logging
logging.disable(logging.CRITICAL)

# TMP directory for mlruns
MLRUNS_TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_experiments', 'mlruns')
LOCAL_TRACKING_URI = pathlib.Path(MLRUNS_TMP_DIR).as_uri()


class MLflowLoggerTests(unittest.TestCase):
    '''Main class to test mlflow_logger'''

    @classmethod
    def setUpClass(cls):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)
        # Clean mlruns directory (if exists)
        if os.path.exists(MLRUNS_TMP_DIR):
            shutil.rmtree(MLRUNS_TMP_DIR)

    @classmethod
    def tearDownClass(cls):
        # Remove mlruns directory
        if os.path.exists(MLRUNS_TMP_DIR):
            shutil.rmtree(MLRUNS_TMP_DIR)

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
        artifact_location = urllib.request.url2pathname(urllib.parse.urlparse(mlflow.get_experiment_by_name(experiment_name).artifact_location).path)
        run_id = mlflow.last_active_run().info.run_id
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'metrics', 'test')))
        # Clear
        mlflow_logger.end_run()

    def test04_mlflow_logger_log_metrics(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_metrics'''
        experiment_name = 'test_mlflow_logger_log_metrics'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        mlflow_logger.log_metrics({'test': 5, 'test2': 24})
        mlflow_logger.log_metrics({'test': 5}, step=2)
        artifact_location = urllib.request.url2pathname(urllib.parse.urlparse(mlflow.get_experiment_by_name(experiment_name).artifact_location).path)
        run_id = mlflow.last_active_run().info.run_id
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'metrics', 'test')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'metrics', 'test2')))
        # Clear
        mlflow_logger.end_run()

    def test05_mlflow_logger_log_param(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_param'''
        experiment_name = 'test_mlflow_logger_log_param'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        mlflow_logger.log_param('test', 5)
        artifact_location = urllib.request.url2pathname(urllib.parse.urlparse(mlflow.get_experiment_by_name(experiment_name).artifact_location).path)
        run_id = mlflow.last_active_run().info.run_id
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'params', 'test')))
        # Clear
        mlflow_logger.end_run()

    def test06_mlflow_logger_log_params(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_params'''
        experiment_name = 'test_mlflow_logger_log_params'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        mlflow_logger.log_params({'test': 5, 'test2': 24})
        artifact_location = urllib.request.url2pathname(urllib.parse.urlparse(mlflow.get_experiment_by_name(experiment_name).artifact_location).path)
        run_id = mlflow.last_active_run().info.run_id
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'params', 'test')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'params', 'test2')))
        # Clear
        mlflow_logger.end_run()

    def test07_mlflow_logger_set_tag(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.set_tag'''
        experiment_name = 'test_mlflow_logger_set_tag'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        mlflow_logger.set_tag('test', 5)
        artifact_location = urllib.request.url2pathname(urllib.parse.urlparse(mlflow.get_experiment_by_name(experiment_name).artifact_location).path)
        run_id = mlflow.last_active_run().info.run_id
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'tags', 'test')))
        # Clear
        mlflow_logger.end_run()

    def test08_mlflow_logger_set_tags(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.set_tags'''
        experiment_name = 'test_mlflow_logger_set_tags'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        mlflow_logger.set_tags({'test': 5, 'test2': 24})
        artifact_location = urllib.request.url2pathname(urllib.parse.urlparse(mlflow.get_experiment_by_name(experiment_name).artifact_location).path)
        run_id = mlflow.last_active_run().info.run_id
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'tags', 'test')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'tags', 'test2')))
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
            'Label': ['label1', 'label2', 'label3!?', 'label4'],
            'metric1': [0.5, 0.2, 1.0, -1.5],
            'metric2!!!': [None, 0.2, -15, np.NaN]
        })
        mlflow_logger.log_df_stats(df_stats)
        artifact_location = urllib.request.url2pathname(urllib.parse.urlparse(mlflow.get_experiment_by_name(experiment_name).artifact_location).path)
        run_id = mlflow.last_active_run().info.run_id
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'params', 'Label 0')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'params', 'Label 1')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'params', 'Label 2')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'params', 'Label 3')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'metrics', 'label1 --- metric1')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'metrics', 'label1 --- Col 1')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'metrics', 'label2 --- metric1')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'metrics', 'label2 --- Col 1')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'metrics', 'Label 2 --- metric1')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'metrics', 'Label 2 --- Col 1')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'metrics', 'label4 --- metric1')))
        self.assertTrue(os.path.exists(os.path.join(artifact_location, run_id, 'metrics', 'label4 --- Col 1')))

        # Bad label_col
        with self.assertRaises(ValueError):
            mlflow_logger.log_df_stats(df_stats, label_col='NotALabelCol')

        # Clear
        mlflow_logger.end_run()

    def test11_mlflow_logger_log_dict(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_dict'''
        experiment_name = 'test_mlflow_logger_log_dict'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        dict = {'toto': 'titi', 'tata': 5}
        artifact_file = 'test.json'
        mlflow_logger.log_dict(dict, artifact_file)
        artifact_location = urllib.request.url2pathname(urllib.parse.urlparse(mlflow.get_experiment_by_name(experiment_name).artifact_location).path)
        run_id = mlflow.last_active_run().info.run_id
        saved_json_path = os.path.join(artifact_location, run_id, 'artifacts', 'test.json')
        self.assertTrue(os.path.exists(saved_json_path))
        with open(saved_json_path, 'r') as f:
            saved_jason = json.load(f)
        self.assertEqual(saved_jason, dict)
        # Clear
        mlflow_logger.end_run()

    def test12_mlflow_logger_log_text(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_text'''
        experiment_name = 'test_mlflow_logger_log_text'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        text = 'This is a test !!!'
        artifact_file = 'test.txt'
        mlflow_logger.log_text(text, artifact_file)
        artifact_location = urllib.request.url2pathname(urllib.parse.urlparse(mlflow.get_experiment_by_name(experiment_name).artifact_location).path)
        run_id = mlflow.last_active_run().info.run_id
        saved_txt_path = os.path.join(artifact_location, run_id, 'artifacts', 'test.txt')
        self.assertTrue(os.path.exists(saved_txt_path))
        with open(saved_txt_path, 'r') as f:
            saved_txt = f.read()
        self.assertEqual(saved_txt, text)
        # Clear
        mlflow_logger.end_run()

    def test13_mlflow_logger_log_figure(self):
        '''Test of {{package_name}}.monitoring.mlflow_logger.MLflowLogger.log_figure'''
        experiment_name = 'test_mlflow_logger_log_figure'
        mlflow_logger = MLflowLogger(experiment_name=experiment_name, tracking_uri=LOCAL_TRACKING_URI)
        # Nominal case
        plt.pie([0.4, 0.3, 0.3])
        figure = plt.gcf()
        figure.canvas.draw()
        image_from_figure = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        artifact_file = 'figure.png'
        mlflow_logger.log_figure(figure, artifact_file)
        artifact_location = urllib.request.url2pathname(urllib.parse.urlparse(mlflow.get_experiment_by_name(experiment_name).artifact_location).path)
        run_id = mlflow.last_active_run().info.run_id
        saved_png_path = os.path.join(artifact_location, run_id, 'artifacts', 'figure.png')
        self.assertTrue(os.path.exists(saved_png_path))
        saved_png = (plt.imread(saved_png_path) * 255).astype('uint8')
        saved_png = saved_png[:, :, :3]
        self.assertAlmostEqual(np.mean(image_from_figure), np.mean(saved_png))
        # # Clear
        mlflow_logger.end_run()


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
