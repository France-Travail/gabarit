#!/usr/bin/env python3
# Starts all functional tests
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

# utils libs
import os
import sys
import shutil
import subprocess
import pandas as pd
import importlib.util
from pathlib import Path
from datetime import datetime

from test_template_num import utils
from test_template_num.models_training import utils_models
from test_template_num.models_training.classifiers import (model_rf_classifier, model_dense_classifier,
                                                           model_ridge_classifier, model_logistic_regression_classifier,
                                                           model_sgd_classifier, model_svm_classifier, model_knn_classifier,
                                                           model_gbt_classifier, model_lgbm_classifier, model_xgboost_classifier)
from test_template_num.models_training.regressors import (model_rf_regressor, model_dense_regressor,
                                                          model_elasticnet_regressor, model_bayesian_ridge_regressor,
                                                          model_kernel_ridge_regressor, model_svr_regressor,
                                                          model_sgd_regressor, model_knn_regressor, model_pls_regressor,
                                                          model_gbt_regressor, model_xgboost_regressor, model_lgbm_regressor)

class Case1_e2e_pipeline(unittest.TestCase):
    '''Class to test the project end to end'''

    def test01_CreateSamples(self):
        '''Test of the file 0_create_samples.py'''
        print("Test of the file 0_create_samples.py")

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_create_samples.py -f mono_class_mono_label.csv -n 15"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'mono_class_mono_label_15_samples.csv')))
        df = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_class_mono_label_15_samples.csv", sep=';', encoding='utf-8')
        self.assertEqual(df.shape[0], 15)

        # Double files
        fonctionnement_double = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_create_samples.py -f mono_class_mono_label.csv multi_class_mono_label.csv -n 2000"
        self.assertEqual(subprocess.run(fonctionnement_double, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'mono_class_mono_label_2000_samples.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'multi_class_mono_label_2000_samples.csv')))
        df1 = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_class_mono_label_2000_samples.csv", sep=';', encoding='utf-8')
        df2 = pd.read_csv(f"{full_path_lib}/test_template_num-data/multi_class_mono_label_2000_samples.csv", sep=';', encoding='utf-8')
        self.assertEqual(df1.shape[0], 210)
        self.assertEqual(df2.shape[0], 210) # 210 row max

    def test02_MergeFiles(self):
        '''Test of the file 0_merge_files.py'''
        print("Test of the file 0_merge_files.py")

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_merge_files.py -f mono_class_mono_label.csv multi_class_mono_label.csv -c col_1 col_2 y_col -o merged_file.csv"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'merged_file.csv')))
        df = pd.read_csv(f"{full_path_lib}/test_template_num-data/merged_file.csv", sep=';', encoding='utf-8')
        self.assertGreater(df.shape[0], 210) # We check that there are more than 210 elements (ie. the size of one of the two files)

    def test03_SplitTrainValidTest(self):
        '''Test of the file 0_split_train_valid_test.py'''
        print("Test of the file 0_split_train_valid_test.py")

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'mono_class_mono_label_train.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'mono_class_mono_label_valid.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'mono_class_mono_label_test.csv')))
        df_train = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_class_mono_label_train.csv", sep=';', encoding='utf-8')
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_class_mono_label_valid.csv", sep=';', encoding='utf-8')
        df_test = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_class_mono_label_test.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_train.shape[0], 126)
        self.assertEqual(df_valid.shape[0], 42)
        self.assertEqual(df_test.shape[0], 42)

        # Test of perc_x arguments
        test_perc = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type random --perc_train 0.3 --perc_valid 0.6 --perc_test 0.1 --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(test_perc, shell=True).returncode, 0)
        df_train = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_class_mono_label_train.csv", sep=';', encoding='utf-8')
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_class_mono_label_valid.csv", sep=';', encoding='utf-8')
        df_test = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_class_mono_label_test.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_train.shape[0], 63)
        self.assertEqual(df_valid.shape[0], 126)
        self.assertEqual(df_test.shape[0], 21)

        # Test split_type stratified
        test_stratified = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type stratified --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(test_stratified, shell=True).returncode, 0)
        df_train = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_class_mono_label_train.csv", sep=';', encoding='utf-8')
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_class_mono_label_valid.csv", sep=';', encoding='utf-8')
        df_test = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_class_mono_label_test.csv", sep=';', encoding='utf-8')
        # Check number of elements
        self.assertEqual(df_train.shape[0], 126)
        self.assertEqual(df_valid.shape[0], 42)
        self.assertEqual(df_test.shape[0], 42)
        # Check stratified (we have 3/7 "1"s -> 0.428...)
        perc_oui_train = sum(df_train['y_col'] == "oui")/df_train.shape[0]
        self.assertGreater(perc_oui_train, 0.40)
        self.assertLess(perc_oui_train, 0.45)
        perc_oui_valid = sum(df_valid['y_col'] == "oui")/df_valid.shape[0]
        self.assertGreater(perc_oui_valid, 0.40)
        self.assertLess(perc_oui_valid, 0.45)
        perc_oui_test = sum(df_test['y_col'] == "oui")/df_test.shape[0]
        self.assertGreater(perc_oui_test, 0.40)
        self.assertLess(perc_oui_test, 0.45)

        # "Basic" case - regression (for compatibility)
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_output_regression.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'mono_output_regression_train.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'mono_output_regression_valid.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'mono_output_regression_test.csv')))
        df_train = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_output_regression_train.csv", sep=';', encoding='utf-8')
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_output_regression_valid.csv", sep=';', encoding='utf-8')
        df_test = pd.read_csv(f"{full_path_lib}/test_template_num-data/mono_output_regression_test.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_train.shape[0], 126)
        self.assertEqual(df_valid.shape[0], 42)
        self.assertEqual(df_test.shape[0], 42)

    def test04_PreProcessData(self):
        '''Test of the file 1_preprocess_data.py'''
        print("Test of the file 1_preprocess_data.py")

        # "Basic" case - classification
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/1_preprocess_data.py -f mono_class_mono_label_train.csv -p preprocess_P1 --target_cols y_col"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check if exists
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'mono_class_mono_label_train_preprocess_P1.csv')))
        df_train = pd.read_csv(os.path.join(full_path_lib, 'test_template_num-data', 'mono_class_mono_label_train_preprocess_P1.csv'), sep=';', encoding='utf-8', skiprows=1)
        # Check col col_1, col_2 & y_col exists
        self.assertTrue('col_1' in df_train.columns)
        self.assertTrue('col_2' in df_train.columns)
        self.assertTrue('y_col' in df_train.columns)
        # Check col x_col_1 value
        self.assertTrue(df_train['col_1'].values[2], 0.4142585780542456)
        # Check col y_col values
        self.assertEqual(sorted(df_train.y_col.unique()), ["non", "oui"])
        # Check pipeline has been saved
        pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
        self.assertTrue(os.path.exists(pipelines_dirpath))
        self.assertTrue(len(os.listdir(pipelines_dirpath)) >= 1)
        pipeline_path = os.path.join(pipelines_dirpath, os.listdir(pipelines_dirpath)[0])
        self.assertTrue('pipeline.info' in os.listdir(pipeline_path))
        self.assertTrue('pipeline.pkl' in os.listdir(pipeline_path))

        # "Basic" case - regression
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/1_preprocess_data.py -f mono_output_regression_train.csv -p preprocess_P1 --target_cols y_col"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check if exists
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'mono_output_regression_train_preprocess_P1.csv')))
        df_train = pd.read_csv(os.path.join(full_path_lib, 'test_template_num-data', 'mono_output_regression_train_preprocess_P1.csv'), sep=';', encoding='utf-8', skiprows=1)
        # Check col col_1, col_2 & y_col exists
        self.assertTrue('col_1' in df_train.columns)
        self.assertTrue('col_2' in df_train.columns)
        self.assertTrue('y_col' in df_train.columns)
        # Check col x_col_1 value
        self.assertTrue(df_train['col_1'].values[2], 0.4142585780542456)
        # Check pipeline has been saved
        pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
        self.assertTrue(os.path.exists(pipelines_dirpath))
        self.assertTrue(len(os.listdir(pipelines_dirpath)) >= 1)
        pipeline_path = os.path.join(pipelines_dirpath, os.listdir(pipelines_dirpath)[-1])
        self.assertTrue('pipeline.info' in os.listdir(pipeline_path))
        self.assertTrue('pipeline.pkl' in os.listdir(pipeline_path))

    def test05_ApplyPipeline(self):
        '''Test of the file 2_apply_existing_pipeline.py'''
        print("Test of the file 2_apply_existing_pipeline.py")

        pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
        pipeline_name_classification = os.listdir(pipelines_dirpath)[0]
        pipeline_name_regression = os.listdir(pipelines_dirpath)[-1]

        # "Basic" case - classification
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/2_apply_existing_pipeline.py -f mono_class_mono_label_valid.csv -p {pipeline_name_classification} --target_cols y_col"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check if exists
        train_path = os.path.join(full_path_lib, 'test_template_num-data', 'mono_class_mono_label_train_preprocess_P1.csv')
        valid_path = os.path.join(full_path_lib, 'test_template_num-data', 'mono_class_mono_label_train_preprocess_P1.csv')
        self.assertTrue(os.path.exists(valid_path))
        # Check first line equals
        with open(train_path, 'r', encoding='utf-8') as f:
            first_line_train = f.readline()
        with open(valid_path, 'r', encoding='utf-8') as f:
            first_line_valid = f.readline()
        self.assertEqual(first_line_train, first_line_valid)
        # Check same preprocess (we test unique values)
        df_train = pd.read_csv(train_path, sep=';', encoding='utf-8', skiprows=1)
        df_valid = pd.read_csv(valid_path, sep=';', encoding='utf-8', skiprows=1)
        self.assertEqual(sorted(df_train.col_1.unique()), sorted(df_valid.col_1.unique()))
        self.assertEqual(sorted(df_train.col_2.unique()), sorted(df_valid.col_2.unique()))
        self.assertEqual(sorted(df_train.y_col.unique()), sorted(df_valid.y_col.unique()))

        # "Basic" case - regression
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/2_apply_existing_pipeline.py -f mono_output_regression_valid.csv -p {pipeline_name_regression} --target_cols y_col"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check if exists
        train_path = os.path.join(full_path_lib, 'test_template_num-data', 'mono_output_regression_train_preprocess_P1.csv')
        valid_path = os.path.join(full_path_lib, 'test_template_num-data', 'mono_output_regression_train_preprocess_P1.csv')
        self.assertTrue(os.path.exists(valid_path))
        # Check first line equals
        with open(train_path, 'r', encoding='utf-8') as f:
            first_line_train = f.readline()
        with open(valid_path, 'r', encoding='utf-8') as f:
            first_line_valid = f.readline()
        self.assertEqual(first_line_train, first_line_valid)
        # Check same preprocess (we test unique values)
        df_train = pd.read_csv(train_path, sep=';', encoding='utf-8', skiprows=1)
        df_valid = pd.read_csv(valid_path, sep=';', encoding='utf-8', skiprows=1)
        self.assertEqual(sorted(df_train.col_1.unique()), sorted(df_valid.col_1.unique()))
        self.assertEqual(sorted(df_train.col_2.unique()), sorted(df_valid.col_2.unique()))
        self.assertEqual(sorted(df_train.y_col.unique()), sorted(df_valid.y_col.unique()))

    def test06_TrainingE2E(self):
        '''Test of files 3_training_classification.py & 3_training_regression.py'''
        print("Test of files 3_training_classification.py & 3_training_regression.py")

        ################
        # Classification
        ################

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/3_training_classification.py -f mono_class_mono_label_train_preprocess_P1.csv -y y_col --filename_valid mono_class_mono_label_valid_preprocess_P1.csv"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_num-models', 'model_ridge_classifier') # Ridge Classifier by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertGreater(len(listdir), 0)
        # Clean
        shutil.rmtree(save_model_dir)

        # With excluded_cols
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/3_training_classification.py -f mono_class_mono_label_train_preprocess_P1.csv -y y_col --filename_valid mono_class_mono_label_valid_preprocess_P1.csv --excluded_cols col_2"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_num-models', 'model_ridge_classifier') # Ridge Classifier by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertGreater(len(listdir), 0)

        ############
        # Regression
        ############
        # We didn't work on the file

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/3_training_regression.py -f mono_output_regression_train.csv -y y_col --filename_valid mono_output_regression_valid.csv"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_num-models', 'model_elasticnet_regressor') # ElasticNet Regressor by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertGreater(len(listdir), 0)
        # Clean
        shutil.rmtree(save_model_dir)

        # With excluded_cols
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/3_training_regression.py -f mono_output_regression_train.csv -y y_col --filename_valid mono_output_regression_valid.csv --excluded_cols col_2"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_num-models', 'model_elasticnet_regressor') # ElasticNet Regressor by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertGreater(len(listdir), 0)

    def test07_PredictE2E(self):
        '''Test of the file 4_predict.py'''
        print("Test of the file 4_predict.py")

        ################
        # Classification
        ################

        # "Basic" case
        save_model_dir = os.path.join(full_path_lib, 'test_template_num-models', 'model_ridge_classifier') # tfidf svm by default
        listdir = os.listdir(os.path.join(save_model_dir))
        model_name = listdir[0]
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/4_predict.py -f mono_class_mono_label_test.csv -m {model_name}"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check predictions
        save_predictions_dir = os.path.join(full_path_lib, 'test_template_num-data', 'predictions', 'mono_class_mono_label_test')
        self.assertTrue(os.path.exists(save_predictions_dir))
        listdir = os.listdir(os.path.join(save_predictions_dir))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[0], 'predictions.csv')))

        # Run with "y_col"
        run_with_y_col = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/4_predict.py -f mono_class_mono_label_test.csv -y y_col -m {model_name}"
        self.assertEqual(subprocess.run(run_with_y_col, shell=True).returncode, 0)
        # Check predictions
        listdir = sorted(os.listdir(os.path.join(save_predictions_dir)))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[-1], 'predictions_with_y_true.csv'))) # last folder

        ################
        # Regression
        ################

        # "Basic" case
        save_model_dir = os.path.join(full_path_lib, 'test_template_num-models', 'model_elasticnet_regressor') # tfidf svm by default
        listdir = os.listdir(os.path.join(save_model_dir))
        model_name = listdir[0]
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/4_predict.py -f mono_output_regression_test.csv -m {model_name}"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check predictions
        save_predictions_dir = os.path.join(full_path_lib, 'test_template_num-data', 'predictions', 'mono_class_mono_label_test')
        self.assertTrue(os.path.exists(save_predictions_dir))
        listdir = os.listdir(os.path.join(save_predictions_dir))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[0], 'predictions.csv')))

        # Run with "y_col"
        run_with_y_col = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/4_predict.py -f mono_output_regression_test.csv -y y_col -m {model_name}"
        self.assertEqual(subprocess.run(run_with_y_col, shell=True).returncode, 0)
        # Check predictions
        listdir = sorted(os.listdir(os.path.join(save_predictions_dir)))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[-1], 'predictions_with_y_true.csv'))) # last folder


def test_model_mono_class_mono_label(test_class, test_model):
    '''Generic function to test a given model for mono-class/mono-label'''

    # Check if files exist
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    # Try some functions
    df_input_preds = pd.DataFrame({
        'col_1': [-5, 3],
        'col_2': [2, 2],
    })
    df_input_preds_prep = utils_models.apply_pipeline(df_input_preds, test_model.preprocess_pipeline)
    # predict
    preds = test_model.predict(df_input_preds_prep)
    test_class.assertEqual(list(preds), ['non', 'oui'])
    # predict_proba
    index_non = test_model.list_classes.index('non')
    index_oui = test_model.list_classes.index('oui')
    probas = test_model.predict_proba(df_input_preds_prep)
    test_class.assertGreater(probas[0][index_non], 0.5)
    test_class.assertLess(probas[0][index_oui], 0.5)
    test_class.assertGreater(probas[1][index_oui], 0.5)
    test_class.assertLess(probas[1][index_non], 0.5)
    # predict w/ return_proba=True
    probas2 = test_model.predict(df_input_preds_prep, return_proba=True)
    test_class.assertGreater(probas2[0][index_non], 0.5)
    test_class.assertLess(probas2[0][index_oui], 0.5)
    test_class.assertGreater(probas2[1][index_oui], 0.5)
    test_class.assertLess(probas2[1][index_non], 0.5)
    # predict_with_proba
    pred_proba = test_model.predict_with_proba(df_input_preds_prep)
    test_class.assertEqual(list(pred_proba[0]), ['non', 'oui'])
    test_class.assertGreater(pred_proba[1][0][index_non], 0.5)
    test_class.assertLess(pred_proba[1][0][index_oui], 0.5)
    test_class.assertGreater(pred_proba[1][1][index_oui], 0.5)
    test_class.assertLess(pred_proba[1][1][index_non], 0.5)
    # get_predict_position
    df_input_get_predict_position = pd.DataFrame({
        'col_1': [-5, 3, 5],
        'col_2': [2, 2, 5],
    })
    df_input_get_predict_position_prep = utils_models.apply_pipeline(df_input_get_predict_position, test_model.preprocess_pipeline)
    # position start at 1
    test_class.assertEqual(list(test_model.get_predict_position(df_input_get_predict_position_prep, ['oui', 'oui', 'toto'])), [2, 1, -1])
    # get_classes_from_proba
    test_class.assertEqual(list(test_model.get_classes_from_proba(probas)), ['non', 'oui'])
    # get_top_n_from_proba
    with test_class.assertRaises(ValueError):
        test_model.get_top_n_from_proba(probas, n=5) # Only 2 classes in our model
    top_n, top_n_proba = test_model.get_top_n_from_proba(probas, n=2)
    test_class.assertEqual([list(_) for _ in top_n], [['non', 'oui'], ['oui', 'non']])
    test_class.assertEqual([list(_) for _ in top_n_proba], [[probas[0][index_non], probas[0][index_oui]], [probas[1][index_oui], probas[1][index_non]]])
    # inverse_transform
    test_class.assertEqual(list(test_model.inverse_transform(preds)), ['non', 'oui'])


class Case2_MonoClassMonoLabel(unittest.TestCase):
    '''Class to test the mono-class / mono-label case'''

    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for mono-class / mono-label case")

        # Clean repo
        pipelines_dir = os.path.join(full_path_lib, 'test_template_num-pipelines')
        models_dir = os.path.join(full_path_lib, 'test_template_num-models')
        if os.path.isdir(pipelines_dir):
            shutil.rmtree(pipelines_dir)
            os.mkdir(pipelines_dir)
        if os.path.isdir(models_dir):
            shutil.rmtree(models_dir)
            os.mkdir(models_dir)

        # Gen. datasets
        split_train_valid_test = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --y_col y_col --seed 42"
        preprocessing = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/1_preprocess_data.py -f mono_class_mono_label_train.csv -p preprocess_P1 --target_cols y_col"
        # We don't apply the preprocessing on the validation dataset. We will use the train dataset as validation to simplify
        self.assertEqual(subprocess.run(split_train_valid_test, shell=True).returncode, 0)
        self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)

    def test02_Model_RidgeClassifier(self):
        '''Test of the Ridge Classifier'''
        print('            ------------------ >     Test of the Ridge Classifier     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_ridge_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_ridge_classifier.ModelRidgeClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                     preprocess_pipeline=preprocess_pipeline,
                                                                     ridge_params={'alpha': 1.0},
                                                                     multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                     multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_ridge_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_ridge_classifier.ModelRidgeClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                       preprocess_pipeline=preprocess_pipeline,
                                                                       ridge_params={'alpha': 1.0},
                                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                       multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_ridge_classifier_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_ridge_classifier.ModelRidgeClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                          preprocess_pipeline=preprocess_pipeline,
            #                                                          ridge_params={'alpha': 1.0},
            #                                                          multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                          multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_RidgeClassifier failed')

    def test03_Model_LogisticRegressionClassifier(self):
        '''Test of the Logistic Regression Classifier'''
        print('            ------------------ >     Test of the Logistic Regression Classifier     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_logistic_regression_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_logistic_regression_classifier.ModelLogisticRegressionClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                                                preprocess_pipeline=preprocess_pipeline,
                                                                                                lr_params={'penalty': 'l2', 'C': 1.0, 'max_iter': 100},
                                                                                                multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                                                multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_logistic_regression_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_logistic_regression_classifier.ModelLogisticRegressionClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                                                  preprocess_pipeline=preprocess_pipeline,
                                                                                                  lr_params={'penalty': 'l2', 'C': 1.0, 'max_iter': 100},
                                                                                                  multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                                                  multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_logistic_regression_classifier_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_logistic_regression_classifier.ModelLogisticRegressionClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                                                       preprocess_pipeline=preprocess_pipeline,
            #                                                                                       lr_params={'penalty': 'l2', 'C': 1.0, 'max_iter': 100},
            #                                                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                                                       multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_LogisticRegressionClassifier failed')

    def test04_Model_SVMClassifier(self):
        '''Test of the Support Vector Machine Classifier'''
        print('            ------------------ >     Test of the Support Vector Machine Classifier     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_svm_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_svm_classifier.ModelSVMClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 svm_params={'C': 1.0, 'kernel': 'linear'},
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_svm_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_svm_classifier.ModelSVMClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                   preprocess_pipeline=preprocess_pipeline,
                                                                   svm_params={'C': 1.0, 'kernel': 'linear'},
                                                                   multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                   multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_svm_classifier_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_svm_classifier.ModelSVMClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                        preprocess_pipeline=preprocess_pipeline,
            #                                                        svm_params={'C': 1.0, 'kernel': 'linear'},
            #                                                        multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                        multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_SVMClassifier failed')

    def test05_Model_SGDClassifier(self):
        '''Test of the Stochastic Gradient Descent Classifier'''
        print('            ------------------ >     Test of the Stochastic Gradient Descent Classifier     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_sgd_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_sgd_classifier.ModelSGDClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 sgd_params={'loss': 'log', 'penalty': 'elasticnet', 'l1_ratio': 0.5},
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_sgd_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_sgd_classifier.ModelSGDClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                   preprocess_pipeline=preprocess_pipeline,
                                                                   sgd_params={'loss': 'log', 'penalty': 'elasticnet', 'l1_ratio': 0.5},
                                                                   multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                   multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_sgd_classifier_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_sgd_classifier.ModelSGDClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                        preprocess_pipeline=preprocess_pipeline,
            #                                                        sgd_params={'loss': 'log', 'penalty': 'elasticnet', 'l1_ratio': 0.5},
            #                                                        multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                        multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_SGDClassifier failed')

    def test06_Model_KNNClassifier(self):
        '''Test of the K-nearest Neighbors Classifier'''
        print('            ------------------ >     Test of the K-nearest Neighbors Classifier     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_knn_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_knn_classifier.ModelKNNClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 knn_params={'n_neighbors': 1, 'algorithm': 'brute'},
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_knn_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_knn_classifier.ModelKNNClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                   preprocess_pipeline=preprocess_pipeline,
                                                                   knn_params={'n_neighbors': 1, 'algorithm': 'brute'},
                                                                   multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                   multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_knn_classifier_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_knn_classifier.ModelKNNClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                        preprocess_pipeline=preprocess_pipeline,
            #                                                        knn_params={'n_neighbors': 1, 'algorithm': 'brute'},
            #                                                        multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                        multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_KNNClassifier failed')

    def test07_Model_RFClassifier(self):
        '''Test of the Random Forest Classifier'''
        print('            ------------------ >     Test of the Random Forest Classifier     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_rf_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_rf_classifier.ModelRFClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                               preprocess_pipeline=preprocess_pipeline,
                                                               rf_params={'n_estimators': 10, 'max_depth': 5},
                                                               multi_label=False, model_name=model_name, model_dir=model_dir,
                                                               multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_rf_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_rf_classifier.ModelRFClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 rf_params={'n_estimators': 10, 'max_depth': 5},
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_rf_classifier_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_rf_classifier.ModelRFClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                      preprocess_pipeline=preprocess_pipeline,
            #                                                      rf_params={'n_estimators': 10, 'max_depth': 5},
            #                                                      multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                      multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_RFClassifier failed')

    def test08_Model_GBTClassifier(self):
        '''Test of the Gradient Boosted Tree Classifier'''
        print('            ------------------ >     Test of the Gradient Boosted Tree Classifier     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_gbt_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_gbt_classifier.ModelGBTClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 gbt_params={'loss': 'deviance', 'learning_rate': 0.1,
                                                                             'n_estimators': 10, 'subsample': 1.0,
                                                                             'criterion': 'friedman_mse'},
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_gbt_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_gbt_classifier.ModelGBTClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                   preprocess_pipeline=preprocess_pipeline,
                                                                   gbt_params={'loss': 'deviance', 'learning_rate': 0.1,
                                                                               'n_estimators': 10, 'subsample': 1.0,
                                                                               'criterion': 'friedman_mse'},
                                                                   multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                   multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_gbt_classifier_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_gbt_classifier.ModelGBTClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                        preprocess_pipeline=preprocess_pipeline,
            #                                                        gbt_params={'loss': 'deviance', 'learning_rate': 0.1,
            #                                                                    'n_estimators': 10, 'subsample': 1.0,
            #                                                                    'criterion': 'friedman_mse'},
            #                                                        multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                        multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_GBTClassifier failed')

    def test09_Model_XgboostClassifier(self):
        '''Test of the Xgboost'''
        print('            ------------------ >     Test of the Xgboost     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_xgboost_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_xgboost_classifier.ModelXgboostClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                         preprocess_pipeline=preprocess_pipeline,
                                                                         xgboost_params={'n_estimators': 20, 'booster': 'gbtree',
                                                                                         'eta': 0.3, 'gamma': 0, 'max_depth': 6},
                                                                         multi_label=False, model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                    filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_XgboostClassifier failed')

    def test10_Model_LGBMClassifier(self):
        '''Test of the Light GBM'''
        print('            ------------------ >     Test of the Light GBM     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_lgbm_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_lgbm_classifier.ModelLGBMClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                   preprocess_pipeline=preprocess_pipeline,
                                                                   lgbm_params={'num_leaves': 31, 'max_depth': -1,
                                                                                'learning_rate': 0.1, 'n_estimators': 100},
                                                                   multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                   multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_lgbm_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_lgbm_classifier.ModelLGBMClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                     preprocess_pipeline=preprocess_pipeline,
                                                                     lgbm_params={'num_leaves': 31, 'max_depth': -1,
                                                                                  'learning_rate': 0.1, 'n_estimators': 100},
                                                                     multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                     multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_lgbm_classifier_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_lgbm_classifier.ModelLGBMClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                          preprocess_pipeline=preprocess_pipeline,
            #                                                          lgbm_params={'num_leaves': 31, 'max_depth': -1,
            #                                                                       'learning_rate': 0.1, 'n_estimators': 100},
            #                                                          multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                          multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_LGBMClassifier failed')

    def test11_Model_DenseClassifier(self):
        '''Test of the Dense Classifier'''
        print('            ------------------ >     Test of the Dense Classifier     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_dense_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_dense_classifier.ModelDenseClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                     preprocess_pipeline=preprocess_pipeline,
                                                                     batch_size=16, epochs=10, patience=5,
                                                                     multi_label=False, model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                    filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Retrieve model & run a second training (continue training)
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            self.assertNotEqual(model_dir, test_model.model_dir)

            # Test second trained model
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_DenseClassifier failed')


def test_model_mono_class_multi_label(test_class, test_model):
    '''Generic fonction to test a given model for mono-class/multi-labels'''

    # Check if files exist
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    # Try some functions
    df_input_preds = pd.DataFrame({
        'col_1': [-10, -5, 0, 3],
        'col_2': [-1, 2, -8, 2],
    })
    df_input_preds_prep = utils_models.apply_pipeline(df_input_preds, test_model.preprocess_pipeline)
    index_col_1 = test_model.list_classes.index('y_col_1')
    index_col_2 = test_model.list_classes.index('y_col_2')
    pred_none = [0, 0]
    pred_col_1 = [0, 0]
    pred_col_1[index_col_1] = 1
    pred_col_2 = [0, 0]
    pred_col_2[index_col_2] = 1
    pred_all = [1, 1]
    # predict
    preds = test_model.predict(df_input_preds_prep)
    test_class.assertEqual([list(_) for _ in preds], [pred_none, pred_col_1, pred_col_2, pred_all])
    # predict_proba
    probas = test_model.predict_proba(df_input_preds_prep)
    test_class.assertLess(probas[0][index_col_1], 0.5)
    test_class.assertLess(probas[0][index_col_2], 0.5)
    test_class.assertGreater(probas[1][index_col_1], 0.5)
    test_class.assertLess(probas[1][index_col_2], 0.5)
    test_class.assertLess(probas[2][index_col_1], 0.5)
    test_class.assertGreater(probas[2][index_col_2], 0.5)
    test_class.assertGreater(probas[3][index_col_1], 0.5)
    test_class.assertGreater(probas[3][index_col_2], 0.5)
    # predict w/ return_proba=True
    probas2 = test_model.predict(df_input_preds_prep, return_proba=True)
    test_class.assertLess(probas2[0][index_col_1], 0.5)
    test_class.assertLess(probas2[0][index_col_2], 0.5)
    test_class.assertGreater(probas2[1][index_col_1], 0.5)
    test_class.assertLess(probas2[1][index_col_2], 0.5)
    test_class.assertLess(probas2[2][index_col_1], 0.5)
    test_class.assertGreater(probas2[2][index_col_2], 0.5)
    test_class.assertGreater(probas2[3][index_col_1], 0.5)
    test_class.assertGreater(probas2[3][index_col_2], 0.5)
    # predict_with_proba
    pred_proba = test_model.predict_with_proba(df_input_preds_prep)
    test_class.assertEqual([list(_) for _ in pred_proba[0]], [pred_none, pred_col_1, pred_col_2, pred_all])
    test_class.assertLess(pred_proba[1][0][index_col_1], 0.5)
    test_class.assertLess(pred_proba[1][0][index_col_2], 0.5)
    test_class.assertGreater(pred_proba[1][1][index_col_1], 0.5)
    test_class.assertLess(pred_proba[1][1][index_col_2], 0.5)
    test_class.assertLess(pred_proba[1][2][index_col_1], 0.5)
    test_class.assertGreater(pred_proba[1][2][index_col_2], 0.5)
    test_class.assertGreater(pred_proba[1][3][index_col_1], 0.5)
    test_class.assertGreater(pred_proba[1][3][index_col_2], 0.5)
    # get_predict_position
    # position start at 1
    with test_class.assertRaises(ValueError):
        test_model.get_predict_position(df_input_preds_prep, ['toto', 'tata', 'toto', 'titi'])  # Does not work with multi-labels
    # get_classes_from_proba
    test_class.assertEqual([list(_) for _ in test_model.get_classes_from_proba(probas)], [pred_none, pred_col_1, pred_col_2, pred_all])
    # get_top_n_from_proba
    with test_class.assertRaises(ValueError):
        test_model.get_top_n_from_proba(probas, n=2) # Does not work with multi-labels
    # inverse_transform
    test_class.assertEqual(list(test_model.inverse_transform(preds)), [(), ('y_col_1',), ('y_col_2',), ('y_col_1', 'y_col_2')])


class Case3_MonoClassMultiLabel(unittest.TestCase):
    '''Class to test the mono-class / multi-labels case'''

    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for the mono-class / multi-labels case")

        # Clean repo
        pipelines_dir = os.path.join(full_path_lib, 'test_template_num-pipelines')
        models_dir = os.path.join(full_path_lib, 'test_template_num-models')
        if os.path.isdir(pipelines_dir):
            shutil.rmtree(pipelines_dir)
            os.mkdir(pipelines_dir)
        if os.path.isdir(models_dir):
            shutil.rmtree(models_dir)
            os.mkdir(models_dir)

        # Gen. datasets
        split_train_valid_test = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_multi_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --y_col y_col_1 --seed 42"
        preprocessing = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/1_preprocess_data.py -f mono_class_multi_label_train.csv -p preprocess_P1 --target_cols y_col_1 y_col_2"
        # We don't apply the preprocessing on the validation dataset. We will use the train dataset as validation to simplify
        self.assertEqual(subprocess.run(split_train_valid_test, shell=True).returncode, 0)
        self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)

    def test02_Model_RidgeClassifier(self):
        '''Test of the Ridge Classifier'''
        print('            ------------------ >     Test of the Ridge Classifier     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_ridge_classifier_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_ridge_classifier.ModelRidgeClassifier(x_col=['col_1', 'col_2'], y_col=['y_col_1', 'y_col_2'], level_save='HIGH',
                                                                     preprocess_pipeline=preprocess_pipeline,
                                                                     ridge_params={'alpha': 0.01, 'solver': 'lsqr'},
                                                                     multi_label=True, model_name=model_name, model_dir=model_dir,
                                                                     multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_RidgeClassifier failed')

    def test03_Model_LogisticRegressionClassifier(self):
        '''Test of the Logistic Regression Classifier'''
        print('            ------------------ >     Test of the Logistic Regression Classifier     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_logistic_regression_classifier_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_logistic_regression_classifier.ModelLogisticRegressionClassifier(x_col=['col_1', 'col_2'], y_col=['y_col_1', 'y_col_2'], level_save='HIGH',
                                                                                                preprocess_pipeline=preprocess_pipeline,
                                                                                                lr_params={'penalty': 'l2', 'C': 1.0, 'max_iter': 100},
                                                                                                multi_label=True, model_name=model_name, model_dir=model_dir,
                                                                                                multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_LogisticRegressionClassifier failed')

    def test04_Model_SVMClassifier(self):
        '''Test of the Support Vector Machine Classifier'''
        print('            ------------------ >     Test of the Support Vector Machine Classifier     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_svm_classifier_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_svm_classifier.ModelSVMClassifier(x_col=['col_1', 'col_2'], y_col=['y_col_1', 'y_col_2'], level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 svm_params={'C': 1.0, 'kernel': 'linear'},
                                                                 multi_label=True, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_SVMClassifier failed')

    def test05_Model_SGDClassifier(self):
        '''Test of the Stochastic Gradient Descent Classifier'''
        print('            ------------------ >     Test of the Stochastic Gradient Descent Classifier     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_sgd_classifier_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_sgd_classifier.ModelSGDClassifier(x_col=['col_1', 'col_2'], y_col=['y_col_1', 'y_col_2'], level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 sgd_params={'loss': 'log', 'penalty': 'elasticnet', 'l1_ratio': 0.5},
                                                                 multi_label=True, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_SGDClassifier failed')

    def test06_Model_KNNClassifier(self):
        '''Test of the K-nearest Neighbors Classifier'''
        print('            ------------------ >     Test of the K-nearest Neighbors Classifier     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_knn_classifier_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_knn_classifier.ModelKNNClassifier(x_col=['col_1', 'col_2'], y_col=['y_col_1', 'y_col_2'], level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 knn_params={'n_neighbors': 1, 'algorithm': 'brute'},
                                                                 multi_label=True, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_KNNClassifier failed')

    def test07_Model_RFClassifier(self):
        '''Test of the Random Forest Classifier'''
        print('            ------------------ >     Test of the Random Forest Classifier     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_rf_classifier_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_rf_classifier.ModelRFClassifier(x_col=['col_1', 'col_2'], y_col=['y_col_1', 'y_col_2'], level_save='HIGH',
                                                               preprocess_pipeline=preprocess_pipeline,
                                                               rf_params={'n_estimators': 10, 'max_depth': 5},
                                                               multi_label=True, model_name=model_name, model_dir=model_dir,
                                                               multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_RFClassifier failed')

    def test08_Model_GBTClassifier(self):
        '''Test of the Gradient Boosted Tree Classifier'''
        print('            ------------------ >     Test of the Gradient Boosted Tree Classifier     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_gbt_classifier_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_gbt_classifier.ModelGBTClassifier(x_col=['col_1', 'col_2'], y_col=['y_col_1', 'y_col_2'], level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 gbt_params={'loss': 'deviance', 'learning_rate': 0.1,
                                                                             'n_estimators': 10, 'subsample': 1.0,
                                                                             'criterion': 'friedman_mse'},
                                                                 multi_label=True, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_GBTClassifier failed')

    def test09_Model_XgboostClassifier(self):
        '''Test of the Xgboost'''
        print('            ------------------ >     Test of the Xgboost     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_xgboost_classifier_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_xgboost_classifier.ModelXgboostClassifier(x_col=['col_1', 'col_2'], y_col=['y_col_1', 'y_col_2'], level_save='HIGH',
                                                                         preprocess_pipeline=preprocess_pipeline,
                                                                         xgboost_params={'n_estimators': 20, 'booster': 'gbtree',
                                                                                         'eta': 0.3, 'gamma': 0, 'max_depth': 6},
                                                                         multi_label=True, model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_XgboostClassifier failed')

    def test10_Model_LGBMClassifier(self):
        '''Test of the Light GBM'''
        print('            ------------------ >     Test of the Light GBM     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_lgbm_classifier_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_lgbm_classifier.ModelLGBMClassifier(x_col=['col_1', 'col_2'], y_col=['y_col_1', 'y_col_2'], level_save='HIGH',
                                                                   preprocess_pipeline=preprocess_pipeline,
                                                                   lgbm_params={'num_leaves': 31, 'max_depth': -1,
                                                                                'learning_rate': 0.1, 'n_estimators': 100},
                                                                   multi_label=True, model_name=model_name, model_dir=model_dir,
                                                                   multiclass_strategy=None)

            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_LGBMClassifier failed')

    def test11_Model_DenseClassifier(self):
        '''Test of the Dense Classifier'''
        print('            ------------------ >     Test of the Dense Classifier     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_dense_classifier_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_dense_classifier.ModelDenseClassifier(x_col=['col_1', 'col_2'], y_col=['y_col_1', 'y_col_2'], level_save='HIGH',
                                                                     preprocess_pipeline=preprocess_pipeline,
                                                                     batch_size=16, epochs=50, patience=5,
                                                                     multi_label=True, model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)

            # Retrieve model & run a second training (continue training)
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            self.assertNotEqual(model_dir, test_model.model_dir)

            # Test second trained model
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_DenseClassifier failed')


def test_model_multi_class_mono_label(test_class, test_model):
    '''Generic fonction to test a given model for multi-classes/mono-label'''

    # Check if files exist
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    # Try some functions
    df_input_preds = pd.DataFrame({
        'col_1': [12, -6, 5, -10],
        'col_2': [6, 12, -6, -5],
    })
    df_input_preds_prep = utils_models.apply_pipeline(df_input_preds, test_model.preprocess_pipeline)
    index_none = test_model.list_classes.index('none')
    index_a = test_model.list_classes.index('a')
    index_b = test_model.list_classes.index('b')
    index_both = test_model.list_classes.index('both')
    pred_none = [0, 0, 0, 0]
    pred_none[index_none] = 1
    pred_a = [0, 0, 0, 0]
    pred_a[index_a] = 1
    pred_b = [0, 0, 0, 0]
    pred_b[index_b] = 1
    pred_both = [0, 0, 0, 0]
    pred_both[index_both] = 1
    # predict
    preds = test_model.predict(df_input_preds_prep)
    test_class.assertEqual(list(preds), ['none', 'a', 'b', 'both'])
    # predict_proba
    probas = test_model.predict_proba(df_input_preds_prep)
    test_class.assertEqual(round(probas.sum(), 3), 4.) # We round for deep learning models
    test_class.assertGreater(probas[0][index_none], 1/4)
    test_class.assertLess(probas[0][index_a], probas[0][index_none])
    test_class.assertLess(probas[0][index_b], probas[0][index_none])
    test_class.assertLess(probas[0][index_both], probas[0][index_none])
    test_class.assertLess(probas[1][index_none], probas[1][index_a])
    test_class.assertGreater(probas[1][index_a], 1/4)
    test_class.assertLess(probas[1][index_b], probas[1][index_a])
    test_class.assertLess(probas[1][index_both], probas[1][index_a])
    test_class.assertLess(probas[2][index_none], probas[2][index_b])
    test_class.assertLess(probas[2][index_a], probas[2][index_b])
    test_class.assertGreater(probas[2][index_b], 1/4)
    test_class.assertLess(probas[2][index_both], probas[2][index_b])
    test_class.assertLess(probas[3][index_none], probas[3][index_both])
    test_class.assertLess(probas[3][index_a], probas[3][index_both])
    test_class.assertLess(probas[3][index_b], probas[3][index_both])
    test_class.assertGreater(probas[3][index_both], 1/4)
    # predict w/ return_proba=True
    probas2 = test_model.predict(df_input_preds_prep, return_proba=True)
    test_class.assertEqual(round(probas2.sum(), 3), 4.) # We round for deep learning models
    test_class.assertGreater(probas2[0][index_none], 1/4)
    test_class.assertLess(probas2[0][index_a], probas2[0][index_none])
    test_class.assertLess(probas2[0][index_b], probas2[0][index_none])
    test_class.assertLess(probas2[0][index_both], probas2[0][index_none])
    test_class.assertLess(probas2[1][index_none], probas2[1][index_a])
    test_class.assertGreater(probas2[1][index_a], 1/4)
    test_class.assertLess(probas2[1][index_b], probas2[1][index_a])
    test_class.assertLess(probas2[1][index_both], probas2[1][index_a])
    test_class.assertLess(probas2[2][index_none], probas2[2][index_b])
    test_class.assertLess(probas2[2][index_a], probas2[2][index_b])
    test_class.assertGreater(probas2[2][index_b], 1/4)
    test_class.assertLess(probas2[2][index_both], probas2[2][index_b])
    test_class.assertLess(probas2[3][index_none], probas2[3][index_both])
    test_class.assertLess(probas2[3][index_a], probas2[3][index_both])
    test_class.assertLess(probas2[3][index_b], probas2[3][index_both])
    test_class.assertGreater(probas2[3][index_both], 1/4)
    # predict_with_proba
    pred_proba = test_model.predict_with_proba(df_input_preds_prep)
    test_class.assertEqual(list(pred_proba[0]), ['none', 'a', 'b', 'both'])
    test_class.assertEqual(round(pred_proba[1].sum(), 3), 4.) # We round for deep learning models
    test_class.assertGreater(pred_proba[1][0][index_none], 1/4)
    test_class.assertLess(pred_proba[1][0][index_a], pred_proba[1][0][index_none])
    test_class.assertLess(pred_proba[1][0][index_b], pred_proba[1][0][index_none])
    test_class.assertLess(pred_proba[1][0][index_both], pred_proba[1][0][index_none])
    test_class.assertLess(pred_proba[1][1][index_none], pred_proba[1][1][index_a])
    test_class.assertGreater(pred_proba[1][1][index_a], 1/4)
    test_class.assertLess(pred_proba[1][1][index_b], pred_proba[1][1][index_a])
    test_class.assertLess(pred_proba[1][1][index_both], pred_proba[1][1][index_a])
    test_class.assertLess(pred_proba[1][2][index_none], pred_proba[1][2][index_b])
    test_class.assertLess(pred_proba[1][2][index_a], pred_proba[1][2][index_b])
    test_class.assertGreater(pred_proba[1][2][index_b], 1/4)
    test_class.assertLess(pred_proba[1][2][index_both], pred_proba[1][2][index_b])
    test_class.assertLess(pred_proba[1][3][index_none], pred_proba[1][3][index_both])
    test_class.assertLess(pred_proba[1][3][index_a], pred_proba[1][3][index_both])
    test_class.assertLess(pred_proba[1][3][index_b], pred_proba[1][3][index_both])
    test_class.assertGreater(pred_proba[1][3][index_both], 1/4)
    # get_predict_position
    df_input_get_predict_position = pd.DataFrame({
        'col_1': [12, -6, 5, -10, 5],
        'col_2': [6, 12, -6, -5, 9],
    })
    df_input_get_predict_position_prep = utils_models.apply_pipeline(df_input_get_predict_position, test_model.preprocess_pipeline)
    # position start at 1
    predict_pos = test_model.get_predict_position(df_input_get_predict_position_prep, ['none', 'a', 'a', 'both', 'toto'])
    test_class.assertEqual(list(predict_pos[[0, 1, 3, 4]]), [1, 1, 1, -1])
    test_class.assertGreater(predict_pos[2], 1)
    # get_classes_from_proba
    test_class.assertEqual(list(test_model.get_classes_from_proba(probas)), ['none', 'a', 'b', 'both'])
    # get_top_n_from_proba
    with test_class.assertRaises(ValueError):
        test_model.get_top_n_from_proba(probas, n=5) # Only 4 classes in our model
    top_n, top_n_proba = test_model.get_top_n_from_proba(probas, n=4)
    test_class.assertEqual([_[0] for _ in top_n], ['none', 'a', 'b', 'both'])
    test_class.assertEqual(sorted(top_n[0]), sorted(['none', 'a', 'b', 'both']))
    test_class.assertEqual(sorted(top_n[1]), sorted(['none', 'a', 'b', 'both']))
    test_class.assertEqual(sorted(top_n[2]), sorted(['none', 'a', 'b', 'both']))
    test_class.assertEqual(sorted(top_n[3]), sorted(['none', 'a', 'b', 'both']))
    test_class.assertEqual([_[0] for _ in top_n_proba], [probas[0][index_none], probas[1][index_a], probas[2][index_b], probas[3][index_both]])
    # inverse_transform
    test_class.assertEqual(list(test_model.inverse_transform(preds)), ['none', 'a', 'b', 'both'])


class Case4_MultiClassMonoLabel(unittest.TestCase):
    '''Class to test the multi-classes / mono-label case'''

    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for the multi-classes / mono-label case")

        # Clean repo
        pipelines_dir = os.path.join(full_path_lib, 'test_template_num-pipelines')
        models_dir = os.path.join(full_path_lib, 'test_template_num-models')
        if os.path.isdir(pipelines_dir):
            shutil.rmtree(pipelines_dir)
            os.mkdir(pipelines_dir)
        if os.path.isdir(models_dir):
            shutil.rmtree(models_dir)
            os.mkdir(models_dir)

        # Gen. datasets
        split_train_valid_test = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_split_train_valid_test.py --overwrite -f multi_class_mono_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --y_col y_col --seed 42"
        preprocessing = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/1_preprocess_data.py -f multi_class_mono_label_train.csv -p preprocess_P1 --target_cols y_col"
        # We don't apply the preprocessing on the validation dataset. We will use the train dataset as validation to simplify
        self.assertEqual(subprocess.run(split_train_valid_test, shell=True).returncode, 0)
        self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)

    def test02_Model_RidgeClassifier(self):
        '''Test of the Ridge Classifier'''
        print('            ------------------ >     Test of the Ridge Classifier     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_ridge_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_ridge_classifier.ModelRidgeClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                     preprocess_pipeline=preprocess_pipeline,
                                                                     ridge_params={'alpha': 1.0},
                                                                     multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                     multiclass_strategy=None)

            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_ridge_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_ridge_classifier.ModelRidgeClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                     preprocess_pipeline=preprocess_pipeline,
                                                                     ridge_params={'alpha': 1.0},
                                                                     multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                     multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_ridge_classifier_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_ridge_classifier.ModelRidgeClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                          preprocess_pipeline=preprocess_pipeline,
            #                                                          ridge_params={'alpha': 1.0},
            #                                                          multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                          multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_RidgeClassifier failed')

    def test03_Model_LogisticRegressionClassifier(self):
        '''Test of the Logistic Regression Classifier'''
        print('            ------------------ >     Test of the Logistic Regression Classifier     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_logistic_regression_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_logistic_regression_classifier.ModelLogisticRegressionClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                                                preprocess_pipeline=preprocess_pipeline,
                                                                                                lr_params={'penalty': 'l2', 'C': 1.0, 'max_iter': 100},
                                                                                                multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                                                multiclass_strategy=None)

            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_logistic_regression_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_logistic_regression_classifier.ModelLogisticRegressionClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                                                  preprocess_pipeline=preprocess_pipeline,
                                                                                                  lr_params={'penalty': 'l2', 'C': 1.0, 'max_iter': 100},
                                                                                                  multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                                                  multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_logistic_regression_classifier_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_logistic_regression_classifier.ModelLogisticRegressionClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                                                       preprocess_pipeline=preprocess_pipeline,
            #                                                                                       lr_params={'penalty': 'l2', 'C': 1.0, 'max_iter': 100},
            #                                                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                                                       multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_LogisticRegressionClassifier failed')

    def test04_Model_SVMClassifier(self):
        '''Test of the Support Vector Machine Classifier'''
        print('            ------------------ >     Test of the Support Vector Machine Classifier     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_svm_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_svm_classifier.ModelSVMClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 svm_params={'C': 1.0, 'kernel': 'linear'},
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy=None)

            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_svm_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_svm_classifier.ModelSVMClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                   preprocess_pipeline=preprocess_pipeline,
                                                                   svm_params={'C': 1.0, 'kernel': 'linear'},
                                                                   multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                   multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_svm_classifier_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_svm_classifier.ModelSVMClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                        preprocess_pipeline=preprocess_pipeline,
            #                                                        svm_params={'C': 1.0, 'kernel': 'linear'},
            #                                                        multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                        multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_SVMClassifier failed')

    def test05_Model_SGDClassifier(self):
        '''Test of the Stochastic Gradient Descent Classifier'''
        print('            ------------------ >     Test of the Stochastic Gradient Descent Classifier     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_sgd_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_sgd_classifier.ModelSGDClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 sgd_params={'loss': 'log', 'penalty': 'elasticnet', 'l1_ratio': 0.5},
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy=None)

            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_sgd_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_sgd_classifier.ModelSGDClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                   preprocess_pipeline=preprocess_pipeline,
                                                                   sgd_params={'loss': 'log', 'penalty': 'elasticnet', 'l1_ratio': 0.5},
                                                                   multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                   multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_sgd_classifier_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_sgd_classifier.ModelSGDClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                        preprocess_pipeline=preprocess_pipeline,
            #                                                        sgd_params={'loss': 'log', 'penalty': 'elasticnet', 'l1_ratio': 0.5},
            #                                                        multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                        multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_SGDClassifier failed')

    def test06_Model_KNNClassifier(self):
        '''Test of the K-nearest Neighbors Classifier'''
        print('            ------------------ >     Test of the K-nearest Neighbors Classifier     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_knn_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_knn_classifier.ModelKNNClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 knn_params={'n_neighbors': 1, 'algorithm': 'brute'},
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy=None)

            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_knn_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_knn_classifier.ModelKNNClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                   preprocess_pipeline=preprocess_pipeline,
                                                                   knn_params={'n_neighbors': 1, 'algorithm': 'brute'},
                                                                   multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                   multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_knn_classifier_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_knn_classifier.ModelKNNClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                        preprocess_pipeline=preprocess_pipeline,
            #                                                        knn_params={'n_neighbors': 1, 'algorithm': 'brute'},
            #                                                        multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                        multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_KNNClassifier failed')

    def test07_Model_RFClassifier(self):
        '''Test of the Random Forest Classifier'''
        print('            ------------------ >     Test of the Random Forest Classifier     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_rf_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_rf_classifier.ModelRFClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                               preprocess_pipeline=preprocess_pipeline,
                                                               rf_params={'n_estimators': 10, 'max_depth': 5},
                                                               multi_label=False, model_name=model_name, model_dir=model_dir,
                                                               multiclass_strategy=None)

            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_rf_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_rf_classifier.ModelRFClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 rf_params={'n_estimators': 10, 'max_depth': 5},
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_rf_classifier_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_rf_classifier.ModelRFClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                      preprocess_pipeline=preprocess_pipeline,
            #                                                      rf_params={'n_estimators': 10, 'max_depth': 5},
            #                                                      multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                      multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_RFClassifier failed')

    def test08_Model_GBTClassifier(self):
        '''Test of the Gradient Boosted Tree Classifier'''
        print('            ------------------ >     Test of the Gradient Boosted Tree Classifier     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_gbt_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_gbt_classifier.ModelGBTClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                 preprocess_pipeline=preprocess_pipeline,
                                                                 gbt_params={'loss': 'deviance', 'learning_rate': 0.1,
                                                                             'n_estimators': 10, 'subsample': 1.0,
                                                                             'criterion': 'friedman_mse'},
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                 multiclass_strategy=None)

            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_gbt_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_gbt_classifier.ModelGBTClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                   preprocess_pipeline=preprocess_pipeline,
                                                                   gbt_params={'loss': 'deviance', 'learning_rate': 0.1,
                                                                               'n_estimators': 10, 'subsample': 1.0,
                                                                               'criterion': 'friedman_mse'},
                                                                   multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                   multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_gbt_classifier_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_gbt_classifier.ModelGBTClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                        preprocess_pipeline=preprocess_pipeline,
            #                                                        gbt_params={'loss': 'deviance', 'learning_rate': 0.1,
            #                                                                    'n_estimators': 10, 'subsample': 1.0,
            #                                                                    'criterion': 'friedman_mse'},
            #                                                        multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                        multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_GBTClassifier failed')

    def test09_Model_XgboostClassifier(self):
        '''Test of the Xgboost'''
        print('            ------------------ >     Test of the Xgboost     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_xgboost_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_xgboost_classifier.ModelXgboostClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                         preprocess_pipeline=preprocess_pipeline,
                                                                         xgboost_params={'n_estimators': 20, 'booster': 'gbtree',
                                                                                         'eta': 0.3, 'gamma': 0, 'max_depth': 6},
                                                                         multi_label=False, model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                    filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_XgboostClassifier failed')

    def test10_Model_LGBMClassifier(self):
        '''Test of the Light GBM'''
        print('            ------------------ >     Test of the Light GBM     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_lgbm_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_lgbm_classifier.ModelLGBMClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                   preprocess_pipeline=preprocess_pipeline,
                                                                   lgbm_params={'num_leaves': 31, 'max_depth': -1,
                                                                                'learning_rate': 0.1, 'n_estimators': 100},
                                                                   multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                   multiclass_strategy=None)

            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'model_lgbm_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_lgbm_classifier.ModelLGBMClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                     preprocess_pipeline=preprocess_pipeline,
                                                                     lgbm_params={'num_leaves': 31, 'max_depth': -1,
                                                                                  'learning_rate': 0.1, 'n_estimators': 100},
                                                                     multi_label=False, model_name=model_name, model_dir=model_dir,
                                                                     multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' instable
            # model_name = 'model_lgbm_classifier_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_lgbm_classifier.ModelLGBMClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
            #                                                          preprocess_pipeline=preprocess_pipeline,
            #                                                          lgbm_params={'num_leaves': 31, 'max_depth': -1,
            #                                                                       'learning_rate': 0.1, 'n_estimators': 100},
            #                                                          multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                          multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_LGBMClassifier failed')

    def test11_Model_DenseClassifier(self):
        '''Test of the Dense Classifier'''
        print('            ------------------ >     Test of the Dense Classifier     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_classification.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Get pipeline
            pipelines_dirpath = os.path.join(full_path_lib, 'test_template_num-pipelines')
            pipeline_name = os.listdir(pipelines_dirpath)[0]
            preprocess_pipeline, _ = utils_models.load_pipeline(pipeline_name)

            # Set model
            model_name = 'model_dense_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_dense_classifier.ModelDenseClassifier(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                     preprocess_pipeline=preprocess_pipeline,
                                                                     batch_size=16, epochs=10, patience=5,
                                                                     multi_label=False, model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                    filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Retrieve model & run a second training (continue training)
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            self.assertNotEqual(model_dir, test_model.model_dir)

            # Test second trained model
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_DenseClassifier failed')


def test_model_mono_output_regression(test_class, test_model):
    '''Generic fonction to test a given model for mono-output regression'''

    # Check if files exist
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    # Try some functions
    df_input_preds = pd.DataFrame({
        'col_1': [-5, 3],
        'col_2': [2, 2],
    })
    df_input_preds_prep = utils_models.apply_pipeline(df_input_preds, test_model.preprocess_pipeline)
    # predict
    preds = test_model.predict(df_input_preds_prep)
    test_class.assertGreater(preds[0], -4) # should predict -3, test > -4 ...
    test_class.assertLess(preds[0], -2) # ... and < -2
    test_class.assertGreater(preds[1], 4) # should predict 5, test > 4 ...
    test_class.assertLess(preds[1], 6) # ... and < 6
    # predict_proba
    with test_class.assertRaises(ValueError):
        probas = test_model.predict_proba(df_input_preds_prep)
    # predict w/ return_proba=True
    with test_class.assertRaises(ValueError):
        probas2 = test_model.predict(df_input_preds_prep, return_proba=True)
    # inverse_transform
    test_class.assertEqual(list(test_model.inverse_transform(preds)), list(preds))


class Case5_MonoOutputRegression(unittest.TestCase):
    '''Class to test the mono-output regression case'''

    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for the mono-output regression case")

        # Clean repo
        pipelines_dir = os.path.join(full_path_lib, 'test_template_num-pipelines')
        models_dir = os.path.join(full_path_lib, 'test_template_num-models')
        if os.path.isdir(pipelines_dir):
            shutil.rmtree(pipelines_dir)
            os.mkdir(pipelines_dir)
        if os.path.isdir(models_dir):
            shutil.rmtree(models_dir)
            os.mkdir(models_dir)

        # Gen. datasets
        split_train_valid_test = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_output_regression.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(split_train_valid_test, shell=True).returncode, 0)

    def test02_Model_ElasticNetRegressor(self):
        '''Test of the Elastic Net'''
        print('            ------------------ >     Test of the Elastic Net     /   Mono-output Regression')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_regression.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)


            # Set model
            model_name = 'model_elasticnet_regressor_mono_output_regression'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_elasticnet_regressor.ModelElasticNetRegressor(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                             preprocess_pipeline=None,
                                                                             elasticnet_params={'alpha': 0.01, 'l1_ratio': 0.5},
                                                                             model_name=model_name, model_dir=model_dir)

            # Test it (directly on the file with no preprocessing)
            test.main(filename='mono_output_regression_train.csv', y_col='y_col',
                      filename_valid='mono_output_regression_train.csv', model=test_model)
            test_model_mono_output_regression(self, test_model)
        except Exception:
            self.fail('testModel_ElasticNetRegressor failed')

    def test03_Model_BayesianRidgeRegressor(self):
        '''Test of the Linear Regression'''
        print('            ------------------ >     Test of the Linear Regression     /   Mono-output Regression')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_regression.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_bayesian_ridge_regressor_mono_output_regression'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_bayesian_ridge_regressor.ModelBayesianRidgeRegressor(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                                    preprocess_pipeline=None,
                                                                                    bayesian_ridge_params={'n_iter': 300},
                                                                                    model_name=model_name, model_dir=model_dir)

            # Test it (directly on the file with no preprocessing)
            test.main(filename='mono_output_regression_train.csv', y_col='y_col',
                      filename_valid='mono_output_regression_train.csv', model=test_model)
            test_model_mono_output_regression(self, test_model)
        except Exception:
            self.fail('testModel_BayesianRidgeRegressor failed')

    def test04_Model_KernelRidgeRegressor(self):
        '''Test of the Kernel Ridge'''
        print('            ------------------ >     Test of the Kernel Ridge     /   Mono-output Regression')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_regression.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_kernel_ridge_regressor_mono_output_regression'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_kernel_ridge_regressor.ModelKernelRidgeRegressor(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                                preprocess_pipeline=None,
                                                                                kernel_ridge_params={'alpha': 1.0, 'kernel': 'linear'},
                                                                                model_name=model_name, model_dir=model_dir)

            # Test it (directly on the file with no preprocessing)
            test.main(filename='mono_output_regression_train.csv', y_col='y_col',
                      filename_valid='mono_output_regression_train.csv', model=test_model)
            test_model_mono_output_regression(self, test_model)
        except Exception:
            self.fail('testModel_KernelRidgeRegressor failed')

    def test05_Model_SVRRegressor(self):
        '''Test of the Support Vector Regression'''
        print('            ------------------ >     Test of the Support Vector Regression     /   Mono-output Regression')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_regression.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_svr_regressor_mono_output_regression'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_svr_regressor.ModelSVRRegressor(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                               preprocess_pipeline=None,
                                                               svr_params={'kernel': 'linear'},
                                                               model_name=model_name, model_dir=model_dir)

            # Test it (directly on the file with no preprocessing)
            test.main(filename='mono_output_regression_train.csv', y_col='y_col',
                      filename_valid='mono_output_regression_train.csv', model=test_model)
            test_model_mono_output_regression(self, test_model)
        except Exception:
            self.fail('testModel_SVRRegressor failed')

    def test06_Model_SGDRegressor(self):
        '''Test of the Stochastic Gradient Descent'''
        print('            ------------------ >     Test of the Stochastic Gradient Descent     /   Mono-output Regression')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_regression.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_sgd_regressor_mono_output_regression'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_sgd_regressor.ModelSGDRegressor(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                               preprocess_pipeline=None,
                                                               sgd_params={'loss': 'squared_loss', 'penalty': 'elasticnet', 'l1_ratio': 0.5},
                                                               model_name=model_name, model_dir=model_dir)

            # Test it (directly on the file with no preprocessing)
            test.main(filename='mono_output_regression_train.csv', y_col='y_col',
                      filename_valid='mono_output_regression_train.csv', model=test_model)
            test_model_mono_output_regression(self, test_model)
        except Exception:
            self.fail('testModel_SGDRegressor failed')

    def test07_Model_KNNRegressor(self):
        '''Test of the K-nearest Neighbors'''
        print('            ------------------ >     Test of the K-nearest Neighbors     /   Mono-output Regression')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_regression.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_knn_regressor_mono_output_regression'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_knn_regressor.ModelKNNRegressor(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                               preprocess_pipeline=None,
                                                               knn_params={'n_neighbors': 7, 'weights': 'distance'},
                                                               model_name=model_name, model_dir=model_dir)

            # Test it (directly on the file with no preprocessing)
            test.main(filename='mono_output_regression_train.csv', y_col='y_col',
                      filename_valid='mono_output_regression_train.csv', model=test_model)
            test_model_mono_output_regression(self, test_model)
        except Exception:
            self.fail('testModel_KNNRegressor failed')

    def test08_Model_PLSRegressor(self):
        '''Test of the Partial Least Squares'''
        print('            ------------------ >     Test of the Partial Least Squares     /   Mono-output Regression')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_regression.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_pls_regressor_mono_output_regression'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_pls_regressor.ModelPLSRegressor(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                               preprocess_pipeline=None,
                                                               pls_params={'n_components': 2, 'max_iter': 500},
                                                               model_name=model_name, model_dir=model_dir)

            # Test it (directly on the file with no preprocessing)
            test.main(filename='mono_output_regression_train.csv', y_col='y_col',
                      filename_valid='mono_output_regression_train.csv', model=test_model)
            test_model_mono_output_regression(self, test_model)
        except Exception:
            self.fail('testModel_PLSRegressor failed')

    def test09_Model_RFRegressor(self):
        '''Test of the Random Forest'''
        print('            ------------------ >     Test of the Random Forest     /   Mono-output Regression')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_regression.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_rf_regressor_mono_output_regression'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_rf_regressor.ModelRFRegressor(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                             preprocess_pipeline=None,
                                                             rf_params={'n_estimators': 50, 'max_depth': 5},
                                                             model_name=model_name, model_dir=model_dir)

            # Test it (directly on the file with no preprocessing)
            test.main(filename='mono_output_regression_train.csv', y_col='y_col',
                      filename_valid='mono_output_regression_train.csv', model=test_model)
            test_model_mono_output_regression(self, test_model)
        except Exception:
            self.fail('testModel_RFRegressor failed')

    def test10_Model_GBTRegressor(self):
        '''Test of the Gradient Boosting Tree'''
        print('            ------------------ >     Test of the Gradient Boosting Tree     /   Mono-output Regression')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_regression.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_gbt_regressor_mono_output_regression'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_gbt_regressor.ModelGBTRegressor(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                               preprocess_pipeline=None,
                                                               gbt_params={'loss': 'ls', 'learning_rate': 0.1,
                                                                           'n_estimators': 100, 'subsample': 1.0,
                                                                           'criterion': 'friedman_mse'},
                                                               model_name=model_name, model_dir=model_dir)

            # Test it (directly on the file with no preprocessing)
            test.main(filename='mono_output_regression_train.csv', y_col='y_col',
                      filename_valid='mono_output_regression_train.csv', model=test_model)
            test_model_mono_output_regression(self, test_model)
        except Exception:
            self.fail('testModel_GBTRegressor failed')

    def test11_Model_XgboostRegressor(self):
        '''Test of the Xgboost'''
        print('            ------------------ >     Test of the Xgboost     /   Mono-output Regression')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_regression.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_xgboost_regressor_mono_output_regression'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_xgboost_regressor.ModelXgboostRegressor(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                               preprocess_pipeline=None,
                                                               xgboost_params={'n_estimators': 20, 'booster': 'gbtree',
                                                                               'eta': 0.3, 'gamma': 0, 'max_depth': 6},
                                                               model_name=model_name, model_dir=model_dir)

            # Test it (directly on the file with no preprocessing)
            test.main(filename='mono_output_regression_train.csv', y_col='y_col',
                      filename_valid='mono_output_regression_train.csv', model=test_model)
            test_model_mono_output_regression(self, test_model)
        except Exception:
            self.fail('testModel_XgboostRegressor failed')

    def test12_Model_LGBMRegressor(self):
        '''Test of the Light GBM'''
        print('            ------------------ >     Test of the Light GBM     /   Mono-output Regression')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_regression.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_lgbm_regressor_mono_output_regression'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_lgbm_regressor.ModelLGBMRegressor(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                               preprocess_pipeline=None,
                                                               lgbm_params={'num_leaves': 31, 'max_depth': -1,
                                                                            'learning_rate': 0.1, 'n_estimators': 100},
                                                               model_name=model_name, model_dir=model_dir)

            # Test it (directly on the file with no preprocessing)
            test.main(filename='mono_output_regression_train.csv', y_col='y_col',
                      filename_valid='mono_output_regression_train.csv', model=test_model)
            test_model_mono_output_regression(self, test_model)
        except Exception:
            self.fail('testModel_LGBMRegressor failed')

    def test13_Model_DenseRegressor(self):
        '''Test of the Dense'''
        print('            ------------------ >     Test of the Dense     /   Mono-output Regression')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_num-scripts/3_training_regression.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_dense_regressor_mono_output_regression'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_dense_regressor.ModelDenseRegressor(x_col=['col_1', 'col_2'], y_col='y_col', level_save='HIGH',
                                                                   preprocess_pipeline=None,
                                                                   batch_size=16, epochs=50, patience=5,
                                                                   model_name=model_name, model_dir=model_dir)

            # Test it (directly on the file with no preprocessing)
            test.main(filename='mono_output_regression_train.csv', y_col='y_col',
                      filename_valid='mono_output_regression_train.csv', model=test_model)
            test_model_mono_output_regression(self, test_model)
        except Exception:
            self.fail('testModel_DenseRegressor failed')


if __name__ == '__main__':
    # Change directory to script directory parent
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    parentname = str(Path(dname).parent)
    os.chdir(parentname)
    # Manage venv
    full_path_lib = os.path.abspath(os.path.join(os.getcwd(), 'test_template_num'))
    if os.name == 'nt':
        is_windows = True
        # Windows: activate the virtual environment & continue with the other processes
        activate_venv = f"cd {full_path_lib}/venv_test_template_num/Scripts & activate & "
    else:
        is_windows = False
        # UNIX : We can't use "source" so we directly call python/pip from the bin of the virtual environment
        activate_venv = f"{full_path_lib}/venv_test_template_num/bin/"
    # Start tests
    unittest.main()
