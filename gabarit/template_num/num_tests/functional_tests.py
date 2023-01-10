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
import glob
import json
import shutil
import tempfile
import subprocess
import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path
from datetime import datetime

from test_template_num import utils
from test_template_num.models_training import utils_models
from test_template_num.models_training.classifiers import model_xgboost_classifier, model_aggregation_classifier
from test_template_num.models_training.classifiers.models_tensorflow import model_dense_classifier
from test_template_num.models_training.classifiers.models_sklearn import (model_rf_classifier, model_ridge_classifier, model_logistic_regression_classifier,
                                                                          model_sgd_classifier, model_svm_classifier, model_knn_classifier, model_gbt_classifier,
                                                                          model_lgbm_classifier)
from test_template_num.models_training.regressors import model_xgboost_regressor, model_aggregation_regressor
from test_template_num.models_training.regressors.models_tensorflow import model_dense_regressor
from test_template_num.models_training.regressors.models_sklearn import (model_rf_regressor, model_elasticnet_regressor, model_bayesian_ridge_regressor,
                                                                         model_kernel_ridge_regressor, model_svr_regressor, model_sgd_regressor,
                                                                         model_knn_regressor, model_pls_regressor, model_gbt_regressor, model_lgbm_regressor)

def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class Case1_e2e_pipeline(unittest.TestCase):
    '''Class to test the project end to end'''

    def test04_sweetviz_report(self):
        '''Test of the file utils/0_sweetviz_report.py'''
        print("Test of the file utils/0_sweetviz_report.py")

        # We first create a sweetviz configuration file
        config_path = os.path.join(full_path_lib, "test_config.json")
        if os.path.exists(config_path):
            os.remove(config_path)
        with open(config_path, 'w') as f:
            json.dump({"open_browser": False}, f)

        report_path = os.path.join(full_path_lib, "test_template_num-data", "reports", "sweetviz")
        mlruns_artifact_dir = os.path.join(full_path_lib, "test_template_num-data", "experiments", "mlruns_artifacts")

        remove_dir(report_path)
        remove_dir(mlruns_artifact_dir)

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_sweetviz_report.py --overwrite -s mono_class_mono_label.csv --source_names source --config {config_path} --mlflow_experiment sweetviz_experiment_1"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        list_filenames = list(os.walk(report_path))[0][2]
        self.assertTrue(len([filename for filename in list_filenames if "report_source" in filename and "report_source_w" not in filename]) == 1)
        # retry without overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_sweetviz_report.py -s mono_class_mono_label.csv --source_names source --config {config_path} --mlflow_experiment sweetviz_experiment_1"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # retry with overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_sweetviz_report.py --overwrite -s mono_class_mono_label.csv --source_names source --config {config_path} --mlflow_experiment sweetviz_experiment_1"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)

        # Check mlflow report artifact
        print(list(glob.glob(f"{mlruns_artifact_dir}/**/*", recursive=True)))
        self.assertTrue(len(glob.glob(f"{mlruns_artifact_dir}/**/report_source_*.html", recursive=True)) > 0)

        # Compare datasets
        test_compare = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_sweetviz_report.py --overwrite -s mono_class_mono_label_train.csv --source_names train -c mono_class_mono_label_valid.csv mono_class_mono_label_test.csv --compare_names valid test --config {config_path} --mlflow_experiment sweetviz_experiment_2"
        self.assertEqual(subprocess.run(test_compare, shell=True).returncode, 0)
        list_filenames = list(os.walk(report_path))[0][2]
        self.assertTrue(len([filename for filename in list_filenames if "report_train_valid" in filename]) == 1)
        self.assertTrue(len([filename for filename in list_filenames if "report_train_test" in filename]) == 1)

        # Check mlflow report artifact
        print(list(glob.glob(f"{mlruns_artifact_dir}/**/*", recursive=True)))
        self.assertTrue(len(glob.glob(f"{mlruns_artifact_dir}/**/report_train_valid_*.html", recursive=True)) > 0)
        self.assertTrue(len(glob.glob(f"{mlruns_artifact_dir}/**/report_train_test_*.html", recursive=True)) > 0)

        # With target
        # Sweetviz does not with categorical target. Hence, we'll create a temporary dataframe with a binary target.
        data_path = os.path.join(full_path_lib, 'test_template_num-data')
        original_dataset_path = os.path.join(data_path, 'mono_class_mono_label.csv')
        with tempfile.NamedTemporaryFile(dir=data_path) as tmp_file:
            # Read dataset, add a tmp target as binary class & save it in the tmp file
            df = pd.read_csv(original_dataset_path, sep=';', encoding='utf-8')
            df['tmp_target'] = df['y_col'].apply(lambda x: 1. if x == 'oui' else 0.)
            df.to_csv(tmp_file.name, sep=';', encoding='utf-8', index=None)
            test_target = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_sweetviz_report.py --overwrite -s {tmp_file.name} --source_names source_with_target -t tmp_target --config {config_path} --mlflow_experiment sweetviz_experiment_3"
            self.assertEqual(subprocess.run(test_target, shell=True).returncode, 0)
            list_filenames = list(os.walk(report_path))[0][2]
            self.assertTrue(len([filename for filename in list_filenames if "report_source_with_target" in filename]) == 1)

        # Clean up sweetviz config path (useful ?)
        os.remove(config_path)

    def test05_fairness_report(self):
        '''Test of the file utils/0_fairness_report.py'''
        print("Test of the file utils/0_fairness_report.py")

        base_filenames_fairlens = ['data_biased_groups.csv', 'data_distribution_score.csv', 'data_distributions.png']
        set_columns_data_distribution = {'Group', 'Distance', 'Proportion', 'Counts', 'P-Value'}
        set_columns_data_biased = {'Group', 'Distance', 'Proportion', 'Counts', 'P-Value', 'number_of_attributes'}
        base_filenames_fairlearn = ['algo_metrics_by_groups.csv', 'fairness_count_groups.png']
        list_metrics_binary = ['accuracy', 'precision', 'false_positive_rate', 'false_negative_rate', 'f1_score']
        list_metrics_categorical = ['f1_score_weighted', 'f1_score_macro', 'precision_weighted', 'precision_macro', 'accuracy']
        list_metrics_continuous = ['mean_absolute_value', 'root_mean_squared_error', 'mean_absolute_percentage_error', 'R_squared']
        data_path = os.path.join(full_path_lib, 'test_template_num-data')
        mlruns_artifact_dir = os.path.join(full_path_lib, "test_template_num-data", "experiments", "mlruns_artifacts")
        filename_test_fairness = 'test_fairness.csv'

        def test_fairness_script(target, sensitive_cols, nb_bins, with_pred=True):
            output_folder = os.path.join(data_path, 'reports', 'fairness')

            remove_dir(output_folder)
            remove_dir(mlruns_artifact_dir)

            # Constitutes the list of files that should be present
            if target == 'biased_target_binary_int':
                list_metrics = list_metrics_binary
            if target == 'biased_target':
                list_metrics = list_metrics_continuous
            if target == 'biased_target_str':
                list_metrics = list_metrics_categorical
            filenames = ['fairness_algo_barplot_' + metric + '.png' for metric in list_metrics]
            list_filenames_to_check = base_filenames_fairlens.copy()
            if with_pred:
                list_filenames_to_check += base_filenames_fairlearn.copy()
                list_filenames_to_check += filenames.copy()

            # Calculate the theoretical length of the files
            nb_groups_whole = 1
            nb_groups_intersection = 1
            if 'birth_date' in sensitive_cols:
                nb_groups_whole = nb_groups_whole * (nb_bins + 1)
                nb_groups_intersection = nb_groups_intersection * nb_bins
            if 'age' in sensitive_cols:
                nb_groups_whole = nb_groups_whole * (nb_bins + 1)
                nb_groups_intersection = nb_groups_intersection * nb_bins
            if 'citizenship' in sensitive_cols:
                nb_groups_whole = nb_groups_whole * (2 + 1)
                nb_groups_intersection = nb_groups_intersection * 2
            if 'genre' in sensitive_cols:
                nb_groups_whole = nb_groups_whole * (2 + 1)
                nb_groups_intersection = nb_groups_intersection * 2
            nb_groups_whole = nb_groups_whole - 1

            # Run the script
            basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/utils/0_fairness_report.py -f {filename_test_fairness} -t {target} -n {nb_bins} --mlflow_experiment fairness_experiment -s "
            for col in sensitive_cols:
                basic_run += f' {col}'
            if with_pred:
                 basic_run += f' -p {target}_pred'
            self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)

            path_fairness = os.path.join(data_path, 'reports', 'fairness')
            folder_name = list(os.walk(path_fairness))[0][1][0]
            output_path = os.path.join(path_fairness, folder_name)

            # Test the presence (or absence) of files
            for filename in list_filenames_to_check:
                self.assertTrue(os.path.exists(os.path.join(output_path, filename)))
            if not with_pred:
                for filename in base_filenames_fairlearn+filenames:
                    self.assertFalse(os.path.exists(os.path.join(output_path, filename)))

            # Test the file data_distribution_score.csv
            filename = 'data_distribution_score.csv'
            if os.path.exists(os.path.join(output_path, filename)):
                df = pd.read_csv(os.path.join(output_path, filename), sep=';', encoding='utf-8')
                self.assertTrue(set_columns_data_distribution.issubset(set(df.columns)))
                self.assertTrue(len(df) == nb_groups_whole)

            # Test the file data_biased_groups.csv
            filename = 'data_biased_groups.csv'
            if os.path.exists(os.path.join(output_path, filename)):
                df = pd.read_csv(os.path.join(output_path, filename), sep=';', encoding='utf-8')
                self.assertTrue(set_columns_data_biased.issubset(set(df.columns)))

            # Test the file algo_metrics_by_groups.csv
            if with_pred:
                filename = 'algo_metrics_by_groups.csv'
                if os.path.exists(os.path.join(output_path, filename)) and filename[-4:]=='.csv':
                    df = pd.read_csv(os.path.join(output_path, filename), sep=';', encoding='utf-8')
                    set_columns = {'count'}.union(set(list_metrics).union(sensitive_cols))
                    self.assertTrue(set_columns.issubset(set(df.columns)))
                    self.assertTrue(len(df) == (nb_groups_intersection + 1))
            else:
                filename = 'algo_metrics_by_groups.csv'
                self.assertFalse(os.path.exists(os.path.join(output_path, filename)))

            # Check mlflow artifacts
            print(list(glob.glob(f"{mlruns_artifact_dir}/**/*", recursive=True)))
            self.assertTrue(len(glob.glob(f"{mlruns_artifact_dir}/**/data_distributions.png", recursive=True)) > 0)
            self.assertTrue(len(glob.glob(f"{mlruns_artifact_dir}/**/data_distribution_score.json", recursive=True)) > 0)
            self.assertTrue(len(glob.glob(f"{mlruns_artifact_dir}/**/data_biased_groups.json", recursive=True)) > 0)

            # Clean dirs
            remove_dir(output_path)
            remove_dir(mlruns_artifact_dir)

        test_fairness_script(target='biased_target', sensitive_cols=['genre', 'citizenship'], nb_bins=3)
        test_fairness_script(target='biased_target_binary_int', sensitive_cols=['genre', 'citizenship'], nb_bins=3)
        test_fairness_script(target='biased_target_str', sensitive_cols=['genre', 'citizenship'], nb_bins=3)
        test_fairness_script(target='biased_target', sensitive_cols=['genre', 'birth_date'], nb_bins=4)
        test_fairness_script(target='biased_target_binary_int', sensitive_cols=['citizenship', 'birth_date'], nb_bins=2)
        test_fairness_script(target='biased_target_str', sensitive_cols=['age', 'citizenship'], nb_bins=2)
        test_fairness_script(target='biased_target_str', sensitive_cols=['citizenship'], nb_bins=2)
        test_fairness_script(target='biased_target_str', sensitive_cols=['age'], nb_bins=2)
        test_fairness_script(target='biased_target', sensitive_cols=['age', 'citizenship', 'genre'], nb_bins=3)
        test_fairness_script(target='biased_target', sensitive_cols=['genre', 'citizenship'], nb_bins=3, with_pred=False)

    def test08_TrainingE2E(self):
        '''Test of files 3_training_classification.py & 3_training_regression.py'''
        print("Test of files 3_training_classification.py & 3_training_regression.py")

        ################
        # Classification
        ################

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/3_training_classification.py -f mono_class_mono_label_train_preprocess_P1.csv -y y_col --filename_valid mono_class_mono_label_valid_preprocess_P1.csv --mlflow_experiment gabarit_ci/mlflow_test"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_num-models', 'model_ridge_classifier')  # Ridge Classifier by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertEqual(len(listdir), 1)
        model_path = os.path.join(save_model_dir, listdir[0])
        self.assertTrue(os.path.exists(os.path.join(model_path, 'pipeline.info')))
        self.assertTrue(os.path.exists(os.path.join(model_path, 'original_data_samples.csv')))
        # Check mlflow artifact
        mlruns_artifact_dir = os.path.join(full_path_lib, "test_template_num-data", "experiments", "mlruns_artifacts")
        self.assertTrue(len(glob.glob(f"{mlruns_artifact_dir}/**/configurations.json", recursive=True)) > 0)

        # With excluded_cols
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/3_training_classification.py -f mono_class_mono_label_train_preprocess_P1.csv -y y_col --filename_valid mono_class_mono_label_valid_preprocess_P1.csv --excluded_cols col_2 --mlflow_experiment gabarit_ci/mlflow_test"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_num-models', 'model_ridge_classifier')  # Ridge Classifier by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertEqual(len(listdir), 2)
        model_path = os.path.join(save_model_dir, listdir[1])
        self.assertTrue(os.path.exists(os.path.join(model_path, 'pipeline.info')))
        self.assertTrue(os.path.exists(os.path.join(model_path, 'original_data_samples.csv')))

        # Multilabel - no preprocess - no valid
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/3_training_classification.py -f mono_class_multi_label.csv -y y_col_1 y_col_2 --mlflow_experiment gabarit_ci/mlflow_test"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_num-models', 'model_ridge_classifier')  # Ridge Classifier by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertEqual(len(listdir), 3)
        model_path = os.path.join(save_model_dir, listdir[2])
        self.assertTrue(os.path.exists(os.path.join(model_path, 'pipeline.info')))
        self.assertTrue(os.path.exists(os.path.join(model_path, 'original_data_samples.csv')))

        ############
        # Regression
        ############
        # We didn't work on the file

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/3_training_regression.py -f mono_output_regression_train.csv -y y_col --filename_valid mono_output_regression_valid.csv --mlflow_experiment gabarit_ci/mlflow_test"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_num-models', 'model_elasticnet_regressor')  # ElasticNet Regressor by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertEqual(len(listdir), 1)
        model_path = os.path.join(save_model_dir, listdir[0])
        self.assertTrue(os.path.exists(os.path.join(model_path, 'pipeline.info')))
        self.assertTrue(os.path.exists(os.path.join(model_path, 'original_data_samples.csv')))
        # Check mlflow artifact
        mlruns_artifact_dir = os.path.join(full_path_lib, "test_template_num-data", "experiments", "mlruns_artifacts")
        self.assertTrue(len(glob.glob(f"{mlruns_artifact_dir}/**/configurations.json", recursive=True)) > 0)

        # With excluded_cols
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_num-scripts/3_training_regression.py -f mono_output_regression_train.csv -y y_col --filename_valid mono_output_regression_valid.csv --excluded_cols col_2 --mlflow_experiment gabarit_ci/mlflow_test"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_num-models', 'model_elasticnet_regressor')  # ElasticNet Regressor by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertEqual(len(listdir), 2)
        model_path = os.path.join(save_model_dir, listdir[1])
        self.assertTrue(os.path.exists(os.path.join(model_path, 'pipeline.info')))
        self.assertTrue(os.path.exists(os.path.join(model_path, 'original_data_samples.csv')))

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
