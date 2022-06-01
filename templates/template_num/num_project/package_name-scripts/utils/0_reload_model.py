#!/usr/bin/env python3

## Reload a model
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
#
# Ex: python 0_reload_model.py -m best_model -c configurations.json


import os
import json
import ntpath
import logging
import argparse
import pandas as pd
from typing import Union

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training import utils_models, utils_deep_keras
from {{package_name}}.models_training.classifiers import (model_rf_classifier,
                                                          model_dense_classifier,
                                                          model_ridge_classifier,
                                                          model_logistic_regression_classifier,
                                                          model_sgd_classifier,
                                                          model_svm_classifier,
                                                          model_knn_classifier,
                                                          model_gbt_classifier,
                                                          model_lgbm_classifier,
                                                          model_xgboost_classifier)
from {{package_name}}.models_training.regressors import (model_rf_regressor,
                                                         model_dense_regressor,
                                                         model_elasticnet_regressor,
                                                         model_bayesian_ridge_regressor,
                                                         model_kernel_ridge_regressor,
                                                         model_svr_regressor,
                                                         model_sgd_regressor,
                                                         model_knn_regressor,
                                                         model_pls_regressor,
                                                         model_gbt_regressor,
                                                         model_xgboost_regressor,
                                                         model_lgbm_regressor)

# Get logger
logger = logging.getLogger('{{package_name}}.0_reload_model')


def main(model_dir: str, config_file: str = 'configurations.json',
         sklearn_pipeline_file: str = 'sklearn_pipeline_standalone.pkl',
         weights_file: str = 'best.hdf5', xgboost_file: str = 'xbgoost_standalone.model',
         preprocess_pipeline_file: str = 'preprocess_pipeline.pkl') -> None:
    '''Reloads a model

    The idea here is to reload a model that was trained on another package version.
    This is done be reusing 'standalone' files.

    Args:
        model_dir (str): Name of the model to reload (not a path, just the directory name)
    Kwargs:
        config_file (str): Name of the configuration file
        sklearn_pipeline_file (str): Standalone sklearn pipeline file name (pipeline models)
        weights_file (str): Neural Network weights file name (keras models)
        xgboost_file (str): Standalone XGBoost file name (xgboost models)
        preprocess_pipeline_file (str): Name of the preprocessing pipeline file (all models)
    Raises:
        FileNotFoundError: If model can't be found
        FileNotFoundError: If model's configuration does not exist
        ValueError: If the model's type is invalid
    '''
    logger.info(f"Reloading a model ...")

    ##############################################
    # Loading configuration
    ##############################################

    # Get model's path
    models_dir = utils.get_models_path()
    model_path = None
    for path, subdirs, files in os.walk(models_dir):
        for name in subdirs:
            if name == model_dir:
                model_path = os.path.join(path, name)
    if model_path is None:
        raise FileNotFoundError(f"Can't find model {model_dir}")

    # Load conf
    conf_path = os.path.join(model_path, config_file)
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"The file {conf_path} does not exist")
    with open(conf_path, 'r', encoding='{{default_encoding}}') as f:
        configs = json.load(f)


    ##############################################
    # Retrieve model type
    ##############################################

    # Get model type
    model_type_dicts = {
        'model_ridge_classifier': model_ridge_classifier.ModelRidgeClassifier,
        'model_logistic_regression_classifier': model_logistic_regression_classifier.ModelLogisticRegressionClassifier,
        'model_svm_classifier': model_svm_classifier.ModelSVMClassifier,
        'model_sgd_classifier': model_sgd_classifier.ModelSGDClassifier,
        'model_knn_classifier': model_knn_classifier.ModelKNNClassifier,
        'model_rf_classifier': model_rf_classifier.ModelRFClassifier,
        'model_gbt_classifier': model_gbt_classifier.ModelGBTClassifier,
        'model_xgboost_classifier': model_xgboost_classifier.ModelXgboostClassifier,
        'model_lgbm_classifier': model_lgbm_classifier.ModelLGBMClassifier,
        'model_dense_classifier': model_dense_classifier.ModelDenseClassifier,
        'model_dense_regressor': model_dense_regressor.ModelDenseRegressor,
        'model_elasticnet_regressor': model_elasticnet_regressor.ModelElasticNetRegressor,
        'model_bayesian_ridge_regressor': model_bayesian_ridge_regressor.ModelBayesianRidgeRegressor,
        'model_kernel_ridge_regressor': model_kernel_ridge_regressor.ModelKernelRidgeRegressor,
        'model_svr_regressor': model_svr_regressor.ModelSVRRegressor,
        'model_sgd_regressor': model_sgd_regressor.ModelSGDRegressor,
        'model_knn_regressor': model_knn_regressor.ModelKNNRegressor,
        'model_pls_regressor': model_pls_regressor.ModelPLSRegression,
        'model_rf_regressor': model_rf_regressor.ModelRFRegressor,
        'model_gbt_regressor': model_gbt_regressor.ModelGBTRegressor,
        'model_xgboost_regressor': model_xgboost_regressor.ModelXgboostRegressor,
        'model_lgbm_regressor': model_lgbm_regressor.ModelLGBMRegressor,
    }
    model_type = configs['model_name']
    if model_type not in model_type_dicts:
        raise ValueError(f"The model's type {model_type} is invalid.")
    else:
        model_class = model_type_dicts[model_type]


    ##############################################
    # Reload model
    ##############################################

    # Reload model
    model = model_class()
    files_dict = {
        'configuration_path': os.path.join(model_path, config_file) if config_file is not None else None,
        'sklearn_pipeline_path': os.path.join(model_path, sklearn_pipeline_file) if sklearn_pipeline_file is not None else None,
        'hdf5_path': os.path.join(model_path, weights_file) if weights_file is not None else None,
        'xgboost_path': os.path.join(model_path, xgboost_file) if xgboost_file is not None else None,
        'preprocess_pipeline_path': os.path.join(model_path, preprocess_pipeline_file) if preprocess_pipeline_file is not None else None,
    }
    model.reload_from_standalone(**files_dict)


    ##############################################
    # Manage some parameters and save
    ##############################################

    # Save model
    # Reminder: the model's save function prioritize the json_data arg over it's default values
    # hence, it helps with some parameters such as `_get_model`
    list_keys_json_data = ['filename', 'filename_valid', 'min_rows', 'preprocess_str',
                           'fit_time', 'excluded_cols', 'date', '_get_model',
                           '_get_learning_rate_scheduler', 'custom_objects']
    json_data = {key: configs.get(key, None) for key in list_keys_json_data}


    # Add training version
    if 'package_version' in configs:
        # If no trained version yet, use package version
        trained_version = configs.get('trained_version', configs['package_version'])
        if trained_version != utils.get_package_version():
            json_data['trained_version'] = trained_version

    # Save
    json_data = {k: v for k, v in json_data.items() if v is not None}  # Only consider not None values
    model.save(json_data)

    logger.info(f"Model {model_dir} has been successfully reloaded")
    logger.info(f"New model's repository is {model.model_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model_X should be the model's directory name: e.g. model_preprocess_pipeline_svm_2019_12_05-12_57_18
    parser.add_argument('-m', '--model_dir', required=True, help="Name of the model to reload (not a path, just the directory name)")
    parser.add_argument('-c', '--config_file', default='configurations.json', help="Name of the configuration file")
    parser.add_argument('--sklearn_pipeline_file', default='sklearn_pipeline_standalone.pkl', help="Standalone sklearn pipeline file name (pipeline models)")
    parser.add_argument('-w', '--weights_file', default='best.hdf5', help="Neural Network weights file name (keras models)")
    parser.add_argument('--xgboost_file', default='xbgoost_standalone.model', help="Standalone XGBoost file name (xgboost models)")
    parser.add_argument('-p', '--preprocess_pipeline_file', default='preprocess_pipeline.pkl', help="Name of the preprocessing pipeline file (all models)")
    args = parser.parse_args()
    main(model_dir=args.model_dir, config_file=args.config_file,
         sklearn_pipeline_file=args.sklearn_pipeline_file, weights_file=args.weights_file,
         xgboost_file=args.xgboost_file, preprocess_pipeline_file=args.preprocess_pipeline_file)
