#!/usr/bin/env python3

## Training a model - Regression task
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
# Ex: python 3_training_regression.py --filename dataset_train_preprocess_P1.csv --filename_valid dataset_valid_preprocess_P1.csv --y_col Age


import os
# Disable some tensorflow logs right away
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import re
import time
import shutil
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import dill as pickle
from functools import partialmethod
from typing import Union, List, Type, Tuple

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training import utils_models
from {{package_name}}.monitoring.mlflow_logger import MLflowLogger
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.regressors import (
    model_xgboost_regressor,
    model_aggregation_regressor,
)
from {{package_name}}.models_training.regressors.models_tensorflow import (
    model_dense_regressor,
)
from {{package_name}}.models_training.regressors.models_sklearn import (
    model_rf_regressor,
    model_elasticnet_regressor,
    model_bayesian_ridge_regressor,
    model_kernel_ridge_regressor,
    model_svr_regressor,
    model_sgd_regressor,
    model_knn_regressor,
    model_pls_regressor,
    model_gbt_regressor,
    model_lgbm_regressor,
)

# Disable some warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get logger
logger = logging.getLogger('{{package_name}}.3_training_regression')


def main(filename: str, y_col: Union[str, int], excluded_cols: Union[List[Union[str, int]], None] = None,
         filename_valid: Union[str, None] = None, level_save: str = 'HIGH',
         sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}',
         model: Union[Type[ModelClass], None] = None,
         mlflow_experiment: Union[str, None] = None) -> None:
    '''Trains a model

    /!\\ By default, models are fitted on all available columns (except targets) /!\\
    /!\\ Some columns can be excluded with the excluded_cols param /!\\

    Args:
        filename (str): Name of the training dataset (actually a path relative to {{package_name}}-data)
        y_col (str | int): Name of the model's target column - y
    Kwargs:
        excluded_cols (list<str|int>): List of columns to NOT use as model's input (no need to include target columns)
        filename_valid (str): Name of the validation dataset (actually a path relative to {{package_name}}-data)
            If None, we do not use a validation dataset.
                -> for keras models (i.e. Neural Networks), we'll use a portion of the training dataset as the validation.
        level_save (str): Save level
            LOW: statistics + configurations + logger keras - /!\\ the model won't be reusable /!\\ -
            MEDIUM: LOW + hdf5 + pkl + plots
            HIGH: MEDIUM + predictions
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
        model (ModelClass): A model to be fitted. This should only be used for testing purposes.
        mlflow_experiment (str): Name of the current experiment. If None, no experiment will be saved.
    Raises:
        ValueError: If level_save value is not a valid option (['LOW', 'MEDIUM', 'HIGH'])
    '''
    logger.info("Training a model ...")

    if level_save not in ['LOW', 'MEDIUM', 'HIGH']:
        raise ValueError(f"The object level_save ({level_save}) is not a valid option (['LOW', 'MEDIUM', 'HIGH'])")


    ##############################################
    # Manage training dataset
    ##############################################

    # Get dataset
    df_train, preprocess_pipeline_dir = load_dataset(filename, sep=sep, encoding=encoding)

    # Get pipeline
    preprocess_pipeline, preprocess_str = utils_models.load_pipeline(preprocess_pipeline_dir)

    ### INFO
    ### EACH MODEL needs the same target data format:
    ###
    ###   - float
    ###     e.g.
    ###            col_y
    ###            ---
    ###            1.0
    ###            -1.50
    ###            0.4156


    # Ensure target is in float format
    df_train[y_col] = df_train[y_col].astype(float)


    ##############################################
    # Manage validation dataset
    ##############################################

    # Get valid dataset (/!\ we consider that this dataset has the same preprocessing as the training set /!\)
    if filename_valid is not None:
        logger.info(f"Using file {filename_valid} as our validation set.")
        df_valid, preprocess_pipeline_dir_valid = load_dataset(filename_valid, sep=sep, encoding=encoding)
        if preprocess_pipeline_dir_valid != preprocess_pipeline_dir:
            logger.warning("Validation set and training set does not expose the same preprocessing metadata.")
            logger.warning(f"Train : {preprocess_pipeline_dir}")
            logger.warning(f"Valid : {preprocess_pipeline_dir_valid}")
            logger.warning("That will probably lead to bad results !")
            logger.warning("Still continuing...")
        ### INFO: the validation set must have a correct format (cf. traing set info above)
        # Ensure target is in float format
        df_valid[y_col] = df_valid[y_col].astype(float)
    else:
        logger.info("No validation set provided.")
        logger.info("In case of Keras models, we'll use a portion of the training dataset as the validation")


    ##############################################
    # Fit pipeline if "no_preprocess"
    ##############################################

    # If we didn't find any pipeline metadata, it means that the input file is not preprocessed
    # The function `load_pipeline` still returns a "no preprocess" pipeline.
    # This pipeline needs to be fitted (to retrieve info on input columns names, numbers, etc.)
    if preprocess_pipeline_dir is None:
        preprocess_pipeline.fit(df_train.drop(y_col, axis=1), df_train[y_col])


    ##############################################
    # Manage input data
    ##############################################

    # Remove excluded_cols & y cols from model's inputs
    if excluded_cols is None:
        excluded_cols = []
    cols_to_remove = excluded_cols + [y_col]
    x_col = [col for col in df_train.columns if col not in cols_to_remove]

    # We do not check for columns presence in dataframes
    # TODO: add checks ?
    # Get x, y for both train and valid
    x_train, y_train = df_train[x_col], df_train[y_col]
    if filename_valid is not None:
        x_valid, y_valid = df_valid[x_col], df_valid[y_col]
    else:
        x_valid, y_valid = None, None

    ##############################################
    # Model selection
    ##############################################

    # INFO
    # If you want to continue training of a model, it needs to be reloaded here (only some models are compatible)
    # model, _ = utils_models.load_model("dir_model")
    # Then, it is possible to change some parameters such as the learning rate
    # Be careful to work with the same preprocessing as the first training

    # TODO: Add hyperparameters tuning


    if model is None:
        model = model_elasticnet_regressor.ModelElasticNetRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
                                                                    preprocess_pipeline=preprocess_pipeline,
                                                                    elasticnet_params={'alpha': 1.0, 'l1_ratio': 0.5})
        # model = model_bayesian_ridge_regressor.ModelBayesianRidgeRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                                    preprocess_pipeline=preprocess_pipeline,
        #                                                                    bayesian_ridge_params={'n_iter': 300})
        # model = model_kernel_ridge_regressor.ModelKernelRidgeRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                                preprocess_pipeline=preprocess_pipeline,
        #                                                                kernel_ridge_params={'alpha': 1.0, 'kernel': 'linear'})
        # model = model_svr_regressor.ModelSVRRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                               preprocess_pipeline=preprocess_pipeline,
        #                                               svr_params={'kernel': 'linear'})
        # model = model_sgd_regressor.ModelSGDRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                               preprocess_pipeline=preprocess_pipeline,
        #                                               sgd_params={'loss': 'squared_loss', 'penalty': 'elasticnet', 'l1_ratio': 0.5})
        # model = model_knn_regressor.ModelKNNRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                               preprocess_pipeline=preprocess_pipeline,
        #                                               knn_params={'n_neighbors': 7, 'weights': 'distance'})
        # model = model_pls_regressor.ModelPLSRegression(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                preprocess_pipeline=preprocess_pipeline,
        #                                                pls_params={'n_components': 5, 'max_iter': 500})
        # model = model_rf_regressor.ModelRFRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                             preprocess_pipeline=preprocess_pipeline,
        #                                             rf_params={'n_estimators': 50, 'max_depth': 5})
        # model = model_gbt_regressor.ModelGBTRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                            preprocess_pipeline=preprocess_pipeline,
        #                                                            gbt_params={'loss': 'ls', 'learning_rate': 0.1,
        #                                                                        'n_estimators': 100, 'subsample': 1.0,
        #                                                                        'criterion': 'friedman_mse'})
        # model = model_xgboost_regressor.ModelXgboostRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                       preprocess_pipeline=preprocess_pipeline,
        #                                                       xgboost_params={'n_estimators': 20, 'booster': 'gbtree',
        #                                                                       'eta': 0.3, 'gamma': 0, 'max_depth': 6},
        #                                                       early_stopping_rounds=5)
        # model = model_lgbm_regressor.ModelLGBMRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                 preprocess_pipeline=preprocess_pipeline,
        #                                                 lgbm_params={'num_leaves': 31, 'max_depth': -1,
        #                                                              'learning_rate': 0.1, 'n_estimators': 100})
        # model = model_dense_regressor.ModelDenseRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                   preprocess_pipeline=preprocess_pipeline,
        #                                                   batch_size=64, epochs=99, patience=5)
        # modle = model_aggregation_regressor.ModelAggregationRegressor(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                               list_models=[model_sgd_regressor.ModelSGDRegressor(), model_sgd_regressor.ModelSGDRegressor()],
        #                                                               multi_label=multi_label, preprocess_pipeline=preprocess_pipeline,)

    # Display if GPU is being used
    model.display_if_gpu_activated()

    ##############################################
    # Train the model !
    ##############################################

    start_time = time.time()
    logger.info("Starting training the model ...")
    model.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid, with_shuffle=True)
    fit_time = time.time() - start_time


    ##############################################
    # Save trained model
    ##############################################

    # Save model
    model.save(
        json_data={
            'filename': filename,
            'filename_valid': filename_valid,
            'preprocess_str': preprocess_str,
            'fit_time': f"{round(fit_time, 2)}s",
            'excluded_cols': excluded_cols,
        }
    )

    # We also try to save some info from the preprocessing pipeline
    # Define paths
    new_info_path = os.path.join(model.model_dir, 'pipeline.info')
    new_sample_path = os.path.join(model.model_dir, 'original_data_samples.csv')
    # If pipeline already existed (i.e. preprocessed dataset), copy files in the model dir
    if preprocess_pipeline_dir is not None:
        pipeline_path = utils.find_folder_path(preprocess_pipeline_dir, utils.get_pipelines_path())
        info_path = os.path.join(pipeline_path, 'pipeline.info')
        sample_path = os.path.join(pipeline_path, 'dataset_sample.csv')
        if os.path.exists(info_path):
            shutil.copyfile(info_path, new_info_path)
        if os.path.exists(sample_path):
            shutil.copyfile(sample_path, new_sample_path)
    # Else, create files
    else:
        # pipeline.info
        pipeline_dict = {'preprocess_pipeline': preprocess_pipeline, 'preprocess_str': preprocess_str}
        with open(new_info_path, 'wb') as f:
            pickle.dump(pipeline_dict, f)
        # original_data_samples.csv
        df_sample = df_train.sample(min(100, df_train.shape[0]))
        utils.to_csv(df_sample, new_sample_path, sep=sep, encoding=encoding)

    #
    logger.info(f"Model {model.model_name} saved in directory {model.model_dir}")


    ##############################################
    # Model metrics
    ##############################################

    # Series to add
    cols_to_add: List[pd.Series] = []  # You can add columns to save here
    series_to_add_train = [df_train[col] for col in cols_to_add]
    series_to_add_valid = [df_valid[col] for col in cols_to_add]
    gc.collect()  # In some cases, helps with OOMs

    # Get results
    y_pred_train = model.predict(x_train, return_proba=False)
    df_stats = model.get_and_save_metrics(y_train, y_pred_train, df_x=x_train, series_to_add=series_to_add_train, type_data='train')
    gc.collect()  # In some cases, helps with OOMs
    # Get predictions on valid
    if x_valid is not None:
        y_pred_valid = model.predict(x_valid, return_proba=False)
        df_stats = model.get_and_save_metrics(y_valid, y_pred_valid, df_x=x_valid, series_to_add=series_to_add_valid, type_data='valid')
        gc.collect()  # In some cases, helps with OOMs


    ##############################################
    # Logger MLflow
    ##############################################

    # Logging metrics on MLflow
    if mlflow_experiment:
        # Get logger
        mlflow_logger = MLflowLogger(
            experiment_name=f"{{package_name}}/{mlflow_experiment}",
            tracking_uri="{{mlflow_tracking_uri}}",
            artifact_uri="{{mlflow_artifact_uri}}",
        )
        mlflow_logger.set_tag('model_name', f"{os.path.basename(model.model_dir)}")
        mlflow_logger.log_df_stats(df_stats)
        mlflow_logger.log_dict(model.json_dict, "configurations.json")
        # To log more tags/params, you can use mlflow_logger.set_tag(key, value) or mlflow_logger.log_param(key, value)
        # Log a sweetviz report
        report = get_sweetviz_report(df_train=df_train, y_pred_train=y_pred_train, y_col=y_col,
                                     df_valid=df_valid if filename_valid else None,
                                     y_pred_valid=y_pred_valid if filename_valid else None)
        if report:
            mlflow_logger.log_text(report, "sweetviz_train_valid.html")
        # Stop MLflow if started
        mlflow_logger.end_run()


def load_dataset(filename: str, sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> Tuple[pd.DataFrame, str]:
    '''Loads a dataset - retrieves preprocessing metadata

    Args:
        filename (str): Name of the dataset to load (actually a path relative to {{package_name}}-data)
    Kwargs:
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        FileNotFoundError: If the file does not exist in {{package_name}}-data
    Returns:
        pd.DataFrame: Loaded dataframe
        str: Preprocessing used on this dataset (from metadata) - actually a directory name
    '''
    logger.info(f"Loading a dataset ({filename})")

    # Get dataset
    data_path = utils.get_data_path()
    file_path = os.path.join(data_path, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")

    # Load dataset
    df, first_line = utils.read_csv(file_path, sep=sep, encoding=encoding)
    # NaNs must have been handled in the preprocessing !

    # Get preprocessing directory
    if first_line is not None and first_line.startswith('#'):
        preprocess_pipeline_dir = first_line[1:]  # remove # (sharp)
    else:
        preprocess_pipeline_dir = None

    # Return
    return df, preprocess_pipeline_dir


def get_sweetviz_report(df_train: pd.DataFrame, y_pred_train: np.ndarray, y_col: str,
                        df_valid: Union[pd.DataFrame, None], y_pred_valid: Union[np.ndarray, None]) -> str:
    '''Generate a sweetviz report that can be logged into MLflow

    Args:
        df_train (pd.DataFrame): Training data
        y_pred_train (np.ndarray): Model predictions on training data
        y_col (str): Target column
        df_valid (pd.DataFrame): Validation data
        y_pred_valid (np.ndarray): Model predictions on validation data
    Returns:
        str: A HTML sweetviz report
    '''
    logger.info("Producing a sweetviz report ...")

    # SweetViz add too much logs to be use in production
    # https://github.com/fbdesignpro/sweetviz/issues/124
    # Deactivate tqdm and import sweetviz
    try:
        from tqdm import tqdm
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    except ImportError:
        pass

    try:
        import sweetviz
        import tempfile
    except ImportError:
        return None

    # Add predictions to our datasets
    # First, get new columns name
    y_col_name = f"pred_{y_col}"
    # Add predictions to train data
    df_train.loc[:, y_col_name] = y_pred_train
    # Add predictions to validation data
    if df_valid is not None and y_pred_valid is not None:
        df_valid.loc[:, y_col_name] = y_pred_valid

    # Target feature
    target = y_col
    # Prepare SweetViz datasets
    train_data = [df_train, "Training data"]
    valid_data = [df_valid, "Validation data"] if df_valid is not None else None

    # Get sweetviz report
    get_report_without_target = True
    report: sweetviz.DataframeReport
    try:
        report = sweetviz.compare(train_data, valid_data, target_feat=target, pairwise_analysis="off")
    except (KeyError, ValueError):
        logger.info("Can't produce a sweetviz report with 'target_feat'. Should not happen.")
        report = sweetviz.compare(train_data, valid_data, pairwise_analysis="off")

    # We need to call show_html in order to get _page_html
    # Hence, we'll do it on a temp file
    with tempfile.TemporaryDirectory() as tmp_dirname:
        tmp_file = os.path.join(tmp_dirname, "report.html")
        report.show_html(tmp_file, open_browser=False, layout="vertical")

    # Return html code as string
    return report._page_html


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='dataset_preprocess_P1.csv', help="Name of the training dataset (actually a path relative to {{package_name}}-data)")
    parser.add_argument('-y', '--y_col', required=True, help="Name of the model's target column - y")
    parser.add_argument('--excluded_cols', nargs='+', default=None, help="List of columns NOT to use as model's input")
    parser.add_argument('--filename_valid', default=None, help="Name of the validation dataset (actually a path relative to {{package_name}}-data)")
    parser.add_argument('-l', '--level_save', default='HIGH', help="Save level -> ['LOW', 'MEDIUM', 'HIGH']")
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files.")
    parser.add_argument('--encoding', default="{{default_encoding}}", help="Encoding to use with the .csv files.")
    parser.add_argument('--force_cpu', dest='on_cpu', action='store_true', help="Whether to force training on CPU (and not GPU)")
    parser.add_argument('--mlflow_experiment', help="Name of the current experiment. MLflow tracking is activated only if fulfilled.")
    parser.set_defaults(on_cpu=False)
    args = parser.parse_args()
    # Check forced CPU usage
    if args.on_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        logger.info("----------------------------")
        logger.info("CPU USAGE FORCED BY THE USER")
        logger.info("----------------------------")
    # Main
    main(filename=args.filename, y_col=args.y_col, excluded_cols=args.excluded_cols,
         filename_valid=args.filename_valid, level_save=args.level_save,
         sep=args.sep, encoding=args.encoding, mlflow_experiment=args.mlflow_experiment)
