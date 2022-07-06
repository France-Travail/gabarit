#!/usr/bin/env python3

# Apply a Machine Learning algorithm to obtain predictions
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
# Ex: python 3_predict.py  --filename dataset_test.csv --x_col preprocessed_text --model_dir model_tfidf_svm_2021_04_15-10_23_13 --y_col Survived


import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Union, List, Tuple

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training import utils_models

# Get logger
logger = logging.getLogger('{{package_name}}.3_predict')


def main(filename: str, x_col: Union[str, int], model_dir: str, y_col: Union[List[Union[str, int]], None] = None,
         sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> None:
    '''Gets a model's predictions on a given dataset

    Args:
        filename (str): Name of the test dataset (actually a path relative to {{package_name}}-data)
            It must not be preprocessed, hence not have metadata in its first line
        x_col (str | int): Name of the model's input column - x
        model_dir (str): Name of the model to use (not a path, just the directory name)
    Kwargs:
        y_col (list<str|int>): Name of the model's target column(s) - y
            Only used to get some metrics if available
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        ValueError: If multi-labels and bad OHE format
    '''

    ##############################################
    # Loading model & data
    ##############################################

    # Load model
    logger.info("Loading model ...")
    model, model_conf = utils_models.load_model(model_dir=model_dir)

    # Load dataset & preprocess it
    logger.info("Loading and preprocessing test dataset ...")
    df, preprocess_col = load_dataset_test(filename, x_col, model_conf, sep=sep, encoding=encoding)


    ##############################################
    # Get predictions
    ##############################################

    # Get predictions
    logger.info("Getting predictions ...")
    y_pred = model.predict(df[preprocess_col], return_proba=False)
    predictions_col = utils.get_new_column_name(list(df.columns), 'predictions')
    df[predictions_col] = list(model.inverse_transform(np.array(y_pred)))


    ##############################################
    # Save results
    ##############################################

    # Remove preprocessing from dataset (we do not need it anymore)
    df = df[[col for col in df.columns if col != preprocess_col]]

    # Save result
    logger.info("Saving results ...")
    save_dir = os.path.join(utils.get_data_path(), 'predictions', Path(filename).stem, datetime.now().strftime("predictions_%Y_%m_%d-%H_%M_%S"))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_file = "predictions.csv"
    file_path = os.path.join(save_dir, save_file)
    df.to_csv(file_path, sep='{{default_sep}}', encoding='{{default_encoding}}', index=None)

    # Also save some info into a configs file
    conf_file = 'configurations.json'
    conf_path = os.path.join(save_dir, conf_file)
    conf = {
        'model_dir': model_dir,
        'preprocess_str': model_conf['preprocess_str'],
        'model_name': model_conf['model_name']
    }
    with open(conf_path, 'w', encoding='{{default_encoding}}') as f:
        json.dump(conf, f, indent=4)


    ##############################################
    # Get metrics
    ##############################################

    # Get metrics if y_col is not None
    if y_col is not None:

        ### INFO: target data must have a correct format (cf training file)
        if len(y_col) > 1:
            try:
                y_true = df[y_col].astype(int)  # Need to cast OHE var into integers
                for col in y_col:
                    assert sorted(y_true[col].unique()) == [0, 1]
            except Exception:
                raise ValueError("You provided several target columns, but at least one of them does not seem to be in a correct OHE format.")
        else:
            y_true = df[y_col[0]]
            # No need to cast target in string, already done by the data loader

        cols_to_add: List[pd.Series] = []  # You can add columns to save here
        series_to_add = [df[col] for col in cols_to_add]
        # Change model directory to save dir & get preds
        model.model_dir = save_dir
        model.get_and_save_metrics(y_true, y_pred, series_to_add=series_to_add, type_data='with_y_true')


def load_dataset_test(filename: str, x_col: Union[str, int], model_conf: dict,
                      sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> Tuple[pd.DataFrame, str]:
    '''Function to load a test dataset & preprocess it

    Args:
        filename (str): Name of the dataset to load (actually a path relative to {{package_name}}-data)
        x_col (str | int): Name of the model's input column - x
        model_conf (dict): Model configuration
    Kwargs:
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        FileNotFoundError: If the file does not exist in {{package_name}}-data
    Returns:
        pd.DataFrame: Loaded & preprocessed dataframe
        str: Column name of the preprocessed data
    '''
    logger.info(f"Loading a dataset ({filename})")

    # Get dataset
    data_path = utils.get_data_path()
    file_path = os.path.join(data_path, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")

    # Get dataset
    df = pd.read_csv(file_path, sep=sep, encoding=encoding, dtype=str).fillna('')

    # Get preprocessor from model conf
    preprocess_str = model_conf['preprocess_str']
    preprocessor = preprocess.get_preprocessor(preprocess_str)

    # Apply preprocessing
    preprocess_col = utils.get_new_column_name(list(df.columns), 'preprocessed_column')
    # We do not check for column presence in dataframe
    # TODO: add check ?
    logger.info("Preprocessing test data")
    df[preprocess_col] = preprocessor(df[x_col])

    # Return
    return df, preprocess_col


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='newdata.csv', help="Name of the test dataset (actually a path relative to {{package_name}}-data)")
    parser.add_argument('-x', '--x_col', required=True, help="Name of the model's input column - x")
    # model_X should be the model's directory name: e.g. model_tfidf_svm_2019_12_05-12_57_18
    parser.add_argument('-m', '--model_dir', required=True, help="Name of the model to use (not a path, just the directory name)")
    parser.add_argument('-y', '--y_col', nargs='+', default=None, help="Name of the model's target column(s) - y")
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files.")
    parser.add_argument('--encoding', default="{{default_encoding}}", help="Encoding to use with the .csv files.")
    parser.add_argument('--force_cpu', dest='on_cpu', action='store_true', help="Whether to force training on CPU (and not GPU)")
    parser.set_defaults(on_cpu=False)
    args = parser.parse_args()
    # Check forced CPU usage
    if args.on_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        logger.info("----------------------------")
        logger.info("CPU USAGE FORCED BY THE USER")
        logger.info("----------------------------")
    # Main
    main(filename=args.filename, x_col=args.x_col, model_dir=args.model_dir,
         y_col=args.y_col, sep=args.sep, encoding=args.encoding)
