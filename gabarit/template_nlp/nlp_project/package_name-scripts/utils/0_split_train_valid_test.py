#!/usr/bin/env python3

## Split a dataset into training / validation / test sets
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
# Ex: python 0_split_train_valid_test.py -f dataset.csv --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2


import os
import random
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Tuple

from {{package_name}} import utils
from {{package_name}}.models_training import utils_models


# Get logger
logger = logging.getLogger('{{package_name}}.0_split_train_valid_test')


def main(filename: str, split_type: str, perc_train: float, perc_valid: float, perc_test: float,
         x_col: Union[str, int, None] = None, y_col: Union[str, int, None] = None, overwrite_dataset: bool = False,
         seed: Union[int, None] = None, sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> None:
    '''Splits a dataset into training / validation / test sets

    Args:
        filename (str): Name of the dataset to be splitted (actually a path relative to {{package_name}}-data)
        split_type (str): Split type (random, stratified, hierarchical)
        perc_train (float): Train dataset fraction
        perc_valid (float): Validation dataset fraction
        perc_test (float): Test dataset fraction
    Kwargs:
        x_col (str | int): Column to be used with hierarchical split
        y_col (str | int): Column to be used with stratified split
        overwrite_dataset (bool): Whether to allow overwriting datasets
        seed (int): Random seed
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        ValueError: If split_type value is not a valid option (['random', 'stratified', 'hierarchical'])
        ValueError: If stratified split, but y_col is None
        ValueError: If hierarchical split, but x_col is None
        ValueError: If perc_train not in in ]0, 1[
        ValueError: If perc_valid not in in [0, 1[
        ValueError: If perc_test not in in [0, 1[
        ValueError: If abs(perc_train + perc_valid + perc_test - 1) > 0.0001
        ValueError: If perc_valid & perc_test are both null
        FileNotFoundError: If the file does not exist in {{package_name}}-data
        FileExistsError: If any save file already exists & not overwrite_dataset
    '''
    logger.info(f"Splits {filename} into training / validation / test sets ...")

    # Manage errors
    if split_type not in ['random', 'stratified', 'hierarchical']:
        raise ValueError(f"split_type value ({split_type}) is not a valid option (['random', 'stratified', 'hierarchical']).")
    if split_type == 'stratified' and y_col is None:
        raise ValueError("y_col must be set with 'stratified' split option")
    if split_type == 'hierarchical' and x_col is None:
        raise ValueError("x_col must be set with 'hierarchical' split option")
    if not 0 < perc_train < 1:
        raise ValueError("perc_train must be positive in ]0, 1[")
    if not 0 <= perc_valid < 1:
        raise ValueError("perc_valid must be positive in [0, 1[")
    if not 0 <= perc_test < 1:
        raise ValueError("perc_test must be positive in [0, 1[")
    if abs(perc_train + perc_valid + perc_test - 1) > 0.0001:
        raise ValueError(f"The sum of perc_train, perc_valid and perc_test should be equal to 1, not {perc_train + perc_valid + perc_test}")
    if perc_valid == 0. and perc_test == 0.:
        raise ValueError("perc_valid & perc_test can't both be null.")

    # Set seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Manage paths
    data_path = utils.get_data_path()
    file_path = os.path.join(data_path, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")

    # Manage save paths
    basename = Path(filename).stem
    train_path =  os.path.join(data_path, f"{basename}_train.csv")
    valid_path =  os.path.join(data_path, f"{basename}_valid.csv")
    test_path =  os.path.join(data_path, f"{basename}_test.csv")
    if os.path.exists(train_path) and not overwrite_dataset:
        raise FileExistsError(f"{train_path} already exists. This error can be bypassed with the argument --overwrite.")
    if os.path.exists(valid_path) and not overwrite_dataset and perc_valid != 0.:
        raise FileExistsError(f"{valid_path} already exists. This error can be bypassed with the argument --overwrite.")
    if os.path.exists(test_path) and not overwrite_dataset and perc_test != 0.:
        raise FileExistsError(f"{test_path} already exists. This error can be bypassed with the argument --overwrite.")

    # Get dataframe
    df, first_line = utils.read_csv(file_path, sep=sep, encoding=encoding, dtype=str)
    df = df.fillna('')  # Compulsory as some sentences might be empty after preprocessing, and reloaded as NAs
    nb_lines = df.shape[0]

    # Percentage
    logger.info(f'Train percentage : {perc_train * 100}%')
    logger.info(f'Validation percentage : {perc_valid * 100}%')
    logger.info(f'Test percentage : {perc_test * 100}%')

    # Split
    if split_type == 'random':
        df_train, df_valid, df_test = split_random(df, perc_train, perc_valid, perc_test, seed=seed)
    elif split_type == 'stratified':
        df_train, df_valid, df_test = split_stratified(df, y_col, perc_train, perc_valid, perc_test, seed=seed)
    else:  # split_type == 'hierarchical'
        df_train, df_valid, df_test = split_hierarchical(df, x_col, perc_train, perc_valid, perc_test, seed=seed)

    # Display info
    logger.info(f"Number of lines in the original dataset : {nb_lines}")
    logger.info(f"Number of lines in the train dataset : {df_train.shape[0]} ({df_train.shape[0] / nb_lines * 100} %)")
    if df_valid is not None:
        logger.info(f"Number of lines in the validation dataset : {df_valid.shape[0]} ({df_valid.shape[0] / nb_lines * 100} %)")
    else:
        logger.info("No validation dataset generated (perc. at 0.0% )")
    if df_test is not None:
        logger.info(f"Number of lines in the test dataset : {df_test.shape[0]} ({df_test.shape[0] / nb_lines * 100} %)")
    else:
        logger.info("No test dataset generated (perc. at 0.0% )")

    # Save
    utils.to_csv(df_train, train_path, first_line=first_line, sep='{{default_sep}}', encoding='{{default_encoding}}')
    if df_valid is not None:
        utils.to_csv(df_valid, valid_path, first_line=first_line, sep='{{default_sep}}', encoding='{{default_encoding}}')
    if df_test is not None:
        utils.to_csv(df_test, test_path, first_line=first_line, sep='{{default_sep}}', encoding='{{default_encoding}}')


def split_random(df: pd.DataFrame, perc_train: float, perc_valid: float, perc_test: float, seed: Union[int, None] = None)\
        -> Tuple[pd.DataFrame, Union[pd.DataFrame, None], Union[pd.DataFrame, None]]:
    '''Splits a dataset into training / validation / test sets - `Random` strategy

    Args:
        df (pd.DataFrame): Dataset to be splitted
        perc_train (float): Train dataset fraction
        perc_valid (float): Validation dataset fraction
        perc_test (float): Test dataset fraction
    Kwargs:
        seed (int): Seed to use
    Returns:
        pd.DataFrame: Train dataset
        pd.DataFrame: Validation dataset
        pd.DataFrame: Test dataset
    '''
    # Select train
    df_train = df.sample(frac=perc_train, random_state=seed)
    idx_train = df_train.index
    df = df[~df.index.isin(idx_train)]  # Remove train from main data
    # Select valid
    if perc_valid > 0.0:
        if perc_test == 0.0:
            df_valid = df  # no test, remaining data is validation
        else:
            # Update perc_valid to match remaining data
            perc_valid = perc_valid / (perc_test + perc_valid)
            df_valid = df.sample(frac=perc_valid, random_state=seed)
            idx_valid = df_valid.index
            df = df[~df.index.isin(idx_valid)]  # Remove valid from main data
    else:
        df_valid = None
    # Select test - remaining data
    df_test = df if perc_test > 0.0 else None
    # Return
    return df_train, df_valid, df_test


def split_stratified(df: pd.DataFrame, y_col: Union[str, int], perc_train: float, perc_valid: float, perc_test: float,
                     seed: Union[int, None] = None) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None], Union[pd.DataFrame, None]]:
    '''Splits a dataset into training / validation / test sets - `Stratified` strategy

    Args:
        df (pd.DataFrame): Dataset to be splitted
        y_col (str | int): Column to be used with stratified split
        perc_train (float): Train dataset fraction
        perc_valid (float): Validation dataset fraction
        perc_test (float): Test dataset fraction
    Kwargs:
        seed (int): Seed to use
    Returns:
        pd.DataFrame: Train dataset
        pd.DataFrame: Validation dataset
        pd.DataFrame: Test dataset
    '''
    # Select train
    df_train, df_valid_test = utils_models.stratified_split(df, y_col, test_size=perc_valid + perc_test, seed=seed)
    # Nominal case
    if perc_valid > 0 and perc_test > 0:
        # Update perc_test to match remaining data
        perc_test = perc_test / (perc_test + perc_valid)
        df_valid, df_test = utils_models.stratified_split(df_valid_test, y_col, test_size=perc_test, seed=seed)
        # Some indexes might be "lost" (small_classes), we retrieve them
        df_lost = df[~df.index.isin(df_train.index.append(df_test.index).append(df_valid.index))]
    # No validation set
    elif perc_valid == 0.0:
        df_valid = None
        df_test = df_valid_test
        # Some indexes might be "lost" (small_classes), we retrieve them
        df_lost = df[~df.index.isin(df_train.index.append(df_test.index))]
    # No test set
    else:
        df_valid = df_valid_test
        df_test = None
        # Some indexes might be "lost" (small_classes), we retrieve them
        df_lost = df[~df.index.isin(df_train.index.append(df_valid.index))]
    # We finally append "lost" indexes to the training set
    df_train = pd.concat([df_train, df_lost], sort=False)
    # Return
    return df_train, df_valid, df_test


def split_hierarchical(df: pd.DataFrame, x_col: Union[str, int], perc_train: float, perc_valid: float, perc_test: float,
                       seed: Union[int, None] = None) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None], Union[pd.DataFrame, None]]:
    '''Splits a dataset into training / validation / test sets - `Hierarchical` strategy

    Args:
        df (pd.DataFrame): Dataset to be splitted
        x_col (str | int): Column to be used with hierarchical split
        perc_train (float): Train dataset fraction
        perc_valid (float): Validation dataset fraction
        perc_test (float): Test dataset fraction
    Kwargs:
        seed (int): Seed to use
    Returns:
        pd.DataFrame: Train dataset
        pd.DataFrame: Validation dataset
        pd.DataFrame: Test dataset
    '''
    # Select train
    df_train, df_valid_test = utils_models.hierarchical_split(df, x_col, test_size=perc_valid + perc_test, seed=seed)
    # Nominal case
    if perc_valid > 0 and perc_test > 0:
        # Update perc_test to match remaining data
        perc_test = perc_test / (perc_test + perc_valid)
        df_valid, df_test = utils_models.hierarchical_split(df_valid_test, x_col, test_size=perc_test, seed=seed)
    # No validation set
    elif perc_valid == 0.0:
        df_valid = None
        df_test = df_valid_test
    # No test set
    else:
        df_valid = df_valid_test
        df_test = None
    # Return
    return df_train, df_valid, df_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='dataset.csv', help="Name of the dataset to be splitted (actually a path relative to {{package_name}}-data)")
    parser.add_argument('--split_type', default='random', help="Split type (random, stratified, hierarchical)")
    parser.add_argument('--perc_train', default=0.6, type=float, help="Train dataset fraction")
    parser.add_argument('--perc_valid', default=0.2, type=float, help="Validation dataset fraction")
    parser.add_argument('--perc_test', default=0.2, type=float, help="Test dataset fraction")
    parser.add_argument('--x_col', default=None, help="Column to be used with hierarchical split")
    parser.add_argument('--y_col', default=None, help="Column to be used with stratified split")
    parser.add_argument('--overwrite', dest='overwrite_dataset', action='store_true', help="Whether to allow overwriting datasets")
    parser.add_argument('--seed', default=None, type=int, help="Random seed")
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files")
    parser.add_argument('--encoding', default="{{default_encoding}}", help="Encoding to use with the .csv files")
    parser.set_defaults(overwrite_dataset=False)
    args = parser.parse_args()
    main(filename=args.filename, split_type=args.split_type, perc_train=args.perc_train,
         perc_valid=args.perc_valid, perc_test=args.perc_test, x_col=args.x_col, y_col=args.y_col,
         overwrite_dataset=args.overwrite_dataset, seed=args.seed, sep=args.sep, encoding=args.encoding)
