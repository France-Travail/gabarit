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
# Ex: python 0_split_train_valid_test.py -d dataset_v1 --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2


import os
import random
import shutil
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Union, Tuple

from {{package_name}} import utils
from {{package_name}}.models_training import utils_models


# Get logger
logger = logging.getLogger('{{package_name}}.0_split_train_valid_test')


def main(directory: str, split_type: str, perc_train: float, perc_valid: float, perc_test: float,
         overwrite_dataset: bool = False, seed: Union[int, None] = None, sep: str = '{{default_sep}}',
         encoding: str = '{{default_encoding}}') -> None:
    '''Splits a dataset into training / validation / test sets

    Args:
        directory (str): Directory to be splitted (actually a path relative to {{package_name}}-data)
        split_type (str): Split type (random, stratified)
        perc_train (float): Train dataset fraction
        perc_valid (float): Validation dataset fraction
        perc_test (float): Test dataset fraction
    Kwargs:
        overwrite_dataset (bool): Whether to allow overwriting datasets
        seed (int): Random seed
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        ValueError: If split_type value is not a valid option (['random', 'stratified'])
        ValueError: If perc_train not in in ]0, 1[
        ValueError: If perc_valid not in in [0, 1[
        ValueError: If perc_test not in in [0, 1[
        ValueError: If abs(perc_train + perc_valid + perc_test - 1) > 0.0001
        ValueError: If perc_valid & perc_test are both null
        FileNotFoundError: If the directory does not exist in {{package_name}}-data
        NotADirectoryError: If the argument `directory` is not a directory
        FileExistsError: If any save directory already exists & not overwrite_dataset
        ValueError: If stratified split but object detection task identified
    '''
    logger.info(f"Splits {directory} into training / validation / test sets ...")

    # Manage errors
    if split_type not in ['random', 'stratified']:
        raise ValueError(f"split_type value ({split_type}) is not a valid option (['random', 'stratified']).")
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
    directory_path = os.path.join(data_path, directory)
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"{directory_path} path does not exist'")
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"{directory_path} is not a valid directory")

    # Manage save paths
    new_directory_train = os.path.join(data_path, f"{directory}_train")
    new_directory_valid = os.path.join(data_path, f"{directory}_valid")
    new_directory_test = os.path.join(data_path, f"{directory}_test")
    if os.path.exists(new_directory_train) and not overwrite_dataset:
        raise FileExistsError(f"{new_directory_train} already exists. This error can be bypassed with the argument --overwrite.")
    if os.path.exists(new_directory_valid) and not overwrite_dataset and perc_valid != 0.:
        raise FileExistsError(f"{new_directory_valid} already exists. This error can be bypassed with the argument --overwrite.")
    if os.path.exists(new_directory_test) and not overwrite_dataset and perc_test != 0.:
        raise FileExistsError(f"{new_directory_test} already exists. This error can be bypassed with the argument --overwrite.")
    # Create directories
    for path, perc in {new_directory_train: perc_train, new_directory_valid: perc_valid, new_directory_test: perc_test}.items():
        # If exists, delete it (we already checked for --overwrite)
        if os.path.isdir(path):
            shutil.rmtree(path)
        # Create dir
        if abs(perc) > 0.000001:
            os.makedirs(path)

    # Load data
    logger.info("Loading data ...")
    path_list, classes_or_bboxes_list, _, task_type = utils.read_folder(directory_path, sep=sep, encoding=encoding)
    df = pd.DataFrame({'path': path_list, 'class_or_bboxes': classes_or_bboxes_list})
    if task_type == 'object_detection' and split_type == 'stratified':
        raise ValueError("Stratified split is not supported with object detection tasks.")
    nb_images = df.shape[0]

    # Percentages
    logger.info(f'Train percentage : {perc_train * 100}%')
    logger.info(f'Validation percentage : {perc_valid * 100}%')
    logger.info(f'Test percentage : {perc_test * 100}%')

    # Split
    if split_type == 'random':
        df_train, df_valid, df_test = split_random(df, perc_train, perc_valid, perc_test, seed=seed)
    else:  # stratified
        df_train, df_valid, df_test = split_stratified(df, perc_train, perc_valid, perc_test, seed=seed)

    # Display info
    logger.info(f"Number of images in the original dataset : {nb_images}")
    logger.info(f"Number of images in the train dataset : {df_train.shape[0]} ({df_train.shape[0] / nb_images * 100} %)")
    if df_valid is not None:
        logger.info(f"Number of images in the validation dataset : {df_valid.shape[0]} ({df_valid.shape[0] / nb_images * 100} %)")
    else:
        logger.info("No validation dataset generated (perc. at 0.0% )")
    if df_test is not None:
        logger.info(f"Number of images in the test dataset : {df_test.shape[0]} ({df_test.shape[0] / nb_images * 100} %)")
    else:
        logger.info("No test dataset generated (perc. at 0.0% )")

    # Save
    _save(df_train, directory_path, new_directory_train, task_type, sep=sep, encoding=encoding)
    if df_valid is not None:
        _save(df_valid, directory_path, new_directory_valid, task_type, sep=sep, encoding=encoding)
    if df_test is not None:
        _save(df_test, directory_path, new_directory_test, task_type, sep=sep, encoding=encoding)


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


def split_stratified(df: pd.DataFrame, perc_train: float, perc_valid: float, perc_test: float,
                     seed: Union[int, None] = None) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None], Union[pd.DataFrame, None]]:
    '''Splits a dataset into training / validation / test sets - `Stratified` strategy

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
    df_train, df_valid_test = utils_models.stratified_split(df, 'class_or_bboxes', test_size=perc_valid + perc_test, seed=seed)
    # Nominal case
    if perc_valid > 0 and perc_test > 0:
        # Update perc_test to match remaining data
        perc_test = perc_test / (perc_test + perc_valid)
        df_valid, df_test = utils_models.stratified_split(df_valid_test, 'class_or_bboxes', test_size=perc_test, seed=seed)
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


def _save(df: pd.DataFrame, original_directory: str, new_directory: str, task_type: str,
          sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> None:
    '''Creates a new directory with selected images and new metadata file

    Args:
        df (pd.DataFrame): DataFrame with the selected images (path / classes / bboxes)
        original_directory (str): Original directory (abs path)
        new_directory (str): New directory (abs path) (target dir)
        task_type (str): Task associated with the data (classification ? object detection ?)
    Kwargs:
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    '''
    # Get path list & classes_list
    path_list = list(df['path'].values)
    classes_or_bboxes_list = list(df['class_or_bboxes'].values)
    # Remove relative path
    new_path_list = [os.path.relpath(f, original_directory) for f in path_list]
    # Add new dir
    new_path_list = [os.path.join(new_directory, f) for f in new_path_list]
    # Copy images
    for src, dst in zip(path_list, new_path_list):
        dst_dir_path = os.path.dirname(dst)
        if not os.path.exists(dst_dir_path):
            os.makedirs(dst_dir_path)
        shutil.copy(src, dst)

    # Generate the preprocessing file associated with the final directory
    preprocessing_path = os.path.join(original_directory, 'preprocess_pipeline.conf')
    if os.path.exists(preprocessing_path):
        new_preprocessing_path = os.path.join(new_directory, 'preprocess_pipeline.conf')
        shutil.copy(preprocessing_path, new_preprocessing_path)

    # Generate the metadata file when adapted
    filenames_list = [os.path.relpath(f, new_directory) for f in new_path_list]
    if task_type == 'classification':
        metadata_path = os.path.join(original_directory, 'metadata.csv')
        # Create only if needed
        if os.path.exists(metadata_path):
            new_metadata_path = os.path.join(new_directory, 'metadata.csv')
            metadata_df = utils.rebuild_metadata_classification(filenames_list, classes_or_bboxes_list)
            metadata_df.to_csv(new_metadata_path, sep=sep, encoding=encoding, index=None)
    else:
        # Metadata compulsory for object detection tasks
        new_metadata_path = os.path.join(new_directory, 'metadata_bboxes.csv')
        metadata_df = utils.rebuild_metadata_object_detection(filenames_list, classes_or_bboxes_list)
        metadata_df.to_csv(new_metadata_path, sep=sep, encoding=encoding, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', required=True, help='Directory to be splitted (actually a path relative to {{package_name}}-data)')
    parser.add_argument('--split_type', default='random', help="Split type (random, stratified)")
    parser.add_argument('--perc_train', default=0.6, type=float, help="Train dataset fraction")
    parser.add_argument('--perc_valid', default=0.2, type=float, help="Validation dataset fraction")
    parser.add_argument('--perc_test', default=0.2, type=float, help="Test dataset fraction")
    parser.add_argument('--overwrite', dest='overwrite_dataset', action='store_true', help="Whether to allow overwriting datasets")
    parser.add_argument('--seed', default=None, type=int, help="Random seed")
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files")
    parser.add_argument('--encoding', default="{{default_encoding}}", help="Encoding to use with the .csv files")
    parser.set_defaults(overwrite_dataset=False)
    args = parser.parse_args()
    main(directory=args.directory, split_type=args.split_type, perc_train=args.perc_train,
         perc_valid=args.perc_valid, perc_test=args.perc_test, overwrite_dataset=args.overwrite_dataset,
         sep=args.sep, encoding=args.encoding, seed=args.seed)
