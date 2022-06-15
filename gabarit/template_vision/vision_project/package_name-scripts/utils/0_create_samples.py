#!/usr/bin/env python3

## Extract samples from data directories
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
# Ex: python 0_create_samples.py -d dataset_v1 -n 100


import os
import random
import ntpath
import shutil
import logging
import argparse
from typing import List

import pandas as pd
from {{package_name}} import utils


# Get logger
logger = logging.getLogger('{{package_name}}.0_create_samples')


def main(directories: List[str], n_samples: int = 100, sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}'):
    '''Extracts data subsets from a list of directories

    Args:
        directories (list<str>): Datasets directories (actually a path relative to {{package_name}}-data)
    Kwargs:
        n_samples (int): Number of samples to extract
        sep (str): Separator to use with the metadata file - if exists
        encoding (str): Encoding to use with the metadata file - if exists
    Raises:
        FileNotFoundError: If a given directory does not exist in {{package_name}}-data
        NotADirectoryError: If a given path is not a directory
    '''
    logger.info("Extracting samples ...")

    # Get data path
    data_path = utils.get_data_path()

    # Process directory by directory
    for directory in directories:
        logger.info(f"Working on directory {directory}")

        # Check path
        directory_path = os.path.join(data_path, directory)
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"{directory_path} path does not exist'")
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"{directory_path} is not a valid directory")

        # Create new directory
        new_directory = os.path.join(data_path, f"{directory}_{n_samples}_samples")
        # We do not trigger an error if the directory exists
        if os.path.exists(new_directory):
            logger.info(f"{new_directory} already exists. Pass.")
            continue
        os.makedirs(new_directory)

        # load data
        logger.info("Loading data ...")
        path_list, classes_or_bboxes_list, _, task_type = utils.read_folder(directory_path, sep=sep, encoding=encoding)
        n_samples = min(n_samples, len(path_list))  # We keep everything if n_samples > dataset size

        # Get extract
        logger.info("Sampling ...")
        matches = list(zip(path_list, classes_or_bboxes_list))
        samples = random.sample(matches, n_samples)
        path_list, classes_or_bboxes_list = [_[0] for _ in samples], [_[1] for _ in samples]

        # Save
        logger.info("Saving ...")
        # Remove relative path
        new_path_list = [os.path.relpath(f, directory_path) for f in path_list]
        # Add new dir
        new_path_list = [os.path.join(new_directory, f) for f in new_path_list]
        # Copy
        for src, dst in zip(path_list, new_path_list):
            dst_dir_path = os.path.dirname(dst)
            if not os.path.exists(dst_dir_path):
                os.makedirs(dst_dir_path)
            shutil.copy(src, dst)

        # We do not forget to add the preprocess file (easy)
        preprocessing_path = os.path.join(directory_path, 'preprocess_pipeline.conf')
        if os.path.exists(preprocessing_path):
            new_preprocessing_path = os.path.join(new_directory, 'preprocess_pipeline.conf')
            shutil.copy(preprocessing_path, new_preprocessing_path)

        # We do not forget to add the metadata file (needs to be created)
        filenames_list = [os.path.relpath(f, new_directory) for f in new_path_list]
        if task_type == 'classification':
            metadata_path = os.path.join(directory_path, 'metadata.csv')
            if os.path.exists(metadata_path):
                new_metadata_path = os.path.join(new_directory, 'metadata.csv')
                metadata_df = utils.rebuild_metadata_classification(filenames_list, classes_or_bboxes_list)
                metadata_df.to_csv(new_metadata_path, sep=sep, encoding=encoding, index=None)
        else:  # Object detection
            new_metadata_path = os.path.join(new_directory, 'metadata_bboxes.csv')
            metadata_df = utils.rebuild_metadata_object_detection(filenames_list, classes_or_bboxes_list)
            metadata_df.to_csv(new_metadata_path, sep=sep, encoding=encoding, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directories', nargs='+', required=True, help='Datasets directories (actually a path relative to {{package_name}}-data)')
    parser.add_argument('-n', '--n_samples', type=int, default=100, help="Number of samples to extract")
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files")
    parser.add_argument('--encoding', default="{{default_encoding}}", help="Encoding to use with the .csv files")
    args = parser.parse_args()
    main(directories=args.directories, n_samples=args.n_samples, sep=args.sep, encoding=args.encoding)
