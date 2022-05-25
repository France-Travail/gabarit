#!/usr/bin/env python3

## Data preprocessing
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
# Ex: python 1_preprocess_data.py -f dataset_train.csv --target_cols Survived


import os
import gc
import time
import logging
import argparse
import pandas as pd
import dill as pickle
from pathlib import Path
from datetime import datetime
from typing import Union, List, Tuple

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess

# Get logger
logger = logging.getLogger('{{package_name}}.1_preprocess_data')


def main(filenames: List[str], preprocessing: Union[str, None], target_cols: List[Union[str, int]],
         sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> None:
    '''Preprocesses some datasets

    Idea:
        - For each file/dataset:
            - We retrieve a NEW pipeline
            - We `fit_transform` it on the dataset
            - We save the pipeline
            - We save the preprocessed dataset

    /!\ `validation` dataset MUST NOT be preprocessed here, as this is the fitting part /!\
    To apply an existing pipeline to another file, you should use 2_apply_existing_pipeline.py.

    Args:
        filenames (list<str>): Datasets filenames (actually paths relative to {{package_name}}-data)
        preprocessing (str): Preprocessing to be applied. All preprocessings are applied if None.
        target_cols (list<str|int>): List of target columns (i.e. Y).
            Warning: If you have several targets, your preprocessing must be compatible.
    Kwargs:
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        ValueError: if the preprocessing is not known
        FileNotFoundError: If a given file does not exist in {{package_name}}-data
    '''
    logger.info("Data preprocessing ...")

    ##############################################
    # Manage preprocessing pipelines
    ##############################################

    # Get preprocess dictionnary
    pipelines_dict = preprocess.get_pipelines_dict()

    # Get preprocessing(s) to be applied
    if preprocessing is not None:
        # Check presence in pipelines_dict
        if preprocessing not in pipelines_dict.keys():
            raise ValueError(f"The given preprocessing {preprocessing} is not known.")
        preprocessing_list = [preprocessing]
    # By default, we apply every preprocessings
    else:
        preprocessing_list = list(pipelines_dict.keys())


    ##############################################
    # Process each file, one by one
    ##############################################

    # Process each file, one by one
    for filename in filenames:

        # Get paths
        data_path = utils.get_data_path()
        dataset_path = os.path.join(data_path, filename)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"The file {dataset_path} does not exist")

        # For each preprocess: get the dataframe, `fit_transform` & save both file and fitted pipeline
        for preprocess_str in preprocessing_list:

            # 'no_preprocess' must be ignored
            if preprocess_str == 'no_preprocess':
                continue
            gc.collect()  # Fix some OOM in case of huge datasets being preprocessed
            logger.info(f'Applying preprocessing {preprocess_str} on filename {filename}')

            # Get pipeline -> we create a new pipeline for each file
            preprocess_pipeline = preprocess.get_pipeline(preprocess_str)
            # Get dataset
            df = pd.read_csv(dataset_path, sep=sep, encoding=encoding)
            # Split X, y
            y = df[target_cols]
            X = df.drop(target_cols, axis=1)
            # Apply pipeline
            new_X = preprocess_pipeline.fit_transform(X, y)
            # Try to retrieve new columns name (experimental)
            new_df = pd.DataFrame(new_X)
            new_df = preprocess.retrieve_columns_from_pipeline(new_df, preprocess_pipeline)
            # Reinject y -> check if a new column does not have the same name as one of the target column
            for col in target_cols:
                if col in new_df.columns:
                    new_df.rename(columns={col: f'new_{col}'}, inplace=True)
            new_df[target_cols] = y

            # Save pipeline
            # Idea: pipelines are saved in {{package_name}}-pipelines and should be reloaded to preprocess `valid` dataset.
            # Moreover, they will be saved alongside models during the training part. Hence, the models will be independent.
            pipeline_dir, pipeline_name = get_pipeline_dir(preprocess_str)
            pipeline_path = os.path.join(pipeline_dir, 'pipeline.pkl')
            # We save the pipeline as a dictionnary (pipeline object + preprocess name)
            pipeline_dict = {
                                'preprocess_pipeline': preprocess_pipeline,
                                'preprocess_str': preprocess_str,
                            }
            with open(pipeline_path, 'wb') as f:
                pickle.dump(pipeline_dict, f)
            # We also save a readable file alongside the pkl file (only informative)
            info_path = os.path.join(pipeline_dir, 'pipeline.info')
            with open(info_path, 'w', encoding='{{default_encoding}}') as f:
                f.write(f"'preprocess_str': {preprocess_str}\n")
                f.write(f"'preprocess_pipeline': {str(preprocess_pipeline)}")

            # Save preprocessed dataframe ({{default_encoding}}, '{{default_sep}}')
            # First line is a metadata with the name of the pipeline file
            basename = Path(filename).stem
            dataset_preprocessed_path = os.path.join(data_path, f'{basename}_{preprocess_str}.csv')
            utils.to_csv(new_df, dataset_preprocessed_path, first_line=f'#{pipeline_name}', sep=sep, encoding=encoding)


def get_pipeline_dir(preprocess_str: str) -> Tuple[str, str]:
    '''Retrieves a new directory to save a pipeline

    Args:
        preprocess_str (str): name of the preprocessing being used
    Returns:
        str: absolute path to the directory
        str: name of the pipeline (= name of the directory)
    '''
    pipelines_path = utils.get_pipelines_path()
    pipeline_name = datetime.now().strftime(f"{preprocess_str}_%Y_%m_%d-%H_%M_%S")
    pipeline_dir = os.path.join(pipelines_path, pipeline_name)
    if os.path.isdir(pipeline_dir):
        # Trick : if the directory already exists (two preprocess in the same second), we wait 1 second so that the 'date' changes...
        time.sleep(1)
        return get_pipeline_dir(preprocess_str)
    else:
        os.makedirs(pipeline_dir)
    return pipeline_dir, pipeline_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filenames', nargs='+', required=True, help='Datasets filenames (actually paths relative to {{package_name}}-data).')
    parser.add_argument('-p', '--preprocessing', default=None, help='Preprocessing to be applied. All preprocessings are applied if None.')
    parser.add_argument('--target_cols', nargs='+', required=True, help='Y columns.')
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files.")
    parser.add_argument('--encoding', default="{{default_encoding}}", help="Encoding to use with the .csv files.")
    args = parser.parse_args()
    main(filenames=args.filenames, preprocessing=args.preprocessing, target_cols=args.target_cols, sep=args.sep, encoding=args.encoding)
