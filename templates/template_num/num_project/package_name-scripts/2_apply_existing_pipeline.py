#!/usr/bin/env python3

## Data preprocessing - Applying an already fitted pipeline to other datasets (i.e. `validation` dataset)
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
# Ex: python 2_apply_existing_pipeline.py -f dataset_valid.csv -p preprocess_P1_2021_04_09-14_34_48 --target_cols Survived


import os
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import Union, List

from {{package_name}} import utils
from {{package_name}}.models_training import utils_models
from {{package_name}}.preprocessing import preprocess

# Get logger
logger = logging.getLogger('{{package_name}}.2_apply_existing_pipeline')


def main(filenames: List[str], pipeline: str, target_cols: List[Union[str, int]],
         sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> None:
    '''Applies an already fitted pipeline to some datasets - usually `validation` datasets

    Args:
        filenames (list<str>): Datasets filenames (actually paths relative to {{package_name}}-data)
        pipeline (str): Already fitted pipeline to apply (relative to {{package_name}}-pipelines)
        target_cols (list<str|int>): List of target columns (i.e. Y).
    Kwargs:
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        FileNotFoundError: If a given file does not exist in {{package_name}}-data
    '''
    logger.info("Data preprocessing - Applying already fitted pipeline ...")

    # Get pipeline
    preprocess_pipeline, preprocess_str = utils_models.load_pipeline(pipeline)

    # Apply this pipeline to each file, one by one
    for filename in filenames:

        # Get paths
        data_path = utils.get_data_path()
        dataset_path = os.path.join(data_path, filename)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"The file {dataset_path} does not exist")

        # Get dataset
        df = pd.read_csv(dataset_path, sep=sep, encoding=encoding)
        # Split X, y
        y = df[target_cols]
        X = df.drop(target_cols, axis=1)
        # Apply pipeline
        new_df = utils_models.apply_pipeline(X, preprocess_pipeline)
        # Reinject y -> check if a new column does not have the same name as one of the target column
        for col in target_cols:
            if col in new_df.columns:
                new_df.rename(columns={col: f'new_{col}'}, inplace=True)
        new_df[target_cols] = y

        # Save preprocessed dataframe ({{default_encoding}}, '{{default_sep}}')
        # First line is a metadata with the name of the pipeline file
        basename = Path(filename).stem
        dataset_preprocessed_path = os.path.join(data_path, f'{basename}_{preprocess_str}.csv')
        utils.to_csv(new_df, dataset_preprocessed_path, first_line=f'#{pipeline}', sep=sep, encoding=encoding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filenames', nargs='+', required=True, help='Datasets filenames (actually paths relative to {{package_name}}-data).')
    parser.add_argument('-p', '--pipeline', default=None, help='Already pipeline to apply (relative to {{package_name}}-pipelines).')
    parser.add_argument('--target_cols', nargs='+', required=True, help='Y columns.')
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files.")
    parser.add_argument('--encoding', default="{{default_encoding}}", help="Encoding to use with the .csv files.")
    args = parser.parse_args()
    main(filenames=args.filenames, pipeline=args.pipeline, target_cols=args.target_cols, sep=args.sep, encoding=args.encoding)
