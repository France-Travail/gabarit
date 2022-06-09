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
# Ex: python 1_preprocess_data.py -f original_dataset.csv train.csv test.csv --encoding utf-8 --sep ; --input_col text


import os
import gc
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import Union, List

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess

# Get logger
logger = logging.getLogger('{{package_name}}.1_preprocess_data')


def main(filenames: List[str], preprocessing: Union[str, None], input_col: Union[str, int],
         sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> None:
    '''Preprocesses some datasets

    Args:
        filenames (list<str>): Datasets filenames (actually paths relative to {{package_name}}-data)
        preprocessing (str): Preprocessing to be applied. All preprocessings are applied if None.
        input_col (str | int): Column to be preprocessed
    Kwargs:
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        FileNotFoundError: If a given file does not exist in {{package_name}}-data
    '''
    logger.info("Data preprocessing ...")

    ##############################################
    # Manage preprocessing pipelines
    ##############################################

    # Get preprocess dictionnary
    preprocessors_dict = preprocess.get_preprocessors_dict()

    # Get preprocessing(s) to be applied
    if preprocessing is not None:
        # Check presence in preprocessors_dict
        if preprocessing not in preprocessors_dict.keys():
            raise ValueError(f"The given preprocessing {preprocessing} is not known.")
        preprocessing_list = [preprocessing]
    # By default, we apply every preprocessings
    else:
        preprocessing_list = list(preprocessors_dict.keys())


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

        # For each preprocess: get the dataframe, apply the preprocessing & save
        for preprocess_str in preprocessing_list:

            # 'no_preprocess' must be ignored
            if preprocess_str == 'no_preprocess':
                continue
            gc.collect()  # Fix some OOM in case of huge datasets being preprocessed
            logger.info(f'Applying preprocessing {preprocess_str} on filename {filename}')

            # Get preprocessor
            preprocessor = preprocess.get_preprocessor(preprocess_str)
            # Get dataset
            df = pd.read_csv(dataset_path, sep=sep, encoding=encoding, dtype=str).fillna('')
            # Preprocess input_col
            x_col = utils.get_new_column_name(list(df.columns), 'preprocessed_text')
            df[x_col] = preprocessor(df[input_col])

            # Save preprocessed dataframe ({{default_encoding}}, '{{default_sep}}')
            # First line is a metadata with the name of the preprocess
            basename = Path(filename).stem
            dataset_preprocessed_path = os.path.join(data_path, f'{basename}_{preprocess_str}.csv')
            utils.to_csv(df, dataset_preprocessed_path, first_line=f'#{preprocess_str}', sep='{{default_sep}}', encoding='{{default_encoding}}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filenames', nargs='+', required=True, help='Datasets filenames (actually paths relative to {{package_name}}-data).')
    parser.add_argument('-p', '--preprocessing', default=None, help='Preprocessing to be applied. All preprocessings are applied if None.')
    parser.add_argument('--input_col', default=None, help='Column to be preprocessed')
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files.")
    parser.add_argument('--encoding', default="{{default_encoding}}", help="Encoding to use with the .csv files.")
    args = parser.parse_args()
    main(filenames=args.filenames, preprocessing=args.preprocessing, input_col=args.input_col, sep=args.sep, encoding=args.encoding)
