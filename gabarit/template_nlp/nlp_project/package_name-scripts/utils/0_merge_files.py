#!/usr/bin/env python3

## Just a script to merge some csv file
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
# Ex: python 0_merge_files.py -f dataset1.csv dataset2.csv dataset3.csv --sep ; --encoding utf-8 -o merged_dataset.csv -c col1 col2


import os
import logging
import argparse
import pandas as pd
from typing import List, Union

from {{package_name}} import utils

# Get logger
logger = logging.getLogger('{{package_name}}.0_merge_files')


def main(filenames: List[str], cols: Union[List[Union[str, int]], None] = None, output: str = 'dataset.csv', overwrite_dataset: bool = False,
         sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> None:
    '''Merges several datasets

    Args:
        filenames (list<str>): Datasets filenames (actually paths relative to {{package_name}}-data)
    Kwargs:
        cols (list<str|int>): List of columns to keep, default all
        output (str): Output filename
        overwrite_dataset (bool): Whether to allow overwriting datasets
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        FileNotFoundError: If a given file does not exist in {{package_name}}-data
        FileExistsError: If the output file already exists & not overwrite_dataset
        ValueError: If any input file has a metadata line (not supported)
    '''
    logger.info("Merging several files ...")

    # Get path
    data_path = utils.get_data_path()
    file_paths = [os.path.join(data_path, filename) for filename in filenames]
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")

    # Manage new file
    output_path = os.path.join(data_path, output)
    if os.path.isfile(output_path) and not overwrite_dataset:
        raise FileNotFoundError(f"The file {output_path} already exists.")

    # Init. dataframe
    df = pd.DataFrame(columns=cols)
    # Concat with all files
    for file_path in file_paths:
        # Check if first line starts with '#'
        with open(file_path, 'r', encoding=encoding) as f:
            first_line = f.readline()
        if first_line.startswith('#'):
            raise ValueError(f"Metadata line found for file {file_path} - not yet supported.")
        # Load data & concat (read everything as str to avoid some errors)
        df_tmp = pd.read_csv(file_path, sep=sep, encoding=encoding, dtype=str).fillna('')
        # TODO: manage error if cols not in dataset ?
        if cols is not None:
            df_tmp = df_tmp[cols]
        df = pd.concat([df, df_tmp]).reset_index(drop=True)

    # Display final size
    utils.display_shape(df)

    # Save
    df.to_csv(output_path, sep='{{default_sep}}', encoding='{{default_encoding}}', index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filenames', nargs='+', required=True, help="Datasets filenames (actually paths relative to {{package_name}}-data)")
    parser.add_argument('-c', '--cols', nargs='+', default=None, help="List of columns to keep")
    parser.add_argument('-o', '--output', default='dataset.csv', help="Output filename")
    parser.add_argument('--overwrite', dest='overwrite_dataset', action='store_true', help="Whether to allow overwriting datasets")
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files")
    parser.add_argument('--encoding', default="{{default_encoding}}", help="Encoding to use with the .csv files")
    parser.set_defaults(overwrite_dataset=False)
    args = parser.parse_args()
    main(filenames=args.filenames, cols=args.cols, output=args.output,
         overwrite_dataset=args.overwrite_dataset, sep=args.sep, encoding=args.encoding)
