#!/usr/bin/env python3

## Extract samples from data files
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
# Ex: python 0_create_samples.py -f dataset1.csv dataset2.csv --encoding utf-8 --sep ; -n 100


import os
import ntpath
import logging
import argparse
import pandas as pd
from typing import List

from {{package_name}} import utils

# Get logger
logger = logging.getLogger('{{package_name}}.0_create_samples')


def main(filenames: List[str], n_samples: int = 100, sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> None:
    '''Extracts data subsets from a list of files

    Args:
        filenames (list<str>): Datasets filenames (actually paths relative to {{package_name}}-data)
    Kwargs:
        n_samples (int): Number of samples to extract
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        FileNotFoundError: If a given file does not exist in {{package_name}}-data
    '''
    logger.info("Extracting samples ...")

    # Get data path
    data_path = utils.get_data_path()

    # Process file by file
    for filename in filenames:
        logger.info(f"Working on file {filename}")

        # Check path
        file_path = os.path.join(data_path, filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {filename} does not exist")

        # Get new file name/path
        base_file_name = '.'.join(ntpath.basename(file_path).split('.')[:-1])
        new_file_name = f"{base_file_name}_{n_samples}_samples.csv"
        new_path = os.path.join(data_path, new_file_name)
        # We do not trigger an error if the file exists
        if os.path.exists(new_path):
            logger.info(f"{new_path} already exists. Pass.")
            continue

        # Process
        logger.info(f"Processing {base_file_name}.")
        # Retrieve data & first line metadata
        df, first_line = utils.read_csv(file_path, sep=sep, encoding=encoding, dtype=str)
        # Get extract
        extract = df.sample(n=min(n_samples, df.shape[0]))
        # Save
        utils.to_csv(extract, new_path, first_line=first_line, sep='{{default_sep}}', encoding='{{default_encoding}}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filenames', nargs='+', required=True, help="Datasets filenames (actually paths relative to {{package_name}}-data)")
    parser.add_argument('-n', '--n_samples', type=int, default=100, help="Number of samples to extract")
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files")
    parser.add_argument('--encoding', default="{{default_encoding}}", help="Encoding to use with the .csv files")
    args = parser.parse_args()
    main(filenames=args.filenames, n_samples=args.n_samples, sep=args.sep, encoding=args.encoding)
