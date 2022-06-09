#!/usr/bin/env python3

## Create an embedding matrix from .vec files (fasttext format)
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
# Ex: python 0_get_embedding_dict.py  --filename cc.fr.300.vec


import os
import pickle
import logging
import argparse
import numpy as np
import pandas as pd

from {{package_name}} import utils

# Get logger
logger = logging.getLogger('{{package_name}}.0_get_embedding_dict')


def main(filename: str, encoding: str = '{{default_encoding}}') -> None:
    '''Creates an embedding matrix from .vec files (fasttext format)

    Those files can be downloaded here (text files): https://fasttext.cc/docs/en/crawl-vectors.html

    Args:
        filename (str): A .vec file name (actually a path relative to {{package_name}}-data)
    Kwargs:
        encoding (str): Encoding to use with the .vec file
    Raises:
        FileNotFoundError: If the file does not exist in {{package_name}}-data
    '''
    logger.info("Creating an embedding matrix ...")

    # Manage path
    data_path = utils.get_data_path()
    embedding_path = os.path.join(data_path, filename)
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"The file {embedding_path} does not exist")

    # Get embedding indexes
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    embedding_indexes = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding=encoding))

    # Save
    file_path = os.path.join(data_path, '.'.join(filename.split('.')[:-1]) + '.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(embedding_indexes, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', required=True, help='A .vec file name (actually a path relative to {{package_name}}-data)')
    parser.add_argument('--encoding', default="utf-8", help="Encoding to use with the .vec file.")
    args = parser.parse_args()
    main(filename=args.filename, encoding=args.encoding)
