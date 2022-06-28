#!/usr/bin/env python3

## Utils - tools-functions
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
# Functions :
# - read_csv -> Reads a .csv and parses the first line
# - to_csv -> Writes a .csv and manages the first line
# - display_shape -> Displays the number of lines and columns of a table
# - get_new_column_name -> Gets a new column name from a list of existing ones & a wanted name
# - get_chunk_limits -> Gets chunk limits from a pandas series or dataframe
# - trained_needed -> Decorator to ensure a model has already been trained
# - get_data_path -> Returns the path of the data folder
# - get_models_path -> Returns the path of the models folder
# - get_pipelines_path -> Returns the path of the pipeline folder
# - get_ressources_path -> Returns the path of the ressources folder
# - get_package_version -> Returns the current package version
# - flatten -> Flattens a list of mixed elements (ie. some iterable, some not)
# Classes :
# - NpEncoder -> JSON encoder to manage numpy objects


import os
import uuid
import json
import logging
import numpy as np
import pandas as pd
import pkg_resources
from collections.abc import Iterable
from typing import Tuple, Union, Callable, Generator, List, Any

# Get logger
logger = logging.getLogger(__name__)

DIR_PATH = None  # IMPORTANT : THIS VARIABLE MUST BE SET IN PRODUCTION TO POINT TO DATA AND MODELS PARENT FOLDER


def read_csv(file_path: str, sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}', **kwargs) -> Tuple[pd.DataFrame, Union[str, None]]:
    '''Reads a .csv file and parses the first line.

    Args:
        file_path (str): Path to the .csv file containing the data
    Kwargs:
        sep (str): Separator of the data file
        encoding (str): Encoding of the data file
        kwargs: Pandas' kwargs
    Raises:
        FileNotFoundError: If the file_path object does not point to an existing file
    Returns:
        pd.DataFrame: Data
        str: First line of the .csv (None if not beginning with #) and with no line break
    '''
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")

    # We get the first line
    with open(file_path, 'r', encoding=encoding) as f:
        first_line = f.readline()
    # We check if the first line contains metadata
    has_metada = True if first_line.startswith('#') else False
    # We load the dataset
    if has_metada:
        df = pd.read_csv(file_path, sep=sep, encoding=encoding, skiprows=1, **kwargs)
    else:
        df = pd.read_csv(file_path, sep=sep, encoding=encoding, **kwargs)

    # If no metadata, return only the dataframe
    if not has_metada:
        return df, None
    # Else process the first_line
    else:
        # Deletion of the line break
        if first_line is not None and first_line.endswith('\n'):
            first_line = first_line[:-1]
        # Deletion of the return carriage
        if first_line is not None and first_line.endswith('\r'):
            first_line = first_line[:-1]
        # Return
        return df, first_line


def to_csv(df: pd.DataFrame, file_path: str, first_line: Union[str, None] = None, sep: str = '{{default_sep}}',
           encoding: str = '{{default_encoding}}', **kwargs) -> None:
    '''Writes a .csv and manages the first line.

    Args:
        df (pd.DataFrame): Data to write
        file_path (str): Path to the file to create
    Kwargs:
        first_line (str): First line to write (without line break which is done in this function)
        sep (str): Separator for the data file
        encoding (str): Encoding of the data file
        kwargs: pandas' kwargs
    '''
    # We get the first line
    with open(file_path, 'w', encoding=encoding) as f:
        if first_line is not None:
            f.write(first_line + '\n')  # We add the first line if metadata are present
        df.to_csv(f, sep=sep, encoding=encoding, index=None, **kwargs)


def display_shape(df: pd.DataFrame) -> None:
    '''Displays the number of line and of column of a table.

    Args:
        df (pd.DataFrame): Table to parse
    '''
    # Display
    logger.info(f"Number of lines : {df.shape[0]}. Number of columns : {df.shape[1]}.")


def get_new_column_name(column_list: list, wanted_name: str) -> str:
    '''Gets a new column name from a list of existing ones & a wanted name

    If the wanted name does not exists, return it.
    Otherwise get a new column prefixed by the wanted name.

    Args:
        column_list (list): List of existing columns
        wanted_name (str): Wanted name
    '''
    if wanted_name not in column_list:
        return wanted_name
    else:
        new_name = f'{wanted_name}_{str(uuid.uuid4())[:8]}'
        # It should not happen, but we still check if new_name is available (bad luck ?)
        return get_new_column_name(column_list, new_name)


def get_chunk_limits(x: Union[pd.DataFrame, pd.Series], chunksize: int = 10000) -> List[Tuple[int]]:
    '''Gets chunk limits from a pandas series or dataframe.

    Args:
        x (pd.Series or pd.DataFrame): Documents to consider
    Kwargs:
        chunksize (int): The chunk size
    Raises:
        ValueError: If the chunk size is negative
    Returns:
        list<tuple>: the chunk limits
    '''
    if chunksize < 0:
        raise ValueError('The object chunksize must not be negative.')
    # Processs
    if chunksize == 0 or chunksize >= x.shape[0]:
        chunks_limits = [(0, x.shape[0])]
    else:
        chunks_limits = [(i * chunksize, min((i + 1) * chunksize, x.shape[0]))
                         for i in range(1 + ((x.shape[0] - 1) // chunksize))]
    return chunks_limits  # type: ignore


def trained_needed(function: Callable) -> Callable:
    '''Decorator to ensure that a model has been trained.

    Args:
        function (func): Function to decorate
    Returns:
        function: The decorated function
    '''
    # Get wrapper
    def wrapper(self, *args, **kwargs):
        '''Wrapper'''
        if not self.trained:
            raise AttributeError(f"The function {function.__name__} can't be called as long as the model hasn't been fitted")
        else:
            return function(self, *args, **kwargs)
    return wrapper


def get_data_path() -> str:
    '''Returns the path to the data folder

    Returns:
        str: Path of the data folder
    '''
    if DIR_PATH is None:
        dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '{{package_name}}-data')
    else:
        dir_path = os.path.join(os.path.abspath(DIR_PATH), '{{package_name}}-data')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return os.path.abspath(dir_path)


def get_models_path() -> str:
    '''Returns the path to the models folder

    Returns:
        str: Path of the models folder
    '''
    if DIR_PATH is None:
        dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '{{package_name}}-models')
    else:
        dir_path = os.path.join(os.path.abspath(DIR_PATH), '{{package_name}}-models')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return os.path.abspath(dir_path)


def get_pipelines_path() -> str:
    '''Returns the path to the pipelines folder

    Returns:
        str: Path of the pipelines folder
    '''
    if DIR_PATH is None:
        dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '{{package_name}}-pipelines')
    else:
        dir_path = os.path.join(os.path.abspath(DIR_PATH), '{{package_name}}-pipelines')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return os.path.abspath(dir_path)


def get_ressources_path() -> str:
    '''Returns the path to the ressources folder

    Returns:
        str: Path of the ressources folder
    '''
    dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '{{package_name}}-ressources')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return os.path.abspath(dir_path)


def get_package_version() -> str:
    '''Returns the current version of the package

    Returns:
        str: version of the package
    '''
    version = pkg_resources.get_distribution('{{package_name}}').version
    return version


def flatten(my_list: Iterable) -> Generator:
    '''Flattens a list of mixed elements (ie. some iterable, some not)
    e.g. [[1, 2], 3, [4]] -> [1, 2, 3, 4]
    From : https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists

    Args:
        my_list (Iterable): List to consider
    Results:
        generator: Flattened list (generator format)
    '''
    for el in my_list:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


# JSON encoder to manage numpy objects
class NpEncoder(json.JSONEncoder):
    '''JSON encoder to manage numpy objects'''
    def default(self, obj) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
