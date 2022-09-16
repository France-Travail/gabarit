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
# - read_folder -> Loads images and classes / bboxes from a directory of images
# - read_folder_object_detection -> Loads images and bboxes from a directory of images - object detection task
# - read_folder_classification -> Loads images and classes from a directory of images - classification task
# - rebuild_metadata_object_detection -> Rebuilds a metadata file from files names and associated bboxes - object detection task
# - rebuild_metadata_classification -> Rebuilds a metadata file from files names and associated classes - classification task
# - display_shape -> Displays the number of lines and columns of a table
# - get_new_column_name -> Gets a new column name from a list of existing ones & a wanted name
# - get_chunk_limits -> Gets chunk limits from a pandas series or dataframe
# - data_agnostic_str_to_list -> Decorator to transform a string into a list of one element.
# - trained_needed -> Decorator to ensure a model has already been trained
# - get_data_path -> Returns the path of the data folder
# - get_models_path -> Returns the path of the models folder
# - get_ressources_path -> Returns the path of the ressources folder
# - get_package_version -> Returns the current package version
# Classes :
# - HiddenPrints -> Hides all prints
# - DownloadProgressBar -> Displays a progress bar
# - NpEncoder -> JSON encoder to manage numpy objects


import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import pkg_resources
from tqdm import tqdm
from urllib import request
from typing import Tuple, Union, Callable, List, Any

# Get logger
logger = logging.getLogger(__name__)

DIR_PATH = None  # IMPORTANT : THIS VARIABLE MUST BE SET IN PRODUCTION TO POINT TO DATA AND MODELS PARENT FOLDER


def read_folder(folder_path: str, images_ext: tuple = ('.jpg', '.jpeg', '.png'),
                sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}',
                accept_no_metadata: bool = False) -> Tuple[list, list, str, str]:
    '''Loads images and classes / bboxes from a directory of images

    Args:
        folder_path (str): Directory with the images to be loaded - abs path
    Kwargs:
        images_ext (tuple): Accepted images extensions if automatic detection (i.e. no metadata file)
        sep (str): Separator of the metadata file - if exists
        encoding (str): Encoding of the metadata file - if exists
        accept_no_metadata (bool): If we allow no targets metadata (i.e. returns only file paths, useful for predictions)
    Returns:
        list: List of images path
        list: List of classes associated with images if classification task, bboxes if objet detection task
        str: Name of the prerprocessing pipeline used
        str: Task type ('classification' or 'object_detection')
    '''
    metadata_object_detection = os.path.join(folder_path, 'metadata_bboxes.csv')
    # Object detection
    if os.path.exists(metadata_object_detection):
        logger.info("Object detection task - Loading folder ...")
        path_list, bboxes_list, preprocess_str = read_folder_object_detection(folder_path, images_ext=images_ext,
                                                                              sep=sep, encoding=encoding,
                                                                              accept_no_metadata=accept_no_metadata)
        return path_list, bboxes_list, preprocess_str, 'object_detection'
    # Classifier
    else:
        logger.info("Classification task - Loading folder ...")
        path_list, classes_list, preprocess_str = read_folder_classification(folder_path, images_ext=images_ext,
                                                                             sep=sep, encoding=encoding,
                                                                             accept_no_metadata=accept_no_metadata)
        return path_list, classes_list, preprocess_str, 'classification'


def read_folder_object_detection(folder_path: str, images_ext: tuple = ('.jpg', '.jpeg', '.png'),
                                 sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}',
                                 accept_no_metadata: bool = False) -> Tuple[list, list, str]:
    '''Loads images and bboxes from a directory of images - object detection task

    Solution 1: usage of a metadata file (metadata_bboxes.csv)
    Solution 2: read images from root directory - no targets

    Args:
        folder_path (str): Directory with the images to be loaded - abs path
    Kwargs:
        images_ext (tuple): Accepted images extensions if automatic detection (i.e. no metadata file)
        sep (str): Separator of the metadata file - if exists
        encoding (str): Encoding of the metadata file - if exists
        accept_no_metadata (bool): If we allow no targets metadata (i.e. returns only file paths, useful for predictions)
    Raises:
        FileNotFoundError: If folder path does not exists
        NotADirectoryError: If the provided path is not a directory
        ValueError: If column 'filename' does not exists in the metadata file
        RuntimeError: If no loading solution found
    Returns:
        list: List of images path
        list: List of bboxes associated with images
        str: Name of the prerprocessing pipeline used
    '''
    logger.info(f"Loading folder {folder_path} ...")

    # Check path exists and it's a directory
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The path {folder_path} does not exist")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path} is not a valid directory")

    # We first check for a preprocessing file
    preprocess_file = os.path.join(folder_path, 'preprocess_pipeline.conf')
    if os.path.exists(preprocess_file):
        logger.info("Found a preprocessing file")
        with open(preprocess_file, 'r', encoding=encoding) as f:
            preprocess_str = f.readline()
    else:
        logger.info("Can't find a preprocessing file, backup on 'no_preprocess'")
        preprocess_str = 'no_preprocess'

    # Solution 1: we try to load the directory by reading a metadata file
    # This file must be named metadata_bboxes.csv and contain a column `filename`
    metadata_file = os.path.join(folder_path, 'metadata_bboxes.csv')
    if os.path.exists(metadata_file):
        logger.info("Found a metadata file")

        # Loading metadata file
        metadata_df = pd.read_csv(metadata_file, sep=sep, encoding=encoding)
        if 'filename' not in metadata_df.columns:
            raise ValueError("The metadata file must contain a column 'filename'")

        # Retrieving information (path & bboxes)
        filenames = list(metadata_df['filename'].unique())
        path_list = [os.path.join(folder_path, f) for f in filenames]
        # Try to read bboxes
        if all([_ in metadata_df.columns for _ in ['class', 'x1', 'x2', 'y1', 'y2']]):
            bboxes_list = []
            for filename in filenames:
                filtered_bboxes = metadata_df[metadata_df.filename == filename]
                tmp_bboxes_list = []
                for i, row in filtered_bboxes.iterrows():
                    tmp_bboxes_list.append({
                        'class': str(row['class']),  # We ensure all classes are strings
                        'x1': row['x1'],
                        'x2': row['x2'],
                        'y1': row['y1'],
                        'y2': row['y2'],
                    })
                bboxes_list.append(tmp_bboxes_list)
        else:
            logger.info("Can't retrieve bboxes")
            bboxes_list = None

    # Solution 2: if accept no metadata, we retrieve all images inside the root directory
    elif accept_no_metadata:
        folder_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        folder_list = [f for f in folder_list if f != 'preprocess_pipeline.conf']  # Do not consider preprocessing file
        folder_list = [f for f in folder_list if f.endswith(images_ext)]  # Keep only images
        path_list = [os.path.join(folder_path, f) for f in folder_list]  # Get abs paths
        bboxes_list = None  # No targets

    # No solution found, raise error
    else:
        raise RuntimeError(f"No loading solution found for folder ({folder_path})")

    # Return
    return path_list, bboxes_list, preprocess_str


def read_folder_classification(folder_path: str, images_ext: tuple = ('.jpg', '.jpeg', '.png'),
                               sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}',
                               accept_no_metadata: bool = False) -> Tuple[list, list, str]:
    '''Loads images and classes from a directory of images - classification task

    Solution 1: usage of a metadata file (metadata.csv)
    Solution 2: all images at root directory, and prefixed with class names (e.g. class_filename.ext)
    Solution 3: all images saved in class subdirectories
    Solution 4: read images from root directory - no targets

    Args:
        folder_path (str): Directory with the images to be loaded - abs path
    Kwargs:
        images_ext (tuple): Accepted images extensions if automatic detection (i.e. no metadata file)
        sep (str): Separator of the metadata file - if exists
        encoding (str): Encoding of the metadata file - if exists
        accept_no_metadata (bool): If we allow no targets metadata (i.e. returns only file paths, useful for predictions)
    Raises:
        FileNotFoundError: If folder path does not exists
        NotADirectoryError: If the provided path is not a directory
        ValueError: If column 'filename' does not exists in the metadata file
        ValueError: If column 'class' does not exists in the metadata file and accept_no_metadata is False
        RuntimeError: If no loading solution found
    Returns:
        list: List of images path
        list: List of classes associated with images
        str: Name of the prerprocessing pipeline used
    '''
    logger.info(f"Loading folder {folder_path} ...")

    # Check path exists and it's a directory
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The path {folder_path} does not exist")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path} is not a valid directory")

    # We first check for a preprocessing file
    preprocess_file = os.path.join(folder_path, 'preprocess_pipeline.conf')
    if os.path.exists(preprocess_file):
        logger.info("Found a preprocessing file")
        with open(preprocess_file, 'r', encoding=encoding) as f:
            preprocess_str = f.readline()
    else:
        logger.info("Can't find a preprocessing file, backup on 'no_preprocess'")
        preprocess_str = 'no_preprocess'

    # Solution 1: we try to load the directory by reading a metadata file
    # This file must be named metadata.csv and contain a column `filename`
    metadata_file = os.path.join(folder_path, 'metadata.csv')
    if os.path.exists(metadata_file):
        logger.info("Found a metadata file")

        # Loading metadata file
        metadata_df = pd.read_csv(metadata_file, sep=sep, encoding=encoding)
        if 'filename' not in metadata_df.columns:
            raise ValueError("The metadata file must contain a column 'filename'")

        # Retrieving information (path & classes)
        path_list = list(metadata_df['filename'].values)
        path_list = [os.path.join(folder_path, f) for f in path_list]
        if 'class' in metadata_df.columns:
            classes_list = [str(cl) for cl in metadata_df['class'].values]
        elif accept_no_metadata:
            logger.info("Can't retrieve classes (missing 'class' column in metadata file)")
            classes_list = None
        else:
            raise ValueError("The metadata file must contain a column 'class' with argument `accept_no_metadata` at False")

        # Return here
        return path_list, classes_list, preprocess_str

    # Solution 2: we check if all files are inside the root directory and if they are all prefixed (i.e. prefix_filename.ext)
    folder_list = os.listdir(folder_path)
    folder_list = [f for f in folder_list if f != 'preprocess_pipeline.conf']  # Do not consider preprocessing file
    # Check if all files are images
    if all([f.endswith(images_ext) for f in folder_list]):
        logger.info("Try to load images from root directory")
        path_list = [os.path.join(folder_path, f) for f in folder_list]

        # Check prefixes
        if all([len(f.split('_')) > 1 for f in folder_list]) and all([len(f.split('_')[0]) > 0 for f in folder_list]):
            classes_list = [f.split('_')[0] for f in folder_list]
        else:
            logger.info("Can't retrieve classes (files are not prefixed)")
            classes_list = None

        # Return here
        return path_list, classes_list, preprocess_str

    # Solution 3: check if images are saved in class subdirectories
    folders_elements = os.listdir(folder_path)
    folders_elements = [f for f in folders_elements if f != 'preprocess_pipeline.conf']  # Do not consider preprocessing file
    # Check if only subdirectories
    if all([os.path.isdir(os.path.join(folder_path, f)) for f in folders_elements]):
        # Check if each subdirectories contain images
        if all([all([f2.endswith(images_ext) for f2 in os.listdir(os.path.join(folder_path, f))]) for f in folders_elements]):
            logger.info("Try to load images from class subdirectories")

            # Retrieving information (path & classes)
            path_list = []
            classes_list = []
            for folder in folders_elements:
                tmp_path_list = [os.path.join(folder_path, folder, f) for f in os.listdir(os.path.join(folder_path, folder))]
                tmp_classes_list = [folder] * len(tmp_path_list)
                path_list += tmp_path_list
                classes_list += tmp_classes_list
            return path_list, classes_list, preprocess_str

    # Solution 4: if accept no metadata, we retrieve all images inside the root directory
    elif accept_no_metadata:
        folder_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        folder_list = [f for f in folder_list if f != 'preprocess_pipeline.conf']  # Do not consider preprocessing file
        folder_list = [f for f in folder_list if f.endswith(images_ext)]  # Keep only images
        path_list = [os.path.join(folder_path, f) for f in folder_list]  # Get abs paths
        return path_list, None, preprocess_str  # No targets

    # No more solution, raise error
    raise RuntimeError(f"No loading solution found for folder ({folder_path})")


def rebuild_metadata_object_detection(filenames_list: list, bboxes_list: list) -> pd.DataFrame:
    '''Rebuilds a metadata file from files names and associated bboxes - object detection task

    Args:
        filenames_list (list): List of files names (actually a path relative to files parent directory)
        bboxes_list (list): List of bboxes
    Raises:
        AssertionError: Both list must be of same length
    Returns:
        pd.DataFrame: The new metadata dataframe
    '''
    assert len(filenames_list) == len(bboxes_list), "Both list 'filenames_list' & 'bboxes_list' must be of same length"
    metadata_df = pd.DataFrame(columns=['filename', 'class', 'x1', 'x2', 'y1', 'y2'])
    for filename, bboxes in zip(filenames_list, bboxes_list):
        for bbox in bboxes:
            new_row = {'filename': filename, 'class': bbox['class'], 'x1': bbox['x1'], 'x2': bbox['x2'], 'y1': bbox['y1'], 'y2': bbox['y2']}
            metadata_df = metadata_df.append(new_row, ignore_index=True)
    return metadata_df


def rebuild_metadata_classification(filenames_list: list, classes_list: list) -> pd.DataFrame:
    '''Rebuilds a metadata file from files names and associated classes - classification task

    Args:
        filenames_list (list): List of files names (actually a path relative to files parent directory)
        classes_list (list): List of classes
    Raises:
        AssertionError: Both list must be of same length
    Returns:
        pd.DataFrame: The new metadata dataframe
    '''
    assert len(filenames_list) == len(classes_list), "Both list 'filenames_list' & 'classes_list' must be of same length"
    return pd.DataFrame({'filename': filenames_list, 'class': classes_list})


def display_shape(df: pd.DataFrame) -> None:
    '''Displays the number of line and of column of a table.

    Args:
        df (pd.DataFrame): Table to parse
    '''
    # Display
    logger.info(f"Number of lines : {df.shape[0]}. Number of columns : {df.shape[1]}.")


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


def data_agnostic_str_to_list(function: Callable) -> Callable:
    '''Decorator to transform a string into a list of one element.
    DO NOT CAST BACK TO ONE ELEMENT.

    Args:
        function (func): Function to decorate
    Returns:
        function: The decorated function
    '''
    # Get wrapper
    def wrapper(x, *args, **kwargs):
        '''Wrapper'''
        if type(x) == str:
            # Cast str into a single element list
            my_list = [x]
            # Call function
            results = function(my_list, *args, **kwargs)
        else:
            results = function(x, *args, **kwargs)
        # Return
        return results
    return wrapper


@data_agnostic_str_to_list
def download_url(urls: list, output_path: str) -> None:
    '''Downloads an object from a list of URLs.
    This function will try every URL until it find an available one.

    Args:
        urls (list): List of URL to try
        output_path (str): Where to save the downloaded object
    Raises:
        ConnectionError: If no URL is available
    '''
    # Start by creating output directory if does not exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Test each url
    is_downloaded = False
    for url in urls:
        if not is_downloaded:
            try:
                # From https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                    request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
                is_downloaded = True  # Download ok
            except Exception:
                logger.warning(f"Can't download from URL {url}.")
    if not is_downloaded:
        raise ConnectionError("Couldn't find a working URL")


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


# Class (contexte manager) to temporary mute all prints
# It helps with some annoying external libraries
class HiddenPrints:
    '''Hides all prints'''
    def __enter__(self) -> None:
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Class (contexte manager) to display a progress bar with downloads
# From https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
class DownloadProgressBar(tqdm):
    '''Displays a progress bar'''
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Any = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


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


# TODO: test trained_needed


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
