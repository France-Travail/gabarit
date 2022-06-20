#!/usr/bin/env python3

## Utils - Tools for training
# Copyright (C) <2018-2021>  <Agence Data Services, DSI Pôle Emploi>
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
# - normal_split -> Splits a DataFrame into train and test sets
# - stratified_split -> Splits a DataFrame into train and test sets - Stratified strategy
# - remove_small_classes -> Deletes under-represented classes
# - display_train_test_shape -> Displays the size of a train/test split
# - load_model -> Loads a model from a path
# - predict -> Gets predictions on a dataset
# - predict_with_proba -> Gets predictions with probabilities on a dataset


import os
import json
import math
import tempfile
import dill as pickle
import logging
import numpy as np
import pandas as pd
from PIL import Image
from typing import Union, Tuple, Any, List

from sklearn.model_selection import train_test_split

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess


# Get logger
logger = logging.getLogger(__name__)


def normal_split(df: pd.DataFrame, test_size: float = 0.25, seed: Union[int, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Splits a DataFrame into train and test sets

    Args:
        df (pd.DataFrame): Dataframe containing the data
    Kwargs:
        test_size (float): Proportion representing the size of the expected test set
        seed (int): random seed
    Raises:
        ValueError: If the object test_size is not between 0 and 1
    Returns:
        DataFrame: Train dataframe
        DataFrame: Test dataframe
    '''
    if not 0 <= test_size <= 1:
        raise ValueError('The object test_size must be between 0 and 1')

    # Normal split
    logger.info("Normal split")
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)

    # Display
    display_train_test_shape(df_train, df_test, df_shape=df.shape[0])

    # Return
    return df_train, df_test


def stratified_split(df: pd.DataFrame, col: Union[str, int], test_size: float = 0.25,
                     seed: Union[int, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Splits a DataFrame into train and test sets - Stratified strategy

    Args:
        df (pd.DataFrame): Dataframe containing the data
        col (str or int): column on which to do the stratified split
    Kwargs:
        test_size (float): Proportion representing the size of the expected test set
        seed (int): Random seed
    Raises:
        ValueError: If the object test_size is not between 0 and 1
    Returns:
        DataFrame: Train dataframe
        DataFrame: Test dataframe
    '''
    if not 0 <= test_size <= 1:
        raise ValueError('The object test_size must be between 0 and 1')

    # Stratified split
    logger.info("Stratified split")
    df = remove_small_classes(df, col, min_rows=math.ceil(1 / test_size))  # minimum lines number per category to split
    df_train, df_test = train_test_split(df, stratify=df[col], test_size=test_size, random_state=seed)

    # Display
    display_train_test_shape(df_train, df_test, df_shape=df.shape[0])

    # Return
    return df_train, df_test


def remove_small_classes(df: pd.DataFrame, col: Union[str, int], min_rows: int = 2) -> pd.DataFrame:
    '''Deletes the classes with small numbers of elements

    Args:
        df (pd.DataFrame): Dataframe containing the data
        col (str | int): Columns containing the classes
    Kwargs:
        min_rows (int): Minimal number of lines in the training set (default: 2)
    Raises:
        ValueError: If the object min_rows is not positive
    Returns:
        pd.DataFrame: New dataset
    '''
    if min_rows < 1:
        raise ValueError("The object min_rows must be positive")

    # Looking for classes with less than min_rows lines
    v_count = df[col].value_counts()
    classes_to_remove = list(v_count[v_count < min_rows].index.values)
    for cl in classes_to_remove:
        logger.warning(f"/!\\ /!\\ /!\\ Class {cl} has less than {min_rows} lines in the training set.")
        logger.warning("/!\\ /!\\ /!\\ This class is automatically removed from the dataset.")
    return df[~df[col].isin(classes_to_remove)]


def display_train_test_shape(df_train: pd.DataFrame, df_test: pd.DataFrame, df_shape: Union[int, None] = None) -> None:
    '''Displays the size of a train/test split

    Args:
        df_train (pd.DataFrame): Train dataset
        df_test (pd.DataFrame): Test dataset
    Kwargs:
        df_shape (int): Size of the initial dataset
    Raises:
        ValueError: If the object df_shape is not positive
    '''
    if df_shape is not None and df_shape < 1:
        raise ValueError("The object df_shape must be positive")

    # Process
    if df_shape is None:
        df_shape = df_train.shape[0] + df_test.shape[0]
    logger.info(f"There are {df_train.shape[0]} lines in the train dataset and {df_test.shape[0]} in the test dataset.")
    logger.info(f"{round(100 * df_train.shape[0] / df_shape, 2)}% of data are in the train set")
    logger.info(f"{round(100 * df_test.shape[0] / df_shape, 2)}% of data are in the test set")


def load_model(model_dir: str, is_path: bool = False) -> Tuple[Any, dict]:
    '''Loads a model from a path

    Args:
        model_dir (str): Name of the folder containing the model (e.g. model_autres_2019_11_07-13_43_19)
    Kwargs:
        is_path (bool): If folder path instead of name (permits to load model from elsewhere)
    Raises:
        FileNotFoundError: If the folder model_dir does not exist
    Returns:
        ?: Model
        dict: Model configurations
    '''

    # Find model path
    if not is_path:
        models_dir = utils.get_models_path()
        model_path = None
        for path, subdirs, files in os.walk(models_dir):
            for name in subdirs:
                if name == model_dir:
                    model_path = os.path.join(path, name)
        if model_path is None:
            raise FileNotFoundError(f"Can't find model {model_dir}")
    else:
        model_path = model_dir
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Can't find model {model_path} (considered as a path)")

    # Get configs
    configuration_path = os.path.join(model_path, 'configurations.json')
    with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
        configs = json.load(f)
    # Can't set int as keys in json, so need to cast it after reloading
    # dict_classes keys are always ints
    if 'dict_classes' in configs.keys() and configs['dict_classes'] is not None:
        configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}

    # Load model
    pkl_path = os.path.join(model_path, f"{configs['model_name']}.pkl")
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)

    # Change model_dir if diff
    if model_path != model.model_dir:
        model.model_dir = model_path
        configs['model_dir'] = model_path

    # Load specifics
    hdf5_path = os.path.join(model_path, 'best.hdf5')

    # Check for keras model
    if os.path.exists(hdf5_path):
        # If a specific reload function has been defined (e.g. faster RCNN), we use it
        if hasattr(model, 'reload_models_from_hdf5'):
            model.reload_models_from_hdf5(hdf5_path)
        else:
            model.model = model.reload_model(hdf5_path)

    # Display if GPU is being used
    model.display_if_gpu_activated()

    # Return model & configs
    return model, configs


def predict(data_input: Union[str, List[str], np.ndarray, pd.DataFrame], model, model_conf: dict,
            return_proba: bool = False, **kwargs) -> Union[str, List[str], np.ndarray]:
    '''Gets predictions of a model on images

    Args:
        data_input (str | list<str> | np.ndarray): New content to be predicted
            - str: abs. path to an image
            - list<str>: list of abs. path to an image
            - np.ndarray: an already loaded image
                Possibility to have several images if 4 dim (i.e (nb_images, width, height, channels))
            - pd.DataFrame: Dataframe with a column file_path (abs. paths to images)
        model (ModelClass): Model to use
        model_conf (dict): Model configurations
    Kwargs:
        return_proba (bool): If probabilities must be return instead
    Raises:
        NotImplementedError: If model is object detection task
        FileNotFoundError: If the input file does not exist (input type == str)
        FileNotFoundError: If one of the input files does not exist (input type == list)
        ValueError: If the input image format is not compatible (input type == np.ndarray)
        ValueError: If the input array is not compatible (input type == np.ndarray)
        ValueError: If the input DataFrame does not contains a 'file_path' column (input type == pd.DataFrame)
        ValueError: If the input type is not a valid type option
    Returns:
        str, List[str], np.ndarray: predictions or probabilities
            - If return_proba -> np.ndarray (shape depends on number of inputs)
            - Else str or list<str> (depends on number of inputs)
    '''
    # TODO
    # TODO
    # TODO: Make this works with object_detector !!!
    # TODO
    # TODO
    if model.model_type == 'object_detector':
        raise NotImplementedError("`predict` is not yet implemented for object detection task")

    ##############################################
    # Retrieve data - PIL format (list)
    ##############################################

    # Type 1: absolute path
    if isinstance(data_input, str):
        if not os.path.exists(data_input):
            raise FileNotFoundError(f"The file {data_input} does not exist")
        images = [Image.open(data_input)]

    # Type 2: list of absolute paths
    elif isinstance(data_input, list):
        if not all([os.path.exists(_) for _ in data_input]):
            raise FileNotFoundError("At least one of the input path does not exist")
        images = [Image.open(_) for _ in data_input]

    # Type 3: numpy array
    elif isinstance(data_input, np.ndarray):
        # If only one image (shape = 3), exapnd a 4th image
        if len(data_input.shape) == 3:
            data_input = np.expand_dims(data_input, 0)
        # Consider input as image list
        if len(data_input.shape) == 4:
            images = []
            for i in range(data_input.shape[0]):
                np_image = data_input[i]
                # RGB
                if np_image.shape[-1] == 3:
                    images.append(Image.fromarray(np_image, 'RGB'))
                elif np_image.shape[-1] == 4:
                    images.append(Image.fromarray(np_image, 'RGBA'))
                else:
                    raise ValueError(f"Input image format ({np_image.shape}) is not compatible")
        else:
            raise ValueError(f"Input array format ({type(data_input)}) is not valid")

    # Type 4: pd.DataFrame
    elif isinstance(data_input, pd.DataFrame):
        if 'file_path' not in data_input.columns:
            raise ValueError("The input DataFrame does not contains a 'file_path' column (mandatory)")
        file_paths = list(data_input['file_path'].values)
        if not all([os.path.exists(_) for _ in file_paths]):
            raise FileNotFoundError("At least one of the input path does not exist")
        images = [Image.open(_) for _ in file_paths]

    # No solution
    else:
        raise ValueError(f"Input type ({type(data_input)}) is not a valid type option.")

    ##############################################
    # Apply preprocessing
    ##############################################

    # Get preprocessor
    if 'preprocess_str' in model_conf.keys():
        preprocess_str = model_conf['preprocess_str']
    else:
        preprocess_str = "no_preprocess"
    preprocessor = preprocess.get_preprocessor(preprocess_str)

    # Preprocess
    images_preprocessed = preprocessor(images)

    ##############################################
    # Save all preprocessed images in a temporary directory
    ##############################################

    # We'll create a temporary folder to save preprocessed images
    with tempfile.TemporaryDirectory(dir=utils.get_data_path()) as tmp_folder:
        # Save images
        images_path = []
        for i, img in enumerate(images_preprocessed):
            img_path = os.path.join(tmp_folder, f"image_{i}.png")
            img.save(img_path, format='PNG')
            images_path.append(img_path)

        # Get predictions
        df = pd.DataFrame({'file_path': images_path})
        predictions, probas = model.predict_with_proba(df)

    # Getting out of the context, all temporary data is deleted

    ##############################################
    # Return result
    ##############################################

    # Return one element if only one input, else return all
    if return_proba:
        return probas[0] if len(probas) == 1 else probas
    else:
        predictions = model.inverse_transform(predictions)
        return predictions[0] if len(predictions) == 1 else predictions


def predict_with_proba(data_input: Union[str, List[str], np.ndarray, pd.DataFrame], model,
                       model_conf: dict) -> Tuple[Union[str, List[str]], Union[float, List[float]]]:
    '''Gets probabilities predictions of a model on a dataset

    Args:
        data_input (str | list<str> | np.ndarray): New content to be predicted
            - str: abs. path to an image
            - list<str>: list of abs. path to an image
            - np.ndarray: an already loaded image
                Possibility to have several images if 4 dim (i.e (nb_images, width, height, channels))
            - pd.DataFrame: Dataframe with a column file_path (abs. paths to images)
        model (ModelClass): Model to use
        model_conf (dict): Model configurations
    Raises:
        NotImplementedError: If model is object detection task
    Returns:
        str: prediction
        float: probability

        If several elements -> list
    '''
    if model.model_type == 'object_detector':
        raise NotImplementedError("`predict_with_proba` is not yet implemented for object detection task")

    # Get probas
    probas: np.ndarray = predict(data_input, model, model_conf, return_proba=True)

    # Manage cases with only one element
    if len(probas.shape) == 1:
        predictions = model.get_classes_from_proba(np.expand_dims(probas, 0))
        predictions = model.inverse_transform(predictions)[0]
        probas = max(probas)
    # Several elements
    else:
        predictions = model.get_classes_from_proba(probas)
        predictions = model.inverse_transform(predictions)
        probas = list(probas.max(axis=1))

    # Return
    return predictions, probas


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
