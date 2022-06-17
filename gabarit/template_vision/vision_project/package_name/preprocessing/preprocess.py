#!/usr/bin/env python3

## Preprocessing functions
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


import tqdm
import logging
import functools
from PIL import Image
from io import BytesIO
from typing import Callable

from {{package_name}}.preprocessing import manage_white_borders


# Get logger
logger = logging.getLogger(__name__)

# TODO: manage .pdf files ?

# The following preprocessing are "simple" ones and they return "readable" images
# e.g. resizing, cast RGB, orientation ...
# For more specific preprocessing (scaling, brightness shift, zoom, ...), this is usually done
# inside the model (with object like ImageDataGenerator)


def get_preprocessors_dict() -> dict:
    '''Gets a dictionary of available preprocessing

    Returns:
        dict: Dictionary of preprocessing
    '''
    preprocessors_dict = {
        'no_preprocess': lambda x: x,  # - /!\ DO NOT DELETE -> necessary for compatibility /!\ -
        'preprocess_convert_rgb': preprocess_convert_rgb,  # Simple RGB converter
        'preprocess_docs': preprocess_docs,  # Example pipeline with documents (remove white borders, 3/4 ratio, resize 224x224)
    }
    return preprocessors_dict


def get_preprocessor(preprocess_str: str) -> Callable:
    '''Gets a preprocessing (function) from its name

    Args:
        preprocess_str (str): Name of the preprocess
    Raises:
        ValueError: If the name of the preprocess is not known
    Returns:
        Callable: Function to be used for the preprocessing
    '''
    # Process
    preprocessors_dict = get_preprocessors_dict()
    if preprocess_str not in preprocessors_dict.keys():
        raise ValueError(f"The preprocess {preprocess_str} is not known.")
    # Get preprocessor
    preprocessor = preprocessors_dict[preprocess_str]
    # Return
    return preprocessor


def apply_pipeline(images: list, pipeline: list) -> list:
    '''Applies a pipeline (i.e. list of transformations)

    Args:
        images (list): List of images to be transformed
        pipeline (list): List of transformation to be applied
    Returns:
        list: Preprocessed images
    '''
    # Process
    results = None
    for transformer in pipeline:
        if results is None:
            results = transformer(images)
        else:
            results = transformer(results)
    return results


def preprocess_convert_rgb(images: list) -> list:
    '''Applies a simple RGB conversion

    Args:
        images (list): List of images to be transformed
    Returns:
        list: Preprocessed images
    '''
    # Process
    pipeline = [convert_rgb]
    return apply_pipeline(images, pipeline=pipeline)


def preprocess_docs(images: list) -> list:
    '''Applies a list of usual transformations with scanned documents

    Args:
        images (list): List of images to be transformed
    Returns:
        list: Preprocessed images
    '''
    # Process
    pipeline = [convert_rgb, functools.partial(manage_white_borders.remove_white_borders, image_ratio_strategy='fill', image_ratio=0.75, with_rotation=True), resize]
    return apply_pipeline(images, pipeline=pipeline)


def convert_rgb(images: list) -> list:
    '''Converts a list of image into RGB images

    Args:
        images (list): List of images to be converted
    Returns:
        list: RGB images
    '''
    return [im.convert('RGB') for im in images]


def resize(images: list, width: int = 224, height: int = 224) -> list:
    '''Resizes images

    Args:
        images (list): List of images to be resized
    Kwargs:
        width (int): Wanted width
        height (int): Wanted height
    Raises:
        ValueError: If width < 1
        ValueError: If height < 1
    Returns:
        list: Resized images
    '''
    if width < 1:
        raise ValueError('Width must be strictly positive.')
    if height < 1:
        raise ValueError('Height must be strictly positive.')
    # Process images one by one
    results = []
    for i, im in enumerate(tqdm.tqdm(images)):
        results.append(im.resize((width, height)))
    return results


def jpeg_compression(images: list, quality: int = 75) -> list:
    '''Simulates a JPEG compression
    Might be useful for prediction if a model is trained with JPEG compressed images.

    Args:
        images (list): List of images to be compressed
    Kwargs:
        quality (int): Wanted quality
    Returns:
        list: Compressed images
    '''
    # Process images one by one
    results = []
    for i, im in enumerate(tqdm.tqdm(images)):
        out = BytesIO()
        im.save(out, format='JPEG', quality=quality)
        out.seek(0)
        tmp_im = Image.open(out)
        tmp_im.load()  # Cumpulsory to avoid "Too many open files" issue
        results.append(tmp_im)
    return results


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
