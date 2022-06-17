#!/usr/bin/env python3

## Functions to remove white borders
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
import numpy as np
from PIL import Image
from math import floor
from typing import Union


# Get logger
logger = logging.getLogger(__name__)


def remove_white_borders(images: list, image_ratio_strategy: Union[str, None] = None, image_ratio: float = 0.75, with_rotation: bool = True) -> list:
    '''Removes white border
    Also change the image ratio and rotate (if wanted) along largest dim. (i.e. portrait mode)

    Args:
        images (list): Images to be processed
    Kwargs:
        image_ratio_strategy (str): wanted strategy to apply new image_ratio
            - None: no change in image ratio
            - 'fill': add white borders on the smallest dimension
            - 'stretch': stretch the images such that they have the wanted ratio
        image_ratio (float): Wanted final image ratio (unused if image_ratio_strategy is None)
        with_rotation (bool): If the images must be rotated along largest dim. (i.e. portrait mode)
    Raises:
        ValueError: If image_ratio_strategy value is not a valid option ([None, 'fill', 'stretch'])
    Returns:
        list: Images sans les marges blanches
    '''
    if image_ratio_strategy not in [None, 'fill', 'stretch']:
        raise ValueError(f"image ratio strategy (image_ratio_strategy) '{image_ratio_strategy}' is not a valid option ([None, 'fill', 'stretch'])")
    # Get 'True' white
    # TODO : to be improved !
    true_white = _rgb2gray(np.array([255, 255, 255]))
    # Process each image, one by one
    results = []
    for i, im in enumerate(tqdm.tqdm(images)):
        # Remove white borders
        pixels = _rgb2gray(np.array(im))
        # x : horizontal
        # y : vertical
        first_x = _get_first_x(pixels, true_white)  # Left
        first_y = _get_first_y(pixels, true_white)  # Upper
        last_x = _get_last_x(pixels, true_white)  # Right
        last_y = _get_last_y(pixels, true_white)  # Lower
        # If first_x -1 -> no 'non-white' pixel, do nothing
        if first_x == -1:
            continue
        else:
            # Crop image (left, upper, right, lower)
            im = im.crop((first_x, first_y, last_x + 1, last_y + 1))
            # Rotate image if wanted
            if with_rotation:
                im = rotate_image_largest_dim(im)
            # Manage new ratio strategy
            if image_ratio_strategy == 'fill':  # fill with white borders to get correct format
                im = fill_with_white(im, image_ratio)
            elif image_ratio_strategy == 'stretch':
                im = stretch_image(im, image_ratio)
            # If None, do nothign
        # Update list
        results.append(im)
    # Return
    return results


def _rgb2gray(rgb: np.ndarray) -> np.ndarray:
    '''Gets gray value from an rgb pixel

    Args:
        rgb (np.ndarray): Image to be processed
    Returns:
        np.ndarray: New image
    '''
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def _get_first_y(pixels: np.ndarray, true_white: np.ndarray) -> int:
    '''Gets first non-white y (vertical) pixel

    Args:
        pixels (np.ndarray): Image to be processed
        true_white (np.ndarray): "True" white (to be improved)
    Returns:
        int: Pixel's index
    '''
    mins = np.amin(pixels, axis=1)  # Get min pix per line
    # Look for first row with non-white pixel(s)
    for i, pix in enumerate(mins):
        if not pix == true_white:
            return i
    # No match ? Return -1
    return -1


def _get_first_x(pixels: np.ndarray, true_white: np.ndarray) -> int:
    '''Gets first non-white x (horizontal) pixel

    Args:
        pixels (np.ndarray): Image to be processed
        true_white (np.ndarray): "True" white (to be improved)
    Returns:
        int: Pixel's index
    '''
    new_pixels = np.transpose(pixels)
    return _get_first_y(new_pixels, true_white)


def _get_last_y(pixels: np.ndarray, true_white: np.ndarray) -> int:
    '''Gets last non-white y (vertical) pixel

    Args:
        pixels (np.ndarray): Image to be processed
        true_white (np.ndarray): "True" white (to be improved)
    Returns:
        int: Pixel's index
    '''
    new_pixels = np.flip(pixels, 0)
    tmp_y = _get_first_y(new_pixels, true_white)
    if tmp_y == -1:
        return -1
    else:
        return pixels.shape[0] - 1 - tmp_y


def _get_last_x(pixels: np.ndarray, true_white: np.ndarray) -> int:
    '''Gets last non-white x (horizontal) pixel

    Args:
        pixels (np.ndarray): Image to be processed
        true_white (np.ndarray): "True" white (to be improved)
    Returns:
        int: Pixel's index
    '''
    new_pixels = np.transpose(np.flip(pixels, 1))
    tmp_x = _get_first_y(new_pixels, true_white)
    if tmp_x == -1:
        return -1
    else:
        return pixels.shape[1] - 1 - tmp_x


def rotate_image_largest_dim(im: Image) -> Image:
    '''Rotates an image along largest dim. (i.e. portrait mode)

    Args:
        im (Image): Image to be processed
    Returns:
        Image: Rotated image
    '''
    orientation = _get_orientation(im)
    if orientation != 0:
        im = im.rotate(orientation, expand=True)
    return im


def fill_with_white(im: Image, image_ratio: float) -> Image:
    '''Fills an image with white such that it respects a wanted ratio

    Args:
        im (Image): Image to be processed
    Kwargs:
        image_ratio (float): Wanted image ratio
    Returns:
        Image: Transformed image
    '''
    width, height = im.size
    ratio = width / height
    if ratio > image_ratio:
        # Increase height
        wanted_height = round(width / image_ratio)
        new_size = (width, wanted_height)
        old_size = (width, height)
        # Set new image
        new_im = Image.new("RGB", new_size, (255, 255, 255))
        # Fill it with the old image, centered
        # (use floor to ensure the old image fits into the new one)
        x_pos = 0
        y_pos = floor((new_size[1] - old_size[1]) / 2)
        new_im.paste(im, (x_pos, y_pos))
    elif ratio < image_ratio:
        # Increase width
        wanted_width = round(height * image_ratio)
        new_size = (wanted_width, height)
        old_size = (width, height)
        # Set new image
        new_im = Image.new("RGB", new_size, (255, 255, 255))
        # Fill it with the old image, centered
        # (use floor to ensure the old image fits into the new one)
        x_pos = floor((new_size[0] - old_size[0]) / 2)
        y_pos = 0
        new_im.paste(im, (x_pos, y_pos))
    else:  # Already correct ratio
        new_im = im
    return new_im


def stretch_image(im: Image, image_ratio: float) -> Image:
    '''Stretch an image such that it respects a wanted ratio

    Args:
        im (Image): Image to be processed
    Kwargs:
        image_ratio (float): Wanted image ratio
    Returns:
        Image: Transformed image
    '''
    width, height = im.size
    ratio = width / height
    if ratio > image_ratio:
        # Increase height
        wanted_height = round(width / image_ratio)
        new_size = (width, wanted_height)
        new_im = im.resize(new_size)
    elif ratio < image_ratio:
        # Increase width
        wanted_width = round(height * image_ratio)
        new_size = (wanted_width, height)
        new_im = im.resize(new_size)
    else:  # Already correct ratio
        new_im = im
    return new_im


def _get_orientation(im: Image) -> int:
    '''Gets an image orientation based on largest dimension

    Args:
        im (Image): Image to be processed
    Returns:
        int: Image orientation
    '''
    if im.size[0] > im.size[1]:
        return 90
    else:
        return 0


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
