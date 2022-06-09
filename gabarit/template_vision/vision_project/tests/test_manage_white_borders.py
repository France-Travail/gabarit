#!/usr/bin/env python3
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

# Libs unittest
import unittest
from unittest.mock import Mock
from unittest.mock import patch

# Utils libs
import os
import numpy as np
import pandas as pd
from PIL import Image

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess, manage_white_borders

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class PreprocessTests(unittest.TestCase):
    '''Main class to test all functions in {{package_name}}.preprocessing.manage_white_borders'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_remove_white_borders(self):
        '''Test of the function manage_white_borders.remove_white_borders'''
        # We define some basic images
        any_color = [125, 204, 10]
        white_pixel = [255, 255, 255]
        pixels_1 = np.array([
            [white_pixel, white_pixel, white_pixel, white_pixel, white_pixel, white_pixel, white_pixel],
            [white_pixel, white_pixel, any_color, any_color, any_color, any_color, white_pixel],
            [white_pixel, white_pixel, any_color, any_color, any_color, any_color, white_pixel],
            [white_pixel, white_pixel, any_color, any_color, any_color, any_color, white_pixel],
            [white_pixel, white_pixel, white_pixel, white_pixel, white_pixel, white_pixel, white_pixel],
        ])
        image_1 = Image.fromarray(pixels_1.astype(np.uint8))
        pixels_2 = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        image_2 = Image.fromarray(pixels_2.astype(np.uint8))
        pixels_3 = np.array([
            [any_color, any_color, any_color, white_pixel, white_pixel],
            [any_color, any_color, any_color, white_pixel, white_pixel],
            [any_color, any_color, any_color, white_pixel, white_pixel],
            [any_color, any_color, any_color, white_pixel, white_pixel],
            [white_pixel, white_pixel, white_pixel, white_pixel, white_pixel],
            [white_pixel, white_pixel, white_pixel, white_pixel, white_pixel],
        ])
        image_3 = Image.fromarray(pixels_3.astype(np.uint8))

        # Nominal case
        wanted_pixels_1 = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        wanted_image_1 = Image.fromarray(wanted_pixels_1.astype(np.uint8))
        wanted_pixels_2 = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        wanted_image_2 = Image.fromarray(wanted_pixels_2.astype(np.uint8))
        wanted_pixels_3 = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        wanted_image_3 = Image.fromarray(wanted_pixels_3.astype(np.uint8))
        res1, res2, res3 = manage_white_borders.remove_white_borders(images=[image_1, image_2, image_3],
                                                                     image_ratio_strategy=None,
                                                                     image_ratio=3/4, with_rotation=True)
        self.assertEqual(res1, wanted_image_1)
        self.assertEqual(res2, wanted_image_2)
        self.assertEqual(res3, wanted_image_3)

        # Without rotation
        wanted_pixels_1 = np.array([
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
        ])
        wanted_image_1 = Image.fromarray(wanted_pixels_1.astype(np.uint8))
        wanted_pixels_2 = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        wanted_image_2 = Image.fromarray(wanted_pixels_2.astype(np.uint8))
        wanted_pixels_3 = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        wanted_image_3 = Image.fromarray(wanted_pixels_3.astype(np.uint8))
        res1, res2, res3 = manage_white_borders.remove_white_borders(images=[image_1, image_2, image_3],
                                                                     image_ratio_strategy=None,
                                                                     image_ratio=3/4, with_rotation=False)
        self.assertEqual(res1, wanted_image_1)
        self.assertEqual(res2, wanted_image_2)
        self.assertEqual(res3, wanted_image_3)

        # Strategy fill - we test with ratio 1. because it is simpler to find suitable tests
        wanted_pixels_1 = np.array([
            [any_color, any_color, any_color, white_pixel],
            [any_color, any_color, any_color, white_pixel],
            [any_color, any_color, any_color, white_pixel],
            [any_color, any_color, any_color, white_pixel],
        ])
        wanted_image_1 = Image.fromarray(wanted_pixels_1.astype(np.uint8))
        wanted_pixels_2 = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        wanted_image_2 = Image.fromarray(wanted_pixels_2.astype(np.uint8))
        wanted_pixels_3 = np.array([
            [any_color, any_color, any_color, white_pixel],
            [any_color, any_color, any_color, white_pixel],
            [any_color, any_color, any_color, white_pixel],
            [any_color, any_color, any_color, white_pixel],
        ])
        wanted_image_3 = Image.fromarray(wanted_pixels_3.astype(np.uint8))
        res1, res2, res3 = manage_white_borders.remove_white_borders(images=[image_1, image_2, image_3],
                                                                     image_ratio_strategy='fill',
                                                                     image_ratio=1., with_rotation=True)
        self.assertEqual(res1, wanted_image_1)
        self.assertEqual(res2, wanted_image_2)
        self.assertEqual(res3, wanted_image_3)

        # Strategy stretch - we test with ratio 1. because it is simpler to find suitable tests & rotation False
        wanted_pixels_1 = np.array([
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
        ])
        wanted_image_1 = Image.fromarray(wanted_pixels_1.astype(np.uint8))
        wanted_pixels_2 = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        wanted_image_2 = Image.fromarray(wanted_pixels_2.astype(np.uint8))
        wanted_pixels_3 = np.array([
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
        ])
        wanted_image_3 = Image.fromarray(wanted_pixels_3.astype(np.uint8))
        res1, res2, res3 = manage_white_borders.remove_white_borders(images=[image_1, image_2, image_3],
                                                                     image_ratio_strategy='stretch',
                                                                     image_ratio=1., with_rotation=False)
        self.assertEqual(res1, wanted_image_1)
        self.assertEqual(res2, wanted_image_2)
        self.assertEqual(res3, wanted_image_3)

        # Manage errors
        with self.assertRaises(ValueError):
             manage_white_borders.remove_white_borders(images=[image_1, image_2, image_3],
                                                       image_ratio_strategy='toto',
                                                       image_ratio=1., with_rotation=True)

    def test02_rgb2gray(self):
        '''Test of the function manage_white_borders._rgb2gray'''
        # We get an image for the tests
        current_dir = os.getcwd()
        im_path = os.path.join(current_dir, 'test_data', '{{package_name}}-data', 'apple.jpg')
        image = Image.open(im_path)
        image.load()

        # First convert as RGB
        image = preprocess.convert_rgb([image])[0]

        # Nominal case
        gray_pixels = manage_white_borders._rgb2gray(np.array(image))
        self.assertEqual(type(gray_pixels), np.ndarray)
        self.assertEqual(gray_pixels.shape[0], np.array(image).shape[0])
        self.assertEqual(gray_pixels.shape[1], np.array(image).shape[1])

    def test03_get_first_y(self):
        '''Test of the function manage_white_borders._get_first_y'''
        true_white = manage_white_borders._rgb2gray(np.array([255,255,255]))

        # Nominal case
        pixels = np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [10, 30, 50, true_white],
            [true_white, true_white, 30, true_white],
            [true_white, true_white, true_white, true_white]
        ])
        first_y = manage_white_borders._get_first_y(pixels, true_white)
        self.assertEqual(first_y, 2)

        # Nominal case - 2
        pixels = np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, 10, true_white],
            [true_white, true_white, true_white, true_white]
        ])
        first_y = manage_white_borders._get_first_y(pixels, true_white)
        self.assertEqual(first_y, 3)

        # Nominal case - 3
        pixels = np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [10, true_white, true_white, true_white]
        ])
        first_y = manage_white_borders._get_first_y(pixels, true_white)
        self.assertEqual(first_y, 4)

        # Full white
        full_white_pixels =  np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
        ])
        first_y = manage_white_borders._get_first_y(full_white_pixels, true_white)
        self.assertEqual(first_y, -1)

    def test04_get_first_x(self):
        '''Test of the function manage_white_borders._get_first_y'''
        true_white = manage_white_borders._rgb2gray(np.array([255,255,255]))

        # Nominal case
        pixels = np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [10, 30, 50, true_white],
            [true_white, true_white, 30, true_white],
            [true_white, true_white, true_white, true_white]
        ])
        first_x = manage_white_borders._get_first_x(pixels, true_white)
        self.assertEqual(first_x, 0)

        # Nominal case - 2
        pixels = np.array([
            [true_white, true_white, 10, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white]
        ])
        first_x = manage_white_borders._get_first_x(pixels, true_white)
        self.assertEqual(first_x, 2)

        # Nominal case - 3
        pixels = np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, 27]
        ])
        first_x = manage_white_borders._get_first_x(pixels, true_white)
        self.assertEqual(first_x, 3)

        # Full white
        full_white_pixels =  np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
        ])
        first_x = manage_white_borders._get_first_x(full_white_pixels, true_white)
        self.assertEqual(first_x, -1)

    def test05_get_last_y(self):
        '''Test of the function manage_white_borders._get_last_y'''
        true_white = manage_white_borders._rgb2gray(np.array([255,255,255]))

        # Nominal case
        pixels = np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [10, 30, 50, true_white],
            [true_white, true_white, 30, true_white],
            [true_white, true_white, true_white, true_white]
        ])
        last_y = manage_white_borders._get_last_y(pixels, true_white)
        self.assertEqual(last_y, 3)

        # Nominal case - 2
        pixels = np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, 10, true_white]
        ])
        last_y = manage_white_borders._get_last_y(pixels, true_white)
        self.assertEqual(last_y, 4)

        # Nominal case - 3
        pixels = np.array([
            [true_white, true_white, 10, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white]
        ])
        last_y = manage_white_borders._get_last_y(pixels, true_white)
        self.assertEqual(last_y, 0)

        # Full white
        full_white_pixels =  np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
        ])
        last_y = manage_white_borders._get_last_y(full_white_pixels, true_white)
        self.assertEqual(last_y, -1)

    def test05_get_last_x(self):
        '''Test of the function manage_white_borders._get_last_x'''
        true_white = manage_white_borders._rgb2gray(np.array([255,255,255]))

        # Nominal case
        pixels = np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [10, 30, 50, true_white],
            [true_white, true_white, 30, true_white],
            [true_white, true_white, true_white, true_white]
        ])
        last_x = manage_white_borders._get_last_x(pixels, true_white)
        self.assertEqual(last_x, 2)

        # Nominal case - 2
        pixels = np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [10, true_white, true_white, true_white]
        ])
        last_x = manage_white_borders._get_last_x(pixels, true_white)
        self.assertEqual(last_x, 0)

        # Nominal case - 3
        pixels = np.array([
            [true_white, true_white, true_white, 10],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white]
        ])
        last_x = manage_white_borders._get_last_x(pixels, true_white)
        self.assertEqual(last_x, 3)

        # Full white
        full_white_pixels =  np.array([
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
            [true_white, true_white, true_white, true_white],
        ])
        last_x = manage_white_borders._get_last_x(full_white_pixels, true_white)
        self.assertEqual(last_x, -1)

    def test06_rotate_image_largest_dim(self):
        '''Test of the function manage_white_borders.rotate_image_largest_dim'''

        # Nominal case - no change
        any_color = [125, 204, 10]
        pixels = np.array([
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
        ])
        image = Image.fromarray(pixels.astype(np.uint8))
        new_image = manage_white_borders.rotate_image_largest_dim(image)
        self.assertEqual(new_image, image)

        # Nominal case - rotate 90
        any_color_2 = [60, 89, 10]
        pixels = np.array([
            [any_color, any_color_2, any_color_2, any_color, any_color],
            [any_color, any_color, any_color_2, any_color, any_color],
            [any_color, any_color, any_color, any_color_2, any_color],
            [any_color, any_color, any_color, any_color, any_color],
        ])
        target_pixels = np.array([
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color_2, any_color],
            [any_color_2, any_color_2, any_color, any_color],
            [any_color_2, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
        ])
        image = Image.fromarray(pixels.astype(np.uint8))
        target_image = Image.fromarray(target_pixels.astype(np.uint8))
        new_image = manage_white_borders.rotate_image_largest_dim(image)
        self.assertEqual(new_image, target_image)

    def test07_fill_with_white(self):
        '''Test of the function manage_white_borders.fill_with_white'''

        # We define an initial image
        any_color = [125, 204, 10]
        pixels = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        image = Image.fromarray(pixels.astype(np.uint8))

        # Nominal case - ratio 1.
        new_image = manage_white_borders.fill_with_white(image, 1.)
        self.assertEqual(new_image, image)

        # Nominal case - ratio 3/5
        white_pixel = [255, 255, 255]
        target_pixels = np.array([
            [white_pixel, white_pixel, white_pixel],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [white_pixel, white_pixel, white_pixel],
        ])
        target_image = Image.fromarray(target_pixels.astype(np.uint8))
        new_image = manage_white_borders.fill_with_white(image, 3/5)
        self.assertEqual(new_image, target_image)

        # Nominal case - ratio 5/3
        target_pixels = np.array([
            [white_pixel, any_color, any_color, any_color, white_pixel],
            [white_pixel, any_color, any_color, any_color, white_pixel],
            [white_pixel, any_color, any_color, any_color, white_pixel],
        ])
        target_image = Image.fromarray(target_pixels.astype(np.uint8))
        new_image = manage_white_borders.fill_with_white(image, 5/3)
        self.assertEqual(new_image, target_image)

    def test08_stretch_image(self):
        '''Test of the function manage_white_borders.stretch_image'''

        # We define an initial image
        any_color = [125, 204, 10]
        pixels = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        image = Image.fromarray(pixels.astype(np.uint8))

        # Nominal case - ratio 1.
        new_image = manage_white_borders.stretch_image(image, 1.)
        self.assertEqual(new_image, image)

        # Nominal case - ratio 3/5
        target_pixels = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        target_image = Image.fromarray(target_pixels.astype(np.uint8))
        new_image = manage_white_borders.stretch_image(image, 3/5)
        self.assertEqual(new_image, target_image)

        # Nominal case - ratio 5/3
        target_pixels = np.array([
            [any_color, any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color, any_color],
        ])
        target_image = Image.fromarray(target_pixels.astype(np.uint8))
        new_image = manage_white_borders.stretch_image(image, 5/3)
        self.assertEqual(new_image, target_image)

    def test09_get_orientation(self):
        '''Test of the function manage_white_borders.stretch_image'''

        # We define some images
        any_color = [125, 204, 10]
        pixels1 = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        im1 = Image.fromarray(pixels1.astype(np.uint8))
        pixels2 = np.array([
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
            [any_color, any_color, any_color],
        ])
        im2 = Image.fromarray(pixels2.astype(np.uint8))
        pixels3 = np.array([
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
            [any_color, any_color, any_color, any_color],
        ])
        im3 = Image.fromarray(pixels3.astype(np.uint8))

        # Nominal case
        self.assertEqual(manage_white_borders._get_orientation(im1), 0)
        self.assertEqual(manage_white_borders._get_orientation(im2), 0)
        self.assertEqual(manage_white_borders._get_orientation(im3), 90)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
