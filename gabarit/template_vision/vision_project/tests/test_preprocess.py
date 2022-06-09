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
    '''Main class to test all functions in {{package_name}}.preprocessing.preprocess'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_get_preprocessors_dict(self):
        '''Test of the function preprocess.get_preprocessors_dict'''
        # We get the preprocesses and we test an image
        current_dir = os.getcwd()
        im_path = os.path.join(current_dir, 'test_data', '{{package_name}}-data', 'test_data_1', 'Birman_4.jpg')
        image = Image.open(im_path)
        image.load()

        # Nominal case
        preprocessors_dict = preprocess.get_preprocessors_dict()
        self.assertEqual(type(preprocessors_dict), dict)
        self.assertTrue('no_preprocess' in preprocessors_dict.keys())

        # We test each returned function
        for preprocessor in preprocessors_dict.values():
            transformed_images = preprocessor([image])
            self.assertEqual(type(transformed_images), list)
            self.assertTrue(len(transformed_images), 1)
            self.assertTrue(str(type(transformed_images[0])).startswith("<class 'PIL."))  # Can be different depending on the preprocess

    def test02_get_preprocessor(self):
        '''Test of the function preprocess.get_preprocessor'''
        # Valids to test
        # We take a preprocessing 'at random'
        preprocessor_str = list(preprocess.get_preprocessors_dict().keys())[0]
        preprocessor_val = list(preprocess.get_preprocessors_dict().values())[0]

        # Nominal case
        # To check if 2 functions are "equal": https://stackoverflow.com/questions/20059011/check-if-two-python-functions-are-equal
        preprocessor_res = preprocess.get_preprocessor(preprocessor_str)
        self.assertEqual(preprocessor_res.__code__.co_code, preprocessor_val.__code__.co_code)

        # Check the input(s) type(s)
        with self.assertRaises(ValueError):
            preprocess.get_preprocessor('NOT A VALID PREPROCESS')

    def test03_apply_pipeline(self):
        '''Test of the function preprocess.apply_pipeline'''
        # We get 2 images for the tests
        current_dir = os.getcwd()
        images = []
        for im_name in ['Birman_4.jpg', 'Bombay_1.png']:
            im_path = os.path.join(current_dir, 'test_data', '{{package_name}}-data', 'test_data_1', im_name)
            image = Image.open(im_path)
            image.load()
            images.append(image)

        # We will test a pipeline
        pipeline = [preprocess.convert_rgb, preprocess.resize]

        # We test each returned function
        result = preprocess.apply_pipeline(images, pipeline)
        self.assertEqual(type(result), list)
        self.assertTrue(len(result), 2)
        self.assertTrue(str(type(result[0])).startswith("<class 'PIL."))  # Can be different depending on the preprocess

    def test04_convert_rgb(self):
        '''Test of the function preprocess.convert_rgb'''
        # We get 1 image for the tests
        current_dir = os.getcwd()
        im_path = os.path.join(current_dir, 'test_data', '{{package_name}}-data', 'test_data_1', 'Birman_4.jpg')
        image = Image.open(im_path)
        image.load()

        # Nominal case
        new_images = preprocess.convert_rgb([image])
        self.assertEqual(type(new_images), list)
        self.assertTrue(len(new_images), 1)
        self.assertTrue(str(type(new_images[0])).startswith("<class 'PIL."))  # Can be different depending on the preprocess
        new_image = new_images[0]
        self.assertEqual(np.array(new_image).shape[-1], 3)

    def test05_resize(self):
        '''Test of the function preprocess.resize'''
        # We get 1 image for the tests
        current_dir = os.getcwd()
        im_path = os.path.join(current_dir, 'test_data', '{{package_name}}-data', 'test_data_1', 'Birman_4.jpg')
        image = Image.open(im_path)
        image.load()

        # Nominal case
        new_images = preprocess.resize([image], width=123, height=123)
        self.assertEqual(type(new_images), list)
        self.assertTrue(len(new_images), 1)
        self.assertTrue(str(type(new_images[0])).startswith("<class 'PIL."))  # Can be different depending on the preprocess
        new_image = new_images[0]
        self.assertEqual(np.array(new_image).shape[0], 123)
        self.assertEqual(np.array(new_image).shape[1], 123)

        # Nominal case - bis
        new_images = preprocess.resize([image], width=846, height=10)
        self.assertEqual(type(new_images), list)
        self.assertTrue(len(new_images), 1)
        self.assertTrue(str(type(new_images[0])).startswith("<class 'PIL."))  # Can be different depending on the preprocess
        new_image = new_images[0]
        self.assertEqual(np.array(new_image).shape[0], 10)
        self.assertEqual(np.array(new_image).shape[1], 846)

        # Check errors
        with self.assertRaises(ValueError):
            preprocess.resize([image], width=0, height=10)
        with self.assertRaises(ValueError):
            preprocess.resize([image], width=10, height=0)

    def test06_jpeg_compression(self):
        '''Test of the function preprocess.jpeg_compression'''
        # We get 1 image for the tests
        current_dir = os.getcwd()
        im_path = os.path.join(current_dir, 'test_data', '{{package_name}}-data', 'test_data_1', 'Birman_4.jpg')
        image = Image.open(im_path)
        image.load()

        # First convert as RGB
        image = preprocess.convert_rgb([image])[0]

        # Nominal case
        new_images = preprocess.jpeg_compression([image])
        self.assertEqual(type(new_images), list)
        self.assertTrue(len(new_images), 1)
        self.assertTrue(str(type(new_images[0])).startswith("<class 'PIL."))  # Can be different depending on the preprocess


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
