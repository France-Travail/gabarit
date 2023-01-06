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
import random
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io

from {{package_name}} import utils
from {{package_name}}.models_training.object_detectors import utils_object_detectors

# Disable logging
import logging
logging.disable(logging.CRITICAL)


class UtilsObjectDetectorTests(unittest.TestCase):
    '''Main class to test all functions in utils_object_detectors.py'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_draw_bboxes_from_file(self):
        '''Test of the function utils_object_detectors.draw_bboxes_from_file'''

        # Set vars
        input_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'apple.jpg')
        with Image.open(input_path) as im:
            w, h = im.size
        gt_bboxes = [{'x1': 2, 'y1': 2, 'x2': 4, 'y2': 4, 'class': 'test', 'toto': 8}, {'x1': 1, 'y1': 1, 'x2': 10, 'y2': 8, 'titi': 'tata'}]
        predicted_bboxes = [{'x1': 3, 'y1': 2, 'x2': 4, 'y2': 4, 'class': 'titi', 'toto': 0.1}, {'x1': 5, 'y1': 1, 'x2': 8, 'y2': 22, 'titi': 'tata'}]

        # Nominal case
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_folder:
            output_path = os.path.join(tmp_folder, 'tmp_image.png')
            output_im = utils_object_detectors.draw_bboxes_from_file(input_path, output_path=output_path, gt_bboxes=gt_bboxes, predicted_bboxes=predicted_bboxes)
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(type(output_im) == np.ndarray)
            self.assertEqual(output_im.shape[0], h)
            self.assertEqual(output_im.shape[1], w)

        # Without output
        output_im = utils_object_detectors.draw_bboxes_from_file(input_path, output_path=None, gt_bboxes=gt_bboxes, predicted_bboxes=predicted_bboxes)
        self.assertTrue(type(output_im) == np.ndarray)
        self.assertEqual(output_im.shape[0], h)
        self.assertEqual(output_im.shape[1], w)

        # Without gt_bboxes
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_folder:
            output_path = os.path.join(tmp_folder, 'tmp_image.png')
            output_im = utils_object_detectors.draw_bboxes_from_file(input_path, output_path=output_path, gt_bboxes=None, predicted_bboxes=predicted_bboxes)
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(type(output_im) == np.ndarray)
            self.assertEqual(output_im.shape[0], h)
            self.assertEqual(output_im.shape[1], w)

        # Without predicted_bboxes
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_folder:
            output_path = os.path.join(tmp_folder, 'tmp_image.png')
            output_im = utils_object_detectors.draw_bboxes_from_file(input_path, output_path=output_path, gt_bboxes=gt_bboxes, predicted_bboxes=None)
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(type(output_im) == np.ndarray)
            self.assertEqual(output_im.shape[0], h)
            self.assertEqual(output_im.shape[1], w)

        # Without gt_bboxes & predicted_bboxes
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_folder:
            output_path = os.path.join(tmp_folder, 'tmp_image.png')
            output_im = utils_object_detectors.draw_bboxes_from_file(input_path, output_path=output_path, gt_bboxes=None, predicted_bboxes=None)
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(type(output_im) == np.ndarray)
            self.assertEqual(output_im.shape[0], h)
            self.assertEqual(output_im.shape[1], w)

        # Check errors
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_folder:
            fake_path = os.path.join(tmp_folder, 'fake.jpg')
            with self.assertRaises(FileNotFoundError):
                output_im = utils_object_detectors.draw_bboxes_from_file(fake_path, output_path=output_path, gt_bboxes=gt_bboxes, predicted_bboxes=predicted_bboxes)

    def test02_draw_bboxes(self):
        '''Test of the function utils_object_detectors.draw_bboxes'''

        # Set vars
        input_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'apple.jpg')
        input_img = io.imread(input_path)
        h, w, depth = input_img.shape[0], input_img.shape[1], input_img.shape[2]
        gt_bboxes = [{'x1': 2, 'y1': 2, 'x2': 4, 'y2': 4, 'class': 'test', 'toto': 8}, {'x1': 1, 'y1': 1, 'x2': 10, 'y2': 8, 'titi': 'tata'}]
        predicted_bboxes = [{'x1': 3, 'y1': 2, 'x2': 4, 'y2': 4, 'class': 'titi', 'toto': 0.1}, {'x1': 5, 'y1': 1, 'x2': 8, 'y2': 22, 'titi': 'tata'}]

        # Nominal case
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_folder:
            output_path = os.path.join(tmp_folder, 'tmp_image.png')
            output_im = utils_object_detectors.draw_bboxes(input_img, output_path=output_path, gt_bboxes=gt_bboxes, predicted_bboxes=predicted_bboxes)
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(type(output_im) == np.ndarray)
            self.assertEqual(output_im.shape[0], h)
            self.assertEqual(output_im.shape[1], w)
            self.assertEqual(output_im.shape[2], depth)

        # Without output
        output_im = utils_object_detectors.draw_bboxes(input_img, output_path=None, gt_bboxes=gt_bboxes, predicted_bboxes=predicted_bboxes)
        self.assertTrue(type(output_im) == np.ndarray)
        self.assertEqual(output_im.shape[0], h)
        self.assertEqual(output_im.shape[1], w)
        self.assertEqual(output_im.shape[2], depth)

        # Without gt_bboxes
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_folder:
            output_path = os.path.join(tmp_folder, 'tmp_image.png')
            output_im = utils_object_detectors.draw_bboxes(input_img, output_path=output_path, gt_bboxes=None, predicted_bboxes=predicted_bboxes)
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(type(output_im) == np.ndarray)
            self.assertEqual(output_im.shape[0], h)
            self.assertEqual(output_im.shape[1], w)
            self.assertEqual(output_im.shape[2], depth)

        # Without predicted_bboxes
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_folder:
            output_path = os.path.join(tmp_folder, 'tmp_image.png')
            output_im = utils_object_detectors.draw_bboxes(input_img, output_path=output_path, gt_bboxes=gt_bboxes, predicted_bboxes=None)
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(type(output_im) == np.ndarray)
            self.assertEqual(output_im.shape[0], h)
            self.assertEqual(output_im.shape[1], w)
            self.assertEqual(output_im.shape[2], depth)

        # Without gt_bboxes & predicted_bboxes
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_folder:
            output_path = os.path.join(tmp_folder, 'tmp_image.png')
            output_im = utils_object_detectors.draw_bboxes(input_img, output_path=output_path, gt_bboxes=None, predicted_bboxes=None)
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(type(output_im) == np.ndarray)
            self.assertEqual(output_im.shape[0], h)
            self.assertEqual(output_im.shape[1], w)

        # Check errors
        with tempfile.TemporaryFile(dir=os.getcwd()) as tmp_file:
            fake_path = tmp_file.name
            with self.assertRaises(FileExistsError):
                output_im = utils_object_detectors.draw_bboxes(input_img, output_path=fake_path, gt_bboxes=gt_bboxes, predicted_bboxes=predicted_bboxes)

    def test03_draw_rectangle_from_bbox(self):
        '''Test of the function utils_object_detectors.draw_rectangle_from_bbox'''

        # Set vars
        input_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'apple.jpg')
        input_img = io.imread(input_path)
        bbox_with_class = {'x1': 2, 'y1': 2, 'x2': 4, 'y2': 4, 'class': 'test', 'toto': 8}
        bbox_without_class = {'x1': 2, 'y1': 2, 'x2': 4, 'y2': 4}

        # Nominal case - we only test that the function does not give an error
        utils_object_detectors.draw_rectangle_from_bbox(input_img, bbox_with_class, color=None, thickness=None, with_center=False)
        # Without class
        utils_object_detectors.draw_rectangle_from_bbox(input_img, bbox_without_class, color=None, thickness=None, with_center=False)
        # Test the other parameters
        utils_object_detectors.draw_rectangle_from_bbox(input_img, bbox_without_class, color=(0, 255, 0, 255), thickness=5, with_center=True)

        # Check errors
        with self.assertRaises(ValueError):
            utils_object_detectors.draw_rectangle_from_bbox(input_img, {'y1': 2, 'x2': 4, 'y2': 4}, color=None, thickness=None, with_center=False)
        with self.assertRaises(ValueError):
            utils_object_detectors.draw_rectangle_from_bbox(input_img, {'x1': 2, 'x2': 4, 'y2': 4}, color=None, thickness=None, with_center=False)
        with self.assertRaises(ValueError):
            utils_object_detectors.draw_rectangle_from_bbox(input_img, {'x1': 2, 'y1': 2, 'y2': 4}, color=None, thickness=None, with_center=False)
        with self.assertRaises(ValueError):
            utils_object_detectors.draw_rectangle_from_bbox(input_img, {'x1': 2, 'y1': 2, 'x2': 4}, color=None, thickness=None, with_center=False)

    def test04_check_coordinates_validity(self):
        '''Test of the function utils_object_detectors.check_coordinates_validity'''

        # Set vars
        def fonction_support_args(x1, x2, y1, y2, x, y, w, h):
            pass
        def fonction_support_kwargs(x1=None, x2=None, y1=None, y2=None, x=None, y=None, w=None, h=None):
            pass
        def fonction_support_mix(x1, y1, x, w, x2=None, y2=None, y=None, h=None):
            pass

        # Nominal case
        utils_object_detectors.check_coordinates_validity(fonction_support_args)(0, 1, 0, 1, 0, 0, 1, 1)
        utils_object_detectors.check_coordinates_validity(fonction_support_kwargs)(x1=0, y1=0, x=0, y=0, x2=1, w=1, h=1, y2=1)
        utils_object_detectors.check_coordinates_validity(fonction_support_mix)(0, 0, 0, 1, x2=1, y2=1, y=0, h=1)

        # Check errors - args
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_args)(-1, 1, 0, 1, 0, 0, 1, 1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_args)(0, -1, 0, 1, 0, 0, 1, 1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_args)(0, 1, -1, 1, 0, 0, 1, 1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_args)(0, 1, 0, -1, 0, 0, 1, 1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_args)(0, 1, 0, 1, -1, 0, 1, 1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_args)(0, 1, 0, 1, 0, -1, 1, 1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_args)(0, 1, 0, 1, 0, 1, -1, 1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_args)(0, 1, 0, 1, 0, 1, 1, -1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_args)(2, 1, 0, 1, 0, 0, 1, 1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_args)(0, 1, 2, 1, 0, 0, 1, 1)

        # Check errors - kwargs
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_kwargs)(x1=-1, y1=0, x=0, y=0, x2=1, w=1, h=1, y2=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_kwargs)(x1=0, y1=-1, x=0, y=0, x2=1, w=1, h=1, y2=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_kwargs)(x1=0, y1=0, x=-1, y=0, x2=1, w=1, h=1, y2=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_kwargs)(x1=0, y1=0, x=0, y=-1, x2=1, w=1, h=1, y2=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_kwargs)(x1=0, y1=0, x=0, y=0, x2=-1, w=1, h=1, y2=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_kwargs)(x1=0, y1=0, x=0, y=0, x2=1, w=-1, h=1, y2=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_kwargs)(x1=0, y1=0, x=0, y=0, x2=1, w=1, h=-1, y2=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_kwargs)(x1=0, y1=0, x=0, y=0, x2=1, w=1, h=1, y2=-1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_kwargs)(x1=2, y1=0, x=0, y=0, x2=1, w=1, h=1, y2=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_kwargs)(x1=0, y1=2, x=0, y=0, x2=1, w=1, h=1, y2=1)

        # Check errors - mix
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_mix)(-1, 0, 0, 1, x2=1, y2=1, y=0, h=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_mix)(0, -1, 0, 1, x2=1, y2=1, y=0, h=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_mix)(0, 0, -1, 1, x2=1, y2=1, y=0, h=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_mix)(0, 0, 0, -1, x2=1, y2=1, y=0, h=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_mix)(0, 0, 0, 1, x2=-1, y2=1, y=0, h=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_mix)(0, 0, 0, 1, x2=1, y2=-1, y=0, h=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_mix)(0, 0, 0, 1, x2=1, y2=1, y=-1, h=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_mix)(0, 0, 0, 1, x2=1, y2=1, y=0, h=-1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_mix)(2, 0, 0, 1, x2=1, y2=1, y=0, h=1)
        with self.assertRaises(AssertionError):
            utils_object_detectors.check_coordinates_validity(fonction_support_mix)(0, 2, 0, 1, x2=1, y2=1, y=0, h=1)

    def test05_xyxy_to_xyhw(self):
        '''Test of the function utils_object_detectors.xyxy_to_xyhw'''

        # Set vars
        x1, y1, x2, y2 = (2, 4, 10, 15)
        x, y, h, w = (2, 4, 11, 8)  # Targets

        # Nominal case
        self.assertEqual((x, y, h, w), utils_object_detectors.xyxy_to_xyhw(x1, y1, x2, y2))

    def test06_xyhw_to_xyxy(self):
        '''Test of the function utils_object_detectors.xyhw_to_xyxy'''

        # Set vars
        x, y, h, w = (2, 4, 11, 8)
        x1, y1, x2, y2 = (2, 4, 10, 15)  # Targets

        # Nominal case
        self.assertEqual((x1, y1, x2, y2), utils_object_detectors.xyhw_to_xyxy(x, y, h, w))
        self.assertEqual((x1, y1, x2, y2), utils_object_detectors.xyhw_to_xyxy(*utils_object_detectors.xyxy_to_xyhw(x1, y1, x2, y2)))  # Reciprocal fonctions

    def test07_xyxy_to_cxcyhw(self):
        '''Test of the function utils_object_detectors.xyxy_to_cxcyhw'''

        # Set vars
        x1, y1, x2, y2 = (2, 4, 10, 15)
        cx, cy, h, w = (6, 9.5, 11, 8)  # Targets

        # Nominal case
        self.assertEqual((cx, cy, h, w), utils_object_detectors.xyxy_to_cxcyhw(x1, y1, x2, y2))

    def test08_get_area_from_xyxy(self):
        '''Test of the function utils_object_detectors.get_area_from_xyxy'''

        # Set vars
        x1, y1, x2, y2 = (2, 4, 10, 15)
        area = 88  # Target

        # Nominal case
        self.assertEqual(area, utils_object_detectors.get_area_from_xyxy(x1, y1, x2, y2))

    def test09_get_iou(self):
        '''Test of the function utils_object_detectors.get_iou'''

        # Set vars
        coordinatesA = (2, 2, 5, 5)
        coordinatesB = (20, 20, 25, 25)
        coordinatesC = (3, 3, 7, 7)
        intersection_A_C = 2 * 2
        union_A_C = (1 * 3 + 1 * 2) + (2 * 4 + 2 * 2) + intersection_A_C  # Cf. figure below
        IOU_A_C = intersection_A_C / union_A_C

        #   2  3       5      7
        # 2  _   _   _
        #   |    _   _ | _  _
        #   |  |       |      |
        # 5 |_ | _   _ |      |
        #      |              |
        # 7    | _   _   _  _ |

        # Nominal case
        self.assertEqual(1, utils_object_detectors.get_iou(coordinatesA, coordinatesA))  # Same rectangle
        self.assertEqual(0, utils_object_detectors.get_iou(coordinatesA, coordinatesB))  # No union
        self.assertEqual(0, utils_object_detectors.get_iou(coordinatesB, coordinatesA))  # No union
        self.assertAlmostEqual(IOU_A_C, utils_object_detectors.get_iou(coordinatesA, coordinatesC))
        self.assertAlmostEqual(IOU_A_C, utils_object_detectors.get_iou(coordinatesC, coordinatesA))

    def test10_get_new_img_size_from_min_side_size(self):
        '''Test of the function utils_object_detectors.get_new_img_size_from_min_side_size'''

        # Set vars
        h1, w1, min1 = (50, 100, 150)
        h1_new, w1_new = (150, 300)
        h2, w2, min2 = (50, 100, 25)
        h2_new, w2_new = (25, 50)
        h3, w3, min3 = (100, 50, 150)
        h3_new, w3_new = (300, 150)
        h4, w4, min4 = (100, 50, 25)
        h4_new, w4_new = (50, 25)
        h5, w5, min5 = (100, 100, 200)
        h5_new, w5_new = (200, 200)

        # Nominal case
        self.assertEqual((h1_new, w1_new), utils_object_detectors.get_new_img_size_from_min_side_size(h1, w1, min1))
        self.assertEqual((h2_new, w2_new), utils_object_detectors.get_new_img_size_from_min_side_size(h2, w2, min2))
        self.assertEqual((h3_new, w3_new), utils_object_detectors.get_new_img_size_from_min_side_size(h3, w3, min3))
        self.assertEqual((h4_new, w4_new), utils_object_detectors.get_new_img_size_from_min_side_size(h4, w4, min4))
        self.assertEqual((h5_new, w5_new), utils_object_detectors.get_new_img_size_from_min_side_size(h5, w5, min5))

        # Check errors
        with self.assertRaises(ValueError):
            utils_object_detectors.get_new_img_size_from_min_side_size(0, 10, 10)
        with self.assertRaises(ValueError):
            utils_object_detectors.get_new_img_size_from_min_side_size(10, 0, 10)
        with self.assertRaises(ValueError):
            utils_object_detectors.get_new_img_size_from_min_side_size(10, 10, 0)

    def test11_get_feature_map_size(self):
        '''Test of the function utils_object_detectors.get_feature_map_size'''

        # Set vars
        h1, w1, ratio1 = (50, 100, 10)
        h1_new, w1_new = (5, 10)
        h2, w2, ratio2 = (200, 100, 25)
        h2_new, w2_new = (8, 4)
        h3, w3, ratio3 = (100, 200, 15)
        h3_new, w3_new = (6, 13)

        # Nominal case
        self.assertEqual((h1_new, w1_new), utils_object_detectors.get_feature_map_size(h1, w1, ratio1))
        self.assertEqual((h2_new, w2_new), utils_object_detectors.get_feature_map_size(h2, w2, ratio2))
        self.assertEqual((h3_new, w3_new), utils_object_detectors.get_feature_map_size(h3, w3, ratio3))

        # Check errors
        with self.assertRaises(ValueError):
            utils_object_detectors.get_feature_map_size(0, 10, 10)
        with self.assertRaises(ValueError):
            utils_object_detectors.get_feature_map_size(10, 0, 10)
        with self.assertRaises(ValueError):
            utils_object_detectors.get_feature_map_size(10, 10, 0)

    def test12_calc_regr(self):
        '''Test of the function utils_object_detectors.calc_regr'''

        # Set vars
        bbox = (5, 13, 9, 18)  # Centers 7 / 15.5, width 4, height 5
        anchor = (10, 8, 16, 20)  # Centers 13 / 14, width 6, height 12
        tx, ty = (-1., 0.125)  # -6 / 6, 1.5 / 12
        th, tw = (np.log(5/12), np.log(4/6))  # 5 / 12, 4 / 6

        # Nominal case
        self.assertEqual((tx, ty, th, tw), utils_object_detectors.calc_regr(bbox, anchor))

    def test13_apply_regression(self):
        '''Test of the function utils_object_detectors.apply_regression'''

        # Set vars
        bbox = (5, 13, 5, 4)  # x, y, h, w
        x_anc, y_anc, h_anc, w_anc = (10, 8, 12, 6)
        tx, ty = (-1., 0.125)
        th, tw = (np.log(5/12), np.log(4/6))
        coordinates_and_regression = (x_anc, y_anc, h_anc, w_anc, tx, ty, th, tw)

        # Nominal case
        self.assertEqual(bbox, utils_object_detectors.apply_regression(coordinates_and_regression))

        # Check reciprocal functions
        bbox = (5, 13, 9, 18)
        anchor = (10, 8, 16, 20)
        bbox_target = utils_object_detectors.xyxy_to_xyhw(*bbox)
        tx, ty, th, tw = utils_object_detectors.calc_regr(bbox, anchor)
        x_anc, y_anc, h_anc, w_anc = utils_object_detectors.xyxy_to_xyhw(*anchor)
        coordinates_and_regression = (x_anc, y_anc, h_anc, w_anc, tx, ty, th, tw)
        self.assertEqual(bbox_target, utils_object_detectors.apply_regression(coordinates_and_regression))

    def test14_non_max_suppression_fast(self):
        '''Test of the function utils_object_detectors.non_max_suppression_fast'''

        # Set vars
        img_boxes_coordinates = np.array([
            [2, 2, 4, 4],
            [3, 2, 4, 4],  # IOU 1/2 with first,
            [3, 2, 5, 4],  # IOU 1/3 with first, 1/2 with second
        ])

        ### Nominal case
        # Test 1
        img_boxes_probas_1 = np.array([0.9, 0.5, 0.7])
        nms_overlap_threshold_1 = 0.4
        nms_max_boxes_1 = 10
        img_boxes_classes_1 = None
        img_boxes_coordinates_res_1 = np.array([
            [2, 2, 4, 4],
            [3, 2, 5, 4],
        ])
        img_boxes_probas_res_1 = np.array([0.9, 0.7])
        img_boxes_classes_res_1 = None
        res_11, res_12, res_13 = utils_object_detectors.non_max_suppression_fast(img_boxes_coordinates, img_boxes_probas_1,
                                                                                 nms_overlap_threshold_1, nms_max_boxes_1,
                                                                                 img_boxes_classes_1)
        np.testing.assert_array_equal(res_11, img_boxes_coordinates_res_1)
        np.testing.assert_array_equal(res_12, img_boxes_probas_res_1)
        np.testing.assert_array_equal(res_13, img_boxes_classes_res_1)

        # Test 2 - higher threshold  & check proba order
        img_boxes_probas_2 = np.array([0.9, 0.5, 0.7])
        nms_overlap_threshold_2 = 0.9
        nms_max_boxes_2 = 10
        img_boxes_classes_2 = None
        img_boxes_coordinates_res_2 = np.array([
            [2, 2, 4, 4],
            [3, 2, 5, 4],
            [3, 2, 4, 4],
        ])
        img_boxes_probas_res_2 = np.array([0.9, 0.7, 0.5])
        img_boxes_classes_res_2 = None
        res_21, res_22, res_23 = utils_object_detectors.non_max_suppression_fast(img_boxes_coordinates, img_boxes_probas_2,
                                                                                 nms_overlap_threshold_2, nms_max_boxes_2,
                                                                                 img_boxes_classes_2)
        np.testing.assert_array_equal(res_21, img_boxes_coordinates_res_2)
        np.testing.assert_array_equal(res_22, img_boxes_probas_res_2)
        np.testing.assert_array_equal(res_23, img_boxes_classes_res_2)

        # Test 3 - lower threshold
        img_boxes_probas_3 = np.array([0.9, 0.5, 0.7])
        nms_overlap_threshold_3 = 0.1
        nms_max_boxes_3 = 10
        img_boxes_classes_3 = None
        img_boxes_coordinates_res_3 = np.array([
            [2, 2, 4, 4],
        ])
        img_boxes_probas_res_3 = np.array([0.9])
        img_boxes_classes_res_3 = None
        res_31, res_32, res_33 = utils_object_detectors.non_max_suppression_fast(img_boxes_coordinates, img_boxes_probas_3,
                                                                                 nms_overlap_threshold_3, nms_max_boxes_3,
                                                                                 img_boxes_classes_3)
        np.testing.assert_array_equal(res_31, img_boxes_coordinates_res_3)
        np.testing.assert_array_equal(res_32, img_boxes_probas_res_3)
        np.testing.assert_array_equal(res_33, img_boxes_classes_res_3)

        # Test 4 - we diminish the maximal number of boxes
        img_boxes_probas_4 = np.array([0.9, 0.5, 0.7])
        nms_overlap_threshold_4 = 0.4
        nms_max_boxes_4 = 1
        img_boxes_classes_4 = None
        img_boxes_coordinates_res_4 = np.array([
            [2, 2, 4, 4],
        ])
        img_boxes_probas_res_4 = np.array([0.9])
        img_boxes_classes_res_4 = None
        res_41, res_42, res_43 = utils_object_detectors.non_max_suppression_fast(img_boxes_coordinates, img_boxes_probas_4,
                                                                                 nms_overlap_threshold_4, nms_max_boxes_4,
                                                                                 img_boxes_classes_4)
        np.testing.assert_array_equal(res_41, img_boxes_coordinates_res_4)
        np.testing.assert_array_equal(res_42, img_boxes_probas_res_4)
        np.testing.assert_array_equal(res_43, img_boxes_classes_res_4)

        # Test 5 - with img_boxes_classes
        img_boxes_probas_5 = np.array([0.9, 0.5, 0.7])
        nms_overlap_threshold_5 = 0.4
        nms_max_boxes_5 = 10
        img_boxes_classes_5 = np.array(['titi', 'tata', 'toto'])
        img_boxes_coordinates_res_5 = np.array([
            [2, 2, 4, 4],
            [3, 2, 5, 4],
        ])
        img_boxes_probas_res_5 = np.array([0.9, 0.7])
        img_boxes_classes_res_5 = np.array(['titi', 'toto'])
        res_51, res_52, res_53 = utils_object_detectors.non_max_suppression_fast(img_boxes_coordinates, img_boxes_probas_5,
                                                                                 nms_overlap_threshold_5, nms_max_boxes_5,
                                                                                 img_boxes_classes_5)
        np.testing.assert_array_equal(res_51, img_boxes_coordinates_res_5)
        np.testing.assert_array_equal(res_52, img_boxes_probas_res_5)
        np.testing.assert_array_equal(res_53, img_boxes_classes_res_5)

        # Test 6 - with a different proba order
        img_boxes_probas_6 = np.array([0.5, 0.9, 0.7])
        nms_overlap_threshold_6 = 0.4
        nms_max_boxes_6 = 10
        img_boxes_classes_6 = np.array(['titi', 'tata', 'toto'])
        img_boxes_coordinates_res_6 = np.array([
            [3, 2, 4, 4]
        ])
        img_boxes_probas_res_6 = np.array([0.9])
        img_boxes_classes_res_6 = np.array(['tata'])
        res_61, res_62, res_63 = utils_object_detectors.non_max_suppression_fast(img_boxes_coordinates, img_boxes_probas_6,
                                                                                 nms_overlap_threshold_6, nms_max_boxes_6,
                                                                                 img_boxes_classes_6)
        np.testing.assert_array_equal(res_61, img_boxes_coordinates_res_6)
        np.testing.assert_array_equal(res_62, img_boxes_probas_res_6)
        np.testing.assert_array_equal(res_63, img_boxes_classes_res_6)

        # Check errors
        with self.assertRaises(ValueError):
            utils_object_detectors.non_max_suppression_fast(img_boxes_coordinates, np.array([0.5, 0.9]),
                                                            nms_overlap_threshold_6, nms_max_boxes_6,
                                                            img_boxes_classes_6)
        with self.assertRaises(ValueError):
            utils_object_detectors.non_max_suppression_fast(img_boxes_coordinates, img_boxes_probas_6,
                                                            0, nms_max_boxes_6,
                                                            img_boxes_classes_6)
        with self.assertRaises(ValueError):
            utils_object_detectors.non_max_suppression_fast(img_boxes_coordinates, img_boxes_probas_6,
                                                            1.01, nms_max_boxes_6,
                                                            img_boxes_classes_6)
        with self.assertRaises(ValueError):
            utils_object_detectors.non_max_suppression_fast(img_boxes_coordinates, img_boxes_probas_6,
                                                            nms_overlap_threshold_6, 0,
                                                            img_boxes_classes_6)
        with self.assertRaises(ValueError):
            utils_object_detectors.non_max_suppression_fast(img_boxes_coordinates, img_boxes_probas_6,
                                                            nms_overlap_threshold_6, nms_max_boxes_6,
                                                            np.array(['titi', 'tata']))

    def test15_get_all_viable_anchors_boxes(self):
        '''Test of the function utils_object_detectors.get_all_viable_anchors_boxes'''
        base_anchors = [(10, 10), (20, 20)]
        subsampling_ratio = 10
        feature_map_height = 2
        feature_map_width = 2
        im_resized_height = 100
        im_resized_width = 100

        viable_anchor_boxes = utils_object_detectors.get_all_viable_anchors_boxes(
            base_anchors, subsampling_ratio, feature_map_height, feature_map_width, im_resized_height, im_resized_width
        )

        self.assertEqual(
            viable_anchor_boxes, 
            {
                (0, 0, 0): {'anchor_img_coordinates': (0.0, 0.0, 10.0, 10.0)}, 
                (1, 0, 0): {'anchor_img_coordinates': (0.0, 10.0, 10.0, 20.0)}, 
                (0, 1, 0): {'anchor_img_coordinates': (10.0, 0.0, 20.0, 10.0)},
                (1, 1, 0): {'anchor_img_coordinates': (10.0, 10.0, 20.0, 20.0)}, 
                (1, 1, 1): {'anchor_img_coordinates': (5.0, 5.0, 25.0, 25.0)}
            }
        )

        # Test with no viable anchor
        with self.assertRaises(RuntimeError):
            base_anchors = [(100, 100)]
            utils_object_detectors.get_all_viable_anchors_boxes(
                base_anchors, subsampling_ratio, feature_map_height, feature_map_width, im_resized_height, im_resized_width
            )
        
    def test16_get_iou_anchors_bboxes(self):
        '''Test of the function utils_object_detectors.get_iou_anchors_bboxes'''
        anchors = {
            (0, 0, 0): {'anchor_img_coordinates': (0.0, 0.0, 10.0, 10.0)}, 
            (1, 0, 0): {'anchor_img_coordinates': (0.0, 10.0, 10.0, 20.0)}, 
            (0, 1, 0): {'anchor_img_coordinates': (10.0, 0.0, 20.0, 10.0)},
            (1, 1, 0): {'anchor_img_coordinates': (10.0, 10.0, 20.0, 20.0)}, 
            (1, 1, 1): {'anchor_img_coordinates': (5.0, 5.0, 25.0, 25.0)}
        }

        image_bboxes = [{"x1": 0.0, "y1": 0.0, "x2": 5.0, "y2": 5.0}]

        expected_result = {
            (0, 0, 0): {'anchor_img_coordinates': (0.0, 0.0, 10.0, 10.0), 'bboxes': {0: {"iou": 0.25, "bbox_img_coordinates": (0.0, 0.0, 5.0, 5.0)}}}, 
            (1, 0, 0): {'anchor_img_coordinates': (0.0, 10.0, 10.0, 20.0), 'bboxes': {0: {"iou": 0.0, "bbox_img_coordinates": (0.0, 0.0, 5.0, 5.0)}}}, 
            (0, 1, 0): {'anchor_img_coordinates': (10.0, 0.0, 20.0, 10.0), 'bboxes': {0: {"iou": 0.0, "bbox_img_coordinates": (0.0, 0.0, 5.0, 5.0)}}},
            (1, 1, 0): {'anchor_img_coordinates': (10.0, 10.0, 20.0, 20.0), 'bboxes': {0: {"iou": 0.0, "bbox_img_coordinates": (0.0, 0.0, 5.0, 5.0)}}}, 
            (1, 1, 1): {'anchor_img_coordinates': (5.0, 5.0, 25.0, 25.0), 'bboxes': {0: {"iou": 0.0, "bbox_img_coordinates": (0.0, 0.0, 5.0, 5.0)}}}
        }

        self.assertEqual(expected_result, utils_object_detectors.get_iou_anchors_bboxes(anchors, image_bboxes))

    def test17_set_anchors_type_validity(self):
        '''Test of the function utils_object_detectors.set_anchors_type_validity'''
        anchor_boxes_dict = {
            (0, 0, 0): {'bboxes': {0: {"iou": 0.3}, 1: {"iou": 0.05}}}, 
            (1, 0, 0): {'bboxes': {0: {"iou": 0.2}, 1: {"iou": 0.05}}}, 
            (0, 1, 0): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.05}}},
            (1, 1, 0): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.05}}}, 
            (1, 1, 1): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.05}}}
        }

        image_bboxes =[0, 1]
        rpn_min_overlap = 0.1
        rpn_max_overlap = 0.2
        
        expected_anchor_boxes_dict = {
            (0, 0, 0): {'bboxes': {0: {"iou": 0.3}, 1: {"iou": 0.05}}, "anchor_type": "pos", "anchor_validity": 1, "best_bbox_index": 0}, 
            (1, 0, 0): {'bboxes': {0: {"iou": 0.2}, 1: {"iou": 0.05}}, "anchor_type": "neutral", "anchor_validity": 0, "best_bbox_index": -1}, 
            (0, 1, 0): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.05}}, "anchor_type": "neg", "anchor_validity": 1, "best_bbox_index": -1},
            (1, 1, 0): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.05}}, "anchor_type": "neg", "anchor_validity": 1, "best_bbox_index": -1}, 
            (1, 1, 1): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.05}}, "anchor_type": "neg", "anchor_validity": 1, "best_bbox_index": -1}
        }

        expected_bboxes_index_with_no_positive = [1]
        expected_result = (expected_anchor_boxes_dict, expected_bboxes_index_with_no_positive)

        result = utils_object_detectors.set_anchors_type_validity(anchor_boxes_dict, image_bboxes, rpn_min_overlap, rpn_max_overlap)

        self.assertEqual(expected_result, result)

    def test18_complete_at_least_one_anchor_per_bbox(self):
        '''Test of the function utils_object_detectors.complete_at_least_one_anchor_per_bbox'''
        anchor_boxes_dict = {
            (0, 0, 0): {'bboxes': {0: {"iou": 0.3}, 1: {"iou": 0.01}}, "anchor_type": "pos", "anchor_validity": 1, "best_bbox_index": 0}, 
            (1, 0, 0): {'bboxes': {0: {"iou": 0.2}, 1: {"iou": 0.02}}, "anchor_type": "neutral", "anchor_validity": 0, "best_bbox_index": -1}, 
            (0, 1, 0): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.03}}, "anchor_type": "neg", "anchor_validity": 1, "best_bbox_index": -1},
            (1, 1, 0): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.04}}, "anchor_type": "neg", "anchor_validity": 1, "best_bbox_index": -1}, 
            (1, 1, 1): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.05}}, "anchor_type": "neg", "anchor_validity": 1, "best_bbox_index": -1}
        }

        bboxes_index_with_no_positive = [1]

        expected_result = {
            (0, 0, 0): {'bboxes': {0: {"iou": 0.3}, 1: {"iou": 0.01}}, "anchor_type": "pos", "anchor_validity": 1, "best_bbox_index": 0}, 
            (1, 0, 0): {'bboxes': {0: {"iou": 0.2}, 1: {"iou": 0.02}}, "anchor_type": "neutral", "anchor_validity": 0, "best_bbox_index": -1}, 
            (0, 1, 0): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.03}}, "anchor_type": "neg", "anchor_validity": 1, "best_bbox_index": -1},
            (1, 1, 0): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.04}}, "anchor_type": "neg", "anchor_validity": 1, "best_bbox_index": -1}, 
            (1, 1, 1): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.05}}, "anchor_type": "pos", "anchor_validity": 1, "best_bbox_index": 1}
        }

        result = utils_object_detectors.complete_at_least_one_anchor_per_bbox(anchor_boxes_dict, bboxes_index_with_no_positive)

        self.assertEqual(result, expected_result)

    def test19_restrict_valid_to_n_regions(self):
        '''Test of the function utils_object_detectors.restrict_valid_to_n_regions'''
        anchor_boxes_dict = {
            (0, 0, 0): {'bboxes': {0: {"iou": 0.3}, 1: {"iou": 0.01}}, "anchor_type": "pos", "anchor_validity": 1, "best_bbox_index": 0}, 
            (1, 0, 0): {'bboxes': {0: {"iou": 0.2}, 1: {"iou": 0.02}}, "anchor_type": "neutral", "anchor_validity": 0, "best_bbox_index": -1}, 
            (0, 1, 0): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.03}}, "anchor_type": "neg", "anchor_validity": 1, "best_bbox_index": -1},
            (1, 1, 0): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.04}}, "anchor_type": "neg", "anchor_validity": 1, "best_bbox_index": -1}, 
            (1, 1, 1): {'bboxes': {0: {"iou": 0.0}, 1: {"iou": 0.05}}, "anchor_type": "pos", "anchor_validity": 1, "best_bbox_index": 1}
        }

        num_regions = 2

        expected_result = {
            (0, 0, 0): {'bboxes': {0: {'iou': 0.3}, 1: {'iou': 0.01}}, 'anchor_type': 'pos', 'anchor_validity': 1, 'best_bbox_index': 0}, 
            (1, 0, 0): {'bboxes': {0: {'iou': 0.2}, 1: {'iou': 0.02}}, 'anchor_type': 'neutral', 'anchor_validity': 0, 'best_bbox_index': -1}, 
            (0, 1, 0): {'bboxes': {0: {'iou': 0.0}, 1: {'iou': 0.03}}, 'anchor_type': 'neg', 'anchor_validity': 1, 'best_bbox_index': -1}, 
            (1, 1, 0): {'bboxes': {0: {'iou': 0.0}, 1: {'iou': 0.04}}, 'anchor_type': 'neg', 'anchor_validity': 0, 'best_bbox_index': -1}, 
            (1, 1, 1): {'bboxes': {0: {'iou': 0.0}, 1: {'iou': 0.05}}, 'anchor_type': 'pos', 'anchor_validity': 0, 'best_bbox_index': 1}
        }

        random.seed(0) # Mandatory for determinism
        result = utils_object_detectors.restrict_valid_to_n_regions(anchor_boxes_dict, num_regions)
        self.assertEqual(result, expected_result)

    def test20_add_regression_target_to_pos_valid(self):
        '''Test of the function utils_object_detectors.add_regression_target_to_pos_valid'''
        anchor_boxes_dict = {
            (0, 0, 0): {
                'anchor_img_coordinates': (10, 8, 16, 20),
                'bboxes': {
                    0: {'iou': 0.3, "bbox_img_coordinates": (5, 13, 9, 18)}, 
                    1: {'iou': 0.0, "bbox_img_coordinates": (0.0, 5.0, 5.0, 10.)}
                }, 
                'anchor_type': 'pos', 
                'anchor_validity': 1, 
                'best_bbox_index': 0
            },
            (1, 1, 1): {
                'anchor_img_coordinates': (0.0, 0.0, 10.0, 10.0),
                'bboxes': {
                    0: {'iou': 0.0, "bbox_img_coordinates": (0.0, 0.0, 5.0, 5.0)}, 
                    1: {'iou': 0.1, "bbox_img_coordinates": (0.0, 5.0, 5.0, 10.)}
                }, 
                'anchor_type': 'neg', 
                'anchor_validity': 1, 
                'best_bbox_index': -1
            },
        }
        
        tx, ty = (-1., 0.125)  # -6 / 6, 1.5 / 12
        th, tw = (np.log(5/12), np.log(4/6))  # 5 / 12, 4 / 6
        expected_result = {
            (0, 0, 0): {
                'anchor_img_coordinates': (10, 8, 16, 20),
                'bboxes': {
                    0: {'iou': 0.3, "bbox_img_coordinates": (5, 13, 9, 18)}, 
                    1: {'iou': 0.0, "bbox_img_coordinates": (0.0, 5.0, 5.0, 10.)}
                }, 
                'anchor_type': 'pos', 
                'anchor_validity': 1, 
                'best_bbox_index': 0,
                'regression_target': (tx, ty, th, tw)
            },
            (1, 1, 1): {
                'anchor_img_coordinates': (0.0, 0.0, 10.0, 10.0),
                'bboxes': {
                    0: {'iou': 0.0, "bbox_img_coordinates": (0.0, 0.0, 5.0, 5.0)}, 
                    1: {'iou': 0.1, "bbox_img_coordinates": (0.0, 5.0, 5.0, 10.)}
                }, 
                'anchor_type': 'neg', 
                'anchor_validity': 1, 
                'best_bbox_index': -1,
                'regression_target': (0, 0, 0, 0)
            },
        }

        result = utils_object_detectors.add_regression_target_to_pos_valid(anchor_boxes_dict)
        self.assertEqual(result, expected_result)

    def test21_get_rpn_targets(self):
        '''Test of the function utils_object_detectors.get_rpn_targets'''
        class Model:
            def __init__(self):
                    self.list_anchors = [(1, 1), (3, 3)]
                    self.nb_anchors = 2
                    self.shared_model_subsampling = 10
                    self.rpn_min_overlap = 0.1
                    self.rpn_max_overlap = 0.2
                    self.rpn_regr_scaling = 1
                    self.rpn_restrict_num_regions = 2

        img_data_batch = [
            {
                "bboxes": [{"x1": 0.0, "y1": 0.0, "x2": 2.0, "y2": 2.0}],
                "batch_height": 10, 
                "batch_width": 10, 
                "resized_height": 10, 
                "resized_width": 10,
            }
        ]
        model = Model()

        Y1, Y2 = utils_object_detectors.get_rpn_targets(model, img_data_batch)
        expected_Y1 = np.array([[[[1., 1., 0., 0.]]]])
        expected_Y2 = np.array([[[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]])

        np.testing.assert_almost_equal(Y1, expected_Y1)
        np.testing.assert_almost_equal(Y2, expected_Y2)

    def test_22_get_roi_from_rpn_predictions(self):
        '''Test of the function utils_object_detectors.get_roi_from_rpn_predictions'''
        class Model:
            def __init__(self):
                    self.list_anchors = [(1, 1), (3, 3), (5, 5), (7, 7)]
                    self.nb_anchors = 4
                    self.shared_model_subsampling = 10
                    self.rpn_min_overlap = 0.1
                    self.rpn_max_overlap = 0.2
                    self.rpn_regr_scaling = 1
                    self.rpn_restrict_num_regions = 2
                    self.roi_nms_overlap_threshold = 0.2
                    self.nms_max_boxes = 2

        img_data_batch = [
            {
                "bboxes": [{"x1": 0.0, "y1": 0.0, "x2": 2.0, "y2": 2.0}],
                "batch_height": 10, 
                "batch_width": 10, 
                "resized_height": 10, 
                "resized_width": 10,
            }
        ]
        model = Model()
        batch_y_cls = np.array([[[[1., 1., 0., 0.]]]])
        batch_y_regr = np.array([[[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]])

        result = utils_object_detectors.get_roi_from_rpn_predictions(model, img_data_batch, batch_y_cls, batch_y_regr)
        expected_result = [np.array([[0, 0, 1, 1]])]

        for r, expected_r in zip(result, expected_result):
            np.testing.assert_almost_equal(r, expected_r)

    def test23_restrict_and_convert_roi_boxes(self):
        '''Test of the function utils_object_detectors.restrict_and_convert_roi_boxes'''
        bbox_coordinates = (-5, -5, 20, 20, 5, 5)
        expected_result = (0, 0, 5, 5)
        result = utils_object_detectors.restrict_and_convert_roi_boxes(bbox_coordinates)
        self.assertEqual(result, expected_result)

    def test24_select_final_rois(self):
        '''Test of the function utils_object_detectors.select_final_rois'''
        rois_on_feature_maps = np.array([[
            [0.45, 0.45, 1., 1.],
            [0.35, 0.35, 1., 1.],
            [0.25, 0.25, 1., 1.],
            [0.15, 0.15, 1., 1.],
        ]])
        rois_probas = np.array([[1., 1., 0., 0.]])
        roi_nms_overlap_threshold = 0.2
        nms_max_boxes = 2
        feature_map_sizes= np.array([[1, 1]])

        expected_result = [np.array([[0, 0, 1, 1]])]
        result = utils_object_detectors.select_final_rois(rois_on_feature_maps, rois_probas, roi_nms_overlap_threshold, nms_max_boxes, feature_map_sizes)
        
        for r, expected_r in zip(result, expected_result):
            np.testing.assert_almost_equal(r, expected_r)

    def test25_get_classifier_train_inputs_and_targets(self):
        '''Test of the function utils_object_detectors.get_classifier_train_inputs_and_targets'''
        class Model:
            def __init__(self):
                    self.nb_rois_classifier = 1
                    self.shared_model_subsampling = 10
                    self.classifier_regr_scaling = 1
                    self.classifier_min_overlap = 0.1
                    self.classifier_max_overlap = 0.2
                    self.dict_classes = {0: "ok"}


        img_data_batch = [
            {
                "bboxes": [{"x1": 0.0, "y1": 0.0, "x2": 2.0, "y2": 2.0, "class": 0}],
                "batch_height": 10, 
                "batch_width": 10, 
                "resized_height": 10, 
                "resized_width": 10,
            }
        ]
        model = Model()
        rois_coordinates = [np.array([[0, 0, 1, 1]])]
        result = utils_object_detectors.get_classifier_train_inputs_and_targets(model, img_data_batch, rois_coordinates)
        expected_result = (
            np.array([[[0., 0., 1., 1.]]]), 
            np.array([[[0., 1.]]]), 
            np.array([[[0., 0., 0., 0., 0., 0., 0., 0.]]])
        )

        for r, expected_r in zip(result, expected_result):
            np.testing.assert_almost_equal(r, expected_r)

    def test26_get_rois_bboxes_iou(self):
        '''Test of the function utils_object_detectors.get_rois_bboxes_iou'''
        rois = np.array([[0, 0, 1, 1]])
        img_data = {"bboxes": [{"x1": 0.0, "y1": 0.0, "x2": 2.0, "y2": 2.0, "class": 0}]}
        subsampling_ratio = 8
        result = utils_object_detectors.get_rois_bboxes_iou(rois, img_data, subsampling_ratio)
        expected_result = {
            0: {
                'coordinates': {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1, 'h': 1, 'w': 1}, 
                'bboxes': {
                    0: {
                        'coordinates': {'x1': 0.0, 'y1': 0.0, 'x2': 0.25, 'y2': 0.25}, 
                        'iou': 0.0625, 
                        'class': 0
                    }
                }
            }
        }
        self.assertEqual(result, expected_result)

    def test27_get_rois_targets(self):
        '''Test of the function utils_object_detectors.get_rois_targets'''
        dict_rois = {
            0: {
                'coordinates': {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1, 'h': 1, 'w': 1}, 
                'bboxes': {
                    0: {
                        'coordinates': {'x1': 0.0, 'y1': 0.0, 'x2': 0.25, 'y2': 0.25}, 
                        'iou': 0.0625, 
                        'class': 0
                    }
                }
            }
        }
        classifier_min_overlap = 0.01
        classifier_max_overlap = 0.02
        result = utils_object_detectors.get_rois_targets(dict_rois, classifier_min_overlap, classifier_max_overlap)
        expected_result = {
            0: {
                'coordinates': {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1, 'h': 1, 'w': 1}, 
                'bboxes': {
                    0: {
                        'coordinates': {'x1': 0.0, 'y1': 0.0, 'x2': 0.25, 'y2': 0.25}, 
                        'iou': 0.0625, 'class': 0
                    }
                }, 
                'best_bbox_index': 0, 
                'best_iou': 0.0625, 
                'classifier_regression_target': (-0.375, -0.375, -1.3862943611198906, -1.3862943611198906),
                'classifier_class_target': 0
            }
        }

        self.assertEqual(result, expected_result)
    
    def test28_limit_rois_targets(self):
        '''Test of the function utils_object_detectors.limit_rois_targets'''
        dict_rois_targets = {
            0: {
                'coordinates': {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1, 'h': 1, 'w': 1}, 
                'bboxes': {
                    0: {
                        'coordinates': {'x1': 0.0, 'y1': 0.0, 'x2': 0.25, 'y2': 0.25}, 
                        'iou': 0.0625, 
                        'class': 0
                    }
                }, 
                'best_bbox_index': 0, 
                'best_iou': 0.0625, 
                'classifier_regression_target': (-0.375, -0.375, -1.3862943611198906, -1.3862943611198906),
                'classifier_class_target': 0
            }
        }
        nb_rois_per_img = 2
        result = utils_object_detectors.limit_rois_targets(dict_rois_targets, nb_rois_per_img)
        expected_result = {
            0: {
                'coordinates': {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1, 'h': 1, 'w': 1}, 
                'bboxes': {
                    0: {
                        'coordinates': {'x1': 0.0, 'y1': 0.0, 'x2': 0.25, 'y2': 0.25}, 
                        'iou': 0.0625, 
                        'class': 0
                    }
                }, 
                'best_bbox_index': 0, 
                'best_iou': 0.0625, 
                'classifier_regression_target': (-0.375, -0.375, -1.3862943611198906, -1.3862943611198906),
                'classifier_class_target': 0
            }, 
            1: {
                'coordinates': {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1, 'h': 1, 'w': 1}, 
                'bboxes': {
                    0: {
                        'coordinates': {'x1': 0.0, 'y1': 0.0, 'x2': 0.25, 'y2': 0.25}, 
                        'iou': 0.0625, 
                        'class': 0
                    }
                }, 
                'best_bbox_index': 0,
                'best_iou': 0.0625,
                'classifier_regression_target': (-0.375, -0.375, -1.3862943611198906, -1.3862943611198906),
                'classifier_class_target': 0
            }
        }
        self.assertEqual(result, expected_result)

    def test29_create_fake_dict_rois_targets(self):
        '''Test of the function utils_object_detectors.create_fake_dict_rois_targets'''
        img_data = {"resized_height": 10,  "resized_width": 10}
        subsampling_ratio = 10
        nb_rois_per_img = 1
        result =  utils_object_detectors.create_fake_dict_rois_targets(img_data, subsampling_ratio, nb_rois_per_img)
        expected_result = {
            0: {
                'coordinates': {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1, 'h': 1, 'w': 1}, 
                'classifier_regression_target': (0, 0, 0, 0), 
                'classifier_class_target': 'bg'
            }
        }
        self.assertEqual(result, expected_result)

    def test30_format_classifier_inputs_and_targets(self):
        '''Test of the function utils_object_detectors.format_classifier_inputs_and_targets'''
        dict_rois_targets = {
            0: {
                'coordinates': {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1, 'h': 1, 'w': 1},
                'classifier_regression_target': (0, 0, 0, 0), 
                'classifier_class_target': 'bg'
            }
        }
        dict_classes = {0: 'ok'}
        classifier_regr_scaling = 1

        result = utils_object_detectors.format_classifier_inputs_and_targets(dict_rois_targets, dict_classes, classifier_regr_scaling)
        expected_result = (
            np.array([[[0., 0., 1., 1.]]]), 
            np.array([[[0., 1.]]]), 
            np.array([[[0., 0., 0., 0., 0., 0., 0., 0.]]])
        )
        for r, expected_r in zip(result, expected_result):
            np.testing.assert_almost_equal(r, expected_r)

    def test31_get_classifier_test_inputs(self):
        '''Test of the function utils_object_detectors.get_classifier_test_inputs'''
        rois_coordinates = [np.array([[5, 5, 10, 10]])]
        result = utils_object_detectors.get_classifier_test_inputs(rois_coordinates)
        expected_result = np.array([[[5, 5, 5, 5]]])
        np.testing.assert_almost_equal(result, expected_result)

    def test32_get_valid_fm_boxes_from_proba(self):
        '''Test of the function utils_object_detectors.get_valid_fm_boxes_from_proba'''
        probas = np.array([[0.1, 0.2]])
        proba_threshold = 0.05
        bg_index = 0
        result = utils_object_detectors.get_valid_fm_boxes_from_proba(probas, proba_threshold, bg_index)
        expected_result = [(0, 1, 0.2)]
        self.assertEqual(result, expected_result)

    def test33_get_valid_boxes_from_coordinates(self):
        '''Test of the function utils_object_detectors.get_valid_boxes_from_coordinates'''
        input_img = np.zeros(shape=(20, 20))
        input_rois = np.array([[5, 5, 5, 5]])
        fm_boxes_candidates = [(0, 0, 0.7)]
        regr_coordinates = np.array([[0, 0, 0, 0]])
        classifier_regr_scaling = [1, 1, 1, 1]
        subsampling_ratio = 1
        dict_classes = {0: "ok"}

        result = utils_object_detectors.get_valid_boxes_from_coordinates(input_img, input_rois, fm_boxes_candidates,
                                                                         regr_coordinates, classifier_regr_scaling,
                                                                         subsampling_ratio, dict_classes)
        expected_result = [('ok', 0.7, (5.0, 5.0, 10.0, 10.0))]
        self.assertEqual(result, expected_result)

    def test34_non_max_suppression_fast_on_preds(self):
        '''Test of the function utils_object_detectors.non_max_suppression_fast_on_preds'''
        boxes_candidates = [('ok', 0.7, (5.0, 5.0, 10.0, 10.0)), ('ok', 0.6, (6.0, 6.0, 10.0, 10.0))]
        nms_overlap_threshold = 0.4

        result = utils_object_detectors.non_max_suppression_fast_on_preds(boxes_candidates, nms_overlap_threshold)
        expected_result = [('ok', 0.7, np.array([ 5.,  5., 10., 10.]))]

        for (c, p, bbox), (expected_c, expected_p, expected_bbox) in zip(result, expected_result):
            self.assertEqual(c, expected_c)
            self.assertEqual(p, expected_p)
            np.testing.assert_almost_equal(bbox, expected_bbox)

    def test35_get_final_bboxes(self):
        '''Test of the function utils_object_detectors.get_final_bboxes'''
        final_boxes = [('ok', 0.7, np.array([ 5.,  5., 10., 10.]))]
        img_data = {
            "resized_width": 1,
            "resized_height": 1,
            "original_width": 2,
            "original_height": 2
        }

        result = utils_object_detectors.get_final_bboxes(final_boxes, img_data)
        expected_result = [{'class': 'ok', 'proba': 0.7, 'x1': 10, 'y1': 10, 'x2': 20, 'y2': 20}]
        self.assertEqual(result, expected_result)

# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
