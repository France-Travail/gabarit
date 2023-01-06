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

from {{package_name}}.models_training.object_detectors import utils_faster_rcnn

# Disable logging
import logging
logging.disable(logging.CRITICAL)

class UtilsFasterRCNNTests(unittest.TestCase):
    '''Main class to test all functions in utils_faster_rcnn.py'''
    
    @classmethod
    def setUpClass(cls):
        '''SetUp fonction'''
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))

    def test01_RoiPoolingLayer(self):
        '''Test of the class utils_faster_rcnn.RoiPoolingLayer'''
        layer = utils_faster_rcnn.RoiPoolingLayer(2)

        features_maps = np.zeros(shape=(1, 10, 10, 1)) # Shape (batch_size, cols, rows, channels)
        rois = np.array([[[0, 0, 2, 2], [2, 2, 3, 3]]])

        result = layer([features_maps, rois])
        self.assertTrue(result.shape, (1, 2, 2, 2, 1))

    def test02_get_rpn_loss_regr(self):
        '''Test of the function utils_faster_rcnn.get_rpn_loss_regr'''
        loss = utils_faster_rcnn.get_rpn_loss_regr(1)

        y_true = np.ones(shape=(1, 10, 10, 2*4*1))
        y_pred = np.zeros(shape=(1, 10, 10, 4*1))

        result = loss(y_true, y_pred)
        self.assertAlmostEqual(result, 0.5, delta=0.1)

    def test02_get_rpn_loss_cls(self):
        '''Test of the function utils_faster_rcnn.get_rpn_loss_cls'''
        loss = utils_faster_rcnn.get_rpn_loss_cls(1)

        y_true = np.ones(shape=(1, 10, 10, 2*1))
        y_pred = np.zeros(shape=(1, 10, 10, 1))

        result = loss(y_true, y_pred)
        self.assertAlmostEqual(result.numpy(), 15.4, delta=0.1)

    def test03_get_class_loss_regr(self):
        '''Test of the function utils_faster_rcnn.get_class_loss_regr'''
        loss = utils_faster_rcnn.get_class_loss_regr(1)

        y_true = np.ones(shape=(1, 2, 2*4*1))
        y_pred = np.zeros(shape=(1, 2, 4*1))

        result = loss(y_true, y_pred)
        self.assertAlmostEqual(result, 0.5, delta=0.1)

    def test04_class_loss_cls(self):
        '''Test of the function utils_faster_rcnn.class_loss_cls'''
        y_true = [[0, 1, 0], [0, 0, 1]]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

        result = utils_faster_rcnn.class_loss_cls(y_true, y_pred)
        self.assertAlmostEqual(result.numpy(), 1.17, delta=0.1)

    def test05_get_custom_objects_faster_rcnn(self):
        '''Test of the function utils_faster_rcnn.get_custom_objects_faster_rcnn'''
        result = utils_faster_rcnn.get_custom_objects_faster_rcnn(1, 2)
        for key in ("RoiPoolingLayer", "rpn_loss_regr", "rpn_loss_cls", "class_loss_cls", "class_loss_regr"):
            self.assertIn(key, result)