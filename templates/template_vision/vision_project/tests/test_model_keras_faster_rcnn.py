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
import io
import cv2
import copy
import math
import json
import psutil
import shutil
import numpy as np
import pandas as pd
from PIL import Image

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.monitoring.model_logger import ModelLogger
from {{package_name}}.models_training.object_detectors import utils_faster_rcnn
from {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn import (ModelKerasFasterRcnnObjectDetector,
                                                                                       ModelCheckpointAll,
                                                                                       CustomGeneratorRpn,
                                                                                       CustomGeneratorClassifier)

import tensorflow as tf
from tensorflow.keras.layers import Input


# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)
    tf.keras.backend.clear_session()


# Mocks

def download_url_crash(x, y):
    raise ConnectionError("error")


class MockCustomGeneratorRpn():
    def __init__(self, img_data_list: list, batch_size: int, shuffle: bool, seed: int, model, **kwargs) -> None:
        self.n = len(img_data_list)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.img_data_list = img_data_list
        self.model = model
        # Default arguments
        self.horizontal_flip = kwargs.get('horizontal_flip', False)
        self.vertical_flip = kwargs.get('vertical_flip', False)
        self.rot_90 = kwargs.get('rot_90', False)
        self.data_type = kwargs.get('data_type', 'train')
        self.with_img_data = kwargs.get('with_img_data', False)

class MockCustomGeneratorClassifier(MockCustomGeneratorRpn):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Default arguments
        self.shared_model_trainable = kwargs.get('shared_model_trainable', False)

class MockModelKerasFasterRcnnObjectDetector():
    def __init__(self) -> None:
        self.anchor_box_sizes = [64, 128, 256]  # In the paper : [128, 256, 512]
        self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]  # In the paper : [1, 1], [1, 2], [2, 1]]
        self.nb_anchors = len(self.anchor_box_sizes) * len(self.anchor_box_ratios)
        self.list_anchors = [[anchor_size * anchor_ratio[0], anchor_size * anchor_ratio[1]]
                             for anchor_size in self.anchor_box_sizes for anchor_ratio in self.anchor_box_ratios]
        self.shared_model_subsampling = 16
        self.rpn_min_overlap = 0.2
        self.rpn_max_overlap = 0.8
        self.rpn_regr_scaling = 4
        self.rpn_restrict_num_regions = 100
        self.logger = logging.getLogger(__name__)
        self.preprocess_input = self._get_preprocess_input()
        # Manage RPN model
        class FakeModelRpn():
            def __init__(self) -> None:
                pass
            def set_weights(self, model):
                pass
            def get_weights(self):
                return None
        self.model_rpn = FakeModelRpn()
    def _get_preprocess_input(self):
        def preprocess_input(img):
            return img
        return preprocess_input
    def _generate_images_with_bboxes(self, img_data: dict, horizontal_flip: bool = False,
                                     vertical_flip: bool = False, rot_90: bool = False):
        with open(img_data['file_path'], 'rb') as f:
            img = Image.open(io.BytesIO(f.read()))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.asarray(img)
        bboxes = copy.deepcopy(img_data.get('bboxes', []))  # Empty if test
        original_height, original_width = img.shape[0], img.shape[1]
        img = self.preprocess_input(img)
        resized_height, resized_width = img.shape[0], img.shape[1]
        for bbox in bboxes:
            bbox['x1'] = bbox['x1'] * (resized_width / original_width)
            bbox['x2'] = bbox['x2'] * (resized_width / original_width)
            bbox['y1'] = bbox['y1'] * (resized_height / original_height)
            bbox['y2'] = bbox['y2'] * (resized_height / original_height)
        prepared_data = {'img': img, 'bboxes': bboxes, 'original_height': original_height,
                         'original_width': original_width, 'resized_height': resized_height,
                         'resized_width': resized_width}
        return prepared_data

def mock_get_rpn_targets(model, img_data_batch: list):
    return 'toto', 'titi'

def mock_clone_model_fn(model, input_tensors=None, clone_function=None):
    return model

# Tests

class ModelKerasFasterRcnnObjectDetectorTests(unittest.TestCase):
    '''Main class to test model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector'''


    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test01_model_keras_faster_rcnn_init(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.model_type, 'object_detector')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model, None)
        self.assertEqual(model.shared_model, None)
        self.assertEqual(model.model_rpn, None)
        self.assertEqual(model.model_classifier, None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, epochs=8)
        self.assertEqual(model.epochs, 8)
        self.assertEqual(model.epochs_rpn_trainable_true, 8)
        self.assertEqual(model.epochs_classifier_trainable_true, 8)
        self.assertEqual(model.epochs_rpn_trainable_false, 8)
        self.assertEqual(model.epochs_classifier_trainable_false, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, epochs=8, epochs_rpn_trainable_true=2,
                                                   epochs_classifier_trainable_true=3, epochs_rpn_trainable_false=4,
                                                   epochs_classifier_trainable_false=5)
        self.assertEqual(model.epochs, 8)
        self.assertEqual(model.epochs_rpn_trainable_true, 2)
        self.assertEqual(model.epochs_classifier_trainable_true, 3)
        self.assertEqual(model.epochs_rpn_trainable_false, 4)
        self.assertEqual(model.epochs_classifier_trainable_false, 5)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, batch_size=8)
        self.assertEqual(model.batch_size, 8)
        self.assertEqual(model.batch_size_rpn_trainable_true, 8)
        self.assertEqual(model.batch_size_classifier_trainable_true, 8)
        self.assertEqual(model.batch_size_rpn_trainable_false, 8)
        self.assertEqual(model.batch_size_classifier_trainable_false, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, batch_size=8, batch_size_rpn_trainable_true=2,
                                                   batch_size_classifier_trainable_true=3, batch_size_rpn_trainable_false=4,
                                                   batch_size_classifier_trainable_false=5)
        self.assertEqual(model.batch_size, 8)
        self.assertEqual(model.batch_size_rpn_trainable_true, 2)
        self.assertEqual(model.batch_size_classifier_trainable_true, 3)
        self.assertEqual(model.batch_size_rpn_trainable_false, 4)
        self.assertEqual(model.batch_size_classifier_trainable_false, 5)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, patience=8)
        self.assertEqual(model.patience, 8)
        self.assertEqual(model.patience_rpn_trainable_true, 8)
        self.assertEqual(model.patience_classifier_trainable_true, 8)
        self.assertEqual(model.patience_rpn_trainable_false, 8)
        self.assertEqual(model.patience_classifier_trainable_false, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, patience=8, patience_rpn_trainable_true=2,
                                                   patience_classifier_trainable_true=3, patience_rpn_trainable_false=4,
                                                   patience_classifier_trainable_false=5)
        self.assertEqual(model.patience, 8)
        self.assertEqual(model.patience_rpn_trainable_true, 2)
        self.assertEqual(model.patience_classifier_trainable_true, 3)
        self.assertEqual(model.patience_rpn_trainable_false, 4)
        self.assertEqual(model.patience_classifier_trainable_false, 5)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, lr_rpn_trainable_true=0.2, lr_classifier_trainable_true=0.3,
                                                   lr_rpn_trainable_false=0.4, lr_classifier_trainable_false=0.5)
        self.assertEqual(model.keras_params['lr_rpn_trainable_true'], 0.2)
        self.assertEqual(model.keras_params['lr_classifier_trainable_true'], 0.3)
        self.assertEqual(model.keras_params['lr_rpn_trainable_false'], 0.4)
        self.assertEqual(model.keras_params['lr_classifier_trainable_false'], 0.5)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=64)
        self.assertEqual(model.img_min_side_size, 64)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=0.8, rpn_max_overlap=0.9)
        self.assertEqual(model.rpn_min_overlap, 0.8)
        self.assertEqual(model.rpn_max_overlap, 0.9)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_restrict_num_regions=8)
        self.assertEqual(model.rpn_restrict_num_regions, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pool_resize_classifier=8)
        self.assertEqual(model.pool_resize_classifier, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, nb_rois_classifier=8)
        self.assertEqual(model.nb_rois_classifier, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, roi_nms_overlap_threshold=0.8)
        self.assertEqual(model.roi_nms_overlap_threshold, 0.8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, nms_max_boxes=8)
        self.assertEqual(model.nms_max_boxes, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, classifier_min_overlap=0.8, classifier_max_overlap=0.9)
        self.assertEqual(model.classifier_min_overlap, 0.8)
        self.assertEqual(model.classifier_max_overlap, 0.9)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pred_bbox_proba_threshold=0.8)
        self.assertEqual(model.pred_bbox_proba_threshold, 0.8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pred_nms_overlap_threshold=0.8)
        self.assertEqual(model.pred_nms_overlap_threshold, 0.8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params={'toto': 'titi'})
        self.assertEqual(model.data_augmentation_params, {'toto': 'titi'})
        remove_dir(model_dir)

        # Check errors
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=31)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_max_overlap=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_max_overlap=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=0.3, rpn_max_overlap=0.1)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_restrict_num_regions=0)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pool_resize_classifier=0)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, nb_rois_classifier=0)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, roi_nms_overlap_threshold=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, roi_nms_overlap_threshold=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, nms_max_boxes=0)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, classifier_min_overlap=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, classifier_min_overlap=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, classifier_max_overlap=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, classifier_max_overlap=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, classifier_min_overlap=0.3, classifier_max_overlap=0.1)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pred_bbox_proba_threshold=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pred_bbox_proba_threshold=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pred_nms_overlap_threshold=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pred_nms_overlap_threshold=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, color_mode='rgba')
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, color_mode='grayscale')

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    @patch('{{package_name}}.utils.download_url', side_effect=download_url_crash)
    def test02_model_keras_faster_rcnn_init_offline(self, mock_download_url):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector.__init__
        - No access to a base model
        '''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Clean cache path if exists
        cache_path = os.path.join(utils.get_data_path(), 'transfer_learning_weights')
        if os.path.isdir(cache_path):
            shutil.rmtree(cache_path)

        # Init., test all parameters
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.model_type, 'object_detector')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model, None)
        self.assertEqual(model.shared_model, None)
        self.assertEqual(model.model_rpn, None)
        self.assertEqual(model.model_classifier, None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, epochs=8)
        self.assertEqual(model.epochs, 8)
        self.assertEqual(model.epochs_rpn_trainable_true, 8)
        self.assertEqual(model.epochs_classifier_trainable_true, 8)
        self.assertEqual(model.epochs_rpn_trainable_false, 8)
        self.assertEqual(model.epochs_classifier_trainable_false, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, epochs=8, epochs_rpn_trainable_true=2,
                                                   epochs_classifier_trainable_true=3, epochs_rpn_trainable_false=4,
                                                   epochs_classifier_trainable_false=5)
        self.assertEqual(model.epochs, 8)
        self.assertEqual(model.epochs_rpn_trainable_true, 2)
        self.assertEqual(model.epochs_classifier_trainable_true, 3)
        self.assertEqual(model.epochs_rpn_trainable_false, 4)
        self.assertEqual(model.epochs_classifier_trainable_false, 5)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, batch_size=8)
        self.assertEqual(model.batch_size, 8)
        self.assertEqual(model.batch_size_rpn_trainable_true, 8)
        self.assertEqual(model.batch_size_classifier_trainable_true, 8)
        self.assertEqual(model.batch_size_rpn_trainable_false, 8)
        self.assertEqual(model.batch_size_classifier_trainable_false, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, batch_size=8, batch_size_rpn_trainable_true=2,
                                                   batch_size_classifier_trainable_true=3, batch_size_rpn_trainable_false=4,
                                                   batch_size_classifier_trainable_false=5)
        self.assertEqual(model.batch_size, 8)
        self.assertEqual(model.batch_size_rpn_trainable_true, 2)
        self.assertEqual(model.batch_size_classifier_trainable_true, 3)
        self.assertEqual(model.batch_size_rpn_trainable_false, 4)
        self.assertEqual(model.batch_size_classifier_trainable_false, 5)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, patience=8)
        self.assertEqual(model.patience, 8)
        self.assertEqual(model.patience_rpn_trainable_true, 8)
        self.assertEqual(model.patience_classifier_trainable_true, 8)
        self.assertEqual(model.patience_rpn_trainable_false, 8)
        self.assertEqual(model.patience_classifier_trainable_false, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, patience=8, patience_rpn_trainable_true=2,
                                                   patience_classifier_trainable_true=3, patience_rpn_trainable_false=4,
                                                   patience_classifier_trainable_false=5)
        self.assertEqual(model.patience, 8)
        self.assertEqual(model.patience_rpn_trainable_true, 2)
        self.assertEqual(model.patience_classifier_trainable_true, 3)
        self.assertEqual(model.patience_rpn_trainable_false, 4)
        self.assertEqual(model.patience_classifier_trainable_false, 5)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, lr_rpn_trainable_true=0.2, lr_classifier_trainable_true=0.3,
                                                   lr_rpn_trainable_false=0.4, lr_classifier_trainable_false=0.5)
        self.assertEqual(model.keras_params['lr_rpn_trainable_true'], 0.2)
        self.assertEqual(model.keras_params['lr_classifier_trainable_true'], 0.3)
        self.assertEqual(model.keras_params['lr_rpn_trainable_false'], 0.4)
        self.assertEqual(model.keras_params['lr_classifier_trainable_false'], 0.5)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=64)
        self.assertEqual(model.img_min_side_size, 64)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=0.8, rpn_max_overlap=0.9)
        self.assertEqual(model.rpn_min_overlap, 0.8)
        self.assertEqual(model.rpn_max_overlap, 0.9)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_restrict_num_regions=8)
        self.assertEqual(model.rpn_restrict_num_regions, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pool_resize_classifier=8)
        self.assertEqual(model.pool_resize_classifier, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, nb_rois_classifier=8)
        self.assertEqual(model.nb_rois_classifier, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, roi_nms_overlap_threshold=0.8)
        self.assertEqual(model.roi_nms_overlap_threshold, 0.8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, nms_max_boxes=8)
        self.assertEqual(model.nms_max_boxes, 8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, classifier_min_overlap=0.8, classifier_max_overlap=0.9)
        self.assertEqual(model.classifier_min_overlap, 0.8)
        self.assertEqual(model.classifier_max_overlap, 0.9)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pred_bbox_proba_threshold=0.8)
        self.assertEqual(model.pred_bbox_proba_threshold, 0.8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pred_nms_overlap_threshold=0.8)
        self.assertEqual(model.pred_nms_overlap_threshold, 0.8)
        remove_dir(model_dir)

        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params={'toto': 'titi'})
        self.assertEqual(model.data_augmentation_params, {'toto': 'titi'})
        remove_dir(model_dir)

        # Check errors
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=31)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_max_overlap=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_max_overlap=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=0.3, rpn_max_overlap=0.1)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, rpn_restrict_num_regions=0)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pool_resize_classifier=0)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, nb_rois_classifier=0)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, roi_nms_overlap_threshold=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, roi_nms_overlap_threshold=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, nms_max_boxes=0)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, classifier_min_overlap=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, classifier_min_overlap=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, classifier_max_overlap=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, classifier_max_overlap=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, classifier_min_overlap=0.3, classifier_max_overlap=0.1)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pred_bbox_proba_threshold=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pred_bbox_proba_threshold=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pred_nms_overlap_threshold=-0.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, pred_nms_overlap_threshold=1.01)
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, color_mode='rgba')
        with self.assertRaises(ValueError):
            ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, color_mode='grayscale')

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test03_model_keras_faster_rcnn_get_model(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._get_model'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        model.custom_objects = {**utils_faster_rcnn.get_custom_objects_faster_rcnn(model.nb_anchors, len(model.list_classes)), **model.custom_objects}
        model._get_model()  # Nothing to test in particular ...

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test04_model_keras_faster_rcnn_compile_model_rpn(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._compile_model_rpn'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, lr_rpn_trainable_true=0.1)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        model.custom_objects = {**utils_faster_rcnn.get_custom_objects_faster_rcnn(model.nb_anchors, len(model.list_classes)), **model.custom_objects}
        model.shared_model, model.model_rpn, model.model_classifier, model.model = model._get_model()
        self.assertAlmostEqual(0.1, float(model.model_rpn.optimizer.lr.numpy()))
        model._compile_model_rpn(model.model_rpn, 0.1234)
        self.assertAlmostEqual(0.1234, float(model.model_rpn.optimizer.lr.numpy()))

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test05_model_keras_faster_rcnn_compile_model_classifier(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._compile_model_classifier'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, lr_classifier_trainable_true=0.1)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        model.custom_objects = {**utils_faster_rcnn.get_custom_objects_faster_rcnn(model.nb_anchors, len(model.list_classes)), **model.custom_objects}
        model.shared_model, model.model_rpn, model.model_classifier, model.model = model._get_model()
        self.assertAlmostEqual(0.1, float(model.model_classifier.optimizer.lr.numpy()))
        model._compile_model_classifier(model.model_classifier, 0.1234)
        self.assertAlmostEqual(0.1234, float(model.model_classifier.optimizer.lr.numpy()))

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test06_model_keras_faster_rcnn_get_shared_model_structure(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._get_shared_model_structure'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        input_img = Input(shape=(None, None, 3), name='input_img')
        model._get_shared_model_structure(input_img)  # Nothing to test in particular ...

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test07_model_keras_faster_rcnn_add_rpn_layers(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._add_rpn_layers'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        input_img = Input(shape=(None, None, 3), name='input_img')
        shared_model_layers = model._get_shared_model_structure(input_img)
        model._add_rpn_layers(shared_model_layers)  # Nothing to test in particular ...

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test08_model_keras_faster_rcnn_add_classifier_layers(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._add_classifier_layers'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        model.custom_objects = {**utils_faster_rcnn.get_custom_objects_faster_rcnn(model.nb_anchors, len(model.list_classes)), **model.custom_objects}
        input_img = Input(shape=(None, None, 3), name='input_img')
        input_rois = Input(shape=(None, model.nb_rois_classifier), name='input_rois')
        shared_model_layers = model._get_shared_model_structure(input_img)
        rpn_layers = model._add_rpn_layers(shared_model_layers)
        model._add_classifier_layers(shared_model_layers, input_rois)  # Nothing to test in particular ...

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    @patch('{{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.CustomGeneratorClassifier', side_effect=MockCustomGeneratorClassifier)
    @patch('{{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.CustomGeneratorRpn', side_effect=MockCustomGeneratorRpn)
    def test09_model_keras_faster_rcnn_get_generator(self, mock_custom_generator_rpn, mock_custom_generator_classifier):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._get_generator'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars (no need to put the real data, the generator is mocked)
        df = pd.DataFrame({
            'file_path': ['a', 'c', 'e', 'b'],
            'bboxes': [[[1, 2], [4, 2]], [[1, 5]], [[3, 8]], [[5, 2], [4, 3]]],
        })

        # Nominal case - RPN
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params={'horizontal_flip': True, 'vertical_flip': True, 'rot_90': True})
        generator = model._get_generator(df, data_type='train', batch_size=5, generator_type='rpn', with_img_data=False)
        self.assertEqual(generator.n, df.shape[0])
        self.assertEqual(generator.batch_size, 5)
        self.assertEqual(generator.shuffle, True)
        self.assertEqual(generator.seed, None)
        self.assertEqual(len(generator.img_data_list), 4)
        self.assertEqual(generator.img_data_list[0]['bboxes'], [[1, 2], [4, 2]])
        self.assertEqual(model.model_name, generator.model.model_name)
        self.assertEqual(generator.horizontal_flip, True)
        self.assertEqual(generator.vertical_flip, True)
        self.assertEqual(generator.rot_90, True)
        self.assertEqual(generator.data_type, 'train')
        self.assertEqual(generator.with_img_data, False)
        remove_dir(model_dir)
        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params={'horizontal_flip': False, 'vertical_flip': False, 'rot_90': False})
        generator = model._get_generator(df, data_type='train', batch_size=5, generator_type='rpn', with_img_data=False)
        self.assertEqual(generator.n, df.shape[0])
        self.assertEqual(generator.batch_size, 5)
        self.assertEqual(generator.shuffle, True)
        self.assertEqual(generator.seed, None)
        self.assertEqual(len(generator.img_data_list), 4)
        self.assertEqual(generator.img_data_list[0]['bboxes'], [[1, 2], [4, 2]])
        self.assertEqual(model.model_name, generator.model.model_name)
        self.assertEqual(generator.horizontal_flip, False)
        self.assertEqual(generator.vertical_flip, False)
        self.assertEqual(generator.rot_90, False)
        self.assertEqual(generator.data_type, 'train')
        self.assertEqual(generator.with_img_data, False)
        remove_dir(model_dir)
        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params={'horizontal_flip': True, 'vertical_flip': True, 'rot_90': True})
        generator = model._get_generator(df, data_type='valid', batch_size=10, generator_type='rpn', with_img_data=True)
        self.assertEqual(generator.n, df.shape[0])
        self.assertEqual(generator.batch_size, 10)
        self.assertEqual(generator.shuffle, False)
        self.assertEqual(generator.seed, None)
        self.assertEqual(len(generator.img_data_list), 4)
        self.assertEqual(generator.img_data_list[0]['bboxes'], [[1, 2], [4, 2]])
        self.assertEqual(model.model_name, generator.model.model_name)
        self.assertEqual(generator.horizontal_flip, False)
        self.assertEqual(generator.vertical_flip, False)
        self.assertEqual(generator.rot_90, False)
        self.assertEqual(generator.data_type, 'valid')
        self.assertEqual(generator.with_img_data, True)
        remove_dir(model_dir)
        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params={'horizontal_flip': True, 'vertical_flip': True, 'rot_90': True})
        generator = model._get_generator(df, data_type='test', batch_size=5, generator_type='rpn', with_img_data=False)
        self.assertEqual(generator.n, df.shape[0])
        self.assertEqual(generator.batch_size, 5)
        self.assertEqual(generator.shuffle, False)
        self.assertEqual(generator.seed, None)
        self.assertEqual(len(generator.img_data_list), 4)
        self.assertEqual(generator.img_data_list[0]['bboxes'], [[1, 2], [4, 2]])
        self.assertEqual(model.model_name, generator.model.model_name)
        self.assertEqual(generator.horizontal_flip, False)
        self.assertEqual(generator.vertical_flip, False)
        self.assertEqual(generator.rot_90, False)
        self.assertEqual(generator.data_type, 'test')
        self.assertEqual(generator.with_img_data, False)
        remove_dir(model_dir)
        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params={'horizontal_flip': True, 'vertical_flip': True, 'rot_90': True})
        generator = model._get_generator(df[['file_path']], data_type='test', batch_size=5, generator_type='rpn', with_img_data=False)
        self.assertEqual(generator.n, df.shape[0])
        self.assertEqual(generator.batch_size, 5)
        self.assertEqual(generator.shuffle, False)
        self.assertEqual(generator.seed, None)
        self.assertEqual(len(generator.img_data_list), 4)
        self.assertEqual(model.model_name, generator.model.model_name)
        self.assertEqual(generator.horizontal_flip, False)
        self.assertEqual(generator.vertical_flip, False)
        self.assertEqual(generator.rot_90, False)
        self.assertEqual(generator.data_type, 'test')
        self.assertEqual(generator.with_img_data, False)
        remove_dir(model_dir)

        # Nominal case - Classifier
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params={'horizontal_flip': True, 'vertical_flip': True, 'rot_90': True})
        generator = model._get_generator(df, data_type='train', batch_size=5, generator_type='classifier', with_img_data=False, shared_model_trainable=True)
        self.assertEqual(generator.n, df.shape[0])
        self.assertEqual(generator.batch_size, 5)
        self.assertEqual(generator.shuffle, True)
        self.assertEqual(generator.seed, None)
        self.assertEqual(len(generator.img_data_list), 4)
        self.assertEqual(generator.img_data_list[0]['bboxes'], [[1, 2], [4, 2]])
        self.assertEqual(model.model_name, generator.model.model_name)
        self.assertEqual(generator.horizontal_flip, True)
        self.assertEqual(generator.vertical_flip, True)
        self.assertEqual(generator.rot_90, True)
        self.assertEqual(generator.data_type, 'train')
        self.assertEqual(generator.with_img_data, False)
        self.assertEqual(generator.shared_model_trainable, True)
        remove_dir(model_dir)
        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params={'horizontal_flip': False, 'vertical_flip': False, 'rot_90': False})
        generator = model._get_generator(df, data_type='train', batch_size=5, generator_type='classifier', with_img_data=False, shared_model_trainable=True)
        self.assertEqual(generator.n, df.shape[0])
        self.assertEqual(generator.batch_size, 5)
        self.assertEqual(generator.shuffle, True)
        self.assertEqual(generator.seed, None)
        self.assertEqual(len(generator.img_data_list), 4)
        self.assertEqual(generator.img_data_list[0]['bboxes'], [[1, 2], [4, 2]])
        self.assertEqual(model.model_name, generator.model.model_name)
        self.assertEqual(generator.horizontal_flip, False)
        self.assertEqual(generator.vertical_flip, False)
        self.assertEqual(generator.rot_90, False)
        self.assertEqual(generator.data_type, 'train')
        self.assertEqual(generator.with_img_data, False)
        self.assertEqual(generator.shared_model_trainable, True)
        remove_dir(model_dir)
        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params={'horizontal_flip': True, 'vertical_flip': True, 'rot_90': True})
        generator = model._get_generator(df, data_type='valid', batch_size=10, generator_type='classifier', with_img_data=True, shared_model_trainable=False)
        self.assertEqual(generator.n, df.shape[0])
        self.assertEqual(generator.batch_size, 10)
        self.assertEqual(generator.shuffle, False)
        self.assertEqual(generator.seed, None)
        self.assertEqual(len(generator.img_data_list), 4)
        self.assertEqual(generator.img_data_list[0]['bboxes'], [[1, 2], [4, 2]])
        self.assertEqual(model.model_name, generator.model.model_name)
        self.assertEqual(generator.horizontal_flip, False)
        self.assertEqual(generator.vertical_flip, False)
        self.assertEqual(generator.rot_90, False)
        self.assertEqual(generator.data_type, 'valid')
        self.assertEqual(generator.with_img_data, True)
        self.assertEqual(generator.shared_model_trainable, False)
        remove_dir(model_dir)
        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params={'horizontal_flip': True, 'vertical_flip': True, 'rot_90': True})
        generator = model._get_generator(df, data_type='test', batch_size=5, generator_type='classifier', with_img_data=False, shared_model_trainable=True)
        self.assertEqual(generator.n, df.shape[0])
        self.assertEqual(generator.batch_size, 5)
        self.assertEqual(generator.shuffle, False)
        self.assertEqual(generator.seed, None)
        self.assertEqual(len(generator.img_data_list), 4)
        self.assertEqual(generator.img_data_list[0]['bboxes'], [[1, 2], [4, 2]])
        self.assertEqual(model.model_name, generator.model.model_name)
        self.assertEqual(generator.horizontal_flip, False)
        self.assertEqual(generator.vertical_flip, False)
        self.assertEqual(generator.rot_90, False)
        self.assertEqual(generator.data_type, 'test')
        self.assertEqual(generator.with_img_data, False)
        self.assertEqual(generator.shared_model_trainable, True)
        remove_dir(model_dir)
        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params={'horizontal_flip': True, 'vertical_flip': True, 'rot_90': True})
        generator = model._get_generator(df[['file_path']], data_type='test', batch_size=5, generator_type='classifier', with_img_data=False, shared_model_trainable=True)
        self.assertEqual(generator.n, df.shape[0])
        self.assertEqual(generator.batch_size, 5)
        self.assertEqual(generator.shuffle, False)
        self.assertEqual(generator.seed, None)
        self.assertEqual(len(generator.img_data_list), 4)
        self.assertEqual(model.model_name, generator.model.model_name)
        self.assertEqual(generator.horizontal_flip, False)
        self.assertEqual(generator.vertical_flip, False)
        self.assertEqual(generator.rot_90, False)
        self.assertEqual(generator.data_type, 'test')
        self.assertEqual(generator.with_img_data, False)
        self.assertEqual(generator.shared_model_trainable, True)
        remove_dir(model_dir)

        # Manage errors
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        with self.assertRaises(ValueError):
            generator = model._get_generator(df, data_type='toto', batch_size=5, generator_type='rpn')
        with self.assertRaises(ValueError):
            generator = model._get_generator(df, data_type='train', batch_size=5, generator_type='toto')
        with self.assertRaises(ValueError):
            generator = model._get_generator(pd.DataFrame(columns=['toto']), data_type='train', batch_size=5, generator_type='rpn')
        with self.assertRaises(ValueError):
            generator = model._get_generator(pd.DataFrame(columns=['file_path']), data_type='train', batch_size=5, generator_type='rpn')
        with self.assertRaises(ValueError):
            generator = model._get_generator(pd.DataFrame(columns=['file_path']), data_type='valid', batch_size=5, generator_type='rpn')
        with self.assertRaises(ValueError):
            model.model_type = 'toto'
            generator = model._get_generator(df, data_type='train', batch_size=5, generator_type='rpn')

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    @patch('numpy.random.choice')
    def test10_model_keras_faster_rcnn_generate_images_with_bboxes(self, mock_np_random_choice):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._generate_images_with_bboxes'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        im_path = os.path.join(os.getcwd(), 'test_data', 'apple_36.jpg')
        img_data = {'file_path': im_path, 'bboxes': [{'x1': 10, 'x2': 30, 'y1': 10, 'y2': 50}, {'x1': 5, 'x2': 25, 'y1': 10, 'y2': 45}]}
        img_data_simple = {'file_path': im_path}
        with Image.open(im_path) as im:
            w, h = im.size
        resize_w, resize_h = (250, 300)
        # Mock preprocess
        def fake_preprocess(img):
            img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
            return img

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        model.preprocess_input = fake_preprocess
        prepared_data = model._generate_images_with_bboxes(img_data, horizontal_flip=False, vertical_flip=False, rot_90=False)
        self.assertEqual(prepared_data['img'].shape[0], resize_h)
        self.assertEqual(prepared_data['img'].shape[1], resize_w)
        self.assertEqual(len(prepared_data['bboxes']), len(img_data['bboxes']))
        self.assertNotEqual(prepared_data['bboxes'][0]['x1'], img_data['bboxes'][0]['x1'])
        self.assertNotEqual(prepared_data['bboxes'][0]['y1'], img_data['bboxes'][0]['y1'])
        self.assertEqual(prepared_data['original_height'], h)
        self.assertEqual(prepared_data['original_width'], w)
        self.assertEqual(prepared_data['resized_height'], resize_h)
        self.assertEqual(prepared_data['resized_width'], resize_w)
        # Without bbox
        prepared_data = model._generate_images_with_bboxes(img_data_simple, horizontal_flip=False, vertical_flip=False, rot_90=False)
        self.assertEqual(prepared_data['img'].shape[0], resize_h)
        self.assertEqual(prepared_data['img'].shape[1], resize_w)
        self.assertEqual(prepared_data['bboxes'], [])
        self.assertEqual(prepared_data['original_height'], h)
        self.assertEqual(prepared_data['original_width'], w)
        self.assertEqual(prepared_data['resized_height'], resize_h)
        self.assertEqual(prepared_data['resized_width'], resize_w)
        remove_dir(model_dir)

        # Tests augmentation
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        model.preprocess_input = fake_preprocess
        prepared_data = model._generate_images_with_bboxes(img_data, horizontal_flip=True, vertical_flip=False, rot_90=False)
        self.assertEqual(prepared_data['img'].shape[0], resize_h)
        self.assertEqual(prepared_data['img'].shape[1], resize_w)
        self.assertEqual(len(prepared_data['bboxes']), len(img_data['bboxes']))
        self.assertNotEqual(prepared_data['bboxes'][0]['x1'], img_data['bboxes'][0]['x1'])
        self.assertNotEqual(prepared_data['bboxes'][0]['y1'], img_data['bboxes'][0]['y1'])
        self.assertEqual(prepared_data['original_height'], h)
        self.assertEqual(prepared_data['original_width'], w)
        self.assertEqual(prepared_data['resized_height'], resize_h)
        self.assertEqual(prepared_data['resized_width'], resize_w)
        #
        prepared_data = model._generate_images_with_bboxes(img_data_simple, horizontal_flip=False, vertical_flip=True, rot_90=False)
        self.assertEqual(prepared_data['img'].shape[0], resize_h)
        self.assertEqual(prepared_data['img'].shape[1], resize_w)
        self.assertEqual(prepared_data['bboxes'], [])
        self.assertEqual(prepared_data['original_height'], h)
        self.assertEqual(prepared_data['original_width'], w)
        self.assertEqual(prepared_data['resized_height'], resize_h)
        self.assertEqual(prepared_data['resized_width'], resize_w)
        remove_dir(model_dir)
        #
        mock_np_random_choice.return_value = [90]
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        model.preprocess_input = fake_preprocess
        prepared_data = model._generate_images_with_bboxes(img_data, horizontal_flip=False, vertical_flip=False, rot_90=True)
        self.assertEqual(prepared_data['img'].shape[0], resize_h)
        self.assertEqual(prepared_data['img'].shape[1], resize_w)
        self.assertEqual(len(prepared_data['bboxes']), len(img_data['bboxes']))
        self.assertNotEqual(prepared_data['bboxes'][0]['x1'], img_data['bboxes'][0]['x1'])
        self.assertNotEqual(prepared_data['bboxes'][0]['y1'], img_data['bboxes'][0]['y1'])
        self.assertEqual(prepared_data['original_height'], w)  # Rot 90
        self.assertEqual(prepared_data['original_width'], h)  # Rot 90
        self.assertEqual(prepared_data['resized_height'], resize_h)
        self.assertEqual(prepared_data['resized_width'], resize_w)
        #
        mock_np_random_choice.return_value = [180]
        prepared_data = model._generate_images_with_bboxes(img_data_simple, horizontal_flip=False, vertical_flip=False, rot_90=True)
        self.assertEqual(prepared_data['img'].shape[0], resize_h)
        self.assertEqual(prepared_data['img'].shape[1], resize_w)
        self.assertEqual(prepared_data['bboxes'], [])
        self.assertEqual(prepared_data['original_height'], h)  # Rot 180
        self.assertEqual(prepared_data['original_width'], w)  # Rot 180
        self.assertEqual(prepared_data['resized_height'], resize_h)
        self.assertEqual(prepared_data['resized_width'], resize_w)
        remove_dir(model_dir)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test11_model_keras_faster_rcnn_get_preprocess_input(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._get_preprocess_input'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        im_path = os.path.join(os.getcwd(), 'test_data', 'apple_36.jpg')
        with open(im_path, 'rb') as f:
            img = Image.open(io.BytesIO(f.read()))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Convert to array
            img = np.asarray(img)
        h, w = img.shape[:2]  # H > W

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=100)
        preprocess_input = model._get_preprocess_input()
        new_img = preprocess_input(img)
        new_h, new_w = new_img.shape[:2]
        self.assertEqual(new_w, 100)
        remove_dir(model_dir)
        #
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=350)
        preprocess_input = model._get_preprocess_input()
        new_img = preprocess_input(img)
        new_h, new_w = new_img.shape[:2]
        self.assertEqual(new_w, 350)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test12_model_keras_faster_rcnn_fit_object_detector(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._fit_object_detector'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(data_path, images_ext=('.jpg', '.jpeg', '.png'))
        df_data = pd.DataFrame({'file_path': path_list, 'bboxes': bboxes_list}).copy()

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=100, epochs=1, batch_size=2, nms_max_boxes=10,
                                                   lr_rpn_trainable_true=0.011, lr_classifier_trainable_true=0.012, lr_rpn_trainable_false=0.013,
                                                   lr_classifier_trainable_false=0.014, keras_params={'decay_rpn': 0.0, 'decay_classifier': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model._fit_object_detector(df_data, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['apple', 'banana', 'orange'])
        self.assertTrue(model.model_rpn._is_compiled)
        self.assertTrue(model.model_classifier._is_compiled)
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertAlmostEqual(0.013, float(model.model_rpn.optimizer._decayed_lr(tf.float32).numpy()))
        self.assertAlmostEqual(0.014, float(model.model_classifier.optimizer._decayed_lr(tf.float32).numpy()))
        remove_dir(model_dir)
        # With valid & shuffle à False
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=100, epochs=1, batch_size=2, nms_max_boxes=10,
                                                   lr_rpn_trainable_true=0.011, lr_classifier_trainable_true=0.012, lr_rpn_trainable_false=0.013,
                                                   lr_classifier_trainable_false=0.014, keras_params={'decay_rpn': 0.0, 'decay_classifier': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model._fit_object_detector(df_data, df_valid=df_data, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['apple', 'banana', 'orange'])
        self.assertTrue(model.model_rpn._is_compiled)
        self.assertTrue(model.model_classifier._is_compiled)
        self.assertTrue(model.model._is_compiled)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertAlmostEqual(0.013, float(model.model_rpn.optimizer._decayed_lr(tf.float32).numpy()))
        self.assertAlmostEqual(0.014, float(model.model_classifier.optimizer._decayed_lr(tf.float32).numpy()))
        remove_dir(model_dir)

        # Test continue training
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=100, epochs=1, batch_size=2, nms_max_boxes=10,
                                                   lr_rpn_trainable_true=0.011, lr_classifier_trainable_true=0.012, lr_rpn_trainable_false=0.013,
                                                   lr_classifier_trainable_false=0.014, keras_params={'decay_rpn': 0.0, 'decay_classifier': 0.0})
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(df_data, df_valid=df_data, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        model.save()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # 2nd fit
        model.fit(df_data, df_valid=df_data, with_shuffle=True)  # We fit again with the same data, not important
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        # We do not save on purpose
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # 3rd fit
        model.fit(df_data, df_valid=df_data, with_shuffle=True)  # We fit again with the same data, not important
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        # Assertion errors (bad classes)
        old_dict_classes = model.dict_classes.copy()
        model.dict_classes = {'a': 0, 'b': 1}
        with self.assertRaises(AssertionError):
            model.fit(df_data, df_valid=df_data, with_shuffle=True)
        model.dict_classes = old_dict_classes
        model.list_classes = ['a', 'b']
        with self.assertRaises(AssertionError):
            model.fit(df_data, df_valid=df_data, with_shuffle=True)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)

        # Manage errors
        df_data['bboxes'][0][0]['class'] = 'bg'
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=100, epochs=1, batch_size=2, nms_max_boxes=10,
                                                   lr_rpn_trainable_true=0.011, lr_classifier_trainable_true=0.012, lr_rpn_trainable_false=0.013,
                                                   lr_classifier_trainable_false=0.014, keras_params={'decay_rpn': 0.0, 'decay_classifier': 0.0})
        with self.assertRaises(ValueError):
            model.fit(df_data, df_valid=df_data, with_shuffle=True)

        # Clean
        remove_dir(model_dir)


    # We do not test _fit_object_detector_RPN & _fit_object_detector_classifier, already done via _fit_object_detector

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test13_model_keras_faster_rcnn_get_callbacks(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._get_callbacks'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        callbacks = model._get_callbacks(patience=12)
        callbacks_types = [type(_) for _ in callbacks]
        self.assertTrue(tf.keras.callbacks.EarlyStopping in callbacks_types)
        self.assertTrue(ModelCheckpointAll in callbacks_types)
        self.assertTrue(tf.keras.callbacks.CSVLogger in callbacks_types)
        self.assertTrue(tf.keras.callbacks.TerminateOnNaN in callbacks_types)
        checkpoint = callbacks[callbacks_types.index(ModelCheckpointAll)]
        csv_logger = callbacks[callbacks_types.index(tf.keras.callbacks.CSVLogger)]
        early_stopping = callbacks[callbacks_types.index(tf.keras.callbacks.EarlyStopping)]
        self.assertEqual(checkpoint.filepath, os.path.join(model.model_dir, 'best.hdf5'))
        self.assertEqual(csv_logger.filename, os.path.join(model.model_dir, 'logger.csv'))
        self.assertEqual(early_stopping.patience, 12)

        # level save 'LOW'
        model.level_save = 'LOW'
        callbacks = model._get_callbacks(patience=14)
        callbacks_types = [type(_) for _ in callbacks]
        self.assertTrue(tf.keras.callbacks.EarlyStopping in callbacks_types)
        self.assertFalse(ModelCheckpointAll in callbacks_types)
        self.assertTrue(tf.keras.callbacks.CSVLogger in callbacks_types)
        self.assertTrue(tf.keras.callbacks.TerminateOnNaN in callbacks_types)
        csv_logger = callbacks[callbacks_types.index(tf.keras.callbacks.CSVLogger)]
        early_stopping = callbacks[callbacks_types.index(tf.keras.callbacks.EarlyStopping)]
        self.assertEqual(csv_logger.filename, os.path.join(model.model_dir, 'logger.csv'))
        self.assertEqual(early_stopping.patience, 14)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test14_model_keras_faster_rcnn_get_learning_rate_scheduler(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._get_learning_rate_scheduler'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        self.assertEqual(model._get_learning_rate_scheduler(), None)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test15_model_keras_faster_rcnn_predict_object_detector(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector._predict_object_detector'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(data_path, images_ext=('.jpg', '.jpeg', '.png'))
        df_data = pd.DataFrame({'file_path': path_list, 'bboxes': bboxes_list}).copy()

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=100, epochs=1, batch_size=2, nms_max_boxes=10)
        model.fit(df_data, df_valid=df_data)
        preds = model.predict(df_data)
        self.assertEqual(len(preds), df_data.shape[0])
        # We can't test lots of things ...
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
            model.predict(df_data)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test16_model_keras_faster_rcnn_rcnn_plot_metrics_and_loss(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn_rcnn.ModelDetectronFasterRcnnObjectDetector._plot_metrics_and_loss'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        class FitHistory():
            def __init__(self, history) -> None:
                self.history = history
        fit_history = FitHistory({
            'loss': [1, 2, 3], 'val_loss': [1, 2, 3],
            'rpn_class_loss': [1, 2, 3], 'val_rpn_class_loss': [1, 2, 3],
            'rpn_regr_loss': [1, 2, 3], 'val_rpn_regr_loss': [1, 2, 3],
            'rpn_class_accuracy': [1, 2, 3], 'val_rpn_class_accuracy': [1, 2, 3],
            'dense_class_loss': [1, 2, 3], 'val_dense_class_loss': [1, 2, 3],
            'dense_regr_loss': [1, 2, 3], 'val_dense_regr_loss': [1, 2, 3],
            'dense_class_accuracy': [1, 2, 3], 'val_dense_class_accuracy': [1, 2, 3],
        })
        fit_history_small = FitHistory({
            'loss': [1, 2, 3], 'val_loss': [1, 2, 3],
            'rpn_regr_loss': [1, 2, 3], 'val_rpn_regr_loss': [1, 2, 3],
        })
        #
        targets_name =  {f'{model_type}_{trainable}': [
                            f'loss_{model_type}_trainable_{trainable}',
                            f'loss_class_{model_type}_trainable_{trainable}',
                            f'loss_regr_{model_type}_trainable_{trainable}',
                            f'accuracy_class_{model_type}_trainable_{trainable}',
                        ] for model_type in ['rpn', 'classifier'] for trainable in [False, True]}

        # Set a model
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir)
        plots_path = os.path.join(model.model_dir, 'plots')

        # Nominal case
        model._plot_metrics_and_loss(fit_history, model_type='rpn', trainable=True)
        list_files = os.listdir(plots_path)
        self.assertEqual(len(list_files), 4)
        for filename in targets_name['rpn_True']:
            self.assertTrue(os.path.exists(os.path.join(plots_path, f"{filename}.jpeg")))
        remove_dir(plots_path)
        #
        model._plot_metrics_and_loss(fit_history, model_type='classifier', trainable=True)
        list_files = os.listdir(plots_path)
        self.assertEqual(len(list_files), 4)
        for filename in targets_name['classifier_True']:
            self.assertTrue(os.path.exists(os.path.join(plots_path, f"{filename}.jpeg")))
        remove_dir(plots_path)
        #
        model._plot_metrics_and_loss(fit_history, model_type='rpn', trainable=False)
        list_files = os.listdir(plots_path)
        self.assertEqual(len(list_files), 4)
        for filename in targets_name['rpn_False']:
            self.assertTrue(os.path.exists(os.path.join(plots_path, f"{filename}.jpeg")))
        remove_dir(plots_path)
        #
        model._plot_metrics_and_loss(fit_history, model_type='classifier', trainable=False)
        list_files = os.listdir(plots_path)
        self.assertEqual(len(list_files), 4)
        for filename in targets_name['classifier_False']:
            self.assertTrue(os.path.exists(os.path.join(plots_path, f"{filename}.jpeg")))
        remove_dir(plots_path)

        # Less metrics
        model._plot_metrics_and_loss(fit_history_small, model_type='rpn', trainable=True)
        list_files = os.listdir(plots_path)
        self.assertEqual(len(list_files), 2)
        remove_dir(plots_path)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test17_model_keras_faster_rcnn_save(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn_rcnn.ModelDetectronFasterRcnnObjectDetector.save'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(data_path, images_ext=('.jpg', '.jpeg', '.png'))
        df_data = pd.DataFrame({'file_path': path_list, 'bboxes': bboxes_list}).copy()

        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=100, epochs=1, batch_size=2, nms_max_boxes=10)
        model.fit(df_data)
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.hdf5')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"preprocess_input.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertTrue('mainteners' in configs.keys())
        self.assertTrue('date' in configs.keys())
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('model_type' in configs.keys())
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'keras')
        self.assertTrue('batch_size' in configs.keys())
        self.assertTrue('epochs' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('patience' in configs.keys())
        self.assertTrue('width' in configs.keys())
        self.assertTrue('height' in configs.keys())
        self.assertTrue('depth' in configs.keys())
        self.assertTrue('color_mode' in configs.keys())
        self.assertTrue('in_memory' in configs.keys())
        self.assertTrue('data_augmentation_params' in configs.keys())
        self.assertTrue('nb_train_generator_images_to_save' in configs.keys())
        self.assertTrue('keras_params' in configs.keys())
        self.assertTrue('_get_model' in configs.keys())
        self.assertTrue('_get_learning_rate_scheduler' in configs.keys())
        self.assertTrue('custom_objects' in configs.keys())
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('vgg_filename' in configs.keys())
        self.assertTrue('shared_model_subsampling' in configs.keys())
        self.assertTrue('anchor_box_sizes' in configs.keys())
        self.assertTrue('anchor_box_ratios' in configs.keys())
        self.assertTrue('nb_anchors' in configs.keys())
        self.assertTrue('list_anchors' in configs.keys())
        self.assertTrue('img_min_side_size' in configs.keys())
        self.assertTrue('pool_resize_classifier' in configs.keys())
        self.assertTrue('rpn_regr_scaling' in configs.keys())
        self.assertTrue('classifier_regr_scaling' in configs.keys())
        self.assertTrue('rpn_min_overlap' in configs.keys())
        self.assertTrue('rpn_max_overlap' in configs.keys())
        self.assertTrue('rpn_restrict_num_regions' in configs.keys())
        self.assertTrue('nb_rois_classifier' in configs.keys())
        self.assertTrue('roi_nms_overlap_threshold' in configs.keys())
        self.assertTrue('nms_max_boxes' in configs.keys())
        self.assertTrue('classifier_min_overlap' in configs.keys())
        self.assertTrue('classifier_max_overlap' in configs.keys())
        self.assertTrue('pred_bbox_proba_threshold' in configs.keys())
        self.assertTrue('pred_nms_overlap_threshold' in configs.keys())
        self.assertTrue('batch_size_rpn_trainable_true' in configs.keys())
        self.assertTrue('batch_size_classifier_trainable_true' in configs.keys())
        self.assertTrue('batch_size_rpn_trainable_false' in configs.keys())
        self.assertTrue('batch_size_classifier_trainable_false' in configs.keys())
        self.assertTrue('epochs_rpn_trainable_true' in configs.keys())
        self.assertTrue('epochs_classifier_trainable_true' in configs.keys())
        self.assertTrue('epochs_rpn_trainable_false' in configs.keys())
        self.assertTrue('epochs_classifier_trainable_false' in configs.keys())
        self.assertTrue('patience_rpn_trainable_true' in configs.keys())
        self.assertTrue('patience_classifier_trainable_true' in configs.keys())
        self.assertTrue('patience_rpn_trainable_false' in configs.keys())
        self.assertTrue('patience_classifier_trainable_false' in configs.keys())
        self.assertTrue('_add_rpn_layers' in configs.keys())
        self.assertTrue('_add_classifier_layers' in configs.keys())
        self.assertTrue('_compile_model_rpn' in configs.keys())
        self.assertTrue('_compile_model_classifier' in configs.keys())
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test18_model_keras_faster_rcnn_reload_model(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn_rcnn.ModelDetectronFasterRcnnObjectDetector.reload_model'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(data_path, images_ext=('.jpg', '.jpeg', '.png'))
        df_data = pd.DataFrame({'file_path': path_list, 'bboxes': bboxes_list}).copy()


        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=100, epochs=1, batch_size=2, nms_max_boxes=10)
        model.fit(df_data)
        model.save()
        # Reload keras
        hdf5_path = os.path.join(model.model_dir, 'best.hdf5')
        reloaded_model = model.reload_model(hdf5_path)
        # We can't really test the predictions because chances are high that no bboxes are predicted

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test19_model_keras_faster_rcnn_reload_from_standalone(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn_rcnn.ModelDetectronFasterRcnnObjectDetector.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(data_path, images_ext=('.jpg', '.jpeg', '.png'))
        df_data = pd.DataFrame({'file_path': path_list, 'bboxes': bboxes_list}).copy()


        # Nominal case
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=100, epochs=1, batch_size=2, nms_max_boxes=10)
        fit_params = model.fit(df_data)
        model.save()
        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        hdf5_path = os.path.join(model.model_dir, "best.hdf5")
        preprocess_input_path = os.path.join(model.model_dir, "preprocess_input.pkl")
        new_model = ModelKerasFasterRcnnObjectDetector()
        new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
                                         preprocess_input_path=preprocess_input_path)
        # Tests
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.model_type, new_model.model_type)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.batch_size, new_model.batch_size)
        self.assertEqual(model.epochs, new_model.epochs)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.patience, new_model.patience)
        self.assertEqual(model.width, new_model.width)
        self.assertEqual(model.height, new_model.height)
        self.assertEqual(model.depth, new_model.depth)
        self.assertEqual(model.color_mode, new_model.color_mode)
        self.assertEqual(model.in_memory, new_model.in_memory)
        self.assertEqual(model.data_augmentation_params, new_model.data_augmentation_params)
        self.assertEqual(model.nb_train_generator_images_to_save, new_model.nb_train_generator_images_to_save)
        self.assertEqual(model.vgg_filename, new_model.vgg_filename)
        self.assertEqual(model.vgg_path, new_model.vgg_path)
        self.assertEqual(model.shared_model_subsampling, new_model.shared_model_subsampling)
        self.assertEqual(model.anchor_box_sizes, new_model.anchor_box_sizes)
        self.assertEqual(model.anchor_box_ratios, new_model.anchor_box_ratios)
        self.assertEqual(model.nb_anchors, new_model.nb_anchors)
        self.assertEqual(model.list_anchors, new_model.list_anchors)
        self.assertEqual(model.img_min_side_size, new_model.img_min_side_size)
        self.assertEqual(model.pool_resize_classifier, new_model.pool_resize_classifier)
        self.assertEqual(model.rpn_regr_scaling, new_model.rpn_regr_scaling)
        self.assertEqual(model.classifier_regr_scaling, new_model.classifier_regr_scaling)
        self.assertEqual(model.rpn_min_overlap, new_model.rpn_min_overlap)
        self.assertEqual(model.rpn_max_overlap, new_model.rpn_max_overlap)
        self.assertEqual(model.rpn_restrict_num_regions, new_model.rpn_restrict_num_regions)
        self.assertEqual(model.nb_rois_classifier, new_model.nb_rois_classifier)
        self.assertEqual(model.roi_nms_overlap_threshold, new_model.roi_nms_overlap_threshold)
        self.assertEqual(model.nms_max_boxes, new_model.nms_max_boxes)
        self.assertEqual(model.classifier_min_overlap, new_model.classifier_min_overlap)
        self.assertEqual(model.classifier_max_overlap, new_model.classifier_max_overlap)
        self.assertEqual(model.pred_bbox_proba_threshold, new_model.pred_bbox_proba_threshold)
        self.assertEqual(model.pred_nms_overlap_threshold, new_model.pred_nms_overlap_threshold)
        self.assertEqual(model.batch_size_rpn_trainable_true, new_model.batch_size_rpn_trainable_true)
        self.assertEqual(model.batch_size_classifier_trainable_true, new_model.batch_size_classifier_trainable_true)
        self.assertEqual(model.batch_size_classifier_trainable_true, new_model.batch_size_classifier_trainable_true)
        self.assertEqual(model.batch_size_rpn_trainable_false, new_model.batch_size_rpn_trainable_false)
        self.assertEqual(model.batch_size_rpn_trainable_false, new_model.batch_size_rpn_trainable_false)
        self.assertEqual(model.batch_size_classifier_trainable_false, new_model.batch_size_classifier_trainable_false)
        self.assertEqual(model.batch_size_classifier_trainable_false, new_model.batch_size_classifier_trainable_false)
        self.assertEqual(model.epochs_rpn_trainable_true, new_model.epochs_rpn_trainable_true)
        self.assertEqual(model.epochs_classifier_trainable_true, new_model.epochs_classifier_trainable_true)
        self.assertEqual(model.epochs_classifier_trainable_true, new_model.epochs_classifier_trainable_true)
        self.assertEqual(model.epochs_rpn_trainable_false, new_model.epochs_rpn_trainable_false)
        self.assertEqual(model.epochs_classifier_trainable_false, new_model.epochs_classifier_trainable_false)
        self.assertEqual(model.epochs_classifier_trainable_false, new_model.epochs_classifier_trainable_false)
        self.assertEqual(model.patience_rpn_trainable_true, new_model.patience_rpn_trainable_true)
        self.assertEqual(model.patience_classifier_trainable_true, new_model.patience_classifier_trainable_true)
        self.assertEqual(model.patience_classifier_trainable_true, new_model.patience_classifier_trainable_true)
        self.assertEqual(model.patience_rpn_trainable_false, new_model.patience_rpn_trainable_false)
        self.assertEqual(model.patience_classifier_trainable_false, new_model.patience_classifier_trainable_false)
        self.assertEqual(model.patience_classifier_trainable_false, new_model.patience_classifier_trainable_false)
        self.assertEqual(model.keras_params, new_model.keras_params)
        # self.assertEqual(model.custom_objects, new_model.custom_objects)  # Do not work because functions are wrapped ...
        # We can't really test the pipeline so we test predictions
        self.assertEqual(model.predict(df_data), new_model.predict(df_data))  # Useful ?
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)


        ############################################
        # Errors
        ############################################

        # with self.assertRaises(FileNotFoundError):
        #     new_model = ModelCnnClassifier()
        #     new_model.reload_from_standalone(configuration_path='toto.json', hdf5_path=hdf5_path,
        #                                      preprocess_input_path=preprocess_input_path)
        # with self.assertRaises(FileNotFoundError):
        #     new_model = ModelCnnClassifier()
        #     new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path='toto.pkl',
        #                                      preprocess_input_path=preprocess_input_path)
        # with self.assertRaises(FileNotFoundError):
        #     new_model = ModelCnnClassifier()
        #     new_model.reload_from_standalone(configuration_path=conf_path, hdf5_path=hdf5_path,
        #                                      preprocess_input_path='toto.pkl')

        # Clean
        remove_dir(model_dir)


class CustomGeneratorRpnTests(unittest.TestCase):
    '''Main class to test model_keras_faster_rcnn.CustomGeneratorRpn'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    @patch('{{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector', side_effect=MockModelKerasFasterRcnnObjectDetector)
    def test01_custom_generator_rpn_init(self, mock_model):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.CustomGeneratorRpn.__init__'''

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(data_path, images_ext=('.jpg', '.jpeg', '.png'))
        img_data_list = []
        for filepath, bboxes in zip(path_list, bboxes_list):
            img_data_list.append({'file_path': filepath, 'bboxes': bboxes})
        fake_model = mock_model()

        # Init., test all parameters
        generator = CustomGeneratorRpn(img_data_list, batch_size=5, shuffle=True, seed=42, model=fake_model,
                                       horizontal_flip=False, vertical_flip=False, rot_90=False, data_type='train',
                                       with_img_data=False)
        self.assertEqual(generator.batch_size, 5)
        self.assertEqual(generator.shuffle, True)
        self.assertEqual(generator.seed, 42)
        self.assertEqual(generator.img_data_list, img_data_list)
        self.assertEqual(generator.model, fake_model)
        self.assertEqual(generator.horizontal_flip, False)
        self.assertEqual(generator.vertical_flip, False)
        self.assertEqual(generator.rot_90, False)
        self.assertEqual(generator.with_img_data, False)
        self.assertEqual(generator.is_test, False)
        # Other params
        generator = CustomGeneratorRpn(img_data_list, batch_size=10, shuffle=False, seed=64, model=fake_model,
                                       horizontal_flip=True, vertical_flip=True, rot_90=True, data_type='test',
                                       with_img_data=True)
        self.assertEqual(generator.batch_size, 1)  # test data, force batch size 1
        self.assertEqual(generator.shuffle, False)
        self.assertEqual(generator.seed, 64)
        self.assertEqual(generator.img_data_list, img_data_list)
        self.assertEqual(generator.model, fake_model)
        self.assertEqual(generator.horizontal_flip, True)
        self.assertEqual(generator.vertical_flip, True)
        self.assertEqual(generator.rot_90, True)
        self.assertEqual(generator.with_img_data, True)
        self.assertEqual(generator.is_test, True)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    @patch('{{package_name}}.models_training.object_detectors.utils_object_detectors.get_rpn_targets', side_effect=mock_get_rpn_targets)
    @patch('{{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector', side_effect=MockModelKerasFasterRcnnObjectDetector)
    def test02_custom_generator_rpn_get_batches_of_transformed_samples(self, mock_model, mock_rpn_targets):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.CustomGeneratorRpn._get_batches_of_transformed_samples'''

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(data_path, images_ext=('.jpg', '.jpeg', '.png'))
        img_data_list = []
        for filepath, bboxes in zip(path_list, bboxes_list):
            img_data_list.append({'file_path': filepath, 'bboxes': bboxes})
        fake_model = mock_model()

        # Nominal case
        generator = CustomGeneratorRpn(img_data_list, batch_size=2, shuffle=True, seed=42, model=fake_model,
                                       data_type='train', with_img_data=False)
        X, Y = generator._get_batches_of_transformed_samples(index_array=np.array([3, 1]))
        self.assertEqual(list(X.keys()), ['input_img'])
        self.assertEqual(X['input_img'].shape[0], 2)
        self.assertEqual(X['input_img'].shape[-1], 3)
        self.assertEqual(sorted(list(Y.keys())), sorted(['rpn_class', 'rpn_regr']))
        # with_img_data = True
        generator = CustomGeneratorRpn(img_data_list, batch_size=2, shuffle=True, seed=42, model=fake_model,
                                       data_type='train', with_img_data=True)
        X, Y, batch_data = generator._get_batches_of_transformed_samples(index_array=np.array([3, 1]))
        self.assertEqual(list(X.keys()), ['input_img'])
        self.assertEqual(X['input_img'].shape[0], 2)
        self.assertEqual(X['input_img'].shape[-1], 3)
        self.assertEqual(sorted(list(Y.keys())), sorted(['rpn_class', 'rpn_regr']))
        self.assertEqual(len(batch_data), 2)
        self.assertTrue('batch_width' in batch_data[0].keys())
        self.assertTrue('batch_height' in batch_data[0].keys())
        # data_type == 'test'
        generator = CustomGeneratorRpn(img_data_list, batch_size=2, shuffle=True, seed=42, model=fake_model,
                                       data_type='test', with_img_data=False)
        X = generator._get_batches_of_transformed_samples(index_array=np.array([2]))
        self.assertEqual(list(X.keys()), ['input_img'])
        self.assertEqual(X['input_img'].shape[0], 1)
        self.assertEqual(X['input_img'].shape[-1], 3)
        # data_type == 'test' & with_img_data == True
        generator = CustomGeneratorRpn(img_data_list, batch_size=2, shuffle=True, seed=42, model=fake_model,
                                       data_type='test', with_img_data=True)
        X, batch_data = generator._get_batches_of_transformed_samples(index_array=np.array([2]))
        self.assertEqual(list(X.keys()), ['input_img'])
        self.assertEqual(X['input_img'].shape[0], 1)
        self.assertEqual(X['input_img'].shape[-1], 3)
        self.assertEqual(len(batch_data), 1)
        self.assertTrue('batch_width' in batch_data[0].keys())
        self.assertTrue('batch_height' in batch_data[0].keys())


class CustomGeneratorClassifierTests(unittest.TestCase):
    '''Main class to test model_keras_faster_rcnn.CustomGeneratorClassifier'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    @patch('{{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.clone_model', side_effect=mock_clone_model_fn)
    @patch('{{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector', side_effect=MockModelKerasFasterRcnnObjectDetector)
    def test01_custom_generator_classifier_init(self, mock_model, mock_clone_model):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.CustomGeneratorClassifier.__init__'''

        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(data_path, images_ext=('.jpg', '.jpeg', '.png'))
        img_data_list = []
        for filepath, bboxes in zip(path_list, bboxes_list):
            img_data_list.append({'file_path': filepath, 'bboxes': bboxes})
        fake_model = mock_model()

        # Init., test all parameters
        generator = CustomGeneratorClassifier(img_data_list, batch_size=5, shuffle=True, seed=42, model=fake_model, shared_model_trainable=False,
                                              horizontal_flip=False, vertical_flip=False, rot_90=False, data_type='train', with_img_data=False)
        self.assertEqual(generator.batch_size, 5)
        self.assertEqual(generator.shuffle, True)
        self.assertEqual(generator.seed, 42)
        self.assertEqual(generator.img_data_list, img_data_list)
        self.assertEqual(generator.model, fake_model)
        self.assertEqual(generator.horizontal_flip, False)
        self.assertEqual(generator.vertical_flip, False)
        self.assertEqual(generator.rot_90, False)
        self.assertEqual(generator.with_img_data, False)
        self.assertEqual(generator.is_test, False)
        self.assertTrue(generator.rpn_clone is None)
        # Other params
        generator = CustomGeneratorClassifier(img_data_list, batch_size=10, shuffle=False, seed=64, model=fake_model, shared_model_trainable=True,
                                              horizontal_flip=True, vertical_flip=True, rot_90=True, data_type='test', with_img_data=True)
        self.assertEqual(generator.batch_size, 1)  # test data, force batch size 1
        self.assertEqual(generator.shuffle, False)
        self.assertEqual(generator.seed, 64)
        self.assertEqual(generator.img_data_list, img_data_list)
        self.assertEqual(generator.model, fake_model)
        self.assertEqual(generator.horizontal_flip, True)
        self.assertEqual(generator.vertical_flip, True)
        self.assertEqual(generator.rot_90, True)
        self.assertEqual(generator.with_img_data, True)
        self.assertEqual(generator.is_test, True)
        self.assertFalse(generator.rpn_clone is None)

    # We do not mock anything, too complex
    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test02_custom_generator_classifier_get_batches_of_transformed_samples(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_keras_faster_rcnn.CustomGeneratorClassifier._get_batches_of_transformed_samples'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars & model
        data_path = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(data_path, images_ext=('.jpg', '.jpeg', '.png'))
        df_data = pd.DataFrame({'file_path': path_list, 'bboxes': bboxes_list}).copy()
        img_data_list = []
        for filepath, bboxes in zip(path_list, bboxes_list):
            img_data_list.append({'file_path': filepath, 'bboxes': bboxes})
        model = ModelKerasFasterRcnnObjectDetector(model_dir=model_dir, img_min_side_size=100, epochs=1, batch_size=2, nms_max_boxes=10, nb_rois_classifier=3)
        model.fit(df_data)

        # Nominal case
        generator = CustomGeneratorClassifier(img_data_list, batch_size=2, shuffle=True, seed=42, model=model,
                                              shared_model_trainable=False, data_type='train', with_img_data=False)
        X, Y = generator._get_batches_of_transformed_samples(index_array=np.array([3, 1]))
        self.assertEqual(sorted(list(X.keys())), sorted(['input_img', 'input_rois']))
        self.assertEqual(X['input_img'].shape[0], 2)
        self.assertEqual(X['input_img'].shape[-1], 3)
        self.assertEqual(X['input_rois'].shape[0], 2)
        self.assertEqual(X['input_rois'].shape[1], model.nb_rois_classifier)
        self.assertEqual(X['input_rois'].shape[2], 4)
        self.assertEqual(sorted(list(Y.keys())), sorted(['dense_class', 'dense_regr']))
        self.assertEqual(Y['dense_class'].shape[0], 2)
        self.assertEqual(Y['dense_class'].shape[1], model.nb_rois_classifier)
        self.assertEqual(Y['dense_regr'].shape[0], 2)
        self.assertEqual(Y['dense_regr'].shape[1], model.nb_rois_classifier)
        # with_img_data = True & shared_model_trainable à True
        generator = CustomGeneratorClassifier(img_data_list, batch_size=2, shuffle=True, seed=42, model=model,
                                              shared_model_trainable=True, data_type='train', with_img_data=True)
        X, Y, batch_data = generator._get_batches_of_transformed_samples(index_array=np.array([3, 1]))
        self.assertEqual(sorted(list(X.keys())), sorted(['input_img', 'input_rois']))
        self.assertEqual(X['input_img'].shape[0], 2)
        self.assertEqual(X['input_img'].shape[-1], 3)
        self.assertEqual(X['input_rois'].shape[0], 2)
        self.assertEqual(X['input_rois'].shape[1], model.nb_rois_classifier)
        self.assertEqual(X['input_rois'].shape[2], 4)
        self.assertEqual(sorted(list(Y.keys())), sorted(['dense_class', 'dense_regr']))
        self.assertEqual(Y['dense_class'].shape[0], 2)
        self.assertEqual(Y['dense_class'].shape[1], model.nb_rois_classifier)
        self.assertEqual(Y['dense_regr'].shape[0], 2)
        self.assertEqual(Y['dense_regr'].shape[1], model.nb_rois_classifier)
        self.assertEqual(len(batch_data), 2)
        self.assertTrue('batch_width' in batch_data[0].keys())
        self.assertTrue('batch_height' in batch_data[0].keys())
        # data_type == 'test'
        generator = CustomGeneratorClassifier(img_data_list, batch_size=2, shuffle=True, seed=42, model=model,
                                              shared_model_trainable=True, data_type='test', with_img_data=False)
        X = generator._get_batches_of_transformed_samples(index_array=np.array([2]))
        self.assertEqual(sorted(list(X.keys())), sorted(['input_img', 'input_rois']))
        self.assertEqual(X['input_img'].shape[0], 1)
        self.assertEqual(X['input_img'].shape[-1], 3)
        self.assertEqual(X['input_rois'].shape[0], 1)
        # self.assertEqual(X['input_rois'].shape[1], ???) -> We do not know the number of ROIs returned for the test
        self.assertEqual(X['input_rois'].shape[2], 4)
        # data_type == 'test' & with_img_data == True
        generator = CustomGeneratorClassifier(img_data_list, batch_size=2, shuffle=True, seed=42, model=model,
                                              shared_model_trainable=True, data_type='test', with_img_data=True)
        X, batch_data = generator._get_batches_of_transformed_samples(index_array=np.array([2]))
        self.assertEqual(sorted(list(X.keys())), sorted(['input_img', 'input_rois']))
        self.assertEqual(X['input_img'].shape[0], 1)
        self.assertEqual(X['input_img'].shape[-1], 3)
        self.assertEqual(X['input_rois'].shape[0], 1)
        # self.assertEqual(X['input_rois'].shape[1], ???) -> We do not know the number of ROIs returned for the test
        self.assertEqual(X['input_rois'].shape[2], 4)
        self.assertEqual(len(batch_data), 1)
        self.assertTrue('batch_width' in batch_data[0].keys())
        self.assertTrue('batch_height' in batch_data[0].keys())


# We do not test ModelCheckpointAll


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
