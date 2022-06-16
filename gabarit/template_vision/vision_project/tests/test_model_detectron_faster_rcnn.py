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
import gc
import json
import torch
import shutil
import numpy as np
import pandas as pd

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.monitoring.model_logger import ModelLogger
from {{package_name}}.models_training.object_detectors import model_detectron_faster_rcnn
from {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn import ModelDetectronFasterRcnnObjectDetector, TrainerRCNN

from detectron2.structures import BoxMode
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.utils.file_io import PathManager
from detectron2.engine import hooks as module_hooks
from detectron2.data.common import AspectRatioGroupedDataset
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.data import (DatasetCatalog, MetadataCatalog, build_detection_test_loader, DatasetMapper)


# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


def download_url_crash(x, y):
    raise ConnectionError("error")


class ModelDetectronRCNNTests(unittest.TestCase):
    '''Main class to test model_detectron_faster_rcnn'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    @patch('torch.cuda.is_available')
    def test01_model_detectron_faster_rcnn_init(self, mock_cuda_is_available):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.model_type, 'object_detector')
        self.assertTrue(os.path.isdir(model_dir))
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, epochs=8)
        self.assertEqual(model.epochs, 8)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, batch_size=8)
        self.assertEqual(model.cfg.SOLVER.IMS_PER_BATCH, 8)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, validation_split=0.4)
        self.assertEqual(model.validation_split, 0.4)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, lr=0.4)
        self.assertEqual(model.cfg.SOLVER.BASE_LR, 0.4)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, min_delta_es=0.4)
        self.assertEqual(model.min_delta_es, 0.4)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, patience=3)
        self.assertEqual(model.patience, 3)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, restore_best_weights=False)
        self.assertEqual(model.restore_best_weights, False)
        remove_dir(model_dir)

        #
        data_augmentation_params = {'horizontal_flip': True, 'vertical_flip': False}
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params=data_augmentation_params)
        self.assertEqual(model.data_augmentation_params, data_augmentation_params)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=0.2)
        self.assertEqual(model.cfg.MODEL.RPN.IOU_THRESHOLDS[0], 0.2)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_max_overlap=0.8)
        self.assertEqual(model.cfg.MODEL.RPN.IOU_THRESHOLDS[1], 0.8)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_restrict_num_regions=65)
        self.assertEqual(model.cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, 65)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, roi_nms_overlap_threshold=0.6)
        self.assertEqual(model.cfg.MODEL.RPN.NMS_THRESH, 0.6)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, pred_bbox_proba_threshold=0.6)
        self.assertEqual(model.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, 0.6)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, pred_nms_overlap_threshold=0.6)
        self.assertEqual(model.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST, 0.6)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, nb_log_write_per_epoch=10)
        self.assertEqual(model.nb_log_write_per_epoch, 10)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, nb_log_display_per_epoch=20)
        self.assertEqual(model.nb_log_display_per_epoch, 20)
        remove_dir(model_dir)

        #
        mock_cuda_is_available.return_value = True
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        self.assertEqual(model.cfg.MODEL.DEVICE, 'cuda')
        remove_dir(model_dir)

        #
        mock_cuda_is_available.return_value = False
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        self.assertEqual(model.cfg.MODEL.DEVICE, 'cpu')
        remove_dir(model_dir)

        # Check errors
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=-0.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=1.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_max_overlap=-0.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_max_overlap=1.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=0.3, rpn_max_overlap=0.1)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_restrict_num_regions=0)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, roi_nms_overlap_threshold=-0.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, roi_nms_overlap_threshold=1.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, pred_bbox_proba_threshold=-0.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, pred_bbox_proba_threshold=1.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, pred_nms_overlap_threshold=-0.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, pred_nms_overlap_threshold=1.01)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    @patch('{{package_name}}.utils.download_url', side_effect=download_url_crash)
    @patch('torch.cuda.is_available')
    def test02_model_detectron_faster_rcnn_init_offline(self, mock_cuda_is_available, mock_download_url):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector.__init__
        - No access to a base model
        '''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Clean cache path if exists
        cache_path = os.path.join(utils.get_data_path(), 'detectron2_conf_files')
        if os.path.isdir(cache_path):
            shutil.rmtree(cache_path)

        # Init., test all parameters
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertEqual(model.model_type, 'object_detector')
        self.assertTrue(os.path.isdir(model_dir))
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, epochs=8)
        self.assertEqual(model.epochs, 8)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, batch_size=8)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, validation_split=0.4)
        self.assertEqual(model.validation_split, 0.4)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, lr=0.4)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, min_delta_es=0.4)
        self.assertEqual(model.min_delta_es, 0.4)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, patience=3)
        self.assertEqual(model.patience, 3)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, restore_best_weights=False)
        self.assertEqual(model.restore_best_weights, False)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        data_augmentation_params = {'horizontal_flip': True, 'vertical_flip': False}
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, data_augmentation_params=data_augmentation_params)
        self.assertEqual(model.data_augmentation_params, data_augmentation_params)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=0.2)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_max_overlap=0.8)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_restrict_num_regions=65)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, roi_nms_overlap_threshold=0.6)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, pred_bbox_proba_threshold=0.6)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, pred_nms_overlap_threshold=0.6)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, nb_log_write_per_epoch=10)
        self.assertEqual(model.nb_log_write_per_epoch, 10)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, nb_log_display_per_epoch=20)
        self.assertEqual(model.nb_log_display_per_epoch, 20)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        mock_cuda_is_available.return_value = True
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        #
        mock_cuda_is_available.return_value = False
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        self.assertTrue(model.cfg is None)
        remove_dir(model_dir)

        # Check errors
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=-0.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=1.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_max_overlap=-0.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_max_overlap=1.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_min_overlap=0.3, rpn_max_overlap=0.1)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, rpn_restrict_num_regions=0)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, roi_nms_overlap_threshold=-0.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, roi_nms_overlap_threshold=1.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, pred_bbox_proba_threshold=-0.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, pred_bbox_proba_threshold=1.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, pred_nms_overlap_threshold=-0.01)
        with self.assertRaises(ValueError):
            ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, pred_nms_overlap_threshold=1.01)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test03_model_detectron_faster_rcnn_register_dataset(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector._register_dataset'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Model instanciation
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}

        # Nominal case
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        for data_type in ['train', 'valid']:
            if f'dataset_{data_type}' in DatasetCatalog:
                DatasetCatalog.pop(f'dataset_{data_type}')
                MetadataCatalog.pop(f'dataset_{data_type}')
            model._register_dataset(df, data_type)
            self.assertTrue(f'dataset_{data_type}' in DatasetCatalog)
            dataset = DatasetCatalog.get(f'dataset_{data_type}')
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0], {'file_name': os.path.join('test_data', 'apple_36.jpg'),
                                          'image_id': 0,
                                          'height': 171,
                                          'width': 166,
                                          'annotations': [{'bbox': [8, 17, 155, 161],
                                                           'bbox_mode': BoxMode.XYXY_ABS,
                                                           'category_id': 2,
                                                           'iscrowd': 0}]})
            metadata = MetadataCatalog.get(f'dataset_{data_type}')
            self.assertEqual(metadata.thing_classes, model.list_classes)
            self.assertEqual(metadata.name, f'dataset_{data_type}')
            DatasetCatalog.pop(f'dataset_{data_type}')
            MetadataCatalog.pop(f'dataset_{data_type}')

        remove_dir(model_dir)
        # Test register with two different datasets with two different models

        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df, 'train')
        remove_dir(model_dir)

        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['1', '2', '3']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': '3'}]}])
        model._register_dataset(df, 'train')

        remove_dir(model_dir)


        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])

        # Manage errors
        df = pd.DataFrame(columns=['file_path', 'bboxes'])
        with self.assertRaises(ValueError):
            model._register_dataset(df, 'test')

        df = pd.DataFrame(columns=['file_path', 'coucou'])
        with self.assertRaises(ValueError):
            model._register_dataset(df, 'train')

        df = pd.DataFrame(columns=['coucou', 'bboxes'])
        with self.assertRaises(ValueError):
            model._register_dataset(df, 'train')

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test04_model_detectron_faster_rcnn_prepare_dataset_format(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector._prepare_dataset_format'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Model instanciation
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        inv_dict_classes = {value: key for key, value in model.dict_classes.items()}

        # Nominal case
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        dataset_dicts = model._prepare_dataset_format(df, inv_dict_classes)
        dataset_dicts_target = {'file_name': os.path.join('test_data', 'apple_36.jpg'),
                                'image_id': 0,
                                'height': 171,
                                'width': 166,
                                'annotations': [{'bbox': [8, 17, 155, 161],
                                                 'bbox_mode': BoxMode.XYXY_ABS,
                                                 'category_id': 2,
                                                 'iscrowd': 0}]}
        self.assertEqual(dataset_dicts[0], dataset_dicts_target)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test05_model_detectron_faster_rcnn_fit(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector.fit'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Get data
        data_directory = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(data_directory, images_ext=('.jpg', '.jpeg', '.png'))
        df_data = pd.DataFrame({'file_path': path_list, 'bboxes': bboxes_list}).copy()

        # Nominal case
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, epochs=2, rpn_restrict_num_regions=4)
        model.cfg.INPUT.MIN_SIZE_TRAIN = (400,)  # We try to limit the memory impact
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        # Fit
        model.fit(df_data, df_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['apple', 'banana', 'orange'])
        self.assertEqual(model.dict_classes, {i: category for i, category in enumerate(model.list_classes)})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.pth')))
        self.assertEqual(model.cfg.MODEL.ROI_HEADS.NUM_CLASSES, len(model.list_classes))
        self.assertTrue('dataset_train' in DatasetCatalog)
        self.assertTrue('dataset_valid' in DatasetCatalog)
        self.assertEqual(model.cfg.SOLVER.MAX_ITER, max(int(int(len(df_data)*(1-model.validation_split)) / model.cfg.SOLVER.IMS_PER_BATCH), 1)*model.epochs-1)
        self.assertEqual(model.cfg.MODEL.WEIGHTS, os.path.join(model.cfg.OUTPUT_DIR, 'best.pth'))
        self.assertTrue(os.path.exists(os.path.join(model.cfg.OUTPUT_DIR, 'metrics.json')))
        model.save()
        # 2nd fit
        model.fit(df_data, df_valid=None, with_shuffle=True)  # We fit again with the same data, not important
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        second_model_dir = model.model_dir

        # Manage errors - we fit again with data containing the wrong classes
        error_directory = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'error_fruits')
        error_path_list, error_bboxes_list, _ = utils.read_folder_object_detection(error_directory, images_ext=('.jpg', '.jpeg', '.png'))
        error_df_data = pd.DataFrame({'file_path': error_path_list, 'bboxes': error_bboxes_list}).copy()
        with self.assertRaises(AssertionError):
            model.fit(error_df_data, df_valid=None, with_shuffle=True)

        # Clean
        remove_dir(model.model_dir)
        remove_dir(model_dir)
        remove_dir(second_model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test06_model_detectron_faster_rcnn_predict(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector.predict'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Get data
        directory = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(directory, images_ext=('.jpg', '.jpeg', '.png'))
        df_data = pd.DataFrame({'file_path': path_list, 'bboxes': bboxes_list}).copy()

        # Nominal case
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, epochs=2, rpn_restrict_num_regions=4)
        model.cfg.INPUT.MIN_SIZE_TRAIN = (400,)  # We try to limit the memory impact
        model.fit(df_data, df_valid=None, with_shuffle=True)
        # predict (with write_images to False)
        predictions = model.predict(df_data, write_images=False)
        self.assertEqual(len(predictions), len(df_data))
        for bboxes in predictions:
            for bbox in bboxes:
                for bbox_keys in ['x1', 'x2', 'y1', 'y2', 'class', 'proba']:
                    self.assertTrue(bbox_keys in bbox.keys())
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'inference', 'images')))
        # predict (with write_images to True)
        predictions = model.predict(df_data, write_images=True)
        self.assertEqual(len(predictions), len(df_data))
        for bboxes in predictions:
            for bbox in bboxes:
                for bbox_keys in ['x1', 'x2', 'y1', 'y2', 'class', 'proba']:
                    self.assertTrue(bbox_keys in bbox.keys())
        for filepath in path_list:
            filename = os.path.split(filepath)[-1]
            self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'inference', 'images', filename)))
        # predict (with write_images to True & output_dir_image != None)
        output_dir_image = os.path.join(model.model_dir, 'new_inference_images')
        predictions = model.predict(df_data, write_images=True, output_dir_image=output_dir_image)
        for bboxes in predictions:
            for bbox in bboxes:
                for bbox_keys in ['x1', 'x2', 'y1', 'y2', 'class', 'proba']:
                    self.assertTrue(bbox_keys in bbox.keys())
        for filepath in path_list:
            filename = os.path.split(filepath)[-1]
            self.assertTrue(os.path.exists(os.path.join(output_dir_image, filename)))

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test07_model_detectron_faster_rcnn_plot_metrics_and_loss(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector._plot_metrics_and_loss'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We initialize a model with a metrics file
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        metrics_path_in = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fake_metrics.json')
        metrics_path_out = os.path.join(model.cfg.OUTPUT_DIR, 'metrics.json')
        if not os.path.exists(model.cfg.OUTPUT_DIR):
            os.makedirs(model.cfg.OUTPUT_DIR)
        shutil.copyfile(metrics_path_in, metrics_path_out)

        # Nominal case
        model._plot_metrics_and_loss()
        plots_path = os.path.join(model.cfg.OUTPUT_DIR, 'plots')
        list_files = os.listdir(plots_path)
        self.assertEqual(len(list_files), 5)  # To date, we plot 5 figures
        for filename in ['total_loss.jpeg', 'loss_cls_classifier.jpeg', 'loss_reg_classifier.jpeg',
                         'loss_cls_rpn.jpeg', 'loss_reg_rpn.jpeg']:
            self.assertTrue(os.path.exists(os.path.join(plots_path, filename)))

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test08_model_detectron_faster_rcnn_load_metrics_from_json(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector._load_metrics_from_json'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Get data
        directory = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(directory, images_ext=('.jpg', '.jpeg', '.png'))
        df_data = pd.DataFrame({'file_path': path_list, 'bboxes': bboxes_list}).copy()

        # Nominal case
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, epochs=2, rpn_restrict_num_regions=4, level_save='LOW')
        model.cfg.INPUT.MIN_SIZE_TRAIN = (400,)  # We try to limit the memory impact
        model.fit(df_data, df_valid=None, with_shuffle=True)
        # Load json file
        path_json_metrics = os.path.join(model.cfg.OUTPUT_DIR, 'metrics.json')
        metrics = model._load_metrics_from_json(path_json_metrics)
        for metric in ['total_loss', 'loss_cls', 'loss_box_reg', 'loss_rpn_cls', 'loss_rpn_loc']:
            self.assertTrue(metric in metrics.columns)
            self.assertTrue(f'validation_{metric}' in metrics.columns)
        self.assertTrue('iteration' in metrics.columns)
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test09_model_detectron_faster_rcnn_plot_one_metric(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector._plot_one_metric'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        metrics = pd.DataFrame({'total_loss': [1.0, 0.5, 0.3], 'validation_total_loss': [1.1, 0.6, 0.35], 'iteration': [0, 1, 2]})
        name_metric = 'total_loss'
        plots_path = os.path.join(model.cfg.OUTPUT_DIR, 'plots')
        if not os.path.exists(plots_path):
            os.mkdir(plots_path)
        model._plot_one_metric(metrics=metrics, name_metric=name_metric, title='coucou', output_filename=name_metric, plots_path=plots_path)
        self.assertTrue(os.path.exists(os.path.join(plots_path, f'{name_metric}.jpeg')))

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test10_model_detectron_faster_rcnn_save(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector.save'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, f"{model.model_name}.pkl")))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        dict_equal = {'librairie': 'detectron2',
                      'patience': model.patience,
                      'restore_best_weights': model.restore_best_weights,
                      'data_augmentation_params': model.data_augmentation_params,
                      'nb_log_write_per_epoch': model.nb_log_write_per_epoch,
                      'epochs': model.epochs,
                      'nb_log_display_per_epoch': model.nb_log_display_per_epoch,
                      'detectron_config_base_filename': model.detectron_config_base_filename,
                      'detectron_config_filename': model.detectron_config_filename,
                      'detectron_model_filename': model.detectron_model_filename}
        dict_almost_equal = {'validation_split': model.validation_split,
                             'min_delta_es': model.min_delta_es}
        self.assertTrue('cfg' in configs)
        self.assertEqual(configs['test'], 8)
        for key, value in dict_equal.items():
            self.assertTrue(key in configs)
            self.assertEqual(value, configs[key])
        for key, value in dict_almost_equal.items():
            self.assertTrue(key in configs)
            self.assertAlmostEqual(value, configs[key])

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test11_model_detectron_faster_rcnn_reload_from_standalone(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector.reload_from_standalone'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        model_dir_2 = os.path.join(os.getcwd(), 'new_model_test_123456789')
        remove_dir(model_dir)
        remove_dir(model_dir_2)

        # Get data
        directory = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(directory, images_ext=('.jpg', '.jpeg', '.png'))
        df_data = pd.DataFrame({'file_path': path_list, 'bboxes': bboxes_list}).copy()

        # Model instanciation
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, epochs=1, validation_split=0.1,
                                                       min_delta_es=0.1, patience=4, rpn_restrict_num_regions=4)
        model.cfg.INPUT.MIN_SIZE_TRAIN = (400,)  # We try to limit the memory impact
        model.fit(df_data, df_valid=None, with_shuffle=True)
        model.save()

        # Nominal case
        # Reload
        configuration_path = os.path.join(model_dir, 'configurations.json')
        pth_path = os.path.join(model_dir, 'best.pth')
        new_model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir_2, level_save='LOW', batch_size=1,
                                                           lr=0.1, restore_best_weights=False,
                                                           data_augmentation_params={'horizontal_flip': False, 'vertical_flip': False},
                                                           rpn_min_overlap=0.5, rpn_max_overlap=0.9, rpn_restrict_num_regions=16,
                                                           roi_nms_overlap_threshold=0.5, nb_log_write_per_epoch=2, nb_log_display_per_epoch=2)
        new_model.reload_from_standalone(configuration_path=configuration_path, pth_path=pth_path)
        # Check attributes
        list_attributes_to_test_equal = ['model_type', 'list_classes', 'dict_classes', 'level_save', 'nb_fit',
                                         'trained', 'patience', 'restore_best_weights', 'data_augmentation_params',
                                         'nb_log_write_per_epoch', 'epochs', 'nb_log_display_per_epoch', 'detectron_config_base_filename',
                                         'detectron_config_filename', 'detectron_model_filename']
        list_attributes_to_test_almost_equal = ['validation_split', 'min_delta_es']
        for attribute in list_attributes_to_test_equal:
            self.assertEqual(getattr(model, attribute), getattr(new_model, attribute), attribute)
        for attribute in list_attributes_to_test_almost_equal:
            self.assertAlmostEqual(getattr(model, attribute), getattr(new_model, attribute), attribute)
        new_pth_path = os.path.join(new_model.model_dir, 'best.pth')
        self.assertTrue(os.path.exists(new_pth_path))
        self.assertEqual(new_model.cfg.MODEL.WEIGHTS, new_pth_path)
        self.assertEqual(new_model.cfg.OUTPUT_DIR, new_model.model_dir)
        self.assertEqual(model.cfg.DATALOADER.NUM_WORKERS, new_model.cfg.DATALOADER.NUM_WORKERS)
        self.assertEqual(model.cfg.SOLVER.IMS_PER_BATCH, new_model.cfg.SOLVER.IMS_PER_BATCH)
        self.assertAlmostEqual(model.cfg.SOLVER.BASE_LR, new_model.cfg.SOLVER.BASE_LR)
        self.assertAlmostEqual(model.cfg.MODEL.RPN.IOU_THRESHOLDS[0], new_model.cfg.MODEL.RPN.IOU_THRESHOLDS[0])
        self.assertAlmostEqual(model.cfg.MODEL.RPN.IOU_THRESHOLDS[1], new_model.cfg.MODEL.RPN.IOU_THRESHOLDS[1])
        self.assertEqual(model.cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, new_model.cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE)
        self.assertEqual(model.cfg.MODEL.RPN.NMS_THRESH, new_model.cfg.MODEL.RPN.NMS_THRESH)
        self.assertAlmostEqual(model.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, new_model.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
        self.assertAlmostEqual(model.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST, new_model.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)

        # Clean
        remove_dir(model_dir)
        remove_dir(model_dir_2)


class TrainerRCNNTests(unittest.TestCase):
    '''Main class to test TrainerRCNN'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test01_trainer_rcnn_init(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.TrainerRCNN.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        cfg = model.cfg

        #
        trainer = TrainerRCNN(cfg, length_epoch=100, nb_iter_per_epoch=50)
        self.assertEqual(cfg.OUTPUT_DIR, trainer.output_dir)
        self.assertEqual(trainer.length_epoch, 100)
        self.assertEqual(trainer.nb_iter_per_epoch, 50)

        #
        trainer = TrainerRCNN(cfg, length_epoch=100, nb_iter_per_epoch=50, nb_iter_log_write=100)
        self.assertEqual(trainer.nb_iter_log_write, 100)

        #
        trainer = TrainerRCNN(cfg, length_epoch=100, nb_iter_per_epoch=50, nb_iter_log_display=100)
        self.assertEqual(trainer.nb_iter_log_display, 100)

        #
        trainer = TrainerRCNN(cfg, length_epoch=100, nb_iter_per_epoch=50, nb_log_write_per_epoch=100)
        self.assertEqual(trainer.nb_log_write_per_epoch, 100)

        #
        trainer = TrainerRCNN(cfg, length_epoch=100, nb_iter_per_epoch=50, min_delta_es=0.1)
        self.assertAlmostEqual(trainer.min_delta_es, 0.1)

        #
        trainer = TrainerRCNN(cfg, length_epoch=100, nb_iter_per_epoch=50, patience=3)
        self.assertEqual(trainer.patience, 3)

        #
        trainer = TrainerRCNN(cfg, length_epoch=100, nb_iter_per_epoch=50, restore_best_weights=True)
        self.assertEqual(trainer.restore_best_weights, True)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test02_trainer_rcnn_build_evaluator(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.TrainerRCNN.build_evaluator'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        cfg = model.cfg

        # Nominal case
        evaluator = TrainerRCNN.build_evaluator(cfg, 'dataset_train')
        self.assertTrue(isinstance(evaluator, COCOEvaluator))
        self.assertTrue(hasattr(evaluator, '_metadata'))
        self.assertEqual(evaluator._metadata.name, 'dataset_train')
        self.assertEqual(evaluator._metadata.thing_classes, model.list_classes)
        self.assertTrue(hasattr(evaluator, '_output_dir'))
        self.assertEqual(evaluator._output_dir, os.path.join(model.model_dir, 'inference'))

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test03_trainer_rcnn_build_train_loader(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.TrainerRCNN.build_train_loader'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        cfg = model.cfg

        # Nominal case
        loader = TrainerRCNN.build_train_loader(cfg)
        self.assertTrue(isinstance(loader, AspectRatioGroupedDataset))
        self.assertTrue(hasattr(loader, 'dataset'))
        self.assertTrue(hasattr(loader, 'batch_size'))
        self.assertEqual(loader.batch_size, cfg.SOLVER.IMS_PER_BATCH)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test04_trainer_rcnn_train(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.TrainerRCNN.train'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        model.cfg.SOLVER.MAX_ITER = 1
        model.cfg.INPUT.MIN_SIZE_TRAIN = (400,)  # We try to limit the memory impact

        # Nominal case
        cfg = model.cfg
        trainer = model_detectron_faster_rcnn.TrainerRCNN(cfg, length_epoch=1, nb_iter_per_epoch=1,
                                                          nb_iter_log_write=1, nb_iter_log_display=1,
                                                          nb_log_write_per_epoch=1, min_delta_es=0.,
                                                          patience=5, restore_best_weights=True)
        trainer.train()
        self.assertTrue(os.path.exists(os.path.join(model_dir, 'best.pth')))
        self.assertTrue(os.path.exists(os.path.join(model_dir, 'metrics.json')))
        # Clean
        del model
        remove_dir(model_dir)
        gc.collect()

        # Test early_stopping
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, lr=0.0000025, epochs=2, rpn_restrict_num_regions=4)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        model.cfg.SOLVER.MAX_ITER = 100
        model.cfg.INPUT.MIN_SIZE_TRAIN = (400,)  # We try to limit the memory impact
        cfg = model.cfg
        # We test several values of patience
        patience_values = [1, 4, 10]
        for patience in patience_values:
            trainer = model_detectron_faster_rcnn.TrainerRCNN(cfg, length_epoch=1, nb_iter_per_epoch=1,
                                                              nb_iter_log_write=1, nb_iter_log_display=1,
                                                              nb_log_write_per_epoch=1, min_delta_es=np.inf,
                                                              patience=patience, restore_best_weights=True)
            trainer.train()
            self.assertTrue(os.path.exists(os.path.join(model_dir, 'metrics.json')))
            metrics = model._load_metrics_from_json(os.path.join(model_dir, 'metrics.json'))
            self.assertEqual(len(metrics), patience + 1)
            os.remove(os.path.join(model_dir, 'metrics.json'))
        # Clean
        del model
        remove_dir(model_dir)
        gc.collect()

        # Test nb_write
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, lr=0.0000025, epochs=2, rpn_restrict_num_regions=4)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        # Get data
        data_directory = os.path.join(os.getcwd(), 'test_data', 'test_data_object_detection', 'fruits')
        path_list, bboxes_list, _ = utils.read_folder_object_detection(data_directory, images_ext=('.jpg', '.jpeg', '.png'))
        df_data = pd.DataFrame({'file_path': path_list, 'bboxes': bboxes_list}).copy()
        self.assertEqual(len(df_data), 4, "The input test DataFrame must contain 4 images, otherwise the test wont't work")
        model._register_dataset(df=df_data, data_type='train')
        model._register_dataset(df=df_data, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        # We will test several combinations of max_iter & nb_log_write_per_epoch
        for max_iter in [3, 7]:
            for nb_log_write_per_epoch in [2, 4]:
                model.cfg.SOLVER.MAX_ITER = max_iter
                cfg = model.cfg
                nb_iter_log_write = int(4 / nb_log_write_per_epoch)
                trainer = model_detectron_faster_rcnn.TrainerRCNN(cfg, length_epoch=4, nb_iter_per_epoch=4,
                                                                  nb_iter_log_write=nb_iter_log_write, nb_iter_log_display=1,
                                                                  nb_log_write_per_epoch=nb_log_write_per_epoch,
                                                                  min_delta_es=0., patience=0, restore_best_weights=True)
                trainer.train()
                self.assertTrue(os.path.exists(os.path.join(model_dir, 'metrics.json')))
                metrics = model._load_metrics_from_json(os.path.join(model_dir, 'metrics.json'))
                self.assertEqual(len(metrics), int((max_iter + 1) / nb_iter_log_write))
                os.remove(os.path.join(model_dir, 'metrics.json'))
                del trainer
                gc.collect()

        # Clean
        del model
        remove_dir(model_dir)
        gc.collect()

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test05_trainer_rcnn_write_model_final(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.TrainerRCNN.write_model_final'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        model.cfg.SOLVER.MAX_ITER = 1
        model.cfg.INPUT.MIN_SIZE_TRAIN = (400,)  # We try to limit the memory impact
        cfg = model.cfg
        # Nominal case
        trainer = model_detectron_faster_rcnn.TrainerRCNN(cfg, length_epoch=1, nb_iter_per_epoch=1,
                                                          nb_iter_log_write=1, nb_iter_log_display=1,
                                                          nb_log_write_per_epoch=1, min_delta_es=0.,
                                                          patience=5, restore_best_weights=True)
        trainer.write_model_final()
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best.pth')))

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test06_trainer_rcnn_early_stopping(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.TrainerRCNN.early_stopping'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        model.cfg.SOLVER.MAX_ITER = 1
        model.cfg.INPUT.MIN_SIZE_TRAIN = (400,)  # We try to limit the memory impact
        cfg = model.cfg

        # Nominal case
        trainer = model_detectron_faster_rcnn.TrainerRCNN(cfg, length_epoch=1, nb_iter_per_epoch=1,
                                                          nb_iter_log_write=1, nb_iter_log_display=1,
                                                          nb_log_write_per_epoch=1, min_delta_es=0.,
                                                          patience=5, restore_best_weights=True)
        # We train the trainer to create trainer.storage
        trainer.train()

        # We create a class to mock history
        class MockHistory(object):
            def __init__(self, values_to_give) -> None:
                self.values_to_give = values_to_give
            def values(self):
                return self.values_to_give

        # Tool function to reinitialize the object trainer
        def reset_trainer(min_delta_es, patience, nb_log_write_per_epoch, restore_best_weights):
            trainer.best_loss = np.inf
            trainer.best_epoch = 0
            trainer.min_delta_es = min_delta_es
            trainer.patience = patience
            trainer.nb_log_write_per_epoch = nb_log_write_per_epoch
            trainer.restore_best_weights = restore_best_weights
            if os.path.exists(os.path.join(trainer.output_dir, 'best.pth')):
                os.remove(os.path.join(trainer.output_dir, 'best.pth'))

        # Function to test the early stopping
        def test_early_stopping(min_delta_es, patience, nb_log_write_per_epoch, restore_best_weights, history,
                                rank_early_stopping, error_message):
            # First, we reset the trainer
            reset_trainer(min_delta_es, patience, nb_log_write_per_epoch, restore_best_weights)
            for i in range(len(history)):
                trainer.storage.histories()['validation_total_loss'] = MockHistory(history[:i+1])
                early_stopping = trainer.early_stopping()
                if i <= rank_early_stopping:
                    if i == rank_early_stopping:
                        self.assertTrue(early_stopping, error_message)
                    else:
                        self.assertFalse(early_stopping, error_message)
                    if restore_best_weights:
                        self.assertTrue(os.path.exists(os.path.join(trainer.output_dir, 'best.pth')), error_message)
                    else:
                        self.assertFalse(os.path.exists(os.path.join(trainer.output_dir, 'best.pth')), error_message)

        test_early_stopping(min_delta_es=0.,
                            patience=3,
                            nb_log_write_per_epoch=1,
                            restore_best_weights=True,
                            history=[(5, 0), (4, 0), (3, 0), (2, 0), (1, 0)],
                            rank_early_stopping=np.inf,
                            error_message = "test : early stopping should not trigger")

        # Run tests
        test_early_stopping(min_delta_es=0.,
                            patience=3,
                            nb_log_write_per_epoch=1,
                            restore_best_weights=True,
                            history=[(5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (2, 0), (3, 0), (2, 0)],
                            rank_early_stopping=7,
                            error_message = "test : early stopping should trigger (no min_delta)")

        test_early_stopping(min_delta_es=1.0,
                            patience=3,
                            nb_log_write_per_epoch=1,
                            restore_best_weights=False,
                            history=[(7, 0), (5, 0), (4.5, 0), (4, 0), (3.5, 0), (3, 0), (2.5, 0), (2, 0), (1.5, 0)],
                            rank_early_stopping=np.inf,
                            error_message = "test : early stopping should trigger (with min_delta)")

        test_early_stopping(min_delta_es=1.0,
                            patience=3,
                            nb_log_write_per_epoch=1,
                            restore_best_weights=False,
                            history=[(7, 0), (5, 0), (4.5, 0), (4.3, 0), (4.2, 0), (4.1, 0), (3.1, 0)],
                            rank_early_stopping=4,
                            error_message = "test : early stopping should trigger (with min_delta)")

        test_early_stopping(min_delta_es=0.,
                            patience=5,
                            nb_log_write_per_epoch=1,
                            restore_best_weights=False,
                            history=[(5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (2, 0), (3, 0), (2, 0), (2, 0), (1.5, 0)],
                            rank_early_stopping=9,
                            error_message = "test : early stopping should trigger (no min_delta, another patience)")

        test_early_stopping(min_delta_es=0.,
                            patience=2,
                            nb_log_write_per_epoch=2,
                            restore_best_weights=False,
                            history=[(5, 0), (4, 0), (3, 0), (2, 0), (2, 0), (1, 0), (3, 0), (2, 0), (2, 0), (1.5, 0), (2, 0), (2, 0)],
                            rank_early_stopping=9,
                            error_message = "test : early stopping should trigger (multiple writes per epoch)")

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test07_trainer_rcnn_build_hooks(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.TrainerRCNN.build_hooks'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        cfg = model.cfg

        # Nominal case
        trainer = model_detectron_faster_rcnn.TrainerRCNN(cfg, length_epoch=1, nb_iter_per_epoch=1)
        hooks = trainer.build_hooks()
        self.assertTrue(isinstance(hooks[0], module_hooks.IterationTimer))
        self.assertTrue(isinstance(hooks[1], module_hooks.LRScheduler))
        self.assertTrue(isinstance(hooks[2], model_detectron_faster_rcnn.LossEvalHook))
        self.assertTrue(isinstance(hooks[3], module_hooks.PeriodicWriter))
        self.assertEqual(len(hooks[3]._writers), 2)
        self.assertTrue(isinstance(hooks[3]._writers[0], model_detectron_faster_rcnn.TrainValMetricPrinter))
        self.assertTrue(isinstance(hooks[3]._writers[1], model_detectron_faster_rcnn.TrainValJSONWriter))
        self.assertEqual(hooks[3]._writers[0].nb_iter_log, trainer.nb_iter_log_write)
        self.assertEqual(hooks[3]._writers[1].nb_iter_log, trainer.nb_iter_log_write)
        self.assertTrue(isinstance(hooks[4], module_hooks.PeriodicWriter))
        self.assertEqual(len(hooks[4]._writers), 1)
        self.assertTrue(isinstance(hooks[4]._writers[0], model_detectron_faster_rcnn.TrainValMetricPrinter))
        self.assertEqual(hooks[4]._writers[0].nb_iter_log, trainer.nb_iter_log_display)

        # Clean
        remove_dir(model_dir)


class LossEvalHookTests(unittest.TestCase):
    '''Main class to test LossEvalHook'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test01_loss_eval_hook_init(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.LossEvalHook.__init__'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        cfg = model.cfg

        # Nominal case
        torch_model = build_model(cfg)
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], DatasetMapper(cfg, True))
        loss_eval_hook = model_detectron_faster_rcnn.LossEvalHook(eval_period=2, model=torch_model, data_loader=data_loader)
        self.assertEqual(loss_eval_hook._period, 2)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test02_loss_eval_hook_get_loss(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.LossEvalHook._get_loss'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, lr=0.0000025)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        model.cfg.SOLVER.MAX_ITER = 1
        cfg = model.cfg

        # Nominal case
        # We define a LossEvalHook
        torch_model = build_model(cfg)
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], DatasetMapper(cfg, True))
        loss_eval_hook = model_detectron_faster_rcnn.LossEvalHook(eval_period=1, model=torch_model, data_loader=data_loader)
        # Process each data
        for data in data_loader:
            with EventStorage(10) as storage:
                results = loss_eval_hook._get_loss(data)
            self.assertTrue(isinstance(results, dict))
            set_losses = {'loss_cls', 'loss_box_reg', 'loss_rpn_cls', 'loss_rpn_loc', 'total_loss'}
            self.assertEqual(set_losses, set(results))

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test03_loss_eval_hook_do_loss_eval(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.LossEvalHook._do_loss_eval'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, lr=0.0000025)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        model.cfg.SOLVER.MAX_ITER = 1
        cfg = model.cfg

        # Nominal case
        # We define a LossEvalHook
        torch_model = build_model(cfg)
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], DatasetMapper(cfg, True))
        loss_eval_hook = model_detectron_faster_rcnn.LossEvalHook(eval_period=1, model=torch_model, data_loader=data_loader)
        # We must also instanciate a trainer to fill its storage
        trainer = model_detectron_faster_rcnn.TrainerRCNN(cfg, length_epoch=1, nb_iter_per_epoch=1)
        loss_eval_hook.trainer = trainer
        # Get losses
        with EventStorage(10) as trainer.storage:
            losses = loss_eval_hook._do_loss_eval()
        # Tests
        set_losses = {'loss_cls', 'loss_box_reg', 'loss_rpn_cls', 'loss_rpn_loc', 'total_loss'}
        self.assertEqual(set(losses), set_losses)
        for loss in set_losses:
            self.assertTrue(isinstance(losses[loss], list))
            self.assertAlmostEqual(trainer.storage.histories()[f'validation_{loss}'].values()[0][0], losses[loss][0])
            self.assertEqual(trainer.storage.histories()[f'validation_{loss}'].values()[0][1], 10)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test04_loss_eval_hook_after_step(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.LossEvalHook.after_step'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, lr=0.0000025)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        model.cfg.SOLVER.MAX_ITER = 100
        cfg = model.cfg

        # Nominal case
        # We define a LossEvalHook
        torch_model = build_model(cfg)
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], DatasetMapper(cfg, True))
        loss_eval_hook = model_detectron_faster_rcnn.LossEvalHook(eval_period=3, model=torch_model, data_loader=data_loader)
        # We must also instanciate a trainer to fill its storage
        trainer = model_detectron_faster_rcnn.TrainerRCNN(cfg, length_epoch=1, nb_iter_per_epoch=1)
        loss_eval_hook.trainer = trainer
        # Tests
        set_losses = {'loss_cls', 'loss_box_reg', 'loss_rpn_cls', 'loss_rpn_loc', 'total_loss'}
        for nb_iter in [2, 8, 32, 65, 99]:
            loss_eval_hook.trainer.iter = nb_iter
            with EventStorage(nb_iter) as trainer.storage:
                loss_eval_hook.after_step()
            for loss in set_losses:
                self.assertEqual(len(trainer.storage.histories()[f'validation_{loss}'].values()), 1)
                self.assertEqual(trainer.storage.histories()[f'validation_{loss}'].values()[0][1], nb_iter)
        for nb_iter in [3, 10, 33, 67]:
            loss_eval_hook.trainer.iter = nb_iter
            with EventStorage(nb_iter) as trainer.storage:
                loss_eval_hook.after_step()
            for loss in set_losses:
                self.assertEqual(len(trainer.storage.histories()[f'validation_{loss}'].values()), 0)

        remove_dir(model_dir)


class TrainValMetricPrinterTests(unittest.TestCase):
    '''Main class to test TrainValMetricPrinter'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test01_train_val_metric_printer_init(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.TrainValMetricPrinter.__init__'''

        # Nominal case
        train_val_metric_printer = model_detectron_faster_rcnn.TrainValMetricPrinter(cfg={},
                                                                                     with_valid=False,
                                                                                     nb_iter_log=100,
                                                                                     length_epoch=10,
                                                                                     nb_iter_per_epoch=2)
        self.assertEqual(train_val_metric_printer.cfg, {})
        self.assertEqual(train_val_metric_printer.with_valid, False)
        self.assertEqual(train_val_metric_printer.nb_iter_log, 100)
        self.assertEqual(train_val_metric_printer.length_epoch, 10)
        self.assertEqual(train_val_metric_printer.nb_iter_per_epoch, 2)
        self.assertTrue(hasattr(train_val_metric_printer, 'logger'))
        self.assertTrue(train_val_metric_printer.logger, logging.Logger)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test02_train_val_metric_printer_write(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.TrainValMetricPrinter.write'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, lr=0.0000025)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                             'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        model.cfg.SOLVER.MAX_ITER = 100
        cfg = model.cfg

        # We mock the logger
        class MockLogger(object):
            def __init__(self) -> None:
                self.messages = []
            def info(self, message):
                self.messages.append(message)

        # We also instanciate a trainer to get some info
        trainer = model_detectron_faster_rcnn.TrainerRCNN(cfg, length_epoch=1, nb_iter_per_epoch=1)
        # We test with with_valid to False and nb_iter_log to 100
        train_val_metric_printer = model_detectron_faster_rcnn.TrainValMetricPrinter(cfg=cfg,
                                                                                     with_valid=False,
                                                                                     nb_iter_log=100,
                                                                                     length_epoch=10,
                                                                                     nb_iter_per_epoch=2)
        train_val_metric_printer.logger = MockLogger()
        # We test when we should write in the logger
        for iteration in [99, 199, 299, 399]:
            train_val_metric_printer.logger.messages = []
            with EventStorage(iteration) as trainer.storage:
                train_val_metric_printer.write()
            self.assertEqual(len(train_val_metric_printer.logger.messages), 1)
        # We test when we should not write in the logger
        for iteration in [14, 245, 340, 270]:
            train_val_metric_printer.logger.messages = []
            with EventStorage(iteration) as trainer.storage:
                train_val_metric_printer.write()
            self.assertEqual(len(train_val_metric_printer.logger.messages), 0)

        # We test with with_valid to True and nb_iter_log to 33
        train_val_metric_printer = model_detectron_faster_rcnn.TrainValMetricPrinter(cfg=cfg,
                                                                                     with_valid=True,
                                                                                     nb_iter_log=33,
                                                                                     length_epoch=10,
                                                                                     nb_iter_per_epoch=2)
        train_val_metric_printer.logger = MockLogger()
        # We test when we should write in the logger
        for iteration in [32, 65, (33*342)-1, 98]:
            train_val_metric_printer.logger.messages = []
            with EventStorage(iteration) as trainer.storage:
                train_val_metric_printer.write()
            self.assertEqual(len(train_val_metric_printer.logger.messages), 2)
        # We test when we should not write in the logger
        for iteration in [33*12, 132, 3453, 2312]:
            train_val_metric_printer.logger.messages = []
            with EventStorage(iteration) as trainer.storage:
                train_val_metric_printer.write()
            self.assertEqual(len(train_val_metric_printer.logger.messages), 0)

        # Clean
        remove_dir(model_dir)


class TrainValJSONWriterTests(unittest.TestCase):
    '''Main class to test TrainValJSONWriter'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test01_train_val_json_writer_init(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.TrainValJSONWriter.__init__'''

        # Nominal case
        path_json_file = os.path.join(os.getcwd(), 'test_train_val_json_writer_init.json')
        # We make sure that the file is not present
        if os.path.exists(path_json_file):
            os.remove(path_json_file)
        # Writer
        train_val_json_writer = model_detectron_faster_rcnn.TrainValJSONWriter(json_file=path_json_file,
                                                                               length_epoch=10,
                                                                               nb_iter_per_epoch=100,
                                                                               nb_iter_log=2)
        # Tests
        self.assertEqual(train_val_json_writer.length_epoch, 10)
        self.assertEqual(train_val_json_writer.nb_iter_per_epoch, 100)
        self.assertEqual(train_val_json_writer.nb_iter_log, 2)
        self.assertEqual(train_val_json_writer.json_file, path_json_file)
        self.assertTrue(os.path.exists(path_json_file))

        # Clean
        os.remove(path_json_file)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test02_train_val_json_writer_write(self):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.TrainValJSONWriter.write'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model to get the cfg and give the training datasets
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, lr=0.0000025, nb_log_display_per_epoch=1,
                                                      epochs=4)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        model._register_dataset(df=df, data_type='train')
        model._register_dataset(df=df, data_type='valid')
        model.cfg.DATASETS.TRAIN = ("dataset_train", )
        model.cfg.DATASETS.TEST = ("dataset_valid", )
        model.cfg.SOLVER.MAX_ITER = 1
        model.cfg.INPUT.MIN_SIZE_TRAIN = (400,)  # We try to limit the memory impact
        cfg = model.cfg

        # Nominal case
        set_losses = {'loss_cls', 'loss_box_reg', 'loss_rpn_cls', 'loss_rpn_loc', 'total_loss'}
        # We also instanciate a trainer to get some info
        trainer = model_detectron_faster_rcnn.TrainerRCNN(cfg, length_epoch=1, nb_iter_per_epoch=1,
                                                          nb_iter_log_write=1, nb_log_write_per_epoch=1)
        trainer.train()
        # Target file
        path_json_file = os.path.join(os.getcwd(), 'test_train_val_json_writer_init.json')
        # We make sure that the file is not present
        if os.path.exists(path_json_file):
            os.remove(path_json_file)

        # We define a TrainValJSONWriter with nb_iter_log=3
        train_val_json_writer = model_detectron_faster_rcnn.TrainValJSONWriter(json_file=path_json_file,
                                                                               length_epoch=1,
                                                                               nb_iter_per_epoch=1,
                                                                               nb_iter_log=3)
        # We write the file
        for iteration in range(100):
            with trainer.storage as storage:
                storage.iter = iteration
                train_val_json_writer.write()
        # We test if the file is properly written
        metrics = model._load_metrics_from_json(path_json_file)
        self.assertEqual(list(metrics['iteration']), list(range(2, 100, 3)))
        self.assertEqual(set(metrics.columns), {loss for loss in set_losses}.union({f"validation_{loss}" for loss in set_losses}.union({'iteration'})))
        train_val_json_writer.close()
        # Clean
        os.remove(path_json_file)

        # We define a TrainValJSONWriter with nb_iter_log=7
        train_val_json_writer = model_detectron_faster_rcnn.TrainValJSONWriter(json_file=path_json_file,
                                                                               length_epoch=1,
                                                                               nb_iter_per_epoch=1,
                                                                               nb_iter_log=7)
        # We write the file
        for iteration in range(100):
            with trainer.storage as storage:
                storage.iter = iteration
                train_val_json_writer.write()
        # We test if the file is properly written
        metrics = model._load_metrics_from_json(path_json_file)
        self.assertEqual(list(metrics['iteration']), list(range(6, 100, 7)))
        self.assertEqual(set(metrics.columns), {loss for loss in set_losses}.union({f"validation_{loss}" for loss in set_losses}.union({'iteration'})))
        train_val_json_writer.close()

        # Clean
        os.remove(path_json_file)
        remove_dir(model_dir)


class ModelDetectronFasterRCNNModuleTests(unittest.TestCase):
    '''Main class to test the module model_detectron_faster_rcnn'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    @patch('numpy.random.choice')
    def test01_data_augmentation_mapper_rot90(self, mock_np_random_choice):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.data_augmentation_mapper
        for the argument rot_90'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model in order to use _prepare_dataset_format
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, lr=0.0000025, nb_log_display_per_epoch=1, epochs=4)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        dataset_dict = model._prepare_dataset_format(df, {value: key for key, value in model.dict_classes.items()})[0]

        # We test with a 90 degrees rotation
        mock_np_random_choice.return_value = [90]
        dataset_dict_augmented = model_detectron_faster_rcnn.data_augmentation_mapper(dataset_dict,
                                                                                      horizontal_flip=False,
                                                                                      vertical_flip=False,
                                                                                      rot_90=True)
        self.assertEqual(dataset_dict_augmented['instances']._image_size, (166, 171))
        self.assertTrue(isinstance(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor, torch.Tensor))
        self.assertEqual(dataset_dict_augmented['width'], 171)
        self.assertEqual(dataset_dict_augmented['height'], 166)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[0], 17.)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[1], 11.)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[2], 161.)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[3], 158.)
        self.assertAlmostEqual(dataset_dict_augmented['image'].numpy()[0, 40, 40], 151.)

        # We test with a 270 degrees rotation
        mock_np_random_choice.return_value = [270]
        dataset_dict_augmented = model_detectron_faster_rcnn.data_augmentation_mapper(dataset_dict,
                                                                                      horizontal_flip=False,
                                                                                      vertical_flip=False,
                                                                                      rot_90=True)
        self.assertEqual(dataset_dict_augmented['instances']._image_size, (166, 171))
        self.assertTrue(isinstance(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor, torch.Tensor))
        self.assertEqual(dataset_dict_augmented['width'], 171)
        self.assertEqual(dataset_dict_augmented['height'], 166)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[0], 10.)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[1], 8.)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[2], 154.)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[3], 155.)
        self.assertAlmostEqual(dataset_dict_augmented['image'].numpy()[0, 40, 40], 86.)

        # Clean
        remove_dir(model_dir)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    @patch('detectron2.data.transforms.RandomFlip._rand_range')
    def test02_data_augmentation_mapper_flip(self, mock_rand_range):
        '''Test of {{package_name}}.models_training.object_detectors.model_detectron_faster_rcnn.data_augmentation_mapper for
        the flipping arguments'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # We instanciate a model in order to use _prepare_dataset_format
        model = ModelDetectronFasterRcnnObjectDetector(model_dir=model_dir, lr=0.0000025, nb_log_display_per_epoch=1, epochs=4)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        df = pd.DataFrame([{'file_path': os.path.join('test_data', 'apple_36.jpg'),
                            'bboxes': [{'x1': 8, 'y1': 17, 'x2': 155, 'y2': 161, 'class': 'apple'}]}])
        dataset_dict = model._prepare_dataset_format(df, {value: key for key, value in model.dict_classes.items()})[0]

        # We test with a horizontal flip
        mock_rand_range.return_value = -1
        dataset_dict_augmented = model_detectron_faster_rcnn.data_augmentation_mapper(dataset_dict,
                                                                                      horizontal_flip=True,
                                                                                      vertical_flip=False,
                                                                                      rot_90=False)
        self.assertEqual(dataset_dict_augmented['instances']._image_size, (171, 166))
        self.assertTrue(isinstance(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor, torch.Tensor))
        self.assertEqual(dataset_dict_augmented['width'], 166)
        self.assertEqual(dataset_dict_augmented['height'], 171)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[0], 11.)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[1], 17.)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[2], 158.)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[3], 161.)
        self.assertAlmostEqual(dataset_dict_augmented['image'].numpy()[0, 40, 40], 151.)

        # We test with a vertical flip
        dataset_dict_augmented = model_detectron_faster_rcnn.data_augmentation_mapper(dataset_dict,
                                                                                      horizontal_flip=False,
                                                                                      vertical_flip=True,
                                                                                      rot_90=False)
        self.assertEqual(dataset_dict_augmented['instances']._image_size, (171, 166))
        self.assertTrue(isinstance(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor, torch.Tensor))
        self.assertEqual(dataset_dict_augmented['width'], 166)
        self.assertEqual(dataset_dict_augmented['height'], 171)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[0], 8.)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[1], 10.)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[2], 155.)
        self.assertAlmostEqual(dataset_dict_augmented['instances']._fields['gt_boxes'].tensor[0].numpy()[3], 154.)
        self.assertAlmostEqual(dataset_dict_augmented['image'].numpy()[0, 40, 40], 86.)

        # Clean
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
