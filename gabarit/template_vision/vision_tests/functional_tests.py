#!/usr/bin/env python3
# Starts all functional tests
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

# utils libs
import os
import gc
import sys
import psutil
import shutil
import logging
import subprocess
import numpy as np
import pandas as pd
import importlib.util
from typing import Any
from PIL import Image
from pathlib import Path
from datetime import datetime

import tensorflow as tf

from test_template_vision import utils
from test_template_vision.models_training.classifiers.model_cnn_classifier import ModelCnnClassifier
from test_template_vision.models_training.object_detectors.model_keras_faster_rcnn import ModelKerasFasterRcnnObjectDetector
from test_template_vision.models_training.classifiers.model_transfer_learning_classifier import ModelTransferLearningClassifier
from test_template_vision.models_training.object_detectors.model_detectron_faster_rcnn import ModelDetectronFasterRcnnObjectDetector


class Case1_e2e_pipeline(unittest.TestCase):
    '''Class to test the project end to end'''


    def test01_CreateSamples(self):
        '''Test of the file 0_create_samples.py'''
        print("Test of the file 0_create_samples.py")

        # dataset_v1
        dataset_v1 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_create_samples.py -d dataset_v1 -n 50"
        self.assertEqual(subprocess.run(dataset_v1, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_50_samples','metadata.csv')))
        nb = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_50_samples')))
        self.assertTrue(nb, 51)  # 51 images + csv file
        df = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_v1_50_samples/metadata.csv", sep=';', encoding='utf-8')
        self.assertEqual(df.shape[0], 50)

        # dataset_v2
        dataset_v2 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_create_samples.py -d dataset_v2 -n 50"
        self.assertEqual(subprocess.run(dataset_v2, shell=True).returncode, 0)
        nb = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_50_samples')))
        self.assertTrue(nb, 50)

        # dataset_v3
        dataset_v3 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_create_samples.py -d dataset_v3 -n 50"
        self.assertEqual(subprocess.run(dataset_v3, shell=True).returncode, 0)
        nb1 = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3')))
        nb2 = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_50_samples')))
        self.assertEqual(nb1, nb2)
        filelist = []
        for _, _, files in os.walk(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_50_samples')):
            for file in files:
                filelist.append(file)
        self.assertEqual(len(filelist), 50)

        # dataset_object_detection
        dataset_object_detection = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_create_samples.py -d dataset_object_detection -n 10"
        self.assertEqual(subprocess.run(dataset_object_detection, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_object_detection_10_samples','metadata_bboxes.csv')))
        nb = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_object_detection_10_samples')))
        self.assertTrue(nb, 11)  # 10 images + csv file
        df = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_object_detection_10_samples/metadata_bboxes.csv", sep=';', encoding='utf-8')
        self.assertTrue(df.shape[0] >= 10)  # Potentially several bboxes per image


    def test02_SplitTrainValidTest(self):
        '''Test of the file 0_split_train_valid_test.py'''
        print("Test of the file 0_split_train_valid_test.py")

        # "Basic" case dataset_v1
        dataset_v1 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_split_train_valid_test.py --overwrite -d dataset_v1 --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --seed 42"
        self.assertEqual(subprocess.run(dataset_v1, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_train','metadata.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_valid','metadata.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_test','metadata.csv')))
        nb_train = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_train')))
        self.assertTrue(nb_train, 86)
        nb_valid = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_valid')))
        self.assertTrue(nb_valid, 29)
        nb_test = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_test')))
        self.assertTrue(nb_test, 29)
        df_train = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_v1_train/metadata.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_train.shape[0], 86)
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_v1_valid/metadata.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_valid.shape[0], 29)
        df_test = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_v1_test/metadata.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_test.shape[0], 29)

        # "Basic" case dataset_v2
        dataset_v2 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_split_train_valid_test.py --overwrite -d dataset_v2 --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --seed 42"
        self.assertEqual(subprocess.run(dataset_v2, shell=True).returncode, 0)
        nb_train = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_train')))
        self.assertTrue(nb_train, 86)
        nb_valid = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_valid')))
        self.assertTrue(nb_valid, 29)
        nb_test = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_test')))
        self.assertTrue(nb_test, 29)

        # "Basic" case dataset_v3
        dataset_v3 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_split_train_valid_test.py --overwrite -d dataset_v3 --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --seed 42"
        self.assertEqual(subprocess.run(dataset_v3, shell=True).returncode, 0)
        nb = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3')))
        nb_train = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train')))
        self.assertEqual(nb, nb_train)
        nb_valid = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid')))
        self.assertEqual(nb, nb_valid)
        nb_test = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_test')))
        self.assertEqual(nb, nb_test)
        filelist = []
        for _, _, files in os.walk(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train')):
            for file in files:
                filelist.append(file)
        self.assertEqual(len(filelist), 86)
        filelist = []
        for _, _, files in os.walk(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid')):
            for file in files:
                filelist.append(file)
        self.assertEqual(len(filelist), 29)
        filelist = []
        for _, _, files in os.walk(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_test')):
            for file in files:
                filelist.append(file)
        self.assertEqual(len(filelist), 29)

        # Test of perc_x arguments dataset_v1
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_train'))
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_valid'))
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_test'))
        dataset_v1 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_split_train_valid_test.py --overwrite -d dataset_v1 --split_type random --perc_train 0.3 --perc_valid 0.6 --perc_test 0.1 --seed 42"
        self.assertEqual(subprocess.run(dataset_v1, shell=True).returncode, 0)
        nb_train = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_train')))
        self.assertTrue(nb_train, 43)
        nb_valid = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_valid')))
        self.assertTrue(nb_valid, 87)
        nb_test = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_test')))
        self.assertTrue(nb_test, 14)
        df_train = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_v1_train/metadata.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_train.shape[0], 43)
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_v1_valid/metadata.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_valid.shape[0], 87)
        df_test = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_v1_test/metadata.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_test.shape[0], 14)

        # Test of perc_x arguments dataset_v2
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_train'))
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_valid'))
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_test'))
        dataset_v2 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_split_train_valid_test.py --overwrite -d dataset_v2 --split_type random --perc_train 0.3 --perc_valid 0.6 --perc_test 0.1 --seed 42"
        self.assertEqual(subprocess.run(dataset_v2, shell=True).returncode, 0)
        nb_train = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_train')))
        self.assertTrue(nb_train, 43)
        nb_valid = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_valid')))
        self.assertTrue(nb_valid, 87)
        nb_test = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_test')))
        self.assertTrue(nb_test, 14)

        # Test of perc_x arguments dataset_v3
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train'))
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid'))
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_test'))
        dataset_v3 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_split_train_valid_test.py --overwrite -d dataset_v3 --split_type random --perc_train 0.75 --perc_valid 0.25 --perc_test 0 --seed 42"
        self.assertEqual(subprocess.run(dataset_v3, shell=True).returncode, 0)
        filelist = []
        for _, _, files in os.walk(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train')):
            for file in files:
                filelist.append(file)
        self.assertEqual(len(filelist), 108)
        filelist = []
        for _, _, files in os.walk(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid')):
            for file in files:
                filelist.append(file)
        self.assertEqual(len(filelist), 36)
        self.assertFalse(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_test')))

        # Test split_type stratified dataset_v1
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_train'))
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_valid'))
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_test'))
        dataset_v1 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_split_train_valid_test.py --overwrite -d dataset_v1 --split_type stratified --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --seed 42"
        self.assertEqual(subprocess.run(dataset_v1, shell=True).returncode, 0)
        nb_train = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_train')))
        self.assertTrue(nb_train, 86)
        nb_valid = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_valid')))
        self.assertTrue(nb_valid, 29)
        nb_test = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_test')))
        self.assertTrue(nb_test, 29)
        df_train = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_v1_train/metadata.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_train.shape[0], 86)
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_v1_valid/metadata.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_valid.shape[0], 29)
        df_test = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_v1_test/metadata.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_test.shape[0], 29)
        # Check stratified (on a birman : 34.72%, bombay : 31.94%, shiba : 33.33%)
        self.assertGreater(sum(df_train['class'] == "birman")/df_train.shape[0], 0.33)
        self.assertLess(sum(df_train['class'] == "birman")/df_train.shape[0], 0.365)
        self.assertGreater(sum(df_train['class'] == "bombay")/df_train.shape[0], 0.30)
        self.assertLess(sum(df_train['class'] == "bombay")/df_train.shape[0], 0.345)
        self.assertGreater(sum(df_train['class'] == "shiba")/df_train.shape[0], 0.31)
        self.assertLess(sum(df_train['class'] == "shiba")/df_train.shape[0], 0.35)
        self.assertGreater(sum(df_valid['class'] == "birman")/df_valid.shape[0], 0.33)
        self.assertLess(sum(df_valid['class'] == "birman")/df_valid.shape[0], 0.365)
        self.assertGreater(sum(df_valid['class'] == "bombay")/df_valid.shape[0], 0.30)
        self.assertLess(sum(df_valid['class'] == "bombay")/df_valid.shape[0], 0.345)
        self.assertGreater(sum(df_valid['class'] == "shiba")/df_valid.shape[0], 0.31)
        self.assertLess(sum(df_valid['class'] == "shiba")/df_valid.shape[0], 0.35)
        self.assertGreater(sum(df_test['class'] == "birman")/df_test.shape[0], 0.33)
        self.assertLess(sum(df_test['class'] == "birman")/df_test.shape[0], 0.365)
        self.assertGreater(sum(df_test['class'] == "bombay")/df_test.shape[0], 0.30)
        self.assertLess(sum(df_test['class'] == "bombay")/df_test.shape[0], 0.345)
        self.assertGreater(sum(df_test['class'] == "shiba")/df_test.shape[0], 0.31)
        self.assertLess(sum(df_test['class'] == "shiba")/df_test.shape[0], 0.35)

        # Test split_type stratified dataset_v2
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_train'))
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_valid'))
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_test'))
        dataset_v2 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_split_train_valid_test.py --overwrite -d dataset_v2 --split_type stratified --perc_train 0.75 --perc_valid 0 --perc_test 0.25 --seed 42"
        self.assertEqual(subprocess.run(dataset_v2, shell=True).returncode, 0)
        nb_train = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_train')))
        self.assertTrue(nb_train, 108)
        self.assertFalse(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_valid')))
        nb_test = len(os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_test')))
        self.assertTrue(nb_test, 36)
        # Check stratified (on a birman : 34.72%, bombay : 31.94%, shiba : 33.33%)
        file_train = os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_train'))
        file_train = [f.split('_')[0] for f in file_train]
        birman = [f for f in file_train if f =='birman']
        bombay = [f for f in file_train if f =='bombay']
        shiba = [f for f in file_train if f =='shiba']
        self.assertGreater(len(birman)/len(file_train), 0.33)
        self.assertLess(len(birman)/len(file_train), 0.365)
        self.assertGreater(len(bombay)/len(file_train), 0.30)
        self.assertLess(len(bombay)/len(file_train), 0.345)
        self.assertGreater(len(shiba)/len(file_train), 0.31)
        self.assertLess(len(shiba)/len(file_train), 0.35)
        file_test = os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_test'))
        file_test = [f.split('_')[0] for f in file_test]
        birman = [f for f in file_test if f =='birman']
        bombay = [f for f in file_test if f =='bombay']
        shiba = [f for f in file_test if f =='shiba']
        self.assertGreater(len(birman)/len(file_test), 0.33)
        self.assertLess(len(birman)/len(file_test), 0.365)
        self.assertGreater(len(bombay)/len(file_test), 0.30)
        self.assertLess(len(bombay)/len(file_test), 0.345)
        self.assertGreater(len(shiba)/len(file_test), 0.31)
        self.assertLess(len(shiba)/len(file_test), 0.35)

        # Test split_type stratified dataset_v3
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train'))
        shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid'))
        dataset_v3 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_split_train_valid_test.py --overwrite -d dataset_v3 --split_type stratified --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --seed 42"
        self.assertEqual(subprocess.run(dataset_v3, shell=True).returncode, 0)
        filelist_train = []
        for _, _, files in os.walk(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train')):
            for file in files:
                filelist_train.append(file)
        self.assertEqual(len(filelist_train), 86)
        filelist_valid = []
        for _, _, files in os.walk(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid')):
            for file in files:
                filelist_valid.append(file)
        self.assertEqual(len(filelist_valid), 29)
        filelist_test = []
        for _, _, files in os.walk(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_test')):
            for file in files:
                filelist_test.append(file)
        self.assertEqual(len(filelist_test), 29)
        # Check stratified (on a birman : 34.72%, bombay : 31.94%, shiba : 33.33%)
        file_train_birman = os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train', 'birman'))
        file_train_bombay = os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train', 'bombay'))
        file_train_shiba = os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train', 'shiba'))
        self.assertGreater(len(file_train_birman)/len(filelist_train), 0.33)
        self.assertLess(len(file_train_birman)/len(filelist_train), 0.365)
        self.assertGreater(len(file_train_bombay)/len(filelist_train), 0.30)
        self.assertLess(len(file_train_bombay)/len(filelist_train), 0.345)
        self.assertGreater(len(file_train_shiba)/len(filelist_train), 0.31)
        self.assertLess(len(file_train_shiba)/len(filelist_train), 0.35)
        file_valid_birman = os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid', 'birman'))
        file_valid_bombay = os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid', 'bombay'))
        file_valid_shiba = os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid', 'shiba'))
        self.assertGreater(len(file_valid_birman)/len(filelist_valid), 0.33)
        self.assertLess(len(file_valid_birman)/len(filelist_valid), 0.365)
        self.assertGreater(len(file_valid_bombay)/len(filelist_valid), 0.30)
        self.assertLess(len(file_valid_bombay)/len(filelist_valid), 0.345)
        self.assertGreater(len(file_valid_shiba)/len(filelist_valid), 0.31)
        self.assertLess(len(file_valid_shiba)/len(filelist_valid), 0.35)
        file_test_birman = os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_test', 'birman'))
        file_test_bombay = os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_test', 'bombay'))
        file_test_shiba = os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_test', 'shiba'))
        self.assertGreater(len(file_test_birman)/len(filelist_test), 0.33)
        self.assertLess(len(file_test_birman)/len(filelist_test), 0.365)
        self.assertGreater(len(file_test_bombay)/len(filelist_test), 0.30)
        self.assertLess(len(file_test_bombay)/len(filelist_test), 0.345)
        self.assertGreater(len(file_test_shiba)/len(filelist_test), 0.31)
        self.assertLess(len(file_test_shiba)/len(filelist_test), 0.35)


    def test03_PreProcessData(self):
        '''Test of the file 1_preprocess_data.py'''
        print("Test of the file 1_preprocess_data.py")

        # "Basic" case dataset_v1
        # We check that some images are present
        src = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1', 'Bombay_2.jpg')
        dst_folder = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_train')
        dst = os.path.join(dst_folder, 'Bombay_2.jpg')
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        src = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1', 'Birman_5.jpg')
        dst_folder = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_valid')
        dst = os.path.join(dst_folder, 'Birman_5.jpg')
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        # Preprocessing
        dataset_v1 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/1_preprocess_data.py -d dataset_v1_train dataset_v1_valid"
        self.assertEqual(subprocess.run(dataset_v1, shell=True).returncode, 0)
        # train
        # Check if exists
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_train_preprocess_docs', 'metadata.csv')))
        df = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_v1_train_preprocess_docs/metadata.csv", sep=';', encoding='utf-8')
        self.assertEqual(df.shape[0], 86)
        # Check preprocess (at least one image)
        im = Image.open(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_train_preprocess_docs', 'Bombay_2.png'))
        im = np.array(im)
        self.assertEqual(im.shape, (224, 224, 3))
        self.assertEqual(im[0][0][0], 255)
        # Check pipeline has been saved
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_train_preprocess_docs', 'preprocess_pipeline.conf')))
        # valid
        # Check if exists
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_valid_preprocess_docs', 'metadata.csv')))
        df = pd.read_csv(f"{full_path_lib}/test_template_vision-data/dataset_v1_valid_preprocess_docs/metadata.csv", sep=';', encoding='utf-8')
        self.assertEqual(df.shape[0], 29)
        # Check preprocess (at least one image)
        im = Image.open(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_valid_preprocess_docs', 'Birman_26.png'))
        im = np.array(im)
        self.assertEqual(im.shape, (224, 224, 3))
        self.assertEqual(im[0][0][0], 255)
        # Check pipeline has been saved
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v1_valid_preprocess_docs', 'preprocess_pipeline.conf')))

        # "Basic" case dataset_v2
        dataset_v2 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/1_preprocess_data.py -d dataset_v2_train dataset_v2_test -p no_preprocess"
        self.assertEqual(subprocess.run(dataset_v2, shell=True).returncode, 0)
        self.assertFalse(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_train_preprocess_docs')))
        self.assertFalse(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v2_test_preprocess_docs')))

        # "Basic" case dataset_v3
        # We check that some images are present
        src = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3', 'birman', 'Birman_1.jpg')
        dst = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train', 'birman', 'Birman_1.jpg')
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        src = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3', 'bombay', 'Bombay_2.jpg')
        dst = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train', 'bombay', 'Bombay_2.jpg')
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        src = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3', 'shiba', 'shiba_inu_3.jpg')
        dst = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train', 'shiba', 'shiba_inu_3.jpg')
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        src = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3', 'birman', 'Birman_3.jpg')
        dst = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid', 'birman', 'Birman_3.jpg')
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        src = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3', 'bombay', 'Bombay_7.jpg')
        dst = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid', 'bombay', 'Bombay_7.jpg')
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        src = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3', 'shiba', 'shiba_inu_1.jpg')
        dst = os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid', 'shiba', 'shiba_inu_1.jpg')
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        # Preprocess
        dataset_v3 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/1_preprocess_data.py -d dataset_v3_train dataset_v3_valid -p preprocess_docs"
        self.assertEqual(subprocess.run(dataset_v3, shell=True).returncode, 0)
        # train
        # Check preprocess (at least one image)
        im = Image.open(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train_preprocess_docs', 'birman', 'Birman_1.png'))
        im = np.array(im)
        self.assertEqual(im.shape, (224, 224, 3))
        self.assertEqual(im[0][0][0], 255)
        im = Image.open(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train_preprocess_docs', 'bombay', 'Bombay_2.png'))
        im = np.array(im)
        self.assertEqual(im.shape, (224, 224, 3))
        self.assertEqual(im[0][0][0], 255)
        im = Image.open(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train_preprocess_docs', 'shiba', 'shiba_inu_3.png'))
        im = np.array(im)
        self.assertEqual(im.shape, (224, 224, 3))
        self.assertEqual(im[0][0][0], 255)
        # Check pipeline has been saved
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_train_preprocess_docs', 'preprocess_pipeline.conf')))
        # valid
        # Check preprocess (at least one image)
        im = Image.open(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid_preprocess_docs', 'birman', 'Birman_3.png'))
        im = np.array(im)
        self.assertEqual(im.shape, (224, 224, 3))
        self.assertEqual(im[0][0][0], 255)
        im = Image.open(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid_preprocess_docs', 'bombay', 'Bombay_7.png'))
        im = np.array(im)
        self.assertEqual(im.shape, (224, 224, 3))
        self.assertEqual(im[0][0][0], 255)
        im = Image.open(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid_preprocess_docs', 'shiba', 'shiba_inu_1.png'))
        im = np.array(im)
        self.assertEqual(im.shape, (224, 224, 3))
        self.assertEqual(im[0][0][0], 255)
        # Check pipeline has been saved
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_vision-data', 'dataset_v3_valid_preprocess_docs', 'preprocess_pipeline.conf')))

    def test04_TrainingE2E_classifier(self):
        '''Test of the file 2_training_classifier.py'''
        print("Test of the file 2_training_classifier.py")

        ################
        # Classification
        ################

        # "Basic" case dataset_v1
        dataset_v1 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/2_training_classifier.py -d dataset_v1_train_preprocess_docs --directory_valid dataset_v1_valid_preprocess_docs"
        self.assertEqual(subprocess.run(dataset_v1, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_vision-models', 'model_cnn_classifier') # cnn by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertGreater(len(listdir), 0)

        # "Basic" case dataset_v2
        dataset_v2 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/2_training_classifier.py -d dataset_v2_train"
        self.assertEqual(subprocess.run(dataset_v2, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_vision-models', 'model_cnn_classifier') # cnn by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertGreater(len(listdir), 1)

        # "Basic" case dataset_v3
        dataset_v3 = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/2_training_classifier.py -d dataset_v3_train_preprocess_docs --directory_valid dataset_v3_valid_preprocess_docs"
        self.assertEqual(subprocess.run(dataset_v3, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_vision-models', 'model_cnn_classifier') # cnn by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertGreater(len(listdir), 2)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test05_TrainingE2E_object_detector(self):
        '''Test of the file 2_training_object_detector.py'''
        print("Test of the file 2_training_object_detector.py")

        ##################
        # Object detection
        ##################

        # "Basic" case dataset_object_detection_mini
        # We are forced to overide the default model, otherwise is MUCH too long !
        spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_vision-scripts/2_training_object_detector.py')
        test = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test)
        # Set model
        model_name = 'model_detectron_faster_rcnn_e2e'
        model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
        os.makedirs(model_dir)
        test_model = ModelDetectronFasterRcnnObjectDetector(level_save='HIGH', epochs=1, batch_size=1, rpn_restrict_num_regions=4, model_name=model_name, model_dir=model_dir)
        test_model.cfg.INPUT.MIN_SIZE_TRAIN = (400,)  # On essaie de diminuer la charge mémoire

        # Test it
        test.main(directory='dataset_object_detection_mini', directory_valid='dataset_object_detection_mini', model=test_model)

        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_vision-models', 'model_detectron_faster_rcnn_e2e')
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertGreater(len(listdir), 0)

    def test06_PredictE2E_classifier(self):
        '''Test of the file 3_predict.py with a classifier'''
        print("Test of the file 3_predict.py with a classifier")

        ################
        # Classification
        ################

        # "Basic" case dataset_v1_test
        save_model_dir = os.path.join(full_path_lib, 'test_template_vision-models', 'model_cnn_classifier') # cnn by default
        listdir = os.listdir(os.path.join(save_model_dir))
        model_name = listdir[0]
        fonctionnement_basique = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/3_predict.py -d dataset_v1_test -m {model_name}"
        self.assertEqual(subprocess.run(fonctionnement_basique, shell=True).returncode, 0)
        # Check predictions
        save_predictions_dir = os.path.join(full_path_lib, 'test_template_vision-data', 'predictions', 'dataset_v1_test')
        self.assertTrue(os.path.exists(save_predictions_dir))
        listdir = os.listdir(os.path.join(save_predictions_dir))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[0], 'predictions.csv')))

        # "Basic" case dataset_v2_test
        fonctionnement_basique = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/3_predict.py -d dataset_v2_test -m {model_name}"
        self.assertEqual(subprocess.run(fonctionnement_basique, shell=True).returncode, 0)
        # Check predictions
        save_predictions_dir = os.path.join(full_path_lib, 'test_template_vision-data', 'predictions', 'dataset_v2_test')
        self.assertTrue(os.path.exists(save_predictions_dir))
        listdir = os.listdir(os.path.join(save_predictions_dir))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[0], 'predictions.csv')))

        # "Basic" case dataset_v3_valid
        fonctionnement_basique = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/3_predict.py -d dataset_v3_valid -m {model_name}"
        self.assertEqual(subprocess.run(fonctionnement_basique, shell=True).returncode, 0)
        # Check predictions
        save_predictions_dir = os.path.join(full_path_lib, 'test_template_vision-data', 'predictions', 'dataset_v3_valid')
        self.assertTrue(os.path.exists(save_predictions_dir))
        listdir = os.listdir(os.path.join(save_predictions_dir))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[0], 'predictions.csv')))

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test07_PredictE2E(self):
        '''Test of the file 3_predict.py with an object detector'''
        print("Test of the file 3_predict.py with an object detector")

        ##################
        # Object detection
        ##################

        # "Basic" case dataset_object_detection_mini
        save_model_dir = os.path.join(full_path_lib, 'test_template_vision-models', 'model_detectron_faster_rcnn_e2e')
        listdir = os.listdir(os.path.join(save_model_dir))
        model_name = listdir[0]
        fonctionnement_basique = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/3_predict.py -d dataset_object_detection_mini -m {model_name}"
        self.assertEqual(subprocess.run(fonctionnement_basique, shell=True).returncode, 0)
        # Check predictions
        save_predictions_dir = os.path.join(full_path_lib, 'test_template_vision-data', 'predictions', 'dataset_object_detection_mini')
        self.assertTrue(os.path.exists(save_predictions_dir))
        listdir = os.listdir(os.path.join(save_predictions_dir))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[0], 'predictions.csv')))

    def test08_ReloadModel(self):
        '''Test of the file 0_reload_model.py'''
        print("Test of the file 0_reload_model.py")

        # "Basic" case
        save_model_dir = os.path.join(full_path_lib, 'test_template_vision-models', 'model_cnn_classifier') # cnn by default
        listdir = os.listdir(os.path.join(save_model_dir))
        model_name = listdir[0]
        fonctionnement_basique = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/utils/0_reload_model.py -m {model_name}"
        self.assertEqual(subprocess.run(fonctionnement_basique, shell=True).returncode, 0)
        self.assertGreater(len(listdir), 2)


class ModelMockCnnClassifier(ModelCnnClassifier):
    '''We mock _get_model to create a simpler model for the mnist tests'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def _get_model(self) -> Any:
        '''Gets a model structure'''
        # Get input/output dimensions
        input_shape = (self.width, self.height, self.depth)
        num_classes = len(self.list_classes)
        # Process
        input_layer = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer="he_uniform")(input_layer)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        # Last layer
        out = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')(x)
        # Set model
        model = tf.keras.models.Model(inputs=input_layer, outputs=[out])
        # Set optimizer
        lr = self.keras_params['learning_rate'] if 'learning_rate' in self.keras_params.keys() else 0.001
        decay = self.keras_params['decay'] if 'decay' in self.keras_params.keys() else 0.0
        self.logger.info(f"Learning rate utilisée : {lr}")
        self.logger.info(f"Decay utilisé : {decay}")
        optimizer = tf.keras.optimizers.Adam(lr=lr, decay=decay)
        # Compile model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        if self.logger.getEffectiveLevel() < logging.ERROR:
            model.summary()
        # Try to save model as png if level_save > 'LOW'
        if self.level_save in ['MEDIUM', 'HIGH']:
            self._save_model_png(model)
        # Return
        return model


def test_model_mono_class_mono_label(test_class, test_model):
    '''Generic fonction to test a given model for mono-class/mono-label'''

    # Check files exists
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    test_class.assertEqual(len(test_model.list_classes), 2)
    # Try some functions
    df = pd.DataFrame([os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', '663.png'),  # Un
                       os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', '10.png')]  # Zero
                      , columns=['file_path'])
    df2 = pd.DataFrame([os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', '663.png'),  # Un
                        os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', '10.png'),  # Zero
                        os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', '2.png')]  # Un
                       , columns=['file_path'])
    # predict
    preds = test_model.predict(df)
    test_class.assertEqual(list(preds), ['Un', 'Zero'])
    # predict_proba
    index_un = test_model.list_classes.index('Un')
    index_zero = test_model.list_classes.index('Zero')
    probas = test_model.predict_proba(df)
    test_class.assertGreater(probas[0][index_un], 0.5)
    test_class.assertLess(probas[0][index_zero], 0.5)
    test_class.assertGreater(probas[1][index_zero], 0.5)
    test_class.assertLess(probas[1][index_un], 0.5)
    # predict w/ return_proba=True
    probas2 = test_model.predict(df, return_proba=True)
    test_class.assertGreater(probas2[0][index_un], 0.5)
    test_class.assertLess(probas2[0][index_zero], 0.5)
    test_class.assertGreater(probas2[1][index_zero], 0.5)
    test_class.assertLess(probas2[1][index_un], 0.5)
    # predict_with_proba
    pred_proba = test_model.predict_with_proba(df)
    test_class.assertEqual(list(pred_proba[0]), ['Un', 'Zero'])
    test_class.assertGreater(pred_proba[1][0][index_un], 0.5)
    test_class.assertLess(pred_proba[1][0][index_zero], 0.5)
    test_class.assertGreater(pred_proba[1][1][index_zero], 0.5)
    test_class.assertLess(pred_proba[1][1][index_un], 0.5)
    # get_predict_position
    # position start at 1
    test_class.assertEqual(list(test_model.get_predict_position(df2, ['Un', 'Un', 'Quatre'])), [1, 2, -1])
    # get_classes_from_proba
    test_class.assertEqual(list(test_model.get_classes_from_proba(probas)), ['Un', 'Zero'])
    # get_top_n_from_proba
    with test_class.assertRaises(ValueError):
        test_model.get_top_n_from_proba(probas, n=5) # Only 2 classes in our model
    top_n, top_n_proba = test_model.get_top_n_from_proba(probas, n=2)
    test_class.assertEqual([list(_) for _ in top_n], [['Un', 'Zero'], ['Zero', 'Un']])
    test_class.assertEqual([list(_) for _ in top_n_proba], [[probas[0][index_un], probas[0][index_zero]], [probas[1][index_zero], probas[1][index_un]]])
    # inverse_transform
    test_class.assertEqual(list(test_model.inverse_transform(preds)), ['Un', 'Zero'])


class Case2_MonoClassMonoLabel(unittest.TestCase):
    '''Class to test the mono-class / mono-label case'''


    def setUp(self):
        '''On essaie de clean la mémoire'''
        gc.collect()
        tf.keras.backend.clear_session()


    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for mono-class / mono-label case")

        # Clean repo
        models_dir = os.path.join(full_path_lib, 'test_template_vision-models')
        if os.path.isdir(models_dir):
            shutil.rmtree(models_dir)
            os.mkdir(models_dir)
        data_path = os.path.join(full_path_lib, 'test_template_vision-data')
        for d in os.listdir(data_path):
            if d not in ['dataset_v1', 'dataset_v2', 'dataset_v3', 'mnist_v1', 'mnist_v2', 'mnist_v3', 'dataset_object_detection', 'dataset_object_detection_mini']:
                shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', d))

        # Gen. datasets mnist_v1
        preprocessing = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/1_preprocess_data.py -d mnist_v1"
        # We don't apply the preprocessing on the validation dataset. We will use the train as val in order to simplify the process
        self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)
        # Only keep the 1s and the 0s in order to have 2 classes (for mono class) (remove 'Quatre's)
        df = pd.read_csv(f"{full_path_lib}/test_template_vision-data/mnist_v1_preprocess_docs/metadata.csv", sep=';', encoding='utf-8')
        df = df[~df['class'].isin(['Zero', 'Un'])]
        for f in list(df['filename']):
            os.remove(os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', f))
        # Load only Zero & One
        df = pd.read_csv(f"{full_path_lib}/test_template_vision-data/mnist_v1_preprocess_docs/metadata.csv", sep=';', encoding='utf-8')
        df = df[df['class'].isin(['Zero', 'Un'])]
        df.to_csv(f"{full_path_lib}/test_template_vision-data/mnist_v1_preprocess_docs/metadata.csv", sep=';', encoding='utf-8', index=False)

        # # Gen. datasets mnist_v2
        # preprocessing = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/1_preprocess_data.py -d mnist_v2"
        # # We don't apply the preprocessing on the validation dataset. We will use the train as val in order to simplify the process
        # self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)
        # # Only keep the 1s and the 0s in order to have 2 classes (for mono class)
        # for f in os.listdir(os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v2_preprocess_docs')):
        #     if f.split('_')[0] not in ['Zero', 'Un']:
        #         os.remove(os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v2_preprocess_docs', f))
        #
        # # Gen. datasets mnist_v3
        # preprocessing = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/1_preprocess_data.py -d mnist_v3"
        # # We don't apply the preprocessing on the validation dataset. We will use the train as val in order to simplify the process
        # self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)
        # # Only keep the 1s and the 0s in order to have 2 classes (for mono class)
        # subdir = os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v3_preprocess_docs')
        # for dir in os.listdir(subdir):
        #     if dir not in ['Zero', 'Un']:
        #         shutil.rmtree(os.path.join(subdir, dir))

    def test02_Model_CnnClassifier(self):
        '''Test of the model CNN Classifier'''
        print('            ------------------ >     Test of the model CNN Classifier     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_vision-scripts/2_training_classifier.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_cnn_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = ModelMockCnnClassifier(level_save='HIGH', epochs=30, patience=30, model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(directory='mnist_v1_preprocess_docs', directory_valid='mnist_v1_preprocess_docs', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_CnnClassifier failed')

    def test03_Model_TransferLearningClassifier(self):
        '''Test of the Transfer Learning Classifier model'''
        print('            ------------------ >     Test of the Transfer Learning Classifier model     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_vision-scripts/2_training_classifier.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_transfer_learning_classifier_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = ModelTransferLearningClassifier(level_save='HIGH', epochs=10, patience=10, second_epochs=1, batch_size=1,
                                                         model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(directory='mnist_v1_preprocess_docs', directory_valid='mnist_v1_preprocess_docs', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_TransferLearningClassifier failed')


def test_model_multi_class_mono_label(test_class, test_model):
    '''Generic fonction to test a given model for multi-classes/mono-label'''

    # Check files exists
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    test_class.assertEqual(len(test_model.list_classes), 3)
    # Try some functions
    df = pd.DataFrame([os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', '4.png'),  # Quatre
                       os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', '37.png'),  # Un
                       os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', '13.png')]  # Zero
                      , columns=['file_path'])
    df2 = pd.DataFrame([os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', '4.png'),  # Quatre
                        os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', '37.png'),  # Un
                        os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', '13.png'),  # Zero
                        os.path.join(full_path_lib, 'test_template_vision-data', 'mnist_v1_preprocess_docs', '584.png')]  # Un
                       , columns=['file_path'])
    index_quatre = test_model.list_classes.index('Quatre')
    index_un = test_model.list_classes.index('Un')
    index_zero = test_model.list_classes.index('Zero')
    # predict
    preds = test_model.predict(df)
    test_class.assertEqual(list(preds), ['Quatre', 'Un', 'Zero'])
    # predict_proba
    probas = test_model.predict_proba(df)
    test_class.assertEqual(round(probas.sum(), 3), 3.) # We round for deep learning models
    test_class.assertGreater(probas[0][index_quatre], 1/3)
    test_class.assertLess(probas[0][index_un], probas[0][index_quatre])
    test_class.assertLess(probas[0][index_zero], probas[0][index_quatre])
    test_class.assertLess(probas[1][index_quatre], probas[1][index_un])
    test_class.assertGreater(probas[1][index_un], 1/3)
    test_class.assertLess(probas[1][index_zero], probas[1][index_un])
    test_class.assertLess(probas[2][index_quatre], probas[2][index_zero])
    test_class.assertLess(probas[2][index_un], probas[2][index_zero])
    test_class.assertGreater(probas[2][index_zero], 1/3)
    # predict w/ return_proba=True
    probas2 = test_model.predict(df, return_proba=True)
    test_class.assertEqual(round(probas2.sum(), 3), 3.) # We round for deep learning models
    test_class.assertGreater(probas2[0][index_quatre], 1/3)
    test_class.assertLess(probas2[0][index_un], probas2[0][index_quatre])
    test_class.assertLess(probas2[0][index_zero], probas2[0][index_quatre])
    test_class.assertLess(probas2[1][index_quatre], probas2[1][index_un])
    test_class.assertGreater(probas2[1][index_un], 1/3)
    test_class.assertLess(probas2[1][index_zero], probas2[1][index_un])
    test_class.assertLess(probas2[2][index_quatre], probas2[2][index_zero])
    test_class.assertLess(probas2[2][index_un], probas2[2][index_zero])
    test_class.assertGreater(probas2[2][index_zero], 1/3)
    # predict_with_proba
    pred_proba = test_model.predict_with_proba(df)
    test_class.assertEqual(list(pred_proba[0]), ['Quatre', 'Un', 'Zero'])
    test_class.assertEqual(round(pred_proba[1].sum(), 3), 3.) # We round for deep learning models
    test_class.assertGreater(pred_proba[1][0][index_quatre], 1/3)
    test_class.assertLess(pred_proba[1][0][index_un], pred_proba[1][0][index_quatre])
    test_class.assertLess(pred_proba[1][0][index_zero], pred_proba[1][0][index_quatre])
    test_class.assertLess(pred_proba[1][1][index_quatre], pred_proba[1][1][index_un])
    test_class.assertGreater(pred_proba[1][1][index_un], 1/3)
    test_class.assertLess(pred_proba[1][1][index_zero], pred_proba[1][1][index_un])
    test_class.assertLess(pred_proba[1][2][index_quatre], pred_proba[1][2][index_zero])
    test_class.assertLess(pred_proba[1][2][index_un], pred_proba[1][2][index_zero])
    test_class.assertGreater(pred_proba[1][2][index_zero], 1/3)
    # get_predict_position
    # position start at 1
    predict_pos = test_model.get_predict_position(df2, ['Quatre', 'Un', 'Un', 'Trois'])
    test_class.assertEqual(list(predict_pos[[0, 1, 3]]), [1, 1, -1])
    test_class.assertGreater(predict_pos[2], 1)
    # get_classes_from_proba
    test_class.assertEqual(list(test_model.get_classes_from_proba(probas)), ['Quatre', 'Un', 'Zero'])
    # get_top_n_from_proba
    with test_class.assertRaises(ValueError):
        test_model.get_top_n_from_proba(probas, n=5) # Only 3 classes in our model
    top_n, top_n_proba = test_model.get_top_n_from_proba(probas, n=3)
    test_class.assertEqual([_[0] for _ in top_n], ['Quatre', 'Un', 'Zero'])
    test_class.assertEqual(sorted(top_n[0]), sorted(['Quatre', 'Un', 'Zero']))
    test_class.assertEqual(sorted(top_n[1]), sorted(['Quatre', 'Un', 'Zero']))
    test_class.assertEqual(sorted(top_n[2]), sorted(['Quatre', 'Un', 'Zero']))
    test_class.assertEqual([_[0] for _ in top_n_proba], [probas[0][index_quatre], probas[1][index_un], probas[2][index_zero]])
    # inverse_transform
    test_class.assertEqual(list(test_model.inverse_transform(preds)), ['Quatre', 'Un', 'Zero'])


class Case3_MultiClassMonoLabel(unittest.TestCase):
    '''Class to test the multi-classes / mono-label case'''


    def setUp(self):
        '''On essaie de clean la mémoire'''
        gc.collect()
        tf.keras.backend.clear_session()


    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for the multi-classes / mono-label case")

        # Clean repo
        models_dir = os.path.join(full_path_lib, 'test_template_vision-models')
        if os.path.isdir(models_dir):
            shutil.rmtree(models_dir)
            os.mkdir(models_dir)
        data_path = os.path.join(full_path_lib, 'test_template_vision-data')
        for d in os.listdir(data_path):
            if d not in ['dataset_v1', 'dataset_v2', 'dataset_v3', 'mnist_v1', 'mnist_v2', 'mnist_v3', 'dataset_object_detection', 'dataset_object_detection_mini']:
                shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', d))


        # Gen. datasets mnist_v1
        preprocessing = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/1_preprocess_data.py -d mnist_v1"
        # We don't apply the preprocessing on the validation dataset. We will use the train as val in order to simplify the process
        self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)

        # # Gen. datasets mnist_v2
        # preprocessing = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/1_preprocess_data.py -d mnist_v2"
        # # We don't apply the preprocessing on the validation dataset. We will use the train as val in order to simplify the process
        # self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)
        #
        # # Gen. datasets mnist_v3
        # preprocessing = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/1_preprocess_data.py -d mnist_v3"
        # # We don't apply the preprocessing on the validation dataset. We will use the train as val in order to simplify the process
        # self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)

    def test02_Model_CnnClassifier(self):
        '''Test of the model CNN Classifier'''
        print('            ------------------ >     Test of the model CNN Classifier     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_vision-scripts/2_training_classifier.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model mnist_v1
            model_name = 'model_cnn_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = ModelMockCnnClassifier(level_save='HIGH', epochs=30, patience=30, model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(directory='mnist_v1_preprocess_docs', directory_valid='mnist_v1_preprocess_docs', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_CnnClassifier failed')

    def test03_Model_TransferLearningClassifier(self):
        '''Test of the Transfer Learning Classifier model'''
        print('            ------------------ >     Test of the Transfer Learning Classifier model     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_vision-scripts/2_training_classifier.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model mnist_v1
            model_name = 'model_transfer_learning_classifier_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = ModelTransferLearningClassifier(level_save='HIGH', epochs=10, patience=10, second_epochs=1, batch_size=1,
                                                         model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(directory='mnist_v1_preprocess_docs', directory_valid='mnist_v1_preprocess_docs', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_TransferLearningClassifier failed')


def test_model_object_detector(test_class, test_model):
    '''Generic fonction to test a given model for object detection'''

    # Check files exists
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    # Check nb classes
    test_class.assertEqual(len(test_model.list_classes), 3)
    # For now, we have not identified suitable performance test sufficiently "fast"
    # Thus, we prefer to do fast training and skip the performances tests


class Case4_ObjectDetection(unittest.TestCase):
    '''Class to test the object detection case'''


    def setUp(self):
        '''On essaie de clean la mémoire'''
        gc.collect()
        tf.keras.backend.clear_session()


    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for the object detection case")

        # Clean repo
        models_dir = os.path.join(full_path_lib, 'test_template_vision-models')
        if os.path.isdir(models_dir):
            shutil.rmtree(models_dir)
            os.mkdir(models_dir)
        data_path = os.path.join(full_path_lib, 'test_template_vision-data')
        for d in os.listdir(data_path):
            if d not in ['dataset_v1', 'dataset_v2', 'dataset_v3', 'mnist_v1', 'mnist_v2', 'mnist_v3', 'dataset_object_detection', 'dataset_object_detection_mini']:
                shutil.rmtree(os.path.join(full_path_lib, 'test_template_vision-data', d))


        # Gen. datasets object_detection
        preprocessing = f"{activate_venv}python {full_path_lib}/test_template_vision-scripts/1_preprocess_data.py -d dataset_object_detection_mini"
        # We don't apply the preprocessing on the validation dataset. We will use the train as val in order to simplify the process
        self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test02_Model_Detectron(self):
        '''Test of the model detctron'''
        print('            ------------------ >     Test of the model detectron     /   Object Detection')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_vision-scripts/2_training_object_detector.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'model_detectron_faster_rcnn_object_detector_tests'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We train on only 1 epoch. It is far too long otherwise. We will not evaluate performances
            test_model = ModelDetectronFasterRcnnObjectDetector(level_save='HIGH', epochs=1, batch_size=1, rpn_restrict_num_regions=4, model_name=model_name, model_dir=model_dir)
            test_model.cfg.INPUT.MIN_SIZE_TRAIN = (400,)  # On essaie de diminuer la charge mémoire

            # Test it
            test.main(directory='dataset_object_detection_mini_preprocess_docs', directory_valid='dataset_object_detection_mini_preprocess_docs', model=test_model)
            test_model_object_detector(self, test_model)
        except Exception:
            self.fail('testModel_Detectron failed')

    @unittest.skip('This test should pass but is skipped to avoid OOM in the deployment CI')
    def test03_Model_FasterRCNN(self):
        '''Test of the 'homemade' model Faster RCNN'''
        print('            ------------------ >     Test of the model Faster RCNN maison     /   Object Detection')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_vision-scripts/2_training_object_detector.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model - 1
            model_name = 'model_keras_faster_rcnn_object_detector_detector_tests'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We train on only 1 epoch. It is far too long otherwise. We will not evaluate performances
            test_model_1 = ModelKerasFasterRcnnObjectDetector(level_save='HIGH', model_name=model_name, model_dir=model_dir,
                                                              img_min_side_size=100, epochs=1, batch_size=2, nms_max_boxes=10)

            # Test it
            test.main(directory='dataset_object_detection_mini_preprocess_docs', directory_valid='dataset_object_detection_mini_preprocess_docs', model=test_model_1)
            test_model_object_detector(self, test_model_1)
            tf.keras.backend.clear_session()

            # Set model - 2
            model_name = 'model_keras_faster_rcnn_object_detector_detector_tests'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We put all epochs to 0 on purpose : there should be no training but the model is still initialized
            test_model_2 = ModelKerasFasterRcnnObjectDetector(level_save='HIGH', model_name=model_name, model_dir=model_dir,
                                                              img_min_side_size=100, batch_size=2, nms_max_boxes=10,
                                                              epochs_rpn_trainable_true=0, epochs_classifier_trainable_true=0,
                                                              epochs_rpn_trainable_false=0, epochs_classifier_trainable_false=0)

            # Test it
            test.main(directory='dataset_object_detection_mini_preprocess_docs', directory_valid='dataset_object_detection_mini_preprocess_docs', model=test_model_2)
            test_model_object_detector(self, test_model_2)
            tf.keras.backend.clear_session()
        except Exception:
            self.fail('testModel_FasterRCNN failed')


if __name__ == '__main__':
    # Change directory to script directory parent
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    parentname = str(Path(dname).parent)
    os.chdir(parentname)
    # Manage venv
    full_path_lib = os.path.abspath(os.path.join(os.getcwd(), 'test_template_vision'))
    if os.name == 'nt':
        is_windows = True
        # Windows: activate the virtual environment & continue with the other processes
        activate_venv = f"cd {full_path_lib}/venv_test_template_vision/Scripts & activate & "
    else:
        is_windows = False
        # UNIX : We can't use "source" so we directly call python/pip from the bin of the virtual environment
        activate_venv = f"{full_path_lib}/venv_test_template_vision/bin/"
    # Start tests
    unittest.main()
