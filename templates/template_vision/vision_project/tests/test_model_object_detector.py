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
import json
import shutil
import random
import numpy as np
import pandas as pd

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.monitoring.model_logger import ModelLogger
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.object_detectors import model_object_detector
from {{package_name}}.models_training.object_detectors.model_object_detector import ModelObjectDetectorMixin

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelMockObjectDetector(ModelObjectDetectorMixin, ModelClass):
    '''We need a mock implementation of the Mixin class'''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    def fit(self, df_train: pd.DataFrame, df_valid: pd.DataFrame = None, with_shuffle: bool = True, **kwargs):
        '''Simplified version of fit'''
        set_classes = set()
        for bboxes in df_train['bboxes'].to_dict().values():
            set_classes = set_classes.union({bbox['class'] for bbox in bboxes})
        list_classes = sorted(list(set_classes))
        # Also set dict_classes
        dict_classes = {i: col for i, col in enumerate(list_classes)}
        self.list_classes = list_classes
        self.dict_classes = dict_classes
        self.trained = True
        self.nb_fit += 1
    def predict(self, df_test, **kwargs):
        '''Simplified version of predict'''
        predictions = []
        for i, row in df_test.iterrows():
            tmp_predictions = self.predict_on_name(row['file_path'])
            predictions.append(tmp_predictions.copy())
        predictions = np.array(predictions)
        return predictions
    def predict_on_name(self, name: str):
        if 'toto' in name:
            return [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'class': 'banana', 'proba': 0.8}]
        elif 'titi' in name:
            return [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'class': 'orange', 'proba': 0.34},
                    {'x1': 50, 'y1': 50, 'x2': 100, 'y2': 100, 'class': 'apple', 'proba': 0.5},
                    {'x1': 150, 'y1': 150, 'x2': 200, 'y2': 250, 'class': 'banana', 'proba': 0.8}]
        elif 'tata' in name:
            return [{'x1': 1000, 'y1': 1000, 'x2': 1100, 'y2': 1100, 'class': 'banana', 'proba': 0.7},
                    {'x1': 50, 'y1': 50, 'x2': 100, 'y2': 100, 'class': 'banana', 'proba': 0.9}]
        else:
            return [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'class': 'banana', 'proba': 0.98}]


class ModelClassifierMixinTests(unittest.TestCase):
    '''Main class to test model_classifier'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_object_detector_init(self):
        '''Test of the initialization of {{package_name}}.models_training.object_detectors.model_object_detector.ModelObjectDetectorMixin'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Nominal case
        model = ModelMockObjectDetector(model_dir=model_dir, model_name=model_name)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.list_classes, None)
        self.assertEqual(model.dict_classes, None)
        self.assertEqual(model.model_type, 'object_detector')
        remove_dir(model_dir)

        # Test level_save
        model = ModelMockObjectDetector(model_dir=model_dir, model_name=model_name, level_save='HIGH')
        self.assertEqual(model.level_save, 'HIGH')
        remove_dir(model_dir)
        #
        model = ModelMockObjectDetector(model_dir=model_dir, model_name=model_name, level_save='MEDIUM')
        self.assertEqual(model.level_save, 'MEDIUM')
        remove_dir(model_dir)
        #
        model = ModelMockObjectDetector(model_dir=model_dir, model_name=model_name, level_save='LOW')
        self.assertEqual(model.level_save, 'LOW')
        remove_dir(model_dir)

        # Manage errors
        with self.assertRaises(ValueError):
            ModelMockObjectDetector(model_dir=model_dir, model_name=model_name, level_save='toto')
        remove_dir(model_dir)

    def test02_model_object_detector_get_and_save_metrics(self):
        '''Test of the method {{package_name}}.models_training.object_detectors.model_object_detector.ModelObjectDetectorMixin.get_and_save_metrics'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        type_data = 'test_type_data'

        # Nominal case
        model = ModelMockObjectDetector(model_dir=model_dir, model_name=model_name)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        with open(os.path.join('test_data', 'test_map.json'), 'r') as json_file:
            dict_test = json.load(json_file)
        # We will test several cases
        for index, dict_test_true_pred in dict_test.items():
            # We get info from the json file and we format them
            y_true = dict_test_true_pred['y_true']
            y_pred = dict_test_true_pred['y_pred']
            df_metrics_target = dict_test_true_pred['df_metrics']
            coco_map = df_metrics_target[0][2]
            df_metrics_target = np.array(df_metrics_target, dtype='object')
            error_message = dict_test_true_pred['comment']
            # We use the function to test
            df_metrics = model.get_and_save_metrics(y_true, y_pred, type_data=type_data)
            df_metrics = df_metrics.fillna('None')
            # We test
            np.testing.assert_array_equal(df_metrics_target, df_metrics, error_message)
            file_path = os.path.join(model_dir, f"map_coco{'_' + type_data if len(type_data) > 0 else ''}@{round(coco_map, 4)}.csv")
            self.assertTrue(os.path.exists(file_path))
            df_preds = pd.read_csv(os.path.join(model.model_dir, f'predictions_{type_data}.csv'), sep='{{default_sep}}', encoding='{{default_encoding}}')
            self.assertTrue('y_true' in df_preds.columns)
            self.assertTrue('y_pred' in df_preds.columns)
            self.assertFalse('file_path' in df_preds.columns)
        remove_dir(model_dir)

        # With other parameters
        model = ModelMockObjectDetector(model_dir=model_dir, model_name=model_name)
        model.list_classes = ['1', '2', '3']
        dict_translate_target = {'orange': '1', 'banana': '2', 'apple': '3'}
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        type_data = 'toto'
        model_logger = ModelLogger(
            tracking_uri="http://toto.titi.tata.test",
            experiment_name="test"
        )
        # We will test several cases
        for index, dict_test_true_pred in dict_test.items():
            # We get info from the json file and we format them
            y_true = dict_test_true_pred['y_true']
            for list_bboxes in y_true:
                for bbox in list_bboxes:
                    bbox['class'] = dict_translate_target[bbox['class']]
            y_pred = dict_test_true_pred['y_pred']
            for list_bboxes in y_pred:
                for bbox in list_bboxes:
                    bbox['class'] = dict_translate_target[bbox['class']]
            df_metrics_target = dict_test_true_pred['df_metrics']
            df_metrics_target_translated = []
            for row in df_metrics_target:
                new_row = []
                for value in row:
                    if isinstance(value, str):
                        new_row.append(dict_translate_target.get(value, value))
                    else:
                        new_row.append(value)
                df_metrics_target_translated.append(new_row.copy())
            df_metrics_target = df_metrics_target_translated
            coco_map = df_metrics_target[0][2]
            df_metrics_target = np.array(df_metrics_target, dtype='object')
            error_message = dict_test_true_pred['comment']
            list_files_x = [random.choice(['toto.png', 'titi.png', 'tata.png']) for _ in range(len(y_true))]
            # We use the function to test
            df_metrics = model.get_and_save_metrics(y_true, y_pred, list_files_x=list_files_x, type_data=type_data, model_logger=model_logger)
            df_metrics = df_metrics.fillna('None')
           # We test
            np.testing.assert_array_equal(df_metrics_target, df_metrics, error_message)
            file_path = os.path.join(model_dir, f"map_coco{'_' + type_data if len(type_data) > 0 else ''}@{round(coco_map, 4)}.csv")
            self.assertTrue(os.path.exists(file_path))
            df_preds = pd.read_csv(os.path.join(model.model_dir, f'predictions_{type_data}.csv'), sep='{{default_sep}}', encoding='{{default_encoding}}')
            self.assertTrue('y_true' in df_preds.columns)
            self.assertTrue('y_pred' in df_preds.columns)
            self.assertTrue('file_path' in df_preds.columns)
        remove_dir(model_dir)

    def test03_model_object_detector_get_coco_ap(self):
        '''Test of the method {{package_name}}.models_training.object_detectors.model_object_detector.ModelObjectDetectorMixin._get_coco_ap'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Nominal case
        model = ModelMockObjectDetector(model_dir=model_dir, model_name=model_name)
        model.list_classes = ['orange', 'banana', 'apple']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        with open(os.path.join('test_data', 'test_map.json'), 'r') as json_file:
            dict_test = json.load(json_file)
        # We will test several cases
        for index, dict_test_true_pred in dict_test.items():
            # We get info from the json file and we format them
            y_true = dict_test_true_pred['y_true']
            y_pred = dict_test_true_pred['y_pred']
            error_message = dict_test_true_pred['comment']
            list_metrics_target = dict_test_true_pred['df_metrics'][1:]
            dict_ap_target = {list_metrics_class[0]: list_metrics_class[1] for list_metrics_class in list_metrics_target}
            # We apply the function to test and change the nan in 'None' since this is how they written
            # registered in the json
            dict_ap = model._get_coco_ap(y_true, y_pred)
            self.assertEqual(set(dict_ap_target), set(dict_ap))
            for key, value in dict_ap.items():
                if isinstance(value, float) and np.isnan(value):
                    dict_ap[key] = 'None'
            for key, value in dict_ap.items():
                if isinstance(value, str):
                    self.assertEqual(value, dict_ap_target[key])
                else:
                    self.assertAlmostEqual(value, dict_ap_target[key])
        # Clean
        remove_dir(model_dir)

    def test04_model_object_detector_put_bboxes_in_coco_format(self):
        '''Test of the method {{package_name}}.models_training.object_detectors.model_object_detector._put_bboxes_in_coco_format'''

        # We instanciate inv_dict_classes
        list_classes = ['orange', 'banana', 'apple']
        dict_classes = {i: col for i, col in enumerate(list_classes)}
        inv_dict_classes = {value: key for key, value in dict_classes.items()}
        # We define the inputs and the expected results
        bbox_1 = {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'class': 'banana', 'proba': 0.2}
        annotation_1 = {'id': 1,
                        'image_id': 1,
                        'category_id': 1,
                        'bbox': np.array([0, 0, 100, 100]),
                        'area': 10000,
                        'iscrowd': 0,
                        'score': 0.2}
        bbox_2 = {'x1': 200, 'y1': 200, 'x2': 1100, 'y2': 1100, 'class': 'apple', 'proba':0.8}
        annotation_2 = {'id': 2,
                        'image_id': 1,
                        'category_id': 2,
                        'bbox': np.array([200, 200, 900, 900]),
                        'area': 810000,
                        'iscrowd': 0,
                        'score': 0.8}
        bbox_3 = {'x1': 1200, 'y1': 1200, 'x2': 2100, 'y2': 1400, 'class': 'orange'}
        annotation_3 = {'id': 3,
                        'image_id': 2,
                        'category_id': 0,
                        'bbox': np.array([1200, 1200, 900, 200]),
                        'area': 180000,
                        'iscrowd': 0,
                        'score': 1}

        # Nominal case
        annotations = model_object_detector.ModelObjectDetectorMixin._put_bboxes_in_coco_format([[bbox_1, bbox_2], [bbox_3]], inv_dict_classes)
        annotations_target = [annotation_1, annotation_2, annotation_3]
       # We test
        self.assertEqual(len(annotations), len(annotations_target))
        for i in range(len(annotations)):
            annotation = annotations[i]
            annotation_target = annotations_target[i]
            # We can't test directly the dictionaries because one of the value is a np.array
            self.assertEqual(set(annotation), set(annotation_target))
            for key in annotation:
                value = annotation[key]
                value_target = annotation_target[key]
                if key != 'bbox':
                    self.assertEqual(value, value_target)
                else:
                    np.testing.assert_array_equal(value, value_target)

    def test05_model_object_detector_get_coco_evaluations(self):
        '''Test of the assert of the method {{package_name}}.models_training.object_detectors.model_object_detector.ModelObjectDetectorMixin._get_coco_evaluations
        via a call to {{package_name}}.models_training.object_detectors.model_object_detector.ModelObjectDetectorMixin._get_coco_ap'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Nominal case
        model = ModelMockObjectDetector(model_dir=model_dir, model_name=model_name)
        # We consider a model with only one class
        model.list_classes = ['orange']
        model.dict_classes = {i: col for i, col in enumerate(model.list_classes)}
        # We load the json containing y_true and y_pred
        with open(os.path.join('test_data', 'test_map.json'), 'r') as json_file:
            dict_test = json.load(json_file)
        # We will test several cases
        for index, dict_test_true_pred in dict_test.items():
            # We get the info contained in the json file
            y_true = dict_test_true_pred['y_true']
            y_pred = dict_test_true_pred['y_pred']
            set_classes_true = {bbox['class'] for list_bboxes in y_true for bbox in list_bboxes}
            # If there are more than a class considred, we must raise an error
            if len(set_classes_true) > 1:
                with self.assertRaises(Exception):
                    dict_ap = model._get_coco_ap(y_true, y_pred)
        remove_dir(model_dir)

    def test06_model_object_detector_save(self):
        '''Test of the method {{package_name}}.models_training.object_detectors.model_object_detector.ModelObjectDetectorMixin.save'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # test save
        model = ModelMockObjectDetector(model_dir=model_dir, model_name=model_name)
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
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
        self.assertEqual(configs['librairie'], None)
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        remove_dir(model_dir)

        # test save, level_save = 'LOW'
        model = ModelMockObjectDetector(model_dir=model_dir, model_name=model_name, level_save='LOW')
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        model.save(json_data={'test': 8})
        self.assertFalse(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
