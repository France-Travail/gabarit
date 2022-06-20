#!/usr/bin/env python3
# type: ignore

# We ignore the type check on mixins : too complicated

## Definition of a parent class for the  object detection models
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
#
# Classes :
# - ModelObjectDetectorMixin -> Parent class object_detector


# Cf. fix https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Union
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from {{package_name}} import utils

sns.set(style="darkgrid")


class ModelObjectDetectorMixin:
    '''Parent class (Mixin) for the model of type object detector'''

    # Not implemented :
    # -> predict : to be implemented by the parent class using this mixin

    def __init__(self, level_save: str = 'HIGH', **kwargs) -> None:
        '''Initialization of the parent class - Object detector

        Kwargs:
            level_save (str): Level of saving
                LOW: stats + configurations + logger keras - /!\\ The model can't be reused /!\\ -
                MEDIUM: LOW + hdf5 + pkl + plots
                HIGH: MEDIUM + predictions
        Raises:
            ValueError: If the object level_save is not a valid option (['LOW', 'MEDIUM', 'HIGH'])
        '''
        super().__init__(level_save=level_save, **kwargs)  # forwards level_save & all unused arguments

        if level_save not in ['LOW', 'MEDIUM', 'HIGH']:
            raise ValueError(f"The object level_save ({level_save}) is not a valid option (['LOW', 'MEDIUM', 'HIGH'])")

        # Get logger
        self.logger = logging.getLogger(__name__)

        # Model type
        self.model_type = 'object_detector'

        # List of classes to consider (set on fit)
        self.list_classes = None
        self.dict_classes = None

        # Other options
        self.level_save = level_save

    def inverse_transform(self, y) -> list:
        '''Gets a list of classes from predictions.
        Useless here, used solely for compatibility.

        Args:
            y (?): Array-like, shape = [n_samples, n_features], arrays of 0s and 1s
        Returns:
            (list)
        '''
        return list(y) if isinstance(y, np.ndarray) else y

    def get_and_save_metrics(self, y_true: list, y_pred: list, list_files_x: list = None, type_data: str = '', model_logger=None, **kwargs):
        '''Gets and saves the metrics of a model

        Args:
            y_true (list): Bboxes list, one entry corresponds to the bboxes of one file - truth
                format bbox : {'class': ..., 'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}
            y_pred (list): Bboxes list, one entry corresponds to the bboxes of one file - predicted
                format bbox : {'class': ..., 'x1': ..., 'y1': ..., 'x2': ..., 'y2': ..., 'proba': ...}
        Kwargs:
            list_files_x (?): List of input files for the prediction
            type_data (str): Type of the dataset (validation, test, ...)
            model_logger (ModelLogger): Custom class to log the metrics with MLflow
        Returns:
            pd.DataFrame: The dataframe containing statistics
        '''
        # Manage errors
        if len(y_true) != len(y_pred):
            raise ValueError(f"The size of the two lists (y_true et y_pred) must be equal ({len(y_true)} != {len(y_pred)})")
        if list_files_x is not None and len(y_true) != len(list_files_x):
            raise ValueError(f"The size of the two lists (y_true et list_files_x) must be equal ({len(y_true)} != {len(list_files_x)})")

        # Construction dataframe
        if list_files_x is None:
            df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        else:
            df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'file_path': list_files_x})

        # Save a prediction file if wanted
        if self.level_save == 'HIGH':
            file_path = os.path.join(self.model_dir, f"predictions{'_' + type_data if len(type_data) > 0 else ''}.csv")
            if 'file_path' in df.columns:
                df = df.sort_values('file_path')
            df.to_csv(file_path, sep='{{default_sep}}', index=None, encoding='{{default_encoding}}')

        # Print info on missing classes and the impact on metrics
        gt_classes = set([bbox['class'] for bboxes in y_true for bbox in bboxes])
        gt_classes_not_in_model = gt_classes.difference(set(self.list_classes))
        model_classes_not_in_gt = set(self.list_classes).difference(gt_classes)
        # Prints
        if len(gt_classes_not_in_model):
            self.logger.info(f"Classes {gt_classes_not_in_model} are not predicted by the model.")
            self.logger.info("We won't take them into account in the calculation of the metrics.")
        if len(model_classes_not_in_gt):
            self.logger.info(f"Classes {model_classes_not_in_gt} are not present in the dataset used to calculate the metrics.")
            self.logger.info("Metrics on these classes won't be accurate.")

        # Get the classes support
        total_bbox = sum([1 for image in y_true for bbox in image if bbox['class'] in self.list_classes])
        classes_support = {}
        if total_bbox == 0:
            total_bbox = 1
        for cl in self.list_classes:
            classes_support[cl] = sum([bbox['class'] == cl for image in y_true for bbox in image]) / total_bbox

        # Get metrics
        # We use the COCO method to get the Average Precision (AP)
        dict_ap_coco = self._get_coco_ap(y_true, y_pred)

        # Calculate the mean Average Precision (mAP) (weighted or not)
        coco_map = np.mean([value for value in list(dict_ap_coco.values()) if not np.isnan(value)])
        coco_wap = sum([dict_ap_coco[cl] * classes_support[cl] for cl in self.list_classes if classes_support[cl] > 0])

        # Global statistics
        self.logger.info('-- * * * * * * * * * * * * * * --')
        self.logger.info(f"Statistics mAP{' ' + type_data if len(type_data) > 0 else ''}")
        self.logger.info('--------------------------------')
        self.logger.info(f"mean Average Precision (mAP) - COCO method : {round(coco_map, 4)}")
        self.logger.info('--------------------------------')
        self.logger.info(f"weighted Average Precision (wAP) - COCO method : {round(coco_wap, 4)}")
        self.logger.info('--------------------------------')

        # Statistics per classes
        for cl in self.list_classes:
            self.logger.info(f"Class {cl}: AP COCO = {round(dict_ap_coco[cl], 4)} /// Support = {round(classes_support[cl], 4)}")
        self.logger.info('--------------------------------')

        # Construction df_stats
        df_stats = pd.DataFrame(columns=['Label', 'AP COCO', 'Support'])
        df_stats = df_stats.append({'Label': 'All', 'AP COCO': coco_map, 'Support': 1.0}, ignore_index=True)
        for cl in self.list_classes:
            df_stats = df_stats.append({'Label': cl, 'AP COCO': dict_ap_coco[cl], 'Support': classes_support[cl]}, ignore_index=True)

        # Save csv
        file_path = os.path.join(self.model_dir, f"map_coco{'_' + type_data if len(type_data) > 0 else ''}@{round(coco_map, 4)}.csv")
        df_stats.to_csv(file_path, sep='{{default_sep}}', index=False, encoding='{{default_encoding}}')

        # Upload metrics in mlflow (or another)
        if model_logger is not None:
            # TODO : To put in a function
            # Prepare parameters
            label_col = 'Label'
            metrics_columns = [col for col in df_stats.columns if col != label_col]

            # Log labels
            labels = df_stats[label_col].values
            for i, label in enumerate(labels):
                model_logger.log_param(f'Label {i}', label)
            # Log metrics
            ml_flow_metrics = {}
            for i, row in df_stats.iterrows():
                for c in metrics_columns:
                    metric_key = f"{row[label_col]} --- {c}"
                    # Check that mlflow accepts the key, otherwise, replace it
                    if not model_logger.valid_name(metric_key):
                        metric_key = f"Label {i} --- {c}"
                    ml_flow_metrics[metric_key] = row[c]
            # Log metrics
            model_logger.log_metrics(ml_flow_metrics)

        # Return df_stats
        return df_stats

    def _get_coco_ap(self, y_true: list, y_pred: list) -> dict:
        '''Calculate COCO's AP for each of the class and gives the result in a dictionary
        where the keys are the classes and the valeus, the corresponding AP value

         Args:
            y_true (list): Bboxes list, one entry corresponds to the bboxes of one file - truth
                format bbox : {'class': ..., 'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}
            y_pred (list): Bboxes list, one entry corresponds to the bboxes of one file - predicted
                format bbox : {'class': ..., 'x1': ..., 'y1': ..., 'x2': ..., 'y2': ..., 'proba': ...}
        Returns:
            The dictionary containing AP for each class
        '''
        inv_dict_classes = {value: key for key, value in self.dict_classes.items()}
        # Put the bboxes in COCO format
        coco_true = self._put_bboxes_in_coco_format(y_true, inv_dict_classes)
        coco_pred = self._put_bboxes_in_coco_format(y_pred, inv_dict_classes)
        images = [{'id': i + 1} for i in range(len(y_true))]
        categories = [{'id': class_id, 'name': class_name, 'supercategory': 'none'}
                      for class_id, class_name in self.dict_classes.items()]
        dataset_coco_true = {'type': 'instances',
                             'images': images.copy(),
                             'categories': categories.copy(),
                             'annotations': coco_true}
        dataset_coco_pred = {'images': images.copy(),
                             'categories': categories.copy(),
                             'annotations': coco_pred}
        # Call pycocotools API to calculate the AP
        coco_eval = self._get_coco_evaluations(dataset_coco_true, dataset_coco_pred)
        dict_ap = self._get_ap_for_classes(coco_eval)
        return dict_ap

    @classmethod
    def _put_bboxes_in_coco_format(self, bboxes: List[List[dict]], inv_dict_classes: dict) -> List[dict]:
        '''Puts a list of list of bboxes (for example from a prediction) in the right format for pycocotools API.

        Args:
            bboxes (list<list<dict>>) : A list of list of bboxes. The first level of list corresponds to the images and the second level to the
            bboxes of this image.
            inv_dict_classes (dict) : The dictionary of classes in the format {class_name: class_id}
        Returns:
            A list of bboxes
        '''
        annotations = []
        idx_bbox = 1  # WARNING: index begins at 1
        for idx_img, list_bboxes in enumerate(bboxes):
            for bbox in list_bboxes:
                dict_bbox = {'id': idx_bbox,
                             'image_id': idx_img + 1,  # WARNING : index begins at 1
                             'category_id': inv_dict_classes[bbox['class']],
                             'bbox': np.array([bbox['x1'], bbox['y1'], bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1']]),
                             'area': (bbox['y2'] - bbox['y1']) * (bbox['x2'] - bbox['x1']),
                             'iscrowd': 0,
                             'score': bbox.get('proba', 1)}
                idx_bbox += 1
                annotations.append(dict_bbox.copy())
        return annotations

    @classmethod
    def _get_coco_evaluations(self, dataset_coco_true: dict, dataset_coco_pred: dict) -> COCOeval:
        '''Calculates the AP from true and predicted datasets in the COCO format, the returns COCOeval,
        the pycocotools API containing all the results.

        Args:
            dataset_coco_true (dict) : Ground truth bboxes in COCO format
            dataset_coco_pred (dict) : Predicted bboxes in COCO format
        Returns:
            A COCOeval (pycocotools API) containing the AP
        '''
        # Everything on mute ! pycocotools library prints too much logs and there are no level settings
        with utils.HiddenPrints():
            # Put the ground truth bboxes in the pycocotools API
            coco_ds = COCO()
            coco_ds.dataset = dataset_coco_true.copy()
            coco_ds.createIndex()

            # Put the predicted bboxes in the pycocotools API
            coco_dt = COCO()
            coco_dt.dataset = dataset_coco_pred.copy()
            coco_dt.createIndex()

            # Get image IDs
            imgIds = sorted(coco_ds.getImgIds())

            # Set evaluator
            cocoEval = COCOeval(coco_ds, coco_dt, 'bbox')
            cocoEval.params.imgIds = imgIds
            cocoEval.params.useCats = True
            cocoEval.params.iouType = "bbox"

            # Evaluate
            cocoEval.evaluate()
            cocoEval.accumulate()

        # Return evaluator
        return cocoEval

    def _get_ap_for_classes(self, coco_eval: COCOeval) -> dict:
        '''Gets the AP per class from cocoEval, the pycocotools API.

        Args:
            coco_eval (COCOeval) : A pycocotools COCOeval which calculated the AP.
                In this function, we just get them, we do not calculate them
        Returns:
            The dictionary containing the AP for each class
        '''
        # Compute per-category AP
        # from https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/coco_evaluation.html
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(self.dict_classes) == precisions.shape[2]

        # Retrieve APs
        dict_ap = {}
        for idx, name in self.dict_classes.items():
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            dict_ap[name] = ap
        return dict_ap

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save model
        if json_data is None:
            json_data = {}

        json_data['list_classes'] = self.list_classes
        json_data['dict_classes'] = self.dict_classes

        # Save
        super().save(json_data=json_data)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
