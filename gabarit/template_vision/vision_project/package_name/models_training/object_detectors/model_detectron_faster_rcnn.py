#!/usr/bin/env python3
# type: ignore

## Faster RCNN model (detectron2) - Object detection
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
# - ModelDetectronFasterRcnnObjectDetector -> Faster RCNN model (detectron 2) for object detection


# Cf. fix https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import os
import cv2
import copy
import json
import shutil
import logging
import numpy as np
import pandas as pd
from functools import partial
from typing import Union, List
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Imports detectron
import torch
from detectron2.utils import comm
from detectron2.structures import BoxMode
from detectron2.data import transforms as T
from detectron2.engine.hooks import HookBase
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import hooks as module_hooks
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.utils.events import EventWriter, get_event_storage, EventStorage
from detectron2.data import (DatasetCatalog, MetadataCatalog, Metadata, detection_utils,
                             build_detection_train_loader, build_detection_test_loader,
                             DatasetMapper)

# Import package utils
from {{package_name}} import utils
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.object_detectors.model_object_detector import ModelObjectDetectorMixin  # type: ignore


class ModelDetectronFasterRcnnObjectDetector(ModelObjectDetectorMixin, ModelClass):
    '''Faster RCNN model (detectron2) for object detection'''

    _default_name = 'model_detectron_faster_rcnn_object_detector'

    def __init__(self, epochs: int = 99, batch_size: int = 1, validation_split: float = 0.2, lr: float = 0.00025,
                 min_delta_es: float = 0., patience: int = 5, restore_best_weights: bool = True,
                 data_augmentation_params: Union[dict, None] = None,
                 rpn_min_overlap: float = 0.3, rpn_max_overlap: float = 0.7, rpn_restrict_num_regions: int = 128,
                 roi_nms_overlap_threshold: float = 0.7, pred_bbox_proba_threshold: float = 0.5,
                 pred_nms_overlap_threshold: float = 0.5, nb_log_write_per_epoch: int = 1, nb_log_display_per_epoch: int = 10,
                 **kwargs) -> None:
        '''Initialization of the class (see ModelClass & ModelObjectDetectorMixin for more arguments)

        Args:
            epochs (float): Maximal number of epochs
            batch_size (int): Number of images in a batch when training
            validation_split (float): Validation split fraction
                Only used if there is no validation dataset as input when fitting
            lr (float): Base (because we can use a lr scheduler) learning rate to use
            min_delta_es (float): Minimal change in losses to be considered an amelioration for early stopping
            patience (int): Early stopping patience. Put to 0 to disable early stopping
            restore_best_weights (bool): If True, when the training is done, save the model with the best
                loss on the validation dataset instead of the last model (even if early stopping is disabled)
            data_augmentation_params (dict): Set of allowed data augmentation
            rpn_min_overlap (float): Under this threshold a region is classified as background (RPN model)
            rpn_max_overlap (float): Above this threshold a region is classified as object (RPN model)
            rpn_restrict_num_regions (int): Maximal number of regions to keep as target for the RPN
            roi_nms_overlap_threshold (float): The NMS deletes overlapping ROIs whose IOU is above this threshold
            pred_bbox_proba_threshold (float): Above this threshold (for probabilities), a ROI is considered to be a match
            pred_nms_overlap_threshold (float): When predicting, the NMS deletes overlapping predictions whose IOU is above this threshold
            nb_log_write_per_epoch (int): Number of metrics logs written during one epoch (losses for the train and the valid)
            nb_log_display_per_epoch (int): Number of metrics logs displayed during one epoch (losses for the train only)
        Raises:
            ValueError: If rpn_min_overlap is not in [0, 1]
            ValueError: If rpn_max_overlap is not in [0, 1]
            ValueError: If rpn_min_overlap > rpn_max_overlap
            ValueError: If rpn_restrict_num_regions is not positive
            ValueError: If roi_nms_overlap_threshold is not in [0, 1]
            ValueError: If pred_bbox_proba_threshold is not in [0, 1]
            ValueError: If pred_nms_overlap_threshold is not in [0, 1]
        '''
        # Check errors
        if not 0 <= rpn_min_overlap <= 1:
            raise ValueError(f"The argument rpn_min_overlap ({rpn_min_overlap}) must be between 0 and 1, included")
        if not 0 <= rpn_max_overlap <= 1:
            raise ValueError(f"The argument rpn_max_overlap ({rpn_max_overlap}) must be between 0 and 1, included")
        if rpn_min_overlap > rpn_max_overlap:
            raise ValueError(f"The argument rpn_min_overlap ({rpn_min_overlap}) can't be bigger than rpn_max_overlap ({rpn_max_overlap})")
        if rpn_restrict_num_regions < 1:
            raise ValueError(f"The argument rpn_restrict_num_regions ({rpn_restrict_num_regions}) must be positive")
        if not 0 <= roi_nms_overlap_threshold <= 1:
            raise ValueError(f"The argument roi_nms_overlap_threshold ({roi_nms_overlap_threshold}) must be between 0 and 1, included")
        if not 0 <= pred_bbox_proba_threshold <= 1:
            raise ValueError(f"The argument pred_bbox_proba_threshold ({pred_bbox_proba_threshold}) must be between 0 and 1, included")
        if not 0 <= pred_nms_overlap_threshold <= 1:
            raise ValueError(f"The argument pred_nms_overlap_threshold ({pred_nms_overlap_threshold}) must be between 0 and 1, included")

        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Attributes
        self.validation_split = validation_split

        # Early stopping parameters
        self.min_delta_es = min_delta_es
        self.patience = patience
        self.restore_best_weights = restore_best_weights

        # Data augmentation
        if data_augmentation_params is None:
            data_augmentation_params = {'horizontal_flip': True, 'vertical_flip': True, 'rot_90': True}
        self.data_augmentation_params = data_augmentation_params

        # Parameters to "convert" iterations to epochs
        # Detectron works with a number of iterations (ie. number of batch to use during training)
        # In order to be more uniform with other models, we will use "epochs" rather than iterations
        self.nb_log_write_per_epoch = nb_log_write_per_epoch
        self.epochs = epochs
        self.nb_log_display_per_epoch = nb_log_display_per_epoch

        # Load config & pre-trained model
        self.detectron_config_base_filename = 'Base-RCNN-FPN.yaml'
        self.detectron_config_filename = 'faster_rcnn_R_50_FPN_3x.yaml'
        self.detectron_model_filename = 'model_final_280758.pkl'
        detectron_config_base_path = os.path.join(utils.get_data_path(), 'detectron2_conf_files', self.detectron_config_base_filename)
        detectron_config_path = os.path.join(utils.get_data_path(), 'detectron2_conf_files', self.detectron_config_filename)
        detectron_model_path = os.path.join(utils.get_data_path(), 'detectron2_conf_files', self.detectron_model_filename)
        # Backup URLs if the files do not exist
        {% if detectron_config_base_backup_urls is not none %}detectron_config_base_backup_urls = [
        {%- for item in detectron_config_base_backup_urls %}
            '{{item}}',
        {%- endfor %}
        ]{% else %}detectron_config_base_backup_urls = []{% endif %}
        {% if detectron_config_backup_urls is not none %}detectron_config_backup_urls = [
        {%- for item in detectron_config_backup_urls %}
            '{{item}}',
        {%- endfor %}
        ]{% else %}detectron_config_backup_urls = []{% endif %}
        {% if detectron_model_backup_urls is not none %}detectron_model_backup_urls = [
        {%- for item in detectron_model_backup_urls %}
            '{{item}}',
        {%- endfor %}
        ]{% else %}detectron_model_backup_urls = []{% endif %}
        # Check files availability
        files_available = True
        # For each file, we try to download it if does not exists in the projet
        if not os.path.exists(detectron_config_base_path):
            try:
                self.logger.warning("The base configuration file of the faster RCNN of detectron2 is not present in your data folder.")
                self.logger.warning("Trying to download the file.")
                utils.download_url(detectron_config_base_backup_urls, detectron_config_base_path)
            except ConnectionError:
                self.logger.warning("Can't download the file. You can try to get it manually.")
                self.logger.warning("You can find it here https://github.com/facebookresearch/detectron2/blob/main/configs/Base-RCNN-FPN.yaml")
                self.logger.warning("The model won't work, except if it is 'reloaded'")
                self.cfg = None
                files_available = False
        # /!\ WARNING, the key _BASE_ of the configuration file of the RCNN must point to the base configuration file /!\
        # /!\ It won't work if it is not the case !!! /!\
        if not os.path.exists(detectron_config_path):
            try:
                self.logger.warning("The configuration file of the faster RCNN of detectron2 is not present in your data folder.")
                self.logger.warning("Trying to download the file.")
                utils.download_url(detectron_config_backup_urls, detectron_config_path)
            except ConnectionError:
                self.logger.warning("Can't download the file. You can try to get it manually.")
                self.logger.warning("You can find it here https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
                self.logger.warning("You will have to modify the key _BASE_ to point to the base configuration file")
                self.logger.warning("The model won't work, except if it is 'reloaded'")
                self.cfg = None
                files_available = False
        if not os.path.exists(detectron_model_path):
            try:
                self.logger.warning("The weights file of the faster RCNN of detectron2 is not present in your data folder.")
                self.logger.warning("Trying to download the file.")
                utils.download_url(detectron_model_backup_urls, detectron_model_path)
            except ConnectionError:
                self.logger.warning("Can't download the file. You can try to get it manually.")
                self.logger.warning("You can download the weights here : https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl")
                self.logger.warning("The model won't work, except if it is 'reloaded'")
                self.cfg = None
                files_available = False
        # Load if ok
        if files_available:
            cfg = get_cfg()  # Get base config
            try:
                cfg.merge_from_file(detectron_config_path)  # Merge faster RCNN config
            except Exception:
                self.logger.error("Error when reading model configurations")
                self.logger.error("A common issue is that the key _BASE_ of the configuration file of the RCNN must point to the base configuration file")
                self.logger.error("Check your file 'faster_rcnn_R_50_FPN_3x.yaml'")
                raise
            self.cfg = cfg
            # Weights
            self.cfg.MODEL.WEIGHTS = detectron_model_path
            # Training parameters
            self.cfg.DATALOADER.NUM_WORKERS = 2
            self.cfg.SOLVER.IMS_PER_BATCH = batch_size
            self.cfg.SOLVER.BASE_LR = lr
            self.cfg.MODEL.RPN.IOU_THRESHOLDS = [rpn_min_overlap, rpn_max_overlap]
            self.cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = rpn_restrict_num_regions
            self.cfg.MODEL.RPN.NMS_THRESH = roi_nms_overlap_threshold
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = pred_bbox_proba_threshold
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = pred_nms_overlap_threshold
            # We put outputs in the folder of the model
            self.cfg.OUTPUT_DIR = self.model_dir
            # Check that the GPU is available. Otherwise, CPU
            if not torch.cuda.is_available():
                self.logger.warning("Warning, no GPU detected, the model will use CPU")
                self.cfg.MODEL.DEVICE = "cpu"

    #####################
    # Register datasets
    #####################

    def _register_dataset(self, df: pd.DataFrame, data_type: str) -> None:
        '''Registers a dataset in the global variables used by detectron2

        Args:
            df (pd.DataFrame): Dataset to use
                Must contain the column 'file_path' with the path to an image
                Must contain the column 'bboxes' containing the list of bboxes of the image
            data_type (str): Data type, 'train' or 'valid'
        Raises:
            ValueError: If data_type not in ['train', 'valid']
            ValueError: If the dataframe has no 'file_path' column
            ValueError: If the dataframe has no 'bboxes' column
        '''
        if data_type not in ['train', 'valid']:
            raise ValueError(f"The value {data_type} is not a suitable value for the argument data_type.")
        if 'file_path' not in df.columns:
            raise ValueError("The column 'file_path' is mandatory in the input dataframe")
        if 'bboxes' not in df.columns:
            raise ValueError("The column 'bboxes' is mandatory in the input dataframe")

        # Name of the dataset
        name_dataset = f"dataset_{data_type}"
        # Deletes the dataset in the catalog if already present
        if name_dataset in DatasetCatalog:
            DatasetCatalog.pop(name_dataset)
        if name_dataset in MetadataCatalog:
            MetadataCatalog.pop(name_dataset)
        # Register the dataset in the catalogues
        inv_dict_classes = {value: key for key, value in self.dict_classes.items()}
        DatasetCatalog.register(name_dataset, lambda: self._prepare_dataset_format(df, inv_dict_classes))  # register format : (str, func)
        MetadataCatalog.get(name_dataset).set(thing_classes=self.list_classes)

    def _prepare_dataset_format(self, df: pd.DataFrame, inv_dict_classes: dict) -> list:
        '''Puts the dataframe containing the file paths and the bboxes in the suitable format for detectron2

        Args:
            df (pd.DataFrame): Dataset to use
                Must contain the column 'file_path' with the path to an image
                Must contain the column 'bboxes' containing the list of bboxes of the image
        Returns:
            A list of dictionaris each corresponding to an image (and the associated bboxes)
        '''
        # Dictionary mapping of the classes
        inv_dict_classes = {value: key for key, value in self.dict_classes.items()}
        # Get info from the dataset
        path_list = list(df['file_path'])
        bboxes_list = list(df['bboxes'])
        dataset_dicts = []
        # For each image, we will create a dictionary with the elements expected by detectron2
        for idx, (path, bboxes) in enumerate(zip(path_list, bboxes_list)):
            # Get the height and width of the image
            try:
                height, width = cv2.imread(path).shape[:2]
            except Exception:
                self.logger.warning(f"Can't read image {path}. We will skip it when training.")
                continue
            record = {}
            record["file_name"] = path
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            # Creation of the list of the associated bboxes (annotations)
            objs = []
            for bbox in bboxes:
                bbox_coordinates = [bbox[coord] for coord in ['x1', 'y1', 'x2', 'y2']]
                # We put it in the format expected by detectron2
                obj = {
                    "bbox": bbox_coordinates,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": inv_dict_classes[bbox["class"]],
                    "iscrowd": 0
                }
                objs.append(obj)

            # We register all the bboxes in "annotations" ...
            record["annotations"] = objs
            # ... and we append to the dictionary of the dataset
            dataset_dicts.append(record)
        return dataset_dicts

    #####################
    # Fit
    #####################

    def fit(self, df_train: pd.DataFrame, df_valid: Union[pd.DataFrame, None] = None, with_shuffle: bool = True) -> None:
        '''Trains the model

        Args:
            df_train (pd.DataFrame): Training dataset with columns file_path & bboxes
        Kwargs:
            df_valid (pd.DataFrame): Validation dataset with columns file_path & bboxes
            with_shuffle (boolean): If data must be shuffled before fitting
                This should be used if the target is not shuffled as the split_validation takes the lines in order.
                Thus, the validation set might get classes which are not in the train set ...
        Raises:
            AssertionError: If the same classes are not present when comparing an already trained model
                and a new dataset
        '''

        ##############################################
        # Manage retrain
        ##############################################

        # If a model has already been fitted, we make a new folder in order not to overwrite the existing one !
        # And we save the old conf
        if self.trained:
            # Get src files to save
            src_files = [os.path.join(self.model_dir, "configurations.json")]
            if self.nb_fit > 1:
                for i in range(1, self.nb_fit):
                    src_files.append(os.path.join(self.model_dir, f"configurations_fit_{i}.json"))
            # Change model dir
            self.model_dir = self._get_model_dir()
            self.cfg.OUTPUT_DIR = self.model_dir
            # Get dst files
            dst_files = [os.path.join(self.model_dir, f"configurations_fit_{self.nb_fit}.json")]
            if self.nb_fit > 1:
                for i in range(1, self.nb_fit):
                    dst_files.append(os.path.join(self.model_dir, f"configurations_fit_{i}.json"))
            # Copies
            for src, dst in zip(src_files, dst_files):
                try:
                    shutil.copyfile(src, dst)
                except Exception as e:
                    self.logger.error(f"Impossible to copy {src} to {dst}")
                    self.logger.error("We still continue ...")
                    self.logger.error(repr(e))

        ##############################################
        # Prepare dataset
        # Also extract list of classes
        ##############################################

        # Extract list of classes from df_train
        set_classes = set()
        for bboxes in df_train['bboxes'].to_dict().values():
            set_classes = set_classes.union({bbox['class'] for bbox in bboxes})
        list_classes = sorted(list(set_classes))
        # Also set dict_classes
        dict_classes = {i: col for i, col in enumerate(list_classes)}

        # We make sure that we have str for all classes
        # We do not raise an error, detectron2 will do it
        classes_not_string = {cl for cl in set_classes if not isinstance(cl, str)}
        if len(classes_not_string):
            self.logger.warning(f"Warning, the following classes are not strings : {classes_not_string}. Detectron2 requires that all classes are strings.")

        # Validate classes if already trained, else set them
        if self.trained:
            assert self.list_classes == list_classes, \
                "Error: the new dataset does not match with the already fitted model"
            assert self.dict_classes == dict_classes, \
                "Error: the new dataset does not match with the already fitted model"
        else:
            self.list_classes = list_classes
            self.dict_classes = dict_classes

        # Shuffle training dataset if wanted
        # If not, if no validation is provided, the train_test_split could stay in order
        # Hence, for classification task, we might have classes in the validation data that we never met in the training data
        if with_shuffle:
            df_train = df_train.sample(frac=1.).reset_index(drop=True)

        # Manage the absence of a validation dataset
        if df_valid is None:
            self.logger.warning(f"Attention, pas de jeu de validation. On va donc split le jeu de training (fraction valid = {self.validation_split})")
            df_train, df_valid = train_test_split(df_train, test_size=self.validation_split)

        # We register the train and validation datasets
        self._register_dataset(df=df_train, data_type='train')
        self._register_dataset(df=df_valid, data_type='valid')

        # We give the number of classes to the model
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.list_classes)

        # We give the datasets to use to the model (that we have previously registered in the catalogues)
        self.cfg.DATASETS.TRAIN = ("dataset_train", )
        self.cfg.DATASETS.TEST = ("dataset_valid", )

        # "Translate" the number of iterations to "epoch"
        nb_iter_per_epoch = max(int(len(df_train) / self.cfg.SOLVER.IMS_PER_BATCH), 1)
        # Number of iterations between two log writes
        nb_iter_log_write = max(int(nb_iter_per_epoch / self.nb_log_write_per_epoch), 1)
        # Number of iterations between two log displays
        nb_iter_log_display = max(int(nb_iter_per_epoch / self.nb_log_display_per_epoch), 1)
        # Maximal number of iterations
        nb_max_iter = self.epochs * nb_iter_per_epoch - 1
        self.cfg.SOLVER.MAX_ITER = nb_max_iter

        # We have to change the class attribute because it is used in a class method BEFORE instanciation
        TrainerRCNN.data_augmentation_params.update(self.data_augmentation_params)

        # We train
        trainer = TrainerRCNN(self.cfg,
                              length_epoch=len(df_train),
                              nb_iter_per_epoch=nb_iter_per_epoch,
                              nb_iter_log_write=nb_iter_log_write,
                              nb_iter_log_display=nb_iter_log_display,
                              nb_log_write_per_epoch=self.nb_log_write_per_epoch,
                              min_delta_es=self.min_delta_es,
                              patience=self.patience,
                              restore_best_weights=self.restore_best_weights)
        # Resume to False because we automatically save the best weights in a file,
        # and then we point self.cfg.MODEL.WEIGHTS to this file
        trainer.resume_or_load(resume=False)
        trainer.train()

        # Update train status
        self.trained = True
        self.nb_fit += 1

        # We change the weights path to the post-training weights
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'best.pth')

        # Plots losses & metrics
        if self.level_save in ['MEDIUM', 'HIGH']:
            self._plot_metrics_and_loss()

    #####################
    # Predict
    #####################

    @utils.trained_needed
    def predict(self, df_test: pd.DataFrame, write_images: bool = False,
                output_dir_image: Union[str, None] = None, **kwargs) -> List[List[dict]]:
        '''Predictions on test set - batch size must be equal to 1

        Args:
            df_test (pd.DataFrame): Data to predict, with a column 'file_path'
            write_images (bool): If True, we write images with the predicted bboxes
            output_dir_image (str): Path to which we want to write the predicted images (if write_images is True)
        Returns:
            (list<list<dict>>): list (one entry per image) of list of bboxes
        '''
        # First we take care of the case where we want to write images
        if write_images:
            # Metadata used by the Visualizer to draw bboxes
            metadata = Metadata(name='metadata_for_predict')
            metadata.set(thing_classes=self.list_classes)
            # Prepare the folders to write the images
            # Manage case where output_dir_image is None
            if output_dir_image is None:
                output_dir_image = os.path.join(self.cfg.OUTPUT_DIR, 'inference', 'images')
            # Create folder if it does not exist
            if not os.path.exists(output_dir_image):
                os.makedirs(output_dir_image)

        # We define a predictor
        predictor = DefaultPredictor(self.cfg)
        # For each image...
        list_bbox = []
        for file_path in df_test['file_path']:
            list_bboxes_img = []
            # We open the image
            im = cv2.imread(file_path)
            if im is not None:
                # We predict
                outputs = predictor(im)
                if write_images:
                    # We draw bboxes and we write the image
                    filename = os.path.split(file_path)[-1]
                    v = Visualizer(im[:, :, ::-1], metadata, scale=1.0)
                    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    cv2.imwrite(os.path.join(output_dir_image, filename), v.get_image()[:, :, ::-1])
                # We get the bboxes, the scores and the classes
                boxes = np.array(outputs['instances'].get('pred_boxes').tensor.cpu())
                scores = np.array(outputs['instances'].get('scores').cpu())
                classes = np.array(outputs['instances'].get('pred_classes').cpu())
                # For each bbox predicted
                for idx in range(len(boxes)):
                    # We put it in bbox format
                    coordinates = boxes[idx]
                    bbox = {'x1': coordinates[0], 'y1': coordinates[1], 'x2': coordinates[2],
                            'y2': coordinates[3], 'proba': scores[idx],
                            'class': self.dict_classes[classes[idx]]}
                    # An we append it
                    list_bboxes_img.append(bbox.copy())
            list_bbox.append(list_bboxes_img.copy())
        return list_bbox

    #####################
    # Misc.
    #####################

    def _plot_metrics_and_loss(self, **kwargs) -> None:
        '''Plots interesting metrics from training and saves them in files'''
        # Get metrics from detectron2 info
        path_json_metrics = os.path.join(self.cfg.OUTPUT_DIR, 'metrics.json')
        metrics = self._load_metrics_from_json(path_json_metrics)
        # Manage plots
        plots_path = os.path.join(self.cfg.OUTPUT_DIR, 'plots')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        dict_plots = {'total_loss': {'title': 'Total loss', 'output_filename': 'total_loss'},
                      'loss_cls': {'title': 'Classifier classification loss', 'output_filename': 'loss_cls_classifier'},
                      'loss_box_reg': {'title': 'Classifier regression loss', 'output_filename': 'loss_reg_classifier'},
                      'loss_rpn_cls': {'title': 'RPN classification loss', 'output_filename': 'loss_cls_rpn'},
                      'loss_rpn_loc': {'title': 'RPN regression loss', 'output_filename': 'loss_reg_rpn'}}
        # Plot each metric one by one
        for name_metric, char_metric in dict_plots.items():
            self._plot_one_metric(metrics=metrics,
                                  name_metric=name_metric,
                                  title=char_metric['title'],
                                  output_filename=char_metric['output_filename'],
                                  plots_path=plots_path)

    def _load_metrics_from_json(self, json_path: str) -> pd.DataFrame:
        '''Reads the .json written by the training and puts it in a dataframe

        Args:
            json_path (str) : Path to the .json file
        Returns:
            pd.DataFrame: A dataframe containing all the metrics saved during the training
        '''
        lines = []
        with open(json_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        # We get rid of the lines which do not contain validation_total_loss
        lines = [line for line in lines if 'validation_total_loss' in line]
        metrics = pd.DataFrame(lines)
        metrics = metrics.drop_duplicates()
        return metrics

    def _plot_one_metric(self, metrics: pd.DataFrame, name_metric: str, title: str,
                         output_filename: str, plots_path: str) -> None:
        '''Plots the figure of a metric for the train and validation datasets and saves it.

        Args:
            metrics (pd.DataFrame): The dataframe containing the metrics
            name_metric (str): The name of the metric we want to plot
            title (str): The name we want to give to the plot
            output_filename (str): The name of the file we want to save (without the extension)
            plots_path (str): The path to the plot folder
        '''
        # Get the lists of metrics for the train and validation datasets
        list_train = list(metrics[name_metric])
        list_valid = list(metrics[f'validation_{name_metric}'])
        list_iteration = list(metrics['iteration'])
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(list_iteration, list_train)
        plt.plot(list_iteration, list_valid)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel("Number of iterations")
        plt.legend(['Train', 'Validation'], loc='upper left')
        # Save
        filename = f"{output_filename}.jpeg"
        plt.savefig(os.path.join(plots_path, filename))
        plt.close('all')

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save configuration JSON
        if json_data is None:
            json_data = {}
        # Save attributes & cfg (contains all params)
        json_data['librairie'] = 'detectron2'
        json_data['validation_split'] = self.validation_split
        json_data['min_delta_es'] = self.min_delta_es
        json_data['patience'] = self.patience
        json_data['restore_best_weights'] = self.restore_best_weights
        json_data['data_augmentation_params'] = self.data_augmentation_params
        json_data['nb_log_write_per_epoch'] = self.nb_log_write_per_epoch
        json_data['epochs'] = self.epochs
        json_data['nb_log_display_per_epoch'] = self.nb_log_display_per_epoch
        json_data['detectron_config_base_filename'] = self.detectron_config_base_filename
        json_data['detectron_config_filename'] = self.detectron_config_filename
        json_data['detectron_model_filename'] = self.detectron_model_filename
        json_data['cfg'] = self.cfg

        # We save le model with CPU so that there is no problem later
        # when we use the model (with streamlit for example)
        device = self.cfg.MODEL.DEVICE
        self.cfg.MODEL.DEVICE = "cpu"
        super().save(json_data=json_data)
        # We undo what we just did
        self.cfg.MODEL.DEVICE = device

    def reload_from_standalone(self, **kwargs) -> None:
        '''Loads a model from its configuration and the weights of the network
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            pth_path (str): Path to pth file
        Raises:
            ValueError: If configuration_path is None
            ValueError: If pth_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object pth_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        pth_path = kwargs.get('pth_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if pth_path is None:
            raise ValueError("The argument pth_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"The file {pth_path} does not exist")

        # Load confs
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        # Can't set int as keys in json, so need to cast it after reloading
        # dict_classes keys are always ints
        if 'dict_classes' in configs.keys():
            configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}
        elif 'list_classes' in configs.keys():
            configs['dict_classes'] = {i: col for i, col in enumerate(configs['list_classes'])}

        # Set class vars
        # self.model_name = # Keep the created name
        # self.model_dir = # Keep the created folder
        self.nb_fit = configs.get('nb_fit', 1)  # Consider one unique fit by default
        self.trained = configs.get('trained', True)  # Consider trained by default
        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['model_type', 'validation_split', 'min_delta_es', 'patience', 'restore_best_weights',
                          'list_classes', 'dict_classes', 'data_augmentation_params', 'level_save',
                          'nb_log_write_per_epoch', 'epochs', 'nb_log_display_per_epoch', 'detectron_config_base_filename',
                          'detectron_config_filename', 'detectron_model_filename', 'cfg']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Transform cfg into CfgNode
        self.cfg = CfgNode(init_dict=self.cfg)

        # Save best pth in new folder
        new_pth_path = os.path.join(self.model_dir, 'best.pth')
        shutil.copyfile(pth_path, new_pth_path)

        # Reload model
        self.cfg.MODEL.WEIGHTS = new_pth_path

        # Change output path
        self.cfg.OUTPUT_DIR = self.model_dir


class TrainerRCNN(DefaultTrainer):
    '''We overload the class DefaultTraine in order to:
        - change when we save the metrics
        - use the COCOevaluator
        - do data augmentation
        - save the validation metrics when training
        - do early stopping

    '''
    # We define a class attribute because it is used by a class method
    data_augmentation_params = {}

    def __init__(self, cfg, length_epoch: int, nb_iter_per_epoch: int,
                 nb_iter_log_write: int = 20, nb_iter_log_display: int = 20, nb_log_write_per_epoch: int = 1,
                 min_delta_es: float = 0., patience: int = 0, restore_best_weights: bool = False) -> None:
        '''Initialize the Trainer

        Args:
            cfg: Configuration to use
            length_epoch (int): Number of image in an epoch
            nb_iter_per_epoch (int): Number of iterations in an epoch
        Kwargs:
            nb_iter_log_write (int): Number of iterations between two log writes (losses
                on train and validation datasets)
            nb_iter_log_display (int): Number of iterations between two log displays (losses
                on train dataset only)
            nb_log_write_per_epoch (int): Number of metrics logs written during
                one epoch (losses for the train and the valid)
            min_delta_es (float): Minimal change in losses to be considered an amelioration for early stopping
            patience (int): Early stopping patience. Put to 0 to disable early stopping
            restore_best_weights (bool): If True, when the training is done, save the model with the best
                loss on the validation dataset instead of the last model (even if early stopping is disabled)
        '''
        # We must add the definition of some attributes before the super() because we redefine the method
        # build_hooks which is called by super().__init__ and uses them
        self.nb_iter_log_write = nb_iter_log_write
        self.nb_iter_log_display = nb_iter_log_display
        self.length_epoch = length_epoch
        self.output_dir = cfg.OUTPUT_DIR
        self.nb_iter_per_epoch = nb_iter_per_epoch
        super().__init__(cfg)

        # Params early stopping
        self.min_delta_es = min_delta_es
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_loss = np.inf
        self.best_epoch = 0

        # Misc.
        self.nb_log_write_per_epoch = nb_log_write_per_epoch

    @classmethod
    def build_evaluator(self, cfg, dataset_name: str) -> COCOEvaluator:
        '''We redefine the method in order to use the COCOevaluator

        Args:
            cfg: Training configuration
            dataset_name (str): Name of the dataset
        Returns:
            Evaluator to use
        '''
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(self, cfg):
        '''We redefine the method in order to use our own data augmentation

        Args:
            cfg: Training configuration
        Returns:
            train loader to use, with our own data augmentation
        '''
        horizontal_flip = self.data_augmentation_params.get('horizontal_flip', False)
        vertical_flip = self.data_augmentation_params.get('vertical_flip', False)
        rot_90 = self.data_augmentation_params.get('rot_90', False)
        mapper = partial(data_augmentation_mapper, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip, rot_90=rot_90)
        return build_detection_train_loader(cfg, mapper=mapper)

    def train(self):
        '''Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        '''
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))
        self.iter = self.start_iter

        # From https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/train_loop.html#TrainerBase
        # The difference is that we get the result of the early stopping and we stop if triggered
        # We also save the final model if restore_best_weights is set to False
        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                while self.iter < self.max_iter:  # We substitute the for by a while in order to break easily with the early stopping
                    self.before_step()
                    self.run_step()
                    test_early_stopping = self.after_step()
                    if test_early_stopping:
                        logger.info("Early stopping")
                        # We change the number of maximum iteration if we want to stop early
                        # Warning, some hooks are defined with anoter self.max_iter
                        # since build_hooks is called by the __init__ of the trainer
                        self.max_iter = self.iter
                    else:
                        self.iter += 1
                # self.iter == self.max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.

            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
                self.write_model_final()  # * NEW *

        # From https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/defaults.html#DefaultTrainer
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def write_model_final(self) -> None:
        '''If self.restore_best_weights == False, no model is saved during
        training. Thus we save the final model with the name best.pth
        '''

        path_model_best = os.path.join(self.cfg.OUTPUT_DIR, 'best.pth')
        if not os.path.exists(path_model_best):
            self.checkpointer.save("best")

    def after_step(self) -> bool:
        '''Function triggered after each step

        Returns:
            bool: If early stopping has been triggered
        '''
        # We trigger the hooks
        for h in self._hooks:
            h.after_step()
        # We add the early stopping to the hooks
        test_early_stopping = self.early_stopping()
        return test_early_stopping

    def early_stopping(self) -> bool:
        '''Triggers if the condition for early stopping are met. We think in term of epoch. Thus
        the patience is indeed the number of epochs without amelioration

        Returns:
            bool: If early stopping has been triggered
        '''
        # We get all the validation losses
        val_loss_values = self.storage.histories()['validation_total_loss'].values()
        # We only keep those that are at the end of an epoch
        val_loss_values_epoch = val_loss_values[self.nb_log_write_per_epoch-1::self.nb_log_write_per_epoch]
        val_loss_values_epoch = [x[0] for x in val_loss_values_epoch]
        if len(val_loss_values_epoch):
            # We get the minimal loss and the corresponding epoch
            min_loss = np.min(val_loss_values_epoch)
            min_epoch = np.argmin(val_loss_values_epoch)
            # If the loss is better, we save the new minimal loss and the corresponding epoch
            if min_loss < self.best_loss - self.min_delta_es:
                self.best_loss = min_loss
                self.best_epoch = min_epoch
                # We save the model with the best weights
                if self.restore_best_weights:
                    self.checkpointer.save("best")
            # If the patience is up, we trigger early stopping
            if self.best_epoch + self.patience + 1 <= len(val_loss_values_epoch) and self.patience > 0:
                return True
        # Otherwise we do not trigger it
        return False

    def build_hooks(self) -> list:
        '''Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        We rewrite this methos (instead of overloading it) so that we can change
        when we save metrics
        From : https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/defaults.html#DefaultTrainer

        Warning, we deleted the hook on the checkpoint, the early stopping hook takes care of it now !

        Returns:
            list[HookBase]:
        '''
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            module_hooks.IterationTimer(),
            module_hooks.LRScheduler()
        ]

        # We add our custom hook in order to add validation losses
        ret.append(LossEvalHook(
            self.nb_iter_log_write,
            self.model,
            build_detection_test_loader(  # Use of our decorator of build_detection_test_loader
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)  # We keep is_train to true to keep the bboxes (needed for losses calculations)
            )
        ))

        if comm.is_main_process():
            # Here we take care of the writers and the printers
            # There is a printer which displays the results of train and val when we reach
            # a particular number of iteration (self.nb_iter_log_write)
            # There is a writer which writes the results of train and val when we reach
            # a particular number of iteration (self.nb_iter_log_write)
            ret.append(module_hooks.PeriodicWriter([TrainValMetricPrinter(cfg=self.cfg,
                                                                          with_valid=True,
                                                                          length_epoch=self.length_epoch,
                                                                          nb_iter_per_epoch=self.nb_iter_per_epoch,
                                                                          nb_iter_log=self.nb_iter_log_write),
                                                    TrainValJSONWriter(os.path.join(self.output_dir, "metrics.json"),
                                                                       self.length_epoch,
                                                                       self.nb_iter_per_epoch,
                                                                       self.nb_iter_log_write)
                                                    ], period=self.nb_iter_log_write))
            # There is a printer which displays the results of train when we reach
            # a particular number of iteration (self.nb_iter_log_display)
            ret.append(module_hooks.PeriodicWriter([TrainValMetricPrinter(cfg=self.cfg,
                                                                          with_valid=False,
                                                                          length_epoch=self.length_epoch,
                                                                          nb_iter_per_epoch=self.nb_iter_per_epoch,
                                                                          nb_iter_log=self.nb_iter_log_display)], period=1))
        return ret


# FROM https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
# Create the class to calculate the loss on validation data during training.
class LossEvalHook(HookBase):
    '''Hook to save the metrics and losses on the validation dataset'''

    def __init__(self, eval_period: int, model, data_loader) -> None:
        '''Initialization of the class

        Args:
            eval_period (int) : Number of iteration between two losses calculation
            model : Considered model
            data_loader : A dataloader containing the validation data
        '''
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self) -> dict:
        '''Calculates the losses on the validation dataset. Saves them in the storage and
        returns them.

        Return:
            the dict containing the losses

        '''
        # Name of the considered losses
        list_losses = ['loss_cls', 'loss_box_reg', 'loss_rpn_cls', 'loss_rpn_loc', 'total_loss']
        losses = {name_loss: [] for name_loss in list_losses}
        # For each batch in the data_loader...
        for inputs in self._data_loader:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Waits for all kernels in all streams on a CUDA device to complete
            # ... we calculates the losses...
            loss_batch = self._get_loss(inputs)
            # ... and we add them to the dictionary which save the results for each batch
            for name_loss in list_losses:
                losses[name_loss].append(loss_batch[name_loss])
        # We get the mean of the losses on the batches
        mean_losses = {name_loss: np.mean(list_losses_batch) for name_loss, list_losses_batch in losses.items()}
        # We save the losses in the storage of the trainer
        for name_loss, value_loss in mean_losses.items():
            self.trainer.storage.put_scalar(f'validation_{name_loss}', value_loss)
        comm.synchronize()
        return losses

    def _get_loss(self, data) -> dict:
        '''Calculates the losses corresponding to data

        Args:
            data: The data on which to calculate the losses
        Returns:
            dict: A dictionary containing all the losses
        '''
        # Calculates the losses
        metrics_dict = self._model(data)
        # Cast the losses to float and put them in a dictionary
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # Calculate the total loss and add it to the dictionary of losses
        total_loss = sum(loss for loss in metrics_dict.values())
        metrics_dict['total_loss'] = total_loss
        return metrics_dict

    def after_step(self) -> None:
        '''After the training step, check if we are at an iteration where we
        should calculates the losses. If it is the case, calculate them and save
        them in the storage.
        '''
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()


class TrainValMetricPrinter(EventWriter):
    '''Takes care of displaying the metrics on the train (and also on the val)'''

    def __init__(self, cfg, with_valid: bool, length_epoch: int, nb_iter_per_epoch: int, nb_iter_log: int = 20) -> None:
        '''Initialize the class.

        Args:
            cfg: Model configuration
            with valid (bool): If true, also displays the results on the validation dataset
            length_epoch (int): Number of images in an "epoch"
            nb_iter_per_epoch (int): Number of iterations in an "epoch"
        Kwargs:
            nb_iter_log (int): Number of iteration between two displays
        '''
        self.logger = logging.getLogger(__name__)
        self.with_valid = with_valid
        self.nb_iter_log = nb_iter_log
        self.length_epoch = length_epoch
        self.nb_iter_per_epoch = nb_iter_per_epoch
        self.cfg = cfg

    def write(self):
        '''Prints the wanted info'''
        storage = get_event_storage()
        iteration = storage.iter
        # Calculates a number of "epoch" (not necessarily an integer)
        nb_epoch = ((iteration * self.cfg.SOLVER.IMS_PER_BATCH) + 1) / self.length_epoch
        if (iteration + 1) % self.nb_iter_log == 0:
            try:
                lr = "{:.5g}".format(storage.history("lr").latest())
            except KeyError:
                lr = "N/A"
            if torch.cuda.is_available():
                max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            else:
                max_mem_mb = None
            # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
            losses = [loss for loss in storage.histories() if 'loss' in loss]
            losses_valid = [loss for loss in losses if 'validation' in loss]
            losses_train = [loss for loss in losses if loss not in losses_valid]
            # Logs train results
            self.logger.info(
                " iter: {iter}, epoch: {nb_epoch}  {losses}  lr: {lr}  {memory}".format(
                    iter=iteration,
                    losses="  ".join(["{}: {:.4g}".format(k, storage.histories()[k].median(self.nb_iter_per_epoch)) for k in losses_train]),
                    lr=lr,
                    memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
                    nb_epoch=nb_epoch
                )
            )
            if self.with_valid:
                # Logs val results
                self.logger.info(
                    "VALIDATION iter: {iter}, epoch: {nb_epoch}  {losses}".format(
                        iter=iteration,
                        losses="  ".join(["{}: {:.4g}".format(k, storage.histories()[k].latest()) for k in losses_valid]),
                        nb_epoch=nb_epoch
                    )
                )


class TrainValJSONWriter(EventWriter):
    '''Write scalars to a json file.
    It saves scalars as one json per line (instead of a big json) for easy parsing.
    '''

    def __init__(self, json_file: str, length_epoch: int, nb_iter_per_epoch: int, nb_iter_log: int = 20) -> None:
        '''Initialization of the class

        Args:
            json_file (str): File where we save the results
            length_epoch (int): Number of images in an "epoch"
            nb_iter_per_epoch (int): Number of iterations in an "epoch"
        Kwargs:
            nb_iter_log (int): Number of iteration between two writes
        '''
        self.json_file = json_file
        self.length_epoch = length_epoch
        self.nb_iter_per_epoch = nb_iter_per_epoch
        self.nb_iter_log = nb_iter_log
        self.open()
        self.close()

    def open(self) -> None:
        self._file_handle = PathManager.open(self.json_file, "a")

    def write(self) -> None:
        '''Saves the results'''
        self.open()
        storage = get_event_storage()
        to_save = defaultdict(dict)
        iteration = storage.iter
        if (iteration + 1) % self.nb_iter_log == 0:
            losses = [loss for loss in storage.histories() if 'loss' in loss]
            losses_valid = [loss for loss in losses if 'validation' in loss]
            losses_train = [loss for loss in losses if loss not in losses_valid]

            for key in losses_train:
                to_save[iteration][key] = storage.histories()[key].median(self.nb_iter_per_epoch)
            for key in losses_valid:
                to_save[iteration][key] = storage.histories()[key].latest()

            for itr, scalars_per_iter in to_save.items():
                scalars_per_iter["iteration"] = itr
                self._file_handle.write(json.dumps(scalars_per_iter, sort_keys=True) + "\n")
            self._file_handle.flush()
            try:
                os.fsync(self._file_handle.fileno())
            except AttributeError:
                pass
        self.close()

    def close(self):
        '''Close the open file'''
        self._file_handle.close()


def data_augmentation_mapper(dataset_dict: dict, horizontal_flip: bool = False,
                             vertical_flip: bool = False, rot_90: bool = False) -> dict:
    '''Applies the data augmentation on data

    Args:
        dataset_dict (dict) : Data dictionary containing the images on which to do data augmentation
        horizontal_flip (bool) : If True, can do horizontal flip (with 0.5 proba)
        vertical_flip (bool) : If True, can do vertical flip (with 0.5 proba)
        rot_90 (bool) : If True, can do a rotation of 0, 90, 180 or 270 degrees (0.25 proba for each)

    Returns:
        The dictionary after data augmentation
    '''
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")  # Reads the image

    # Add transformations
    transform_list = []
    if rot_90:
        angle = np.random.choice([0, 90, 180, 270], 1)[0]
        transform_list.append(T.RandomRotation(angle=[angle, angle]))
    if horizontal_flip:
        transform_list.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
    if vertical_flip:
        transform_list.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))

    # Apply transformations to the image
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    dataset_dict['height'], dataset_dict['width'] = dataset_dict['image'].shape[1:]

    # Update of the bboxes
    annos = [
        detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    # Transform to "instance" (suitable format for detectron2)
    instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
    # Return
    return dataset_dict


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
