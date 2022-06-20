#!/usr/bin/env python3
# type: ignore

## Faster RCNN model (keras) - Object detection
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
# - ModelKerasFasterRcnnObjectDetector -> Faster RCNN model (Keras) for object detection


# Cf. fix https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import os
import io
import cv2
import copy
import json
import math
import ntpath
import shutil
import logging
import numpy as np
import pandas as pd
import dill as pickle
from PIL import Image
from typing import Union, List, Callable, Any
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPooling2D, TimeDistributed
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard, TerminateOnNaN

from {{package_name}} import utils
from {{package_name}}.models_training.model_keras import ModelKeras
from {{package_name}}.models_training.object_detectors import utils_faster_rcnn
from {{package_name}}.models_training.object_detectors import utils_object_detectors
from {{package_name}}.models_training.object_detectors.model_object_detector import ModelObjectDetectorMixin  # type: ignore

###########################
# The Faster RCNN model is composed of two models
#
# The first one, the RPN (region proposal network) allows to select ROIs (regions of interest) in an image
# This first model is composed of a common structure (VGG16) and of a FCN (fully convolutional network)
# which allows to consider images of any sizes
# It has two targets :
# - Classification to tell if a region is a potential ROI (ie. match on an object)
# - Regression to correct the position of the region
#
# The second model is a Fast-RCNN which will work on the ROIs of an image
# It is composed of a common structure (VGG16) and a dense network
# A crop of the image (more precisely of its feature map which is the output of the common structure)
# is done for each ROI identified by the RPN
# This crop is send to the model which has two targets:
# - Classification to find the class of the ROI (ie. which object is detected)
# - Regression to correct the position of the ROI
###########################


class ModelKerasFasterRcnnObjectDetector(ModelObjectDetectorMixin, ModelKeras):
    '''Faster RCNN model (Keras) for object detection'''

    _default_name = 'model_keras_faster_rcnn_object_detector'

    def __init__(self, img_min_side_size: int = 300, rpn_min_overlap: float = 0.3, rpn_max_overlap: float = 0.7, rpn_restrict_num_regions: int = 256,
                 pool_resize_classifier: int = 7, nb_rois_classifier: int = 4, roi_nms_overlap_threshold: float = 0.7, nms_max_boxes: int = 300,
                 classifier_min_overlap: float = 0.1, classifier_max_overlap: float = 0.5,
                 pred_bbox_proba_threshold: float = 0.6, pred_nms_overlap_threshold: float = 0.2,
                 data_augmentation_params: Union[dict, None] = None,
                 batch_size_rpn_trainable_true: Union[int, None] = None, batch_size_classifier_trainable_true: Union[int, None] = None,
                 batch_size_rpn_trainable_false: Union[int, None] = None, batch_size_classifier_trainable_false: Union[int, None] = None,
                 epochs_rpn_trainable_true: Union[int, None] = None, epochs_classifier_trainable_true: Union[int, None] = None,
                 epochs_rpn_trainable_false: Union[int, None] = None, epochs_classifier_trainable_false: Union[int, None] = None,
                 patience_rpn_trainable_true: Union[int, None] = None, patience_classifier_trainable_true: Union[int, None] = None,
                 patience_rpn_trainable_false: Union[int, None] = None, patience_classifier_trainable_false: Union[int, None] = None,
                 lr_rpn_trainable_true: float = 1e-5, lr_classifier_trainable_true: float = 1e-5, lr_rpn_trainable_false: float = 1e-5,
                 lr_classifier_trainable_false: float = 1e-5, **kwargs) -> None:
        '''Initialization of the class (see ModelClass, ModelKeras & ModelObjectDetectorMixin for more arguments)

        Kwargs:
            img_min_side_size (int): Size to give to the smaller dimension as input of the model
            rpn_min_overlap (float): Under this threshold a region is classified as background (RPN model)
            rpn_max_overlap (float): Above this threshold a region is classified as object (RPN model)
            rpn_restrict_num_regions (int): Maximal number of regions to keep as target for the RPN
            pool_resize_classifier (int): Size to give to the crops done before the classifier (via ROI)
            nb_rois_classifier (int): Maximal number of ROIs per image during classifier training (per image of a batch)
            roi_nms_overlap_threshold (float): The NMS deletes overlapping ROIs whose IOU is above this threshold
            nms_max_boxes (int): Maximal number of ROIs to be returned by the NMS
            classifier_min_overlap (float): Above this threshold a ROI is considered to be a target of the classifier (but can be 'bg')
            classifier_max_overlap (float): Above this threshold a ROI is considered to be matching a bbox (so the target is a class, not 'bg')
            pred_bbox_proba_threshold (float): Above this threshold (for probabilities), a ROI is considered to be a match
            pred_nms_overlap_threshold (float): When predicting, the NMS deletes overlapping predictions whose IOU is above this threshold
            data_augmentation_params (dict): Set of allowed data augmentation
            batch_size_rpn_trainable_true (int): Batch size for the RPN with for first run with trainable set to True
            batch_size_classifier_trainable_true (int): Batch size for the classifier for the first run with trainable set to True
            batch_size_rpn_trainable_false (int): Batch size for the RPN for the second run with trainable set to False
            batch_size_classifier_trainable_false (int): Batch size for the classifier for the second run with trainable set to False
            epochs_rpn_trainable_true (int): Number of epochs for the RPN for the first run with trainable set to True
            epochs_classifier_trainable_true (int): Number of epochs for the classifier for the first run with trainable set to True
            epochs_rpn_trainable_false (int): Number of epochs for the RPN for the second run with trainable set to False
            epochs_classifier_trainable_false (int): lNumber of epochs for the classifier for the second run with trainable set to False
            patience_rpn_trainable_true (int): Patience for the RPN for the first run with trainable set to True
            patience_classifier_trainable_true (int): Patience for the classifier for the first run with trainable set to True
            patience_rpn_trainable_false (int): Patience for the RPN for the second run with trainable set to False
            patience_classifier_trainable_false (int): Patience for the classifier for the second run with trainable set to False
            lr_rpn_trainable_true (float): Learning rate for the RPN for the first run with trainable set to True
            lr_classifier_trainable_true (float): Learning rate for the classifier for the first run with trainable set to True
            lr_rpn_trainable_false (float): Learning rate for the RPN for the second run with trainable set to False
            lr_classifier_trainable_false (float): Learning rate for the classifier for the second run with trainable set to False
        Raises:
            ValueError: If img_min_side_size is not positive
            ValueError: If rpn_min_overlap is not in [0, 1]
            ValueError: If rpn_max_overlap is not in [0, 1]
            ValueError: If rpn_min_overlap > rpn_max_overlap
            ValueError: If rpn_restrict_num_regions is not positive
            ValueError: If pool_resize_classifier is not positive
            ValueError: If nb_rois_classifier is not positive
            ValueError: If roi_nms_overlap_threshold is not in [0, 1]
            ValueError: If nms_max_boxes is not positive
            ValueError: If classifier_min_overlap is not in [0, 1]
            ValueError: If classifier_max_overlap is not in [0, 1]
            ValueError: If classifier_min_overlap > classifier_max_overlap
            ValueError: If pred_bbox_proba_threshold is not in [0, 1]
            ValueError: If pred_nms_overlap_threshold is not in [0, 1]
            ValueError: If color_mode is not 'rgb'
            ValueError: If the minimum size of the image is inferior to twice the subsampling ratio
        '''
        # Manage errors
        if img_min_side_size < 1:
            raise ValueError(f"The argument img_min_side_size ({img_min_side_size}) must be positive")
        if not 0 <= rpn_min_overlap <= 1:
            raise ValueError(f"The argument rpn_min_overlap ({rpn_min_overlap}) must be between 0 and 1, included")
        if not 0 <= rpn_max_overlap <= 1:
            raise ValueError(f"The argument rpn_max_overlap ({rpn_max_overlap}) must be between 0 and 1, included")
        if rpn_min_overlap > rpn_max_overlap:
            raise ValueError(f"The argument rpn_min_overlap ({rpn_min_overlap}) can't be superior to rpn_max_overlap ({rpn_max_overlap})")
        if rpn_restrict_num_regions < 1:
            raise ValueError(f"The argument rpn_restrict_num_regions ({rpn_restrict_num_regions}) must be positive")
        if pool_resize_classifier < 1:
            raise ValueError(f"The argument pool_resize_classifier ({pool_resize_classifier}) must be positive")
        if nb_rois_classifier < 1:
            raise ValueError(f"The argument nb_rois_classifier ({nb_rois_classifier}) must be positive")
        if not 0 <= roi_nms_overlap_threshold <= 1:
            raise ValueError(f"The argument roi_nms_overlap_threshold ({roi_nms_overlap_threshold}) must be between 0 and 1, included")
        if nms_max_boxes < 1:
            raise ValueError(f"The argument nms_max_boxes ({nms_max_boxes}) must be positive")
        if not 0 <= classifier_min_overlap <= 1:
            raise ValueError(f"The argument classifier_min_overlap ({classifier_min_overlap}) must be between 0 and 1, included")
        if not 0 <= classifier_max_overlap <= 1:
            raise ValueError(f"The argument classifier_max_overlap ({classifier_max_overlap}) must be between 0 and 1, included")
        if classifier_min_overlap > classifier_max_overlap:
            raise ValueError(f"The argument classifier_min_overlap ({classifier_min_overlap}) can't be superior to classifier_max_overlap ({classifier_max_overlap})")
        if not 0 <= pred_bbox_proba_threshold <= 1:
            raise ValueError(f"The argument pred_bbox_proba_threshold ({pred_bbox_proba_threshold}) must be between 0 and 1, included")
        if not 0 <= pred_nms_overlap_threshold <= 1:
            raise ValueError(f"The argument pred_nms_overlap_threshold ({pred_nms_overlap_threshold}) must be between 0 and 1, included")

        # Size of the input images (must be defined before the super init because it is used in the method _get_preprocess_input)
        self.img_min_side_size = img_min_side_size  # Default 300, in the paper 600

        # Init. (by default we have some data augmentation)
        if data_augmentation_params is None:
            data_augmentation_params = {'horizontal_flip': True, 'vertical_flip': True, 'rot_90': True}
        super().__init__(data_augmentation_params=data_augmentation_params, **kwargs)
        if self.color_mode != 'rgb':
            raise ValueError("Faster RCNN model only accept color_mode equal to 'rgb' (compatibility VGG16).")

        # Put to None some parameters of model_keras not used by this model
        self.width = None
        self.height = None
        self.depth = None
        self.in_memory = None
        self.nb_train_generator_images_to_save = None

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Models, set on fit
        self.model: Any = None
        self.shared_model = None
        self.model_rpn = None
        self.model_classifier = None

        # Weights
        self.vgg_filename = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        self.vgg_path = os.path.join(utils.get_data_path(), 'transfer_learning_weights', self.vgg_filename)
        {% if vgg16_weights_backup_urls is not none %}vgg16_weights_backup_urls = [
        {%- for item in vgg16_weights_backup_urls %}
            '{{item}}',
        {%- endfor %}
        ]{% else %}vgg16_weights_backup_urls = []{% endif %}
        if not os.path.exists(self.vgg_path):
            try:
                self.logger.warning("The weights file for VGG16 is not present in your data folder.")
                self.logger.warning("Trying to download the file.")
                utils.download_url(vgg16_weights_backup_urls, self.vgg_path)
            except ConnectionError:
                self.logger.warning("Can't download. You can try to download it manually and save it on DVC.")
                self.logger.warning("Building this model will return an error.")
                self.logger.warning("You can download the weights here : https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
                # We don't raise an error because we may reload a trained model

        ### Model configuration

        # Configurations related to the base model
        self.shared_model_subsampling = 16  # VGG 16
        # Error if img_min_side_size < 2 * subsampling rate
        if self.img_min_side_size < 2 * self.shared_model_subsampling:
            raise ValueError("Can't have a minimum size of an image inferior to twice the subsampling ratio")

        # Anchors boxes
        self.anchor_box_sizes = [64, 128, 256]  # In the paper : [128, 256, 512]
        self.anchor_box_ratios = [[1, 1], [1. / math.sqrt(2), 2. / math.sqrt(2)], [2. / math.sqrt(2), 1. / math.sqrt(2)]]  # In the paper : [1, 1], [1, 2], [2, 1]]
        self.nb_anchors = len(self.anchor_box_sizes) * len(self.anchor_box_ratios)
        self.list_anchors = [[anchor_size * anchor_ratio[0], anchor_size * anchor_ratio[1]]
                             for anchor_size in self.anchor_box_sizes for anchor_ratio in self.anchor_box_ratios]

        # Sizes
        self.pool_resize_classifier = pool_resize_classifier  # Def 7

        # Scaling (we could probably do without scaling)
        self.rpn_regr_scaling = 4.0
        self.classifier_regr_scaling = [8.0, 8.0, 4.0, 4.0]

        # Thresholds for the RPN to find positive and negative anchor boxes
        self.rpn_min_overlap = rpn_min_overlap  # Def 0.3
        self.rpn_max_overlap = rpn_max_overlap  # Def 0.7
        # Maximum number of regions targets of the RPN
        self.rpn_restrict_num_regions = rpn_restrict_num_regions

        # Classifier configuration
        self.nb_rois_classifier = nb_rois_classifier  # Def 4
        self.roi_nms_overlap_threshold = roi_nms_overlap_threshold  # Def 0.7
        self.nms_max_boxes = nms_max_boxes  # Def 300
        self.classifier_min_overlap = classifier_min_overlap  # Def 0.1
        self.classifier_max_overlap = classifier_max_overlap  # Def 0.5

        # Prediction Thresholds
        self.pred_bbox_proba_threshold = pred_bbox_proba_threshold  # Def 0.6
        self.pred_nms_overlap_threshold = pred_nms_overlap_threshold  # Def 0.2

        ### Misc.

        # We add the custom objects only when fitting because we need the number of classes

        # Manage batch_size, epochs & patience (back up on global values if not specified)
        self.batch_size_rpn_trainable_true = batch_size_rpn_trainable_true if batch_size_rpn_trainable_true is not None else self.batch_size
        self.batch_size_classifier_trainable_true = batch_size_classifier_trainable_true if batch_size_classifier_trainable_true is not None else self.batch_size
        self.batch_size_rpn_trainable_false = batch_size_rpn_trainable_false if batch_size_rpn_trainable_false is not None else self.batch_size
        self.batch_size_classifier_trainable_false = batch_size_classifier_trainable_false if batch_size_classifier_trainable_false is not None else self.batch_size
        self.epochs_rpn_trainable_true = epochs_rpn_trainable_true if epochs_rpn_trainable_true is not None else self.epochs
        self.epochs_classifier_trainable_true = epochs_classifier_trainable_true if epochs_classifier_trainable_true is not None else self.epochs
        self.epochs_rpn_trainable_false = epochs_rpn_trainable_false if epochs_rpn_trainable_false is not None else self.epochs
        self.epochs_classifier_trainable_false = epochs_classifier_trainable_false if epochs_classifier_trainable_false is not None else self.epochs
        self.patience_rpn_trainable_true = patience_rpn_trainable_true if patience_rpn_trainable_true is not None else self.patience
        self.patience_classifier_trainable_true = patience_classifier_trainable_true if patience_classifier_trainable_true is not None else self.patience
        self.patience_rpn_trainable_false = patience_rpn_trainable_false if patience_rpn_trainable_false is not None else self.patience
        self.patience_classifier_trainable_false = patience_classifier_trainable_false if patience_classifier_trainable_false is not None else self.patience

        # Save learning rates in params_keras
        self.keras_params['lr_rpn_trainable_true'] = lr_rpn_trainable_true
        self.keras_params['lr_classifier_trainable_true'] = lr_classifier_trainable_true
        self.keras_params['lr_rpn_trainable_false'] = lr_rpn_trainable_false
        self.keras_params['lr_classifier_trainable_false'] = lr_classifier_trainable_false

    #####################
    # Modelisation
    #####################

    def _get_model(self) -> Any:
        '''Gets a model structure

        Returns:
            (?): Shared layers of the VGG 16 (not compiled)
            (?): RPN model
            (?): Classifier model
            (?): Global model (for load/save only)
        '''
        # First, we define the inputs
        input_img = Input(shape=(None, None, 3), name='input_img')
        input_rois = Input(shape=(None, 4), name='input_rois')

        # Then we get the various parts of the network
        shared_model_layers = self._get_shared_model_structure(input_img)  # List (class & regr)
        rpn_layers = self._add_rpn_layers(shared_model_layers)  # List (class & regr)
        classifier_layers = self._add_classifier_layers(shared_model_layers, input_rois)

        # Base model (shared layers)
        shared_model = Model(input_img, shared_model_layers)
        # We instanciate our models
        model_rpn = Model(input_img, rpn_layers)
        model_classifier = Model([input_img, input_rois], classifier_layers)
        # Concatenation of the two models, used to load / save the weights
        model_all = Model([input_img, input_rois], rpn_layers + classifier_layers)

        # We load the pre-trained weights
        shared_model.load_weights(self.vgg_path, by_name=True)

        # Compile models
        self._compile_model_rpn(model_rpn, lr=self.keras_params['lr_rpn_trainable_true'])
        self._compile_model_classifier(model_classifier, lr=self.keras_params['lr_classifier_trainable_true'])
        # We also compile model_all
        model_all.compile(optimizer='sgd', loss='mae')

        # Display summaries
        if self.logger.getEffectiveLevel() < logging.ERROR:
            model_all.summary()

        # Try to save models as png if level_save > 'LOW'
        if self.level_save in ['MEDIUM', 'HIGH']:
            self._save_model_png(model_all)

        # Return models
        return shared_model, model_rpn, model_classifier, model_all

    def _compile_model_rpn(self, model_rpn, lr: float) -> None:
        '''Compiles the RPN model using the specified learning rate

        Args:
            model_rpn : RPN model to compile
            lr (float): Learning rate we want to use
        '''
        # Set optimizer
        decay = self.keras_params.get('decay_rpn', 0.0)
        self.logger.info(f"Learning rate used - RPN : {lr}")
        self.logger.info(f"Decay used - RPN : {decay}")
        optimizer_rpn = Adam(lr=lr, decay=decay)

        # Set loss & metrics
        losses_rpn = {'rpn_class': self.custom_objects['rpn_loss_cls'], 'rpn_regr': self.custom_objects['rpn_loss_regr']}
        metrics_rpn = {'rpn_class': 'accuracy'}

        # Compile model
        model_rpn.compile(optimizer=optimizer_rpn, loss=losses_rpn, metrics=metrics_rpn)

    def _compile_model_classifier(self, model_classifier, lr: float) -> None:
        '''Compiles the classifier model using the specified learning rate

        Args:
            model_classifier : Classifier to compule
            lr (float): Learning rate we want to use
        '''
        # Set optimizer
        decay = self.keras_params.get('decay_classifier', 0.0)
        self.logger.info(f"Learning rate used - classifier : {lr}")
        self.logger.info(f"Decay used - classifier : {decay}")
        optimizer_classifier = Adam(lr=lr, decay=decay)

        # Set loss & metrics
        losses_classifier = {'dense_class': self.custom_objects['class_loss_cls'], 'dense_regr': self.custom_objects['class_loss_regr']}
        metrics_classifier = {'dense_class': 'accuracy'}

        # Compile model
        model_classifier.compile(optimizer=optimizer_classifier, loss=losses_classifier, metrics=metrics_classifier)

    def _get_shared_model_structure(self, input_img):
        '''We give the VGG 16 structure

        Args:
            input_img (?): Input layer for the images
        Returns:
            ?: VGG16 structure (without the weights)
        '''

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)

        return x

    def _add_rpn_layers(self, base_layers) -> List[Conv2D]:
        '''Adds the RPN layers to a base model

        Args:
            base_layers: Base model - VGG16
        Returns:
            ?: RPN layers
        '''
        # We add a convolution layer
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
        # Fully convolutional layer for our targets
        # - nb_anchors, feature_map_width, feature_map_height  --->  Classification
        # - nb_anchors * nb_coordinates, feature_map_width, feature_map_height  --->  Regression
        x_class = Conv2D(self.nb_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_class')(x)
        x_regr = Conv2D(self.nb_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_regr')(x)
        # We return the results of the two parts (the format will be managed by the losses)
        return [x_class, x_regr]

    def _add_classifier_layers(self, base_layers, input_rois) -> List[TimeDistributed]:
        '''Adds layers for classification to a base model and a ROIs tensor

        Args:
            base_layers: Base model - VGG16
            input_rois: Tensor with some ROIs
                # Shape (1, num_rois, 4), with coordinates (x, y, w, h)
        Returns:
            list: List with the classification and regression outputs
        '''
        # We get the crops on the features map from the ROIs
        out_roi_pool = utils_faster_rcnn.RoiPoolingLayer(self.pool_resize_classifier, name='roi_pool')([base_layers, input_rois])

        # Add the Dense part (we use TimeDistributed to take care of the ROIs dimension)
        out = TimeDistributed(Flatten(name='flatten'), name='distributed_flatten')(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc1'), name='distributed_fc1')(out)
        out = TimeDistributed(Dropout(0.5), name='distributed_dropout_1')(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='fc2'), name='distributed_fc2')(out)
        out = TimeDistributed(Dropout(0.5), name='distributed_dropout_2')(out)

        # We output two parts, classifier and regressor
        # Classifier : ROI class
        # Regressor : ROI coordinates correction
        nb_classes = len(self.list_classes)
        out_class = TimeDistributed(Dense(nb_classes + 1, activation='softmax', kernel_initializer='zero'), name="dense_class")(out)
        # TODO: Do we really need to a regression for each class ?!
        # TODO: Couldn't we simply have a single regression ? The corresponding loss would be on matching an object (whatever the class).
        # TODO: Couldn't we event make without the regression part here? After all, it is already done by the RPN.
        out_regr = TimeDistributed(Dense(4 * nb_classes, activation='linear', kernel_initializer='zero'), name="dense_regr")(out)

        return [out_class, out_regr]

    #####################
    # Images generation
    #####################

    def _get_generator(self, df: pd.DataFrame, data_type: str, batch_size: int, generator_type: str,
                       shared_model_trainable: bool = False, with_img_data: bool = False):
        '''Gets image generator from a list of files - object detector version

        Args:
            df (pd.DataFrame): Dataset to use must contain :
                - a column 'file_path' with a path to an image
                - a column 'bboxes', the list of the bboxes of the image (if train or val)
            data_type (str): Type of data : 'train', 'valid' or 'test'
            batch_size (int): Batch size to use
            generator_type (str): The generator to use, 'rpn' or 'classifier'
            shared_model_trainable (bool): Classifier & train only - if the shared model is trainable,
                we must clone the RPN in order not to worsen the quality of the ROIs prediction (which
                are an input of the classifier)
            with_img_data (bool): If True, the generator also gives img_data as output
        Raises:
            ValueError: If the type of the model is not object_detector
            ValueError: If data_type is not in ['train', 'valid', 'test']
            ValueError: If the dataframe has no 'file_path' column
            ValueError: If 'train' or 'valid' and the dataframe has no 'bboxes' column
        '''
        # Manage errors
        if self.model_type != 'object_detector':
            raise ValueError(f"Models of type {self.model_type} do not implement the method _get_generator")
        if data_type not in ['train', 'valid', 'test']:
            raise ValueError(f"The value {data_type} is not a suitable value for the argument data_type ['train', 'valid', 'test'].")
        if generator_type not in ['rpn', 'classifier']:
            raise ValueError(f"The value {generator_type} is not a suitable value for the generator_type ['rpn', 'classifier'].")
        if 'file_path' not in df.columns:
            raise ValueError("The column 'file_path' is mandatory in the input dataframe")
        if data_type in ['train', 'valid'] and 'bboxes' not in df.columns:
            raise ValueError(f"The column 'bboxes' is mandatory in the input dataframe when data_type equal to '{data_type}'")

        # Copy
        df = df.copy(deep=True)
        # Extract info
        img_data_list = []
        for i, row in df.iterrows():
            filepath = row['file_path']
            if 'bboxes' in df.columns:
                bboxes = row['bboxes']
                img_data_list.append({'file_path': filepath, 'bboxes': bboxes})
            else:
                img_data_list.append({'file_path': filepath})

        # TODO : Manage incorrect bboxes ?

        # Get the suitable generator class
        custom_generator = CustomGeneratorRpn if generator_type == 'rpn' else CustomGeneratorClassifier

        # Set data_gen (no augmentation nor shuffle if validation/test)
        if data_type == 'train':
            generator = custom_generator(img_data_list=img_data_list, batch_size=batch_size, shuffle=True,
                                         seed=None, model=self, shared_model_trainable=shared_model_trainable,
                                         data_type=data_type, **self.data_augmentation_params, with_img_data=with_img_data)
        else:
            generator = custom_generator(img_data_list=img_data_list, batch_size=batch_size, shuffle=False,
                                         seed=None, model=self, shared_model_trainable=shared_model_trainable,
                                         data_type=data_type, with_img_data=with_img_data)
        return generator

    def _generate_images_with_bboxes(self, img_data: dict, horizontal_flip: bool = False,
                                     vertical_flip: bool = False, rot_90: bool = False) -> dict:
        '''Generates an image and its bboxes from its info (path, etc.)

        Can do data augmentation but with a limited choice because some transformations are not
        compatible with the bboxes (eg. 20 degrees angle)

        Also preprocesses the image

        Args:
            img_data (dict): Data on the image and its bboxes
                Must contain : 'file_path' & 'bboxes'
        Kwargs:
            horizontal_flip (bool): If True, can do horizontal flip (with 0.5 proba)
            vertical_flip (bool): If True, can do vertical flip (with 0.5 proba)
            rot_90 (bool): If True, can do a rotation of 0, 90, 180 or 270 degrees (0.25 proba for each)

                By default the augmentations are not applied (set to False)
        '''
        # Read the image, as TensorFlow does
        with open(img_data['file_path'], 'rb') as f:
            img = Image.open(io.BytesIO(f.read()))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Convert to array
            img = np.asarray(img)
        # Get bboxes & image size
        bboxes = copy.deepcopy(img_data.get('bboxes', []))  # Empty if test
        h, w = img.shape[:2]

        #####################
        ### Augmentations ###
        #####################
        # Horizontal flip
        if horizontal_flip and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            for bbox in bboxes:
                x1, x2 = bbox['x1'], bbox['x2']
                bbox['x2'] = w - x1
                bbox['x1'] = w - x2

        # Vertical flip
        if vertical_flip and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)
            for bbox in bboxes:
                y1, y2 = bbox['y1'], bbox['y2']
                bbox['y2'] = h - y1
                bbox['y1'] = h - y2

        # Rotation 0°, 90°, 180° or 270°
        if rot_90:
            angle = np.random.choice([0, 90, 180, 270], 1)[0]
            if angle == 270:
                img = cv2.flip(np.transpose(img, (1, 0, 2)), 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = cv2.flip(np.transpose(img, (1, 0, 2)), 1)

            for bbox in bboxes:
                x1, x2, y1, y2 = bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = w - x2
                    bbox['y2'] = w - x1
                elif angle == 180:
                    bbox['x2'] = w - x1
                    bbox['x1'] = w - x2
                    bbox['y2'] = h - y1
                    bbox['y1'] = h - y2
                elif angle == 90:
                    bbox['x1'] = h - y2
                    bbox['x2'] = h - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2

        #####################
        ### Preprocessing ###
        #####################

        # Keep original sizes
        original_height, original_width = img.shape[0], img.shape[1]

        # Preprocess
        img = self.preprocess_input(img)

        # Get new sizes
        resized_height, resized_width = img.shape[0], img.shape[1]

        # Resize the bboxes following the preprocessing
        # We could get floats but it is not important at this point
        for bbox in bboxes:
            bbox['x1'] = bbox['x1'] * (resized_width / original_width)
            bbox['x2'] = bbox['x2'] * (resized_width / original_width)
            bbox['y1'] = bbox['y1'] * (resized_height / original_height)
            bbox['y2'] = bbox['y2'] * (resized_height / original_height)

        #####################
        ###  Format Data  ###
        #####################

        prepared_data = {
            'img': img,  # Preprocessed image
            'bboxes': bboxes,  # Bboxes after data augmentation & resizing
            'original_height': original_height,
            'original_width': original_width,
            'resized_height': resized_height,
            'resized_width': resized_width,
        }
        return prepared_data

    def _get_preprocess_input(self) -> Union[Callable, None]:
        '''Gets the preprocessing to be used before feeding images to the NN

        Returns:
            (Callable | None): Preprocessing function
        '''
        # Get preprocessing function (resize + vgg16)
        img_min_side_size = self.img_min_side_size  # We take care not to have references to self in the function
        def preprocess_input(x_img: np.ndarray, **kwargs) -> np.ndarray:
            '''Preprocessing of a numpy image

            Resizes the image + classic VGG 16 preprocessing

            Args:
                x_img (np.ndarray): Image to process
            Returns:
                np.ndarray: Result of the preprocessing
            '''
            # Resize
            height, width = x_img.shape[0], x_img.shape[1]
            resized_height, resized_width = utils_object_detectors.get_new_img_size_from_min_side_size(height, width, img_min_side_size)
            x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)  # Format dimension width, height ...
            return preprocess_input_vgg16(x_img)
        # Returns it
        return preprocess_input

    #####################
    # Fit
    #####################

    def _fit_object_detector(self, df_train: pd.DataFrame, df_valid: Union[pd.DataFrame, None] = None,
                             with_shuffle: bool = True, **kwargs) -> None:
        '''Training of the model

        Args:
            df_train (pd.DataFrame): Training data with file_path & bboxes columns
            df_valid (pd.DataFrame): Validation data with file_path & bboxes columns
        Kwargs:
            with_shuffle (boolean): If data must be shuffled before fitting
                This should be used if the target is not shuffled as the split_validation takes the lines in order.
                Thus, the validation set might get classes which are not in the train set ...
        Raises:
            ValueError: If the type of the model is not object_detector
            ValueError: If the class 'bg' is present in the input data
            AssertionError: If the same classes are not present when comparing an already trained model
                and a new dataset
        '''
        if self.model_type != 'object_detector':
            raise ValueError(f"The models of type {self.model_type} do not implement the method _fit_object_detector")

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
        if 'bg' in set_classes:
            raise ValueError("The 'bg' class must not be present in the bounding boxes classes")

        list_classes = sorted(list(set_classes))
        # Also set dict_classes
        dict_classes = {i: col for i, col in enumerate(list_classes)}

        # Validate classes if already trained, else set them
        if self.trained:
            assert self.list_classes == list_classes, \
                "Error: the new dataset does not match with the already fitted model"
            assert self.dict_classes == dict_classes, \
                "Error: the new dataset does not match with the already fitted model"
        else:
            self.list_classes = list_classes
            self.dict_classes = dict_classes

        # Now that we have the list of the classes, we can define the custom_objects
        self.custom_objects = {**utils_faster_rcnn.get_custom_objects_faster_rcnn(self.nb_anchors, len(self.list_classes)), **self.custom_objects}

        # Shuffle training dataset if wanted
        # It is advised as validation_split from keras does not shufle the data
        # Hence, for classification task, we might have classes in the validation data that we never met in the training data
        if with_shuffle:
            df_train = df_train.sample(frac=1.).reset_index(drop=True)

        # Manage absence of validation datasets
        if df_valid is None:
            self.logger.warning(f"Warning, no validation set. The training set will be splitted (validation fraction = {self.validation_split})")
            df_train, df_valid = train_test_split(df_train, test_size=self.validation_split)

        ##############################################
        # Trainings
        # 4 steps :
        #   - Train RPN - sharable model trainable
        #   - Train classifier - sharable model trainable
        #   - Train RPN - sharable model NOT trainable
        #   - Train classifier - sharable model NOT trainable
        ##############################################

        # Get models (we do not load a new model if this one has already been trained)
        # self.model corresponds to model_all
        if not self.trained:
            self.shared_model, self.model_rpn, self.model_classifier, self.model = self._get_model()

        ### Train RPN - sharable model trainable
        self.logger.info("RPN training - trainable set to True")
        self._fit_object_detector_RPN(df_train, df_valid, shared_trainable=True)

        ### Train classifier - sharable model trainable
        self.logger.info("Classifier training - trainable set to True")
        self._fit_object_detector_classifier(df_train, df_valid, shared_trainable=True)

        ### Train RPN - sharable model NOT trainable
        self.logger.info("RPN training - trainable set to False")
        self._fit_object_detector_RPN(df_train, df_valid, shared_trainable=False)

        ### Train classifier - sharable model NOT trainable
        self.logger.info("Classifier training - trainable set to False")
        self._fit_object_detector_classifier(df_train, df_valid, shared_trainable=False)

        # We update trained & nb_fit
        self.trained = True
        self.nb_fit += 1

    def _fit_object_detector_RPN(self, df_train: pd.DataFrame, df_valid: Union[pd.DataFrame, None] = None,
                                 shared_trainable: bool = True, **kwargs) -> None:
        '''RPN training

        Args:
            df_train (pd.DataFrame): Training data with file_path & bboxes columns
            df_valid (pd.DataFrame): Validation data with file_path & bboxes columns
            shared_trainable (bool): If the shared model is trainable
        '''
        # Manage trainable
        for layer in self.shared_model.layers:
            layer.trainable = shared_trainable
        # We adapt the learning rate
        new_lr = self.keras_params['lr_rpn_trainable_true'] if shared_trainable else self.keras_params['lr_rpn_trainable_false']
        # /!\ Recompile, otherwise the unfreeze is not taken into account ! /!\
        # Cf. https://keras.io/guides/transfer_learning/#fine-tuning
        self._compile_model_rpn(self.model_rpn, lr=new_lr)
        # We adapt the batch_size, the number of epochs and the patience
        batch_size = self.batch_size_rpn_trainable_true if shared_trainable else self.batch_size_rpn_trainable_false
        epochs = self.epochs_rpn_trainable_true if shared_trainable else self.epochs_rpn_trainable_false
        patience = self.patience_rpn_trainable_true if shared_trainable else self.patience_rpn_trainable_false

        # If the number of epoch is 0, we skip the training
        if epochs == 0:
            self.logger.info(f"Number of epochs for RPN training - trainable set to {shared_trainable} is 0. We skip it.")
        # Entrainement
        else:
            # Create generators for the RPN
            self.logger.info("Get a RPN generator for training data.")
            batch_size_train = min(batch_size, len(df_train))
            generator_rpn_train = self._get_generator(df=df_train, data_type='train', batch_size=batch_size_train, generator_type='rpn')
            self.logger.info("Get a RPN generator for validation data.")
            batch_size_valid = min(batch_size, len(df_valid))
            generator_rpn_valid = self._get_generator(df=df_valid, data_type='valid', batch_size=batch_size_valid, generator_type='rpn')

            # Get callbacks (early stopping & checkpoint)
            callbacks = self._get_callbacks(patience=patience)

            # Fit !
            fit_history = self.model_rpn.fit(  # type: ignore
                x=generator_rpn_train,
                epochs=epochs,
                validation_data=generator_rpn_valid,
                callbacks=callbacks,
                verbose=1,
                workers=8,  # TODO : Check if this is ok if there are less CPUs
            )

            # Plots losses & metrics
            if self.level_save in ['MEDIUM', 'HIGH']:
                self._plot_metrics_and_loss(fit_history, model_type='rpn', trainable=shared_trainable)

    def _fit_object_detector_classifier(self, df_train: pd.DataFrame, df_valid: Union[pd.DataFrame, None] = None,
                                        shared_trainable: bool = True, **kwargs) -> None:
        '''Training of the classifier

        Args:
            df_train (pd.DataFrame): Training data with file_path & bboxes columns
            df_valid (pd.DataFrame): Validation data with file_path & bboxes columns
            shared_trainable (bool): If the shared model is trainable
        '''
        # Manage trainable
        for layer in self.shared_model.layers:
            layer.trainable = shared_trainable
        # We adapt the learning rate
        new_lr = self.keras_params['lr_classifier_trainable_true'] if shared_trainable else self.keras_params['lr_classifier_trainable_false']
        # /!\ Recompile, otherwise the unfreeze is not taken into account ! /!\
        # Cf. https://keras.io/guides/transfer_learning/#fine-tuning
        self._compile_model_classifier(self.model_classifier, lr=new_lr)
        # We adapt the batch_size, the number of epochs and the patience
        batch_size = self.batch_size_classifier_trainable_true if shared_trainable else self.batch_size_classifier_trainable_false
        epochs = self.epochs_classifier_trainable_true if shared_trainable else self.epochs_classifier_trainable_false
        patience = self.patience_classifier_trainable_true if shared_trainable else self.patience_classifier_trainable_false

        # If the number of epoch is 0, we skip the training
        if epochs == 0:
            self.logger.info(f"Number of epochs for classifier training - trainable set to {shared_trainable} is 0. We skip it.")
        # Training
        else:
            # Create generators for the classifier
            self.logger.info("Get a classifier generator for training data.")
            batch_size_train = min(batch_size, len(df_train))
            generator_classifier_train = self._get_generator(df=df_train, data_type='train', batch_size=batch_size_train,
                                                             generator_type='classifier', shared_model_trainable=shared_trainable)
            self.logger.info("Get a classifier generator for validation data.")
            batch_size_valid = min(batch_size, len(df_valid))
            generator_classifier_valid = self._get_generator(df=df_valid, data_type='valid', batch_size=batch_size_valid,
                                                             generator_type='classifier', shared_model_trainable=shared_trainable)

            # Get callbacks (early stopping & checkpoint)
            callbacks = self._get_callbacks(patience=patience)

            # Fit !
            fit_history = self.model_classifier.fit(  # type: ignore
                x=generator_classifier_train,
                epochs=epochs,
                validation_data=generator_classifier_valid,
                callbacks=callbacks,
                verbose=1,
                workers=8,  # TODO : Check if this is ok if there are less CPUs
            )

            # Plots losses & metrics
            if self.level_save in ['MEDIUM', 'HIGH']:
                self._plot_metrics_and_loss(fit_history, model_type='classifier', trainable=shared_trainable)

    def _get_callbacks(self, patience: int) -> list:
        '''Gets model callbacks

        Args:
            patience (int): Early stopping patience
        Returns:
            list: List of callbacks
        '''
        # Get classic callbacks
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
        if self.level_save in ['MEDIUM', 'HIGH']:
            callbacks.append(
                ModelCheckpointAll(
                    filepath=os.path.join(self.model_dir, 'best.hdf5'), monitor='val_loss',
                    save_best_only=True, mode='auto', model_all=self.model
                )
            )
        callbacks.append(CSVLogger(filename=os.path.join(self.model_dir, 'logger.csv'), separator='{{default_sep}}', append=False))
        callbacks.append(TerminateOnNaN())

        # Get LearningRateScheduler
        # FOR NOW, WE DO NOT TAKE INTO ACCOUNT LEARNING RATE SCHEDULERS
        # scheduler = self._get_learning_rate_scheduler()
        # if scheduler is not None:
        #     callbacks.append(LearningRateScheduler(scheduler))

        # Manage tensorboard
        if self.level_save in ['HIGH']:
            # Get log directory
            models_path = utils.get_models_path()
            tensorboard_dir = os.path.join(models_path, 'tensorboard_logs')
            # We add a prefix so that the function load_model works correctly (it looks for a sub-folder with model name)
            log_dir = os.path.join(tensorboard_dir, f"tensorboard_{ntpath.basename(self.model_dir)}")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # TODO: Check if this class slows the process
            # -> For now: comment
            # Create custom class to monitor LR changes
            # https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard
            # class LRTensorBoard(TensorBoard):
            #     def __init__(self, log_dir, **kwargs) -> None:  # add other arguments to __init__ if you need
            #         super().__init__(log_dir=log_dir, **kwargs)
            #
            #     def on_epoch_end(self, epoch, logs=None):
            #         logs.update({'lr': K.eval(self.model.optimizer.lr)})
            #         super().on_epoch_end(epoch, logs)

            callbacks.append(TensorBoard(log_dir=log_dir, write_grads=False, write_images=False))
            self.logger.info(f"To start tensorboard: python -m tensorboard.main --logdir {tensorboard_dir} --samples_per_plugin images=10")
            # We use the option samples_per_plugin to avoid a rare problem between matplotlib and tensorboard
            # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread

        return callbacks

    #####################
    # Predict
    #####################

    @utils.trained_needed
    def _predict_object_detector(self, df_test: pd.DataFrame, **kwargs) -> List[List[dict]]:
        '''Predictions on test set - batch size must is equal to 1

        Args:
            df_test (pd.DataFrame): Data to predict, with a column 'file_path'
        Raises:
            ValueError: If the model is not of type object_detector
        Returns:
            (list<list<dict>>): list (one entry per image) of list of bboxes
        '''
        if self.model_type != 'object_detector':
            raise ValueError(f"The models of type {self.model_type} do not implement the method _predict_object_detector")

        # Instanciate the generator for predictions (batch size must be equal to 1)
        test_generator = self._get_generator(df=df_test, data_type='test', batch_size=1, generator_type='classifier', with_img_data=True)
        final_predictions = []
        # For each image in df_test
        for index_img in range(len(df_test)):
            # We get the image after preprocessing and some metadata
            input_data, batch_prepared_img_data = test_generator.next()
            input_img = input_data['input_img'][0]  # Batch size of 1
            input_rois = input_data['input_rois'][0]  # Batch size of 1
            img_data = batch_prepared_img_data[0]  # Batch size of 1
            # We predict thanks to the classifier
            predictions = self.model_classifier.predict(input_data, verbose=0)
            probas = predictions[0][0]  # Probas match, at the level of the features map
            regr_coordinates = predictions[1][0]  # regressions,  at the level of the features map
            # We only keep the boxes which have a class different from the background and with
            # a high enough probability. At the same time, we get the class and the proba
            fm_boxes_candidates = utils_object_detectors.get_valid_fm_boxes_from_proba(probas, self.pred_bbox_proba_threshold, len(self.list_classes))
            # We apply the regression, then we get back to the level input bbox and we only keep the valid boxes
            boxes_candidates = utils_object_detectors.get_valid_boxes_from_coordinates(input_img, input_rois, fm_boxes_candidates, regr_coordinates,
                                                                                       self.classifier_regr_scaling, self.shared_model_subsampling,
                                                                                       self.dict_classes)
            # We apply the NMS algorithm in order to avoid overlaps
            if len(boxes_candidates):
                final_boxes = utils_object_detectors.non_max_suppression_fast_on_preds(boxes_candidates, self.pred_nms_overlap_threshold)
                # Finally we resize the boxes depending on the original size of the imaeg and put it in the desired format
                predicted_bboxes = utils_object_detectors.get_final_bboxes(final_boxes, img_data)
            else:
                predicted_bboxes = []
            # We add the list of bboxes to the total list
            final_predictions.append(copy.deepcopy(predicted_bboxes))
        # Return
        return final_predictions

    #####################
    # Misc.
    #####################

    def _plot_metrics_and_loss(self, fit_history, model_type: str = 'rpn', trainable: bool = True, **kwargs) -> None:
        '''Plots some metrics & losses

        Args:
            fit_history (?) : Fit history
        Kwargs:
            model_type (str): Type of the model (rpn' or 'classifier') used
            shared_trainable (bool): If the shared model is trainable
        '''
        # Manage dir
        plots_path = os.path.join(self.model_dir, 'plots')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        # Get a dictionnary of possible metrics/loss plots for both rpn & classifier
        metrics_dir_rpn = {
            'loss': [f'RPN loss with trainable set to {trainable}', f'loss_{model_type}_trainable_{trainable}'],
            'rpn_class_loss': [f'RPN classification loss with trainable set to {trainable}', f'loss_class_{model_type}_trainable_{trainable}'],
            'rpn_regr_loss': [f'RPN regression loss with trainable set to {trainable}', f'loss_regr_{model_type}_trainable_{trainable}'],
            'rpn_class_accuracy': [f'RPN classification accuracy with trainable set to {trainable}', f'accuracy_class_{model_type}_trainable_{trainable}']
        }
        metrics_dir_classifier = {
            'loss': [f'Classifier loss with trainable set to {trainable}', f'loss_{model_type}_trainable_{trainable}'],
            'dense_class_loss': [f'Classifier classification loss with trainable set to {trainable}', f'loss_class_{model_type}_trainable_{trainable}'],
            'dense_regr_loss': [f'Classifier regression loss with trainable set to {trainable}', f'loss_regr_{model_type}_trainable_{trainable}'],
            'dense_class_accuracy': [f'Classifier classification accuracy with trainable set to {trainable}', f'accuracy_class_{model_type}_trainable_{trainable}']
        }

        # Get correct metrics dir
        if model_type == 'rpn':
            metrics_dir = copy.deepcopy(metrics_dir_rpn)
        else:
            metrics_dir = copy.deepcopy(metrics_dir_classifier)

        # Plots each available metrics & losses
        for metric in fit_history.history.keys():
            if metric in metrics_dir.keys():
                title = metrics_dir[metric][0]
                filename = metrics_dir[metric][1]
                plt.figure(figsize=(10, 8))
                plt.plot(fit_history.history[metric])
                plt.plot(fit_history.history[f'val_{metric}'])
                plt.title(f"Model {title}")
                plt.ylabel(title)
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper left')
                # Save
                filename = f"{filename}.jpeg"
                plt.savefig(os.path.join(plots_path, filename))

                # Close figures
                plt.close('all')

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save configuration JSON
        if json_data is None:
            json_data = {}

        json_data['vgg_filename'] = self.vgg_filename
        json_data['shared_model_subsampling'] = self.shared_model_subsampling
        json_data['anchor_box_sizes'] = self.anchor_box_sizes
        json_data['anchor_box_ratios'] = self.anchor_box_ratios
        json_data['nb_anchors'] = self.nb_anchors
        json_data['list_anchors'] = self.list_anchors
        json_data['img_min_side_size'] = self.img_min_side_size
        json_data['pool_resize_classifier'] = self.pool_resize_classifier
        json_data['rpn_regr_scaling'] = self.rpn_regr_scaling
        json_data['classifier_regr_scaling'] = self.classifier_regr_scaling
        json_data['rpn_min_overlap'] = self.rpn_min_overlap
        json_data['rpn_max_overlap'] = self.rpn_max_overlap
        json_data['rpn_restrict_num_regions'] = self.rpn_restrict_num_regions
        json_data['nb_rois_classifier'] = self.nb_rois_classifier
        json_data['roi_nms_overlap_threshold'] = self.roi_nms_overlap_threshold
        json_data['nms_max_boxes'] = self.nms_max_boxes
        json_data['classifier_min_overlap'] = self.classifier_min_overlap
        json_data['classifier_max_overlap'] = self.classifier_max_overlap
        json_data['pred_bbox_proba_threshold'] = self.pred_bbox_proba_threshold
        json_data['pred_nms_overlap_threshold'] = self.pred_nms_overlap_threshold
        json_data['batch_size_rpn_trainable_true'] = self.batch_size_rpn_trainable_true
        json_data['batch_size_classifier_trainable_true'] = self.batch_size_classifier_trainable_true
        json_data['batch_size_rpn_trainable_false'] = self.batch_size_rpn_trainable_false
        json_data['batch_size_classifier_trainable_false'] = self.batch_size_classifier_trainable_false
        json_data['epochs_rpn_trainable_true'] = self.epochs_rpn_trainable_true
        json_data['epochs_classifier_trainable_true'] = self.epochs_classifier_trainable_true
        json_data['epochs_rpn_trainable_false'] = self.epochs_rpn_trainable_false
        json_data['epochs_classifier_trainable_false'] = self.epochs_classifier_trainable_false
        json_data['patience_rpn_trainable_true'] = self.patience_rpn_trainable_true
        json_data['patience_classifier_trainable_true'] = self.patience_classifier_trainable_true
        json_data['patience_rpn_trainable_false'] = self.patience_rpn_trainable_false
        json_data['patience_classifier_trainable_false'] = self.patience_classifier_trainable_false

        # Add some code if not in json_data:
        for layer_or_compile in ['_add_rpn_layers', '_add_classifier_layers', '_compile_model_rpn', '_compile_model_classifier']:
            if layer_or_compile not in json_data:
                json_data[layer_or_compile] = pickle.source.getsourcelines(getattr(self, layer_or_compile))[0]

        # Save
        # Save strategy :
        # - best.hdf5 already saved in fit() & contains all models
        # - as we can't pickle models, we drop them, save, and reload them
        shared_model = self.shared_model
        model_rpn = self.model_rpn
        model_classifier = self.model_classifier
        self.shared_model = None
        self.model_rpn = None
        self.model_classifier = None
        super().save(json_data=json_data)
        self.shared_model = shared_model
        self.model_rpn = model_rpn
        self.model_classifier = model_classifier

    def reload_models_from_hdf5(self, hdf5_path: str) -> None:
        '''Reloads all models from a unique hdf5 file. This method is specific to the faster RCNN model.

        Args:
            hdf5_path (str): Path to the .hdf5 file with the weightds of model_all
        Raises:
            FileNotFoundError: If the object hdf5_path is not an existing file
        '''
        # Check path exists
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"The file {hdf5_path} does not exist")

        # Reload model (based on get_models)
        # Set layers
        input_img = Input(shape=(None, None, 3), name='input_img')  # Warning, 3 channels !
        input_rois = Input(shape=(None, 4), name='input_rois')
        shared_model_layers = self._get_shared_model_structure(input_img)
        rpn_layers = self._add_rpn_layers(shared_model_layers)
        classifier_layers = self._add_classifier_layers(shared_model_layers, input_rois)
        # Init. models
        self.shared_model = Model(input_img, shared_model_layers)
        self.model_rpn = Model(input_img, rpn_layers)
        self.model_classifier = Model([input_img, input_rois], classifier_layers)
        self.model = Model([input_img, input_rois], rpn_layers + classifier_layers)
        # Load the weights (loading the weights of model all will also load the weights of the other models since they are linked)
        self.model.load_weights(hdf5_path)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration, the network used and its preprocessing
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            hdf5_path (str): Path to hdf5 file
            preprocess_input_path (str): Path to preprocess input file
        Raises:
            ValueError : If configuration_path is None
            ValueError : If hdf5_path is None
            ValueError : If preprocess_input_path is None
            FileNotFoundError : If the object configuration_path is not an existing file
            FileNotFoundError : If the object hdf5_path is not an existing file
            FileNotFoundError : If the object preprocess_input_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        hdf5_path = kwargs.get('hdf5_path', None)
        preprocess_input_path = kwargs.get('preprocess_input_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if hdf5_path is None:
            raise ValueError("The argument hdf5_path can't be None")
        if preprocess_input_path is None:
            raise ValueError("The argument preprocess_input_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"The file {hdf5_path} does not exist")
        if not os.path.exists(preprocess_input_path):
            raise FileNotFoundError(f"The file {preprocess_input_path} does not exist")

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
        for attribute in ['model_type', 'list_classes', 'dict_classes', 'level_save', 'batch_size',
                          'epochs', 'validation_split', 'patience', 'color_mode', 'data_augmentation_params',
                          'vgg_filename', 'shared_model_subsampling', 'anchor_box_sizes', 'anchor_box_ratios',
                          'nb_anchors', 'list_anchors', 'img_min_side_size', 'pool_resize_classifier',
                          'rpn_regr_scaling', 'classifier_regr_scaling', 'rpn_min_overlap', 'rpn_max_overlap',
                          'rpn_restrict_num_regions', 'nb_rois_classifier', 'roi_nms_overlap_threshold',
                          'nms_max_boxes', 'classifier_min_overlap', 'classifier_max_overlap',
                          'pred_bbox_proba_threshold', 'pred_nms_overlap_threshold',
                          'batch_size_rpn_trainable_true', 'batch_size_classifier_trainable_true',
                          'batch_size_rpn_trainable_false', 'batch_size_classifier_trainable_false',
                          'epochs_rpn_trainable_true', 'epochs_classifier_trainable_true',
                          'epochs_rpn_trainable_false', 'epochs_classifier_trainable_false',
                          'patience_rpn_trainable_true', 'patience_classifier_trainable_true',
                          'patience_rpn_trainable_false', 'patience_classifier_trainable_false',
                          'keras_params']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))
        for attribute in ['width', 'height', 'depth']:
            setattr(self, attribute, configs.get(attribute, None))
        self.in_memory = None
        self.nb_train_generator_images_to_save = None
        self.vgg_path = os.path.join(utils.get_data_path(), 'transfer_learning_weights', self.vgg_filename)  # Try to reload from usual path. Not really important if it fails.

        # Set custom objects
        self.custom_objects = {**utils_faster_rcnn.get_custom_objects_faster_rcnn(self.nb_anchors, len(self.list_classes)), **self.custom_objects}

        # Reload model
        self.reload_models_from_hdf5(hdf5_path)

        # Reload preprocess_input
        with open(preprocess_input_path, 'rb') as f:
            self.preprocess_input = pickle.load(f)

        # Save best hdf5 in new folder
        new_hdf5_path = os.path.join(self.model_dir, 'best.hdf5')
        shutil.copyfile(hdf5_path, new_hdf5_path)


class CustomGeneratorRpn(Iterator):
    '''RPN generator'''

    def __init__(self, img_data_list: List[dict], batch_size: int, shuffle: bool, seed: Union[int, None],
                 model, horizontal_flip: bool = False, vertical_flip: bool = False,
                 rot_90: bool = False, data_type: str = 'train', with_img_data: bool = False, **kwargs) -> None:
        '''Initialization of the RPN generator

        Args:
            img_data_list (list<dict>): Data list (the dictionaries containing file path and bboxes)
            batch_size (int): Size of the batches to generate
            shuffle (bool): If the data should be shuffled
            seed (int): Random seed to use
            model: Link to the model (nested structure) to access to the other methods of this script
            data_type (str): Data type 'train', 'valid' or 'test'
            with_img_data (bool): If True, also gives img_data as output
        Kwargs:
            horizontal_flip (bool) : If True, can do horizontal flip (with 0.5 proba)
            vertical_flip (bool) : If True, can do vertical flip (with 0.5 proba)
            rot_90 (bool) : If True, can do a rotation of 0, 90, 180 or 270 degrees (0.25 proba for each)
        '''
        # Set is_test
        if data_type == 'test':
            self.is_test = True
        else:
            self.is_test = False

        # If test, the batch size is set to 1
        if self.is_test:
            batch_size = 1

        # Super init.
        super().__init__(n=len(img_data_list), batch_size=batch_size, shuffle=shuffle, seed=seed)

        # Set params
        self.img_data_list = img_data_list
        self.model = model
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rot_90 = rot_90
        self.with_img_data = with_img_data

        # Manage data augmentation & test
        if self.is_test and any([param for param in [self.horizontal_flip, self.vertical_flip, self.rot_90]]):
            model.logger.warning("Warning, data augmentation on the test dataset ! It is most certainly a mistake !")

    def _get_batches_of_transformed_samples(self, index_array: np.ndarray) -> tuple:
        '''Gets a batch of inputs for the RPN model
        Warning, in order to be sure to have the same shape between the various images, we pad them with black pixels

        Args:
            index_array (np.ndarray): List of indices to include in a batch
        Returns:
            np.ndarray: Data batch to be used by the RPN model (x)
                # Shape (bach_size, max_resized_height, max_resized_width, 3)
            dict<np.ndarray>: Data batch to be used by the RPN model (y)
                rpn_class: RPN - target of the classifier
                    # Shape (bach_size, feature_map_height, feature_map_width, 2 * nb_anchors)
                rpn_class: RPN - target of the regressor
                    # Shape (bach_size, feature_map_height, feature_map_width, 2 * 4 * nb_anchors)
            if self.with_img_data : the metadata of the images
        '''
        # For each selected index, get image & preprocess it, and retrieve max resized_width & resized_height
        batch_prepared_img_data = []
        max_resized_width = 0
        max_resized_height = 0
        for ind in index_array:
            img_data = copy.deepcopy(self.img_data_list[ind])
            prepared_img_data = self.model._generate_images_with_bboxes(img_data, horizontal_flip=self.horizontal_flip,
                                                                        vertical_flip=self.vertical_flip, rot_90=self.rot_90)
            batch_prepared_img_data.append(prepared_img_data)
            max_resized_width = max(max_resized_width, prepared_img_data['resized_width'])
            max_resized_height = max(max_resized_height, prepared_img_data['resized_height'])

        # Set format X
        batch_shape = (len(index_array), max_resized_height, max_resized_width, 3)
        batch_x = np.zeros(batch_shape)  # Def. black pixels

        # Get input images
        for ind, prepared_img_data in enumerate(batch_prepared_img_data):
            prepared_img_data['batch_width'] = max_resized_width
            prepared_img_data['batch_height'] = max_resized_height
            img = prepared_img_data['img']
            batch_x[ind, :img.shape[0], :img.shape[1], :] = img

        # If test, we return the images
        if self.is_test:
            if self.with_img_data:
                return {'input_img': batch_x}, batch_prepared_img_data
            else:
                return {'input_img': batch_x}
        # Otherwise, we also manage the targets y
        else:
            batch_y_cls, batch_y_regr = utils_object_detectors.get_rpn_targets(self.model, batch_prepared_img_data)
            # Return
            if self.with_img_data:
                return {'input_img': batch_x}, {'rpn_class': batch_y_cls, 'rpn_regr': batch_y_regr}, batch_prepared_img_data
            else:
                return {'input_img': batch_x}, {'rpn_class': batch_y_cls, 'rpn_regr': batch_y_regr}


class CustomGeneratorClassifier(Iterator):
    '''Classifier generator'''

    def __init__(self, img_data_list: List[dict], batch_size: int, shuffle: bool, seed: Union[int, None],
                 model, shared_model_trainable: bool = False, horizontal_flip: bool = False, vertical_flip: bool = False,
                 rot_90: bool = False, data_type: str = 'train', with_img_data: bool = False, **kwargs) -> None:
        '''Initialization of the generator for the classifier

        Args:
            img_data_list (list<dict>): Data list (the dictionaries containing file path and bboxes)
            batch_size (int): Size of the batches to generate
            shuffle (bool): If the data should be shuffled
            seed (int): Random seed to use
            model: Link to the model (nested structure) to access to the other methods of this script
            shared_model_trainable (bool): If the shared model is set to trainable, we must clone the RPN
                in order not to worsen the prediction quality of the ROIs (input of the classifier)
            data_type (str): Data type 'train', 'valid' or 'test'
            with_img_data (bool): If True, also gives img_data as output
        Kwargs:
            horizontal_flip (bool) : If True, can do horizontal flip (with 0.5 proba)
            vertical_flip (bool) : If True, can do vertical flip (with 0.5 proba)
            rot_90 (bool) : If True, can do a rotation of 0, 90, 180 or 270 degrees (0.25 proba for each)
        '''
        # Set is_test
        if data_type == 'test':
            self.is_test = True
        else:
            self.is_test = False

        # Si test, on force batch size à 1
        if self.is_test:
            batch_size = 1

        # Super init.
        super().__init__(n=len(img_data_list), batch_size=batch_size, shuffle=shuffle, seed=seed)

        # Set params
        self.img_data_list = img_data_list
        self.model = model
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rot_90 = rot_90
        self.with_img_data = with_img_data

        # Manage shared_model_trainable
        self.shared_model_trainable = shared_model_trainable
        if self.shared_model_trainable:
            self.rpn_clone = clone_model(self.model.model_rpn)
            self.rpn_clone.set_weights(self.model.model_rpn.get_weights())
        else:
            self.rpn_clone = None

        # Manage data augmentation & test
        if self.is_test and any([param for param in [self.horizontal_flip, self.vertical_flip, self.rot_90]]):
            model.logger.warning("Warning, Data Augmentation detected on the test set! This is certainly not desired!")

    def _get_batches_of_transformed_samples(self, index_array: np.ndarray):
        '''Gets a batch of inputs for the classifier model
        Warning, in order to be sure to have the same shape between the various images, we pad them with black pixels

        Args:
            index_array (np.ndarray): List of indices to include in a batch
        Returns:
            np.ndarray: Data batch to be used by the classifier model (x)
                # Shape (bach_size, max_resized_height, max_resized_width, 3)
            dict<np.ndarray>: Data batch to be used by the classifier model (y)
                dense_class: Classifier - target of the classifier
                    # Shape (bach_size, feature_map_height, feature_map_width, 2 * nb_anchors)
                rpn_class: Classifier - target of the regressor
                    # Shape (bach_size, feature_map_height, feature_map_width, 2 * 4 * nb_anchors)
            if self.with_img_data : the metadata of the images
        '''
        # For each selected index, get image & preprocess it, and retrieve max resized_width & resized_height
        batch_prepared_img_data = []
        max_resized_width = 0
        max_resized_height = 0
        for ind in index_array:
            img_data = copy.deepcopy(self.img_data_list[ind])
            prepared_img_data = self.model._generate_images_with_bboxes(img_data, horizontal_flip=self.horizontal_flip,
                                                                        vertical_flip=self.vertical_flip, rot_90=self.rot_90)
            batch_prepared_img_data.append(prepared_img_data)
            max_resized_width = max(max_resized_width, prepared_img_data['resized_width'])
            max_resized_height = max(max_resized_height, prepared_img_data['resized_height'])

        # Set format X images
        batch_img_shape = (len(index_array), max_resized_height, max_resized_width, 3)
        batch_x_img = np.zeros(batch_img_shape)  # Def. pixels noirs

        # Get input images
        for ind, prepared_img_data in enumerate(batch_prepared_img_data):
            prepared_img_data['batch_width'] = max_resized_width
            prepared_img_data['batch_height'] = max_resized_height
            img = prepared_img_data['img']
            batch_x_img[ind, :img.shape[0], :img.shape[1], :] = img

        # Get the input ROIs - need to be formatted
        if self.shared_model_trainable:
            rpn_predictions_cls, rpn_predictions_regr = self.rpn_clone.predict(batch_x_img)
        else:
            rpn_predictions_cls, rpn_predictions_regr = self.model.model_rpn.predict(batch_x_img)
        rois_coordinates = utils_object_detectors.get_roi_from_rpn_predictions(self.model, batch_prepared_img_data,
                                                                               rpn_predictions_cls, rpn_predictions_regr)

        # If test, we return the images and the associated ROIs
        if self.is_test:
            # Get the ROIs with the right format
            batch_x_rois = utils_object_detectors.get_classifier_test_inputs(rois_coordinates)
            if self.with_img_data:
                return {'input_img': batch_x_img, 'input_rois': batch_x_rois}, batch_prepared_img_data
            else:  # Usually, this case never happens
                return {'input_img': batch_x_img, 'input_rois': batch_x_rois}
        # Otherwise, we also manage the targets y
        else:
            # Get the inputs and the targets of the classifier
            batch_x_rois, Y1_classifier, Y2_classifier = utils_object_detectors.get_classifier_train_inputs_and_targets(self.model, batch_prepared_img_data, rois_coordinates)
            if batch_x_rois is None:
                self.model.logger.warning("We have an image batch without ROI !!!")
                return next(self)  # We try another batch ...
            if self.with_img_data:
                return {'input_img': batch_x_img, 'input_rois': batch_x_rois}, {'dense_class': Y1_classifier, 'dense_regr': Y2_classifier}, batch_prepared_img_data
            else:
                return {'input_img': batch_x_img, 'input_rois': batch_x_rois}, {'dense_class': Y1_classifier, 'dense_regr': Y2_classifier}


class ModelCheckpointAll(ModelCheckpoint):
    '''A Callback to save the whole model and not only the model currently being fitted.
    In order to do so, we overload the class ModelCheckpoint by redefining its method _save_model
    '''

    def __init__(self, model_all, **kwargs) -> None:
        '''Initialization of the class

        Args:
            model_all (Model): Whole model (RPN & Classifier) of the Faster RCNN
        '''
        super().__init__(**kwargs)
        self.model_all = model_all

    def _save_model(self, epoch: int, batch, logs: dict) -> None:
        """Saves the model.

        Small trick : we temporarily set the model to model_all, save it, and then reload model_main

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
            is set to `epoch`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        # Small trick on the models
        model_main = self.model
        self.model = self.model_all
        # Save
        super()._save_model(epoch, batch, logs)
        # Fix trick
        self.model = model_main


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
