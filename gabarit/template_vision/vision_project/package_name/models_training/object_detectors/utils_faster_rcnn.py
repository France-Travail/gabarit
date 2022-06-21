#!/usr/bin/env python3
# type: ignore

## Utils - tools-functions for Faster RCNN model
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


import logging
from typing import Callable

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import categorical_crossentropy

# Get logger
logger = logging.getLogger(__name__)


class RoiPoolingLayer(Layer):
    '''Layer selecting a zone from a ROI in a features map and resize it

    # Input shape
        List of two 4D tensors [X_img, X_roi] with shape:
        X_img : list of images
            (batch_size, cols, rows, channels)
        X_roi : list of ROI with 4 coordinates (x, y, w, h)
            (batch_size, nb_rois, 4)
    # Output shape
        5D tensor with shape:
            (batch_size, nb_rois, pool_size, pool_size, channels)
            pool_size is a parameter of resizing of the features map
    '''
    def __init__(self, pool_size: int, **kwargs) -> None:
        '''Initialization of the layer

        Args:
            pool_size (int): Output size of the layer
        '''
        self.pool_size = pool_size
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, None, self.pool_size, self.pool_size, self.nb_channels

    def cut_feature_map(self, feature_map, roi):
        '''Cuts a features map with a ROI

        Args:
            feature_map : input features map
                # Shape : (cols, rows, channels)
            roi : input ROI
                # Shape : (4,)
        '''
        x, y, h, w = roi[0], roi[1], roi[2], roi[3]
        return tf.image.resize(feature_map[y: y + h, x: x + w, :], (self.pool_size, self.pool_size))

    def call(self, x: list, mask=None):
        '''Call to the layer

        Args:
            x (list): List of two tensors
                0 -> features maps # Shape (batch_size, cols, rows, channels)
                1 -> rois # Shape (batch_size, nb_rois, 4)
        Returns:
            tensor: images (features maps) cut with the ROIs
                # Shape (batch_size, nb_rois, cols, rows, nb_channel)
        '''
        # Get the tensors
        feature_maps = x[0]  # Shape (batch_size, cols, rows, channels)
        rois = K.cast(x[1], 'int32')  # Shape (batch_size, nb_rois, 4)

        # We loop on each batch, and then, we loop on each ROI
        # We crop each image with the ROIs of the batch
        # We also resize (cf. pool_size)
        # We format the results and return it
        # TODO: WARNING, WE MUST HAVE THE SAME NUMBER OF ROIS PER IMAGE :'(
        output = tf.map_fn(lambda batch:
            # IN : fmap (h, w, nb_channel)
            # IN : rois (nb_rois, 4)
            tf.map_fn(
                # IN : fmap (h, w, nb_channel)
                # IN : roi (4,)
                lambda roi: self.cut_feature_map(batch[0], roi)
                , batch[1], fn_output_signature=tf.float32
                # OUT (pool_size, pool_size, nb_channel)
            )
            # OUT : (nb_rois, pool_size, pool_size, nb_channel)
        , (feature_maps, rois), fn_output_signature=tf.float32)
        # OUT : (batch_size, nb_rois, pool_size, pool_size, nb_channel)
        return output

    def get_config(self) -> dict:
        config = {'pool_size': self.pool_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_rpn_loss_regr(nb_anchors: int) -> Callable:
    '''Gets the RPN regression loss depending on the number of anchor of the model

    Args:
        nb_anchors (int): Number of anchors of the model
    Returns:
        Callable: RPN regression loss
    '''
    def rpn_loss_regr(y_true, y_pred) -> float:
        '''Calculates the RPN regression loss (Huber loss)
               0.5*x² (if x_abs < 1)
               x_abs - 0.5 (otherwise)

        Args:
            y_true: Model's targets. Shape (batch_size, height, width, 2*4*nb_anchors)
                first part : class of the anchor boxes -> object or background
                                the loss is calculated only on anchor boxes associated to an object
                second part : regression
            y_pred: Outputs of the model. Shape (batch_size, height, width, 4*nb_anchors)
        Returns:
            float: Calculated loss
        '''
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Evaluate difference
        ind_sep = 4 * nb_anchors  # Separation index of the two parts of y_true
        x = y_true[:, :, :, ind_sep:] - y_pred
        x_abs = K.abs(x)

        # If x_abs <= 1.0, x_bool = 1
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        # robust loss function (smooth L1)
        return K.sum(y_true[:, :, :, :ind_sep] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(1e-4 + y_true[:, :, :, :ind_sep])
    # Return loss
    return rpn_loss_regr


def get_rpn_loss_cls(nb_anchors: int) -> Callable:
    '''Gets the RPN classification loss depending on the number of anchor of the model

    Args:
        nb_anchors (int): Number of anchors of the model
    Returns:
        Callable: RPN classification loss
    '''
    def rpn_loss_cls(y_true, y_pred) -> float:
        '''Calculates the RPN classification loss (Cross entropy)

        Args:
            y_true: Model's targets. Shape (batch_size, height, width, 2*nb_anchors)
                first part : validity of the anchor box. Valid if positive (object match) or negative (background match),
                                                         Not valid if neutral (in between object and background)
                             the loss is calculated only on valid anchor boxes
                second part : classe of the anchor box
                                 0  --> background
                                 1  --> object
            y_pred: Outputs of the model. Shape (batch_size, height, width, nb_anchors)
        Returns:
            float: Calculated loss
        '''
        ind_sep = nb_anchors  # Separation index of the two parts of y_true
        return K.sum(y_true[:, :, :, :ind_sep] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, ind_sep:])) / K.sum(1e-4 + y_true[:, :, :, :ind_sep])
    # Return loss
    return rpn_loss_cls


def get_class_loss_regr(nb_classes: int) -> Callable:
    '''Gets the classifier regression loss depending on the number of classes of the model

    Args:
        nb_classes (int): Number of classes of the model
    Returns:
        Callable: Classifier regression loss
    '''
    def class_loss_regr(y_true, y_pred):
        '''Calculates the classifier regression loss (Huber loss)
               0.5*x² (if x_abs < 1)
               x_abs - 0.5 (otherwise)

        Args:
            y_true: Model's targets. Shape (batch_size, nb_bboxes, 2*4*num_classes)
                first part : ROI class
                                  Example with two classes (without taking 'bg' into account)
                                    -> [0, 0, 0, 0, 0, 0, 0, 0] : background
                                    -> [0, 0, 0, 0, 1, 1, 1, 1] : classe n°1
                                    etc...
                                    the loss is calculated only on ROIs associated to an object
                second part : regression
            y_pred: Outputs of the model. Shape (batch_size, nb_bboxes, 4*num_classes)
        Returns:
            float: Calculated loss
        '''
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Evaluate difference
        ind_sep = 4 * nb_classes  # Separation index of the two parts of y_true
        x = y_true[:, :, ind_sep:] - y_pred
        x_abs = K.abs(x)

        # If x_abs <= 1.0, x_bool = 1
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return K.sum(y_true[:, :, :ind_sep] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(1e-4 + y_true[:, :, :ind_sep])
    # Return loss
    return class_loss_regr


def class_loss_cls(y_true, y_pred):
    '''Calculates the classifier classification loss (Cross entropy)

    Args:
        y_true: Model's target. Shape (batch_size, nb_bboxes, nb_classes)
        y_pred: Outputs of the model. Shape (batch_size, nb_bboxes, nb_classes)
    Returns:
        float: Calculated loss
    '''
    return K.mean(categorical_crossentropy(y_true, y_pred))


def get_custom_objects_faster_rcnn(nb_anchors: int, nb_classes: int) -> dict:
    '''Gets the keras custom_objects depending of the number of anchors and of classes of a model

    Args:
        nb_anchors (int): Number of anchors of the model
        nb_classes (int): Number of classes of the model
    Returns:
        dict: Set of customs objects
    '''

    # /!\ Important -> This dictionary defines the "custom" objets used in our Faster RCNN models
    # /!\ Important -> They are mandatory in order to serialize and save the model
    # /!\ Important -> All customs objects must be added to it
    custom_objects_faster_rcnn = {
        'RoiPoolingLayer': RoiPoolingLayer,
        'rpn_loss_regr': get_rpn_loss_regr(nb_anchors),
        'rpn_loss_cls': get_rpn_loss_cls(nb_anchors),
        'class_loss_cls': class_loss_cls,
        'class_loss_regr': get_class_loss_regr(nb_classes),
    }
    return custom_objects_faster_rcnn


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
