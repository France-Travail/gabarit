#!/usr/bin/env python3

## Utils - tools-functions for deep_learning keras models
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
# Functions :
# - f1 -> f1 score, to use as custom metrics
# - f1_loss -> f1 loss, to use as custom loss


import logging
from functools import partial
from typing import Callable, Any

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax


def recall(y_true, y_pred) -> float:
    '''Recall to use as a custom metrics

    Args:
        y_true: Ground truth values
        y_pred: The predicted values
    Returns:
        float: metric
    '''
    y_pred = K.round(y_pred)
    y_true = K.cast(y_true, 'float32')  # Fix : TypeError: Input 'y' of 'Mul' Op has type float32 that does not match type int32 of argument 'x'.

    ground_positives = K.sum(y_true, axis=0) + K.epsilon()  # We add an epsilon -> manage the case where a class is absent in the batch

    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    recall = tp / (tp + fn + K.epsilon())
    recall = tf.where(tf.math.is_nan(recall), tf.zeros_like(recall), recall)

    weighted_recall = recall * ground_positives / K.sum(ground_positives)
    weighted_recall = K.sum(weighted_recall)

    return weighted_recall


def precision(y_true, y_pred) -> float:
    '''Precision, to use as custom metrics

    Args:
        y_true: Ground truth values
        y_pred: The predicted values
    Returns:
        float: metric
    '''
    y_pred = K.round(y_pred)
    y_true = K.cast(y_true, 'float32')  # Fix : TypeError: Input 'y' of 'Mul' Op has type float32 that does not match type int32 of argument 'x'.

    ground_positives = K.sum(y_true, axis=0) + K.epsilon()  # We add an epsilon -> manage the case where a class is absent in the batch

    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    precision = tf.where(tf.math.is_nan(precision), tf.zeros_like(precision), precision)

    weighted_precision = precision * ground_positives / K.sum(ground_positives)
    weighted_precision = K.sum(weighted_precision)

    return weighted_precision


def f1(y_true, y_pred) -> float:
    '''f1 score, to use as custom metrics

    - /!\\ To use with a big batch size /!\\ -

    From:
        https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
        https://stackoverflow.com/questions/59963911/how-to-write-a-custom-f1-loss-function-with-weighted-average-for-keras

    Args:
        y_true: Ground truth values
        y_pred: The predicted values
    Returns:
        float: metric
    '''
    # Round pred to 0 & 1
    y_pred = K.round(y_pred)
    y_true = K.cast(y_true, 'float32')  # Fix : TypeError: Input 'y' of 'Mul' Op has type float32 that does not match type int32 of argument 'x'.

    ground_positives = K.sum(y_true, axis=0) + K.epsilon()  # We add an epsilon -> manage the case where a class is absent in the batch

    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
    weighted_f1 = K.sum(weighted_f1)

    return weighted_f1


def f1_loss(y_true, y_pred) -> float:
    '''f1 loss, to use as custom loss

    - /!\\ To use with a big batch size /!\\ -

    From:
        https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
        https://stackoverflow.com/questions/59963911/how-to-write-a-custom-f1-loss-function-with-weighted-average-for-keras

    Args:
        y_true: Ground truth values
        y_pred: The predicted values
    Returns:
        float: metric
    '''
    # TODO : Find a mean of rounding y_pred
    # TODO : Problem : models will quickly converge on probabilities 1.0 & 0.0 to optimize this loss....
    # We can't round here :(
    # Please make sure that all of your ops have a gradient defined (i.e. are differentiable).
    # Common ops without gradient: K.argmax, K.round, K.eval.
    y_true = K.cast(y_true, 'float32')  # Fix : TypeError: Input 'y' of 'Mul' Op has type float32 that does not match type int32 of argument 'x'.

    ground_positives = K.sum(y_true, axis=0) + K.epsilon()  # We add an epsilon -> manage the case where a class is absent in the batch

    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
    weighted_f1 = K.sum(weighted_f1)

    return 1 - weighted_f1


def fb_loss(b: float, y_true, y_pred) -> float:
    '''fB loss, to use as custom loss

    - /!\\ To use with a big batch size /!\\ -

    From:
        https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
        https://stackoverflow.com/questions/59963911/how-to-write-a-custom-f1-loss-function-with-weighted-average-for-keras

    Args:
        b (float): importance recall in the calculation of the fB score
        y_true: Ground truth values
        y_pred: The predicted values
    Returns:
        float: metric
    '''
    # TODO : Find a mean of rounding y_pred
    # TODO : Problem : models will quickly converge on probabilities 1.0 & 0.0 to optimize this loss....
    # We can't round here :(
    # Please make sure that all of your ops have a gradient defined (i.e. are differentiable).
    # Common ops without gradient: K.argmax, K.round, K.eval.
    y_true = K.cast(y_true, 'float32')  # Fix : TypeError: Input 'y' of 'Mul' Op has type float32 that does not match type int32 of argument 'x'.

    ground_positives = K.sum(y_true, axis=0) + K.epsilon()  # We add an epsilon -> manage the case where a class is absent in the batch

    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    fb = (1 + b**2) * p * r / ((p * b**2) + r + K.epsilon())
    fb = tf.where(tf.math.is_nan(fb), tf.zeros_like(fb), fb)

    weighted_fb = fb * ground_positives / K.sum(ground_positives)
    weighted_fb = K.sum(weighted_fb)

    return 1 - weighted_fb


def get_fb_loss(b: float = 2.0) -> Callable:
    ''' Gets a fB-score loss

    Args:
        b (float): importance recall in the calculation of the fB score
    Returns:
        Callable: fb_loss
    '''
    # - /!\ Utilisation partial obligatoire pour pouvoir pickle des fonctions dynamiques ! /!\ -
    fn = partial(fb_loss, b)
    # FIX:  AttributeError: 'functools.partial' object has no attribute '__name__'
    fn.__name__ = 'fb_loss'  # type: ignore
    return fn


def weighted_binary_crossentropy(pos_weight: float, target, output) -> float:
    '''Weighted binary crossentropy between an output tensor
    and a target tensor. pos_weight is used as a multiplier
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits

    Args:
        pos_weight (float): poid classe positive, to be tuned
        target: Target tensor
        output: Output tensor
    Returns:
        float: metric
    '''
    target = K.cast(target, 'float32')
    output = K.cast(output, 'float32')
    # transform back to logits
    _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(target, output, pos_weight=pos_weight)
    return tf.reduce_mean(loss, axis=-1)


def get_weighted_binary_crossentropy(pos_weight: float = 10.0) -> Callable:
    ''' Gets a "weighted binary crossentropy" loss
    From https://stats.stackexchange.com/questions/261128/neural-network-for-multi-label-classification-with-large-number-of-classes-outpu
    TO BE ADDED IN custom_objects : 'weighted_binary_crossentropy': utils_deep_keras.get_weighted_binary_crossentropy(pos_weight=...)

    Args:
        pos_weight (float): Weight of the positive class, to be tuned
    Returns:
        Callable: Weighted binary crossentropy loss
    '''
    # - /!\ Use of partial mandatory in order to be able to pickle dynamical functions ! /!\ -
    fn = partial(weighted_binary_crossentropy, pos_weight)
    # FIX:  AttributeError: 'functools.partial' object has no attribute '__name__'
    fn.__name__ = 'weighted_binary_crossentropy'  # type: ignore
    return fn


# ** EXPERIMENTAL **
# ** EXPERIMENTAL **
# ** EXPERIMENTAL **

# From Gaëlle JOUIS Thesis

class AttentionAverage(Layer):
    def __init__(self, attention_hops, **kwargs) -> None:
        self.attention_hops = attention_hops
        self.applied_axis = 1
        super(AttentionAverage, self).__init__()

    def get_config(self) -> Any:
        '''Gets the config'''
        config = super().get_config().copy()
        config.update({
            'attention_hops': self.attention_hops
        })
        return config

    def call(self, input) -> Any:
        return tf.divide(tf.reduce_sum(input, self.applied_axis), self.attention_hops)


class AttentionWithContext(Layer):
    '''Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
    '''
    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None, bias=True,
                 return_attention=False, **kwargs):
        self.return_attention = return_attention
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def get_config(self) -> Any:
        config = super().get_config().copy()
        config.update({
            'return_attention': self.return_attention,
            'bias': self.bias
        })
        return config

    def build(self, input_shape) -> None:
        assert len(input_shape) == 3
        input_shape_list = input_shape.as_list()

        self.W = self.add_weight(shape=((input_shape_list[-1], input_shape_list[-1])),
                                 name='{}_W'.format(self.name))
        if self.bias:
            self.b = self.add_weight(shape=(input_shape_list[-1],),
                                     name='{}_b'.format(self.name))

        self.u = self.add_weight(shape=(input_shape_list[-1],),
                                 name='{}_u'.format(self.name))

        super(AttentionWithContext, self).build(input_shape.as_list())

    def compute_mask(self, input, input_mask=None) -> Any:
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None) -> Any:
        uit = tf.tensordot(x, self.W, axes=1)
        if self.bias:
            uit += self.b
        uit = activations.tanh(uit)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = activations.exponential(ait)
        # Apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= tf.cast(mask, K.floatx())
        # In some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= tf.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape) -> Any:
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
        else:
            return tf.TensorShape([input_shape[0].value, input_shape[-1].value])


def softmax_axis(x) -> float:
    return softmax(x, axis=1)

# ** EXPERIMENTAL **
# ** EXPERIMENTAL **
# ** EXPERIMENTAL **


# /!\ Important -> This dictionary defines the "custom" objets used in our models
# /!\ Important -> They are mandatory in order to serialize and save the model
# /!\ Important -> All customs objects must be added to it
# TODO: to be improved
custom_objects = {
    'f1': f1,
    'f1_loss': f1_loss,
    'recall': recall,
    'precision': precision,
    'AttentionWithContext': AttentionWithContext,
    'AttentionAverage': AttentionAverage,
    'softmax_axis': softmax_axis,
}


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
