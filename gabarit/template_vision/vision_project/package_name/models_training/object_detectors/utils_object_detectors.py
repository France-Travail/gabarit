#!/usr/bin/env python3
# type: ignore

## Utils - tools-functions for object detection tasks
# Copyright (C) <2018-2021>  <Agence Data Services, DSI Pôle Emploi>
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
# - draw_bboxes -> Adds bboxes to an image


import os
import cv2
import copy
import random
import inspect
import logging
import numpy as np
from skimage import io
from typing import Union, List, Callable, Tuple

# Get logger
logger = logging.getLogger(__name__)


#######################
# Display functions
#######################


def draw_bboxes_from_file(input_path: str, output_path: Union[str, None] = None, gt_bboxes: Union[List[dict], None] = None,
                          predicted_bboxes: Union[List[dict], None] = None) -> np.ndarray:
    '''Adds bboxes to an image from a file

    Args:
        input_path (str): Path to the input image
    Kwargs:
        output_path (str): Path to the output file. If None, the result is not saved
        gt_bboxes (list): List of "ground truth" bboxes to display
            Each entry must be a dictionary with keys x1, y1, x2, y2 and (optional) class
        predicted_bboxes (list): List of bboxes coming from a prediction (same format as gt_bboxes)
    Raises:
        FileNotFoundError: If the input file does not exist
    Returns:
        (np.ndarray) : The image with the boxes
    '''
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file {input_path} does not exist")
    # Load image
    input_img = io.imread(input_path)
    return draw_bboxes(input_img, output_path, gt_bboxes, predicted_bboxes)


def draw_bboxes(input_img: np.ndarray, output_path: Union[str, None] = None, gt_bboxes: Union[List[dict], None] = None,
                predicted_bboxes: Union[List[dict], None] = None) -> np.ndarray:
    '''Adds bboxes to an image (np.ndarray)

    Args:
        input_img (np.ndarray): Input image
    Kwargs:
        output_path (str): Path to the output file. If None, the result is not saved
        gt_bboxes (list): List of "ground truth" bboxes to display
            Each entry must be a dictionary with keys x1, y1, x2, y2 and (optional) class
        predicted_bboxes (list): List of bboxes coming from a prediction (same format as gt_bboxes)
    Raises:
        FileExistsError: If the output file already exists
    Returns:
        (np.ndarray) : The image with the boxes
    '''
    if output_path is not None and os.path.exists(output_path):
        raise FileExistsError(f"The file {output_path} already exists")

    if gt_bboxes is None:
        gt_bboxes = []
    if predicted_bboxes is None:
        predicted_bboxes = []
    # Define colors
    green = (0, 255, 0, 255)
    red = (255, 0, 0, 255)
    # Create green rectangles for each bbox
    for bbox in gt_bboxes:
        draw_rectangle_from_bbox(input_img, bbox, color=green, thickness=5)
    # Create red rectangles for each predicted bbox
    for bbox in predicted_bboxes:
        draw_rectangle_from_bbox(input_img, bbox, color=red, thickness=5)
    if output_path is not None:
        io.imsave(output_path, input_img)
        logger.info(f"Image saved here : {output_path}")
    return input_img


def draw_rectangle_from_bbox(img: np.array, bbox: dict, color: Union[tuple, None] = None,
                             thickness: Union[int, None] = None, with_center: bool = False):
    '''Draws a rectangle in the image and adds a text (optional)

    Args:
        img (np.ndarray): The considered image
        bbox (dict): The dictionary containing the coordinates and the text
        color (tuple): A RGB tuple giving the color of the rectangle
        thickness (int): The thickness of the rectangle
        with_center (bool): If True, also draws the center of the rectangle
    Raises:
        ValueError: If one of the keys 'x1', 'y1', 'x2', 'y2' is missing
    '''
    # Check mandatory keys
    if any([key not in bbox.keys() for key in ['x1', 'y1', 'x2', 'y2']]):
        raise ValueError("One of the mandatory keys ('x1', 'y1', 'x2', 'y2') is missing in the object bbox.")
    # Process
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    class_name = bbox.get('class', None)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if class_name is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        if 'proba' in bbox:
            proba = format(bbox['proba'], ".2f")
            class_name = class_name + f" ({proba})"
        cv2.putText(img, class_name, (x1 + 5, y1 + 30), font, 1, color, 2)
    if with_center:
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(img, (center_x, center_y), 3, color, -1)


#######################
# Coordinates functions
#######################

# Notations
# x, x1 -> x coordinate of the upper left corner of a bbox
# y, y1 -> y coordinate of the upper left corner of a bbox
# x2 -> x coordinate of the bottom right corner of a bbox
# y2 -> y coordinate of the bottom right corner of a bbox
# cx -> x coordinate of the center of a bbox
# cy -> y coordinate of the center of a bbox
# w -> width of a bbox
# h -> height of a bbox
# xyxy -> x1, y1, x2, y2 format (opposite points format)
# xyhw -> x, y, h, w format
# cxcyhw -> cx, cy, h, w format


def check_coordinates_validity(function: Callable) -> Callable:
    '''Decorator to make sure that the coordinates are valid

    Args:
        function (Callable): Function to decorate
    Raises:
        ValueError: If a set of coordinates is impossible
    Returns:
        function: The decorated function
    '''
    # Get wrapper
    def wrapper(*args, **kwargs):
        '''Wrapper'''
        # Get vars.
        f_args = inspect.getfullargspec(function).args
        x1 = kwargs['x1'] if 'x1' in kwargs.keys() else (args[f_args.index('x1')] if 'x1' in f_args else None)
        y1 = kwargs['y1'] if 'y1' in kwargs.keys() else (args[f_args.index('y1')] if 'y1' in f_args else None)
        x2 = kwargs['x2'] if 'x2' in kwargs.keys() else (args[f_args.index('x2')] if 'x2' in f_args else None)
        y2 = kwargs['y2'] if 'y2' in kwargs.keys() else (args[f_args.index('y2')] if 'y2' in f_args else None)
        x = kwargs['x'] if 'x' in kwargs.keys() else (args[f_args.index('x')] if 'x' in f_args else None)
        y = kwargs['y'] if 'y' in kwargs.keys() else (args[f_args.index('y')] if 'y' in f_args else None)
        w = kwargs['w'] if 'w' in kwargs.keys() else (args[f_args.index('w')] if 'w' in f_args else None)
        h = kwargs['h'] if 'h' in kwargs.keys() else (args[f_args.index('h')] if 'h' in f_args else None)
        # Apply checks
        if x1 is not None:
            assert x1 >= 0, 'x1 must be non negative'
        if y1 is not None:
            assert y1 >= 0, 'y1 must be non negative'
        if x2 is not None:
            assert x2 >= 0, 'x2 must be non negative'
        if y2 is not None:
            assert y2 >= 0, 'y2 must be non negative'
        if x is not None:
            assert x >= 0, 'x must be non negative'
        if y is not None:
            assert y >= 0, 'y must be non negative'
        if w is not None:
            assert w > 0, 'w must be positive'
        if h is not None:
            assert h > 0, 'h must be positive'
        if x1 is not None and x2 is not None:
            assert x2 > x1, 'x2 must be bigger than x1'
        if y1 is not None and y2 is not None:
            assert y2 > y1, 'y2 must be bigger than y1'
        # Return
        return function(*args, **kwargs)
    return wrapper


@check_coordinates_validity
def xyxy_to_xyhw(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    '''Changes a rectangle in the format xyxy (x1, y1, x2, y2) to
    the format xyhw (x, y, h, w)

    Args :
        x1 (float): x coordinate of the upper left point
        y1 (float): y coordinate of the upper left point
        x2 (float): x coordinate of the bottom right point
        y2 (float): y coordinate of the bottom right point
    Returns:
        float: x coordinate of the upper left point
        float: y coordinate of the upper left point
        float: height of the rectangle
        float: width of the rectangle
    '''
    h = y2 - y1
    w = x2 - x1
    return x1, y1, h, w


@check_coordinates_validity
def xyhw_to_xyxy(x: float, y: float, h: float, w: float) -> Tuple[float, float, float, float]:
    '''Changes a rectangle in the format xyhw (x, y, h, w) to
    the format xyxy (x1, y1, x2, y2)

    Args :
        x (float): x coordinate of the upper left point
        y (float): y coordinate of the upper left point
        h (float): height of the rectangle
        w (float): width of the rectangle
    Returns:
        float: x coordinate of the upper left point
        float: y coordinate of the upper left point
        float: x coordinate of the bottom right point
        float: y coordinate of the bottom right point
    '''
    x2 = x + w
    y2 = y + h
    return x, y, x2, y2


@check_coordinates_validity
def xyxy_to_cxcyhw(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    '''Changes a rectangle in the format xyxy (x1, y1, x2, y2) to
    the format cxcyhw (cx, cy, h, w)

    Args :
        x1 (float): x coordinate of the upper left point
        y1 (float): y coordinate of the upper left point
        x2 (float): x coordinate of the bottom right point
        y2 (float): y coordinate of the bottom right point
    Returns:
        float: x coordinate of the center of the rectangle
        float: y coordinate of the center of the rectangle
        float: height of the rectangle
        float: width of the rectangle
    '''
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    return cx, cy, height, width


@check_coordinates_validity
def get_area_from_xyxy(x1: float, y1: float, x2: float, y2: float) -> float:
    '''Gives the area (absolute, not relative) of a rectangle in opposite points format

    Args :
        x1 (float): x coordinate of the upper left point
        y1 (float): y coordinate of the upper left point
        x2 (float): x coordinate of the bottom right point
        y2 (float): y coordinate of the bottom right point

    Returns:
        float : The absolute area of the rectangle
    '''
    return abs((x2 - x1) * (y2 - y1))


def get_iou(coordinatesA: Tuple[float, float, float, float], coordinatesB: Tuple[float, float, float, float]) -> float:
    '''Gives the intersection over union (iou) from the coordinates of two
    rectangles (in opposite points format)

    Args:
        coordinatesA (tuple<float>): The coordinates of the first rectangle in the format (x1, y1, x2, y2)
        coordinatesB (tuple<float>): The coordinates of the second rectangle in the format (x1, y1, x2, y2)
    Returns:
        float: Intersection over union of the two rectangles
    '''
    # Get areas of A and B
    areaA = get_area_from_xyxy(*coordinatesA)
    areaB = get_area_from_xyxy(*coordinatesB)
    # If any null, iou is equal to 0
    if areaA == 0 or areaB == 0:
        return 0

    x1A, y1A, x2A, y2A = coordinatesA
    x1B, y1B, x2B, y2B = coordinatesB
    # Get coordinates of the intersection
    x1_inter = max(x1A, x1B)
    y1_inter = max(y1A, y1B)
    x2_inter = min(x2A, x2B)
    y2_inter = min(y2A, y2B)
    # Get intersection area
    if x2_inter > x1_inter and y2_inter > y1_inter:
        area_inter = get_area_from_xyxy(x1_inter, y1_inter, x2_inter, y2_inter)
    else:
        area_inter = 0

    # Return IOU
    return area_inter / (areaA + areaB - area_inter)


######################################
# Functions on image size and features maps
######################################


def get_new_img_size_from_min_side_size(height: int, width: int, img_min_side_size: int = 300) -> Tuple[int, int]:
    '''Gets the new dimensions of an image so that the smaller dimension is equal to img_min_side_size
    but keeping the ratio.

    Args:
        height (int): Height of the base image
        width (int): Width of the base image
    Kwargs:
        img_min_side_size (int): Final size of the smaller dimension
    Raises:
        ValueError: If incorrect dimension of the image (< 1)
        ValueError: If img_min_side_size is incorrect (< 1)
    Returns:
        int: Resized height
        int: Resized width
    '''
    # Manage errors
    if height < 1 or width < 1:
        raise ValueError(f"Incorrect dimension of the image (H : {height} / W : {width})")
    if img_min_side_size < 1:
        raise ValueError(f"Minimal size wanted incorrect ({img_min_side_size})")
    # Width smaller than height, we calculates the new height and set the width to img_min_side_size
    if width <= height:
        f = float(img_min_side_size) / width
        resized_height = int(f * height)
        resized_width = img_min_side_size
    # Height smaller than width, we calculates the new width and set the height to img_min_side_size
    else:
        f = float(img_min_side_size) / height
        resized_width = int(f * width)
        resized_height = img_min_side_size
    # Return
    return resized_height, resized_width


def get_feature_map_size(input_height: int, input_width: int, subsampling_ratio: int) -> Tuple[int, int]:
    '''Gives the size of the features map given the height and width of the image using
    the subsampling_ratio of the shared model. For exemple, for VGG16, the subsampling_ratio is 16

    Args:
        input_height (int): Height of the image
        input_width (int): Width of the image
        subsampling_ratio (int): Subsampling ratio of the shared model
    Raises:
        ValueError: If incorrect dimension of the image (< 1)
        ValueError: If the subsampling_ratio is incorrect (< 1)
    Returns:
        int: Height of the features map
        int: Width of the features map
    '''
    # Manage errors
    if input_height < 1 or input_width < 1:
        raise ValueError(f"Bad image shape (H : {input_height} / W : {input_width})")
    if subsampling_ratio < 1:
        raise ValueError(f"Bad subsampling ratio ({subsampling_ratio})")
    # Process
    return input_height // subsampling_ratio, input_width // subsampling_ratio


#############################################
# Functions related to the Faster RCNN model
#############################################

######
# Useful functions
######


def calc_regr(coordinates_bbox: Tuple[float, float, float, float],
              coordinates_anchor: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    '''Gives the target of a regression given the coordinates of a bbox and of an anchor (or a ROI)

    Args:
        coordinates_bbox (tuple): The coordinates of a bbox in opposite points format
        coordinates_anchor (tuple): The coordinates of an anchor (or a ROI) in opposite points format
    Returns:
        float: Gap between the centers (x coordinate) normalized by the width of the anchor
        float: Gap between the centers (y coordinate) normalized by the height of the anchor
        float: Height ratio : bbox / anchor (log version)
        float: Width ratio : bbox / anchor (log version)
    '''
    cx_bbox, cy_bbox, height_bbox, width_bbox = xyxy_to_cxcyhw(*coordinates_bbox)
    cx_anchor, cy_anchor, height_anchor, width_anchor = xyxy_to_cxcyhw(*coordinates_anchor)
    tx = (cx_bbox - cx_anchor) / width_anchor
    ty = (cy_bbox - cy_anchor) / height_anchor
    th = np.log(height_bbox / height_anchor)
    tw = np.log(width_bbox / width_anchor)
    return tx, ty, th, tw


def apply_regression(coordinates_and_regression: np.ndarray) -> Tuple[float, float, float, float]:
    '''Applies the result of a regression on an anchor box (or a ROI) given in xyhw format.

    Args:
        coordinates_and_regression (np.ndarray): An array composed of 8 objects x_anc, y_anc, h_anc, w_anc, tx, ty, th, tw.
            (x_anc, y_anc, h_anc, w_anc) are the coordinates of the anchor (or of a ROI)
            (tx, ty, th, tw) are the predictions of a regression
            # Shape (8,)
    Returns:
        float: coordinates after regression applied on the anchor box (or on the ROI) - x coordinate of the upper left corner
        float: coordinates after regression applied on the anchor box (or on the ROI) - y coordinate of the upper left corner
        float: coordinates after regression applied on the anchor box (or on the ROI) - height
        float: coordinates after regression applied on the anchor box (or on the ROI) - width
    '''
    x_anc, y_anc, h_anc, w_anc, tx, ty, th, tw = coordinates_and_regression
    w_roi = np.exp(tw) * w_anc  # Take the inverse of the log and get rid of the normalization
    h_roi = np.exp(th) * h_anc  # Take the inverse of the log and get rid of the normalization
    x_roi = (tx * w_anc + (x_anc + w_anc / 2)) - w_roi / 2  # Get rid of the normalization, then add the center of the anchor = center of ROI, then remove half the width to get x1
    y_roi = (ty * h_anc + (y_anc + h_anc / 2)) - h_roi / 2  # Get rid of the normalization, then add the center of the anchor = center of ROI, then remove half the height to get y1
    return x_roi, y_roi, h_roi, w_roi


def non_max_suppression_fast(img_boxes_coordinates: np.ndarray, img_boxes_probas: np.ndarray, nms_overlap_threshold: float,
                             nms_max_boxes: int, img_boxes_classes: Union[np.ndarray, None] = None) -> np.ndarray:
    '''Filters boxes in order to limit overlaps on the same object using a list of boxes (ROIs or final predictions)
    and the probabilities of matching with an object.

    Args:
        img_boxes_coordinates (np.ndarray): The coordinates of the boxes (in opposite points format)
            # shape: (nb_boxes, 4)
        img_boxes_probas (np.ndarray): The probabilities associated to the boxes
            # shape: (nb_boxes)
        nms_overlap_threshold (float): The iou value above which we assume that two boxes overlap
        nms_max_boxes (int): The maximal number of boxes that this function can return
    Kwargs:
        img_boxes_classes (np.ndarray): The classes associated with the boxes (optional)
            # shape: (nb_boxes)
    Raises:
        ValueError: If img_boxes_probas is not the same length as img_boxes_coordinates
        ValueError: If nms_overlap_threshold <= 0 or > 1
        ValueError: If nms_max_boxes < 1
        ValueError: If img_boxes_classes is not the same length as img_boxes_coordinates (if != None)
    Returns:
        np.ndarray: List of kept boxes
            # shape: (nb_boxes_kept, 4)
        np.ndarray: Associated probabilities
        np.ndarray: Associated classes (if prediction)
    '''
    # code taken from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, returns an empty list

    # Manage errors
    if img_boxes_coordinates.shape[0] != img_boxes_probas.shape[0]:
        raise ValueError("The arrays img_boxes_coordinates and img_boxes_probas must have the same length.")
    if not 0 < nms_overlap_threshold <= 1:
        raise ValueError("The value of nms_overlap_threshold must be between 0 and 1 (0 excluded, 1 included)")
    if nms_max_boxes < 1:
        raise ValueError("The argument nms_max_boxes must be positive")
    if img_boxes_classes is not None and img_boxes_coordinates.shape[0] != img_boxes_classes.shape[0]:
        raise ValueError("The arrays img_boxes_coordinates and img_boxes_classes must have the same length.")

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
    #   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list

    # If empty, return an empty array
    if len(img_boxes_coordinates) == 0:
        return np.array([]), np.array([]), np.array([])

    # Grab the coordinates of the boxes & calculate the areas
    x1_box, y1_box, x2_box, y2_box = (img_boxes_coordinates[:, i] for i in range(4))
    boxes_areas = (x2_box - x1_box) * (y2_box - y1_box)

    # We now loop over each boxes, sorted by max probas
    picked_index = []
    idxs = np.argsort(img_boxes_probas)
    # Keep looping while some indexes still remain in the list
    while len(idxs) > 0:
        # If we have enough boxes, break
        if len(picked_index) >= nms_max_boxes:
            break

        # Add highest proba remaining to picked indexes
        picked_index.append(idxs[-1])

        # Find intersection area between picked box & remaining candidates
        xx1_int = np.maximum(x1_box[idxs[-1]], x1_box[idxs[:-1]])
        yy1_int = np.maximum(y1_box[idxs[-1]], y1_box[idxs[:-1]])
        xx2_int = np.minimum(x2_box[idxs[-1]], x2_box[idxs[:-1]])
        yy2_int = np.minimum(y2_box[idxs[-1]], y2_box[idxs[:-1]])
        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)
        area_int = ww_int * hh_int

        # Get the union
        area_union = boxes_areas[idxs[-1]] + boxes_areas[idxs[:-1]] - area_int

        # Compute the overlap (i.e. iou)
        overlap = area_int / (area_union + 1e-6)

        # Delete last index (selected) & all indexes from the index list that have an IOU higher than a given threhsold
        idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1], np.where(overlap > nms_overlap_threshold)[0])))

    # Return only the boxes that were picked
    img_boxes_coordinates = img_boxes_coordinates[picked_index, :]
    if img_boxes_classes is not None:
        return img_boxes_coordinates, img_boxes_probas[picked_index], img_boxes_classes[picked_index]
    else:
        return img_boxes_coordinates, img_boxes_probas[picked_index], None


######
# Get targets for the RPN for an image
# Main function : get_rpn_targets
######


def get_rpn_targets(model, img_data_batch: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    '''Gives the classification and regression targets for the RPN

    Process : We defined a set of possible anchor boxes (def. 9). For each point of the features map,
              we look at the possible anchor boxes. We get back to the input image space and keep only
              the anchor boxes which are totally included in the image. Then, for each anchor box, we check
              if it matches with a bbox (via iou) and we define our target : match bbox vs match background
              and gap between anchor box and bbox for the regression part (only if there is a match on a bbox).
              We use this process for each image

    Args:
        model (ModelKerasFasterRcnnObjectDetector): Model used (contains all the necessary configs)
        img_data_batch (list): The list of img_data (dict) for the batch.
            Each entry is a dictionary with the content of an image (already preprocessed) and associated metadata:
                    - 'img' -> image in the numpy format (h, w, c), preprocessed and ready to be used by the model
                    - 'bboxes' -> (dict) associated bboxes (preprocessed image format)
                         'x1', 'x2', 'y1', 'y2'
                    - 'original_width' -> Original width of the image
                    - 'original_height' -> Original height of the image
                    - 'resized_width' -> Resized width of the image (ie. smaller dim set to img_min_side_size px (def 300))
                    - 'resized_height' -> Resized height of the image (ie. smaller dim set to img_min_side_size px (def 300))
                    - 'batch_width' -> Width of the images in the batch (max width of the batch, we pad the smaller images with zeroes)
                    - 'batch_height' -> Height of the images in the batch (max height of the batch, we pad the smaller images with zeroes)
    Returns:
        np.ndarray: Classification targets : [y_is_box_valid] + [y_rpn_overlap] for each image with :
                    - y_is_box_valid -> if a box is valid (and thus, should enter in the classification loss)
                    - y_rpn_overlap -> target of the classification ('pos', 'neg' or 'neutral')
            # Shape (batch_size, feature_map_height, feature_map_width, nb_anchors * 2)
        np.ndarray: Regression targets : [y_rpn_overlap (repeated x 4)] + [y_rpn_regr] for each image with :
                    - y_rpn_overlap -> if a box is an object (and thus, should enter in the regression loss)
                        repeated to account for the 4 coordinates
                    - y_rpn_regr -> regression targets
            # Shape (batch_size, feature_map_height, feature_map_width, nb_anchors * 2 * 4)
    '''
    # Extract params from model
    base_anchors = model.list_anchors
    nb_anchors = model.nb_anchors
    subsampling_ratio = model.shared_model_subsampling
    rpn_min_overlap = model.rpn_min_overlap
    rpn_max_overlap = model.rpn_max_overlap
    rpn_regr_scaling = model.rpn_regr_scaling
    num_regions = model.rpn_restrict_num_regions

    # Info batch size
    batch_size = len(img_data_batch)
    # Get size of the features map of the batch (for example by taking the first image)
    feature_map_height, feature_map_width = get_feature_map_size(img_data_batch[0]['batch_height'], img_data_batch[0]['batch_width'], subsampling_ratio)

    # Setup target arrays
    Y1 = np.zeros((batch_size, feature_map_height, feature_map_width, nb_anchors * 2))
    Y2 = np.zeros((batch_size, feature_map_height, feature_map_width, nb_anchors * 2 * 4))

    # We process each image
    for ind, img_data in enumerate(img_data_batch):

        # Info image data
        im_resized_height, im_resized_width = img_data['resized_height'], img_data['resized_width']
        image_bboxes = img_data['bboxes']

        # Get the "viable" anchor boxes : one for each couple (point features map, base anchor) except if it does not fit
        # in the image
        anchor_boxes_dict = get_all_viable_anchors_boxes(base_anchors, subsampling_ratio, feature_map_height,
                                                         feature_map_width, im_resized_height, im_resized_width)
        # Get iou for each couple (anchor box / bbox)
        anchor_boxes_dict = get_iou_anchors_bboxes(anchor_boxes_dict, image_bboxes)

        # Set anchor validity & type for each anchor
        # - pos & valid if match on a bbox (ie. an object)
        # - neg & valid if match on background
        # - neutral & invalid otherwise
        anchor_boxes_dict, bboxes_index_with_no_positive = set_anchors_type_validity(anchor_boxes_dict, image_bboxes, rpn_min_overlap, rpn_max_overlap)

        # We add at least one positive anchor box for each bbox which does not have one match
        # (in some cases, it is not possible, in that case : skip)
        anchor_boxes_dict = complete_at_least_one_anchor_per_bbox(anchor_boxes_dict, bboxes_index_with_no_positive)

        # Invalidate some anchors in order not to have too many
        anchor_boxes_dict = restrict_valid_to_n_regions(anchor_boxes_dict, num_regions=num_regions)

        # Add the regression target for the positive and valid anchors
        anchor_boxes_dict = add_regression_target_to_pos_valid(anchor_boxes_dict)

        # We have the anchors, their type and validity and the regression targets for the pos/valid anchors
        # We format the result. Here we initialize
        y_rpn_overlap = np.zeros((feature_map_height, feature_map_width, nb_anchors))  # Target classifier
        y_is_box_valid = np.zeros((feature_map_height, feature_map_width, nb_anchors))  # couples (ix, anchor) which will enter the loss (ie. are not neutral)
        y_rpn_regr = np.zeros((feature_map_height, feature_map_width, nb_anchors * 4))

        # For each anchor, we add data.
        # The deleted anchors (because not 'viable') are not in anchor_boxes_dict, BUT all their characteristics should be zero
        # which is the case thanks to the initialization
        for anchor_idx, anchor in anchor_boxes_dict.items():
            y_rpn_overlap[anchor_idx[0], anchor_idx[1], anchor_idx[2]] = 1 if anchor['anchor_type'] == 'pos' else 0
            y_is_box_valid[anchor_idx[0], anchor_idx[1], anchor_idx[2]] = anchor['anchor_validity']
            start_regr_index = 4 * anchor_idx[2]
            y_rpn_regr[anchor_idx[0], anchor_idx[1], start_regr_index: start_regr_index + 4] = anchor['regression_target']

        # We then concat all final arrays
        # For regression part, we add y_rpn_overlap (repeated) to y_rpn_regr in order to identify data that should be used by the regression loss
        y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=2)
        y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=2), y_rpn_regr], axis=2)

        # We scale the regression target
        y_rpn_regr[:, :, y_rpn_regr.shape[2] // 2:] *= rpn_regr_scaling

        # We finally update the output arrays
        Y1[ind, :, :, :] = y_rpn_cls
        Y2[ind, :, :, :] = y_rpn_regr

    # Return
    return Y1, Y2


def get_all_viable_anchors_boxes(base_anchors: List[tuple], subsampling_ratio: int, feature_map_height: int,
                                 feature_map_width: int, im_resized_height: int, im_resized_width: int) -> dict:
    '''Gets a dictionary of 'viable' anchor boxes.

    From a list of "base" anchors, we will take each point of a features map, get its initial coordinates
    (input of the model) and build as many anchors as "base" anchors with this point as a center.
    Then we filter out the ones which are outside the image

    Args:
        base_anchors (list): List of base anchors
        subsampling_ratio (int): Subsampling ratio of the shared model
        feature_map_height (int): Height of the features map
        feature_map_width (int): Width of the features map
        im_resized_height (int): Height of the input image (preprocessed, without padding)
        im_resized_width (int): Width of the input image (preprocessed, without padding)
    Returns:
        dict : set of 'viable' anchor boxes identified by (y, x, index_anchor)
    '''
    viable_anchor_boxes = {}
    # For each anchor...
    for index_anchor, (height_anchor, width_anchor) in enumerate(base_anchors):
        # For each point of the features map...
        for x_feature_map in range(feature_map_width):
            # x coordinate of the anchor, input format (ie in image space)
            x1_anchor = subsampling_ratio * (x_feature_map + 0.5) - width_anchor / 2  # center - width / 2
            x2_anchor = subsampling_ratio * (x_feature_map + 0.5) + width_anchor / 2  # center + width / 2
            # We do not consider the anchors outside the image (before padding)
            if x1_anchor < 0 or x2_anchor >= im_resized_width:
                continue
            for y_feature_map in range(feature_map_height):
                # y coordinate of the anchor, input format (ie in image space)
                y1_anchor = subsampling_ratio * (y_feature_map + 0.5) - height_anchor / 2  # center - height / 2
                y2_anchor = subsampling_ratio * (y_feature_map + 0.5) + height_anchor / 2  # center + height / 2
                # We do not consider the anchors outside the image (before padding)
                if y1_anchor < 0 or y2_anchor >= im_resized_height:
                    continue
                # We update the dictionary of the 'viable' anchor boxes (y, x, anchor)
                id_key = (y_feature_map, x_feature_map, index_anchor)
                viable_anchor_boxes[id_key] = {
                    'anchor_img_coordinates': (x1_anchor, y1_anchor, x2_anchor, y2_anchor)
                }
    # Check errors
    if len(viable_anchor_boxes) == 0:
        logger.error("No viable bbox for one of the input images.")
        logger.error("The size of the preprocessed images may be too small when compared to the list of anchors of the model.")
        raise RuntimeError("No viable bbox for one of the input images.")
    # Return
    return viable_anchor_boxes


def get_iou_anchors_bboxes(anchor_boxes_dict: dict, image_bboxes: List[dict]) -> dict:
    '''Gives the iou for each anchor boxes with all the bboxes of a list (for example, all the
    bboxes of an image)

    Args:
        anchor_boxes_dict (dict): Anchor boxes dictionary
            - 'anchor_img_coordinates': xyxy coordinates of the anchor boxe (input format, ie. image space)
        image_bboxes (list<dict>): List of bboxes
    Returns:
        dict: The input dictionary to which we added a bboxes field containing coordinates and iou
    '''
    # For each anchor ...
    for anchor_idx, anchor in anchor_boxes_dict.items():
        anchor['bboxes'] = {}
        anchor_img_coordinates = anchor['anchor_img_coordinates']
        # ... and for each bbox in the list ...
        for index_bbox, bbox in enumerate(image_bboxes):
            # ... we calculate the iou and add the info to the dictionary of anchors
            bbox_coordinates = (bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'])
            iou = get_iou(anchor_img_coordinates, bbox_coordinates)
            anchor['bboxes'][index_bbox] = {
                'iou': iou,
                'bbox_img_coordinates': bbox_coordinates,
            }
        anchor_boxes_dict[anchor_idx] = anchor
    # Return
    return anchor_boxes_dict


def set_anchors_type_validity(anchor_boxes_dict: dict, image_bboxes: List[dict], rpn_min_overlap: float,
                              rpn_max_overlap: float) -> Tuple[dict, list]:
    '''Defines the type and the validity of each anchor
        Type:
            - pos -> Match between the anchor and a bbox
            - neg -> Match between the anchor and the background
            - neutral -> In between the two, won't be used by the model
        Validity:
            - 1 -> If pos or neg
            - 0 -> If neutral
    Args:
        anchor_boxes_dict (dict): Anchor boxes dictionary
            - 'anchor_img_coordinates': Coordinates of the anchor box (input format ie. image space)
            - 'bboxes': bboxes with their coordinates xyxy (in image space) and iou
        image_bboxes (list<dict>): List of bboxes of the image
        rpn_min_overlap (float): Threshold below which a bbox is marked as negative
        rpn_max_overlap (float): Threshold aboce which a bbox is marked as positive
    Returns:
        dict: Dictionary of the anchors boxes (input dictionary) with type and validity added ('anchor_type', 'anchor_validity')
        list: Liste of the bboxes with no positive anchor associated
    '''
    bboxes_index_with_positive = set()
    # For each anchor ...
    for anchor_idx, anchor in anchor_boxes_dict.items():
        # Get the dictionary where the keys are the bboxes and the values, the iou
        dict_iou = {index_bbox: dict_bbox['iou'] for index_bbox, dict_bbox in anchor['bboxes'].items()}
        # Get max iou (if for some reason no bbox, set it to 0 (i.e 'neg'))
        max_iou = max(dict_iou.values()) if len(dict_iou) > 0 else 0
        # If we are above threshold max, the anchor is positive and valid
        if max_iou > rpn_max_overlap:
            anchor['anchor_type'] = 'pos'
            anchor['anchor_validity'] = 1
            best_bbox_index = max(dict_iou, key=dict_iou.get)
            anchor['best_bbox_index'] = best_bbox_index
            bboxes_index_with_positive.add(best_bbox_index)
        # If we are below threshold min, the anchor is negative and valid
        elif 0 <= max_iou < rpn_min_overlap:
            anchor['anchor_type'] = 'neg'
            anchor['anchor_validity'] = 1
            anchor['best_bbox_index'] = -1
        # Otherwise, it is invalid (and we set it to neutral)
        else:
            anchor['anchor_type'] = 'neutral'
            anchor['anchor_validity'] = 0
            anchor['best_bbox_index'] = -1
        anchor_boxes_dict[anchor_idx] = anchor
    # Get list of bboxes index without positive anchor
    bboxes_index_with_no_positive = [index_bbox for index_bbox in range(len(image_bboxes))
                                     if index_bbox not in bboxes_index_with_positive]
    return anchor_boxes_dict, bboxes_index_with_no_positive


def complete_at_least_one_anchor_per_bbox(anchor_boxes_dict: dict, bboxes_index_with_no_positive: List[dict]) -> dict:
    '''Completes the dictionary of anchor to have at least one positive anchor per bbox if it is not
    already the case.
    If a bbox is not associated to an anchor, we associate it to the anchor with which it has
    the biggest iou (if this anchor is not already associated with another bbox)
    Args:
        anchor_boxes_dict (dict): Anchor boxes dictionary
            - 'anchor_img_coordinates': Coordinates of the anchor box (input format ie. image space)
            - 'bboxes': bboxes with their coordinates xyxy (in image space) and iou
            - 'anchor_type': anchor type (pos, neg or neutral)
            - 'anchor_validity': anchor validity
            - 'best_bbox_index': bbox associated to this anchor
        bboxes_index_with_no_positive (list): List of bboxes with no positive anchor associated
    Returns:
        dict: Updated anchor boxes dictionary
    '''
    # For each missing bbox ...
    for index_bbox in bboxes_index_with_no_positive:
        # ... we look for the anchor box with the best iou ...
        best_iou = -1  # We could set it to 0 but there exists rare case where all the anchor boxes have a 0 iou.
        best_anchor_idx = -1
        for anchor_idx, anchor in anchor_boxes_dict.items():
            iou = anchor['bboxes'][index_bbox]['iou']
            if iou > best_iou:
                best_iou = iou
                best_anchor_idx = anchor_idx
        # ... and we update this anchor if it is not already positive
        anchor = anchor_boxes_dict[best_anchor_idx]
        if anchor['anchor_type'] != 'pos' and best_iou > 0:
            anchor['anchor_type'] = 'pos'
            anchor['anchor_validity'] = 1
            anchor['best_bbox_index'] = index_bbox
            anchor_boxes_dict[best_anchor_idx] = anchor  # Update
    # Return
    return anchor_boxes_dict


def restrict_valid_to_n_regions(anchor_boxes_dict: dict, num_regions: int) -> dict:
    '''Restricts the number of valid anchor boxes.
    If there are more positive anchor boxes than hald of num_regions :
        - we invalidate positive anchors until there are less than num_regions / 2
        - Then, we invalidate positive anchors until the number of valid anchors is equal to num_regions

    Args:
        anchor_boxes_dict (dict): Anchor boxes dictionary
            - 'anchor_img_coordinates': Coordinates of the anchor box (input format ie. image space)
            - 'bboxes': bboxes with their coordinates xyxy (in image space) and iou
            - 'anchor_type': anchor type (pos, neg or neutral)
            - 'anchor_validity': anchor validity
            - 'best_bbox_index': bbox associated to this anchor
        num_regions (int): The number of valid anchors we want to consider
    Returns:
        dict: Updated anchor boxes dictionary
    '''
    # We look at both positive and negative anchor boxes
    # No need to test for validity at this point, positive and negative anchor boxes are necessarily valid
    positive_anchor_indexes = [anchor_idx for anchor_idx, anchor in anchor_boxes_dict.items() if anchor['anchor_type'] == 'pos']
    negative_anchor_indexes = [anchor_idx for anchor_idx, anchor in anchor_boxes_dict.items() if anchor['anchor_type'] == 'neg']

    # First we invalidate the surplus of positive anchors if needed ...
    nb_pos = len(positive_anchor_indexes)
    nb_pos_to_invalidate = max(0, nb_pos - int(num_regions / 2))
    if nb_pos_to_invalidate > 0:
        # Random select
        anchors_indexes_to_unvalid = random.sample(positive_anchor_indexes, nb_pos_to_invalidate)
        for anchor_idx in anchors_indexes_to_unvalid:
            anchor_boxes_dict[anchor_idx]['anchor_validity'] = 0
        nb_pos = int(num_regions / 2)

    # ... Then we invalidate negative regions until we have num_regions valid anchor boxes
    nb_neg_to_invalidate = len(negative_anchor_indexes) + nb_pos - num_regions
    if nb_neg_to_invalidate > 0:
        # Random select
        anchors_indexes_to_unvalid = random.sample(negative_anchor_indexes, nb_neg_to_invalidate)
        for anchor_idx in anchors_indexes_to_unvalid:
            anchor_boxes_dict[anchor_idx]['anchor_validity'] = 0

    # Return updated dict
    return anchor_boxes_dict


def add_regression_target_to_pos_valid(anchor_boxes_dict: dict) -> dict:
    '''Add the regression target for positive and valid anchors
    Otherwise, keep (0, 0, 0, 0) and won't be used by the loss

    Args:
        anchor_boxes_dict (dict): Anchor boxes dictionary
            - 'anchor_img_coordinates': Coordinates of the anchor box (input format ie. image space)
            - 'bboxes': bboxes with their coordinates xyxy (in image space) and iou
            - 'anchor_type': anchor type (pos, neg or neutral)
            - 'anchor_validity': anchor validity
            - 'best_bbox_index': bbox associated to this anchor
    Returns:
        dict: Updated anchor boxes withe the regression targets
    '''
    # For each anchor ...
    for anchor_idx, anchor in anchor_boxes_dict.items():
        # ... and if the anchor is positive and valid ...
        if anchor['anchor_type'] == 'pos' and anchor['anchor_validity'] == 1:
            # ... we get the regression target between this anchor and the closest bbox (best iou)
            coordinates_anchor = anchor['anchor_img_coordinates']
            best_bbox_index = anchor['best_bbox_index']
            coordinates_bbox = anchor['bboxes'][best_bbox_index]['bbox_img_coordinates']
            anchor['regression_target'] = calc_regr(coordinates_bbox, coordinates_anchor)
        # Otherwise, default to 0
        else:
            anchor['regression_target'] = (0, 0, 0, 0)
        anchor_boxes_dict[anchor_idx] = anchor
    # Return updated dict
    return anchor_boxes_dict


######
# Get ROIs from RPN predictions
# Main function : get_roi_from_rpn_predictions
######


def get_roi_from_rpn_predictions(model, img_data_batch: List[dict], rpn_predictions_cls: np.ndarray,
                                 rpn_predictions_regr: np.ndarray) -> List[np.ndarray]:
    '''Converts the output layers of the RPN (classification and regression) in ROIs


    Process : We get the prediction results of the RPN and we want to select regions of interest (ROIs) for the
              classifier part. For each point and each base anchor, we apply the results of the regression. Then
              we crop the resulting ROIs in order to stay in the limit of the image. Then we delete the unsuitable
              ROIs (ie. invalid) and finally we apply a Non Max Suppression (NMS) algorithm to remove the ROIs
              which overlap too much.

    Note : We work with float coordinates. It is no big deal, we will recast them to int to display them.

    Args:
        model (ModelKerasFasterRcnnObjectDetector): Model used (contains all the necessary configs)
        img_data_batch (list<dict>): List of img_data of the batch
            Here, it is used to get the (preprocessed) size of the images in order to remove the ROIs which
            are outside the image.
            Each entry must contain 'resized_height' & 'resized_width'
        rpn_predictions_cls (np.ndarray): Classification prediction (output RPN)
            # shape: (batch_size, height_feature_map, width_feature_map, nb_anchor)
        rpn_predictions_regr (np.ndarray): Regression prediction (output RPN)
            # shape: (batch_size, height_feature_map, width_feature_map, 4 * nb_anchor)
    Returns:
        list<np.ndarray> : Final ROIs list selected for the classifier part (coordinates in features map space)
            Each element is a numpy array of the ROIs coordinates calculated for an image of the batch (variable number).
            The coordinates are returned as int (whereas they were float as output of the RPN)
            # Format x1, y1, x2, y2
            Note : We can't return a numpy array because there are not the same number of ROIs for each image,
                   thus, we return a list
    '''

    # Get model attributes and info from the input shapes
    rpn_regr_scaling = model.rpn_regr_scaling
    subsampling_ratio = model.shared_model_subsampling
    base_anchors = model.list_anchors
    nb_anchors = model.nb_anchors
    roi_nms_overlap_threshold = model.roi_nms_overlap_threshold
    nms_max_boxes = model.nms_max_boxes
    batch_size, height_feature_map, width_feature_map, _ = rpn_predictions_cls.shape

    # First we unscale the regression prediction (we scaled the target of the RPN)
    rpn_predictions_regr = rpn_predictions_regr / rpn_regr_scaling
    # We get the base anchor base in features map space
    base_anchors_feature_maps = [(size[0] / subsampling_ratio, size[1] / subsampling_ratio) for size in base_anchors]

    # First we get all the possible anchor boxes on the features map
    # ie., for each point and each base anchor, we get the coordinates of the anchor box centered on the point
    # TODO : check if the + 0.5 are necessary
    anchor_on_feature_maps = np.array([
        [
            [
                [(x + 0.5 - width_anchor / 2, y + 0.5 - height_anchor / 2, height_anchor, width_anchor) for height_anchor, width_anchor in base_anchors_feature_maps]
                for x in range(width_feature_map)
            ]
            for y in range(height_feature_map)
        ]
        for i in range(batch_size)
    ])  # Format (batch_size, height, width, nb_anchors, nb_coords (format xyhw -> 4))
    # Then we apply the regression result to these anchors
    # First we put together the coordinates of the anchor box and the regression next to each other for each point /anchor box /image
    # Format (batch_size, height, width, nb_anchors, 8 (x_anc, y_anc, h_anc, w_anc, tx, ty, th, tw))
    # Note : first we reshape the regression where the results of each anchors were concatenated
    rpn_predictions_regr = rpn_predictions_regr.reshape((batch_size, height_feature_map, width_feature_map, nb_anchors, 4))
    concatenation_anchor_regr = np.concatenate([anchor_on_feature_maps, rpn_predictions_regr], axis=4)
    # Then we apply the regression for each entry and obtain the candidate ROIs
    # Format (batch_size, height, width, nb_anchors, 4 (x_roi, y_roi, h_roi, w_roi))
    rois_on_feature_maps = np.apply_along_axis(func1d=apply_regression, axis=4, arr=concatenation_anchor_regr)  # Format x, y, h, w
    # Then we crop the ROIs to stay inside the image
    # Problem : in a batch, we padded the images so that they all have the same size,
    #           and we want to crop the ROIs with respect to the initial size (unpadded)
    # Solution : We apply same trick as before, ie. we put the limit size of each image after the coordinates of the associated ROIs
    feature_map_sizes = np.array([get_feature_map_size(img_data['resized_height'], img_data['resized_width'], subsampling_ratio)
                                  for img_data in img_data_batch])
    array_img_size = np.broadcast_to(feature_map_sizes, (height_feature_map, width_feature_map, nb_anchors, batch_size, 2))
    array_img_size = np.transpose(array_img_size, (3, 0, 1, 2, 4))  # Format (batch_size, height, width, nb_anchors, 2)
    rois_on_feature_maps = np.concatenate([rois_on_feature_maps, array_img_size], axis=4)  # We add the sizes to the coordinates
    # We do some work on ROI coordinates
    # Format (batch_size, height, width, nb_anchors, 4 (x_roi, y_roi, h_roi, w_roi))
    rois_on_feature_maps = np.apply_along_axis(func1d=restrict_and_convert_roi_boxes, axis=4, arr=rois_on_feature_maps)   # Format x1, y1, x2, y2

    # Reshape the ROIs in order to have (batch_size, nb_rois, 4), nb_rois = height_feature_map * width_feature_map * nb_anchors
    rois_on_feature_maps = np.reshape(rois_on_feature_maps.transpose((0, 4, 1, 2, 3)), (batch_size, 4, -1)).transpose((0, 2, 1))
    # Same thing with RPN probabilities, ie. shape (batch_size, nb_rois)
    rois_probas = rpn_predictions_cls.reshape((batch_size, -1))

    # Finally we select the final ROIs by deleting the invalid ones and by limiting the overlaps
    # We cast the coordinateds to int in order to use them when cutting the ROIs
    # TODO : Could we try to always get 300? --> Shape consistente
    rois_on_feature_maps = select_final_rois(rois_on_feature_maps, rois_probas, roi_nms_overlap_threshold,
                                             nms_max_boxes, feature_map_sizes)
    return rois_on_feature_maps


def restrict_and_convert_roi_boxes(bbox_coordinates: np.ndarray) -> Tuple[float, float, float, float]:
    '''Resizes the box to have the minimal size and crops it to stay in the features map. Finally,
    converts it in xyxy coordinates.

    Args:
        bbox_coordinates (np.ndarray): An array composed of 6 objects : x_roi, y_roi, h_roi, w_roi, height_img_in_feature_map, width_img_in_feature_map.
            (x_roi, y_roi, h_roi, w_roi) are the coordinates of a ROI
            (height_img_in_feature_map, width_img_in_feature_map) sont les tailles avant padding de l'image correspondantes, puis downsampled au format feature map
    Returns:
        float: Coordinates of the ROI after correction - x coordinate of the upper left point
        float: Coordinates of the ROI after correction - y coordinate of the upper left point
        float: Coordinates of the ROI after correction - x coordinate of the bottom right point
        float: Coordinates of the ROI after correction - y coordinate of the bottom right point
    '''
    x, y, h, w, height_img_in_feature_map, width_img_in_feature_map = bbox_coordinates
    # We want the box to have a size of at least 1
    h = np.maximum(1, h)
    w = np.maximum(1, w)
    # We want the upper left point to be in the image (projected on the features map)
    x = np.maximum(0, x)
    y = np.maximum(0, y)
    # Convert in xyxy (opposite points format)
    x1, y1, x2, y2 = xyhw_to_xyxy(x, y, h, w)
    # We want the bottom right point to be in the image (projected on the features map)
    x2 = np.minimum(width_img_in_feature_map, x2)
    y2 = np.minimum(height_img_in_feature_map, y2)
    # Return new coordinates
    return x1, y1, x2, y2


def select_final_rois(rois_coordinates: np.ndarray, rois_probas: np.ndarray, roi_nms_overlap_threshold: float,
                      nms_max_boxes: int, feature_map_sizes: np.ndarray) -> List[np.ndarray]:
    '''Deletes the invalid ROIs and selects some of them to limit the overlaps

    Args:
        rois_coordinates (np.ndarray): The set of all selected ROIs for all the images of the batch
            # shape: (batch_size, nb_rois, 4)
        rois_probas (np.ndarray): The probabilities associated to each selected ROIsfor all the images
            # shape: (batch_size, nb_rois)
        roi_nms_overlap_threshold (float): Above this threshold for the iou, we assume that two ROIs overlap
        nms_max_boxes (int): Maximal number of ROIs that this function can return for each image
        feature_map_sizes (np.ndarray): Theoretical heights and widths of the features maps - useful if we have no valid ROI anymore.
            Allows to manage the fact that, in a batch, we padded the images so that they all have the same size
                # shape: (batch_size, 2)
    Returns:
        list<np.ndarray>: Final list of the ROIs selected for the classifier part
    '''
    # We process each image of the batch separately and stocks the results in list_rois
    # We can't return a numpy array because the number of ROIs for each image is not the same -> we return a list
    list_rois = []
    for img_index in range(rois_coordinates.shape[0]):
        # Get infos
        img_rois_coordinates = rois_coordinates[img_index]
        img_rois_probas = rois_probas[img_index]
        x1, y1, x2, y2 = (img_rois_coordinates[:, i] for i in range(4))
        # Eliminate invalid anchors
        idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
        img_rois_coordinates = np.delete(img_rois_coordinates, idxs, 0)
        img_rois_probas = np.delete(img_rois_probas, idxs, 0)
        # In the rare cases where there are no more ROIs, we create one artificially (the whole image)
        if img_rois_coordinates.shape[0] == 0:
            logger.warning("Warning, there is an image for which we can't find a valid ROI.")
            logger.warning("By default, we create an artificial ROI which cover the whole image.")
            height_img_in_feature_map, width_img_in_feature_map = feature_map_sizes[img_index, :]
            img_rois_coordinates = np.array([[0, 0, width_img_in_feature_map, height_img_in_feature_map]])  # x1, y1, x2, y2
        # Otherwise, we continue the process
        else:
            # We keep the ROIs which do not overlap
            img_rois_coordinates, _, _ = non_max_suppression_fast(img_rois_coordinates, img_rois_probas, roi_nms_overlap_threshold, nms_max_boxes)
            # Finally, we cast to int (because we will cut the features map per index)
            # Round
            img_rois_coordinates = np.around(img_rois_coordinates).astype("int")
            # Delete invalid ROIs again (after rounding)
            x1, y1, x2, y2 = (img_rois_coordinates[:, i] for i in range(4))
            idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
            img_rois_coordinates = np.delete(img_rois_coordinates, idxs, 0)
        # Append result
        list_rois.append(img_rois_coordinates)
    # Return ROIs
    return list_rois


######
# Get the target of the classifier for an image and its ROIs (RPN prediction)
# Main function : get_classifier_train_inputs_and_targets
######


def get_classifier_train_inputs_and_targets(model, img_data_batch: List[dict],
                                            rois_coordinates: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Gives the regression and classification of the classifier from the ROIs given by the RPN

    Process : We got the ROIs from the RPN prediction (and transformed them via get_roi_from_rpn_predictions)
              For each image we will :
              - Calculate the ious between bboxes and ROIs
              - Keep, for each ROI, the bbox with the biggest iou (if the iou is bigger than a threshold)
              - Filter the ROIs to only some of them:
                  - Allows to keep OOM in check
                  - We respect to the loss, we will, of course, only take into account the selected ROIs
                  - We will keep a balance between positive ROIs (match with an object) and negative ROIs (match with 'bg')
              - Format the inputs and targets of the model

    Args:
        model (ModelKerasFasterRcnnObjectDetector): Model used (contains all the necessary configs)
        img_data_batch (list<dict>): List of img_data for the batch
            Used here to get the bboxes of the images to define the targets of the classifier
        rois_coordinates (list<np.ndarray>): Final list of the ROIs selected for the classifier part.
            Each element is a numpy array containing the coordinates of the ROIs calculated for an image of the batch
            # Format x1, y1, x2, y2 (opposite points)
    Returns:
        np.ndarray : ROIs coordinates in input of the model - Format x, y, h, w
            # Shape : (batch_size, nb_rois_per_img, 4), format x, y, h, w
        np.ndarray : Targets of the classifier - classification
            # Shape (batch_size, nb_rois_per_img, (nb_classes + 1))
        np.ndarray : Targets of the classifier - regression
            # Shape (batch_size, nb_rois_per_img, 2 * nb_classes * 4)
    '''

    # Get model attributes
    subsampling_ratio = model.shared_model_subsampling
    classifier_min_overlap = model.classifier_min_overlap
    classifier_max_overlap = model.classifier_max_overlap
    nb_rois_per_img = model.nb_rois_classifier
    classifier_regr_scaling = model.classifier_regr_scaling
    dict_classes = model.dict_classes

    # Init. of output arrays
    X, Y1, Y2 = None, None, None

    # Preprocess one image at a time
    for img_data, rois in zip(img_data_batch, rois_coordinates):
        # Get all the ious betwee, ROIs and bboxes
        dict_rois = get_rois_bboxes_iou(rois, img_data, subsampling_ratio)
        # Find the best bbox for each ROI et the corresponding regression
        dict_rois_targets = get_rois_targets(dict_rois, classifier_min_overlap, classifier_max_overlap)
        # Limit the number of ROI
        dict_rois_targets = limit_rois_targets(dict_rois_targets, nb_rois_per_img)
        # If we have no more targets (very rare !), we consider the entire image as 'bg'
        if dict_rois_targets is None:
            logger.warning("There is an image with no suitable target for the classifier. We consider the entire image as background.")
            dict_rois_targets = create_fake_dict_rois_targets(img_data, subsampling_ratio, nb_rois_per_img)
        # Format the classification and regression targets
        X_tmp, Y1_tmp, Y2_tmp = format_classifier_inputs_and_targets(dict_rois_targets, dict_classes, classifier_regr_scaling)
        # Increment output
        if X is None:
            X, Y1, Y2 = X_tmp, Y1_tmp, Y2_tmp
        else:
            # TODO : get rid of the concatenate in the loop and concatenate a list one time at the end instead (much faster)
            X = np.concatenate((X, X_tmp), axis=0)
            Y1 = np.concatenate((Y1, Y1_tmp), axis=0)
            Y2 = np.concatenate((Y2, Y2_tmp), axis=0)
    # We return the result
    return X, Y1, Y2


def get_rois_bboxes_iou(rois: np.ndarray, img_data: dict, subsampling_ratio: int) -> dict:
    '''Gives the ious between the ROIs (in rois) and the bboxes (in img_data).

    Args:
        rois (np.ndarray): ROIs given by the RPN (ie. by the function get_roi_from_rpn_predictions())
            # Shape (N, 4), N corresponds to the number of given ROIs (in general max 300, cf. model.nms_max_boxes)
        img_data (dict): Metadata of the image after the preprocessing. In particular, the bboxes have been resized
            and rotated if the image has been resized and rotated. We only use the 'bboxes' field
        subsampling_ratio (int): Subsampling of the base model (shared layers) - to apply to bboxes (which are in image space)
    Returns:
        dict: Dictionary containing all the IOUs betwee, ROIs and bboxes of the image
            Keys : (index_roi) -> 'coordinates' -> 'x1', 'y1', 'x2', 'y2', 'h', 'w'
                               -> (index_bbox)  -> 'coordinates': 'x1', 'y1', 'x2', 'y2'
                                                -> 'iou'
                                                -> 'class'
    '''
    # Init. output dictionary
    dict_rois = {}
    # For each ROI, we get its coordinates, and the ious with each bbox of the image
    for index_roi in range(rois.shape[0]):
        # Get the coordinates of each ROI
        x1_roi, y1_roi, x2_roi, y2_roi = rois[index_roi, :]
        _, _, h_roi, w_roi = xyxy_to_xyhw(*rois[index_roi, :])
        dict_roi = {
            'coordinates': {'x1': x1_roi, 'y1': y1_roi, 'x2': x2_roi, 'y2': y2_roi, 'h': h_roi, 'w': w_roi},
            'bboxes': {}
        }
        # Get the iou of each bbox
        for index_bbox, bbox in enumerate(img_data['bboxes']):
            # bbox coordinates - input image format
            # Bbox coordinates (input format, ie. image space)
            bbox_coordinates = (bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'])
            # Coordinates transformation to features map space
            x1_bbox, y1_bbox, x2_bbox, y2_bbox = (coord / subsampling_ratio for coord in bbox_coordinates)
            # Calculus iou
            iou = get_iou((x1_bbox, y1_bbox, x2_bbox, y2_bbox), (x1_roi, y1_roi, x2_roi, y2_roi))
            dict_roi['bboxes'][index_bbox] = {
                'coordinates': {'x1': x1_bbox, 'y1': y1_bbox, 'x2': x2_bbox, 'y2': y2_bbox},
                'iou': iou,
                'class': bbox['class']
            }
        # Append results
        dict_rois[index_roi] = dict_roi
    # Returns
    return dict_rois


def get_rois_targets(dict_rois: dict, classifier_min_overlap: float, classifier_max_overlap: float) -> dict:
    '''Finds the bbox with the biggest iou with an ROI and associate them. Then associates the class
    of this bbox to the ROI and, if the iou is sufficiently big, gives the associated regression.

    Args:
        dict_rois (dict): Dictionary containing all the ious between the ROIs and the bboxes of the image
        classifier_min_overlap (float): Minimal threshold to consider a ROI as a target of the classifier (which can still be 'bg')
        classifier_max_overlap (float): Minimal threshold to consider a ROI as matching with a bbox (so with a class which is not 'bg')
    Returns:
        dict: Dictionary containing the 'viable' ROIs and their classification and regression targets
    '''
    dict_rois_targets = {}
    # For each ROI ...
    for roi_index, dict_roi in dict_rois.items():
        # ... get the coordinates ...
        coords_roi = dict_roi['coordinates']
        x1_roi, y1_roi, x2_roi, y2_roi = (coords_roi['x1'], coords_roi['y1'], coords_roi['x2'], coords_roi['y2'])
        # ... get the best associated bbox (highest iou) ...
        dict_iou = {bbox_index: dict_roi['bboxes'][bbox_index]['iou'] for bbox_index in dict_roi['bboxes']}
        best_bbox_index = max(dict_iou, key=dict_iou.get)
        best_iou = dict_iou[best_bbox_index]
        # ... if best_iou lower than a threshold, we ignore this ROI ...
        if best_iou < classifier_min_overlap:
            continue
        # ... otherwise, we define the best bbox and we complete the targets
        dict_roi['best_bbox_index'] = best_bbox_index
        dict_roi['best_iou'] = best_iou
        dict_roi['classifier_regression_target'] = (0, 0, 0, 0)
        # ... if best_iou is above a threshold, we consider a match on a class,
        # and get the regression target ...
        if best_iou >= classifier_max_overlap:
            # class
            dict_roi['classifier_class_target'] = dict_roi['bboxes'][best_bbox_index]['class']
            # regression
            coords_bbox = dict_roi['bboxes'][best_bbox_index]['coordinates']
            x1_bbox, y1_bbox, x2_bbox, y2_bbox = coords_bbox['x1'], coords_bbox['y1'], coords_bbox['x2'], coords_bbox['y2']
            dict_roi['classifier_regression_target'] = calc_regr((x1_bbox, y1_bbox, x2_bbox, y2_bbox), (x1_roi, y1_roi, x2_roi, y2_roi))
        # ... otherwise, we consider the ROI to be background (ie. 'bg')
        # Note : the regression target is not calculated if background
        else:
            dict_roi['classifier_class_target'] = 'bg'
        # Append results
        dict_rois_targets[roi_index] = dict_roi
    # Returns
    return dict_rois_targets


def limit_rois_targets(dict_rois_targets: dict, nb_rois_per_img: int) -> Union[dict, None]:
    '''Limits the number of input / output for each image in order not to have OOM

    Args:
        dict_rois (dict): Dictionary containing the possible inputs / targets of the classifier
        nb_rois_per_img (int): Maximal number of ROIs to return for each image
            In the rare case where there are not enough ROIs, we clone the ROIs in order to have enough
            If no ROI, we return None. This case is then handled by the function create_fake_dict_rois_targets
    Returns:
        dict: The dictionary containing the "selected" dictionary
    '''
    # Get the positive and negative ROIs
    pos_rois_indexes = [roi_index for roi_index, roi in dict_rois_targets.items() if roi['classifier_class_target'] != 'bg']
    neg_rois_indexes = [roi_index for roi_index, roi in dict_rois_targets.items() if roi['classifier_class_target'] == 'bg']
    # Case 1 : no ROI (very rare ?!), return None
    if len(pos_rois_indexes) + len(neg_rois_indexes) == 0:
        logger.warning("Warning, there is an image for which we do not have a target ROI for the classifier.")
        return None
    # Case 2 : not enough ROIs
    elif len(pos_rois_indexes) + len(neg_rois_indexes) < nb_rois_per_img:
        logger.warning(f"Warning, there is an image for which we have less than {nb_rois_per_img} target ROIs. We randomly clone some ROIs")
        selected_indexes = pos_rois_indexes + neg_rois_indexes
        selected_indexes = random.sample(selected_indexes, k=len(selected_indexes))
        selected_indexes = list(np.resize(selected_indexes, nb_rois_per_img))
    # Case 3 : enough ROIs
    else:
        # Case 3.1 : not enough positive ROIs
        if len(pos_rois_indexes) < nb_rois_per_img // 2:
            selected_neg_indexes = random.sample(neg_rois_indexes, k=(nb_rois_per_img - len(pos_rois_indexes)))
            selected_indexes = pos_rois_indexes + selected_neg_indexes
            selected_indexes = random.sample(selected_indexes, k=len(selected_indexes))
        # Cas 3.2 : not enough negative ROIs
        elif len(neg_rois_indexes) < nb_rois_per_img // 2:
            selected_pos_indexes = random.sample(pos_rois_indexes, k=(nb_rois_per_img - len(neg_rois_indexes)))
            selected_indexes = selected_pos_indexes + neg_rois_indexes
            selected_indexes = random.sample(selected_indexes, k=len(selected_indexes))
        # Cas 3.3 : nominal case, we have everything we need
        else:
            selected_pos_indexes = random.sample(pos_rois_indexes, k=nb_rois_per_img // 2)
            selected_neg_indexes = random.sample(neg_rois_indexes, k=(nb_rois_per_img - len(selected_pos_indexes)))
            selected_indexes = selected_pos_indexes + selected_neg_indexes
            selected_indexes = random.sample(selected_indexes, k=len(selected_indexes))
    # We return the ROIs whose index are in selected_indexes
    # We are careful to manage "duplicates" in the list of selected indices
    return {i: copy.deepcopy(dict_rois_targets[roi_index]) for i, roi_index in enumerate(selected_indexes)}


def create_fake_dict_rois_targets(img_data: dict, subsampling_ratio: int, nb_rois_per_img: int) -> dict:
    '''Creates fake dict_rois_targets in the rare cases where the function limit_rois_targets gives an empty object (None).

        Process : we return ROIs on the entire image, considered as background

    Args:
        img_data (dict): Metadata of the image after the preprocessing. In particular, the size of the image
        subsampling_ratio (int): Subsampling of the base model (shared layers)
        nb_rois_per_img (int): Number of fake ROIs to return
    Returns:
        dict: The dictionary of fake "selected" ROIs
    '''
    # Get the size of the image in the features map
    height_img_in_feature_map, width_img_in_feature_map = get_feature_map_size(img_data['resized_height'], img_data['resized_width'], subsampling_ratio)
    # Create a dictionary with an unique entry : a ROI on the entire image, considered as background
    dict_rois_targets = {
        0: {
            'coordinates': {'x1': 0, 'y1': 0, 'x2': width_img_in_feature_map, 'y2': height_img_in_feature_map,
                            'h': height_img_in_feature_map, 'w': width_img_in_feature_map},
            'classifier_regression_target': (0, 0, 0, 0),
            'classifier_class_target': 'bg',
        }
    }
    # We clone this ROI to have as many as wanted
    dict_rois_targets = {i: copy.deepcopy(dict_rois_targets[0]) for i in range(nb_rois_per_img)}
    # Return
    return dict_rois_targets


def format_classifier_inputs_and_targets(dict_rois_targets: dict, dict_classes: dict,
                                         classifier_regr_scaling: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Transforms a dictionary of target ROIs into a suitable format for the classifier model

    Args:
        dict_rois (dict): Dictionary containing the possible inputs / targets of the classifier
        dict_classes (dict): Mapping of the classes of the model (must not contain 'bg'), format :  {idx: label}
        classifier_regr_scaling (list<float>): Regression coefficient to apply to coordinates
    Returns:
        np.ndarray: coordinates of each selected ROIs
            # Shape : (1, nb_rois, 4), format x, y, h, w
        np.ndarray: Classification target of the classifier (with the background)
            # Shape (1, nb_rois, (nb_classes + 1))
        np.ndarray: Two parts array:
            # Shape (1, nb_rois, 2 * nb_classes * 4)
            -> First half : identification class ground truth to calculate the regression loss for the classifier
                # Shape (1, nb_rois, nb_classes * 4)
            -> Second hald : regression target for the classifier (one regression per class)
                # Shape (1, nb_rois, nb_classes * 4)
    '''

    # Get the number of selected ROIs (the same for each image) and info on classes
    nb_rois = len(dict_rois_targets)
    nb_classes = len(dict_classes)
    class_mapping = {name_class: index for index, name_class in dict_classes.items()}
    class_mapping['bg'] = len(class_mapping)  # Add background to the mapping

    # Init. of output arrays
    X = np.zeros((nb_rois, 4))  # ROIs coordinates
    Y1 = np.zeros((nb_rois, (nb_classes + 1)))  # + 1 for background
    Y2_1 = np.zeros((nb_rois, 4 * nb_classes))  # OHE class (without background), but repeated 4 times (one for each coordinate), will be used by the loss
    Y2_2 = np.zeros((nb_rois, 4 * nb_classes))  # One regression per class (# TODO verify if we can't do only one regression)

    # For each ROI, we fill up the output arrays
    for i, (roi_index, roi) in enumerate(dict_rois_targets.items()):
        ### ROI coordinates
        X[i, :] = (roi['coordinates']['x1'], roi['coordinates']['y1'], roi['coordinates']['h'], roi['coordinates']['w'])  # Format x, y, h, w

        ### Targets of the classifier model
        # ROI class
        gt_class = roi['classifier_class_target']
        idx_gt_class = class_mapping[gt_class]
        ohe_target = [0] * (nb_classes + 1)
        ohe_target[idx_gt_class] = 1  # --> e.g. [0, 1, 0] / 2 classes + 'bg'
        Y1[i, :] = ohe_target

        # ROI regression - loss target
        ohe_target_no_bg = ohe_target[:nb_classes]  # We get rid of the background, no regression here --> e.g. [0, 1]
        ohe_target_no_bg_repeated = np.repeat(ohe_target_no_bg, 4)  # We repeat the OHE targets four times 4 fois (one for each coordinate)
        Y2_1[i, :] = ohe_target_no_bg_repeated  # e.g. [0, 0, 0, 0, 1, 1, 1, 1]

        # ROI regression - regression - only if not background
        if gt_class != 'bg':
            target_regression = [0.] * nb_classes * 4
            regression_values = [a * b for a, b in zip(roi['classifier_regression_target'], classifier_regr_scaling)]  # Apply a scaling
            target_regression[idx_gt_class * 4: (idx_gt_class + 1) * 4] = regression_values  # e.g. [0, 0, 0, 0, 0.2, -0.3, 0.1, 0.9]
            Y2_2[i, :] = target_regression

    # Concatenate Y2
    Y2 = np.concatenate([Y2_1, Y2_2], axis=1)
    # Add batch dimension
    X = np.expand_dims(X, axis=0)
    Y1 = np.expand_dims(Y1, axis=0)
    Y2 = np.expand_dims(Y2, axis=0)
    # Returns
    return X, Y1, Y2


######
# Utils function for the prediction of a Faster RCNN model
######


def get_classifier_test_inputs(rois_coordinates: List[np.ndarray]) -> np.ndarray:
    '''Formats the inputs for the classifier from ROIs proposed by the RPN (test case)

    Process : For each ROI, we simply get the format x, y, h, w

    Args:
        rois_coordinates (list<np.ndarray>): ROIs to transform
            rois_coordinates must be a list with only one entry : the ROIs of the current image (for prediction, the batch_size is forced to 1)
            The unique entry is a numpy array:
                # Shape (nb_rois, 4)
                # Format x1, y1, x2, y2
    Raises:
        ValueError: If the number of elements in the list is different from 1
    Returns:
        np.ndarray : ROIs to use as inputs of the model
            # Shape : (1, nb_rois, 4), format x, y, h, w
    '''
    if len(rois_coordinates) != 1:
        raise ValueError("In prediction mode, the batch_size must be 1.")
    # Init. of the output array
    nb_rois = rois_coordinates[0].shape[0]
    output_shape = (1, nb_rois, 4)
    X = np.zeros(output_shape)
    # We process ROIs one at a time
    for i, roi in enumerate(rois_coordinates[0]):
        # Get format x, y, h, w
        x1, y1, x2, y2 = roi[0], roi[1], roi[2], roi[3]
        x1, y1, h, w = xyxy_to_xyhw(x1, y1, x2, y2)
        # Update output array
        X[0, i, :] = (x1, y1, h, w)
    # Return result
    return X


def get_valid_fm_boxes_from_proba(probas: np.ndarray, proba_threshold: float, bg_index: int) -> List[tuple]:
    '''Keeps predicted (in features map space) boxes whose probability is above a threshold. Also deletes
    all the boxes which matched on background

    Args:
        probas (np.ndarray): Probabilities of the boxes predicted by the model
        proba_threshold (float): Threshold below which, boxes are eliminated
    Returns:
        A list of boxes (in features map space) valid from a probability point of view
            # Format [(index, index_cl, proba), (...), ...)
    '''
    fm_boxes_candidates = []
    # For each box ...
    for index_box, box_probas in enumerate(probas):
        # ... get the class ...
        predicted_class = np.argmax(box_probas)
        # ... get the corresponding probability ...
        predicted_proba = box_probas[predicted_class]
        # ..., and, if we are above the threshold and the predicted class is not the background...
        if predicted_proba >= proba_threshold and predicted_class != bg_index:
            # ... we add the box to the list
            fm_boxes_candidates.append((index_box, predicted_class, predicted_proba))
    return fm_boxes_candidates


def get_valid_boxes_from_coordinates(input_img: np.ndarray, input_rois: np.ndarray, fm_boxes_candidates: List[tuple],
                                     regr_coordinates: np.ndarray, classifier_regr_scaling: List[float], subsampling_ratio: int,
                                     dict_classes: dict) -> List[tuple]:
    '''Calculates the coordinates (in image space) after application of the regression of the boxes (in features map space) whose
    probability is sufficiently high. Then restricts them to the image and keeps only the valid boxes

    Args:
        input_img (np.ndarray): Resized image (useful to get the dimensions)
        input_rois (np.ndarray): ROIs given by the RPN
        fm_boxes_candidates (list): The boxes (in features map space) valid with respect to their proba
        regr_coordinates (np.ndarray): Regression prediction for the boxes
        classifier_regr_scaling (list): Scaling to remove from the regression results
        subsampling_ratio (int): Subsampling of the base model (shared layers) - to apply to bboxes (which are in image space)
        dict_classes (dict): Dictionary of the classes of the model
    Returns:
        A list of boxes valid from a probability AND coordinates xyxy points of view
            # Format [(cl, proba, coordinates), (...), ...)
    '''
    boxes_candidates = []
    # For each box (in features map space)...
    for index_box, predicted_class, predicted_proba in fm_boxes_candidates:
        roi_coordinates = input_rois[index_box]  # Get corresponding ROI
        regr_predicted = regr_coordinates[index_box][predicted_class * 4: (predicted_class + 1) * 4]  # Get the regression associated to this class
        regr_predicted = np.array([a / b for a, b in zip(regr_predicted, classifier_regr_scaling)])  # Remove the scaling
        # Apply predicted regression
        coordinates_after_regr = list(apply_regression(np.concatenate([roi_coordinates, regr_predicted])))
        # Make sure that the upper left point is in the features map
        coordinates_after_regr[0] = max(0, coordinates_after_regr[0])
        coordinates_after_regr[1] = max(0, coordinates_after_regr[1])
        bbox_fm_coords = xyhw_to_xyxy(*coordinates_after_regr)
        # Get the coordinates in the input format (ie. in image space)
        x1_bbox, y1_bbox, x2_bbox, y2_bbox = (coord * subsampling_ratio for coord in bbox_fm_coords)
        # Make sure that the point defining the box are in the image
        x1_bbox = max(0, x1_bbox)
        y1_bbox = max(0, y1_bbox)
        x2_bbox = min(input_img.shape[1], x2_bbox)
        y2_bbox = min(input_img.shape[0], y2_bbox)
        # If the box is valid ...
        if x1_bbox < x2_bbox and y1_bbox < y2_bbox:
            # ... we add it to the list
            box_infos = (dict_classes[predicted_class], predicted_proba, (x1_bbox, y1_bbox, x2_bbox, y2_bbox))
            boxes_candidates.append(box_infos)
    return boxes_candidates


def non_max_suppression_fast_on_preds(boxes_candidates: List[tuple], nms_overlap_threshold: float) -> List[tuple]:
    '''Applies the NMS algorithm on the valid predicted boxes to avoid overlaps.

    Args:
        boxes_candidates (list): Valid predicted boxes
            # Format [(cl, proba, coordinates), (...), ...)
        nms_overlap_threshold (float): Above this threshold for the iou, two boxes are said to be overlapping
    Returns:
        A list of boxes valid from a probability AND coordinates xyxy points of view and with "no" overlap
            # Format [(cl, proba, coordinates), (...), ...)
    '''
    # If there are no valid boxes
    if len(boxes_candidates) == 0:
        return []
    # First we format the inputs to the format for the NMS
    img_boxes_classes = np.array([cl for cl, _, _ in boxes_candidates])
    img_boxes_probas = np.array([proba for _, proba, _ in boxes_candidates])
    img_boxes_coordinates = np.array([coordinates for _, _, coordinates in boxes_candidates])
    # Apply NMS
    nms_result = non_max_suppression_fast(img_boxes_coordinates, img_boxes_probas, nms_overlap_threshold, np.inf, img_boxes_classes=img_boxes_classes)
    img_boxes_coordinates, img_boxes_probas, img_boxes_classes = nms_result
    # Format final result
    final_boxes = []
    for i in range(img_boxes_coordinates.shape[0]):
        final_boxes.append((img_boxes_classes[i], img_boxes_probas[i], img_boxes_coordinates[i]))
    # Return
    return final_boxes


def get_final_bboxes(final_boxes: List[tuple], img_data: dict) -> List[dict]:
    '''Resizes the final predicted boxes to image space and formats them.

    Args:
        final_boxes (list) : list of boxes valid from a probability AND coordinates xyxy points of view and with "no" overlap
            # Format [(cl, proba, coordinates), (...), ...)
        img_data (dict) : Metadata associated with the image (used to resize predictions)
    Returns:
        A list of bboxes corresponding to the model predictions
    '''
    list_bboxes = []
    resized_width = img_data['resized_width']
    original_width = img_data['original_width']
    resized_height = img_data['resized_height']
    original_height = img_data['original_height']
    for cl, proba, coordinates in final_boxes:
        bbox = {'class': cl, 'proba': proba, 'x1': coordinates[0],
                'y1': coordinates[1], 'x2': coordinates[2], 'y2': coordinates[3]}
        bbox['x1'] = int(bbox['x1'] * (original_width / resized_width))
        bbox['x2'] = int(bbox['x2'] * (original_width / resized_width))
        bbox['y1'] = int(bbox['y1'] * (original_height / resized_height))
        bbox['y2'] = int(bbox['y2'] * (original_height / resized_height))
        list_bboxes.append(copy.deepcopy(bbox))
    return list_bboxes


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
