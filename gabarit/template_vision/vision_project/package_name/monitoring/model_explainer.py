#!/usr/bin/env python3

## Classes to explain models predictions
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
# - Explainer -> Parent class for the explainers
# - LimeExplainer -> Lime Explainer wrapper class

import os
import logging
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
from typing import Type, Union, Any
from lime.lime_image import LimeImageExplainer

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training.model_class import ModelClass


class Explainer:
    '''Parent class for the explainers'''

    def __init__(self) -> None:
        '''Initialization of the parent class'''
        self.logger = logging.getLogger(__name__)

    def explain_instance(self, text: str, **kwargs) -> Any:
        '''Explains a prediction

        Args:
            text (str): Text to be explained
        Returns:
            (?): An explanation object
        '''
        raise NotImplementedError("'explain_instance' needs to be overridden")

    def explain_instance_as_html(self, text: str, **kwargs) -> str:
        '''Explains a prediction - returns an HTML object

        Args:
            text (str): Text to be explained
        Returns:
            str: An HTML code with the explanation
        '''
        raise NotImplementedError("'explain_instance_as_html' needs to be overridden")

    def explain_instance_as_list(self, text: str, **kwargs) -> list:
        '''Explains a prediction - returns a list object

        Args:
            text (str): Text to be explained
        Returns:
            list: List of tuples with words and corresponding weights
        '''
        raise NotImplementedError("'explain_instance_as_list' needs to be overridden")


class LimeExplainer(Explainer):
    '''Lime Explainer wrapper class'''

    def __init__(self, model: Type[ModelClass], model_conf: dict) -> None:
        ''' Initialization

        Args:
            model: A model instance with predict & predict_proba functions, and list_classes attribute
            model_conf (dict): The model's configuration
        Raises:
            ValueError: If the provided model is not a classifier
            TypeError: If the provided model does not implement a `predict` function
            TypeError: If the provided model does not implement a `predict_proba` function
            TypeError: If the provided model does not have a `list_classes` attribute
        '''
        super().__init__()
        pred_op = getattr(model, "predict", None)
        pred_proba_op = getattr(model, "predict_proba", None)

        # Check needed methods
        if pred_op is None or not callable(pred_op):
            raise TypeError("The supplied model must implement a predict() function")
        if pred_proba_op is None or not callable(pred_proba_op):
            raise TypeError("The supplied model must implement a predict_proba() function")
        if getattr(model, "list_classes", None) is None:
            raise TypeError("The supplied model must have a list_classes attribute")
        # Check classifier
        if not model.model_type == 'classifier':
            raise ValueError("LimeExplainer only supported with classifier models")

        self.model = model
        self.model_conf = model_conf
        self.class_names = self.model.list_classes
        self.explainer = LimeImageExplainer()

    def explain_instance(self, img: Image.Image, classes: Union[list, None] = None, num_samples: int = 100,
                         batch_size: int = 100, hide_color=0, top_labels: int = 3, **kwargs):
        '''Explains a prediction

        This function calls the Lime module. It generates neighborhood data by randomly perturbing features from the instance.
        Then, it learns locally weighted linear models on this neighborhood data to explain each of the classes in an interpretable way.

        Args:
            img (Image.Image): Image to be explained
        Kwargs:
            classes (list): Classes to be compared - names
            num_samples (int): size of the neighborhood to learn the linear model (cf. Lime documentation)
            batch_size (int): classifier_fn will be called on batches of this size (cf. Lime documentation)
            hide_color (?): TODO
            top_labels (int): Number of labels to consider (sort by proba)
        Returns:
            (?): An explanation object
        '''
        # Define classifier_fn
        def classifier_fn_lime(images: np.ndarray) -> np.ndarray:
            '''Function to be used by Lime, returns probas per classes

            Args:
                images (np.ndarray): array of images
            Returns:
                np.array: probabilities
            '''
            # Preprocess images
            images = [Image.fromarray(img, 'RGB') for img in images]
            if 'preprocess_str' in self.model_conf.keys():
                preprocess_str = self.model_conf['preprocess_str']
            else:
                preprocess_str = "no_preprocess"
            preprocessor = preprocess.get_preprocessor(preprocess_str)
            images_preprocessed = preprocessor(images)
            # Temporary folder
            with tempfile.TemporaryDirectory(dir=utils.get_data_path()) as tmp_folder:
                # Save images
                images_path = [os.path.join(tmp_folder, f'image_{i}.png') for i in range(len(images_preprocessed))]
                for i, img_preprocessed in enumerate(images_preprocessed):
                    img_preprocessed.save(images_path[i], format='PNG')
                # Get predictions
                df = pd.DataFrame({'file_path': images_path})
                predictions, probas = self.model.predict_with_proba(df)
            # Return probas
            return probas

        # Get explanations (images must be convert into rgb, then into np array)
        return self.explainer.explain_instance(np.array(img.convert('RGB')), classifier_fn_lime,
                                               num_samples=num_samples, batch_size=batch_size,
                                               hide_color=hide_color, top_labels=top_labels)

    def explain_instance_as_html(self, *args, **kwargs):
        '''Explains a prediction - returns an HTML object
        ** NOT IMPLEMENTED **
        '''
        raise NotImplementedError("'explain_instance_as_html' is not defined for LimeExplainer")

    def explain_instance_as_list(self, *args, **kwargs):
        '''Explains a prediction - returns a list object
        ** NOT IMPLEMENTED **
        '''
        raise NotImplementedError("'explain_instance_as_list' is not defined for LimeExplainer")


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
