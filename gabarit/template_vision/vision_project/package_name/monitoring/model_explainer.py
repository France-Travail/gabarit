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

from .. import utils
from ..preprocessing import preprocess
from ..models_training.classifiers.model_classifier import ModelClassifierMixin  # type: ignore
# ModelClassifierMixin import must be ignored for mypy as the Mixin class is itself ignored (too complicated to manage)


class Explainer:
    '''Parent class for the explainers'''

    def __init__(self, *args, **kwargs) -> None:
        '''Initialization of the parent class'''
        self.logger = logging.getLogger(__name__)

    def explain_instance(self, content: Image.Image, **kwargs) -> Any:
        '''Explains a prediction

        Args:
            content (Image.Image): Image to be explained
        Returns:
            (?): An explanation object
        '''
        raise NotImplementedError("'explain_instance' needs to be overridden")

    def explain_instance_as_html(self, content: Image.Image, **kwargs) -> str:
        '''Explains a prediction - returns an HTML object

        Args:
            content (Image.Image): Image to be explained
        Returns:
            str: An HTML code with the explanation
        '''
        raise NotImplementedError("'explain_instance_as_html' needs to be overridden")

    def explain_instance_as_json(self, content: Image.Image, **kwargs) -> Union[dict, list]:
        '''Explains a prediction - returns an JSON serializable object

        Args:
            content (str): Text to be explained
        Returns:
            str: A JSON serializable object with the explanation
        '''
        raise NotImplementedError("'explain_instance_as_json' needs to be overridden")

class LimeExplainer(Explainer):
    '''Lime Explainer wrapper class'''

    def __init__(self, model: Type[ModelClassifierMixin], model_conf: dict) -> None:
        ''' Initialization

        Args:
            model: A model instance with predict & predict_proba functions, and list_classes attribute
            model_conf (dict): The model's configuration
        Raises:
            ValueError: If the provided model is not a classifier
            TypeError: If the provided model does not implement a `predict_proba` function
            TypeError: If the provided model does not have a `list_classes` attribute
        '''
        super().__init__()
        pred_proba_op = getattr(model, "predict_proba", None)

        # Check classifier
        if not model.model_type == 'classifier':
            raise ValueError("LimeExplainer only supported with classifier models")
        # Check needed methods
        if pred_proba_op is None or not callable(pred_proba_op):
            raise TypeError("The supplied model must implement a predict_proba() function")
        if getattr(model, "list_classes", None) is None:
            raise TypeError("The supplied model must have a list_classes attribute")

        self.model = model
        self.model_conf = model_conf
        self.class_names = self.model.list_classes
        # Our explainers will explain a prediction for a given class / label
        # These atributes are set on the fly
        self.current_class_index = 0
        # Create the explainer
        self.explainer = LimeImageExplainer()

    def classifier_fn(self, content_arrays: np.ndarray) -> np.ndarray:
        '''Function to get probabilities from a list of (not preprocessed) images

        Args:
            content_arrays (np.ndarray): images to be considered
        Returns:
            np.array: probabilities
        '''
        # Get preprocessor
        if 'preprocess_str' in self.model_conf.keys():
            preprocess_str = self.model_conf['preprocess_str']
        else:
            preprocess_str = "no_preprocess"
        preprocessor = preprocess.get_preprocessor(preprocess_str)
        # Preprocess images
        images = [Image.fromarray(img, 'RGB') for img in content_arrays]
        images_preprocessed = preprocessor(images)
        # Temporary folder
        with tempfile.TemporaryDirectory(dir=utils.get_data_path()) as tmp_folder:
            # Save images
            images_path = [os.path.join(tmp_folder, f'image_{i}.png') for i in range(len(images_preprocessed))]
            for i, img_preprocessed in enumerate(images_preprocessed):
                img_preprocessed.save(images_path[i], format='PNG')
            # Get predictions
            df = pd.DataFrame({'file_path': images_path})
            probas = self.model.predict_proba(df)
        # Return probas
        return probas

    def explain_instance(self, content: Image.Image, class_index: Union[int, None] = None,
                         num_samples: int = 100, batch_size: int = 100, hide_color=0, **kwargs):
        '''Explains a prediction

        This function calls the Lime module. It generates neighborhood data by randomly perturbing features from the instance.
        Then, it learns locally weighted linear models on this neighborhood data to explain each of the classes in an interpretable way.

        Args:
            img (Image.Image): Image to be explained
        Kwargs:
            class_index (int): for classification only. Class or label index to be considered.
            num_samples (int): size of the neighborhood to learn the linear model (cf. Lime documentation)
            batch_size (int): classifier_fn will be called on batches of this size (cf. Lime documentation)
            hide_color (?): TODO
        Returns:
            (?): An explanation object
        '''
        # Set index
        if class_index is not None:
            self.current_class_index = class_index
        else:
            self.current_class_index = 1  # Def to 1
        # Get explanations (images must be convert into rgb, then into np array)
        return self.explainer.explain_instance(np.array(content.convert('RGB')), self.classifier_fn,
                                               labels=(self.current_class_index,),
                                               num_samples=num_samples, batch_size=batch_size,
                                               hide_color=hide_color, top_labels=None)

    def explain_instance_as_html(self, content: Image.Image, **kwargs) -> str:
        '''Explains a prediction - returns an HTML object
        ** NOT IMPLEMENTED **
        '''
        raise NotImplementedError("'explain_instance_as_html' is not defined for LimeExplainer")

    def explain_instance_as_json(self, content: Image.Image, **kwargs) -> Union[dict, list]:
        '''Explains a prediction - returns an JSON serializable object
        ** NOT IMPLEMENTED **
        '''
        raise NotImplementedError("'explain_instance_as_json' is not defined for LimeExplainer")

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
