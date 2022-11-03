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
# - ShapExplainer -> Shap Explainer wrapper class

import shap
import base64
import logging
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Type, Union, Any

from {{package_name}}.models_training import utils_models
from {{package_name}}.models_training.model_class import ModelClass


class Explainer:
    '''Parent class for the explainers'''

    def __init__(self) -> None:
        '''Initialization of the parent class'''
        self.logger = logging.getLogger(__name__)

    def explain_instance(self, content: pd.DataFrame, **kwargs) -> Any:
        '''Explains a prediction

        Args:
            content (pd.DataFrame): Single entry to be explained
        Returns:
            (?): An explanation object
        '''
        raise NotImplementedError("'explain_instance' needs to be overridden")

    def explain_instance_as_html(self, content: pd.DataFrame, **kwargs) -> str:
        '''Explains a prediction - returns an HTML object

        Args:
            content (pd.DataFrame): Single entry to be explained
        Returns:
            str: An HTML code with the explanation
        '''
        raise NotImplementedError("'explain_instance_as_html' needs to be overridden")


class ShapExplainer(Explainer):
    '''Shap Explainer wrapper class'''

    def __init__(self, model: Type[ModelClass], anchor_data: pd.DataFrame, anchor_preprocessed: bool = False) -> None:
        ''' Initialization

        Args:
            model: A model instance with predict & predict_proba functions, and list_classes attribute
            anchor_data (pd.DataFrame): data anchor needed by shap (usually 100 data points)
        Kwargs:
            anchor_preprocessed (bool): If the anchor data has already been preprocessed
        Raises:
            TypeError: If the provided model does not implement a `predict` function
            TypeError: If the provided model is a classifier and does not implement a `predict_proba` function
            TypeError: If the provided model is a classifier and does not have a `list_classes` attribute
        '''
        super().__init__()
        pred_op = getattr(model, "predict", None)
        pred_proba_op = getattr(model, "predict_proba", None)

        if pred_op is None or not callable(pred_op):
            raise TypeError("The supplied model must implement a predict() function")
        # Check classifier
        if model.model_type == 'classifier':
            if pred_proba_op is None or not callable(pred_proba_op):
                raise TypeError("The supplied model must implement a predict_proba() function")
            if getattr(model, "list_classes", None) is None:
                raise TypeError("The supplied model must have a list_classes attribute")

        # Set attributes
        self.model = model
        self.model_type = model.model_type
        self.class_names = getattr(model, "list_classes", None)
        # Our explainers will explain a prediction for a given class / label
        # These atributes are set on the fly and will change the proba function used by the explainer
        self.current_class_or_label_index = 0
        fn_output = self.classifier_fn if self.model_type == 'classifier' else self.regressor_fn

        # Preprocess the anchor data
        if not anchor_preprocessed:
            if self.model.preprocess_pipeline is not None:
                anchor_prep = utils_models.apply_pipeline(anchor_data, self.model.preprocess_pipeline)
            else:
                anchor_prep = anchor_data.copy()
        else:
            # Check columns
            try:
                anchor_prep = anchor_prep[self.model.x_col]
            except:
                raise ValueError("Provided anchor data (already preprocessed) do not match model's inputs columns")
        # Create the explainer
        self.explainer = shap.Explainer(fn_output, anchor_prep)

    def classifier_fn(self, content_prep: pd.DataFrame) -> np.ndarray:
        '''Function to get probabilities from a dataset (already preprocessed) - classifiers

        Args:
            content_prep (pd.DataFrame): dataset (already preprocessed) to be considered
        Returns:
            np.array: probabilities
        '''
        # Get probabilities
        return self.model.predict_proba(content_prep)[:, self.current_class_or_label_index]

    def regressor_fn(self, content_prep: pd.DataFrame) -> np.ndarray:
        '''Function to get predictions from a dataset (already preprocessed) - regressors

        Args:
            content_prep (pd.DataFrame): dataset (already preprocessed) to be considered
        Returns:
            np.array: predictions
        '''
        # Get predictions
        return self.model.predict(content_prep)

    def explain_instance(self, content: pd.DataFrame, class_or_label_index: int = None, **kwargs) -> Any:
        '''Explains a prediction

        This function calls the Shap module.

        Args:
            content (pd.DataFrame): Single entry to be explained
        Kwargs:
            class_or_label_index (int): for classification only. Class or label index to be considered.
        Returns:
            (?): Shap values
        '''
        # Apply preprocessing
        if self.model.preprocess_pipeline is not None:
            df_prep = utils_models.apply_pipeline(content, self.model.preprocess_pipeline)
        else:
            df_prep = content.copy()
            logger.warning("No preprocessing pipeline found - we consider no preprocessing, but it should not be so !")
        # Set index (if needed)
        if class_or_label_index is not None:
            self.current_class_or_label_index = class_or_label_index
        # Get explanations
        return self.explainer(df_prep)  # Shap values

    def explain_instance_as_html(self, content: pd.DataFrame, class_or_label_index: int = None, **kwargs) -> str:
        '''Explains a prediction - returns an HTML object

        Args:
            content (pd.DataFrame): Single entry to be explained
        Kwargs:
            class_or_label_index (int): for classification only. Class or label index to be considered.
        Returns:
            str: An HTML code with the explanation
        '''
        shap_values = self.explain_instance(content, class_or_label_index=class_or_label_index)
        # Waterfall figure
        plt.clf()
        waterfall_fig = shap.plots.waterfall(shap_values[0], show=False)
        with tempfile.TemporaryFile('w+b') as plt_file:
            waterfall_fig.savefig(plt_file, format='png', bbox_inches='tight')
            plt_file.seek(0)
            encoded = base64.b64encode(plt_file.read())
        plt.clf()
        html_waterfall = f"<img src=\'data:image/png;base64, {encoded.decode('utf-8')}\'>"
        # Force figure
        html_force = shap.plots.force(shap_values[0]).html()
        # Combine & return
        final_html = f"{html_waterfall}<br>{html_force}"
        return final_html


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
