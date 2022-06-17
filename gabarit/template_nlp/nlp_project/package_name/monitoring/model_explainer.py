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
# ** EXPERIMENTAL ** - AttentionExplainer -> Attention Explainer wrapper class

import logging
import numpy as np
from typing import Type, Union, Any
from lime.explanation import Explanation
from words_n_fun.preprocessing import api
from lime.lime_text import IndexedString, TextDomainMapper, LimeTextExplainer

from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.model_embedding_lstm_structured_attention import ModelEmbeddingLstmStructuredAttention


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
            TypeError: If the provided model does not implement a `predict` function
            TypeError: If the provided model does not implement a `predict_proba` function
            TypeError: If the provided model does not have a `list_classes` attribute
        '''
        super().__init__()
        pred_op = getattr(model, "predict", None)
        pred_proba_op = getattr(model, "predict_proba", None)

        if pred_op is None or not callable(pred_op):
            raise TypeError("The supplied model must implement a predict() function")
        if pred_proba_op is None or not callable(pred_proba_op):
            raise TypeError("The supplied model must implement a predict_proba() function")
        if getattr(model, "list_classes", None) is None:
            raise TypeError("The supplied model must have a list_classes attribute")

        self.model = model
        self.model_conf = model_conf
        self.class_names = self.model.list_classes
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def explain_instance(self, text: str, classes: Union[list, None] = None, max_features: int = 15, **kwargs):
        '''Explains a prediction

        This function calls the Lime module. It creates a linear model around the input text to evaluate
        the weight of each word in the final prediction.

        Args:
            text (str): Text to be explained
        Kwargs:
            classes (list): Classes to be compared
            max_features (int): Maximum number of features (cf. Lime documentation)
        Returns:
            (?): An explanation object
        '''
        # If no class provided, we only consider the predicted one against all others
        if classes is None:
            classes = [self.class_names.index(self.model.predict(text))]
        # Ohterwise we consider the provided ones
        else:
            classes = [self.class_names.index(x) for x in classes]

        # Define classifier_fn
        def classifier_fn(list_text: list) -> np.ndarray:
            '''Classifier function - retrieves proba'''
            # Get preprocessor
            if 'preprocess_str' in self.model_conf.keys():
                preprocess_str = self.model_conf['preprocess_str']
            else:
                preprocess_str = 'no_preprocess'
            preprocessor = preprocess.get_preprocessor(preprocess_str)
            # Preprocess
            list_text_preprocessed = preprocessor(list_text)
            # Get probas & return
            return self.model.predict_proba(list_text_preprocessed)

        # Get explanations
        return self.explainer.explain_instance(text, classifier_fn, labels=classes, num_features=max_features)

    def explain_instance_as_html(self, text: str, classes: Union[list, None] = None, max_features: int = 15, **kwargs) -> str:
        '''Explains a prediction - returns an HTML object

        Args:
            text (str): Text to be explained
        Kwargs:
            classes (list): Classes to be compared
            max_features (int): Maximum number of features (cf. Lime documentation)
        Returns:
            str: An HTML code with the explanation
        '''
        return self.explain_instance(text, classes, max_features).as_html()

    def explain_instance_as_list(self, text: str, classes: Union[list, None] = None, max_features: int = 15, **kwargs) -> list:
        '''Explains a prediction - returns a list object

        Args:
            text (str): Text to be explained
        Kwargs:
            classes (list): Classes to be compared
            max_features (int): Maximum number of features (cf. Lime documentation)
        Returns:
            list: List of tuples with words and corresponding weights
        '''
        return self.explain_instance(text, classes, max_features).as_list()


# ** EXPERIMENTAL **
# ** EXPERIMENTAL **
# ** EXPERIMENTAL **


class AttentionExplainer(Explainer):
    '''Attention Explainer wrapper class

    From Gaëlle JOUIS Thesis
    '''

    def __init__(self, model: Type[ModelClass]) -> None:
        ''' Initialization

        Args:
            model: A model instance with predict & predict_proba functions, and list_classes attribute
        Raises:
            TypeError: If the provided model is not a ModelEmbeddingLstmStructuredAttention model
            TypeError: If the provided model does not have a `list_classes` attribute
        '''
        super().__init__()
        if not isinstance(model, ModelEmbeddingLstmStructuredAttention):
            raise TypeError("At the moment AttentionExplainer is only available for ModelEmbeddingLstmStructuredAttention models")
        if getattr(model, "list_classes", None) is None:
            raise TypeError("The supplied model must have a list_classes attribute")

        self.model: Any = model
        self.class_names: list = self.model.list_classes

    def explain_instance(self, text: str, classes: Union[list, None] = None, max_features: int = 15, pipeline: list = None, **kwargs):
        '''Explains a prediction

        This function is based one Gaëlle JOUIS works.

        Args:
            text (str): Text to be explained
        Kwargs:
            classes (list): Classes to be compared
            max_features (int): Maximum number of features (cf. Lime documentation)
            pipeline (list):  ???? To be confirmed with gaëlle JOUIS
        Returns:
            (?): An explanation object
        '''
        # If no class provided, we only consider the predicted one against all others
        if classes is None:
            classes = [self.class_names.index(self.model.predict(text))]
        # Ohterwise we consider the provided ones
        else:
            classes = [self.class_names.index(x) for x in classes]

        # ???
        # TODO: shouldn't it be the model preprocess pipeline ?
        if pipeline is None:
            pipeline = ['to_lower', 'remove_accents']  # ???
            text = api.preprocess_pipeline(text, pipeline)

        # Prepare explanations
        indexed_string = IndexedString(text, split_expression=self.model.tokenizer.split, **kwargs)
        domain_mapper = TextDomainMapper(indexed_string)
        exp = Explanation(domain_mapper=domain_mapper, class_names=self.class_names + ["Important"])
        exp.predict_proba = self.model.predict_proba(text)
        exp_from_model = self.model.explain(text, attention_threshold=0, fix_index=True)

        # Format results
        word_list = []
        for word in exp_from_model.keys():
            word_list.append((word, float(exp_from_model[word][1])))
        exp.local_exp = {len(self.class_names): word_list}  # exp.local_exp = sorted list of tuples where each tuple (x,y) corresponds to the feature id (x) and the local weight (y).
        exp.intercept = '0.42'  # exp.intercept = can be ignored
        exp.score = '0.42'  # exp.score = the R^2 value of the returned explanation. ???
        exp.local_pred = '0.42'  # exp.local_pred = the prediction of the explanation model on the original instance. Can be ignored.

        return exp

    def explain_instance_as_html(self, text: str, classes: Union[list, None] = None, max_features: int = 15, **kwargs) -> str:
        '''Explains a prediction - returns an HTML object

        Args:
            text (str): Text to be explained
        Kwargs:
            classes (list): Classes to be compared
            max_features (int): Maximum number of features (cf. Lime documentation)
        Returns:
            str: An HTML code with the explanation
        '''
        return self.explain_instance(text, classes, max_features).as_html()

    def explain_instance_as_list(self, text: str, classes: Union[list, None] = None, max_features: int = 15, **kwargs) -> list:
        '''Explains a prediction - returns a list object

        Args:
            text (str): Text to be explained
        Kwargs:
            classes (list): Classes to be compared
            max_features (int): Maximum number of features (cf. Lime documentation)
        Returns:
            list: List of tuples with words and corresponding weights
        '''
        return self.explain_instance(text, classes, max_features).as_list(label=len(self.class_names))


# ** EXPERIMENTAL **
# ** EXPERIMENTAL **
# ** EXPERIMENTAL **


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
