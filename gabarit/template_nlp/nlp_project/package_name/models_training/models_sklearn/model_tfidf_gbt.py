#!/usr/bin/env python3

## Model TFIDF GBT
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
# - ModelTfidfGbt -> Model for predictions via TF-IDF + GBT

import logging
import numpy as np
from typing import Union, Any

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from ... import utils
from .model_pipeline import ModelPipeline


class ModelTfidfGbt(ModelPipeline):
    '''Model for predictions via TF-IDF + GBT'''

    _default_name = 'model_tfidf_gbt'

    def __init__(self, tfidf_params: Union[dict, None] = None, gbt_params: Union[dict, None] = None, multiclass_strategy: Union[str, None] = None, **kwargs) -> None:
        '''Initialization of the class (see ModelPipeline & ModelClass for more arguments)

        Kwargs:
            tfidf_params (dict) : Parameters for the tfidf
            gbt_params (dict) : parameters for the gbt
            multiclass_strategy (str): Multi-classes strategy, 'ovr' (OneVsRest), or 'ovo' (OneVsOne). If None, use the default of the algorithm.
        Raises:
            ValueError: If multiclass_strategy is not 'ovo', 'ovr' or None
        '''
        if multiclass_strategy is not None and multiclass_strategy not in ['ovo', 'ovr']:
            raise ValueError(f"The value of 'multiclass_strategy' must be 'ovo' or 'ovr' (not {multiclass_strategy})")
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Manage model
        if tfidf_params is None:
            tfidf_params = {}
        self.tfidf = TfidfVectorizer(**tfidf_params)
        if gbt_params is None:
            gbt_params = {}
        self.gbt = GradientBoostingClassifier(**gbt_params)
        self.multiclass_strategy = multiclass_strategy

        # Can't do multi-labels / multi-classes
        if not self.multi_label:
            # If not multi-classes : no impact
            if multiclass_strategy == 'ovr':
                self.pipeline = Pipeline([('tfidf', self.tfidf), ('gbt', OneVsRestClassifier(self.gbt))])
            elif multiclass_strategy == 'ovo':
                self.pipeline = Pipeline([('tfidf', self.tfidf), ('gbt', OneVsOneClassifier(self.gbt))])
            else:
                self.pipeline = Pipeline([('tfidf', self.tfidf), ('gbt', self.gbt)])

        # Manage multi-labels -> add a MultiOutputClassifier
        # The GBT does not natively support multi-labels
        if self.multi_label:
            self.pipeline = Pipeline([('tfidf', self.tfidf), ('gbt', MultiOutputClassifier(self.gbt))])

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Probabilities prediction on the test dataset
            'ovo' can't predict probabilities. By default we return 1 if it is the predicted class, 0 otherwise.

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples, n_features]
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        # Use super() of Pipeline class if != 'ovo' or multi-labels
        if self.multi_label or self.multiclass_strategy != 'ovo':
            return super().predict_proba(x_test, **kwargs)
        # We return 1 if predicted, otherwise 0
        else:
            preds = self.pipeline.predict(x_test)
            # Format ['a', 'b', 'c', 'a', ..., 'b']
            # Transform to "proba"
            transform_dict = {col: [0. if _ != i else 1. for _ in range(len(self.list_classes))] for i, col in enumerate(self.list_classes)}
            probas = np.array([transform_dict[x] for x in preds])
        return probas

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save model
        if json_data is None:
            json_data = {}

        # No need to save the parameters of the pipeline steps, it is already done in ModelPipeline
        json_data['multiclass_strategy'] = self.multiclass_strategy

        # Save
        super().save(json_data=json_data)

    @classmethod
    def _init_new_instance_from_configs(cls, configs):
        '''Inits a new instance from a set of configurations

        Args:
            configs: a set of configurations of a model to be reloaded
        Returns:
            ModelClass: the newly generated class
        '''
        # Call parent
        model = super()._init_new_instance_from_configs(configs)

        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['multiclass_strategy']:
            setattr(model, attribute, configs.get(attribute, getattr(model, attribute)))

        # Return the new model
        return model

    def _load_standalone_files(self, default_model_dir: Union[str, None] = None, *args, **kwargs):
        '''Loads standalone files for a newly created model via _init_new_instance_from_configs

        Kwargs:
            default_model_dir (str): a path to look for default file paths
                                     If None, standalone files path should all be provided
        '''
        # Call parent
        super()._load_standalone_files(default_model_dir=default_model_dir, **kwargs)

        # Reload pipeline elements
        self.tfidf = self.pipeline['tfidf']

        # Manage multi-labels or multi-classes
        if not self.multi_label and self.multiclass_strategy is None:
            self.gbt = self.pipeline['gbt']
        else:
            self.gbt = self.pipeline['gbt'].estimator


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
