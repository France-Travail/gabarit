#!/usr/bin/env python3

## Xgboost model
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
# - ModelXgboostClassifier -> Xgboost model for classification


import os
import json
import logging
import numpy as np
import pandas as pd
import dill as pickle
from xgboost import XGBClassifier
from joblib import Parallel, delayed
from typing import Optional, List, Any, Union

from sklearn.base import is_classifier
from sklearn.multioutput import _fit_estimator
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_fit_params, _deprecate_positional_args, has_fit_parameter

from {{package_name}} import utils
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.classifiers.model_classifier import ModelClassifierMixin  # type: ignore


class ModelXgboostClassifier(ModelClassifierMixin, ModelClass):
    '''Xgboost model for classification'''

    _default_name = 'model_xgboost_classifier'

    def __init__(self, xgboost_params: Union[dict, None] = None, early_stopping_rounds: int = 5, validation_split: float = 0.2, **kwargs) -> None:
        '''Initialization of the class  (see ModelClass & ModelClassifierMixin for more arguments)

        Kwargs:
            xgboost_params (dict): Parameters for the Xgboost
                -> https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
            early_stopping_rounds (int): Number of rounds for early stopping
            validation_split (float): Validation split fraction.
                Only used if not validation dataset in the fit input
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Set parameters
        if xgboost_params is None:
            xgboost_params = {}
        self.xgboost_params = xgboost_params
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_split = validation_split

        # Set objective (if not in params) & init. model
        if 'objective' not in self.xgboost_params.keys():
            self.xgboost_params['objective'] = 'binary:logistic'
            #  List of objectives https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
        # WARNING, if multi-classes, AUTOMATIC backup on multi:softprob (by xgboost)
        # https://stackoverflow.com/questions/57986259/multiclass-classification-with-xgboost-classifier
        self.model = XGBClassifier(**self.xgboost_params)

        # If multi-labels, we use MultiOutputClassifier
        if self.multi_label:
            self.model = MyMultiOutputClassifier(self.model)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, with_shuffle: bool = True, **kwargs) -> None:
        '''Trains the model
           **kwargs permits compatibility with Keras model

        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train (?): Array-like, shape = [n_samples, n_targets]
            x_valid (?): Array-like, shape = [n_samples, n_features]
            y_valid (?): Array-like, shape = [n_samples, n_targets]
        Kwargs:
            with_shuffle (bool): If x, y must be shuffled before fitting
        Raises:
            RuntimeError: If the model is already fitted
        '''
        # TODO: Check if the training can be continued
        if self.trained:
            self.logger.error("We can't train again a xgboost model")
            self.logger.error("Please train a new model")
            raise RuntimeError("We can't train again a xgboost model")

        # We check input format
        x_train, y_train = self._check_input_format(x_train, y_train, fit_function=True)
        # If there is a validation set, we also check the format (but fit_function to False)
        if y_valid is not None and x_valid is not None:
            x_valid, y_valid = self._check_input_format(x_valid, y_valid, fit_function=False)
        # Otherwise, we do a random split
        else:
            self.logger.warning(f"Warning, no validation dataset. We split the training set (fraction valid = {self.validation_split})")
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=self.validation_split)

        # Gets the input columns
        original_list_classes: Optional[List[Any]] = None  # None if no 'columns' attribute
        if hasattr(y_train, 'columns'):
            original_list_classes = list(y_train.columns)

        # Shuffle x, y if wanted
        if with_shuffle:
            p = np.random.permutation(len(x_train))
            x_train = np.array(x_train)[p]
            y_train = np.array(y_train)[p]
        # Else still transform to numpy array
        else:
            x_train = np.array(x_train)
            y_train = np.array(y_train)

        # Also get x_valid & y_valid as numpy
        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        # Set eval set and train
        eval_set = [(x_train, y_train), (x_valid, y_valid)]  # If there’s more than one item in eval_set, the last entry will be used for early stopping.
        prior_objective = self.model.objective if not self.multi_label else self.model.estimator.objective
        self.model.fit(x_train, y_train, eval_set=eval_set, early_stopping_rounds=self.early_stopping_rounds, verbose=True)
        post_objective = self.model.objective if not self.multi_label else self.model.estimator.objective
        if prior_objective != post_objective:
            self.logger.warning("Warning: the objective function was automatically changed by XGBOOST")
            self.logger.warning(f"Before: {prior_objective}")
            self.logger.warning(f"After: {post_objective}")

        # Set list classes
        if not self.multi_label:
            self.list_classes = list(self.model.classes_)
        else:
            if original_list_classes is not None:
                self.list_classes = original_list_classes
            else:
                self.logger.warning(
                    "Can't read the name of the columns of y_train -> inverse transformation won't be possible"
                )
                # We still create a list of classes in order to be compatible with other functions
                self.list_classes = [str(_) for _ in range(pd.DataFrame(y_train).shape[1])]

        # Set dict_classes based on list classes
        self.dict_classes = {i: col for i, col in enumerate(self.list_classes)}

        # Set trained
        self.trained = True
        self.nb_fit += 1

    @utils.trained_needed
    def predict(self, x_test: pd.DataFrame, return_proba: bool = False, **kwargs) -> np.ndarray:
        '''Predictions on test set

        Args:
            x_test (pd.DataFrame): DataFrame with the test data to be predicted
        Kwargs:
            return_proba (bool): If the function should return the probabilities instead of the classes (Keras compatibility)
        Returns:
            (np.ndarray): Array
                # If not return_proba, shape = [n_samples,] or [n_samples, n_classes]
                # Else, shape = [n_samples, n_classes]
        '''
        # If we want probabilities, we use predict_proba
        if return_proba:
            return self.predict_proba(x_test, **kwargs)
        # Otherwise, returns the prediction :
        else:
            # We check input format
            x_test, _ = self._check_input_format(x_test)
            # Warning, "The method returns the model from the last iteration"
            # But : "Predict with X. If the model is trained with early stopping, then best_iteration is used automatically."
            y_pred = self.model.predict(x_test)
            return y_pred

    @utils.trained_needed
    def predict_proba(self, x_test: pd.DataFrame, **kwargs) -> np.ndarray:
        '''Predicts the probabilities on the test set

        Args:
            x_test (pd.DataFrame): DataFrame to be predicted -> retrieve the probabilities
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        # We check input format
        x_test, _ = self._check_input_format(x_test)

        #
        probas = np.array(self.model.predict_proba(x_test))
        # If use of MultiOutputClassifier ->  returns probabilities of 0 and 1 for all elements and all classes
        # Correction in cas where we detect a shape of length > 2 (ie. equals to 3)
        # Reminder : we do not manage multi-labels multi-classes
        if len(probas.shape) > 2:
            probas = np.swapaxes(probas[:, :, 1], 0, 1)
        return probas

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save model
        if json_data is None:
            json_data = {}

        json_data['librairie'] = 'xgboost'
        json_data['xgboost_params'] = self.xgboost_params
        json_data['early_stopping_rounds'] = self.early_stopping_rounds
        json_data['validation_split'] = self.validation_split

        # Save xgboost standalone
        if self.level_save in ['MEDIUM', 'HIGH']:
            if not self.multi_label:
                if self.trained:
                    save_path = os.path.join(self.model_dir, 'xbgoost_standalone.model')
                    self.model.save_model(save_path)
                else:
                    self.logger.warning("Can't save the XGboost in standalone because it hasn't been already fitted")
            else:
                # If multi-labels, we use a multi-output and fits several xgboost (cf. strategy sklearn)
                # We can't save only one xgboost, so we use pickle to save
                # Problem : the pickle won't be compatible with updates :'(
                save_path = os.path.join(self.model_dir, f"{self.model_name}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(self.model, f)

        # Save
        super().save(json_data=json_data)

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            xgboost_path (str): Path to standalone xgboost
            preprocess_pipeline_path (str): Path to preprocess pipeline
        Raises:
            ValueError: If configuration_path is None
            ValueError: If xgboost_path is None
            ValueError: If preprocess_pipeline_path is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object xgboost_path is not an existing file
            FileNotFoundError: If the object preprocess_pipeline_path is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        xgboost_path = kwargs.get('xgboost_path', None)
        preprocess_pipeline_path = kwargs.get('preprocess_pipeline_path', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if xgboost_path is None:
            raise ValueError("The argument xgboost_path can't be None")
        if preprocess_pipeline_path is None:
            raise ValueError("The argument preprocess_pipeline_path can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(xgboost_path):
            raise FileNotFoundError(f"The file {xgboost_path} does not exist")
        if not os.path.exists(preprocess_pipeline_path):
            raise FileNotFoundError(f"The file {preprocess_pipeline_path} does not exist")

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
        for attribute in ['model_type', 'x_col', 'y_col', 'columns_in', 'mandatory_columns',
                          'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'xgboost_params', 'early_stopping_rounds', 'validation_split']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload xgboost model
        if not self.multi_label:
            self.model.load_model(xgboost_path)
        else:
            with open(xgboost_path, 'rb') as f:  # type: ignore
                self.model = pickle.load(f)

        # Reload pipeline preprocessing
        with open(preprocess_pipeline_path, 'rb') as f:
            self.preprocess_pipeline = pickle.load(f)


# Problem : We want to use the MultiOutputClassifier for the multi-labels cases but, by default,
# it gives the same parameters for each fit. However, for the XGboost, we want to validate
# on the right label (error if we do nothing because it is not the right format).
# Solution : We modify the MultiOutputClassifier class to give the right subset of y_valid for each fit
# From : https://stackoverflow.com/questions/66785587/how-do-i-use-validation-sets-on-multioutputregressor-for-xgbregressor
# From : https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/multioutput.py#L293
# From : https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/multioutput.py#L64
class MyMultiOutputClassifier(MultiOutputClassifier):

    @_deprecate_positional_args
    def __init__(self, estimator, *, n_jobs=None) -> None:
        super().__init__(estimator, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None, **fit_params) -> Any:
        ''' Fit the model to data.
        Fit a separate model for each output variable.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.
        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.
            .. versionadded:: 0.23
        Returns
        -------
        self : object
        '''
        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement"
                             " a fit method")

        X, y = self._validate_data(X, y,
                                   force_all_finite=False,
                                   multi_output=True, accept_sparse=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")

        if (sample_weight is not None and not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")

        fit_params_validated = _check_fit_params(X, fit_params)

        # New : extract eval_set
        if 'eval_set' in fit_params_validated.keys():
            eval_set = fit_params_validated.pop('eval_set')
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X, y[:, i], sample_weight,
                    **fit_params_validated,
                    eval_set=[(X_test, Y_test[:, i]) for X_test, Y_test in eval_set])
                for i in range(y.shape[1]))
        # Pas d'eval_set
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X, y[:, i], sample_weight,
                    **fit_params_validated)
                for i in range(y.shape[1]))
        return self


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
