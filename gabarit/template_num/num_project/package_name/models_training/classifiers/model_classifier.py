#!/usr/bin/env python3
# type: ignore
# Too complicated to manage Mixin & types

## Definition of a parent class for the classifier models
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
# - ModelClassifierMixin -> Parent class for classifier models


import os
import re
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             multilabel_confusion_matrix, precision_score, recall_score)

from {{package_name}} import utils
from {{package_name}}.monitoring.model_logger import ModelLogger

sns.set(style="darkgrid")


class ModelClassifierMixin:
    '''Parent class (Mixin) for classifier models'''

    # Not implemented :
    # -> predict : To be implementd by the parent class when using this mixin

    def __init__(self, level_save: str = 'HIGH', multi_label: bool = False, **kwargs) -> None:
        '''Initialization of the class

        Kwargs:
            level_save (str): Level of saving
                LOW: stats + configurations + logger keras - /!\\ The model can't be reused /!\\ -
                MEDIUM: LOWlevel_save + hdf5 + pkl + plots
                HIGH: MEDIUM + predictions
            multi_label (bool): If the classification must be multi-labels
        Raises:
            ValueError: If the object level_save is not a valid option (['LOW', 'MEDIUM', 'HIGH'])
        '''
        super().__init__(level_save=level_save, **kwargs)  # forwards level_save & all unused arguments

        if level_save not in ['LOW', 'MEDIUM', 'HIGH']:
            raise ValueError(f"The object level_save ({level_save}) is not a valid option (['LOW', 'MEDIUM', 'HIGH'])")

        # Get logger
        self.logger = logging.getLogger(__name__)

        # Model type
        self.model_type = 'classifier'

        # Multi-labels ?
        self.multi_label = multi_label

        # Classes list to use (set on fit)
        self.list_classes = None
        self.dict_classes = None

        # Other options
        self.level_save = level_save

    @utils.trained_needed
    def predict_with_proba(self, x_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        '''Predictions on test with probabilities

        Args:
            x_test (pd.DataFrame): DataFrame with the test data to be predicted
        Returns:
            predicted_class (np.ndarray): The predicted classes, shape = [n_samples, n_classes] if multi-labels, shape = [n_samples, 1] otherwise
            predicted_proba (np.ndarray): The predicted probabilities for each class, shape = [n_samples, n_classes]
        '''
        # Process
        predicted_proba = self.predict(x_test, return_proba=True)
        predicted_class = self.get_classes_from_proba(predicted_proba)
        return predicted_class, predicted_proba

    @utils.trained_needed
    def get_predict_position(self, x_test: pd.DataFrame, y_true) -> np.ndarray:
        '''Gets the order of predictions of y_true.
        Positions start at 1 (not 0)

        Args:
            x_test (pd.DataFrame): DataFrame with the test data to be predicted
            y_true (?): Array-like, shape = [n_samples, n_targets]
        Raises:
            ValueError: Can't use this method with multi-labels tasks
        Returns:
            predict_positions (np.ndarray): The order of prediction of y_true shape = [n_samples, ]
        '''
        if self.multi_label:
            raise ValueError("The method 'get_predict_position' is unavailable with multi-labels tasks")
        # Process
        # Cast as pd.Series
        y_true = pd.Series(y_true)
        # Get predicted probabilities
        predicted_proba = self.predict(x_test, return_proba=True)
        # Get position
        order = predicted_proba.argsort()
        ranks = len(self.dict_classes.values()) - order.argsort()
        df_probas = pd.DataFrame(ranks, columns=self.dict_classes.values())
        predict_positions = np.array([df_probas.loc[i, cl] if cl in df_probas.columns else -1 for i, cl in enumerate(y_true)])
        return predict_positions

    def get_classes_from_proba(self, predicted_proba: np.ndarray) -> np.ndarray:
        '''Gets the classes from probabilities

        Args:
            predicted_proba (np.ndarray): The probabilities predicted by the model, shape = [n_samples, n_classes]
        Returns:
            predicted_class (np.ndarray): Shape = [n_samples, n_classes] if multi-labels, shape = [n_samples] otherwise
        '''
        if not self.multi_label:
            predicted_class = np.vectorize(lambda x: self.dict_classes[x])(predicted_proba.argmax(axis=-1))
        else:
            # If multi-labels, returns a list of 0 and 1
            predicted_class = np.vectorize(lambda x: 1 if x >= 0.5 else 0)(predicted_proba)
        return predicted_class

    def get_top_n_from_proba(self, predicted_proba: np.ndarray, n: int = 5) -> Tuple[list, list]:
        '''Gets the Top n predictions from probabilities

        Args:
            predicted_proba (np.ndarray): Predicted probabilities = [n_samples, n_classes]
        kwargs:
            n (int): Number of classes to return
        Raises:
            ValueError: If the number of classes to return is greater than the number of classes of the model
        Returns:
            top_n (list): Top n predicted classes
            top_n_proba (list): Top n probabilities (corresponding to the top_n list of classes)
        '''
        # TODO: Make this method available with multi-labels tasks
        if self.multi_label:
            raise ValueError("The method 'get_top_n_from_proba' is unavailable with multi-labels tasks")
        if self.list_classes is not None and n > len(self.list_classes):
            raise ValueError("The number of classes to return is greater than the number of classes of the model")
        # Process
        idx = predicted_proba.argsort()[:, -n:][:, ::-1]
        top_n_proba = list(np.take_along_axis(predicted_proba, idx, axis=1))
        top_n = list(np.vectorize(lambda x: self.dict_classes[x])(idx))
        return top_n, top_n_proba

    def inverse_transform(self, y: Union[list, np.ndarray]) -> Union[list, tuple]:
        '''Gets a list of classes from the predictions (mainly useful for multi-labels)

        Args:
            y (list | np.ndarray): Array-like, shape = [n_samples, n_classes], arrays of 0s and 1s
                   OR 1D array shape = [n_classes] (only one prediction)
        Raises:
            ValueError: If the size of y does not correspond to the number of classes of the model
        Returns:
            List of tuple if multi-labels and several predictions
            Tuple if multi-labels and one prediction
            List of classes if mono-label
        '''
        # If multi-labels, get classes in tuple
        if self.multi_label:
            if y.shape[-1] != len(self.list_classes):  # We consider "-1" in order to take care of the case where y is 1D
                raise ValueError(f"The size of y ({y.shape[-1]}) does not correspond"
                                 f" to the number of classes of the model : ({len(self.list_classes)})")
            # Manage 1D array (only one prediction)
            if len(y.shape) == 1:
                return tuple(np.array(self.list_classes).compress(y))
            # Several predictions
            else:
                return [tuple(np.array(self.list_classes).compress(indicators)) for indicators in y]
        # If mono-label, just cast in list if y is np array
        else:
            return list(y) if isinstance(y, np.ndarray) else y

    def get_and_save_metrics(self, y_true, y_pred, df_x: Union[pd.DataFrame, None] = None,
                             series_to_add: Union[List[pd.Series], None] = None, type_data: str = '',
                             model_logger: Union[ModelLogger, None] = None) -> pd.DataFrame:
        '''Gets and saves the metrics of a model

        Args:
            y_true (?): Array-like, shape = [n_samples, n_targets]
            y_pred (?): Array-like, shape = [n_samples, n_targets]
        Kwargs:
            df_x (pd.DataFrame or None): Input dataFrame used for the prediction
            series_to_add (list<pd.Series>): List of pd.Series to add to the dataframe
            type_data (str): Type of dataset (validation, test, ...)
            model_logger (ModelLogger): Custom class to log the metrics with MLflow
        Returns:
            pd.DataFrame: The dataframe containing the statistics
        '''
        # Cast to np.array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Check shapes
        if not self.multi_label:
            if len(y_true.shape) == 2 and y_true.shape[1] == 1:
                y_true = np.ravel(y_true)
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
                y_pred = np.ravel(y_pred)

        # Save a predictionn file if wanted
        if self.level_save == 'HIGH':
            # Inverse transform
            y_true_df = list(self.inverse_transform(y_true))
            y_pred_df = list(self.inverse_transform(y_pred))

            # Concat in a dataframe
            if df_x is not None:
                df = df_x.copy()
                df['y_true'] = y_true_df
                df['y_pred'] = y_pred_df
            else:
                df = pd.DataFrame({'y_true': y_true_df, 'y_pred': y_pred_df})
            # Add a matched column
            df['matched'] = (df['y_true'] == df['y_pred']).astype(int)
            # Add some more columns
            if series_to_add is not None:
                for ser in series_to_add:
                    df[ser.name] = ser.reset_index(drop=True).reindex(index=df.index)  # Reindex

            # Save predictions
            file_path = os.path.join(self.model_dir, f"predictions{'_' + type_data if len(type_data) > 0 else ''}.csv")
            df.sort_values('matched', ascending=True).to_csv(file_path, sep='{{default_sep}}', index=None, encoding='{{default_encoding}}')

        # Gets global f1 score / acc_tot / trues / falses / precision / recall / support
        if self.multi_label:
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            trues = np.sum(np.all(np.equal(y_true, y_pred), axis=1))
            falses = len(y_true) - trues
            acc_tot = trues / len(y_true)
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            support = list(pd.DataFrame(y_true).sum().values)
            support = [_ / sum(support) for _ in support] + [1.0]
        else:
            # We use 'weighted' even in the mono-label case since there can be several classes !
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            trues = np.sum(y_true == y_pred)
            falses = np.sum(y_true != y_pred)
            acc_tot = accuracy_score(y_true, y_pred)
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            labels_tmp, counts_tmp = np.unique(y_true, return_counts=True)
            support = [0.] * len(self.list_classes) + [1.0]
            for i, cl in enumerate(self.list_classes):
                if cl in labels_tmp:
                    idx_tmp = list(labels_tmp).index(cl)
                    support[i] = counts_tmp[idx_tmp] / y_pred.shape[0]

        # Global Statistics
        self.logger.info('-- * * * * * * * * * * * * * * --')
        self.logger.info(f"Statistics f1-score{' ' + type_data if len(type_data) > 0 else ''}")
        self.logger.info('--------------------------------')
        self.logger.info(f"Total accuracy : {round(acc_tot * 100, 2)}% \t Trues: {trues} \t Falses: {falses}")
        self.logger.info(f"F1-score (weighted) : {round(f1_weighted, 5)}")
        self.logger.info(f"Precision (weighted) : {round(precision_weighted, 5)}")
        self.logger.info(f"Recall (weighted) : {round(recall_weighted, 5)}")
        self.logger.info('--------------------------------')

        # Metrics file
        df_stats = pd.DataFrame(columns=['Label', 'F1-Score', 'Accuracy',
                                         'Precision', 'Recall', 'Trues', 'Falses',
                                         'True positive', 'True negative',
                                         'False positive', 'False negative',
                                         'Condition positive', 'Condition negative',
                                         'Predicted positive', 'Predicted negative'])

        # Add metrics depending on mono/multi labels & manage confusion matrices
        labels = self.list_classes
        log_stats = len(labels) < 50
        if self.multi_label:
            # Details per category
            mcm = multilabel_confusion_matrix(y_true, y_pred)
            for i, label in enumerate(labels):
                c_mat = mcm[i]
                df_stats = df_stats.append(self._update_info_from_c_mat(c_mat, label, log_info=log_stats), ignore_index=True)
                # Plot individual confusion matrix if level_save > LOW
                if self.level_save in ['MEDIUM', 'HIGH']:
                    none_class = 'not_' + label
                    tmp_label = re.sub(r',|:|\s', '_', label)
                    self._plot_confusion_matrix(c_mat, [none_class, label], type_data=f"{tmp_label}_{type_data}",
                                                normalized=False, subdir=type_data)
                    self._plot_confusion_matrix(c_mat, [none_class, label], type_data=f"{tmp_label}_{type_data}",
                                                normalized=True, subdir=type_data)
        else:
            # Plot confusion matrices if level_save > LOW
            if self.level_save in ['MEDIUM', 'HIGH']:
                if len(labels) > 50:
                    self.logger.warning(
                        f"Warning, there are {len(labels)} categories to plot in the confusion matrix.\n"
                        "Heavy chances of slowness/display bugs/crashes...\n"
                        "SKIP the plots"
                    )
                else:
                    # Global statistics
                    c_mat = confusion_matrix(y_true, y_pred, labels=labels)
                    self._plot_confusion_matrix(c_mat, labels, type_data=type_data, normalized=False)
                    self._plot_confusion_matrix(c_mat, labels, type_data=type_data, normalized=True)

            # Get statistics per class
            for label in labels:
                label_str = str(label)  # Fix : If label is an int, can cause some problems (e.g. only zeroes in the confusion matrix)
                none_class = 'None' if label_str != 'None' else 'others'  # Check that the class is not already 'None'
                y_true_tmp = [label_str if _ == label else none_class for _ in y_true]
                y_pred_tmp = [label_str if _ == label else none_class for _ in y_pred]
                c_mat_tmp = confusion_matrix(y_true_tmp, y_pred_tmp, labels=[none_class, label_str])
                df_stats = df_stats.append(self._update_info_from_c_mat(c_mat_tmp, label, log_info=False), ignore_index=True)

        # Add global statistics
        global_stats = {
            'Label': 'All',
            'F1-Score': f1_weighted,
            'Accuracy': acc_tot,
            'Precision': precision_weighted,
            'Recall': recall_weighted,
            'Trues': trues,
            'Falses': falses,
            'True positive': None,
            'True negative': None,
            'False positive': None,
            'False negative': None,
            'Condition positive': None,
            'Condition negative': None,
            'Predicted positive': None,
            'Predicted negative': None,
        }
        df_stats = df_stats.append(global_stats, ignore_index=True)

        # Add support
        df_stats['Support'] = support

        # Save .csv
        file_path = os.path.join(self.model_dir, f"f1{'_' + type_data if len(type_data) > 0 else ''}@{f1_weighted}.csv")
        df_stats.to_csv(file_path, sep='{{default_sep}}', index=False, encoding='{{default_encoding}}')

        # Save accuracy
        acc_path = os.path.join(self.model_dir, f"acc{'_' + type_data if len(type_data) > 0 else ''}@{round(acc_tot, 5)}")
        with open(acc_path, 'w'):
            pass

        # Upload metrics in mlflow (or another)
        if model_logger is not None:
            # TODO : To put in a function
            # Prepare parameters
            label_col = 'Label'
            metrics_columns = [col for col in df_stats.columns if col != label_col]

            # Log labels
            labels = df_stats[label_col].values
            for i, label in enumerate(labels):
                model_logger.log_param(f'Label {i}', label)
            # Log metrics
            ml_flow_metrics = {}
            for i, row in df_stats.iterrows():
                for c in metrics_columns:
                    metric_key = f"{row[label_col]} --- {c}"
                    # Check that mlflow accepts the key, otherwise, replace it
                    if not model_logger.valid_name(metric_key):
                        metric_key = f"Label {i} --- {c}"
                    ml_flow_metrics[metric_key] = row[c]
            # Log metrics
            model_logger.log_metrics(ml_flow_metrics)

        return df_stats

    def get_metrics_simple_monolabel(self, y_true, y_pred) -> pd.DataFrame:
        '''Gets metrics on mono-label predictions
        Same as the method get_and_save_metrics but without all the fluff (save, etc.)

        Args:
            y_true (?): Array-like, shape = [n_samples,]
            y_pred (?): Array-like, shape = [n_samples,]
        Raises:
            ValueError: If not in mono-label mode
        Returns:
            pd.DataFrame: The dataframe containing statistics
        '''
        if self.multi_label:
            raise ValueError("The method get_metrics_simple_monolabel only works for the mono-label case")

        # Cast to np.array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Check shapes
        if len(y_true.shape) == 2 and y_true.shape[1] == 1:
            y_true = np.ravel(y_true)
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
            y_pred = np.ravel(y_pred)

        # Gets global f1 score / acc_tot / trues / falses / precision / recall / support
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        trues = np.sum(y_true == y_pred)
        falses = np.sum(y_true != y_pred)
        acc_tot = accuracy_score(y_true, y_pred)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        labels_tmp, counts_tmp = np.unique(y_true, return_counts=True)
        support = [0.] * len(self.list_classes) + [1.0]
        for i, cl in enumerate(self.list_classes):
            if cl in labels_tmp:
                idx_tmp = list(labels_tmp).index(cl)
                support[i] = counts_tmp[idx_tmp] / y_pred.shape[0]

        # DataFrame metrics
        df_stats = pd.DataFrame(columns=['Label', 'F1-Score', 'Accuracy',
                                         'Precision', 'Recall', 'Trues', 'Falses',
                                         'True positive', 'True negative',
                                         'False positive', 'False negative',
                                         'Condition positive', 'Condition negative',
                                         'Predicted positive', 'Predicted negative'])

        # Get statistics per class
        labels = self.list_classes
        for label in labels:
            label_str = str(label)  # Fix : If label is an int, can cause some problems (e.g. only zeroes in the confusion matrix)
            none_class = 'None' if label_str != 'None' else 'others'  # Check that the class is not already 'None'
            y_true_tmp = [label_str if _ == label else none_class for _ in y_true]
            y_pred_tmp = [label_str if _ == label else none_class for _ in y_pred]
            c_mat_tmp = confusion_matrix(y_true_tmp, y_pred_tmp, labels=[none_class, label_str])
            df_stats = df_stats.append(self._update_info_from_c_mat(c_mat_tmp, label, log_info=False), ignore_index=True)

        # Add global statistics
        global_stats = {
            'Label': 'All',
            'F1-Score': f1_weighted,
            'Accuracy': acc_tot,
            'Precision': precision_weighted,
            'Recall': recall_weighted,
            'Trues': trues,
            'Falses': falses,
            'True positive': None,
            'True negative': None,
            'False positive': None,
            'False negative': None,
            'Condition positive': None,
            'Condition negative': None,
            'Predicted positive': None,
            'Predicted negative': None,
        }
        df_stats = df_stats.append(global_stats, ignore_index=True)

        # Add support
        df_stats['Support'] = support

        # Return dataframe
        return df_stats

    def get_metrics_simple_multilabel(self, y_true, y_pred) -> pd.DataFrame:
        '''Gets metrics on multi-label predictions
        Same as the method get_and_save_metrics but without all the fluff (save, etc.)

        Args:
            y_true (?): Array-like, shape = [n_samples, n_targets]
            y_pred (?): Array-like, shape = [n_samples, n_targets]
        Raises:
            ValueError: If not with multi-labels tasks
        Returns:
            pd.DataFrame: The dataframe containing statistics
        '''
        if not self.multi_label:
            raise ValueError("The method get_metrics_simple_multilabel only works for multi-labels cases")

        # Cast to np.array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Gets global f1 score / acc_tot / trues / falses / precision / recall / support
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        trues = np.sum(np.all(np.equal(y_true, y_pred), axis=1))
        falses = len(y_true) - trues
        acc_tot = trues / len(y_true)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        support = list(pd.DataFrame(y_true).sum().values)
        support = [_ / sum(support) for _ in support] + [1.0]

        # DataFrame metrics
        df_stats = pd.DataFrame(columns=['Label', 'F1-Score', 'Accuracy',
                                         'Precision', 'Recall', 'Trues', 'Falses',
                                         'True positive', 'True negative',
                                         'False positive', 'False negative',
                                         'Condition positive', 'Condition negative',
                                         'Predicted positive', 'Predicted negative'])

        # Add metrics
        labels = self.list_classes
        # Details per category
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        for i, label in enumerate(labels):
            c_mat = mcm[i]
            df_stats = df_stats.append(self._update_info_from_c_mat(c_mat, label, log_info=False), ignore_index=True)

        # Add global statistics
        global_stats = {
            'Label': 'All',
            'F1-Score': f1_weighted,
            'Accuracy': acc_tot,
            'Precision': precision_weighted,
            'Recall': recall_weighted,
            'Trues': trues,
            'Falses': falses,
            'True positive': None,
            'True negative': None,
            'False positive': None,
            'False negative': None,
            'Condition positive': None,
            'Condition negative': None,
            'Predicted positive': None,
            'Predicted negative': None,
        }
        df_stats = df_stats.append(global_stats, ignore_index=True)

        # Add support
        df_stats['Support'] = support

        # Return dataframe
        return df_stats

    def _update_info_from_c_mat(self, c_mat: np.ndarray, label: str, log_info: bool = True) -> dict:
        '''Updates a dataframe for the method get_and_save_metrics, given a confusion matrix

        Args:
            c_mat (np.ndarray): Confusion matrix
            label (str): Label to use
        Kwargs:
            log_info (bool): If the statistics must be logged
        Returns:
            dict: Dictionary with the information for the update of the dataframe
        '''
        # Extract all needed info from c_mat
        true_negative = c_mat[0][0]
        true_positive = c_mat[1][1]
        false_negative = c_mat[1][0]
        false_positive = c_mat[0][1]
        condition_positive = false_negative + true_positive
        condition_negative = false_positive + true_negative
        predicted_positive = false_positive + true_positive
        predicted_negative = false_negative + true_negative
        trues_cat = true_negative + true_positive
        falses_cat = false_negative + false_positive
        accuracy = (true_negative + true_positive) / (true_negative + true_positive + false_negative + false_positive)
        precision = 0 if predicted_positive == 0 else true_positive / predicted_positive
        recall = 0 if condition_positive == 0 else true_positive / condition_positive
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        # Display some info
        if log_info:
            self.logger.info(
                f"F1-score: {round(f1, 5)}  \t Precision: {round(100 * precision, 2)}% \t"
                f"Recall: {round(100 * recall, 2)}% \t Trues: {trues_cat} \t Falses: {falses_cat} \t\t --- {label} "
            )

        # Return result
        return {
            'Label': f'{label}',
            'F1-Score': f1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Trues': trues_cat,
            'Falses': falses_cat,
            'True positive': true_positive,
            'True negative': true_negative,
            'False positive': false_positive,
            'False negative': false_negative,
            'Condition positive': condition_positive,
            'Condition negative': condition_negative,
            'Predicted positive': predicted_positive,
            'Predicted negative': predicted_negative,
        }

    def _plot_confusion_matrix(self, c_mat: np.ndarray, labels: list, type_data: str = '',
                               normalized: bool = False, subdir: Union[str, None] = None) -> None:
        '''Plots a confusion matrix

        Args:
            c_mat (np.ndarray): Confusion matrix
            labels (list): Labels to plot
        Kwargs:
            type_data (str): Type of dataset (validation, test, ...)
            normalized (bool): If the confusion matrix should be normalized
            subdir (str): Sub-directory for writing the plot
        '''

        # Get title
        if normalized:
            title = f"Normalized confusion matrix{' - ' + type_data if len(type_data) > 0 else ''}"
        else:
            title = f"Confusion matrix, without normalization{' - ' + type_data if len(type_data) > 0 else ''}"

        # Init. plot
        width = round(10 + 0.5 * len(c_mat))
        height = round(4 / 5 * width)
        fig, ax = plt.subplots(figsize=(width, height))

        # Plot
        if normalized:
            c_mat = c_mat.astype('float') / c_mat.sum(axis=1)[:, np.newaxis]
            sns.heatmap(c_mat, annot=True, fmt=".2f", cmap=plt.cm.Blues, ax=ax)
        else:
            sns.heatmap(c_mat, annot=True, fmt="d", cmap=plt.cm.Blues, ax=ax)

        # labels, title and ticks
        ax.set_xlabel('Predicted classes', fontsize=height * 2)
        ax.set_ylabel('Real classes', fontsize=height * 2)
        ax.set_title(title, fontsize=width * 2)
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
        plt.tight_layout()

        # Save
        plots_path = os.path.join(self.model_dir, 'plots')
        if subdir is not None:  # Add subdir
            plots_path = os.path.join(plots_path, subdir)
        file_name = f"{type_data + '_' if len(type_data) > 0 else ''}confusion_matrix{'_normalized' if normalized else ''}.png"
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        plt.savefig(os.path.join(plots_path, file_name))

        # Close figures
        plt.close('all')

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save model
        if json_data is None:
            json_data = {}

        json_data['list_classes'] = self.list_classes
        json_data['dict_classes'] = self.dict_classes
        json_data['multi_label'] = self.multi_label

        # Save
        super().save(json_data=json_data)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
