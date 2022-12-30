#!/usr/bin/env python3

## Definition of a class to abstract how MlFlow works
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
# - MLflowLogger -> Abstracts how MlFlow works


import os
import math
import mlflow
import pathlib
import logging
import pandas as pd
from typing import Union

from .. import utils

# Deactivation of GIT warning for mlflow
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

class MLflowLogger:
    '''Abstracts how MlFlow works'''

    def __init__(self, experiment_name: str, tracking_uri: str = '', artifact_uri: str = '') -> None:
        '''Class initialization
        Args:
            experiment_name (str):  Name of the experiment to activate
        Kwargs:
            tracking_uri (str): URI of the tracking server
            artifact_uri (str): URI where to store artifacts
        '''
        # Get logger
        self.logger = logging.getLogger(__name__)

        # Backup to local save if no uri (i.e. empty string)
        if not tracking_uri:
            tracking_uri = pathlib.Path(os.path.join(utils.get_data_path(), 'experiments', 'mlruns')).as_uri()
        
        if not artifact_uri:
            artifact_uri = pathlib.Path(os.path.join(utils.get_data_path(), 'experiments', 'mlruns_artifacts')).as_uri()

        # Set tracking URI & experiment name
        self.tracking_uri = tracking_uri
        # No need to set_tracking_uri, this is done through the setter decorator
        self.experiment_name = experiment_name
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
        except Exception as e:
            self.logger.error(repr(e))
            raise ConnectionError(f"Can't reach MLflow at {self.tracking_uri}. Please check the URI.")

        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
            mlflow.set_experiment(experiment_id=experiment_id)

        self.logger.info(f'MLflow running, metrics available @ {self.tracking_uri}')

    @property
    def tracking_uri(self) -> str:
        '''Current tracking uri'''
        return mlflow.get_tracking_uri()

    @tracking_uri.setter
    def tracking_uri(self, uri:str) -> None:
        '''Set tracking uri'''
        mlflow.set_tracking_uri(uri)

    def end_run(self) -> None:
        '''Stops an MLflow run'''
        try:
            mlflow.end_run()
        except Exception:
            self.logger.error("Can't stop mlflow run")

    def log_metric(self, key: str, value, step: Union[int, None] = None) -> None:
        '''Logs a metric on mlflow

        Args:
            key (str): Name of the metric
            value (float, ?): Value of the metric
        Kwargs:
            step (int): Step of the metric
        '''
        # Check for None
        if value is None:
            value = math.nan
        # Log metric
        mlflow.log_metric(key, value, step)

    def log_metrics(self, metrics: dict, step: Union[int, None] = None) -> None:
        '''Logs a set of metrics in mlflow

        Args:
            metrics (dict): Metrics to log
        Kwargs:
            step (int): Step of the metric
        '''
        # Check for Nones
        for k, v in metrics.items():
            if v is None:
                metrics[k] = math.nan
        # Log metrics
        mlflow.log_metrics(metrics, step)

    def log_param(self, key: str, value) -> None:
        '''Logs a parameter in mlflow

        Args:
            key (str): Name of the parameter
            value (str, ?): Value of the parameter (which will be cast to str if not already of type str)
        '''
        if value is None:
            value = 'None'
        # Log parameter
        mlflow.log_param(key, value)

    def log_params(self, params: dict) -> None:
        '''Logs a set of parameters in mlflow

        Args:
            params (dict): Name and value of each parameter
        '''
        # Check for Nones
        for k, v in params.items():
            if v is None:
                params[k] = 'None'
        # Log parameters
        mlflow.log_params(params)

    def set_tag(self, key: str, value) -> None:
        '''Logs a tag in mlflow

        Args:
            key (str): Name of the tag
            value (str, ?): Value of the tag (which will be cast to str if not already of type str)
        Raises:
            ValueError: If the object value is None
        '''
        if value is None:
            raise ValueError('value must not be None')
        # Log tag
        mlflow.set_tag(key, value)

    def set_tags(self, tags: dict) -> None:
        '''Logs a set of tags in mlflow

        Args:
            tags (dict): Name and value of each tag
        '''
        # Log tags
        mlflow.set_tags(tags)

    def valid_name(self, key: str) -> bool:
        '''Validates key names

        Args:
            key (str): Key to check
        Returns:
            bool: If key is a valid mlflow key
        '''
        if mlflow.mlflow.utils.validation._VALID_PARAM_AND_METRIC_NAMES.match(key):
            return True
        else:
            return False

    def log_df_stats(self, df_stats: pd.DataFrame, label_col: str = 'Label') -> None:
        '''Log a dataframe containing metrics from a training

        Args:
            df_stats (pd.Dataframe): Dataframe containing metrics from a training
        Kwargs:
            label_col (str): default labelc column name
        '''
        if label_col not in df_stats.columns:
            raise ValueError(f"The provided label column name ({label_col}) not found in df_stats' columns.")

        # Get metrics columns
        metrics_columns = [col for col in df_stats.columns if col != label_col]

        # Log labels
        labels = df_stats[label_col].values
        for i, label in enumerate(labels):  # type: ignore
            self.log_param(f'Label {i}', label)

        # Log metrics
        ml_flow_metrics = {}
        for i, row in df_stats.iterrows():
            for j, col in enumerate(metrics_columns):
                metric_key = f"{row[label_col]} --- {col}"
                # Check that mlflow accepts the key, otherwise, replace it
                # TODO: could be improved ...
                if not self.valid_name(metric_key):
                    metric_key = f"Label {i} --- {col}"
                if not self.valid_name(metric_key):
                    metric_key = f"{row[label_col]} --- Col {j}"
                if not self.valid_name(metric_key):
                    metric_key = f"Label {i} --- Col {j}"
                ml_flow_metrics[metric_key] = row[col]

        # Log metrics
        self.log_metrics(ml_flow_metrics)

    def log_dict(self, dictionary: dict, artifact_file: str) -> None:
        '''Logs a dictionary as an artifact in MLflow

        Args:
            dictionary (dict): A dictionary
            artifact_file (str): The run-relative artifact file path in posixpath format to which the dictionary is saved
        '''
        mlflow.log_dict(dictionary=dictionary, artifact_file=artifact_file)

    def log_text(self, text: str, artifact_file: str) -> None:
        '''Logs a text as an artifact in MLflow

        Args:
            text (str): A text
            artifact_file (str): The run-relative artifact file path in posixpath format to which the dictionary is saved
        '''
        mlflow.log_text(text=text, artifact_file=artifact_file)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
