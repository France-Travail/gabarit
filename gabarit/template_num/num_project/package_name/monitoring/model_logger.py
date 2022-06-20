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
# - ModelLogger -> Abstracts how MlFlow works


import os
import re
import math
import uuid
import mlflow
import socket
import logging
from typing import Callable, Union

# Deactivation of GIT warning for mlflow
os.environ["GIT_PYTHON_REFRESH"] = "quiet"


def is_running(host: str, port: int, logger: logging.Logger) -> bool:
    '''Checks if a host is up & running

    Args:
        host (str): URI of the host
        port (int): Port to check
        logger (logging.Logger): Logger of a  ModelLogger instance
    Returns:
        bool: If the host is reachable
    '''
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    reachable = False
    try:
        host = re.sub(r'(?i)http(s)*://', '', host)  # Remove http:// to test connexion
        sock.connect((host, port))
        sock.shutdown(socket.SHUT_RDWR)
        reachable = True
    except Exception:
        logger.error(f'Monitoring - MlFlow  @ {host} not reachable => nothing will be stored')
    finally:
        sock.close()

    # Return state
    return reachable


def is_local(host: str) -> bool:
    '''Checks if mlflow is running in local

    Args:
        host (str): URI of the host
    Returns:
        bool: If mlflow is running in local
    '''
    l1 = len(host)
    host = re.sub(r'(?i)http(s)*://', '', host)
    l2 = len(host)
    if l1 == l2:  # no http
        return True
    else:
        return False


def is_mlflow_up(func: Callable) -> Callable:
    '''Checks if mlflow server is up & running before calling the decorated function

    Args:
        func (Callable): Function to decorate
    Returns:
        Callable: Wrapper
    '''

    # Define wrapper to check if mlflow is up
    def wrapper(self, *args, **kwargs):

        # We run only if running == True (ie. initial connection ok)
        if self.running:

            # Check if we can run
            if is_local(self.tracking_uri):
                to_run = True  # OK because local
            elif is_running(self.tracking_uri, 80, self.logger):
                to_run = True  # OK because still running
            else:
                to_run = False  # KO

            # run if possible
            if to_run:
                try:
                    func(self, *args, **kwargs)
                except Exception as e:  # Manage mlflow errors (but continues process)
                    self.logger.error("Can't log on mlflow")
                    self.logger.error(repr(e))
            # Else : do nothing (error already logged)

    # For test purposes (ignore wrapper)
    wrapper.wrapped_fn = func  # type: ignore

    return wrapper


class ModelLogger:
    '''Abstracts how MlFlow works'''

    _default_name = f'{{package_name}}-approche-{uuid.uuid4()}'
    _default_tracking_uri = ''

    def __init__(self, tracking_uri: Union[str, None] = None, experiment_name: Union[str, None] = None) -> None:
        '''Class initialization

        Kwargs:
            tracking_uri (str): URI of the tracking server
            experiment_name (str): Name of the experiment to activate
        '''
        # Get logger
        self.logger = logging.getLogger(__name__)
        # Set tracking URI & experiment name
        self.tracking_uri = tracking_uri if tracking_uri is not None else self._default_tracking_uri
        self.experiment_name = experiment_name if experiment_name is not None else self._default_name
        # Initiate tracking
        # There is a try...except in order to test if mlflow is reachable
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(f'/{self.experiment_name}')
            self.logger.info(f'Ml Flow running, metrics available @ {self.tracking_uri}')
            self.running = True
        except Exception:
            self.logger.warning(f"Host {self.tracking_uri} is not reachable. ML flow won't run")
            self.logger.warning("Warning, for a local process, mlflow only accepts relative paths ...")
            self.logger.warning("Take care to use os.path.relpath()")
            self.running = False

    def stop_run(self) -> None:
        '''Stop an MLflow run'''
        try:
            mlflow.end_run()
        except Exception:
            self.logger.error("Can't stop mlflow run")

    @is_mlflow_up
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

    @is_mlflow_up
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

    @is_mlflow_up
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

    @is_mlflow_up
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

    @is_mlflow_up
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

    @is_mlflow_up
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
        return mlflow.mlflow.utils.validation._VALID_PARAM_AND_METRIC_NAMES.match(key)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
