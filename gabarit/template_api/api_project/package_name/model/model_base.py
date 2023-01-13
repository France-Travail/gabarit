#!/usr/bin/env python3
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


"""This module contains the base Model class

Model is the base model class. It contains a loading and downloading methods that are
used by default to download your model into your Docker container and load it into your
application.

To use a custom model class in your application, create a new module such as
model_awesome.py in this package and write a custom class that overwrite _load_model,
download_model or predict depending on your needs.
"""


import dill as pickle
import logging
from pathlib import Path
from pydantic import BaseSettings
from typing import Any, Tuple, Union


# Manage paths
CURRENT_DIR = Path()
DEFAULT_MODELS_DIR = CURRENT_DIR / "{{package_name}}-models"
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "model.pkl"

logger = logging.getLogger(__name__)


class ModelSettings(BaseSettings):
    """Download settings

    This class is used for settings management purpose, have a look at the pydantic
    documentation for more details : https://pydantic-docs.helpmanual.io/usage/settings/

    By default, it looks for environment variables (case insensitive) to set the settings
    if a variable is not found, it looks for a file name .env in your working directory
    where you can declare the values of the variables and finally it sets the values
    to the default ones you can see above.
    """

    model_path: Path = DEFAULT_MODEL_PATH

    class Config:
        env_file = ".env"


class Model:
    """Parent model class.

    This class is given as an exemple, you should probably adapt it to your project.
    This class loads the model from a .pkl file. The model must have a predict function.
    """

    def __init__(self):
        '''Init. model class'''
        self._model = None
        self._model_conf = None
        self._model_explainer = None
        self._loaded = False

    def is_model_loaded(self):
        """return the state of the model"""
        return self._loaded

    def loading(self, **kwargs):
        """load the model"""
        self._load_model(**kwargs)
        self._loaded = True

    def predict(self, *args, **kwargs):
        """Make a prediction thanks to the model"""
        return self._model.predict(*args, **kwargs)

    def explain_as_json(self, *args, **kwargs) -> Union[dict, list]:
        """Compute explanations about a prediction and return a JSON serializable object"""
        return self._model_explainer.explain_instance_as_json(*args, **kwargs)

    def explain_as_html(self, *args, **kwargs) -> str:
        """Compute explanations about a prediction and return an HTML report"""
        return self._model_explainer.explain_instance_as_html(*args, **kwargs)

    def _load_model(self, **kwargs) -> None:
        """Load a model from a file

        Returns:
            Tuple[Any, dict]: A tuple containing the model and a dict of metadata about it.
        """
        settings = ModelSettings(**kwargs)

        logger.info(f"Loading the model from {settings.model_path}")
        with settings.model_path.open("rb") as f:
            self._model = pickle.load(f)

        self._model_conf = {
            "model_path": settings.model_path.name,
            "model_name": settings.model_path.stem,
        }
        logger.info(f"Model loaded")

    @staticmethod
    def download_model(**kwargs) -> bool:
        """You should implement a download method to automatically download your model"""
        logger.info("The function download_model is empty. Implement it to automatically download your model.")
        return True
