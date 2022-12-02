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


"""This module contains a ModelGabarit class you can use for your gabarit generated
projects

ModelGabarit overwrite some methods of the base Model class :
    - download_model method to download a model from a JFrog Artifactory repository ;
    {%- if gabarit_package_spec %}
    - _load_model method to use the {{gabarit_import_name}}.models_training.utils_models.load_model
    function from a typical gabarit project ;
    - predict method to use the the {{gabarit_import_name}}.models_training.utils_models.predict
    function from a typical gabarit project.
    {%- else %}
    - _load_model method to use the gabarit_package.models_training.utils_models.load_model
    function from a typical gabarit project ;
    - predict method to use the the gabarit_package.models_training.utils_models.predict
    function from a typical gabarit project.
    {%- endif %}

"""


import logging
import shutil
import tempfile
import pandas as pd
from pathlib import Path
from typing import Any, Union
from pydantic import BaseSettings

from .model_base import Model

{%- if gabarit_package_spec %}
try:
    from {{gabarit_import_name}} import utils as utils_gabarit
    from {{gabarit_import_name}}.models_training import utils_models
    from {{gabarit_import_name}}.monitoring.model_explainer import Explainer
except ImportError:
    raise ImportError("Package '{{gabarit_package_name}}' not found. Please install it.")
{%- else %}
try:
    from gabarit_package import utils as utils_gabarit
    from gabarit_package.models_training import utils_models
    from gabarit_package.monitoring.model_explainer import Explainer
except ImportError:
    raise ImportError("Package 'gabarit_package' not found. Please install it.")
{%- endif %}



# Manage paths
CURRENT_DIR = Path()
DEFAULT_DATA_DIR = CURRENT_DIR / "{{package_name}}-data"
DEFAULT_MODELS_DIR = CURRENT_DIR / "{{package_name}}-models"

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

    data_dir: Path = DEFAULT_DATA_DIR
    models_dir: Path = DEFAULT_MODELS_DIR
    model_path: Path = DEFAULT_MODELS_DIR / "model"
    artifactory_model_url: str = ""
    artifactory_user: str = ""
    artifactory_password: str = ""
    redownload: bool = False

    class Config:
        env_file = ".env"


class ModelGabarit(Model):
    """Model class for a Gabarit generated project

    - download_model has been redefined to download a model from artifactory based on
    the settings : ARTIFACTORY_MODEL_URL, ARTIFACTORY_USER, ARTIFACTORY_PASSWORD
    - _load_model has been redefined to use utils_models.load_model
    - predict has been redefined to use utils_models.predict
    """
    def __init__(self, *args, **kwargs):
        """Object initialization
        By default, it initialize the attributes _model, _model_config and _loaded

        see the parent __init__ method in {{package_name}}.model.model_base.Model
        """
        super().__init__(*args, **kwargs)

    def predict(self, content: Any, *args, **kwargs) -> Any:
        """Make a prediction by calling utils_models.predict with the loaded model"""
        if isinstance(content, list) or isinstance(content, dict):
            content = pd.DataFrame(content)

        return utils_models.predict(content, model=self._model, model_conf=self._model_conf, **kwargs)

    def explain_as_json(self, content: Any, *args, **kwargs) -> Union[dict, list]:
        """Compute explanations about a prediction and return a JSON serializable object"""
        if isinstance(content, list) or isinstance(content, dict):
            content = pd.DataFrame(content)

        return self._model_explainer.explain_instance_as_json(content, *args, **kwargs)

    def explain_as_html(self, content: Any, *args, **kwargs) -> str:
        """Compute explanations about a prediction and return an HTML report"""
        if isinstance(content, list) or isinstance(content, dict):
            content = pd.DataFrame(content)

        return self._model_explainer.explain_instance_as_html(content, *args, **kwargs)

    def _load_model(self, **kwargs) -> None:
        """Load a model in a gabarit fashion"""
        settings = ModelSettings(**kwargs)

        # Replace get_data_path method from gabarit.utils to use {{package_name}} data directory
        if hasattr(utils_gabarit, "get_data_path"):
            utils_gabarit.get_data_path = lambda: str(settings.data_dir.resolve())

        # Using is_path=True allow to specify a path instead of a folder relative
        # to gabarit_package.utils.DIR_PATH
        model, model_conf = utils_models.load_model(model_dir=settings.model_path, is_path=True)

        # Set attributes
        self._model = model
        self._model_conf = model_conf

        # Create a model explainer
        self._model_explainer = Explainer(model=model, model_conf=model_conf)

    @staticmethod
    def download_model(**kwargs) -> bool:
        """Download the model from an JFrog Artifactory repository"""
        settings = ModelSettings(**kwargs)

        model_path = settings.model_path

        # If the model already exists there is no need to download it
        if not settings.redownload and model_path.is_dir() and not any(model_path.iterdir()):
            logger.info(f"The model is already dowloaded : {model_path} already exists")
            return True

        # Create models directory if it doesn not exists
        models_dir = settings.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        # Download model from artifactory
        try:
            from artifactory import ArtifactoryPath
        except ImportError:
            raise ImportError("Module artifactory not found. Please install it : `pip install dohq-artifactory`")

        model_artifactory_path = ArtifactoryPath(
            settings.artifactory_model_url,
            auth=(settings.artifactory_user, settings.artifactory_password),
            verify=False,
        )

        with tempfile.TemporaryDirectory(dir=models_dir) as tmpdir:
            model_archive_path = Path(tmpdir) / model_artifactory_path.name

            # Download model
            logger.info(f"Downloading the model to : {model_path}")
            with model_archive_path.open("wb") as out:
                model_artifactory_path.writeto(out)

            # Unzip model
            shutil.unpack_archive(model_archive_path, model_path)
            logger.info(f"Model downloaded")

        logger.info(f"Model archive removed")
        return True
