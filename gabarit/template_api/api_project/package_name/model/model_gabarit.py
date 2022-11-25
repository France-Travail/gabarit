import logging
import os
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseSettings

from .model_base import Model

try:
    {%- if gabarit_package %}
    from {{gabarit_package}} import utils as utils_gabarit
    from {{gabarit_package}}.models_training import utils_models
    {%- else %}
    from gabarit_package import utils as utils_gabarit
    from gabarit_package.models_training import utils_models
    {%- endif %}
except ImportError:
    raise ImportError("Package '{{gabarit_package}}' not found. Please install it.")

CURRENT_DIR = Path()

DEFAULT_DATA_DIR = CURRENT_DIR / "{{package_name}}-data"
DEFAULT_MODELS_DIR = CURRENT_DIR / "{{package_name}}-models"

logger = logging.getLogger(__name__)


class ModelSettings(BaseSettings):
    """Download settings"""

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
    def __init__(self, *args, **kwargs):
        """Init object"""
        super().__init__(*args, **kwargs)

    def predict(self, content: Any, **kwargs) -> Any:
        """Make a prediction thanks to the model"""
        return utils_models.predict(content, self._model, **kwargs)

    def _load_model(self, **kwargs):
        """Load a model in a gabarit fashion"""
        settings = ModelSettings(**kwargs)

        # Replace get_data_path method from gabarit.utils to use {{package_name}} data directory
        if hasattr(utils_gabarit, "get_data_path"):
            utils_gabarit.get_data_path = lambda: str(settings.data_dir.resolve())

        # Using is_path=True allow to specify a path instead of a folder relative
        # to {{gabarit_package}}.utils.DIR_PATH
        model, model_conf = utils_models.load_model(
            model_dir=settings.model_path, is_path=True
        )

        return model, model_conf

    @staticmethod
    def download_model(**kwargs) -> bool:
        """Download the model from an JFrog Artifactory repository"""
        settings = ModelSettings(**kwargs)

        model_path = settings.model_path

        # If the model already exists there is no need to download it
        if (
            not settings.redownload
            and model_path.is_dir()
            and next(model_path.iterdir(), False)
        ):
            logger.info(f"The model is already dowloaded : {model_path} already exists")
            return True

        # Create models directory if it doesn not exists
        models_dir = settings.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        # Download model from artifactory
        try:
            from artifactory import ArtifactoryPath
        except ImportError:
            raise ImportError(
                "module artifactory not found. please install it : pip install dohq-artifactory"
            )

        model_artifactory_path = ArtifactoryPath(
            settings.artifactory_model_url,
            auth=(settings.artifactory_user, settings.artifactory_password),
            verify=False,
        )
        model_archive_path = models_dir / model_artifactory_path.name

        logger.info(f"Downloading the model to : {model_path}")
        with model_archive_path.open("wb") as out:
            model_artifactory_path.writeto(out)

        # Unzip model
        shutil.unpack_archive(model_archive_path, model_path)
        logger.info(f"Model downloaded")

        # Remove model archive
        os.remove(model_archive_path)
        logger.info(f"Model archive removed")

        return True
