import logging
import pickle
from pathlib import Path
from typing import Any, Tuple

from pydantic import BaseSettings

CURRENT_DIR = Path()
DEFAULT_MODELS_DIR = CURRENT_DIR / "{{package_name}}-models"
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "model.pkl"

logger = logging.getLogger(__name__)


class ModelSettings(BaseSettings):
    """Download settings"""

    model_path: Path = DEFAULT_MODEL_PATH

    class Config:
        env_file = ".env"


class Model:
    def __init__(self):
        self._model = None
        self._model_conf = None
        self._loaded = False

    def is_model_loaded(self):
        """return the state of the model"""
        return self._loaded

    def loading(self, **kwargs):
        """load the model"""
        self._model, self._model_conf = self._load_model(**kwargs)
        self._loaded = True

    def predict(self, *args, **kwargs):
        """Make a prediction thanks to the model"""
        return self._model.predict(*args, **kwargs)


    def _load_model(self, **kwargs) -> Tuple[Any, dict]:
        """Load a model from a file

        Returns:
            Tuple[Any, dict]: A tuple containing the model and a dict of metadata about it.
        """
        settings = ModelSettings(**kwargs)

        logger.info(f"Loading the model from {settings.model_path}")
        with settings.model_path.open("rb") as f:
            model = pickle.load(f)

        logger.info(f"Model loaded")
        return model, {
            "model_path": settings.model_path.name,
            "model_name": settings.model_path.stem,
        }

    @staticmethod
    def download_model(**kwargs) -> bool:
        """You shloud implement a download method to automatically download your model"""

        logger.info(
            "The function download_model is empty. Implement it to automatically download your model."
        )
        return True
