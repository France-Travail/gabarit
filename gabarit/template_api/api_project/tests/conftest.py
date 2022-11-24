import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from .create_test_model import TestModel

TEST_DIR = Path(__file__).parent.resolve()
TEST_MODELS_DIR = TEST_DIR / "data" / "models"
TEST_MODEL_PATH = TEST_MODELS_DIR / "model.pkl"

# Set environement variables for tests
os.environ["app_name"] = "APP_TESTS"
os.environ["api_entrypoint"] = "/tests"
os.environ["model_path"] = str(TEST_MODEL_PATH)

# Create a test model if needed
if not TEST_MODEL_PATH.exists():
    TestModel().to_pickle(TEST_MODEL_PATH)


@pytest.fixture(scope="session")
def test_base_client() -> TestClient:
    """Basic TestClient that do not run startup and shutdown events"""
    from {{package_name}}.application import app

    return TestClient(app)


@pytest.fixture()
def test_complete_client(monkeypatch) -> TestClient:
    """Complete TestClient that do run startup and shutdown events to load
    the model
    """
    from {{package_name}}.application import app
    from {{package_name}}.core import event_handlers
    from {{package_name}}.model.model_base import Model

    # Use base model for tests
    monkeypatch.setattr(event_handlers, "Model", Model)

    with TestClient(app) as client:
        yield client
