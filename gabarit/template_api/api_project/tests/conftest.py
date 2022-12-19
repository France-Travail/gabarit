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


"""Test configuration

This conftest.py file is the first loaded by pytest when the tests are executed,
see the documentation for more infos :
https://docs.pytest.org/en/7.2.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files

We use it to :
- create a test model
- create a TestClient named test_base_client which does not load the test model
- create a TestClient named test_complete_client which does load the test model
- set some environment variables to check if they are well set in the app settings

> More details about test_base_client and test_complete_client:
>
> By default a TestClient does not fire events so the startup and shutdown events are never fired
> and the model is never loaded.
> We can fire thoses events by using TestClient as a context manager so we use two TestClient in
> our tests : a test_base_client that does not load the model and test_complete_client that does
> load the model thanks to a context manager. It allows us to test the behavior of our application
> when a model is not loaded (which should not happend).

"""

import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from .create_test_model import TestModel

# Set paths
TEST_DIR = Path(__file__).parent.resolve()
TEST_MODELS_DIR = TEST_DIR / "data" / "models"
TEST_MODEL_PATH = TEST_MODELS_DIR / "model.pkl"

# Set environment variables for testing
os.environ["app_name"] = "APP_TESTS"
os.environ["api_entrypoint"] = "/tests"
os.environ["model_path"] = str(TEST_MODEL_PATH)

# Create a test model if needed
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
