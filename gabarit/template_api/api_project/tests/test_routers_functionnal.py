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


import pytest
from fastapi.testclient import TestClient
from .create_test_model import TestExplainer


def test_predict(test_complete_client: TestClient):
    """Test the route predict thanks to the TestModel we created in conftest.py"""
    response = test_complete_client.post(
        "/tests/predict", json={"content": ["gab", "gabarit"]}
    )
    assert response.status_code == 200
    assert response.json() == [
        {"probability": pytest.approx(3 / 7)},
        {"probability": 1},
    ]

def test_explain(test_complete_client: TestClient):
    """Test the route explain"""
    # 501 HTML error
    response = test_complete_client.post(
        "/tests/explain",
        json={"content": ["gab", "gabarit"]},
    )
    assert response.status_code == 501

    # 501 JSON error
    response = test_complete_client.post(
        "/tests/explain",
        json={"content": ["gab", "gabarit"]},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 501

    # Now add a TestExplainer to test the explanations
    test_complete_client.app.state.model._model_explainer = TestExplainer()

    # HTML response
    response = test_complete_client.post(
        "/tests/explain",
        json={"content": ["gab", "gabarit"]},
    )
    assert response.status_code == 200
    assert "<li>gab : GAB____</li>" in response.text

    # JSON response
    response = test_complete_client.post(
        "/tests/explain",
        json={"content": ["gab", "gabarit"]},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    assert response.json() == [
        {"common_letters": "GAB____"},
        {"common_letters": "GABARIT"},
    ]
