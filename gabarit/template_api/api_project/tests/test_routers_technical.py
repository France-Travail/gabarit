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


from fastapi.testclient import TestClient


def test_get_liveness(test_base_client: TestClient):
    """Test the technical route /liveness"""
    response = test_base_client.get("/tests/liveness")
    assert response.status_code == 200
    assert response.json() == {"alive": "ok"}


def test_get_readiness_ko(test_base_client: TestClient):
    """Test the technical route /readiness when the model is not loaded"""
    response = test_base_client.get("/tests/readiness")
    assert response.status_code == 200
    assert response.json() == {"ready": "ko"}


def test_get_readiness_ok(test_complete_client: TestClient):
    """Test the technical route /readiness when the model is fully loaded"""
    response = test_complete_client.get("/tests/readiness")
    assert response.status_code == 200
    assert response.json() == {"ready": "ok"}


def test_info(test_complete_client: TestClient):
    """Test the technical route /info"""
    response = test_complete_client.get("/tests/info")
    assert response.status_code == 200
    assert response.json()["application"] == "APP_TESTS"
