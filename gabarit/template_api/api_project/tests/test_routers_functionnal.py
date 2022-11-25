import pytest
from fastapi.testclient import TestClient


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
