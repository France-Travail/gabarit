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
    
    test_complete_client.app.state.model._model_explainer = TestExplainer()

    # HTML response
    response = test_complete_client.post(
        "/tests/explain", json={"content": ["gab", "gabarit"]}
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