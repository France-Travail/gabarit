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
