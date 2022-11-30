import argparse
from importlib import import_module

from fastapi.testclient import TestClient

parser = argparse.ArgumentParser()
parser.add_argument("package")


if __name__ == "__main__":
    args = parser.parse_args()
    package = args.package

    package_app = import_module(".application", package=package)

    with TestClient(package_app.app) as client:
        response = client.post(
            f"/{package}/rs/v1/explain",
            headers={"Accept": "application/json"},
            json={"content": "test text"},
        )
        print(response)
        assert response.status_code == 200
