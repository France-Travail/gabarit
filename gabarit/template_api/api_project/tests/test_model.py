from {{package_name}}.model.model_base import download_model


def test_download():
    assert download_model()
