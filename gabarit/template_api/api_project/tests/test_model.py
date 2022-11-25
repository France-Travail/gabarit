from {{package_name}}.model.model_base import Model


def test_download():
    assert Model.download_model()
