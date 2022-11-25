from {{package_name}}.model.model_base import Model


def test_download():
    """Call the download method from the base Model that does nothing"""
    assert Model.download_model()
