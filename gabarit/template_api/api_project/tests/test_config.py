from {{package_name}}.core.config import settings


def test_settings():
    """Verify that environment variables set in conftest.py are taken into 
    account in the application settings.
    """
    assert settings.app_name == "APP_TESTS"
    assert settings.api_entrypoint == "/tests"
