from {{package_name}}.core.config import settings


def test_settings():
    assert settings.app_name == "APP_TESTS"
    assert settings.api_entrypoint == "/tests"
