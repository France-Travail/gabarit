import pkg_resources
from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "{{package_name}}"
    app_version: str = pkg_resources.get_distribution("{{package_name}}").version
    api_entrypoint: str = "/{{package_name}}/rs/v1"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
