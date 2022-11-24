from .config import settings


def get_pattern_log():
    return (
        "{"
        '"date": "%(asctime)s", '
        '"level": "%(levelname)s", '
        '"message": "%(message)s", '
        f'"version": "{settings.app_version}", '
        '"function": "File %(pathname)s, line %(lineno)d, in %(funcName)s", '
        '"logger": "%(name)s"'
        "}"
    )
