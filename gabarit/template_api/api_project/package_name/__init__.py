import logging
import os

from .core.config import settings
from .core.logtools import get_pattern_log

level = settings.log_level

logger = logging.getLogger(__name__)
logger.setLevel(level)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatterJson = logging.Formatter(get_pattern_log())
ch.setFormatter(formatterJson)

logger.addHandler(ch)

# Filter some non essential dependency logs :
# words_n_fun
try:  # pragma: no cover
    from words_n_fun import logger as words_n_fun_logger

    words_n_fun_logger.setLevel(logging.ERROR)
except ImportError:  # pragma: no cover
    pass

# tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
