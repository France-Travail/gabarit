#!/usr/bin/env python3
# Copyright (C) <2018-2022>  <Agence Data Services, DSI Pôle Emploi>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


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
