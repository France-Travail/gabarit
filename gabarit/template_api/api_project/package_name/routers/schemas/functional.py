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


"""Functional schemas"""

from typing import Any

import numpy as np
import orjson
from starlette.responses import JSONResponse


class NumpyJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY, default=default)


def default(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "dtype") and hasattr(obj, "astype") and hasattr(obj, "tolist"):
        if np.issubdtype(obj.dtype, np.integer):
            return obj.astype(int).tolist()
        elif np.issubdtype(obj.dtype, np.number):
            return obj.astype(float).tolist()

    raise TypeError
