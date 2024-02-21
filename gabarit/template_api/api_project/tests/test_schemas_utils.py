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

import numpy as np
import orjson
import pytest

from {{package_name}}.routers.schemas import functional
from

def test_numpy_encoder():
    """Test the NumpyArrayEncoder that is used by default to handle numpy objects"""
    obj = np.array([0.1, 0.2], dtype=np.longdouble)
    assert orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY, default=functional.default).decode(
        'utf-8') == "[0.1,0.2]"

    obj = {0.1, 0.2}
    assert orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY, default=functional.default).decode(
        'utf-8') == "[0.1,0.2]"

    obj = np.array([1, 2], dtype=np.int16)
    assert orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY, default=functional.default).decode(
        'utf-8') == "[1,2]"

    obj = np.array(["a", "b"])
    assert orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY, default=functional.default).decode(
        'utf-8') == '["a","b"]'

    with pytest.raises(TypeError):
        assert orjson.dumps(str)
