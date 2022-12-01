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

import json
import pytest
import numpy as np

from {{package_name}}.routers.schemas.utils import NumpyArrayEncoder


def test_numpy_encoder():
    """Test the NumpyArrayEncoder that is used by default to handle numpy objects"""
    obj = np.array([0.1, 0.2], dtype=np.float128)
    assert json.dumps(obj, cls=NumpyArrayEncoder) == "[0.1, 0.2]"

    obj = {0.1, 0.2}
    assert json.dumps(obj, cls=NumpyArrayEncoder) == "[0.1, 0.2]"

    obj = np.array([1, 2], dtype=np.int16)
    assert json.dumps(obj, cls=NumpyArrayEncoder) == "[1, 2]"

    obj = np.array(["a", "b"])
    assert json.dumps(obj, cls=NumpyArrayEncoder) == '["a", "b"]'

    with pytest.raises(TypeError):
        assert json.dumps(str, cls=NumpyArrayEncoder)
