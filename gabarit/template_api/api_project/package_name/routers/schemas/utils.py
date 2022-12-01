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
import numpy as np
from typing import Any


class NumpyArrayEncoder(json.JSONEncoder):
    """JSONEncoder to store python dict or list containing numpy arrays"""

    def default(self, obj: Any) -> Any:
        """Transform numpy arrays into JSON serializable object such as list
        see : https://docs.python.org/3/library/json.html#json.JSONEncoder.default
        """

        # numpy.ndarray have dtype, astype and tolist attribute and methods that we want
        # to use to convert their element into JSON serializable objects
        if hasattr(obj, "dtype") and hasattr(obj, "astype") and hasattr(obj, "tolist"):

            if np.issubdtype(obj.dtype, np.integer):
                return obj.astype(int).tolist()
            elif np.issubdtype(obj.dtype, np.number):
                return obj.astype(float).tolist()
            else:
                return obj.tolist()

        # sets are not json serializable
        elif isinstance(obj, set):
            return list(obj)

        return json.JSONEncoder.default(self, obj)
