import json
from typing import Any

import numpy as np


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
