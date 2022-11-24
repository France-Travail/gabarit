import json

import numpy as np
import pytest

from {{package_name}}.routers.schemas.utils import NumpyArrayEncoder


def test_numpy_encoder():
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
