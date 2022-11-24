"""Functional schemas"""

import json
from typing import Any

from starlette.responses import JSONResponse

from .utils import NumpyArrayEncoder


class PredictionResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(content, cls=NumpyArrayEncoder).encode()
