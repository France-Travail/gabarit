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


import dill as pickle
from pathlib import Path
from itertools import zip_longest


class TestModel:
    """Simulate a model for testing purpose"""

    DETECT_WORD = "GABARIT"

    def predict(self, content, **kwargs) -> dict:
        """Simulate a predict method"""

        if isinstance(content, str):
            content = [content]

        return [
            {
                "probability": (
                    sum(x == y for x, y in zip(s.upper(), self.DETECT_WORD))
                    / max(len(s), len(self.DETECT_WORD))
                )
            }
            for s in content
        ]

    def to_pickle(self, path: Path):
        """Save model as a pickle file"""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)


class TestExplainer:
    """Simulate an explainer object"""

    def explain_instance_as_json(self, content, **kwargs) -> dict:
        """Simulate explain_instance_as_json"""

        if isinstance(content, str):
            content = [content]

        return [
            {
                "common_letters": "".join(
                    [
                        x if x == y else "_"
                        for x, y in zip_longest(
                            s.upper(), TestModel.DETECT_WORD, fillvalue=""
                        )
                    ]
                )
            }
            for s in content
        ]

    def explain_instance_as_html(self, content, **kwargs) -> str:
        """Simulate explain_instance_as_html"""

        if isinstance(content, str):
            content = [content]

        explanations = self.explain_instance_as_json(content, **kwargs)
        explanations_html = "\n".join(
            [
                f"<li>{s} : {explanation['common_letters']}</li>"
                for s, explanation in zip(content, explanations)
            ]
        )
        return f"""<!DOCTYPE html>
        <html>
        <body>
            <ul>
                {explanations_html}
            </ul>
        </body>
        </html>
        """
