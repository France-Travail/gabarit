import pickle
from itertools import zip_longest
from pathlib import Path


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
        