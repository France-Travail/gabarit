import pickle
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
