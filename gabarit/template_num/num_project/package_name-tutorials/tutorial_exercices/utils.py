import inspect
import re
from io import BytesIO
from tokenize import COMMENT, tokenize, untokenize

from IPython import display

comment_pattern = re.compile(r"#")


def get_source(function, strip_comments=True) -> str:
    """Return source code from a function

    Args:
        function (_type_): function to get source code from
        strip_comments (bool, optional): Strip comments from source code. Defaults to True.

    Returns:
        str: python code as a string
    """
    code = inspect.getsource(function)

    if strip_comments:
        code_without_comments = []

        # We collect all non-comment tokens
        for token in tokenize(BytesIO(code.encode("utf-8")).readline):
            if token.type is not COMMENT:
                code_without_comments.append(token)

        # Then convert them back to code string
        code_without_comments = untokenize(code_without_comments).decode("utf-8")

        # And finally remove all blank lines
        code_without_comments_lines = [
            l.rstrip()
            for l in code_without_comments.split("\n")
            if not re.match(r"^[ \t]*$", l)
        ]

        return "\n".join(code_without_comments_lines)
    else:
        return code


def display_source(function, strip_comments=True) -> None:
    """Display source code of a function

    Args:
        function (_type_): function
        strip_comments (bool, optional): Strip comments from source code. Defaults to True.
    """
    return display.Code(
        get_source(function, strip_comments=strip_comments), language="python"
    )
