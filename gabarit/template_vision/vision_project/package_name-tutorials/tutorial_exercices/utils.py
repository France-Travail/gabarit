import inspect
import os
import re
from io import BytesIO
from pathlib import Path
from tokenize import COMMENT, tokenize, untokenize
from typing import List
from urllib.parse import urlparse
from urllib.request import urlretrieve

import requests
from IPython import display
from PIL import Image, UnidentifiedImageError
from {{package_name}}.utils import get_data_path
from tqdm import tqdm

DATA_PATH = get_data_path()
CLASSIF_DATA_URL = (
    "https://api.github.com/repos/OSS-Pole-Emploi/"
    "gabarit/contents/gabarit/template_vision/vision_data/dataset_v3"
)
OBJ_DETECT_DATA_URL = (
    "https://api.github.com/repos/OSS-Pole-Emploi/"
    "gabarit/contents/gabarit/template_vision/vision_data/dataset_object_detection"
)

IMAGE_EXTS = {
    ex[1:] for ex, f in Image.registered_extensions().items() if f in Image.OPEN
}

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


def download_file(
    url: str,
    output_path: str = None,
    overwrite: bool = False,
    verify_image: bool = True,
    nb_retry: int = 2,
) -> None:
    """Download file and URL"""

    if output_path is None:
        filename = url.rsplit("/", 1)[-1]
        filename = filename.rsplit(".", 1)[0]
        output_path = os.path.join(get_data_path(), filename)

    if isinstance(output_path, Path):
        ext = output_path.name.rsplit(".", 1)[-1]
    else:
        ext = output_path.rsplit(".", 1)[-1]

    if os.path.exists(output_path) and not overwrite:
        return None

    if not verify_image or ext not in IMAGE_EXTS:
        nb_retry = 0
    else:
        nb_retry = max(0, nb_retry)

    while nb_retry >= 0:
        urlretrieve(url, filename=output_path)

        if verify_image and ext in IMAGE_EXTS:
            try:
                with Image.open(output_path):
                    break
            except UnidentifiedImageError:
                if nb_retry > 0:
                    print(f"Download {url} failed. Number of retry : {nb_retry}")
                else:
                    print(f"WARNING : {url} could not be downloaded.")
                    os.remove(output_path)
                pass

        nb_retry -= 1


def get_relative_path(path: str, base_path: str):
    """Get path relative to a base_path"""
    assert path.startswith(base_path)
    relative_path = path[len(base_path) :]
    return relative_path if relative_path[:1] != "/" else relative_path[1:]


def verify_github_api_url(github_api_url: str) -> tuple:
    """Verify that the given url is a api.github.com url using contents endpoint
    of repositories api

    Args:
        github_api_url (str): github api url

    Returns:
        tuple: (user, repo, repo_path)
    """
    parsed_url = urlparse(github_api_url)

    # Verify that the given url is a api.github.com url using contents endpoint
    # of repositories api
    assert (
        parsed_url.netloc == "api.github.com"
    ), "This function is meant to be used with api.github.com"

    _, api, user, repo, api_endpoint, repo_path = parsed_url.path.split("/", 5)

    assert api == "repos" and api_endpoint == "contents", (
        "This function is meant to be used with : "
        "https://docs.github.com/en/rest/repos/contents#get-repository-content"
    )

    return user, repo, repo_path


def github_folder_tree(
    github_api_url: str, files_extensions: list = None, base_path: str = None
) -> List[dict]:
    """List all files contained in a directory thanks to api.github.com

    Args:
        github_api_url (str): An api.github url
        files_extensions (list): List of files extensions that files must match
        base_path (str, optional): Used to return path relative to this base path.
            Defaults to last folder in URL.

    Returns:
        List[dict]: List of dict containing file paths and download URLs
    """
    _, _, repo_path = verify_github_api_url(github_api_url)

    if base_path is None:
        base_path = repo_path.rsplit("/", 1)[0]

    files = []
    elements = requests.get(github_api_url).json()

    if isinstance(elements, dict):
        elements = [elements]

    for element in elements:
        element_type = element.get("type", None)
        element_url = element.get("url", None)
        element_path = element.get("path", None)
        element_dl_url = element.get("download_url", None)

        if element_type == "dir" and element_url:
            files += github_folder_tree(
                element_url, files_extensions=files_extensions, base_path=base_path
            )

        elif element_type == "file" and element_dl_url:

            # Checking file extension if needed
            if (
                files_extensions
                and element_path.rsplit(".", 1)[-1] not in files_extensions
            ):
                continue

            # Append file path and download url
            files.append(
                {
                    "path": get_relative_path(element_path, base_path),
                    "download_url": element_dl_url,
                }
            )

    return files


def github_download_folder(
    github_url: str,
    dest_dir: str,
    files_extensions: list = None,
    overwrite: bool = False,
) -> None:
    """Download a github repo folder"""
    files = github_folder_tree(github_url, files_extensions=files_extensions)

    dest_path = Path(dest_dir)
    dest_path.parent

    for file in tqdm(files):
        file_path = dest_path / file["path"]

        os.makedirs(file_path.parent, exist_ok=True)

        download_file(file["download_url"], file_path, overwrite=overwrite)


def github_download_classification_dataset(overwrite=False):
    """Download the vision dataset for classification"""
    return github_download_folder(CLASSIF_DATA_URL, DATA_PATH, overwrite=overwrite)


def github_download_object_detection_dataset(overwrite=False):
    """Download the vision dataset for object detection"""
    return github_download_folder(OBJ_DETECT_DATA_URL, DATA_PATH, overwrite=overwrite)
