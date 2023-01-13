"""Generate the code reference pages and navigation."""
import os
import shutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Callable

import mkdocs_gen_files

from gabarit.template_api import generate_api_project
from gabarit.template_nlp import generate_nlp_project
from gabarit.template_num import generate_num_project
from gabarit.template_vision import generate_vision_project

DIR_DOCS = Path(__file__).resolve().parent
DIR_PROJECT = DIR_DOCS.parent
DIR_GABARIT = DIR_PROJECT / "gabarit"
DIR_DOT_DOCS = DIR_PROJECT / ".mkdocs"
DIR_GEN_TEMPLATES = DIR_DOT_DOCS / "templates"
DIR_DOC_REFERENCE = DIR_DOCS / "reference"

TEMPLATES = (
    ("template_api", generate_api_project.generate, {
        "package_name": "template_api",
        "project_path": DIR_GEN_TEMPLATES / "template_api"
    }),
    ("template_nlp", generate_nlp_project.generate, {
        "project_name": "template_nlp",
        "project_path": DIR_GEN_TEMPLATES / "template_nlp",
        "config_path": DIR_GABARIT / "template_nlp" / "default_config.ini",
        "upload_intructions_path": DIR_GEN_TEMPLATES / "template_nlp" / "instructions.md"
    }),
    ("template_num", generate_num_project.generate, {
        "project_name": "template_num",
        "project_path": DIR_GEN_TEMPLATES / "template_num",
        "config_path": DIR_GABARIT / "template_num" / "default_config.ini",
        "upload_intructions_path": DIR_GEN_TEMPLATES / "template_num" / "instructions.md"
    }),
    ("template_vision", generate_vision_project.generate, {
        "project_name": "template_vision",
        "project_path": DIR_GEN_TEMPLATES / "template_vision",
        "config_path": DIR_GABARIT / "template_vision" / "default_config.ini",
        "upload_intructions_path": DIR_GEN_TEMPLATES / "template_vision" / "instructions.md"
    }),
)

NAV = mkdocs_gen_files.Nav()
DOC_NO_REF = os.environ.get("DOC_NO_REF", "").lower()


def create_dot_mkdocs():
    """Create a .mkdocs fodler that will be ignored by git"""
    # Create .mkdocs/templates
    DIR_GEN_TEMPLATES.mkdir(parents=True, exist_ok=True)

    # Create .gitignore if it does not exists
    FILE_GITIGNORE = DIR_DOT_DOCS / ".gitignore"
    if not FILE_GITIGNORE.exists():
        with open(FILE_GITIGNORE, "w") as f:
            f.write("*")


def pip_install_packages(*packages, editable=True):
    """Pip install a package in the current env

    Args:
        editable (bool, optional): If True, install packages in a editable way. Defaults to True.
    """
    command = [sys.executable, "-m", "pip", "install"]
    if editable: command.append("-e")
    command += packages
    subprocess.check_call(command)


def get_source_package_tree(package_path: Path) -> list:
    """Return a package tree as a list of python files

    Args:
        package_path (Path): path to the package source folder

    Returns:
        list: package tree as a list of python files
    """
    return sorted([file_path.relative_to(package_path).as_posix() for file_path in package_path.rglob("**/*.py")])


def generate_and_install_template(template_name: str, generate_function: Callable, **kwargs: dict) -> bool:
    """Generate a template with gabarit and install it in the current env. Returns true if there is any change
    in the generated package source code

    This is mandatory to be able to use mkdocstrings
    Cf. https://mkdocstrings.github.io/usage/

    Args:
        template_name (str): template name. It is used as template and package name.
        generate_function (Callable): function that render a template

    Returns:
        bool: True if there is any change in the generated template. False otherwise.
    """
    generated_template_path: Path = kwargs["project_path"]

    # Remove previous generated template if it exists
    if generated_template_path.exists():
        package_tree = get_source_package_tree(generated_template_path / template_name)
        shutil.rmtree(generated_template_path)
    else:
        package_tree = None

    # Create a dummy upload_intructions_path for nlp, num and vision templates
    if "upload_intructions_path" in kwargs:
        upload_intructions_path = Path(kwargs["upload_intructions_path"])
        upload_intructions_path.parent.mkdir(parents=True, exist_ok=True)
        if not upload_intructions_path.exists():
            upload_intructions_path.touch()

    # Generate template
    generate_function(**kwargs)

    # Install generated template if not already installed (edit mode)
    try:
        import_module(template_name)
    except ModuleNotFoundError:
        pip_install_packages(generated_template_path)

    return get_source_package_tree(generated_template_path / template_name) != package_tree


def generate_reference_template(template_name: str, dir_template: Path):
    """Generate references doc files

    Based on recipe : https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages

    Args:
        template_name (str): template name. It is used as template and package name.
        dir_template (Path): directory containing the generated template.
    """
    # For each python file in the package directory of the template
    for path in sorted(dir_template.rglob(f"{template_name}/**/*.py")):

        # We get the module path relative to the package directory in a "import format"
        # ex : package.subpackage.module.py -> package.subpackage.module
        module_path = path.relative_to(dir_template).with_suffix("")

        # We create a markdown path "doc_path" relative to that module and a "full_doc_path"
        # relative to "reference" directory inside docs
        doc_path = path.relative_to(dir_template).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        parts = tuple(module_path.parts)

        # __init__.py are converted to index.md
        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")

        # __main__.py are ignored
        elif parts[-1] == "__main__":
            continue

        # the reference doc is added to mkdocs_gen_files.Nav()
        NAV[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}")

        mkdocs_gen_files.set_edit_path(full_doc_path, doc_path)


def main():
    """Main function that generate and install templates and create references doc"""
    # Create a .mkdocs to store generated templates
    create_dot_mkdocs()

    # generate and install templates will tracking changed templates
    changed_templates = [
        generate_and_install_template(template_name, generate_function, **kwargs)
        for template_name, generate_function, kwargs in TEMPLATES
    ]

    # generate references and add them to the NAV object
    for template_name, _, kwargs in TEMPLATES:
        generate_reference_template(template_name, kwargs["project_path"])

    # if at least one of the templates has changed, we rebuild the summary
    # otherwise we don't rebuild the summary since there is a conflict with live-reloading capability of mkdocs
    if any(changed_templates):
        DIR_DOC_REFERENCE.mkdir(exist_ok=True)
        with mkdocs_gen_files.open(f"{DIR_DOC_REFERENCE}/SUMMARY.md", "w") as reference_nav_file:
            reference_nav_file.writelines(NAV.build_literate_nav())


if DOC_NO_REF != "true":
    main()