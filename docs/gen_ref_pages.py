"""Generate the code reference pages and navigation."""
import subprocess
import sys
import shutil
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
    """PIp install a package in the current env

    Args:
        editable (bool, optional): If True, install packages in a editable way. Defaults to True.
    """
    command = [sys.executable, "-m", "pip", "install"]
    if editable: command.append("-e")
    command += packages
    subprocess.check_call(command)


def generate_and_install_template(template_name: str, generate_function: Callable, **kwargs: dict):
    """Generate a template with gabarit and install it in the current env

    This is mandatory to be able to use mkdocstrings
    Cf. https://mkdocstrings.github.io/usage/

    Args:
        template_name (str): template name. It is used as template and package name.
        generate_function (Callable): function that render a template
    """
    generated_template_path: Path = kwargs["project_path"]

    # Remove previous generated template if it exists
    if generated_template_path.exists():
        shutil.rmtree(generated_template_path)

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

    # generate and install templates
    for template_name, generate_function, kwargs in TEMPLATES:
        generate_and_install_template(template_name, generate_function, **kwargs)

    # generate references
    for template_name, _, kwargs in TEMPLATES:
        generate_reference_template(template_name, kwargs["project_path"])


main()

with mkdocs_gen_files.open(f"{DIR_DOC_REFERENCE}/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(NAV.build_literate_nav())