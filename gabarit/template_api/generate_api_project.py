#!/usr/bin/env python3.8

# Génération d'un template python
# Auteurs : Agence dataservices
# Date : 20/12/2019


import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader


def main():
    """Generates a python template"""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Project name")
    parser.add_argument("-p", "--path", required=True, help="Path (relative or absolute) to project directory")
    parser.add_argument("--gabarit_package", help="Gabarit package you want to use")
    parser.add_argument("--gabarit_package_version", help="Gabarit package version you whish to use")
    parser.add_argument(
        "-c",
        "--custom",
        nargs="+",
        help=(
            "Add custom templates such as a custom .env file or a custom Dockerfile. "
            "Example : --custom /path/to/my/.env.custom=.env include/all/from/dir="
        ),
    )
    args = parser.parse_args()

    generate(
        package_name=args.name,
        project_path=args.path,
        gabarit_package=args.gabarit_package,
        gabarit_package_version=args.gabarit_package_version,
        custom_templates=args.custom,
    )


def generate(
    package_name: str,
    project_path: str,
    gabarit_package: str,
    gabarit_package_version: str,
    custom_templates: List[str],
):
    """Generates a API python template from arguments.

    Args:
        package_name (str): Name of the project to generate
        project_path (str): Path to the folder of the new project (which does not need to exist)
        gabarit_package (str) : model python package
        gabarit_package_version (str) : model python package version
        custom_templates (List[str]) : custom templates or directories

    Raises:
        TypeError: if package_name is not of type str
        TypeError: if project_path is not of type str
        TypeError: if gabarit_package is not of type str
        TypeError: if gabarit_package_version is not of type str
    """
    # Start by creating the output directory:
    output_dir = os.path.abspath(project_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get template_api templates directory
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_project")

    # Reference custom templates files in a dict whose keys are custom template file paths
    # and values resulting file path in the generated project
    custom_templates_destinations = {}

    for custom_template in custom_templates:
        try:
            template_file, file_dest = custom_template.rsplit("=", 1)
        except ValueError:
            print(
                f"WARNING : wrong custom template format for '{custom_template}'. "
                f"It should be like : path/to/custom/template=path/to/dest/file"
            )

        # A custom template can be a file...
        if os.path.isfile(template_file):
            custom_templates_destinations[file_dest] = template_file

        # ... or a directory. In this case we recursively add all files in the directory
        # to the custom_templates_destinations dict :
        #
        # rglob return all files and directories from template_file, the filter keeps only files
        # then the map transform Path objects to a str representing path relative to the custom
        # template directory
        elif os.path.isdir(template_file):
            for file_path in map(
                lambda p: str(p.relative_to(template_file)),
                filter(lambda p: p.is_file(), Path(template_file).rglob("**/*")),
            ):
                path_dest = os.path.join(file_dest, file_path)
                path_origin = os.path.join(template_file, file_path)

                custom_templates_destinations[path_dest] = os.path.join(path_origin)
        else:
            print(f"WARNING : custom template or directory '{template_file}' does not exists. Custom template ignored.")

    with tempfile.TemporaryDirectory() as tmpdirname:

        # Copy all custom templates to a temporary directory
        for file_dest, template_file in custom_templates_destinations.items():
            tmp_path_dest = os.path.join(tmpdirname, file_dest)

            os.makedirs(os.path.dirname(tmp_path_dest), exist_ok=True)

            shutil.copy2(template_file, tmp_path_dest)

        # For each of main template directory and custom template directory, render templates and
        # put the results in the output directory
        for dir in [template_dir, tmpdirname]:
            # Then create the Jinja env
            env = Environment(loader=FileSystemLoader(dir))

            # For each template (a.k.a. files), load and fill it, then save into output dir
            for template_name in env.list_templates():
                print(f"Rendering {template_name}")

                # Get render
                template = env.get_template(template_name)
                render = template.render(
                    package_name=package_name,
                    gabarit_package=gabarit_package,
                    gabarit_package_version=gabarit_package_version,
                )

                # Replace package_name in file/directory
                template_name = template_name.replace("package_name", package_name)

                # Save it
                final_path = os.path.join(output_dir, template_name)
                basedir = os.path.dirname(final_path)
                if not os.path.exists(basedir):
                    os.makedirs(basedir)
                with open(final_path, "w", newline="\n", encoding="utf8") as f:
                    f.write(render)

    # Create data and models dir
    for data_dir in (
        os.path.join(output_dir, f"{package_name}-data"),
        os.path.join(output_dir, f"{package_name}-models"),
    ):
        for env_dir in [data_dir]:
            if not os.path.exists(env_dir):
                os.makedirs(env_dir)


if __name__ == "__main__":
    main()
