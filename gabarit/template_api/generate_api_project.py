#!/usr/bin/env python3
# Generates an API template
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


import os
import shutil
import tempfile
import argparse
from typing import List, Union
from pathlib import Path
from pkg_resources import Requirement
from jinja2 import Environment, FileSystemLoader


EXCLUDE_EXTS = {".pyc"}


def main():
    """Generates an API python template"""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Project name")
    parser.add_argument("-p", "--path", required=True, help="Path (relative or absolute) to project directory")
    parser.add_argument("--gabarit_package", help=("Gabarit package you plan to use in your API (let empty otherwise). "
                                                   "Example : my-gabarit-project[explicability]==0.1.2"))
    parser.add_argument("--gabarit_import_name", help=("The import name of your Gabarit package might be different from "
                                                       "the package name as with sickit-learn / sklearn. Use this argument "
                                                       "to specify a different import name"))
    parser.add_argument("-c", "--custom", nargs="+", default=[], help=("Add custom templates such as a custom .env file or a custom Dockerfile. "
                                                                       "Example : --custom /path/to/my/.env.custom=.env include/all/from/dir="))
    parser.add_argument("--gabarit_no_spec", action="store_true", help="Allow to specify a gabarit_package without any specification")
    args = parser.parse_args()

    generate(package_name=args.name, project_path=args.path, gabarit_package_spec=args.gabarit_package,
             gabarit_import_name=args.gabarit_import_name, gabarit_no_spec=args.gabarit_no_spec,
             custom_templates=args.custom)


def generate(package_name: str, project_path: str, gabarit_package_spec: Union[str, None] = None,
             gabarit_import_name: Union[str, None] = None, custom_templates: Union[List[str], None] = None,
             gabarit_no_spec: bool = False) -> None:
    """Generates an API python template from arguments.

    Args:
        package_name (str): Name of the project to generate
        project_path (str): Path to the folder of the new project (which does not need to exist)
    Kwargs:
        gabarit_package_spec (str) : Gabarit dependency
        gabarit_import_name (str) : Gabarit import name
        custom_templates (List[str]) : Custom templates or directories
    """
    # Python package names can not have : "-"
    package_name = package_name.replace("-", "_")
    
    # Parse Gabarit dependency
    if gabarit_package_spec is not None:
        # Parse Gabarit package specs.
        gabarit_requirement = Requirement.parse(gabarit_package_spec)
        gabarit_package_name = gabarit_requirement.project_name
        # Check for gabarit version
        if not gabarit_requirement.specs and not gabarit_no_spec:
            raise Exception(f"Please specify a gabarit_package version. "
                            f"Example : --gabarit_package '{gabarit_package_name}==0.1'")
        # If no import name specified, backup on package_name (with '-' replaces by '_')
        if gabarit_import_name is None:
            gabarit_import_name = gabarit_package_name.replace("-", "_")
            print(f"WARNING : no import name specified, backup on {gabarit_import_name}.")
    else:
        gabarit_package_name = None
        gabarit_package_spec = None
        gabarit_import_name = None

    # Start by creating the output directory:
    output_dir = os.path.abspath(project_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get template_api templates directory
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_project")

    # Reference custom templates files in a dict whose keys are custom template file paths
    # and values resulting file path in the generated project
    custom_templates_destinations = {}
    if custom_templates is None:
        custom_templates = []
    for custom_template in custom_templates:
        # Read arguments
        try:
            template_file, file_dest = custom_template.rsplit("=", 1)
        except ValueError:
            print(f"WARNING : wrong custom template format for '{custom_template}'. "
                  f"It should be like : path/to/custom/template=path/to/dest/file")
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

    # Render the new project -> all the process is made using a temporary folder
    # Idea : we will copy all the files that needs to be rendered, then we will render the whole folder at once
    with tempfile.TemporaryDirectory() as tmp_dirname:

        # Copy all custom templates to a temporary directory
        for file_dest, template_file in custom_templates_destinations.items():
            tmp_path_dest = os.path.join(tmp_dirname, file_dest)
            os.makedirs(os.path.dirname(tmp_path_dest), exist_ok=True)
            shutil.copy2(template_file, tmp_path_dest)

        # For each of main template directory and custom template directory, render templates and
        # put the results in the output directory
        for current_dir in [template_dir, tmp_dirname]:
            # Then create the Jinja env
            env = Environment(loader=FileSystemLoader(current_dir))

            # For each template (a.k.a. files), load and fill it, then save into output dir
            for template_name in env.list_templates():
                # Ignore some extensions
                if any(template_name.endswith(ext) for ext in EXCLUDE_EXTS):
                    continue

                # Get render
                template = env.get_template(template_name)
                render = template.render(package_name=package_name,
                                         gabarit_package_name=gabarit_package_name,
                                         gabarit_package_spec=gabarit_package_spec,
                                         gabarit_import_name=gabarit_import_name)

                # Replace package_name in file/directory
                template_name = template_name.replace("package_name", package_name)

                # Save it
                final_path = os.path.join(output_dir, template_name)
                basedir = os.path.dirname(final_path)
                if not os.path.exists(basedir):
                    os.makedirs(basedir)
                # Encoding scripts in utf-8 !
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(render)

    # Everything is rendered, we just need to create some subdirectories
    data_dir = os.path.join(output_dir, f'{package_name}-data')
    models_dir = os.path.join(output_dir, f'{package_name}-models')
    for new_dir in [data_dir, models_dir]:
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)


if __name__ == "__main__":
    main()
