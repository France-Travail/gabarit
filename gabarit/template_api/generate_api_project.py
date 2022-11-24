#!/usr/bin/env python3.8

# Génération d'un template python
# Auteurs : Agence dataservices
# Date : 20/12/2019


import argparse
import os
import re
import shutil

from jinja2 import Environment, FileSystemLoader

VALID_ENTRY_POINT = re.compile(r'\w+')

def main():
    '''Generates a python template'''
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Project name")
    parser.add_argument("-p", "--path", required=True, help="Path (relative or absolute) to project directory")
    parser.add_argument("-e", "--env", help=".env file for your project")
    parser.add_argument("--gabarit_package", help="Gabarit package you whish to use")
    parser.add_argument("--gabarit_package_version", help="Gabarit package version you whish to use")
    args = parser.parse_args()

    generate(package_name=args.name, project_path=args.path, env_file=args.env,
             gabarit_package=args.gabarit_package, gabarit_package_version=args.gabarit_package_version)


def generate(package_name: str, project_path: str, env_file: str, gabarit_package: str, gabarit_package_version: str):
    '''Generates a API python template from arguments.

    Args:
        package_name (str): Name of the project to generate
        project_path (str): Path to the folder of the new project (which does not need to exist)
        env_file (str): .env file for project configuration
        gabarit_package (str) : model python package
        gabarit_package_version (str) : model python package version
    Raises:
        TypeError: if package_name is not of type str
        TypeError: if project_path is not of type str
        TypeError: if env_file is not of type str
        TypeError: if gabarit_package is not of type str
        TypeError: if gabarit_package_version is not of type str
    '''
    # Start by creating the output directory:
    output_dir = os.path.abspath(project_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get environment
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_project")
    env = Environment(loader=FileSystemLoader(template_dir))

    # For each template (a.k.a. files), load and fill it, then save into output dir
    for template_name in env.list_templates():
        print(f'Rendering {template_name}')

        # Get render
        template = env.get_template(template_name)
        render = template.render(package_name=package_name,
                                 gabarit_package=gabarit_package,
                                 gabarit_package_version=gabarit_package_version)

        # Replace package_name in file/directory
        template_name = template_name.replace("package_name", package_name)

        # Save it
        final_path = os.path.join(output_dir, template_name)
        basedir = os.path.dirname(final_path)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        with open(final_path, "w", newline='\n', encoding='utf8') as f:
            f.write(render)

    # Create data and models dir
    for data_dir in (
        os.path.join(output_dir, f'{package_name}-data'),
        os.path.join(output_dir, f'{package_name}-models'),
    ):
        for env_dir in [data_dir]:
            if not os.path.exists(env_dir):
                os.makedirs(env_dir)

    if env_file and os.path.exists(env_file):
        shutil.copy2(env_file, os.path.join(output_dir, '.env'))
    else:
        print(f"WARNING : {env_file} does not exists. The template .env file was used instead.")

if __name__ == "__main__":
    main()
