#!/usr/bin/env python3

## Generation of a computer vision python template
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
import argparse
import tempfile
import configparser
from typing import Union
from shutil import copyfile
from distutils.dir_util import copy_tree
from jinja2 import Environment, FileSystemLoader

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    '''Generates a python template'''
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True, help='Project name')
    parser.add_argument('-p', '--path', required=True, help='Path (relative or absolute) to project directory')
    parser.add_argument('-c', '--config', default=os.path.join(ROOT_DIR, 'default_config.ini'), help='Configuration file to use (relative or absolute).')
    parser.add_argument('--upload', '--upload_intructions', default=os.path.join(ROOT_DIR, 'default_model_upload_instructions.md'),
                        help='Markdown file with models upload instructions (relative or absolute).')
    parser.add_argument('--dvc', '--dvc_config', default=None, help='DVC configuration file to use (relative or absolute).')
    args = parser.parse_args()
    # Generate project
    generate(project_name=args.name, project_path=args.path, config_path=args.config, upload_intructions_path=args.upload, dvc_config_path=args.dvc)


def generate(project_name: str, project_path: str, config_path: str,
             upload_intructions_path: str, dvc_config_path: Union[str, None] = None) -> None:
    '''Generates a Computer Vision python template from arguments.

    Args:
        project_name (str): Name of the project to generate
        project_path (str): Path to the folder of the new project (which does not need to exist)
        config_path (str): Configuration filepath
        upload_intructions_path (str): Models upload instructions filepath
            The value `model_dir_path_identifier` will be automatically updated for each model with its directory path
    Kwargs:
        dvc_config_path (str): DVC configuration filepath
    Raises:
        FileNotFoundError: If configuration path does not exists
        FileNotFoundError: If upload instructions path does not exists
        FileNotFoundError: If DVC configuration path does not exists
    '''
    # Check input files path
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Filepath {config_path} does not exist")
    if not os.path.exists(upload_intructions_path):
        raise FileNotFoundError(f"Filepath {upload_intructions_path} does not exist")
    if dvc_config_path is not None and not os.path.exists(dvc_config_path):
        raise FileNotFoundError(f"Filepath {dvc_config_path} does not exist")

    # Start by creating the output directory:
    output_dir = os.path.abspath(project_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Retrieve configurations
    config = configparser.ConfigParser(comment_prefixes=';')
    config.read(config_path)

    # For each config, returns None if no value given
    def get_config(config, section, key, fallback=None):
        value = config.get(section, key, fallback=fallback)
        return value if value != '' else None
    default_sep = get_config(config, 'files', 'csv_sep', fallback=None)
    default_encoding = get_config(config, 'files', 'encoding', fallback=None)
    pip_trusted_host = get_config(config, 'pip', 'trusted-host', fallback=None)
    pip_index_url = get_config(config, 'pip', 'index-url', fallback=None)
    mlflow_tracking_uri = get_config(config, 'mlflow', 'tracking_uri', fallback=None)
    additional_pip_packages = get_config(config, 'packages', 'additional_pip_packages', fallback=None)
    vgg16_weights_backup_urls = get_config(config, 'transfer_learning', 'vgg16_weights_backup_urls', fallback=None)
    efficientnetb6_weights_backup_urls = get_config(config, 'transfer_learning', 'efficientnetb6_weights_backup_urls', fallback=None)
    detectron_config_base_backup_urls = get_config(config, 'detectron', 'detectron_config_base_backup_urls', fallback=None)
    detectron_config_backup_urls = get_config(config, 'detectron', 'detectron_config_backup_urls', fallback=None)
    detectron_model_backup_urls = get_config(config, 'detectron', 'detectron_model_backup_urls', fallback=None)
    dvc_config_ok = True if dvc_config_path is not None else False

    # Fix some options that should be list of elements
    vgg16_weights_backup_urls = vgg16_weights_backup_urls.split('\n') if vgg16_weights_backup_urls is not None else None
    efficientnetb6_weights_backup_urls = efficientnetb6_weights_backup_urls.split('\n') if efficientnetb6_weights_backup_urls is not None else None
    detectron_config_base_backup_urls = detectron_config_base_backup_urls.split('\n') if detectron_config_base_backup_urls is not None else None
    detectron_config_backup_urls = detectron_config_backup_urls.split('\n') if detectron_config_backup_urls is not None else None
    detectron_model_backup_urls = detectron_model_backup_urls.split('\n') if detectron_model_backup_urls is not None else None

    # Render the new project -> all the process is made using a temporary folder
    # Idea : we will copy all the files that needs to be rendered + the optionnal instructions / configurations in this folder
    # Then we will render the whole folder at once
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.realpath(__file__))) as tmp_folder:

        # Copy main folder to be rendered
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vision_project')
        copy_tree(template_dir, tmp_folder)

        # Copy models upload instructions
        ressources_path = os.path.join(tmp_folder, f'{project_name}-ressources')
        if not os.path.exists(ressources_path):
            os.makedirs(ressources_path)
        upload_intructions_target_path = os.path.join(ressources_path, 'model_upload_instructions.md')
        copyfile(upload_intructions_path, upload_intructions_target_path)

        # Copy dvc config if available
        if dvc_config_path is not None:
            dvc_config_directory = os.path.join(tmp_folder, '.dvc')
            if not os.path.exists(dvc_config_directory):
                os.makedirs(dvc_config_directory)
            dvc_target_path = os.path.join(dvc_config_directory, 'config')
            copyfile(dvc_config_path, dvc_target_path)

        # Get environment
        env = Environment(loader=FileSystemLoader(tmp_folder))

        # For each template (a.k.a. files), load and fill it, then save into output dir
        for template_name in env.list_templates():

            # Nominal process
            if not template_name.endswith(('.jpg', '.jpeg', '.png', '.pyc', '.pyo')):
                # Get render
                template = env.get_template(template_name)
                render = template.render(package_name=project_name,
                                         default_sep=default_sep,
                                         default_encoding=default_encoding,
                                         pip_trusted_host=pip_trusted_host,
                                         pip_index_url=pip_index_url,
                                         mlflow_tracking_uri=mlflow_tracking_uri,
                                         additional_pip_packages=additional_pip_packages,
                                         vgg16_weights_backup_urls=vgg16_weights_backup_urls,
                                         efficientnetb6_weights_backup_urls=efficientnetb6_weights_backup_urls,
                                         detectron_config_base_backup_urls=detectron_config_base_backup_urls,
                                         detectron_config_backup_urls=detectron_config_backup_urls,
                                         detectron_model_backup_urls=detectron_model_backup_urls,
                                         dvc_config_ok=dvc_config_ok)

                # Ignore empty files
                # This is useful to remove some files when configuration are missing, e.g. for DVC
                if render == '' and not template_name.endswith('__init__.py'):
                    continue

                # Replace package_name in file/directory
                template_name = template_name.replace('package_name', project_name)

                # Save it
                final_path = os.path.join(output_dir, template_name)
                basedir = os.path.dirname(final_path)
                if not os.path.exists(basedir):
                    os.makedirs(basedir)
                # Encoding scripts in utf-8 !
                with open(final_path, 'w', encoding='utf-8') as f:
                    f.write(render)

            # Specials format -> copy/paste
            else:
                initial_path = os.path.join(template_dir, template_name)
                # Replace package_name in file/directory
                template_name = template_name.replace('package_name', project_name)
                final_path = os.path.join(output_dir, template_name)
                dist_dir_path = os.path.dirname(final_path)
                if not os.path.exists(dist_dir_path):
                    os.makedirs(dist_dir_path)
                copyfile(initial_path, final_path)

    # Everything is rendered, we just need to create some subdirectories
    data_dir = os.path.join(output_dir, f'{project_name}-data')
    models_weights_dir = os.path.join(output_dir, f'{project_name}-data', 'transfer_learning_weights')
    detectron2_conf_dir = os.path.join(output_dir, f'{project_name}-data', 'detectron2_conf_files')
    cache_keras_dir = os.path.join(output_dir, f'{project_name}-data', 'cache_keras')
    models_dir = os.path.join(output_dir, f'{project_name}-models')
    exploration_dir = os.path.join(output_dir, f'{project_name}-exploration')
    for new_dir in [data_dir, models_weights_dir, detectron2_conf_dir,
                    cache_keras_dir, models_dir, exploration_dir]:
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)


if __name__ == '__main__':
    main()
