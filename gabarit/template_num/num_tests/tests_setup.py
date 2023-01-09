#!/usr/bin/env python3
# Launches all the initialization tests of the template
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

# Libs unittest
import unittest
from unittest.mock import Mock
from unittest.mock import patch

# utils libs
import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path



class Case1_Env(unittest.TestCase):
    '''Main class to test environnement creation'''
    pip_trusted_host = None
    pip_index_url = None

    def test01_GenerateProject(self):
        '''Checks a project generation'''
        print("Project generation")
        # First remove any old folder
        if os.path.exists(full_path_lib):
            shutil.rmtree(full_path_lib)
        # Generate project and test it
        gen_project = f"generate_num_project -n test_template_num -p {full_path_lib}"
        self.assertEqual(subprocess.run(gen_project, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(full_path_lib))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'setup.py')))

    def test02_GenerateVenv(self):
        '''Checks a venv creation'''
        print("Virtual env generation")
        # Generate venv and test it
        gen_venv = f"python -m venv {full_path_lib}/venv_test_template_num"
        self.assertEqual(subprocess.run(gen_venv, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'venv_test_template_num')))

    def test03_InstallTemplate(self):
        '''Checks the installation of the template'''
        print("Project installation")
        # Install project and test it
        # Get upgrade & requirements command lines
        upgrade_pip = f"{activate_venv}pip install --upgrade pip"
        install_requirements = f"{activate_venv}pip install -r {full_path_lib}/requirements.txt"
        # Add PIP options
        if self.pip_trusted_host is not None:
            upgrade_pip += f" --trusted-host {self.pip_trusted_host}"
            install_requirements += f" --trusted-host {self.pip_trusted_host}"
        if self.pip_index_url is not None:
            upgrade_pip += f" --index-url {self.pip_index_url}"
            install_requirements += f" --index-url {self.pip_index_url}"
        # Get setup command line
        if is_windows:
            install_project = f"{activate_venv} cd {full_path_lib} & python setup.py develop"
        else:
            install_project = f"cd {full_path_lib} && {activate_venv}python setup.py develop"
        self.assertEqual(subprocess.run(upgrade_pip, shell=True).returncode, 0)
        self.assertEqual(subprocess.run(install_requirements, shell=True).returncode, 0)
        self.assertEqual(subprocess.run(install_project, shell=True).returncode, 0)

    def test04_DataExists(self):
        '''Copies the required datasets'''
        print("Copy of the datasets")
        # Copy datasets into the generated project
        if is_windows:
            # cf. https://stackoverflow.com/questions/4601161/copying-all-contents-of-folder-to-another-folder-using-batch-file
            copy_data = f"robocopy num_data {full_path_lib}/test_template_num-data /E"
            returncodes = [0, 1] # https://ss64.com/nt/robocopy-exit.html
        else:
            copy_data = f"cp num_data/* {full_path_lib}/test_template_num-data/"
            returncodes = [0]
        self.assertTrue(subprocess.run(copy_data, shell=True).returncode in returncodes)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_num-data', 'mono_class_mono_label.csv')))


class Case2_functionals_tests(unittest.TestCase):
    '''Main class to test functionals tests'''

    def test01_launchtests(self):
        '''Launches functional tests'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {os.getcwd()}/num_tests/functional_tests.py', shell=True).returncode, 0)
        # Clear models
        models_path = os.path.join(full_path_lib, 'test_template_num-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

if __name__ == '__main__':
    # Retrieve pip params
    # Based on https://stackoverflow.com/questions/1029891/python-unittest-is-there-a-way-to-pass-command-line-options-to-the-app
    # and https://stackoverflow.com/questions/11380413/python-unittest-passing-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pip_trusted_host', default=None, help="PIP trusted-host argument to be used.")
    parser.add_argument('--pip_index_url', default=None, help="PIP index-url argument to be used.")
    parser.add_argument('unittest_args', nargs='*', help="Optional unitest args")
    args = parser.parse_args()
    Case1_Env.pip_trusted_host = args.pip_trusted_host
    Case1_Env.pip_index_url = args.pip_index_url
    sys.argv[1:] = args.unittest_args

    # Change directory to script directory parent
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    parentname = str(Path(dname).parent)
    os.chdir(parentname)

    # Manage venv
    full_path_lib = os.path.abspath(os.path.join(os.getcwd(), 'test_template_num'))
    if os.name == 'nt':
        is_windows = True
        # Windows: activate the virtual environment & continue with the other processes
        activate_venv = f"cd {full_path_lib}/venv_test_template_num/Scripts & activate & cd ../../ & "
    else:
        is_windows = False
        # UNIX : We can't use "source" so we directly call python/pip from the bin of the virtual environment
        activate_venv = f"{full_path_lib}/venv_test_template_num/bin/"

    # Start tests
    unittest.main()
