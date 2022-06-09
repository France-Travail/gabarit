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
        gen_project = f"generate_nlp_project -n test_template_nlp -p {full_path_lib}"
        self.assertEqual(subprocess.run(gen_project, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(full_path_lib))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'setup.py')))

    def test02_GenerateVenv(self):
        '''Checks a venv creation'''
        print("Virtual env generation")
        # Generate venv and test it
        gen_venv = f"python -m venv {full_path_lib}/venv_test_template_nlp"
        self.assertEqual(subprocess.run(gen_venv, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'venv_test_template_nlp')))

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
            copy_data = f"robocopy nlp_data {full_path_lib}/test_template_nlp-data /E"
            returncodes = [0, 1] # https://ss64.com/nt/robocopy-exit.html
        else:
            copy_data = f"cp nlp_data/* {full_path_lib}/test_template_nlp-data/"
            returncodes = [0]
        self.assertTrue(subprocess.run(copy_data, shell=True).returncode in returncodes)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label.csv')))
        # TODO: Try to load flaubert small for torch tests


class Case2_functionals_tests(unittest.TestCase):
    '''Main class to test functionals tests'''

    def test01_launchtests(self):
        '''Launches functional tests'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {os.getcwd()}/nlp_tests/functional_tests.py', shell=True).returncode, 0)
        # Clear models
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)


class Case3_unit_tests(unittest.TestCase):
    '''Main class to test unit tests'''

    def test01_test_static_type(self):
        '''Launches mypy tests'''
        # https://realpython.com/python-type-checking/#static-type-checking
        self.assertEqual(subprocess.run(f'(cd {full_path_lib} && {activate_venv}python -m mypy --ignore-missing-imports --allow-redefinition --no-strict-optional -p test_template_nlp)', shell=True).returncode, 0)

    def test02_test_utils(self):
        '''Launches tests of file utils.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_utils.py', shell=True).returncode, 0)

    def test03_test_utils_models(self):
        '''Launches tests of file utils_models.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_utils_models.py', shell=True).returncode, 0)
        # Clear models
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test04_test_preprocess(self):
        '''Launches tests of file preprocess.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_preprocess.py', shell=True).returncode, 0)

    def test05_test_model_class(self):
        '''Launches tests of file model_class.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_class.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test06_test_model_pipeline(self):
        '''Launches tests of file model_pipeline.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_pipeline.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test07_test_model_tfidf_svm(self):
        '''Launches tests of file model_tfidf_svm.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_tfidf_svm.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test08_test_model_model_tfidf_gbt(self):
        '''Launches tests of file model_tfidf_gbt.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_tfidf_gbt.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test09_test_model_model_tfidf_lgbm(self):
        '''Launches tests of file model_tfidf_lgbm.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_tfidf_lgbm.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test10_test_model_model_tfidf_sgdc(self):
        '''Launches tests of file model_tfidf_sgdc.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_tfidf_sgdc.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test11_test_model_keras(self):
        '''Launches tests of file model_keras.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_keras.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test12_test_model_embedding_cnn(self):
        '''Launches tests of file model_embedding_cnn.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_embedding_cnn.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test13_test_model_embedding_lstm(self):
        '''Launches tests of file model_embedding_lstm.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_embedding_lstm.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test14_test_model_embedding_lstm_attention(self):
        '''Launches tests of file model_embedding_lstm_attention.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_embedding_lstm_attention.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test15_test_model_embedding_lstm_attention(self):
        '''Launches tests of file model_embedding_lstm_structured_attention.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_embedding_lstm_structured_attention.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test16_test_model_embedding_lstm_gru_gpu(self):
        '''Launches tests of file model_embedding_lstm_gru_gpu.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_embedding_lstm_gru_gpu.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test17_test_model_tfidf_dense(self):
        '''Launches tests of file model_tfidf_dense.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_tfidf_dense.py', shell=True).returncode, 0)
        models_path = os.path.join(full_path_lib, 'test_template_nlp-models')
        if os.path.exists(models_path):
            shutil.rmtree(models_path)
            os.makedirs(models_path)

    def test18_test_model_logger(self):
        '''Launches tests of file model_logger.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_logger.py', shell=True).returncode, 0)

    def test19_test_model_explainer(self):
        '''Launches tests of file model_explainer.py'''
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_explainer.py', shell=True).returncode, 0)

    def test20_test_pytorch_transformers(self):
        '''Launches tests of file model_pytorch_transformers.py'''
        # Run only if transformer loaded locally
        transformers_path = os.path.join(full_path_lib, 'test_template_nlp-transformers')
        transformer_path = os.path.join(transformers_path, 'flaubert', 'flaubert_small_cased')
        # TODO: add flaubert_small_cased download
        if not os.path.exists(transformer_path):
            print("WARNING : Can't test the Pytorch Transformer model -> can't find transformer")
            print("How to use : download flaubert_small_cased in the folder of the module to test")
            print("We ignore this test.")
            return None
        # If available, run the test
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_pytorch_transformers.py', shell=True).returncode, 0)

    def test21_test_pytorch_light(self):
        '''Launches tests of file model_pytorch_light.py'''
        # Run only if transformer loaded locally
        transformers_path = os.path.join(full_path_lib, 'test_template_nlp-transformers')
        transformer_path = os.path.join(transformers_path, 'flaubert', 'flaubert_small_cased')
        # TODO: add flaubert_small_cased download
        if not os.path.exists(transformer_path):
            print("WARNING : Can't test the Pytorch Transformer model -> can't find transformer")
            print("How to use : download flaubert_small_cased in the folder of the module to test")
            print("We ignore this test.")
            return None
        # If available, run the test
        self.assertEqual(subprocess.run(f'{activate_venv}python {full_path_lib}/tests/test_model_pytorch_light.py', shell=True).returncode, 0)


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
    full_path_lib = os.path.abspath(os.path.join(os.getcwd(), 'test_template_nlp'))
    if os.name == 'nt':
        is_windows = True
        # Windows: activate the virtual environment & continue with the other processes
        activate_venv = f"cd {full_path_lib}/venv_test_template_nlp/Scripts & activate & cd ../../ & "
    else:
        is_windows = False
        # UNIX : We can't use "source" so we directly call python/pip from the bin of the virtual environment
        activate_venv = f"{full_path_lib}/venv_test_template_nlp/bin/"

    # Start tests
    unittest.main()
