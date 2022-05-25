#!/usr/bin/env python3
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

# Utils libs
import os
import json
import shutil
import numpy as np
import pandas as pd

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training.model_class import ModelClass

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelClassTests(unittest.TestCase):
    '''Main class to test model_class'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_class_init(self):
        '''Test of the initialization of {{package_name}}.models_training.model_class.ModelClass'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Init., test all parameters
        model = ModelClass(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        self.assertEqual(model.model_type, None)
        with self.assertRaises(NotImplementedError):
            model.fit('test')
        with self.assertRaises(NotImplementedError):
            model.predict('test')
        with self.assertRaises(NotImplementedError):
            model.predict_proba('test')
        with self.assertRaises(NotImplementedError):
            model.inverse_transform('test')
        with self.assertRaises(NotImplementedError):
            model.get_and_save_metrics('test', 'test')
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        model = ModelClass(model_dir=model_dir, model_name=model_name)
        self.assertEqual(model.model_name, model_name)
        remove_dir(model_dir)

        model = ModelClass(model_dir=model_dir, level_save='HIGH')
        self.assertEqual(model.level_save, 'HIGH')
        remove_dir(model_dir)

        model = ModelClass(model_dir=model_dir, level_save='MEDIUM')
        self.assertEqual(model.level_save, 'MEDIUM')
        remove_dir(model_dir)

        model = ModelClass(model_dir=model_dir, level_save='LOW')
        self.assertEqual(model.level_save, 'LOW')
        remove_dir(model_dir)

        # Check the input(s) type(s)
        with self.assertRaises(ValueError):
            ModelClass(model_dir=model_dir, level_save='toto')

    def test02_model_class_save(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass.save'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # test save
        model = ModelClass(model_dir=model_dir, model_name=model_name)
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertTrue('mainteners' in configs.keys())
        self.assertTrue('date' in configs.keys())
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('model_type' in configs.keys())
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], None)
        remove_dir(model_dir)

        # test save, level_save = 'LOW'
        model = ModelClass(model_dir=model_dir, model_name=model_name, level_save='LOW')
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        model.save(json_data={'test': 8})
        self.assertFalse(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        remove_dir(model_dir)

    def test03_model_class_save_upload_properties(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass._save_upload_properties'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Nominal case
        json_dict = {
            "mainteners": "c'est nous",
            "date": "01/01/1970 - 00:00:00",
            "bruit": "toto",
            "package_version": "0.0.8",
            "model_name": "hello_model",
            "list_classes": ["c1", "c2", np.int32(9), "c3", 3],
            "autre_bruit": "titi",
            "librairie": "ma_lib",
            "fit_time": "7895s",
        }
        model = ModelClass(model_dir=model_dir, model_name=model_name)
        model._save_upload_properties(json_dict)
        # Checks the presence of a file model_upload_instructions.md
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_upload_instructions.md')))
        with open(os.path.join(model.model_dir, 'model_upload_instructions.md'), 'r', encoding='{{default_encoding}}') as f:
            instructions = f.read()
        self.assertTrue(os.path.abspath(model.model_dir) in instructions)
        # Checks the presence of a file proprietes.json
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'proprietes.json')))
        with open(os.path.join(model.model_dir, 'proprietes.json'), 'r', encoding='{{default_encoding}}') as f:
            proprietes = json.load(f)
        self.assertTrue('mainteners' in proprietes.keys())
        self.assertEqual(proprietes['mainteners'], "c'est nous")
        self.assertTrue('date' in proprietes.keys())
        self.assertEqual(proprietes['date'], "01/01/1970 - 00:00:00")
        self.assertTrue('package_version' in proprietes.keys())
        self.assertEqual(proprietes['package_version'], "0.0.8")
        self.assertTrue('model_name' in proprietes.keys())
        self.assertEqual(proprietes['model_name'], "hello_model")
        self.assertTrue('list_classes' in proprietes.keys())
        self.assertEqual(proprietes['list_classes'], ["c1", "c2", 9, "c3", 3])
        self.assertTrue('librairie' in proprietes.keys())
        self.assertEqual(proprietes['librairie'], "ma_lib")
        self.assertTrue('fit_time' in proprietes.keys())
        self.assertEqual(proprietes['fit_time'], "7895s")
        self.assertFalse('bruit' in proprietes.keys())
        self.assertFalse('autre_bruit' in proprietes.keys())
        remove_dir(model_dir)

        # Same, but via the save function
        json_dict = {
            "mainteners": "c'est nous",
            "date": "01/01/1970 - 00:00:00",
            "bruit": "toto",
            "package_version": "0.0.8",
            "model_name": "hello_model",
            "list_classes": ["c1", "c2", "c8", "c3"],
            "autre_bruit": "titi",
            "librairie": "ma_lib",
            "fit_time": "7895s",
        }
        model = ModelClass(model_dir=model_dir, model_name=model_name)
        model.save(json_dict)
        # Checks the presence of a file model_upload_instructions.md
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_upload_instructions.md')))
        with open(os.path.join(model.model_dir, 'model_upload_instructions.md'), 'r', encoding='{{default_encoding}}') as f:
            instructions = f.read()
        self.assertTrue(os.path.abspath(model.model_dir) in instructions)
        # Checks the presence of a file proprietes.json
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'proprietes.json')))
        with open(os.path.join(model.model_dir, 'proprietes.json'), 'r', encoding='{{default_encoding}}') as f:
            proprietes = json.load(f)
        self.assertTrue('mainteners' in proprietes.keys())
        self.assertEqual(proprietes['mainteners'], "c'est nous")
        self.assertTrue('date' in proprietes.keys())
        self.assertEqual(proprietes['date'], "01/01/1970 - 00:00:00")
        self.assertTrue('package_version' in proprietes.keys())
        self.assertEqual(proprietes['package_version'], "0.0.8")
        self.assertTrue('model_name' in proprietes.keys())
        self.assertEqual(proprietes['model_name'], "hello_model")
        self.assertTrue('list_classes' in proprietes.keys())
        self.assertEqual(proprietes['list_classes'], ["c1", "c2", "c8", "c3"])
        self.assertTrue('librairie' in proprietes.keys())
        self.assertEqual(proprietes['librairie'], "ma_lib")
        self.assertTrue('fit_time' in proprietes.keys())
        self.assertEqual(proprietes['fit_time'], "7895s")
        self.assertFalse('bruit' in proprietes.keys())
        self.assertFalse('autre_bruit' in proprietes.keys())
        remove_dir(model_dir)

        # Empty case
        json_dict = {}
        model = ModelClass(model_dir=model_dir, model_name=model_name)
        model._save_upload_properties(json_dict)
        # Checks the presence of a file model_upload_instructions.md
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_upload_instructions.md')))
        with open(os.path.join(model.model_dir, 'model_upload_instructions.md'), 'r', encoding='{{default_encoding}}') as f:
            instructions = f.read()
        self.assertTrue(os.path.abspath(model.model_dir) in instructions)
        # Checks the presence of a file proprietes.json
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'proprietes.json')))
        with open(os.path.join(model.model_dir, 'proprietes.json'), 'r', encoding='{{default_encoding}}') as f:
            proprietes = json.load(f)
        self.assertFalse('mainteners' in proprietes.keys())
        self.assertFalse('date' in proprietes.keys())
        self.assertFalse('package_version' in proprietes.keys())
        self.assertFalse('model_name' in proprietes.keys())
        self.assertFalse('list_classes' in proprietes.keys())
        self.assertFalse('librairie' in proprietes.keys())
        self.assertFalse('fit_time' in proprietes.keys())
        self.assertFalse('bruit' in proprietes.keys())
        self.assertFalse('autre_bruit' in proprietes.keys())
        remove_dir(model_dir)

    def test04_model_class_get_model_dir(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass._get_model_dir'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'

        # Nominal case
        model = ModelClass(model_dir=model_dir, model_name=model_name)
        expected_dir = os.path.join(utils.get_models_path(), model_name, f"{model_name}_")
        res_dir = model._get_model_dir()
        self.assertTrue(res_dir.startswith(expected_dir))
        remove_dir(model_dir)

    def test05_model_class_is_gpu_activated(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass._is_gpu_activated'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        model_name = 'model_test'
        remove_dir(model_dir)

        # Nominal case
        model = ModelClass(model_dir=model_dir, model_name=model_name)
        self.assertFalse(model._is_gpu_activated())
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
