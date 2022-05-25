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

from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.model_pipeline import ModelPipeline

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
        x_col = ['test_x1', 'test_x2']
        y_col = 'test_y'

        # Init., test all paramameteres
        model = ModelClass(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        with self.assertRaises(NotImplementedError):
            model.fit('test', 'test')
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

        model = ModelClass(model_dir=model_dir, x_col=x_col)
        self.assertEqual(model.x_col, x_col)
        remove_dir(model_dir)

        model = ModelClass(model_dir=model_dir, y_col=y_col)
        self.assertEqual(model.y_col, y_col)
        remove_dir(model_dir)

        preprocess_pipeline = preprocess.get_pipeline("no_preprocess") # Warning, needs to be fitted
        preprocess_pipeline.fit(pd.DataFrame({'test_x1': [1, 2, 3], 'test_x2': [4, 5, 6]})) # Fit the pipeline
        model = ModelClass(model_dir=model_dir, preprocess_pipeline=preprocess_pipeline)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, x_col)
        self.assertEqual(model.mandatory_columns, x_col)
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
        with self.assertRaises(NotFittedError):
            preprocess_pipeline = preprocess.get_pipeline("no_preprocess")
            ModelClass(model_dir=model_dir, preprocess_pipeline=preprocess_pipeline)


    def test02_model_class_save(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass.save'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        preprocess_pipeline = preprocess.get_pipeline("no_preprocess") # Warning, needs to be fitted
        preprocess_pipeline.fit(pd.DataFrame({'test_x1': [1, 2, 3], 'test_x2': [4, 5, 6]}))

        # test save
        model = ModelClass(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline)
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        preprocess_pipeline_path = os.path.join(model.model_dir, 'preprocess_pipeline.pkl')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        self.assertTrue(os.path.exists(preprocess_pipeline_path))
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
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('columns_in' in configs.keys())
        self.assertTrue('mandatory_columns' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], None)
        remove_dir(model_dir)

        # test save, level_save = 'LOW'
        model = ModelClass(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline, level_save='LOW')
        pkl_path = os.path.join(model.model_dir, f"{model.model_name}.pkl")
        configuration_path = os.path.join(model.model_dir, 'configurations.json')
        preprocess_pipeline_path = os.path.join(model.model_dir, 'preprocess_pipeline.pkl')
        model.save(json_data={'test': 8})
        self.assertFalse(os.path.exists(pkl_path))
        self.assertTrue(os.path.exists(configuration_path))
        self.assertFalse(os.path.exists(preprocess_pipeline_path))
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

        # Same, but via the save method
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


    def test05_model_class_check_input_format(self):
        '''Test of the method {{package_name}}.models_training.model_class.ModelClass._check_input_format

        Also tests  columns_in, mandatory_columns, etc.
        '''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model_name = 'test'
        x_col = ['test_x1_prep', 'test_x2_prep']
        y_col = 'test_y'
        columns_in = ['test_x1', 'test_x2', 'useless']
        mandatory_columns = ['test_x1', 'test_x2']

        original_df = pd.DataFrame({'test_x1': [1, 2], 'test_x2': [3, 4], 'useless': [5, 6]})
        x_input = pd.DataFrame({'test_x1_prep': [2, 4], 'test_x2_prep': [6, 8]})
        x_input_bad_order = pd.DataFrame({'test_x2_prep': [6, 8], 'test_x1_prep': [2, 4]})
        x_input_bad_columns = pd.DataFrame({'test_x1_prep': [2, 4], 'toto': [6, 8]})
        x_input_bad_format = pd.DataFrame({'test_x1_prep': [2, 4], 'test_x2_prep': [6, 8], 'test_x3': [5, 6]})
        y_input = pd.Series([0, 1])
        y_input_df = pd.DataFrame({'test_y': [0, 1]})
        y_col_multi = ['test_y1', 'test_y2']
        y_input_multi = pd.DataFrame({'test_y1': [0, 1], 'test_y2': [1, 0]})
        y_input_multi_bad_order = pd.DataFrame({'test_y2': [1, 0], 'test_y1': [0, 1]})
        y_input_multi_bad_columns = pd.DataFrame({'toto': [0, 1], 'titi': [1, 0]})
        y_input_multi_bad_format = pd.DataFrame({'test_y1': [0, 1], 'test_y2': [1, 0], 'toto': [0, 1]})
        class customFunctionTransformer(FunctionTransformer):
            def __init__(self, func=None) -> None:
                super().__init__(func=func)
            def get_feature_names(self, feature_in):
                return [f"{feat}_prep" for feat in feature_in]
        preprocess_pipeline = ColumnTransformer([('test_pipeline', customFunctionTransformer(lambda x: x*2), ['test_x1', 'test_x2'])])
        preprocess_pipeline.fit(original_df)

        ### Nominal case
        model = ModelClass(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline, x_col=x_col, y_col=y_col)
        # OK - no y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, _ = model._check_input_format(x_input, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # OK - with y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input, y_input, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # OK - with y - fit_function=True
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input, y_input, fit_function=True)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # disorder - no y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, _ = model._check_input_format(x_input_bad_order, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # Get it with the right order
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # disorder - with y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input_bad_order, y_input, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # Get it with the right order
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # disorder - with y - fit_function=True
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input_bad_order, y_input, fit_function=True)
        pd.testing.assert_frame_equal(x_output, x_input) # Get it with the right order
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # wrong columns - no y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, _ = model._check_input_format(x_input_bad_columns, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input_bad_columns) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # wrong columns - with y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input_bad_columns, y_input, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input_bad_columns) # No modifications
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # wrong columns - with y - fit_function=True
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input_bad_columns, y_input, fit_function=True)
        pd.testing.assert_frame_equal(x_output, x_input_bad_columns) # No modifications
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        #
        remove_dir(model_dir)

        ### Nominal case - y_col multi
        model = ModelClass(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline, x_col=x_col, y_col=y_col_multi)
        # OK - no y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, _ = model._check_input_format(x_input, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # OK - with y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input, y_input_multi, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        pd.testing.assert_frame_equal(y_output, y_input_multi) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # OK - with y - fit_function=True
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input, y_input_multi, fit_function=True)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        pd.testing.assert_frame_equal(y_output, y_input_multi) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # disorder - no y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, _ = model._check_input_format(x_input_bad_order, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # Get it with the right order
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # disorder - with y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input_bad_order, y_input_multi_bad_order, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # Get it with the right order
        pd.testing.assert_frame_equal(y_output, y_input_multi) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # disorder - with y - fit_function=True
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input_bad_order, y_input_multi_bad_order, fit_function=True)
        pd.testing.assert_frame_equal(x_output, x_input) # Get it with the right order
        pd.testing.assert_frame_equal(y_output, y_input_multi) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # wrong columns - no y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, _ = model._check_input_format(x_input_bad_columns, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input_bad_columns) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # wrong columns - with y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input_bad_columns, y_input_multi_bad_columns, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input_bad_columns) # No modifications
        pd.testing.assert_frame_equal(y_output, y_input_multi_bad_columns) # No modifications
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # wrong columns - with y - fit_function=True
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input_bad_columns, y_input_multi_bad_columns, fit_function=True)
        pd.testing.assert_frame_equal(x_output, x_input_bad_columns) # No modifications
        pd.testing.assert_frame_equal(y_output, y_input_multi_bad_columns) # No modifications
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col_multi)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        #
        remove_dir(model_dir)

        ### Tests x_col to None
        model = ModelClass(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline, x_col=None, y_col=y_col)
        # OK - no y - fit_function=False
        self.assertEqual(model.x_col, None)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, _ = model._check_input_format(x_input, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, None) # Still None because fit_function=False
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # OK - with y - fit_function=False
        self.assertEqual(model.x_col, None)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input, y_input, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, None) # Still None because fit_function=False
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # OK - with y - fit_function=True
        self.assertEqual(model.x_col, None)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input, y_input, fit_function=True)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col) # x_col now set
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        model.x_col = None # reset for further tests
        # OK - with y - fit_function=True - x_input in np array
        self.assertEqual(model.x_col, None)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(np.array(x_input), y_input, fit_function=True)
        np.testing.assert_array_equal(x_output, np.array(x_input)) # No modifications
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, [0, 1]) # x_col now set - but no columns in a numpy array
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # No "disorder" nor "wrong columns"
        #
        remove_dir(model_dir)

        ### Tests y_col to None
        model = ModelClass(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline, x_col=x_col, y_col=None)
        # OK - no y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, None)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, _ = model._check_input_format(x_input, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, None) # Still None because fit_function=False
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # OK - with y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, None)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input, y_input, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, None) # Still None because fit_function=False
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # OK - with y - fit_function=True - y pd Series
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, None)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input, y_input, fit_function=True)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, 0) # y_col now set
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        model.y_col = None # reset for further tests
        # OK - with y - fit_function=True - y pd DataFrame
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, None)
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        x_output, y_output = model._check_input_format(x_input, y_input_df, fit_function=True)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        pd.testing.assert_frame_equal(y_output, y_input_df) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col) # y_col now set
        self.assertEqual(model.preprocess_pipeline, preprocess_pipeline)
        self.assertEqual(model.columns_in, columns_in)
        self.assertEqual(model.mandatory_columns, mandatory_columns)
        # No "disorder" nor "wrong columns"
        #
        remove_dir(model_dir)

        ### Tests preprocess_pipeline to None
        model = ModelClass(model_dir=model_dir, model_name=model_name, preprocess_pipeline=None, x_col=x_col, y_col=y_col)
        # OK - no y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, None)
        self.assertEqual(model.columns_in, None)
        self.assertEqual(model.mandatory_columns, None)
        x_output, _ = model._check_input_format(x_input, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, None)
        self.assertEqual(model.columns_in, None)
        self.assertEqual(model.mandatory_columns, None)
        # OK - with y - fit_function=False
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, None)
        self.assertEqual(model.columns_in, None)
        self.assertEqual(model.mandatory_columns, None)
        x_output, y_output = model._check_input_format(x_input, y_input, fit_function=False)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, None)
        self.assertEqual(model.columns_in, None)
        self.assertEqual(model.mandatory_columns, None)
        # OK - with y - fit_function=True
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        self.assertEqual(model.preprocess_pipeline, None)
        self.assertEqual(model.columns_in, None)
        self.assertEqual(model.mandatory_columns, None)
        x_output, y_output = model._check_input_format(x_input, y_input, fit_function=True)
        pd.testing.assert_frame_equal(x_output, x_input) # No modifications
        pd.testing.assert_series_equal(y_output, y_input) # No modifications
        self.assertEqual(_, None)
        self.assertEqual(model.x_col, x_col)
        self.assertEqual(model.y_col, y_col)
        # We can't test the equality of the pipeline. But we test the identity function on original_df
        x_input_transformed = model.preprocess_pipeline.transform(x_input)
        x_input_transformed = pd.DataFrame(x_input_transformed, columns=x_col) # Recast to dataframe
        pd.testing.assert_frame_equal(x_input_transformed, x_input) # No modifications
        self.assertEqual(model.columns_in, x_col) # Here x_col as columns in
        self.assertEqual(model.mandatory_columns, x_col) # Here x_col as mandatory columns
        #
        remove_dir(model_dir)

        # Check errors
        with self.assertRaises(AttributeError):
            model = ModelClass(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline, x_col=x_col, y_col=y_col)
            model._check_input_format(x_input, fit_function=True)
            remove_dir(model_dir)
        with self.assertRaises(ValueError):
            model = ModelClass(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline, x_col=x_col, y_col=y_col)
            model._check_input_format(x_input_bad_format)
            remove_dir(model_dir)
        with self.assertRaises(ValueError):
            model = ModelClass(model_dir=model_dir, model_name=model_name, preprocess_pipeline=preprocess_pipeline, x_col=x_col, y_col=y_col_multi)
            model._check_input_format(x_input, y_input_multi_bad_format)
            remove_dir(model_dir)


    def test06_model_class_is_gpu_activated(self):
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
