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
import dill as pickle
import shutil
import numpy as np
import pandas as pd
from PIL import Image

from {{package_name}} import utils
from {{package_name}}.monitoring.model_explainer import LimeExplainer
from {{package_name}}.models_training.classifiers.model_cnn_classifier import ModelCnnClassifier


# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelExplainerTest(unittest.TestCase):
    '''Main class to test model_explainer'''

    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)
        # Check if data folder exists
        data_path = utils.get_data_path()
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        # Create a mock embedding
        fake_embedding = {'toto': [0.25, 0.90, 0.12], 'titi': [0.85, 0.12, 0.8], 'test': [0.5, 0.6, 0.1]}
        fake_path = os.path.join(data_path, 'fake_embedding.pkl')
        if os.path.exists(fake_path):
            os.remove(fake_path)
        with open(fake_path, 'wb') as f:
            pickle.dump(fake_embedding, f, pickle.HIGHEST_PROTOCOL)

    def test01_lime_explainer_nominal(self):
        '''Test of the Lime explainer'''

        # Model dir
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        # fake model_conf
        model_conf = {'preprocess_str': 'no_preprocess'}
        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'test_data_1')
        filenames = ['Birman_1.jpg', 'Birman_2.jpg', 'Birman_4.jpg', 'Bombay_1.png', 'Bombay_10.jpg',
                     'Bombay_3.jpg', 'shiba_inu_15.jpg', 'shiba_inu_30.jpg', 'shiba_inu_31.jpg',
                     'shiba_inu_34.jpg', 'subfolder/Birman_36.jpg', 'subfolder/Bombay_19.jpg']
        df_train_mono = pd.DataFrame({
            'file_class': ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'cat'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        df_train_multi = pd.DataFrame({
            'file_class': ['birman', 'birman', 'birman', 'bombay', 'bombay', 'bombay', 'shiba', 'shiba', 'shiba', 'shiba', 'birman', 'bombay'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        img = Image.open(df_train_mono['file_path'].values[0])

        # Mono Class
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
        model.fit(df_train_mono, df_valid=df_train_mono)
        explainer = LimeExplainer(model, model_conf)
        explanation = explainer.explain_instance(content=img, class_or_label_index=0)
        explanation = explainer.explain_instance(content=img, class_or_label_index=1)
        explanation = explainer.explain_instance(content=img, class_or_label_index=None)
        remove_dir(model_dir)

        # Multi classes
        model = ModelCnnClassifier(model_dir=model_dir, batch_size=2, epochs=2)
        model.fit(df_train_multi, df_valid=df_train_multi)
        explainer = LimeExplainer(model, model_conf)
        explanation = explainer.explain_instance(content=img, class_or_label_index=0)
        explanation = explainer.explain_instance(content=img, class_or_label_index=1)
        explanation = explainer.explain_instance(content=img, class_or_label_index=2)
        explanation = explainer.explain_instance(content=img, class_or_label_index=None)
        remove_dir(model_dir)

        # Check errors
        # Not a classifier
        class FakeModelClass1():
            def __init__(self):
                self.list_classes = ['a', 'b', 'c']
                self.model_type = 'regressor'
            def predict_proba(self):
                pass
        with self.assertRaises(ValueError):
            LimeExplainer(FakeModelClass1(), model_conf)
        # No predict_proba
        class FakeModelClass2():
            def __init__(self):
                self.list_classes = ['a', 'b', 'c']
                self.model_type = 'classifier'
        with self.assertRaises(TypeError):
            LimeExplainer(FakeModelClass2(), model_conf)
        # No list_classes
        class FakeModelClass3():
            def __init__(self):
                self.model_type = 'classifier'
            def predict_proba(self):
                pass
        with self.assertRaises(TypeError):
            LimeExplainer(FakeModelClass3(), model_conf)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
