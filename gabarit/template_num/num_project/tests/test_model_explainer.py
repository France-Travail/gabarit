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

from {{package_name}} import utils
from {{package_name}}.models_training import utils_models
from {{package_name}}.monitoring.model_explainer import ShapExplainer
from {{package_name}}.models_training.regressors.models_sklearn.model_rf_regressor import ModelRFRegressor
from {{package_name}}.models_training.classifiers.models_sklearn.model_rf_classifier import ModelRFClassifier


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

    def test01_shap_explainer_nominal(self):
        '''Test of the SHAP explainer'''

        # Model dir
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, -2, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 2, 3, 12, 2] * 10})
        content = pd.DataFrame({'col_1': [-4], 'col_2': [1]})
        y_train_mono_2 = pd.Series([0, 0, 0, 0, 1, 1, 1] * 10)
        y_train_mono_3 = pd.Series([0, 0, 0, 2, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1, 1] * 10, 'y3': [0, 0, 1, 0, 1, 0, 1] * 10})
        y_train_regressor = pd.Series([-3, -2, -8, 0, 5, 6, 5] * 10)
        x_col = ['col_1', 'col_2']
        y_col_mono = ['toto']
        y_col_multi = ['y1', 'y2', 'y3']

        # Classifier - Mono-class - Mono-label
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        model.fit(x_train, y_train_mono_2)
        # Nominal
        explainer = ShapExplainer(model, anchor_data=x_train, anchor_preprocessed=False)
        explanation = explainer.explain_instance(content, class_or_label_index=0)
        explanation = explainer.explain_instance(content, class_or_label_index=1)
        explanation = explainer.explain_instance(content, class_or_label_index=None)
        html = explainer.explain_instance_as_html(content, class_or_label_index=0)
        html = explainer.explain_instance_as_html(content, class_or_label_index=1)
        html = explainer.explain_instance_as_html(content, class_or_label_index=None)
        # Preprocessed anchor data
        explainer = ShapExplainer(model, anchor_data=utils_models.apply_pipeline(x_train, model.preprocess_pipeline), anchor_preprocessed=True)
        explanation = explainer.explain_instance(content, class_or_label_index=0)
        explanation = explainer.explain_instance(content, class_or_label_index=1)
        explanation = explainer.explain_instance(content, class_or_label_index=None)
        html = explainer.explain_instance_as_html(content, class_or_label_index=0)
        html = explainer.explain_instance_as_html(content, class_or_label_index=1)
        html = explainer.explain_instance_as_html(content, class_or_label_index=None)
        remove_dir(model_dir)

        # Classifier - Multi-class - Mono-label
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        model.fit(x_train, y_train_mono_3)
        # Nominal
        explainer = ShapExplainer(model, anchor_data=x_train, anchor_preprocessed=False)
        explanation = explainer.explain_instance(content, class_or_label_index=0)
        explanation = explainer.explain_instance(content, class_or_label_index=1)
        explanation = explainer.explain_instance(content, class_or_label_index=2)
        explanation = explainer.explain_instance(content, class_or_label_index=None)
        html = explainer.explain_instance_as_html(content, class_or_label_index=0)
        html = explainer.explain_instance_as_html(content, class_or_label_index=1)
        html = explainer.explain_instance_as_html(content, class_or_label_index=2)
        html = explainer.explain_instance_as_html(content, class_or_label_index=None)
        # Preprocessed anchor data
        explainer = ShapExplainer(model, anchor_data=utils_models.apply_pipeline(x_train, model.preprocess_pipeline), anchor_preprocessed=True)
        explanation = explainer.explain_instance(content, class_or_label_index=0)
        explanation = explainer.explain_instance(content, class_or_label_index=1)
        explanation = explainer.explain_instance(content, class_or_label_index=2)
        explanation = explainer.explain_instance(content, class_or_label_index=None)
        html = explainer.explain_instance_as_html(content, class_or_label_index=0)
        html = explainer.explain_instance_as_html(content, class_or_label_index=1)
        html = explainer.explain_instance_as_html(content, class_or_label_index=2)
        html = explainer.explain_instance_as_html(content, class_or_label_index=None)
        remove_dir(model_dir)

        # Classifier - Mono-class - Multi-label
        model = ModelRFClassifier(x_col=x_col, y_col=y_col_multi, model_dir=model_dir, multi_label=True)
        model.fit(x_train, y_train_multi)
        # Nominal
        explainer = ShapExplainer(model, anchor_data=x_train, anchor_preprocessed=False)
        explanation = explainer.explain_instance(content, class_or_label_index=0)
        explanation = explainer.explain_instance(content, class_or_label_index=1)
        explanation = explainer.explain_instance(content, class_or_label_index=2)
        explanation = explainer.explain_instance(content, class_or_label_index=None)
        html = explainer.explain_instance_as_html(content, class_or_label_index=0)
        html = explainer.explain_instance_as_html(content, class_or_label_index=1)
        html = explainer.explain_instance_as_html(content, class_or_label_index=2)
        html = explainer.explain_instance_as_html(content, class_or_label_index=None)
        # Preprocessed anchor data
        explainer = ShapExplainer(model, anchor_data=utils_models.apply_pipeline(x_train, model.preprocess_pipeline), anchor_preprocessed=True)
        explanation = explainer.explain_instance(content, class_or_label_index=0)
        explanation = explainer.explain_instance(content, class_or_label_index=1)
        explanation = explainer.explain_instance(content, class_or_label_index=2)
        explanation = explainer.explain_instance(content, class_or_label_index=None)
        html = explainer.explain_instance_as_html(content, class_or_label_index=0)
        html = explainer.explain_instance_as_html(content, class_or_label_index=1)
        html = explainer.explain_instance_as_html(content, class_or_label_index=2)
        html = explainer.explain_instance_as_html(content, class_or_label_index=None)
        remove_dir(model_dir)

        # Regressor
        model = ModelRFRegressor(x_col=x_col, y_col=y_col_mono, model_dir=model_dir)
        model.fit(x_train, y_train_regressor)
        # Nominal
        explainer = ShapExplainer(model, anchor_data=x_train, anchor_preprocessed=False)
        explanation = explainer.explain_instance(content, class_or_label_index=0)
        explanation = explainer.explain_instance(content, class_or_label_index=None)
        html = explainer.explain_instance_as_html(content, class_or_label_index=0)
        html = explainer.explain_instance_as_html(content, class_or_label_index=None)
        # Preprocessed anchor data
        explainer = ShapExplainer(model, anchor_data=utils_models.apply_pipeline(x_train, model.preprocess_pipeline), anchor_preprocessed=True)
        explanation = explainer.explain_instance(content, class_or_label_index=0)
        explanation = explainer.explain_instance(content, class_or_label_index=None)
        html = explainer.explain_instance_as_html(content, class_or_label_index=0)
        html = explainer.explain_instance_as_html(content, class_or_label_index=None)
        remove_dir(model_dir)

        # Check errors
        # Regressor withtout predict
        class FakeModelClass1():
            def __init__(self):
                self.model_type = 'regressor'
            def predict_proba(self):
                pass
        with self.assertRaises(TypeError):
            ShapExplainer(FakeModelClass1(), anchor_data=x_train, anchor_preprocessed=False)
        # Classifier withtout predict_proba
        class FakeModelClass2():
            def __init__(self):
                self.model_type = 'classifier'
            def predict(self):
                pass
        with self.assertRaises(TypeError):
            ShapExplainer(FakeModelClass2(), anchor_data=x_train, anchor_preprocessed=False)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
