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
import pickle
import shutil
import numpy as np
import pandas as pd

from {{package_name}} import utils
from {{package_name}}.models_training.model_tfidf_svm import ModelTfidfSvm
from {{package_name}}.monitoring.model_explainer import LimeExplainer, AttentionExplainer
from {{package_name}}.models_training.model_embedding_lstm_structured_attention import ModelEmbeddingLstmStructuredAttention


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
        '''Test of the mono-class mono-label case'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False)
        # fake model_conf
        model_conf = {'preprocess_str': 'no_preprocess'}
        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"] * 100)
        x_test = np.array(["cela est un test", "ni cela", "non plus", "ici test", "là, rien de rien!"] * 100)
        y_train_mono = np.array([0, 1, 0, 1, 2] * 100)
        y_test_mono = y_train_mono.copy()
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0] * 100, 'test2': [1, 0, 0, 0, 0] * 100, 'test3': [0, 0, 0, 1, 0] * 100})
        y_test_multi = y_train_multi.copy()
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=False)
        model.fit(x_train, y_train_mono)
        exp = LimeExplainer(model, model_conf)
        txt = exp.explain_instance_as_list(text="ceci est un test", classes=[0, 1])
        html = exp.explain_instance_as_html(text="ceci est un test", classes=[0, 1])
        remove_dir(model_dir)

        # Multi-labels
        model = ModelTfidfSvm(model_dir=model_dir, multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        txt = exp.explain_instance_as_list(text="ceci est un test", classes=[0, 1])
        html = exp.explain_instance_as_html(text="ceci est un test", classes=[0, 1])
        remove_dir(model_dir)

    def test02_attention_explainer_nominal(self):
        '''Test of the mono-class mono-label case'''

        # Model creation
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"] * 100)
        x_test = np.array(["cela est un test", "ni cela", "non plus", "ici test", "là, rien de rien!"] * 100)
        y_train_mono = np.array([0, 1, 0, 1, 2] * 100)
        y_test_mono = y_train_mono.copy()
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0] * 100, 'test2': [1, 0, 0, 0, 0] * 100, 'test3': [0, 0, 0, 1, 0] * 100})
        y_test_multi = y_train_multi.copy()
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        model = ModelEmbeddingLstmStructuredAttention(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False,
                                            max_sequence_length=10, max_words=100,
                                            padding='pre', truncating='post',
                                            embedding_name='fake_embedding.pkl')
        model.fit(x_train, y_train_mono)
        exp = AttentionExplainer(model)
        txt = exp.explain_instance_as_list(text="ceci est un test")
        html = exp.explain_instance_as_html(text="ceci est un test")
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
