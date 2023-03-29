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
from datasets.arrow_dataset import Batch
from transformers import TextClassificationPipeline
from transformers.trainer_utils import EvalPrediction
from transformers.models.distilbert.modeling_distilbert import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from {{package_name}} import utils
from {{package_name}}.models_training.model_huggingface import ModelHuggingFace

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelHuggingFaceTests(unittest.TestCase):
    '''Main class to test model_huggingface'''

    def setUp(self):
        '''Setup fonction -> we create a mock embedding'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    def test01_model_huggingface_init(self):
        '''Test of the initialization of {{package_name}}.models_training.model_huggingface.ModelHuggingFace'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Init., test all parameters
        model = ModelHuggingFace(model_dir=model_dir)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.model, None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8)
        self.assertEqual(model.batch_size, 8)
        remove_dir(model_dir)

        #
        model = ModelHuggingFace(model_dir=model_dir, epochs=42)
        self.assertEqual(model.epochs, 42)
        remove_dir(model_dir)

        #
        model = ModelHuggingFace(model_dir=model_dir, validation_split=0.3)
        self.assertEqual(model.validation_split, 0.3)
        remove_dir(model_dir)

        #
        model = ModelHuggingFace(model_dir=model_dir, patience=10)
        self.assertEqual(model.patience, 10)
        remove_dir(model_dir)

        #
        model = ModelHuggingFace(model_dir=model_dir, model_max_length=10)
        self.assertEqual(model.model_max_length, 10)
        remove_dir(model_dir)

        # Can't be tested as this would try to load a transformer called 'toto'
        # We could patch it, but w/e
        # model = ModelHuggingFace(model_dir=model_dir, transformer_name='toto')
        # self.assertEqual(model.transformer_name, 'toto')
        # remove_dir(model_dir)

        # transformer_params must accept anything !
        model = ModelHuggingFace(model_dir=model_dir, transformer_params={'toto': 5})
        self.assertEqual(model.transformer_params, {'toto': 5})
        remove_dir(model_dir)

        # trainer_params
        model = ModelHuggingFace(model_dir=model_dir, trainer_params={'toto': 5})
        self.assertEqual(model.trainer_params, {'toto': 5})
        remove_dir(model_dir)

    def test02_model_huggingface_fit(self):
        '''Test of the method fit of {{package_name}}.models_training.model_huggingface.ModelHuggingFace'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"] * 100)
        x_valid = np.array(["cela est un test", "ni cela", "non plus", "ici test", "là, rien de rien!"] * 100)
        y_train_mono = np.array([0, 1, 0, 1, 2] * 100)
        y_valid_mono = y_train_mono.copy()
        y_valid_mono_missing = y_train_mono.copy()
        y_valid_mono_missing[y_valid_mono_missing == 2] = 0
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0] * 100, 'test2': [1, 0, 0, 0, 0] * 100, 'test3': [0, 0, 0, 1, 0] * 100})
        y_valid_multi = y_train_multi.copy()
        x_train_long = np.array(["cela est un test " * 1000, "là, rien! " * 1000] * 10)
        y_train_long = np.array([0, 1] * 10)

        # Mono-label
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        remove_dir(model_dir)

        # with valid data
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=x_valid, y_valid=y_valid_mono, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        remove_dir(model_dir)

        # no shuffle
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=x_valid, y_valid=y_valid_mono, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        remove_dir(model_dir)

        # Validation with y_train & y_valid of shape 2
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, np.expand_dims(y_train_mono, 1), x_valid=x_valid, y_valid=np.expand_dims(y_valid_mono, 1), with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        remove_dir(model_dir)

        # Missing targets in y_valid
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=x_valid, y_valid=y_valid_mono_missing, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1, 2])
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        remove_dir(model_dir)

        # Multi-labels
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['test1', 'test2', 'test3'])
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        remove_dir(model_dir)

        # with valid
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=x_valid, y_valid=y_valid_multi, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['test1', 'test2', 'test3'])
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        remove_dir(model_dir)

        # No shuffle
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=x_valid, y_valid=y_valid_multi, with_shuffle=False)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), ['test1', 'test2', 'test3'])
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        remove_dir(model_dir)


        ###########
        # Test continue training

        # Test mono-label nominal case
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # Second fit
        y_train_different_order = np.array([1, 0, 0, 1, 2] * 100)
        model.fit(x_train[:50], y_train_different_order[:50], x_valid=None, y_valid=None, with_shuffle=True)
        # We do not save on purpose
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # third fit
        model.fit(x_train[50:], y_train_mono[50:], x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        # Fourth fit
        model.fit(x_train[50:], y_train_mono[50:], x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 4)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_3.json')))
        model_dir_4 = model.model_dir
        self.assertNotEqual(model_dir_3, model_dir_4)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        remove_dir(model_dir_4)

        # Test data errors mono-label
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_mono, x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # Second fit
        y_train_mono_fake = np.array([3, 1, 0, 1, 2] * 100)
        with self.assertRaises(AssertionError):
            model.fit(x_train[:50], y_train_mono_fake[:50], x_valid=None, y_valid=None, with_shuffle=True)
        remove_dir(model_dir)

        # Test multi-labels nominal case
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # Second fit
        model.fit(x_train[:50], y_train_multi[:50], x_valid=None, y_valid=None, with_shuffle=True)
        # We do not save on purpose
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 2)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        model_dir_2 = model.model_dir
        self.assertNotEqual(model_dir, model_dir_2)
        # third fit
        model.fit(x_train[50:], y_train_multi[50:], x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 3)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        model_dir_3 = model.model_dir
        self.assertNotEqual(model_dir_2, model_dir_3)
        # Fourth fit
        model.fit(x_train[50:], y_train_multi[50:], x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 4)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_1.json')))
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_2.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations_fit_3.json')))
        model_dir_4 = model.model_dir
        self.assertNotEqual(model_dir_3, model_dir_4)
        remove_dir(model_dir)
        remove_dir(model_dir_2)
        remove_dir(model_dir_3)
        remove_dir(model_dir_4)


        ###########
        # Misc
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train_long, y_train_long, x_valid=None, y_valid=None, with_shuffle=True)
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertEqual(sorted(model.list_classes), [0, 1])
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        remove_dir(model_dir)

        # Test data errors multi-labels
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        model.fit(x_train, y_train_multi, x_valid=None, y_valid=None, with_shuffle=True)
        model.save()
        self.assertTrue(model.trained)
        self.assertEqual(model.nb_fit, 1)
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json')))
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # Second fit
        y_train_multi_fake = pd.DataFrame({'test3': [0, 0, 0, 1, 0] * 100, 'test2': [1, 0, 0, 0, 0] * 100, 'test1': [0, 0, 0, 1, 0] * 100})
        with self.assertRaises(AssertionError):
            model.fit(x_train[:50], y_train_multi_fake[:50], x_valid=None, y_valid=None, with_shuffle=True)
        remove_dir(model_dir)

    def test03_model_huggingface_predict(self):
        '''Test of the method predict of {{package_name}}.models_training.model_huggingface.ModelHuggingFace'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"] * 100)
        x_valid = np.array(["cela est un test", "ni cela", "non plus", "ici test", "là, rien de rien!"] * 100)
        x_valid_long = np.array(["cela est un test " * 1000] * 10)
        y_train_mono = np.array([0, 1, 0, 1, 2] * 100)
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0] * 100, 'test2': [1, 0, 0, 0, 0] * 100, 'test3': [0, 0, 0, 1, 0] * 100})
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono)
        preds = model.predict(x_valid, return_proba=False)
        self.assertEqual(preds.shape, (len(x_valid),))
        preds = model.predict('test', return_proba=False)
        self.assertEqual(preds, model.predict(['test'], return_proba=False)[0])
        proba = model.predict(x_valid, return_proba=True)
        self.assertEqual(proba.shape, (len(x_valid), 3))
        proba = model.predict('test', return_proba=True)
        self.assertEqual([elem for elem in proba], [elem for elem in model.predict(['test'], return_proba=True)[0]])
        # Predict long sentence (just check no errors)
        model.predict(x_valid_long)
        remove_dir(model_dir)

        # Multi-labels
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        model.fit(x_train, y_train_multi)
        preds = model.predict(x_valid, return_proba=False)
        self.assertEqual(preds.shape, (len(x_valid), len(cols)))
        preds = model.predict('test', return_proba=False)
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict(['test'], return_proba=False)[0]])
        proba = model.predict(x_valid, return_proba=True)
        self.assertEqual(proba.shape, (len(x_valid), len(cols)))
        proba = model.predict('test', return_proba=True)
        self.assertEqual([elem for elem in proba], [elem for elem in model.predict(['test'], return_proba=True)[0]])
        # Predict long sentence (just check no errors)
        model.predict(x_valid_long)
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
            model.predict('test')
        remove_dir(model_dir)

    def test04_model_huggingface_predict_proba(self):
        '''Test of {{package_name}}.models_training.model_huggingface.ModelHuggingFace.predict_proba'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Set vars
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"] * 100)
        x_test = np.array(["cela est un test", "ni cela", "non plus", "ici test", "là, rien de rien!"] * 100)
        x_test_long = np.array(["cela est un test " * 1000] * 10)
        y_train_mono = np.array([0, 1, 0, 1, 2] * 100)
        y_train_multi = pd.DataFrame({'test1': [0, 0, 0, 1, 0] * 100, 'test2': [1, 0, 0, 0, 0] * 100, 'test3': [0, 0, 0, 1, 0] * 100})
        cols = ['test1', 'test2', 'test3']

        # Mono-label
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_test)
        self.assertEqual(preds.shape, (len(x_test), 3))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        # Predict long sentence (just check no errors)
        model.predict_proba(x_test_long)
        remove_dir(model_dir)


        # Multi-labels
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_test)
        self.assertEqual(preds.shape, (len(x_test), len(cols)))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        # Predict long sentence (just check no errors)
        model.predict_proba(x_test_long)
        remove_dir(model_dir)


        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test05_model_huggingface_prepare_x_train(self):
        '''Test of {{package_name}}.models_training.model_huggingface.ModelHuggingFace._prepare_x_train'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        x_train = ['test titi toto', 'toto', 'titi test test toto']
        y_train_dummies = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

        # Nominal case
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        x_train_prepared = model._prepare_x_train(x_train, y_train_dummies)
        # We can't easily test the results, too many dependences
        self.assertEqual(x_train_prepared.shape[0], len(x_train))  # Same nb of lines
        self.assertTrue('text' in x_train_prepared.features)
        self.assertTrue('label' in x_train_prepared.features)
        self.assertTrue('input_ids' in x_train_prepared.features)
        self.assertTrue('attention_mask' in x_train_prepared.features)
        remove_dir(model_dir)

    def test06_model_huggingface_prepare_x_valid(self):
        '''Test of {{package_name}}.models_training.model_huggingface.ModelHuggingFace._prepare_x_valid'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        x_valid = ['test titi toto', 'toto', 'titi test test toto']
        y_valid_dummies = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

        # Nominal case
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        x_valid_prepared = model._prepare_x_valid(x_valid, y_valid_dummies)
        # We can't easily test the results, too many dependences
        self.assertEqual(x_valid_prepared.shape[0], len(x_valid))  # Same nb of lines
        self.assertTrue('text' in x_valid_prepared.features)
        self.assertTrue('label' in x_valid_prepared.features)
        self.assertTrue('input_ids' in x_valid_prepared.features)
        self.assertTrue('attention_mask' in x_valid_prepared.features)
        remove_dir(model_dir)

    def test07_model_huggingface_prepare_x_test(self):
        '''Test of {{package_name}}.models_training.model_huggingface.ModelHuggingFace._prepare_x_test'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        x_test = ['test titi toto', 'toto', 'titi test test toto']

        # Nominal case - tokenizer not set
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        # TMP FIX: init. tokenizer
        model.tokenizer = model._get_tokenizer()
        x_test_prepared = model._prepare_x_test(x_test)
        # We can't easily test the results, too many dependences
        self.assertEqual(x_test_prepared.shape[0], len(x_test))  # Same nb of lines
        self.assertTrue('text' in x_test_prepared.features)
        self.assertTrue('label' not in x_test_prepared.features)
        self.assertTrue('input_ids' in x_test_prepared.features)
        self.assertTrue('attention_mask' in x_test_prepared.features)
        remove_dir(model_dir)

    def test08_model_huggingface_tokenize_function(self):
        '''Test of {{package_name}}.models_training.model_huggingface.ModelHuggingFace._tokenize_function'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        batch = Batch(data={'text': 'titi test test toto', 'label': 1})

        # Nominal case - tokenizer not set
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        # TMP FIX: init. tokenizer
        model.tokenizer = model._get_tokenizer()
        encoded_batch = model._tokenize_function(batch)
        # We can't easily test the results, too many dependences
        self.assertTrue(type(encoded_batch) == BatchEncoding)
        self.assertTrue('input_ids' in encoded_batch.keys())
        self.assertTrue('attention_mask' in encoded_batch.keys())
        remove_dir(model_dir)

    def test09_model_huggingface_get_model(self):
        '''Test of {{package_name}}.models_training.model_huggingface.ModelHuggingFace._get_model'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)

        # Nominal case
        model.list_classes = ['a', 'b']  # We force the creation of a list of classes
        model_res = model._get_model()
        self.assertTrue(isinstance(model_res, PreTrainedModel))

        # Clean
        remove_dir(model_dir)

    def test10_model_huggingface_get_tokenizer(self):
        '''Test of {{package_name}}.models_training.model_huggingface.ModelHuggingFace._get_tokenizer'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)

        # Nominal case
        tokenizer = model._get_tokenizer()
        self.assertTrue(isinstance(tokenizer, PreTrainedTokenizerBase))

        # With model_path
        # TODO

        # Clean
        remove_dir(model_dir)

    def test11_model_huggingface_compute_metrics_mono_label(self):
        '''Test of {{package_name}}.models_training.model_huggingface.ModelHuggingFace._compute_metrics_mono_label'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)

        # Data to compute
        logits = np.array([[0.2, 0.8], [0.51, 0.49], [0.3, 0.7], [0.1, 0.9], [0.8, 0.2]])
        label_ids = np.array([1, 0, 0, 1, 0])
        weight_0 = 3 / 5
        weight_1 = 2 / 5
        eval_pred = EvalPrediction(logits, label_ids)

        # Nominal case
        metrics = model._compute_metrics_mono_label(eval_pred)
        self.assertTrue('accuracy' in metrics.keys())
        self.assertAlmostEqual(metrics['accuracy'], 4/5)
        self.assertTrue('weighted_precision' in metrics.keys())
        precision_0 = 2 / 2
        precision_1 = 2 / 3
        precision = (precision_0 * weight_0) + (precision_1 * weight_1)  # weighted precision
        self.assertAlmostEqual(metrics['weighted_precision'], precision)
        self.assertTrue('weighted_recall' in metrics.keys())
        recall_0 = 2 / 3
        recall_1 = 2 / 2
        recall = (recall_0 * weight_0) + (recall_1 * weight_1)  # weighted recall
        self.assertAlmostEqual(metrics['weighted_recall'], recall)
        self.assertTrue('weighted_f1' in metrics.keys())
        f1_0 =  2 * precision_0 * recall_0 / (precision_0 + recall_0)
        f1_1 =  2 * precision_1 * recall_1 / (precision_1 + recall_1)
        f1 = (f1_0 * weight_0) + (f1_1 * weight_1)  # weighted f1
        self.assertAlmostEqual(metrics['weighted_f1'], f1)

        # Clean
        remove_dir(model_dir)

    def test12_model_huggingface_compute_metrics_multi_label(self):
        '''Test of {{package_name}}.models_training.model_huggingface.ModelHuggingFace._compute_metrics_multi_label'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)

        # Data to compute
        # proba > 0.5 if logit > 0
        logits = np.array([[-1, 1], [0.1, -0.1], [-1, 1], [-1, -1], [1, 1]])
        # predictions = np.array([[0, 1], [1, 0], [0, 1], [0, 0], [1, 1]])
        label_ids = np.array([[0, 1], [1, 1], [1, 1], [0, 0], [0, 1]])
        weight_0 = 2 / 5
        weight_1 = 4 / 5
        total = weight_0 + weight_1
        weight_0 = weight_0 / total
        weight_1 = weight_1 / total
        eval_pred = EvalPrediction(logits, label_ids)

        # Nominal case
        metrics = model._compute_metrics_multi_label(eval_pred)
        self.assertTrue('accuracy' in metrics.keys())
        self.assertAlmostEqual(metrics['accuracy'], 2/5)
        self.assertTrue('weighted_precision' in metrics.keys())
        precision_0 = 1 / 2
        precision_1 = 3 / 3
        precision = (precision_0 * weight_0) + (precision_1 * weight_1)  # weighted precision
        self.assertAlmostEqual(metrics['weighted_precision'], precision)
        self.assertTrue('weighted_recall' in metrics.keys())
        recall_0 = 1 / 2
        recall_1 = 3 / 4
        recall = (recall_0 * weight_0) + (recall_1 * weight_1)  # weighted recall
        self.assertAlmostEqual(metrics['weighted_recall'], recall)
        self.assertTrue('weighted_f1' in metrics.keys())
        f1_0 =  2 * precision_0 * recall_0 / (precision_0 + recall_0)
        f1_1 =  2 * precision_1 * recall_1 / (precision_1 + recall_1)
        f1 = (f1_0 * weight_0) + (f1_1 * weight_1)  # weighted f1
        self.assertAlmostEqual(metrics['weighted_f1'], f1)

        # Clean
        remove_dir(model_dir)

    def test13_model_huggingface_save(self):
        '''Test of the method save of {{package_name}}.models_training.model_huggingface.ModelHuggingFace'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelHuggingFace(model_dir=model_dir, batch_size=8, epochs=2, multi_label=False)
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_model', 'pytorch_model.bin')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'hf_tokenizer', 'tokenizer.json'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_huggingface.pkl')))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertTrue('package_version' in configs.keys())
        self.assertEqual(configs['package_version'], utils.get_package_version())
        self.assertTrue('model_name' in configs.keys())
        self.assertTrue('model_dir' in configs.keys())
        self.assertTrue('trained' in configs.keys())
        self.assertTrue('nb_fit' in configs.keys())
        self.assertTrue('list_classes' in configs.keys())
        self.assertTrue('dict_classes' in configs.keys())
        self.assertTrue('x_col' in configs.keys())
        self.assertTrue('y_col' in configs.keys())
        self.assertTrue('multi_label' in configs.keys())
        self.assertTrue('level_save' in configs.keys())
        self.assertTrue('librairie' in configs.keys())
        self.assertEqual(configs['librairie'], 'huggingface')
        self.assertTrue('batch_size' in configs.keys())
        self.assertTrue('epochs' in configs.keys())
        self.assertTrue('validation_split' in configs.keys())
        self.assertTrue('transformer_name' in configs.keys())
        self.assertTrue('transformer_params' in configs.keys())
        self.assertTrue('trainer_params' in configs.keys())
        self.assertTrue('_get_model' in configs.keys())
        self.assertTrue('_get_tokenizer' in configs.keys())

        # Clean
        remove_dir(model_dir)

    def test14_model_huggingface_init_new_instance_from_configs(self):
        '''Test of the method _init_new_instance_from_configs of {{package_name}}.models_training.models_tensorflow.model_huggingface.ModelHuggingFace'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case
        model = ModelHuggingFace(model_dir=model_dir)
        model.save(json_data={'test': 8})
        configs = model.load_configs(model_dir=model_dir)

        new_model = ModelHuggingFace._init_new_instance_from_configs(configs=configs)
        self.assertTrue(isinstance(new_model, ModelHuggingFace))
        self.assertEqual(new_model.nb_fit, 0)
        self.assertFalse(new_model.trained)
        for attribute in ['x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save', 'batch_size', 'epochs', 'patience',
                          'transformer_name', 'model_max_length']:
            self.assertEqual(getattr(model, attribute), getattr(new_model, attribute))
        for attribute in ['validation_split']:
            self.assertAlmostEqual(getattr(model, attribute), getattr(new_model, attribute))
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Check by changing some attributes
        model = ModelHuggingFace(model_dir=model_dir)
        model.nb_fit = 2
        model.trained = True
        model.x_col = 'coucou'
        model.y_col = 'coucou_2'
        model.list_classes = ['class_1', 'class_2', 'class_3']
        model.dict_classes = {0: 'class_1', 1: 'class_2', 2: 'class_3'}
        model.multi_label = True
        model.level_save = 'MEDIUM'
        model.batch_size = 13
        model.epochs = 42
        model.validation_split = 0.3
        model.patience = 15
        model.transformer_name = 'coucou'
        model.model_max_length = 12
        model.trainer_params = {
                'weight_decay': 0.0,
                'evaluation_strategy': 'epoch',
                'load_best_model_at_end': True
            }
        model.transformer_params = {
                'weight_decay': 0.0,
                'evaluation_strategy': 'epoch',
                'load_best_model_at_end': True
            }
        model.save(json_data={'test': 8})
        configs = model.load_configs(model_dir=model_dir)

        new_model = ModelHuggingFace._init_new_instance_from_configs(configs=configs)
        self.assertTrue(isinstance(new_model, ModelHuggingFace))
        self.assertEqual(new_model.nb_fit, 2)
        self.assertTrue(new_model.trained)
        for attribute in ['x_col', 'y_col', 'list_classes', 'dict_classes', 'multi_label', 'level_save', 'batch_size', 'epochs', 'patience',
                          'transformer_name', 'model_max_length']:
            self.assertEqual(getattr(model, attribute), getattr(new_model, attribute))
        for attribute in ['validation_split']:
            self.assertAlmostEqual(getattr(model, attribute), getattr(new_model, attribute))
        self.assertEqual(set(model.trainer_params), set(new_model.trainer_params))
        self.assertEqual(model.trainer_params['evaluation_strategy'], new_model.trainer_params['evaluation_strategy'])
        self.assertAlmostEqual(model.trainer_params['weight_decay'], new_model.trainer_params['weight_decay'])
        self.assertEqual(model.trainer_params['load_best_model_at_end'], new_model.trainer_params['load_best_model_at_end'])
        self.assertEqual(set(model.transformer_params), set(new_model.transformer_params))
        self.assertEqual(model.transformer_params['evaluation_strategy'], new_model.transformer_params['evaluation_strategy'])
        self.assertAlmostEqual(model.transformer_params['weight_decay'], new_model.transformer_params['weight_decay'])
        self.assertEqual(model.transformer_params['load_best_model_at_end'], new_model.transformer_params['load_best_model_at_end'])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

    def test15_model_huggingface_load_standalone_files(self):
        '''Test of the method _load_standalone_files of {{package_name}}.models_training.models_tensorflow.model_huggingface.ModelHuggingFace'''
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)

        # Nominal case with default_model_dir
        model = ModelHuggingFace(model_dir=model_dir, epochs=2)
        x_train = ['coucou', 'coucou_2', 'coucou_3']
        y_train = ['class_1', 'class_2', 'class_1']
        x_valid = ['coucou_4', 'coucou_5', 'coucou_6']
        y_valid = ['class_1', 'class_2', 'class_1']
        model.fit(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)
        model.save(json_data={'test': 8})

        configs = model.load_configs(model_dir=model_dir)
        new_model = ModelHuggingFace._init_new_instance_from_configs(configs=configs)
        new_model._load_standalone_files(default_model_dir=model_dir)
        self.assertTrue(isinstance(new_model.model, type(model.model)))
        self.assertTrue(isinstance(new_model.tokenizer, type(model.tokenizer)))
        self.assertEqual(model.model.config.n_layers, new_model.model.config.n_layers)
        self.assertEqual(model.model.config.vocab_size, new_model.model.config.vocab_size)
        self.assertEqual(model.model.config.model_type, new_model.model.config.model_type)

        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Nominal case with paths
        model = ModelHuggingFace(model_dir=model_dir, epochs=2)
        x_train = ['coucou', 'coucou_2', 'coucou_3']
        y_train = ['class_1', 'class_2', 'class_1']
        x_valid = ['coucou_4', 'coucou_5', 'coucou_6']
        y_valid = ['class_1', 'class_2', 'class_1']
        model.fit(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)
        model.save(json_data={'test': 8})

        configs = model.load_configs(model_dir=model_dir)
        new_model = ModelHuggingFace._init_new_instance_from_configs(configs=configs)
        new_model._load_standalone_files(hf_model_dir_path=os.path.join(model_dir, "hf_model"),
                                         hf_tokenizer_dir_path=os.path.join(model_dir, "hf_tokenizer"))
        self.assertTrue(isinstance(new_model.model, type(model.model)))
        self.assertTrue(isinstance(new_model.tokenizer, type(model.tokenizer)))
        self.assertEqual(model.model.config.n_layers, new_model.model.config.n_layers)
        self.assertEqual(model.model.config.vocab_size, new_model.model.config.vocab_size)
        self.assertEqual(model.model.config.model_type, new_model.model.config.model_type)

        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Errors
        model = ModelHuggingFace(model_dir=model_dir, epochs=2)
        x_train = ['coucou', 'coucou_2', 'coucou_3']
        y_train = ['class_1', 'class_2', 'class_1']
        x_valid = ['coucou_4', 'coucou_5', 'coucou_6']
        y_valid = ['class_1', 'class_2', 'class_1']
        model.fit(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)
        model.save(json_data={'test': 8})

        configs = model.load_configs(model_dir=model_dir)
        new_model = ModelHuggingFace._init_new_instance_from_configs(configs=configs)
        with self.assertRaises(ValueError):
            new_model._load_standalone_files(hf_tokenizer_dir_path=os.path.join(model_dir, "hf_tokenizer"))
        with self.assertRaises(ValueError):
            new_model._load_standalone_files(hf_model_dir_path=os.path.join(model_dir, "hf_model"))
        with self.assertRaises(FileNotFoundError):
            new_model._load_standalone_files(hf_model_dir_path=os.path.join(model_dir, "hf_model2"),
                                         hf_tokenizer_dir_path=os.path.join(model_dir, "hf_tokenizer"))
        with self.assertRaises(FileNotFoundError):
            new_model._load_standalone_files(hf_model_dir_path=os.path.join(model_dir, "hf_model"),
                                         hf_tokenizer_dir_path=os.path.join(model_dir, "hf_tokenizer2"))

        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        
# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
