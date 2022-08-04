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
import torch
import shutil
import pickle
import numpy as np
import pandas as pd

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer

from {{package_name}} import utils
from {{package_name}}.models_training.model_pytorch_transformers import ModelPyTorchTransformers, TaskClass

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class ModelPyTorchTransformersTests(unittest.TestCase):
    '''Main class to test model_pytorch_transformers'''

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

    def test01_model_pytorch_transformers_init(self):
        '''Test of {{package_name}}.models_training.model_pytorch_transformers.ModelPyTorchTransformers.__init__'''
        true_transformer_name = 'flaubert/flaubert_small_cased'
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        # Init., test all parameters
        model = ModelPyTorchTransformers(model_dir=model_dir, transformer_name=true_transformer_name)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertTrue(model.tokenizer is not None)
        # We test display_if_gpu_activated and _is_gpu_activated just by calling them
        self.assertTrue(type(model._is_gpu_activated()) == bool)
        model.display_if_gpu_activated()
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformers(model_dir=model_dir, transformer_name=true_transformer_name, max_sequence_length=20)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.max_sequence_length, 20)
        self.assertTrue(model.tokenizer is not None)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformers(model_dir=model_dir, transformer_name=true_transformer_name, tokenizer_special_tokens=('totototototototototo'))
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.tokenizer_special_tokens, ('totototototototototo'))
        self.assertTrue('totototototototototo' in model.tokenizer.get_vocab())
        self.assertTrue(model.tokenizer is not None)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformers(model_dir=model_dir, transformer_name=true_transformer_name, padding='longest')
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.padding, 'longest')
        self.assertTrue(model.tokenizer is not None)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformers(model_dir=model_dir, transformer_name=true_transformer_name, truncation=False)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.truncation, False)
        self.assertTrue(model.tokenizer is not None)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformers(model_dir=model_dir, transformer_name=true_transformer_name, batch_size=9)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.batch_size, 9)
        self.assertTrue(model.tokenizer is not None)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformers(model_dir=model_dir, transformer_name=true_transformer_name, epochs=9)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.epochs, 9)
        self.assertTrue(model.tokenizer is not None)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformers(model_dir=model_dir, transformer_name=true_transformer_name, validation_split=0.3)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.validation_split, 0.3)
        self.assertTrue(model.tokenizer is not None)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformers(model_dir=model_dir, transformer_name=true_transformer_name, patience=18)
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.patience, 18)
        self.assertTrue(model.tokenizer is not None)
        remove_dir(model_dir)

        #
        model = ModelPyTorchTransformers(model_dir=model_dir, transformer_name=true_transformer_name, pytorch_params={'titi': 'toto'})
        self.assertEqual(model.model_dir, model_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertEqual(model.pytorch_params, {'titi': 'toto'})
        self.assertTrue(model.tokenizer is not None)
        remove_dir(model_dir)

    def test02_model_pytorch_transformers_predict_proba(self):
        '''Test of {{package_name}}.models_training.model_pytorch_transformers.ModelPyTorchTransformers.predict_proba'''

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
        model = ModelPyTorchTransformers(model_dir=model_dir, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=10,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=False)
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Mono-label - The other way around (to check the management of CPU/GPU)
        model = ModelPyTorchTransformers(model_dir=model_dir, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=10,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=False)
        model.fit(x_train, y_train_mono)
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), 3))
        remove_dir(model_dir)

        # Multi-labels
        model = ModelPyTorchTransformers(model_dir=model_dir, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=10,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=True)
        model.fit(x_train, y_train_multi[cols])
        preds = model.predict_proba(x_train)
        self.assertEqual(preds.shape, (len(x_train), len(cols)))
        preds = model.predict_proba('test')
        self.assertEqual([elem for elem in preds], [elem for elem in model.predict_proba(['test'])[0]])
        remove_dir(model_dir)

        # Model needs to be fitted
        with self.assertRaises(AttributeError):
            model = ModelPyTorchTransformers(model_dir=model_dir, batch_size=8, epochs=2,
                                             patience=10, max_sequence_length=10,
                                             transformer_name='flaubert/flaubert_small_cased',
                                             multi_label=False)
            model.predict_proba('test')
        remove_dir(model_dir)

    def test03_model_pytorch_transformers_convert_inputs(self):
        '''Test of {{package_name}}.models_training.model_pytorch_transformers.ModelPyTorchTransformers._convert_inputs'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        max_sequence_length = 10
        model = ModelPyTorchTransformers(model_dir=model_dir, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=max_sequence_length,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=False)

        # Nominal case
        x_train = ['test titi toto', 'toto', 'titi test test toto']
        y_train = [0, 1, 0]
        all_input_ids, all_input_mask, all_label = model._convert_inputs(x_train)
        # We can't easily test the results, too many dependences
        self.assertEqual(all_input_ids.shape[0], len(x_train))
        self.assertEqual(all_input_ids.shape[1], max_sequence_length)
        self.assertEqual(all_input_mask.shape[0], len(x_train))
        self.assertEqual(all_input_mask.shape[1], max_sequence_length)
        self.assertTrue(all_label is None)

        # With labels
        all_input_ids, all_input_mask, all_label = model._convert_inputs(x_train, y=y_train)
        self.assertEqual(all_input_ids.shape[0], len(x_train))
        self.assertEqual(all_input_ids.shape[1], max_sequence_length)
        self.assertEqual(all_input_mask.shape[0], len(x_train))
        self.assertEqual(all_input_mask.shape[1], max_sequence_length)
        self.assertTrue(torch.all(torch.eq(all_label, torch.tensor(y_train))))

        # Clean
        remove_dir(model_dir)

    def test04_model_pytorch_transformers_get_train_dataloader(self):
        '''Test of {{package_name}}.models_training.model_pytorch_transformers.ModelPyTorchTransformers._get_train_dataloader'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        max_sequence_length = 10
        model = ModelPyTorchTransformers(model_dir=model_dir, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=max_sequence_length,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=False)

        # Nominal case
        x_train = ['test titi toto', 'toto', 'titi test test toto']
        y_train_dummies = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        batch_size = 10
        train_dl = model._get_train_dataloader(batch_size, x_train, y_train_dummies)
        self.assertEqual(train_dl.batch_size, batch_size)
        self.assertTrue('RandomSampler' in train_dl.sampler.__str__())
        self.assertTrue(train_dl.sampler is not None)
        self.assertEqual(len(train_dl._get_iterator().next()), 3)

        # Check raise errors
        with self.assertRaises(ValueError):
            model._get_train_dataloader(batch_size, x_train, None)

        # Clean
        remove_dir(model_dir)

    def test05_model_pytorch_transformers_get_test_dataloader(self):
        '''Test of {{package_name}}.models_training.model_pytorch_transformers.ModelPyTorchTransformers._get_test_dataloader'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        max_sequence_length = 10
        model = ModelPyTorchTransformers(model_dir=model_dir, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=max_sequence_length,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=False)

        # Nominal case
        x_test = ['test titi toto', 'toto', 'titi test test toto']
        y_test_dummies = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        batch_size = 10
        test_dl = model._get_test_dataloader(batch_size, x_test, y_test_dummies)
        self.assertEqual(test_dl.batch_size, batch_size)
        self.assertTrue('SequentialSampler' in test_dl.sampler.__str__())
        self.assertEqual(len(test_dl._get_iterator().next()), 3)

        # Fonctionnement sans y_test_dummies
        test_dl = model._get_test_dataloader(batch_size, x_test, None)
        self.assertEqual(test_dl.batch_size, batch_size)
        self.assertTrue('SequentialSampler' in test_dl.sampler.__str__())
        self.assertEqual(len(test_dl._get_iterator().next()), 2)

        # Clean
        remove_dir(model_dir)

    def test06_model_pytorch_transformers_get_model(self):
        '''Test of {{package_name}}.models_training.model_pytorch_transformers.ModelPyTorchTransformers._get_model'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        model = ModelPyTorchTransformers(model_dir=model_dir, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=10,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=False)

        # Nominal case
        x_train = ['test titi toto', 'toto', 'titi test test toto']
        model.list_classes = ['a', 'b']  # We force the creation of a list of classes ...
        model.dict_classes = {0: 'a', 1: 'b'}  # ... and of the corresponding dict
        model_res = model._get_model()
        self.assertTrue(type(model_res) == TaskClass)

        # With train_dataloader_size
        tokenizer = Tokenizer()
        model_res = model._get_model(train_dataloader_size=28)
        self.assertTrue(type(model_res) == TaskClass)

        # Clean
        remove_dir(model_dir)

    def test07_model_pytorch_transformers_save(self):
        '''Test of the method save of {{package_name}}.models_training.model_pytorch_transformers.ModelPyTorchTransformers'''

        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        remove_dir(model_dir)
        max_sequence_length = 18
        tokenizer_special_tokens = ('taratata', )
        padding = 'longest'
        truncation = False
        pytorch_params = {'toto': 'titi'}

        # Nominal case - without tokenizer
        model = ModelPyTorchTransformers(model_dir=model_dir, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=max_sequence_length,
                                         tokenizer_special_tokens=tokenizer_special_tokens,
                                         padding=padding, truncation=truncation,
                                         pytorch_params=pytorch_params,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=False)

        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_model.ckpt'))) -> no model trained
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'model_pytorch_transformers.pkl')))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertEqual(configs['max_sequence_length'], max_sequence_length)
        self.assertEqual(configs['tokenizer_special_tokens'], list(tokenizer_special_tokens)) # casted in list when saved
        self.assertEqual(configs['padding'], padding)
        self.assertEqual(configs['truncation'], truncation)
        self.assertEqual(configs['pytorch_params']['toto'], 'titi')
        remove_dir(model_dir)

        # Nominal case - level_save = 'LOW'
        model = ModelPyTorchTransformers(model_dir=model_dir, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=max_sequence_length,
                                         tokenizer_special_tokens=tokenizer_special_tokens,
                                         padding=padding, truncation=truncation,
                                         pytorch_params=pytorch_params,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=False, level_save='LOW')
        model.save(json_data={'test': 8})
        self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'configurations.json')))
        # self.assertTrue(os.path.exists(os.path.join(model.model_dir, 'best_model.ckpt'))) -> no model trained
        self.assertFalse(os.path.exists(os.path.join(model.model_dir, 'model_pytorch_transformers.pkl')))
        with open(os.path.join(model.model_dir, 'configurations.json'), 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        self.assertEqual(configs['test'], 8)
        self.assertEqual(configs['max_sequence_length'], max_sequence_length)
        self.assertEqual(configs['tokenizer_special_tokens'], list(tokenizer_special_tokens)) # casted in list when saved
        self.assertEqual(configs['padding'], padding)
        self.assertEqual(configs['truncation'], truncation)
        self.assertEqual(configs['pytorch_params']['toto'], 'titi')
        remove_dir(model_dir)

    def test08_model_pytorch_transformers_reload_model(self):
        '''Test of the method reload_model of {{package_name}}.models_training.model_pytorch_transformers.ModelPyTorchTransformers'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])
        y_train_mono = np.array(['non', 'oui', 'non', 'oui', 'non'])
        model = ModelPyTorchTransformers(model_dir=model_dir, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=10,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=False)
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload pytorch
        checkpoint_path = os.path.join(model.model_dir, 'best_model.ckpt')
        reloaded_model = model.reload_model(checkpoint_path)
        reloaded_model.freeze()

        # Check same preds
        x_val = ['test', 'toto', 'titi']
        initial_probas = model.predict_proba(x_val)
        input_ids, attention_mask, _ = model._convert_inputs(x_val, None)
        if torch.cuda.is_available():
            reloaded_model.to('cuda')
            input_ids = model.model.to_device(input_ids)
            attention_mask = model.model.to_device(attention_mask)
        else:
            reloaded_model.to('cpu')
        logits_torch = reloaded_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = np.vstack([logit.cpu() for logit in logits_torch])
        new_probas = model.model.get_probas_from_logits(logits)
        self.assertEqual([list(_) for _ in new_probas], [list(_) for _ in initial_probas])

        remove_dir(model_dir)

    def test09_test_model_pytorch_transformers_reload_from_standalone(self):
        '''Test of the method {{package_name}}.models_training.model_pytorch_transformers.ModelPyTorchTransformers.reload'''

        # Create model
        model_dir = os.path.join(os.getcwd(), 'model_test_123456789')
        x_train = np.array(["ceci est un test", "pas cela", "cela non plus", "ici test", "là, rien!"])
        x_test = np.array(["ceci est un coucou", "pas lui", "lui non plus", "ici coucou", "là, rien!"])
        y_train_mono = np.array(['non', 'oui', 'non', 'oui', 'non'])
        model = ModelPyTorchTransformers(model_dir=model_dir, batch_size=8, epochs=2,
                                         patience=10, max_sequence_length=10,
                                         transformer_name='flaubert/flaubert_small_cased',
                                         multi_label=False)
        model.fit(x_train, y_train_mono)
        model.save()

        # Reload
        conf_path = os.path.join(model.model_dir, "configurations.json")
        checkpoint_path = os.path.join(model.model_dir, "best_model.ckpt")
        new_model = ModelPyTorchTransformers()
        new_model.reload_from_standalone(conf_path, checkpoint_path)

        # Test
        self.assertEqual(model.model_name, new_model.model_name)
        self.assertEqual(model.x_col, new_model.x_col)
        self.assertEqual(model.y_col, new_model.y_col)
        self.assertEqual(model.list_classes, new_model.list_classes)
        self.assertEqual(model.dict_classes, new_model.dict_classes)
        self.assertEqual(model.multi_label, new_model.multi_label)
        self.assertEqual(model.level_save, new_model.level_save)
        self.assertEqual(model.nb_fit, new_model.nb_fit)
        self.assertEqual(model.trained, new_model.trained)
        self.assertEqual(model.batch_size, new_model.batch_size)
        self.assertEqual(model.epochs, new_model.epochs)
        self.assertEqual(model.validation_split, new_model.validation_split)
        self.assertEqual(model.patience, new_model.patience)
        self.assertEqual(model.embedding_name, new_model.embedding_name)
        self.assertEqual(model.transformer_name, new_model.transformer_name)
        self.assertEqual(model.max_sequence_length, new_model.max_sequence_length)
        self.assertEqual(model.tokenizer_special_tokens, new_model.tokenizer_special_tokens)
        self.assertEqual(model.padding, new_model.padding)
        self.assertEqual(model.truncation, new_model.truncation)
        self.assertEqual(model.pytorch_params, new_model.pytorch_params)
        self.assertEqual([list(_) for _ in model.predict_proba(x_test)], [list(_) for _ in new_model.predict_proba(x_test)])
        remove_dir(model_dir)
        remove_dir(new_model.model_dir)

        # Check errors
        with self.assertRaises(FileNotFoundError):
            new_model = ModelPyTorchTransformers()
            new_model.reload_from_standalone(configuration_path='toto.json', checkpoint_path=checkpoint_path)
        with self.assertRaises(FileNotFoundError):
            new_model = ModelPyTorchTransformers()
            new_model.reload_from_standalone(configuration_path=conf_path, checkpoint_path='toto.ckpt')


# Perform tests
if __name__ == '__main__':
    # Run only if transformer loaded locally
    transformer_path = os.path.join(utils.get_transformers_path(), 'flaubert', 'flaubert_small_cased')
    if os.path.exists(transformer_path):
        # Start tests
        unittest.main()
    else:
        print(f"{transformer_path} does not exists. We won't start the tests for model_pytorch_transformers.py")
        print("Remainder : this is still experimental")
