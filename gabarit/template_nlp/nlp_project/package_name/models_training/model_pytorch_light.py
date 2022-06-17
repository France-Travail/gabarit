#!/usr/bin/env python3

## Model pytorch transformer -- LIGHT (for preds only)
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
#
# Classes :
# - ModelPyTorchTransformersLight -> Model for predictions via tranformers pytorch


import os
import json
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from typing import Union, Any

import torch
from torch.nn import Softmax, Sigmoid
from transformers import CONFIG_NAME, WEIGHTS_NAME, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from {{package_name}} import utils
from {{package_name}}.models_training.model_pytorch import ModelPyTorch

sns.set(style="darkgrid")
tqdm.pandas()


class ModelPyTorchTransformersLight(ModelPyTorch):
    '''Model for predictions via tranformers pytorch
    Version light - only predictions
    '''

    _default_name = 'model_pytorch_light'

    def __init__(self, max_sequence_length: int = 256, tokenizer_special_tokens: Union[tuple, None] = None,
                 padding: str = "max_length", truncation: bool = True, **kwargs):
        '''Initialization of the class (see ModelClass & ModelPyTorch for more arguments)

        Args:
            transformer_name (str): Name of the transformer to use
        Kwargs:
            max_sequence_length (int): Maximum number of words per sequence (ie. sentences)

            tokenizer_special_tokens (tuple): Set of "special tokens" for the tokenizer
            padding (str): Tokenizer's padding strategy
            truncation (bool): Tokenizer's padding truncation
        '''
        # Init.
        super().__init__(**kwargs)

        # Get logger (must be done after super init)
        self.logger = logging.getLogger(__name__)

        # Params
        self.max_sequence_length = max_sequence_length
        if tokenizer_special_tokens is None:
            tokenizer_special_tokens = tuple()
        self.tokenizer_special_tokens = tokenizer_special_tokens
        self.padding = padding
        self.truncation = truncation
        # Tokenizer set on reload
        self.tokenizer: Any = None

    @utils.data_agnostic_str_to_list
    @utils.trained_needed
    def predict_proba(self, x_test, **kwargs) -> np.ndarray:
        '''Predicts probabilities on the test dataset

        Args:
            x_test (?): Array-like or sparse matrix, shape = [n_samples]
        Returns:
            (np.ndarray): Array, shape = [n_samples, n_classes]
        '''
        # Cast to pd.Series
        x_test = pd.Series(x_test)

        # Get probas
        predicted_proba = np.zeros((x_test.shape[0], len(self.list_classes)))
        for i, x_tmp in enumerate(x_test):
            x = self.tokenizer(x_tmp, padding=self.padding, truncation=self.truncation,
                               max_length=self.max_sequence_length)
            with torch.no_grad():
                logits = self.model(input_ids=torch.tensor([x["input_ids"]], dtype=torch.long), attention_mask=torch.tensor([x["attention_mask"]], dtype=torch.long))['logits']
            if self.multi_label:
                predicted_proba[i] = Sigmoid()(logits[0].detach()).numpy()
            else:
                predicted_proba[i] = Softmax(dim=-1)(logits[0].detach()).numpy()

        return predicted_proba

    def freeze(self) -> None:
        '''Freezes the model'''
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze(self) -> None:
        '''Unfreezes the model'''
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    def save(self, json_data: Union[dict, None] = None) -> None:
        '''Saves the model

        Kwargs:
            json_data (dict): Additional configurations to be saved
        '''
        # Save configuration JSON
        if json_data is None:
            json_data = {}

        # Add specific data
        json_data['max_sequence_length'] = self.max_sequence_length
        json_data['tokenizer_special_tokens'] = self.tokenizer_special_tokens
        json_data['padding'] = self.padding
        json_data['truncation'] = self.truncation

        # Save strategy :
        # 1. save torch model & tokenizer
        # 2. can't pickle torch model, so we drop it, save, and reload it
        ### 1.
        output_model_file = os.path.join(self.model_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(self.model_dir, CONFIG_NAME)
        torch.save(self.model.state_dict(), output_model_file)  # type: ignore
        self.model.config.to_json_file(output_config_file)  # type: ignore
        self.tokenizer.save_vocabulary(self.model_dir)  # type: ignore
        ### 2.
        torch_model = self.model
        tokenizer = self.tokenizer
        self.model = None
        self.tokenizer = None
        super().save(json_data=json_data)
        self.model = torch_model
        self.tokenizer = tokenizer

    def reload_model(self, model_path: str, **kwargs) -> Any:
        '''Reloads a model saved in 'light' format

        Args:
            model_path: Model directory / torch dir
        Returns:
            ?: Torch model
        '''
        model = AutoModelForSequenceClassification.from_pretrained(model_path, config=AutoConfig.from_pretrained(model_path))

        # Set trained to true if not already true
        if not self.trained:
            self.trained = True
            self.nb_fit = 1

        return model

    def reload_tokenizer(self, torch_dir: str) -> Any:
        '''Reloads a tokenizer saved in 'light' format

        Args:
            torch_dir: Tokenizer directory
        Returns:
            ?: Torch tokenizer
        '''
        tokenizer = AutoTokenizer.from_pretrained(torch_dir)
        return tokenizer

    def reload_from_standalone(self, **kwargs) -> None:
        '''Reloads a model from its configuration and "standalones" files
        - /!\\ Experimental /!\\ -

        Kwargs:
            configuration_path (str): Path to configuration file
            torch_dir (str): Torch dir path
        Raises:
            ValueError: If configuration_path is None
            ValueError: If torch_dir is None
            FileNotFoundError: If the object configuration_path is not an existing file
            FileNotFoundError: If the object torch_dir is not an existing file
        '''
        # Retrieve args
        configuration_path = kwargs.get('configuration_path', None)
        torch_dir = kwargs.get('torch_dir', None)

        # Checks
        if configuration_path is None:
            raise ValueError("The argument configuration_path can't be None")
        if torch_dir is None:
            raise ValueError("The argument torch_dir can't be None")
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(f"The file {configuration_path} does not exist")
        if not os.path.exists(torch_dir):
            raise FileNotFoundError(f"The file {torch_dir} does not exist")

        # Load confs
        with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
            configs = json.load(f)
        # Can't set int as keys in json, so need to cast it after reloading
        # dict_classes keys are always ints
        if 'dict_classes' in configs.keys():
            configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}
        elif 'list_classes' in configs.keys():
            configs['dict_classes'] = {i: col for i, col in enumerate(configs['list_classes'])}

        # Set class vars
        # self.model_name = # Keep the created name
        # self.model_dir = # Keep the created folder
        self.nb_fit = configs.get('nb_fit', 1)  # Consider one unique fit by default
        self.trained = configs.get('trained', True)  # Consider trained by default
        # Try to read the following attributes from configs and, if absent, keep the current one
        for attribute in ['x_col', 'y_col',
                          'list_classes', 'dict_classes', 'multi_label', 'level_save',
                          'batch_size', 'epochs', 'validation_split', 'patience',
                          'embedding_name', 'max_sequence_length', 'tokenizer_special_tokens',
                          'padding', 'truncation', 'pytorch_params']:
            setattr(self, attribute, configs.get(attribute, getattr(self, attribute)))

        # Reload model
        self.model = self.reload_model(torch_dir)
        self.freeze()  # Do not forget to freeze

        # Reload tokenizer
        self.tokenizer = self.reload_tokenizer(torch_dir)

        # No need to save (done in 0_reload_model.py)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, with_shuffle: bool = True, **kwargs) -> None:
        '''Trains the model

        Args:
            x_train (?): Array-like, shape = [n_samples, n_features]
            y_train (?): Array-like, shape = [n_samples, n_features]
            x_valid (?): Array-like, shape = [n_samples, n_features]
            y_valid (?): Array-like, shape = [n_samples, n_features]
        Kwargs:
            with_shuffle (bool): If x, y must be shuffled before fitting
                This should be used if y is not shuffled as the split_validation takes the lines in order.
                Thus, the validation set might get classes which are not in the train set ...
        Raises:
            AssertionError: If different classes when comparing an already fitted model and a new dataset
        '''
        raise NotImplementedError("A light model can't be trained anymore")

    # model_class.list_classes = list(model.config.id2label.values())
    # model_class.dict_classes = model.config.id2label


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
