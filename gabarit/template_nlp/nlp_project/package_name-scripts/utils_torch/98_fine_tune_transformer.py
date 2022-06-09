#!/usr/bin/env python3

## Finetune a pretrained transformer
#  /!\ NEEDS A LOT OF RESSOURCES /!\
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
# Ex: python 98_fine_tune_transformer.py -d dataset_train.csv -v dataset_valid.csv -p flaubert_small_cased -x text -o my_transformer -e 5 -b 64


import os
# Disable some tensorflow logs right away
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
from transformers import CONFIG_NAME, WEIGHTS_NAME

from {{package_name}} import utils
from {{package_name}}.models_training import model_pytorch_language_model

# Disable some warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get logger
logger = logging.getLogger('{{package_name}}.98_fine_tune_transformer')



def main(dataset_train: str, dataset_valid: str, pretrained_model_name: str, x_col: Union[str, int],
         batch_size: int = 4, epochs: int = 1, output_dir: Union[str, None] = None,
         sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> None:
    '''Finetunes a pretrained transformer

    Args:
        dataset_train (str): Name of the training dataset (actually a path relative to {{package_name}}-data)
        dataset_valid (str): Name of the validation dataset (actually a path relative to {{package_name}}-data)
        pretrained_model_name (str): Name of the tranformer model to be finetuned (actually a path relative to {{package_name}}-transformers)
        x_col (str | int): Name of the model's input column - x
    Kwargs:
        batch_size (int): Model's batch size
        epochs (int): Model's number of epochs
        output_dir (str): Name of the output model's directory (actually a path relative to {{package_name}}-data)
            If None, we just suffix the pretrained model name with '_finetuned'
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        FileNotFoundError: If the training file does not exist in {{package_name}}-data
        FileNotFoundError: If the validation file does not exist in {{package_name}}-data
        FileNotFoundError: If the model does not exist in {{package_name}}-tranformers
        FileExistsError: If the output directory already exists
    '''
    logger.info("Finetuning a transformer model ...")

    # Manage output dir name
    if output_dir is None:
        output_dir = f"{pretrained_model_name}_finetuned"

    # Get data paths
    dataset_train = os.path.join(utils.get_data_path(), dataset_train)
    dataset_valid = os.path.join(utils.get_data_path(), dataset_valid)
    pretrained_model_path = os.path.join(utils.get_transformers_path(), pretrained_model_name)
    output_dir = os.path.join(utils.get_transformers_path(), output_dir)

    if not os.path.exists(dataset_train):
        raise FileNotFoundError(f"The file {dataset_train} does not exist")
    if not os.path.exists(dataset_valid):
        raise FileNotFoundError(f"The file {dataset_valid} does not exist")
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f"The path {pretrained_model_path} does not exist")
    if not os.path.exists(output_dir):
        raise FileExistsError(f"Path {output_dir} already exists.")

    # Load data
    df_train = pd.read_csv(dataset_train, sep=sep, encoding=encoding)
    df_valid = pd.read_csv(dataset_valid, sep=sep, encoding=encoding)
    # Legacy: add a fake target column to comply with ModelPytorch
    y_col = ["dummy"]
    df_train[y_col] = pd.Series(np.random.randint(0, 2, df_train.shape[0]))
    df_valid[y_col] = pd.Series(np.random.randint(0, 2, df_valid.shape[0]))
    x_train, y_train = df_train[x_col], df_train[y_col]
    x_valid, y_valid = df_valid[x_col], df_valid[y_col]

    # Create language model
    model_pytorch_language_model.TORCH_DEVICE
    model = model_pytorch_language_model.ModelPyTorchLanguageModel(pretrained_model_name, multi_label=True, batch_size=batch_size, epochs=epochs)

    # Train it
    model.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid, with_shuffle=True)
    # Add a single forward
    model.forward()

    # Save
    os.makedirs(output_dir)  # We already cheked it does not exist
    model_to_save = model.model.model.module if hasattr(model.model.model, "module") else model.model.model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    model.tokenizer.save_vocabulary(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_train', required=True, help="Name of the training dataset (actually a path relative to {{package_name}}-data)")
    parser.add_argument('-v', '--dataset_valid', required=True, help="Name of the validation dataset (actually a path relative to {{package_name}}-data)")
    parser.add_argument('-p', '--pretrained_model_name', required=True, help="Name of the tranformer model to be finetuned (actually a path relative to {{package_name}}-transformers)")
    parser.add_argument('-x', '--x_col', required=True, help="Name of the model's input column - x")
    parser.add_argument('-o', '--output_dir', default=None, help="Model's batch size")
    parser.add_argument('-e', '--epochs', default=1, type=int, help="Model's number of epochs")
    parser.add_argument('-b', '--batch_size', default=4, type=int, help="Name of the output model's directory (actually a path relative to {{package_name}}-data)")
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files")
    parser.add_argument('--encoding', default="{{default_encoding}}", help="Encoding to use with the .csv files")
    args = parser.parse_args()
    main(output_dir=args.output_dir, pretrained_model_name=args.pretrained_model_name,
         dataset_train=args.dataset_train, dataset_valid=args.dataset_valid,
         x_col = args.x_col, batch_size=args.batch_size, epochs=args.epochs,
         sep=args.sep, encoding=args.encoding)
