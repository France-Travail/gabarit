#!/usr/bin/env python3

## Reload a model
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
# Ex: python 0_reload_model.py -m best_model


import os
import json
import ntpath
import logging
import argparse
import pandas as pd
from typing import Union

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training import (model_tfidf_dense,
                                              model_tfidf_gbt,
                                              model_tfidf_lgbm,
                                              model_tfidf_sgdc,
                                              model_tfidf_svm,
                                              model_embedding_cnn,
                                              model_embedding_lstm,
                                              model_embedding_lstm_attention,
                                              model_embedding_lstm_structured_attention,
                                              model_embedding_lstm_gru_gpu,
                                              model_pytorch_light,
                                              model_pytorch_transformers,
                                              utils_models,
                                              utils_deep_keras,
                                              utils_deep_torch)

# Get logger
logger = logging.getLogger('{{package_name}}.0_reload_model')


def main(model_dir: str, config_file: str = 'configurations.json',
         sklearn_pipeline_file: str = 'sklearn_pipeline_standalone.pkl',
         weights_file: str = 'best.hdf5', tokenizer_file: str = 'embedding_tokenizer.pkl',
         tfidf_file: str = 'tfidf_standalone.pkl', checkpoint_path: str = 'best_model.ckpt',
         torch_dir: Union[str, None] = None) -> None:
    '''Reloads a model

    The idea here is to reload a model that was trained on another package version.
    This is done be reusing 'standalone' files.

    Args:
        model_dir (str): Name of the model to reload (not a path, just the directory name)
    Kwargs:
        config_file (str): Name of the configuration file
        sklearn_pipeline_file (str): Standalone sklearn pipeline file name (pipeline models)
        weights_file (str): Neural Network weights file name (keras models)
        tokenizer_file (str): Tokenizer file name (models with embeddings)
        tfidf_file (str): TFIDF file name (models with tfidf)
        checkpoint_path (str): Pytorch lightning checkpoint name (models pytorch)
        torch_dir (str): Pytorch lightning directory name (models pytorch)
    Raises:
        FileNotFoundError: If model can't be found
        FileNotFoundError: If model's configuration does not exist
        ValueError: If the model's type is invalid
    '''
    logger.info(f"Reloading a model ...")

    ##############################################
    # Loading configuration
    ##############################################

    # Get model's path
    models_dir = utils.get_models_path()
    model_path = None
    for path, subdirs, files in os.walk(models_dir):
        for name in subdirs:
            if name == model_dir:
                model_path = os.path.join(path, name)
    if model_path is None:
        raise FileNotFoundError(f"Can't find model {model_dir}")

    # Load conf
    conf_path = os.path.join(model_path, config_file)
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"The file {conf_path} does not exist")
    with open(conf_path, 'r', encoding='{{default_encoding}}') as f:
        configs = json.load(f)


    ##############################################
    # Retrieve model type
    ##############################################

    # Get model type
    model_type_dicts = {
        'model_tfidf_dense': model_tfidf_dense.ModelTfidfDense,
        'model_tfidf_gbt': model_tfidf_gbt.ModelTfidfGbt,
        'model_tfidf_lgbm': model_tfidf_lgbm.ModelTfidfLgbm,
        'model_tfidf_sgdc': model_tfidf_sgdc.ModelTfidfSgdc,
        'model_tfidf_svm': model_tfidf_svm.ModelTfidfSvm,
        'model_embedding_cnn': model_embedding_cnn.ModelEmbeddingCnn,
        'model_embedding_lstm': model_embedding_lstm.ModelEmbeddingLstm,
        'model_embedding_lstm_attention': model_embedding_lstm_attention.ModelEmbeddingLstmAttention,
        'model_embedding_lstm_structured_attention': model_embedding_lstm_structured_attention.ModelEmbeddingLstmStructuredAttention,
        'model_embedding_lstm_gru_gpu': model_embedding_lstm_gru_gpu.ModelEmbeddingLstmGruGpu,
        'model_pytorch_light': model_pytorch_light.ModelPyTorchTransformersLight,
        'model_pytorch_transformers': model_pytorch_transformers.ModelPyTorchTransformers,
    }
    model_type = configs['model_name']
    if model_type not in model_type_dicts:
        raise ValueError(f"The model's type {model_type} is invalid.")
    else:
        model_class = model_type_dicts[model_type]


    ##############################################
    # Reload model
    ##############################################

    # Reload model
    model = model_class()
    files_dict = {
        'configuration_path': os.path.join(model_path, config_file) if config_file is not None else None,
        'sklearn_pipeline_path': os.path.join(model_path, sklearn_pipeline_file) if sklearn_pipeline_file is not None else None,
        'hdf5_path': os.path.join(model_path, weights_file) if weights_file is not None else None,
        'tokenizer_path': os.path.join(model_path, tokenizer_file) if tokenizer_file is not None else None,
        'tfidf_path': os.path.join(model_path, tfidf_file) if tfidf_file is not None else None,
        'checkpoint_path': os.path.join(model_path, checkpoint_path) if checkpoint_path is not None else None,
        'torch_dir': os.path.join(model_path, torch_dir) if torch_dir is not None else None,
    }
    model.reload_from_standalone(**files_dict)


    ##############################################
    # Manage some parameters and save
    ##############################################

    # We check if the model's preprocessing is defined in the preprocess file
    if 'preprocess_str' in configs.keys():
        if configs['preprocess_str'] not in preprocess.get_preprocessors_dict().keys():
            logging.warning(f"The reloaded model's preprocessing is not defined.")
            logging.warning(f"Please add it in the preprocess.py file.")
            logging.warning(f"The model won't be able to make any prediction otherwise.")

    # Save model
    # Reminder: the model's save function prioritize the json_data arg over it's default values
    # hence, it helps with some parameters such as `_get_model`
    list_keys_json_data = ['filename', 'filename_valid', 'min_rows', 'preprocess_str',
                           'fit_time', 'date', '_get_model', '_get_learning_rate_scheduler', 'custom_objects']
    json_data = {key: configs.get(key, None) for key in list_keys_json_data}

    # Add training version
    if 'package_version' in configs:
        # If no trained version yet, use package version
        trained_version = configs.get('trained_version', configs['package_version'])
        if trained_version != utils.get_package_version():
            json_data['trained_version'] = trained_version

    # Save
    json_data = {k: v for k, v in json_data.items() if v is not None}  # Only consider not None values
    model.save(json_data)

    logger.info(f"Model {model_dir} has been successfully reloaded")
    logger.info(f"New model's repository is {model.model_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model_X should be the model's directory name: e.g. model_tfidf_svm_2019_12_05-12_57_18
    parser.add_argument('-m', '--model_dir', required=True, help="Name of the model to reload (not a path, just the directory name)")
    parser.add_argument('-c', '--config_file', default='configurations.json', help="Name of the configuration file")
    parser.add_argument('--sklearn_pipeline_file', default='sklearn_pipeline_standalone.pkl', help="Standalone sklearn pipeline file name (pipeline models)")
    parser.add_argument('-w', '--weights_file', default='best.hdf5', help="Neural Network weights file name (keras models)")
    parser.add_argument('--tokenizer_file', default='embedding_tokenizer.pkl', help="Tokenizer file name (models with embeddings)")
    parser.add_argument('--tfidf_file', default='tfidf_standalone.pkl', help="TFIDF file name (models with tfidf)")
    parser.add_argument('--checkpoint_path', default='best_model.ckpt', help="Pytorch lightning checkpoint name (models pytorch)")
    parser.add_argument('--torch_dir', default=None, help="Pytorch lightning directory name (models pytorch)")
    args = parser.parse_args()
    main(model_dir=args.model_dir, config_file=args.config_file, weights_file=args.weights_file,
         sklearn_pipeline_file=args.sklearn_pipeline_file, tokenizer_file=args.tokenizer_file,
         tfidf_file=args.tfidf_file, checkpoint_path=args.checkpoint_path, torch_dir=args.torch_dir)
