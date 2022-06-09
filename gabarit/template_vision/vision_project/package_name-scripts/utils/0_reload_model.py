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
# Ex: python 0_reload_model.py -m best_model -c configurations.json


import os
import json
import ntpath
import logging
import argparse
import pandas as pd
from typing import Union

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training import utils_models, utils_deep_keras
from {{package_name}}.models_training.classifiers import model_cnn_classifier, model_transfer_learning_classifier
from {{package_name}}.models_training.object_detectors import model_keras_faster_rcnn, model_detectron_faster_rcnn

# Get logger
logger = logging.getLogger('{{package_name}}.0_reload_model')


def main(model_dir: str, config_file: str = 'configurations.json', weights_file: str = 'best.hdf5',
         detectron_file: str = 'best.pth', preprocess_input_file: str = 'preprocess_input.pkl') -> None:
    '''Reloads a model

    The idea here is to reload a model that was trained on another package version.
    This is done be reusing 'standalone' files.

    Args:
        model_dir (str): Name of the model to reload (not a path, just the directory name)
    Kwargs:
        config_file (str): Name of the configuration file
        weights_file (str): Neural Network weights file name (keras models)
        detectron_file (str): Detectron best model file name (detectron models)
        preprocess_input_file (str): Model's internal preprocessing file (keras models)
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
        'model_cnn_classifier': model_cnn_classifier.ModelCnnClassifier,
        'model_transfer_learning_classifier': model_transfer_learning_classifier.ModelTransferLearningClassifier,
        'model_keras_faster_rcnn_object_detector': model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector,
        'model_detectron_faster_rcnn_object_detector': model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector,
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
        'hdf5_path': os.path.join(model_path, weights_file) if weights_file is not None else None,
        'pth_path': os.path.join(model_path, detectron_file) if detectron_file is not None else None,
        'preprocess_input_path': os.path.join(model_path, preprocess_input_file) if preprocess_input_file is not None else None,
    }
    model.reload_from_standalone(**files_dict)


    ##############################################
    # Manage some parameters and save
    ##############################################

    # Save model
    # Reminder: the model's save function prioritize the json_data arg over it's default values
    # hence, it helps with some parameters such as `_get_model`
    list_keys_json_data = ['directory', 'directory_valid', 'preprocess_str', '_add_classifier_layers',
                           'fit_time', 'date', '_get_model', '_add_rpn_layers', '_get_preprocess_input',
                           '_get_learning_rate_scheduler', '_get_second_learning_rate_scheduler', 'custom_objects']
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
    # model_X should be the model's directory name: e.g. model_preprocess_pipeline_svm_2019_12_05-12_57_18
    parser.add_argument('-m', '--model_dir', required=True, help="Name of the model to reload (not a path, just the directory name)")
    parser.add_argument('-c', '--config_file', default='configurations.json', help="Name of the configuration file")
    parser.add_argument('-w', '--weights_file', default='best.hdf5', help="Neural Network weights file name (keras models)")
    parser.add_argument('-d', '--detectron_file', default='best.pth', help="Detectron best model file name (detectron models)")
    parser.add_argument('-p', '--preprocess_input_file', default='preprocess_input.pkl', help="Model's internal preprocessing file (keras models)")
    args = parser.parse_args()
    main(model_dir=args.model_dir, config_file=args.config_file, weights_file=args.weights_file,
         detectron_file=args.detectron_file, preprocess_input_file=args.preprocess_input_file)
