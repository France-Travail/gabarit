#!/usr/bin/env python3

## Transform a pytorch transformer into a "light" version
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
# Ex: python 99_transformer_to_light.py -m torch_model_dir


import os
import json
import torch
import logging
import argparse
from transformers import CONFIG_NAME, WEIGHTS_NAME

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training import model_pytorch_transformers, model_pytorch_light, utils_models

# Get logger
logger = logging.getLogger('{{package_name}}.99_transformer_to_light')


def main(model_dir: str) -> None:
    '''Transforms a pytorch transformer into a "light" version

    Args:
        model_dir (str): Name of the tranformer model to be transformed (actually a path relative to {{package_name}}-transformers)
    '''
    logger.info(f"Transforming a transformer model into a light version ...")


    ##############################################
    # Relaod model
    ##############################################

    # Load model
    model, model_conf = utils_models.load_model(model_dir)


    ##############################################
    # Create new 'light' model
    ##############################################

    # Init. a new "light" model
    light_conf = {}
    conf_keys = ["multi_label", "max_sequence_length",
                 "tokenizer_special_tokens", "padding", "truncation"]
    for k in conf_keys:
        if k in model_conf.keys():
            light_conf[k] = model_conf[k]
    new_model = model_pytorch_light.ModelPyTorchTransformersLight(**light_conf)

    # Save light files to new models' dir
    output_dir = new_model.model_dir
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model.model.model.state_dict(), output_model_file)
    model.model.model.config.to_json_file(output_config_file)
    model.tokenizer.save_vocabulary(output_dir)

    # Reload
    new_model.trained = model.trained
    new_model.nb_fit = model.nb_fit
    new_model.list_classes = model.list_classes
    new_model.dict_classes = model.dict_classes
    new_model.x_col = model.x_col
    new_model.y_col = model.y_col
    new_model.multi_label = model.multi_label
    new_model.pytorch_params = model.pytorch_params
    new_model.model = new_model.reload_model(output_dir)
    new_model.tokenizer = new_model.reload_tokenizer(output_dir)


    ##############################################
    # Save
    ##############################################

    # Save model
    # Reminder: the model's save function prioritize the json_data arg over it's default values
    # hence, it helps with some parameters such as `_get_model`
    list_keys_json_data = ['filename', 'min_rows', 'preprocess_str', 'fit_time',
                           'date', '_get_model', 'pytorch_model']
    json_data = {key: model_conf.get(key, None) for key in list_keys_json_data}

    # Add training version
    if 'package_version' in model_conf:
        # If no trained version yet, use package version
        trained_version = model_conf.get('trained_version', model_conf['package_version'])
        if trained_version != utils.get_package_version():
            json_data['trained_version'] = trained_version

    # Save
    json_data = {k: v for k, v in json_data.items() if v is not None}  # Only consider not None values
    model.save(json_data)

    logger.info(f"Light version of model {model_dir} has been successfully saved")
    logger.info(f"New model's repository is {model.model_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', default=None, help="Name of the tranformer model to be transformed (actually a path relative to {{package_name}}-transformers)")
    args = parser.parse_args()
    main(model_dir=args.model_dir)
