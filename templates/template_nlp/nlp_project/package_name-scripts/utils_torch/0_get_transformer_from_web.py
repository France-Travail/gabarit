#!/usr/bin/env python3

## Retrieve a transformer from the web
## Saves it in a transformers folder
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
# Ex: python 0_get_transformer_from_web.py --transformer_name flaubert/flaubert_base_cased


import os
import logging
import argparse
from transformers import AutoModel, AutoTokenizer

from {{package_name}} import utils

# Get logger
logger = logging.getLogger('{{package_name}}.0_get_transformer_from_web')


def main(transformer_name: str) -> None:
    '''Retrieves a transformer from the web and save it in {{package_name}}-transformers

    Args:
        transformer_name (str): Name of the transformer to be retrieved
    Raises:
        FileExistsError: If the transformer already exists
    '''
    logger.info("Retrieving a transformer ...")

    # Check path does not already exist
    transformer_path = os.path.join(utils.get_transformers_path(), transformer_name)
    if os.path.exists(transformer_path):
        raise FileExistsError(f"Transformer {transformer_name} already exists. To redownload it, this file has to be deleted.")


    ##############################################
    # Retrieve transformer
    ##############################################

    logger.info(f"Retrieving transformer {transformer_name}")
    config_dict = dict(do_lowercase=False) #TODO: to be parameterized
    model = AutoModel.from_pretrained(transformer_name, config=config_dict)
    tokenizer = AutoTokenizer.from_pretrained(transformer_name, config=config_dict)


    ##############################################
    # Save
    ##############################################

    logger.info(f"Saving transformer to {transformer_path}")
    # Save
    model.save_pretrained(transformer_path)
    tokenizer.save_pretrained(transformer_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # e.g. flaubert/flaubert_base_cased
    parser.add_argument('--transformer_name', required=True, help="Name of the transformer to be retrieved.")
    args = parser.parse_args()
    main(transformer_name=args.transformer_name)
