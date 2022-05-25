#!/usr/bin/env python3

## Data preprocessing
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
# Ex: python 1_preprocess_data.py -d dataset_v1_train dataset_v1_valid -p preprocess_docs


import os
import gc
import time
import tqdm
import shutil
import logging
import argparse
import pandas as pd
import dill as pickle
from PIL import Image
from pathlib import Path
from typing import Union, List

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess

# Get logger
logger = logging.getLogger('{{package_name}}.1_preprocess_data')


def main(directories: List[str], preprocessing: Union[str, None], sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> None:
    '''Preprocesses some datasets

    Args:
        directory (list<str>): Datasets directories (actually a path relative to {{package_name}}-data)
        preprocessing (str): Preprocessing to be applied. All preprocessings are applied if None.
    Kwargs:
        sep (str): Separator to use with the metadata file - if exists
        encoding (str): Encoding to use with the metadata file - if exists
    Raises:
        FileNotFoundError: If a given directory does not exist in {{package_name}}-data
        NotADirectoryError: If a given path is not a directory
    '''
    logger.info("Data preprocessing ...")

    ##############################################
    # Manage preprocessing pipelines
    ##############################################

    # Get preprocess dictionnary
    preprocessors_dict = preprocess.get_preprocessors_dict()

    # Get preprocessing(s) to apply
    if preprocessing is not None:
        # Check presence in preprocessors_dict
        if preprocessing not in preprocessors_dict.keys():
            raise ValueError(f"The given preprocessing {preprocessing} is not known.")
        preprocessing_list = [preprocessing]
    # By default, we apply every preprocessings
    else:
        preprocessing_list = list(preprocessors_dict.keys())


    ##############################################
    # Process each directory, one by one
    ##############################################

    # Process each directory, one by one
    for directory in directories:

        ##############################################
        # Manage paths
        ##############################################

        # Get some paths
        data_path = utils.get_data_path()
        directory_path = os.path.join(data_path, directory)
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"{directory_path} does not exist.'")
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"{directory_path} is not a directory.")


        ##############################################
        # Retrieve informations from images
        ##############################################

        # Read folder
        path_list, classes_or_bboxes_list, _, task_type = utils.read_folder(directory_path, sep=sep, encoding=encoding)


        if task_type == 'object_detection':
            logger.warning("! WARNING !")
            logger.warning("When working with object detection tasks, one should not change the image size & orientation.")
            logger.warning("Otherwise, the Bounding Boxes could be in the wrong place.")
            logger.warning("! WARNING !")


        ##############################################
        # Applying preprocessing one by one
        ##############################################

        # Apply each preprocess one by one
        for preprocess_str in preprocessing_list:

            # 'no_preprocess' must be ignored
            if preprocess_str == 'no_preprocess':
                continue
            gc.collect()  # Fix some OOM in case of huge datasets being preprocessed
            logger.info(f'Applying preprocessing {preprocess_str} on directory {directory}')

            # Get preprocessor
            preprocessor = preprocess.get_preprocessor(preprocess_str)

            # We start by creating a new directory for the preprocessed images
            new_directory = os.path.join(data_path, f"{directory}_{preprocess_str}")
            if os.path.exists(new_directory):
                shutil.rmtree(new_directory)
            os.makedirs(new_directory)

            # Process by chunks of 100s to avoid memory issues
            new_path_list = []
            chunks_limits = utils.get_chunk_limits(pd.Series(path_list), chunksize=100)
            for limits in tqdm.tqdm(chunks_limits):
                # Manage data
                min_l, max_l = limits[0], limits[1]
                tmp_path_list = path_list[min_l:max_l]
                # Load images
                images_tmp = []
                for f in tmp_path_list:
                    tmp_im = Image.open(f)
                    tmp_im.load() # Compulsory to avoid "Too many open files" issue
                    images_tmp.append(tmp_im)
                # Process images
                processed_images_tmp = preprocessor(images_tmp)
                # Get new files list
                new_path_list_tmp = [os.path.relpath(f, directory_path) for f in tmp_path_list]
                # Add new dir
                new_path_list_tmp = [os.path.join(new_directory, f) for f in new_path_list_tmp]
                # Save images as PNG -> lossless compression ! Images must NOT be saved in JPEG format
                new_path_list_tmp = [f"{os.path.splitext(f)[0]}.png" for f in new_path_list_tmp]
                # Save processed images
                for i in range(len(processed_images_tmp)):
                    im = processed_images_tmp[i]
                    im_path = new_path_list_tmp[i]
                    # We ensure that the image's parent directory exists (e.g. in case of subfolders per class format)
                    dst_dir_path = os.path.dirname(im_path)
                    if not os.path.exists(dst_dir_path):
                        os.makedirs(dst_dir_path)
                    im.save(im_path, format='PNG')
                # We add the new paths into final list
                new_path_list += new_path_list_tmp

            # Generate the preprocessing file associated with the final directory to match it back to the preprocess pipeline
            preprocess_pipeline_conf_path = os.path.join(new_directory, 'preprocess_pipeline.conf')
            with open(preprocess_pipeline_conf_path, 'w', encoding=encoding) as f:
                f.write(preprocess_str)

            # Generate the metadata file when adapted
            filenames_list = [os.path.relpath(f, new_directory) for f in new_path_list]
            if task_type == 'classification':
                metadata_path = os.path.join(directory_path, 'metadata.csv')
                # Create only if needed
                if os.path.exists(metadata_path):
                    new_metadata_path = os.path.join(new_directory, 'metadata.csv')
                    metadata_df = utils.rebuild_metadata_classification(filenames_list, classes_or_bboxes_list)
                    metadata_df.to_csv(new_metadata_path, sep=sep, encoding=encoding, index=None)
            else:
                # Metadata compulsory for object detection tasks
                new_metadata_path = os.path.join(new_directory, 'metadata_bboxes.csv')
                metadata_df = utils.rebuild_metadata_object_detection(filenames_list, classes_or_bboxes_list)
                metadata_df.to_csv(new_metadata_path, sep=sep, encoding=encoding, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directories', nargs='+', required=True, help='Datasets directories (actually a path relative to {{package_name}}-data)')
    parser.add_argument('-p', '--preprocessing', default=None, help='Preprocessing to be applied. All preprocessings are applied if None.')
    parser.add_argument('--sep', default='{{default_sep}}', help='Separator to use with the metadata file - if exists.')
    parser.add_argument('--encoding', default="{{default_encoding}}", help='Encoding to use with the metadata file - if exists.')
    args = parser.parse_args()
    main(directories=args.directories, preprocessing=args.preprocessing, sep=args.sep, encoding=args.encoding)
