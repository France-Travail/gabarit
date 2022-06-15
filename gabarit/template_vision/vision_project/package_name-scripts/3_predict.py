#!/usr/bin/env python3

# Apply a Machine Learning algorithm to obtain predictions
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
# Ex: python 4_predict.py --directory dataset_test.csv --model model_cnn_classifier_2021_04_15-10_23_13


import os
import tqdm
import json
import logging
import tempfile
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from typing import List
from pathlib import Path
from datetime import datetime

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training import utils_models

# Get logger
logger = logging.getLogger('{{package_name}}.3_predict')


def main(directory: str, model_dir: str, sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> None:
    '''Gets a model's predictions on a given dataset

    Args:
        directory (str): Name of the test directory (actually a path relative to {{package_name}}-data)
            It must NOT be preprocessed
        model_dir (str): Name of the model to use (not a path, just the directory name)
    Kwargs:
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        FileNotFoundError: If the directory does not exists in {{package_name}}-data
        NotADirectoryError: If the argument `directory` is not a directory
    '''

    ##############################################
    # Loading data
    ##############################################

    # Get data abs path
    data_path = utils.get_data_path()
    directory_path = os.path.join(data_path, directory)
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"{directory_path} path does not exist'")
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"{directory_path} is not a valid directory")

    # Retrieve path/class/bboxes informations
    logger.info("Loading dataset ...")
    path_list, classes_or_bboxes_list, _, task_type = utils.read_folder(directory_path, sep=sep, encoding=encoding, accept_no_metadata=True)


    ##############################################
    # Load model
    ##############################################

    # Load model
    logger.info("Loading model ...")
    model, model_conf = utils_models.load_model(model_dir=model_dir)

    # Retrieve preprocessing
    preprocess_str = model_conf['preprocess_str']


    ##############################################
    # Preprocessing
    ##############################################

    # We'll create a temporary folder to save preprocessed images
    with tempfile.TemporaryDirectory(dir=utils.get_data_path()) as tmp_folder:
        # Preprocessing
        if preprocess_str != 'no_preprocess':
            new_path_list = apply_preprocessing(path_list, preprocess_str, directory_path, tmp_folder)
            df = pd.DataFrame({'file_path': new_path_list, 'original_file_path': path_list})
        else:
            logger.info("No preprocessing to be applied")
            df = pd.DataFrame({'file_path': path_list, 'original_file_path': path_list})


        ##############################################
        # Predictions
        ##############################################

        # Get predictions
        logger.info("Getting predictions ...")
        y_pred = model.predict(df, return_proba=False)
        df['predictions'] = list(model.inverse_transform(np.array(y_pred)))

    # Getting out of the context, all temporary data is deleted


    ##############################################
    # Save results
    ##############################################

    # Add filename & select final columns
    df['filename'] = [os.path.relpath(f, directory_path) for f in path_list]
    df = df[['original_file_path', 'filename', 'predictions']]

    # Save result
    logger.info("Saving results ...")
    save_dir = os.path.join(data_path, 'predictions', directory, datetime.now().strftime("predictions_%Y_%m_%d-%H_%M_%S"))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_file = "predictions.csv"
    file_path = os.path.join(save_dir, save_file)
    df.to_csv(file_path, sep='{{default_sep}}', encoding='{{default_encoding}}', index=None)

    # Also save some info into a configs file
    conf_file = 'configurations.json'
    conf_path = os.path.join(save_dir, conf_file)
    conf = {
        'model_dir': model_dir,
        'preprocess_str': model_conf['preprocess_str'],
        'model_name': model_conf['model_name']
    }
    with open(conf_path, 'w', encoding='{{default_encoding}}') as f:
        json.dump(conf, f, indent=4)


    ##############################################
    # Get metrics
    ##############################################

    # Get metrics if classes_or_bboxes_list is not None
    if classes_or_bboxes_list is not None:
        # Change model directory to save dir & get preds
        model.model_dir = save_dir
        model.get_and_save_metrics(classes_or_bboxes_list, y_pred, list_files_x=list(df['original_file_path'].values), type_data='test')


def apply_preprocessing(path_list: List[str], preprocess_str: str, old_directory: str, tmp_folder: str) -> List[str]:
    '''Function to apply a preprocessing on an image directory

    Args:
        path_list (list<str>): path list of all images to be preprocessed
        preprocess_str (str): preprocessing to be applied
        old_directory (str): directory with the original images - path
        tmp_folder (str): directory where to save the preprocessed images - path
    Returns:
        list<str>: new paths list for the preprocessed images
    '''
    logger.info(f"Applying preprocessing {preprocess_str} on test images ...")

    # Get preprocessor
    preprocessor = preprocess.get_preprocessor(preprocess_str)

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
            tmp_im.load() # Cumpulsory to avoid "Too many open files" issue
            images_tmp.append(tmp_im)
        # Process images
        processed_images_tmp = preprocessor(images_tmp)
        # Get new files list
        new_path_list_tmp = [os.path.relpath(f, old_directory) for f in tmp_path_list]
        new_path_list_tmp = [os.path.join(tmp_folder, f) for f in new_path_list_tmp]
        # Save in PNG format -> lossless compression ! DO NOT SAVE IN JPEG !
        new_path_list_tmp = [f"{os.path.splitext(f)[0]}.png" for f in new_path_list_tmp]
        # Save processed images
        for i in range(len(processed_images_tmp)):
            im = processed_images_tmp[i]
            im_path = new_path_list_tmp[i]
            # Check if folder exists !
            dst_dir_path = os.path.dirname(im_path)
            if not os.path.exists(dst_dir_path):
                os.makedirs(dst_dir_path)
            im.save(im_path, format='PNG')
        # Add new path to the main list
        new_path_list += new_path_list_tmp

    # Return
    return new_path_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default=None, required=True, help="Name of the test directory (actually a path relative to {{package_name}}-data)")
    # model_X should be the model's directory name: e.g. model_tfidf_svm_2019_12_05-12_57_18
    parser.add_argument('-m', '--model_dir', required=True, help="Name of the model to use (not a path, just the directory name)")
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files.")
    parser.add_argument('--encoding', default="{{default_encoding}}", help="Encoding to use with the .csv files.")
    parser.add_argument('--force_cpu', dest='on_cpu', action='store_true', help="Whether to force training on CPU (and not GPU)")
    parser.set_defaults(on_cpu=False)
    args = parser.parse_args()
    # Check forced CPU usage
    if args.on_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        logger.info("----------------------------")
        logger.info("CPU USAGE FORCED BY THE USER")
        logger.info("----------------------------")
    # Main
    main(directory=args.directory, model_dir=args.model_dir, sep=args.sep, encoding=args.encoding)
