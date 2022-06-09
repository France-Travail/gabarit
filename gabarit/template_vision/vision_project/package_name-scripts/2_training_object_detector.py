#!/usr/bin/env python3

## Training a model - Object detection task
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
# Ex: python 2_training_object_detector.py --directory dataset_train_preprocess_P1 --directory_valid dataset_valid_preprocess_P1


import os
# Disable some tensorflow logs right away
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import re
import time
import logging
import argparse
import warnings
import pandas as pd
from typing import Type

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training import utils_models
from {{package_name}}.monitoring.model_logger import ModelLogger
from {{package_name}}.models_training.model_class import ModelClass
from {{package_name}}.models_training.object_detectors import (model_keras_faster_rcnn,
                                                               model_detectron_faster_rcnn)

# Disable some warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get logger
logger = logging.getLogger('{{package_name}}.2_training_object_detector')


def main(directory: str, directory_valid: str = None, level_save: str = 'HIGH',
         sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}',
         model: Type[ModelClass] = None) -> None:
    '''Trains a model

    Args:
        directory (str): Directory with the training data (actually a path relative to {{package_name}}-data)
    Kwargs:
        directory_valid (str): Directory with the validation dataset (actually a path relative to {{package_name}}-data)
        level_save (str): Save level
            LOW: statistics + configurations + logger keras - /!\\ the model won't be reusable /!\\ -
            MEDIUM: LOW + hdf5 + pkl + plots
            HIGH: MEDIUM + predictions
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
        model (ModelClass): A model to be fitted. This should only be used for testing purposes.
    Raises:
        ValueError: If level_save value is not a valid option (['LOW', 'MEDIUM', 'HIGH'])
        FileNotFoundError: If the directory does not exists in {{package_name}}-data
        NotADirectoryError: If the argument `directory` is not a directory
        FileNotFoundError: If the validation directory does not exists {{package_name}}-data
        NotADirectoryError: If the argument `directory_valid` is not a directory
    '''
    logger.info("Training a model ...")

    if level_save not in ['LOW', 'MEDIUM', 'HIGH']:
        raise ValueError(f"The object level_save ({level_save}) is not a valid option (['LOW', 'MEDIUM', 'HIGH'])")


    ##############################################
    # Manage training dataset
    ##############################################

    data_path = utils.get_data_path()
    directory_path = os.path.join(data_path, directory)
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The path {directory_path} does not exist")
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"The path {directory_path} is not a directory")

    # Retrieve path/bboxes informations & used preprocessing (if any)
    path_list, bboxes_list, preprocess_str = utils.read_folder_object_detection(directory_path, sep=sep, encoding=encoding)


    ##############################################
    # Manage validation dataset
    ##############################################

    if directory_valid is not None:
        directory_valid_path = os.path.join(data_path, directory_valid)
        if not os.path.exists(directory_valid_path):
            raise FileNotFoundError(f"The path {directory_valid_path} does not exist")
        if not os.path.isdir(directory_valid_path):
            raise NotADirectoryError(f"The path {directory_valid_path} is not a directory")
        # Retrieve path/class informations & used preprocessing (if any)
        path_list_valid, bboxes_list_valid, preprocess_str_valid = utils.read_folder_object_detection(directory_valid_path, sep=sep, encoding=encoding)

        if preprocess_str_valid != preprocess_str:
            logger.warning("Validation set and training set does not expose the same preprocessing metadata.")
            logger.warning(f"Train : {preprocess_str}")
            logger.warning(f"Valid : {preprocess_str_valid}")
            logger.warning("That will probably lead to bad results !")
            logger.warning("Still continuing...")
    else:
        logger.info("No validation set provided.")
        logger.info("In case of Keras models, we'll use a portion of the training dataset as the validation")


    ##############################################
    # Manage input data
    ##############################################

    df_train = pd.DataFrame({'file_path': path_list, 'bboxes': bboxes_list})
    if directory_valid is not None:
        df_valid = pd.DataFrame({'file_path': path_list_valid, 'bboxes': bboxes_list_valid})
    else:
        df_valid = None


    ##############################################
    # Model selection
    ##############################################

    # INFO
    # If you want to continue training of a model, it needs to be reloaded here (only some models are compatible)
    # model, _ = utils_models.load_model("dir_model")
    # Then, it is possible to change some parameters such as the learning rate
    # Be careful to work with the same preprocessing as the first training

    if model is None:
        model = model_detectron_faster_rcnn.ModelDetectronFasterRcnnObjectDetector(epochs=99, batch_size=1, validation_split=0.2, lr=0.00025,
                                                                                   min_delta_es=0., patience=5, restore_best_weights=True,
                                                                                   data_augmentation_params={'horizontal_flip': True, 'vertical_flip': True, 'rot_90': True},
                                                                                   rpn_min_overlap=0.3, rpn_max_overlap=0.7, rpn_restrict_num_regions=128,
                                                                                   roi_nms_overlap_threshold=0.7, pred_bbox_proba_threshold=0.5,
                                                                                   pred_nms_overlap_threshold=0.5, nb_log_write_per_epoch=1,
                                                                                   nb_log_display_per_epoch=10, level_save=level_save)
        # model = model_keras_faster_rcnn.ModelKerasFasterRcnnObjectDetector(validation_split=0.2, color_mode='rgb', pool_resize_classifier=7, nb_rois_classifier=4,
        #                                                                    nms_max_boxes=300, pred_bbox_proba_threshold=0.6, pred_nms_overlap_threshold=0.2,
        #                                                                    data_augmentation_params={'horizontal_flip': True, 'vertical_flip': True, 'rot_90': True},
        #                                                                    batch_size_rpn_trainable_true=5, batch_size_classifier_trainable_true=5,
        #                                                                    batch_size_rpn_trainable_false=5, batch_size_classifier_trainable_false=5,
        #                                                                    epochs_rpn_trainable_true=99, epochs_classifier_trainable_true=99,
        #                                                                    epochs_rpn_trainable_false=99, epochs_classifier_trainable_false=99,
        #                                                                    patience_rpn_trainable_true=5, patience_classifier_trainable_true=5,
        #                                                                    patience_rpn_trainable_false=5, patience_classifier_trainable_false=5,
        #                                                                    lr_rpn_trainable_true=1e-5, lr_classifier_trainable_true=1e-5,
        #                                                                    lr_rpn_trainable_false=1e-5, lr_classifier_trainable_false=1e-5,
        #                                                                    level_save=level_save)


    # Display if GPU is being used
    model.display_if_gpu_activated()


    ##############################################
    # Train the model !
    ##############################################

    start_time = time.time()
    logger.info("Starting training the model ...")
    model.fit(df_train, df_valid=df_valid, with_shuffle=True)
    fit_time = time.time() - start_time


    ##############################################
    # Save trained model
    ##############################################

    # Save model
    model.save(
        json_data={
            'directory': directory,
            'directory_valid': directory_valid,
            'preprocess_str': preprocess_str,  # A ne surtout pas enlever
            'fit_time': f"{round(fit_time, 2)}s",
        }
    )
    logger.info(f"Model {model.model_name} saved in directory {model.model_dir}")


    ##############################################
    # Model metrics
    ##############################################
    {% if mlflow_tracking_uri is not none %}
    # Logging metrics on MLflow
    model_logger = ModelLogger(
        tracking_uri="{{mlflow_tracking_uri}}",
        experiment_name=f"{{package_name}}",
    )
    model_logger.set_tag('model_name', f"{os.path.basename(model.model_dir)}")
    # To log more tags/params, you can use model_logger.set_tag(key, value) or model_logger.log_param(key, value){% else %}
    # No URI has been defined for MLflow
    # Uncomment the following code if you want to use MLflow with a valid URI
    # # Logging metrics on MLflow
    # model_logger = ModelLogger(
    #     tracking_uri=TO BE DEFINED,
    #     experiment_name=f"{{package_name}}",
    # )
    # model_logger.set_tag('model_name', f"{os.path.basename(model.model_dir)}")
    # # To log more tags/params, you can use model_logger.set_tag(key, value) or model_logger.log_param(key, value)
    model_logger=None{% endif %}

    # Get results
    y_pred_train = model.predict(df_train)
    # model_logger.set_tag(key='type_metric', value='train')
    model.get_and_save_metrics(bboxes_list, y_pred_train, list_files_x=path_list, type_data='train', model_logger=model_logger)
    gc.collect()  # In some cases, helps with OOMs
    # Get predictions on valid
    if df_valid is not None:
        y_pred_valid = model.predict(df_valid)
        # model_logger.set_tag(key='type_metric', value='valid')
        model.get_and_save_metrics(bboxes_list_valid, y_pred_valid, list_files_x=path_list_valid, type_data='valid', model_logger=model_logger)
        gc.collect()  # In some cases, helps with OOMs

    # Stop MLflow if started
    if model_logger is not None:
        model_logger.stop_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default=None, required=True, help="Directory with the training data (actually a path relative to {{package_name}}-data)")
    parser.add_argument('--directory_valid', default=None, help="Directory with the validation dataset (actually a path relative to {{package_name}}-data)")
    parser.add_argument('-l', '--level_save', default='HIGH', help="Save level -> ['LOW', 'MEDIUM', 'HIGH']")
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
    main(directory=args.directory, directory_valid=args.directory_valid,
         level_save=args.level_save, sep=args.sep, encoding=args.encoding)
