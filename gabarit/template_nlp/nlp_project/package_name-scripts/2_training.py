#!/usr/bin/env python3

## Training a model - Classification task
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
# Ex: python 2_training.py  --filename dataset_train_preprocess_P1.csv --filename_valid dataset_valid_preprocess_P1.csv --x_col preprocessed_text --y_col Survived


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
from typing import Union, List, Type, Tuple

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.monitoring.model_logger import ModelLogger
from {{package_name}}.models_training.model_class import ModelClass
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
                                              model_pytorch_transformers,
                                              utils_models)

# Disable some warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get logger
logger = logging.getLogger('{{package_name}}.2_training')


def main(filename: str, x_col: Union[str, int], y_col: List[Union[str, int]], filename_valid: Union[str, None] = None,
         min_rows: Union[int, None] = None, nb_iter_keras: int = 1, level_save: str = 'HIGH',
         sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}',
         model: Union[Type[ModelClass], None] = None) -> None:
    '''Trains a model

    Args:
        filename (str): Name of the training dataset (actually a path relative to {{package_name}}-data)
        x_col (str | int): Name of the model's input column - x
        y_col (list<str|int>): Name of the model's target column(s) - y
    Kwargs:
        min_rows (int): Minimal number of occurrences for a class to be considered by the model
            Corresponding entries are removed from both training & validation dataset - mono-label only
        filename_valid (str): Name of the validation dataset (actually a path relative to {{package_name}}-data)
            If None, we do not use a validation dataset.
                -> for keras models (i.e. Neural Networks), we'll use a portion of the training dataset as the validation.
        nb_iter_keras (int): For Keras models, number of models to train to improve stability (default 1).
        level_save (str): Save level
            LOW: statistics + configurations + logger keras - /!\\ the model won't be reusable /!\\ -
            MEDIUM: LOW + hdf5 + pkl + plots
            HIGH: MEDIUM + predictions
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
        model (ModelClass): A model to be fitted. This should only be used for testing purposes.
    Raises:
        ValueError: If level_save value is not a valid option (['LOW', 'MEDIUM', 'HIGH'])
        ValueError: If multi-labels and bad OHE format
    '''
    logger.info("Training a model ...")

    if level_save not in ['LOW', 'MEDIUM', 'HIGH']:
        raise ValueError(f"level_save value ({level_save}) is not a valid option (['LOW', 'MEDIUM', 'HIGH']).")


    ##############################################
    # Manage training dataset
    ##############################################

    # Get dataset
    df_train, preprocess_str = load_dataset(filename, sep=sep, encoding=encoding)

    ### INFO
    ### EACH MODEL needs the same target data format:
    ###
    ###   - OHE (integers) if multi-label
    ###     e.g.
    ###         col1 col2 col3
    ###            1    0   1
    ###            0    0   0
    ###            0    1   0
    ###
    ###   - a single string column if mono-label (even if 0/1 -> '0'/'1')
    ###     e.g.
    ###         target
    ###           target_1
    ###           target_2
    ###           target_1
    ###           target_3
    ###
    ### Below are some examples to preprocess your datasets with the correct format:
    ###
    ###
    ### - multi-label -> Transform tuples in OHE format
    ### ********
    ### y_col = .... # Column with target in tuple format
    ### # Make sure column is tuples and not strings (optional)
    ### from ast import literal_eval
    ### df_train[y_col] = df_train[y_col].apply(lambda x: literal_eval(x))
    ### # Transform to OHE
    ### df_train, y_col = utils_models.preprocess_model_multilabel(df_train, y_col, classes=None)
    ### ********

    # Check if multi-label, i.e. several target columns
    if len(y_col) > 1:
        multi_label = True
        try:
            df_train[y_col] = df_train[y_col].astype(int)  # Need to cast OHE var into integers
            for col in y_col:
                assert sorted(df_train[col].unique()) == [0, 1]
        except Exception:
            raise ValueError("You provided several target columns, but at least one of them does not seem to be in a correct OHE format.")
    else:
        multi_label = False
        y_col = y_col[0]
        # No need to cast target in string, already done by the data loader


    ##############################################
    # Manage validation dataset
    ##############################################

    # Get valid dataset (/!\ we consider that this dataset has the same preprocessing as the training set /!\)
    if filename_valid is not None:
        logger.info(f"Using file {filename_valid} as our validation set.")
        df_valid, preprocess_str_valid = load_dataset(filename_valid, sep=sep, encoding=encoding)
        if preprocess_str_valid != preprocess_str:
            logger.warning("Validation set and training set does not expose the same preprocessing metadata.")
            logger.warning(f"Train : {preprocess_str}")
            logger.warning(f"Valid : {preprocess_str_valid}")
            logger.warning("That will probably lead to bad results !")
            logger.warning("Still continuing...")
        ### INFO: the validation set must have a correct format (cf. traing set info above)
        # Manage OHE format (ensure int)
        if multi_label:
            df_valid[y_col] = df_valid[y_col].astype(int)
        # No need to cast in str if not multilabel, already done by the data loader
    else:
        logger.info("No validation set provided.")
        logger.info("In case of Keras models, we'll use a portion of the training dataset as the validation")


    ##############################################
    # Manage classes
    ##############################################

    # Remove small classes if wanted (only possible if not multi-labels)
    if min_rows is not None and not multi_label:
        df_train = utils_models.remove_small_classes(df_train, y_col, min_rows=min_rows)

    # Remove classes in valid, but not in train (only possible if not multi-labels)
    # If we do not do that, some model (e.g. Keras models) might not work as intended
    # To get metrics on whole dataset, one can use 4_predict.py
    if filename_valid is not None and not multi_label:
        train_classes = list(df_train[y_col].unique())
        valid_classes = list(df_valid[y_col].unique())
        for cl in valid_classes:
            if cl not in train_classes:
                logger.warning(f"Removing class {cl} from the valid dataset (not present in the training dataset).")
        df_valid = df_valid[df_valid[y_col].isin(train_classes)].reset_index(drop=True)


    ##############################################
    # Manage input data
    ##############################################

    # We do not check for columns presence in dataframes
    # TODO: add checks ?
    # Get x, y for both train and valid
    x_train, y_train = df_train[x_col], df_train[y_col]
    if filename_valid is not None:
        x_valid, y_valid = df_valid[x_col], df_valid[y_col]
    else:
        x_valid, y_valid = None, None


    ##############################################
    # Model selection
    ##############################################

    # INFO
    # If you want to continue training of a model, it needs to be reloaded here (only some models are compatible)
    # model, _ = utils_models.load_model("dir_model")
    # Then, it is possible to change some parameters such as the learning rate
    # Be careful to work with the same preprocessing as the first training

    # INFO
    # To use hyperparameters tuning, we can do something like this (experimental, needs improvements):
    # model_cls = model_tfidf_svm.ModelTfidfSvm  # Model Class to be tuned
    # model_params = {'x_col': x_col, 'y_col': y_col, 'multi_label': multi_label}  # Model's fixed parameters
    # hp_params = {'tfidf_params': [{'analyzer': 'word', 'ngram_range': (1, 2), "max_df":0.1}, {'analyzer': 'word', 'ngram_range': (1, 3), "max_df":0.1}]}  # Parameters to be tested
    # scoring_fn = "f1"  # Scoring function to MAXIMIZE
    # kwargs_fit = {'x_train': x_train, 'y_train': y_train, 'with_shuffle': True}  # Fit arguments (i.e. training data)
    # n_splits = 5  # Number of crossvalidation
    # model = utils_models.search_hp_cv(model_cls, model_params, hp_params, scoring_fn, kwargs_fit, n_splits=n_splits)  # Returns a model with the "best" parameters, to be fitted on the whole dataset


    if model is None:
        model = model_tfidf_svm.ModelTfidfSvm(x_col=x_col, y_col=y_col, level_save=level_save,
                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                              svc_params={'C': 1.0, 'max_iter': 10000},
                                              multi_label=multi_label)
        # model = model_tfidf_gbt.ModelTfidfGbt(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
        #                                       gbt_params={'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 10, 'subsample': 1.0, 'max_features': 'auto'},
        #                                       multi_label=multi_label)
        # model = model_tfidf_lgbm.ModelTfidfLgbm(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
        #                                         lgbm_params={'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 5, 'subsample': 1.0, 'num_leaves': 127},
        #                                         multi_label=multi_label)
        # model = model_tfidf_sgdc.ModelTfidfSgdc(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
        #                                         sgdc_params={'loss': 'hinge', 'max_iter': 1000},
        #                                         multi_label=multi_label)
        # model = model_embedding_lstm.ModelEmbeddingLstm(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                 batch_size=64, epochs=99, patience=5,
        #                                                 max_sequence_length=200, max_words=100000,
        #                                                 multi_label=multi_label, nb_iter_keras=nb_iter_keras)
        # model = model_embedding_lstm_attention.ModelEmbeddingLstmAttention(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                                    batch_size=64, epochs=99, patience=5,
        #                                                                    max_sequence_length=200, max_words=100000,
        #                                                                    multi_label=multi_label, nb_iter_keras=nb_iter_keras)
        # model = model_embedding_lstm_structured_attention.ModelEmbeddingLstmStructuredAttention(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                                                         batch_size=64, epochs=99, patience=5,
        #                                                                                         max_sequence_length=200, max_words=100000,
        #                                                                                         multi_label=multi_label, nb_iter_keras=nb_iter_keras)
        # model = model_embedding_lstm_gru_gpu.ModelEmbeddingLstmGruGpu(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                               batch_size=64, epochs=99, patience=5,
        #                                                               max_sequence_length=60, max_words=100000,
        #                                                               multi_label=multi_label, nb_iter_keras=nb_iter_keras)
        # model = model_embedding_cnn.ModelEmbeddingCnn(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                               batch_size=64, epochs=99, patience=5,
        #                                               max_sequence_length=200, max_words=100000,
        #                                               multi_label=multi_label, nb_iter_keras=nb_iter_keras)
        # model = model_tfidf_dense.ModelTfidfDense(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                           batch_size=64, epochs=99, patience=5,
        #                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
        #                                           multi_label=multi_label, nb_iter_keras=nb_iter_keras)
        # model = model_pytorch_transformers.ModelPyTorchTransformers(x_col=x_col, y_col=y_col, level_save=level_save,
        #                                                             batch_size=64, epochs=10, patience=5,
        #                                                             transformer_name='flaubert/flaubert_base_cased',
        #                                                             max_sequence_length=200,
        #                                                             tokenizer_special_tokens=tuple(),
        #                                                             padding="max_length", truncation=True,
        #                                                             multi_label=multi_label)

    # Display if GPU is being used
    model.display_if_gpu_activated()


    ##############################################
    # Train the model !
    ##############################################

    start_time = time.time()
    logger.info("Starting training the model ...")
    model.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid, with_shuffle=True)
    fit_time = time.time() - start_time


    ##############################################
    # Save trained model
    ##############################################

    # Save model
    model.save(
        json_data={
            'filename': filename,
            'filename_valid': filename_valid,
            'min_rows': min_rows,
            'preprocess_str': preprocess_str,
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

    # Series to add
    cols_to_add: List[pd.Series] = []  # You can add columns to save here
    series_to_add_train = [df_train[col] for col in cols_to_add]
    series_to_add_valid = [df_valid[col] for col in cols_to_add]
    gc.collect()  # In some cases, helps with OOMs

    # Get results
    y_pred_train = model.predict(x_train, return_proba=False)
    # model_logger.set_tag(key='type_metric', value='train')
    model.get_and_save_metrics(y_train, y_pred_train, x=x_train, series_to_add=series_to_add_train, type_data='train', model_logger=model_logger)
    gc.collect()  # In some cases, helps with OOMs
    # Get predictions on valid if exists
    if x_valid is not None:
        y_pred_valid = model.predict(x_valid, return_proba=False)
        # model_logger.set_tag(key='type_metric', value='valid')
        model.get_and_save_metrics(y_valid, y_pred_valid, x=x_valid, series_to_add=series_to_add_valid, type_data='valid', model_logger=model_logger)
        gc.collect()  # In some cases, helps with OOMs

    # Stop MLflow if started
    if model_logger is not None:
        model_logger.stop_run()


def load_dataset(filename: str, sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}') -> Tuple[pd.DataFrame, str]:
    '''Function to load a dataset - retrieves preprocessing metadata

    Args:
        filename (str): Name of the dataset to load (actually a path relative to {{package_name}}-data)
    Kwargs:
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    Raises:
        FileNotFoundError: If the file does not exist in {{package_name}}-data
    Returns:
        pd.DataFrame: Loaded dataframe
        str: Preprocessing used on this dataset (from metadata)
    '''
    logger.info(f"Loading a dataset ({filename})")

    # Get dataset
    data_path = utils.get_data_path()
    file_path = os.path.join(data_path, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")

    # Load dataset
    df, first_line = utils.read_csv(file_path, sep=sep, encoding=encoding, dtype=str)
    df = df.fillna('')  # Compulsory as some sentences might be empty after preprocessing, and reloaded as NAs

    # Get preprocess type
    if first_line is not None and first_line.startswith('#'):
        preprocess_str = first_line[1:]  # remove # (sharp)
    else:
        preprocess_str = 'no_preprocess'

    # Return
    return df, preprocess_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='dataset_preprocess_P1.csv', help="Name of the training dataset (actually a path relative to {{package_name}}-data)")
    parser.add_argument('-x', '--x_col', default='preprocessed_text', help="Name of the model's input column - x")
    parser.add_argument('-y', '--y_col', nargs='+', required=True, help="Name of the model's target column(s) - y")
    parser.add_argument('-m', '--min_rows', type=int, default=None, help="Minimal number of occurrences for a class to be considered by the model")
    parser.add_argument('--filename_valid', default=None, help="Name of the validation dataset (actually a path relative to {{package_name}}-data)")
    parser.add_argument('-i', '--nb_iter_keras', type=int, default=1, help="For Keras models, number of models to train to increase stability")
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
    main(filename=args.filename, x_col=args.x_col, y_col=args.y_col,
         min_rows=args.min_rows, filename_valid=args.filename_valid,
         nb_iter_keras=args.nb_iter_keras, level_save=args.level_save,
         sep=args.sep, encoding=args.encoding)
