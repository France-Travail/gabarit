#!/usr/bin/env python3

## Utils - Tools for training
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
# Functions :
# - normal_split -> Splits a DataFrame into train and test sets
# - stratified_split -> Splits a DataFrame into train and test sets - Stratified strategy
# - remove_small_classes -> Deletes under-represented classes
# - display_train_test_shape -> Displays the size of a train/test split
# - preprocess_model_multilabel -> Prepares a dataframe for a multi-labels model
# - load_pipeline -> Loads a pipeline from the pipelines folder
# - load_model -> Loads a model from a path
# - get_columns_pipeline -> Retrieves a pipeline wanted columns, and mandatory ones
# - apply_pipeline -> Applies a fitted pipeline to a dataframe
# - predict -> Gets predictions on a dataset
# - predict_with_proba -> Gets predictions with probabilities on a dataset
# - search_hp_cv -> Searches for hyperparameters


import os
import gc
import json
import math
import pprint
import shutil
import logging
import numpy as np
import pandas as pd
import dill as pickle
from datetime import datetime
from typing import Union, Tuple, Callable, Any

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from {{package_name}} import utils
from {{package_name}}.preprocessing import preprocess
# We can't import ModelClass here as this creates a circular dependency
# As this is only used for type hitting, we rather prefer to give up on it here
# from {{package_name}}.models_training.model_class import ModelClass


# Get logger
logger = logging.getLogger(__name__)


def normal_split(df: pd.DataFrame, test_size: float = 0.25, seed: Union[int, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Splits a DataFrame into train and test sets

    Args:
        df (pd.DataFrame): Dataframe containing the data
    Kwargs:
        test_size (float): Proportion representing the size of the expected test set
        seed (int): random seed
    Raises:
        ValueError: If the object test_size is not between 0 and 1
    Returns:
        DataFrame: Train dataframe
        DataFrame: Test dataframe
    '''
    if not 0 <= test_size <= 1:
        raise ValueError('The object test_size must be between 0 and 1')

    # Normal split
    logger.info("Normal split")
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)

    # Display
    display_train_test_shape(df_train, df_test, df_shape=df.shape[0])

    # Return
    return df_train, df_test


def stratified_split(df: pd.DataFrame, col: Union[str, int], test_size: float = 0.25, seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Splits a DataFrame into train and test sets - Stratified strategy

    Args:
        df (pd.DataFrame): Dataframe containing the data
        col (str or int): column on which to do the stratified split
    Kwargs:
        test_size (float): Proportion representing the size of the expected test set
        seed (int): Random seed
    Raises:
        ValueError: If the object test_size is not between 0 and 1
    Returns:
        DataFrame: Train dataframe
        DataFrame: Test dataframe
    '''
    if not 0 <= test_size <= 1:
        raise ValueError('The object test_size must be between 0 and 1')

    # Stratified split
    logger.info("Stratified split")
    df = remove_small_classes(df, col, min_rows=math.ceil(1 / test_size))  # minimum lines number per category to split
    df_train, df_test = train_test_split(df, stratify=df[col], test_size=test_size, random_state=seed)

    # Display
    display_train_test_shape(df_train, df_test, df_shape=df.shape[0])

    # Return
    return df_train, df_test


def remove_small_classes(df: pd.DataFrame, col: Union[str, int], min_rows: int = 2) -> pd.DataFrame:
    '''Deletes the classes with small numbers of elements

    Args:
        df (pd.DataFrame): Dataframe containing the data
        col (str | int): Columns containing the classes
    Kwargs:
        min_rows (int): Minimal number of lines in the training set (default: 2)
    Raises:
        ValueError: If the object min_rows is not positive
    Returns:
        pd.DataFrame: New dataset
    '''
    if min_rows < 1:
        raise ValueError("The object min_rows must be positive")

    # Looking for classes with less than min_rows lines
    v_count = df[col].value_counts()
    classes_to_remove = list(v_count[v_count < min_rows].index.values)
    for cl in classes_to_remove:
        logger.warning(f"/!\\ /!\\ /!\\ Class {cl} has less than {min_rows} lines in the training set.")
        logger.warning("/!\\ /!\\ /!\\ This class is automatically removed from the dataset.")
    return df[~df[col].isin(classes_to_remove)]


def display_train_test_shape(df_train: pd.DataFrame, df_test: pd.DataFrame, df_shape: Union[int, None] = None) -> None:
    '''Displays the size of a train/test split

    Args:
        df_train (pd.DataFrame): Train dataset
        df_test (pd.DataFrame): Test dataset
    Kwargs:
        df_shape (int): Size of the initial dataset
    Raises:
        ValueError: If the object df_shape is not positive
    '''
    if df_shape is not None and df_shape < 1:
        raise ValueError("The object df_shape must be positive")

    # Process
    if df_shape is None:
        df_shape = df_train.shape[0] + df_test.shape[0]
    logger.info(f"There are {df_train.shape[0]} lines in the train dataset and {df_test.shape[0]} in the test dataset.")
    logger.info(f"{round(100 * df_train.shape[0] / df_shape, 2)}% of data are in the train set")
    logger.info(f"{round(100 * df_test.shape[0] / df_shape, 2)}% of data are in the test set")


def preprocess_model_multilabel(df: pd.DataFrame, y_col: Union[str, int], classes: Union[list, None] = None) -> Tuple[pd.DataFrame, list]:
    '''Prepares a dataframe for a multi-labels classification

    Args:
        df (pd.DataFrame): Training dataset
            This dataset must be preprocessed.
            Example:
                # Group by & apply tuple to y_col
                x_cols = [col for col in list(df.columns) if col != y_col]
                df = pd.DataFrame(df.groupby(x_cols)[y_col].apply(tuple))
        y_col (str or int): Name of the column to be used for training - y
    Kwargs:
        classes (list): List of classes to consider
    Returns:
        DataFrame: Dataframe for training
        list: List of 'y' columns
    '''
    # TODO: add possibility to have sparse output
    logger.info("Preprocess dataframe for multi-labels model")
    # Process
    logger.info("Preparing dataset for multi-labels format. Might take several minutes.")
    # /!\ The reset_index is compulsory in order to have the same indexes between df, and MLB transformed values
    df = df.reset_index(drop=True)
    # Apply MLB
    mlb = MultiLabelBinarizer(classes=classes)
    df = df.assign(**pd.DataFrame(mlb.fit_transform(df[y_col]), columns=mlb.classes_))
    # Return dataframe & y_cols (i.e. classes)
    return df, list(mlb.classes_)


def load_pipeline(pipeline_dir: Union[str, None], is_path: bool = False) -> Tuple[Pipeline, str]:
    '''Loads a pipeline from the pipelines folder

    Args:
        pipeline_dir (str): Name of the folder containing the pipeline to get. If None,
            backups on "no_preprocess"
    Kwargs:
        is_path (bool): If path to the folder instead of the name (permits the loading from elsewhere)
    Raises:
        FileNotFoundError: If the folder pipeline_dir does not exist
    Returns:
        Pipeline: Reloaded pipeline
        str: Name of the preprocessing used
    '''
    # If pipeline_dir is None, backups on "no_preprocess"
    if pipeline_dir is None:
        logger.warning("The folder of the pipeline is None. Backups on 'no_preprocess'")
        preprocess_str = "no_preprocess"
        preprocess_pipeline = preprocess.get_pipeline(preprocess_str)  # Warning, must be fitted
        return preprocess_pipeline, preprocess_str

    # Otherwise, nominal case
    # Find pipeline path
    if not is_path:
        pipelines_dir = utils.get_pipelines_path()
        pipeline_path = None
        for path, subdirs, files in os.walk(pipelines_dir):
            for name in subdirs:
                if name == pipeline_dir:
                    pipeline_path = os.path.join(path, name)
        if pipeline_path is None:
            raise FileNotFoundError(f"Can't find pipeline {pipeline_dir}")
    else:
        pipeline_path = pipeline_dir
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Can't find pipeline {pipeline_path} (considered as a path)")

    # Get pipeline
    pipeline_path = os.path.join(pipeline_path, 'pipeline.pkl')
    with open(pipeline_path, 'rb') as f:
        pipeline_dict = pickle.load(f)

    # Return
    return pipeline_dict['preprocess_pipeline'], pipeline_dict['preprocess_str']


def load_model(model_dir: str, is_path: bool = False) -> Tuple[Any, dict]:
    '''Loads a model from a path

    Args:
        model_dir (str): Name of the folder containing the model (e.g. model_autres_2019_11_07-13_43_19)
    Kwargs:
        is_path (bool): If folder path instead of name (permits to load model from elsewhere)
    Raises:
        FileNotFoundError: If the folder model_dir does not exist
    Returns:
        ?: Model
        dict: Model configurations
    '''

    # Find model path
    if not is_path:
        models_dir = utils.get_models_path()
        model_path = None
        for path, subdirs, files in os.walk(models_dir):
            for name in subdirs:
                if name == model_dir:
                    model_path = os.path.join(path, name)
        if model_path is None:
            raise FileNotFoundError(f"Can't find model {model_dir}")
    else:
        model_path = model_dir
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Can't find model {model_path} (considered as a path)")

    # Get configs
    configuration_path = os.path.join(model_path, 'configurations.json')
    with open(configuration_path, 'r', encoding='{{default_encoding}}') as f:
        configs = json.load(f)
    # Can't set int as keys in json, so need to cast it after reloading
    # dict_classes keys are always ints
    if 'dict_classes' in configs.keys() and configs['dict_classes'] is not None:
        configs['dict_classes'] = {int(k): v for k, v in configs['dict_classes'].items()}

    # Load model
    pkl_path = os.path.join(model_path, f"{configs['model_name']}.pkl")
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)

    # Change model_dir if diff
    if model_path != model.model_dir:
        model.model_dir = model_path
        configs['model_dir'] = model_path

    # Load specifics
    hdf5_path = os.path.join(model_path, 'best.hdf5')

    # Check for keras model
    if os.path.exists(hdf5_path):
        model.model = model.reload_model(hdf5_path)

    # Display if GPU is being used
    model.display_if_gpu_activated()

    # Return model & configs
    return model, configs


def get_columns_pipeline(preprocess_pipeline: ColumnTransformer) -> Tuple[list, list]:
    '''Retrieves a pipeline wanted columns, and mandatory ones

    Args:
        preprocess_pipeline (ColumnTransformer): Preprocessing pipeline
    Returns:
        list: List of columns in
        list: List of mandatory ones
    '''
    # Checks if the pipeline is fitted
    check_is_fitted(preprocess_pipeline)
    # Gets the names of input columns
    columns_in = preprocess_pipeline._feature_names_in.tolist()
    # Gets the names of the "mandatory" columns
    if preprocess_pipeline._remainder[1] == 'drop':
        # If drop, we get from _columns
        mandatory_columns = list(utils.flatten(preprocess_pipeline._columns))
    else:
        # Otherwise, we need all the columns
        mandatory_columns = columns_in
    # Returns
    return columns_in, mandatory_columns


def apply_pipeline(df: pd.DataFrame, preprocess_pipeline: ColumnTransformer) -> pd.DataFrame:
    '''Applies a fitted pipeline to a dataframe

    Problem :
        The pipeline expects as input the same columns and in the same order
        even if some columns are then dropped (and so useless)
    Solution (experimental 14/04/2021):
        We add the "useless" columns as columns filled with NaNs

    Args:
        df (pd.DataFrame): Dataframe to preprocess
        preprocess_pipeline (ColumnTransformer): Pipeline to use
    Raises:
        ValueError: If some mandatory columns are missing
    Returns:
        pd.DataFrame: Preprocessed dataFrame
    '''
    columns_in, mandatory_columns = get_columns_pipeline(preprocess_pipeline)

    # Removes the useless columns
    df = df[[col for col in df.columns if col in columns_in]]
    optionals_columns = [col for col in columns_in if col not in mandatory_columns]

    # Checks if all the mandatory columns are present
    missing_mandatory_columns = [col for col in mandatory_columns if col not in df.columns]
    if len(missing_mandatory_columns) > 0:
        for missing_col in missing_mandatory_columns:
            logger.error(f"Missing column in your dataset : {missing_col}")
        raise ValueError(f"There are some missing mandatory columns in order to preprocess the dataset : {missing_mandatory_columns}")

    # We add the non mandatory columns if they are not already in df
    # Note : only relevant in the case remainder = "drop" (nominal case)
    missing_optionals_columns = [col for col in optionals_columns if col not in df.columns]
    for col in missing_optionals_columns:
        logger.warning(f'The column {col} is missing in order to apply the preprocessing.')
        logger.warning('Experimental : it should be useless -> creation of an empty column')
        df[col] = np.nan

    # Apply transform on reordered columns
    preprocessed_x = preprocess_pipeline.transform(df[columns_in])
    # Reconstruct dataframe & return
    preprocessed_df = pd.DataFrame(preprocessed_x)
    preprocessed_df = preprocess.retrieve_columns_from_pipeline(preprocessed_df, preprocess_pipeline)
    return preprocessed_df


def predict(content: pd.DataFrame, model, **kwargs) -> Union[float, str, tuple, list]:
    '''Gets predictions of a model on a dataset

    Args:
        content (pd.DataFrame): New dataset to be predicted
        model (ModelClass): Model to use
    Returns:
        REGRESSION :
            float: prediction
        MONO-LABEL CLASSIFICATION:
            str: prediction
        MULTI-LABELS CLASSIFICATION:
            tuple: predictions

        If several elements -> list
    '''
    # Apply preprocessing
    if model.preprocess_pipeline is not None:
        df_prep = apply_pipeline(content, model.preprocess_pipeline)
    else:
        df_prep = content.copy()
        logger.warning("No preprocessing pipeline found - we consider no preprocessing, but it should not be so !")

    # Get predictions
    predictions = model.predict(df_prep)

    # Inverse transform (needed for classification)
    predictions = model.inverse_transform(predictions)

    # Return only first element if dataframe has one row
    if content.shape[0] == 1:
        predictions = predictions[0]

    # Return
    return predictions


def predict_with_proba(content: pd.DataFrame, model) -> Tuple[Union[str, tuple, list], Union[float, tuple, list]]:
    '''Gets probabilities predictions of a model on a dataset

    Args:
        content (pd.DataFrame): New dataset to be predicted
        model (ModelClass): Model to use
    Raises:
        ValueError: If the model type is not classifier
    Returns:
        MONO-LABEL CLASSIFICATION:
            str: prediction
            float: probability
        MULTI-LABELS CLASSIFICATION:
            tuple: predictions
            tuple: probabilities

        If several elements -> list
    '''
    # Regressions
    if not model.model_type == 'classifier':
        raise ValueError(f"The model type ({model.model_type}) is not supported by the method predict_with_proba")

    # Apply preprocessing
    if model.preprocess_pipeline is not None:
        df_prep = apply_pipeline(content, model.preprocess_pipeline)
    else:
        df_prep = content.copy()
        logger.warning("No preprocessing pipeline found - we consider no preprocessing, but it should not be so !")

    # Get predictions
    predictions, probas = model.predict_with_proba(df_prep)

    # Rework format
    if not model.multi_label:
        prediction = model.inverse_transform(predictions)
        proba = list(probas.max(axis=1))
    else:
        prediction = [tuple(np.array(model.list_classes).compress(indicators)) for indicators in predictions]
        proba = [tuple(np.array(probas[i]).compress(indicators)) for i, indicators in enumerate(predictions)]

    # Return only first element if dataframe has one row
    if content.shape[0] == 1:
        prediction = prediction[0]
        proba = proba[0]

    # Return prediction & proba
    return prediction, proba


def search_hp_cv(model_cls, model_params: dict, hp_params: dict, scoring_fn: Union[str, Callable],
                 kwargs_fit: dict, n_splits: int = 5):
    '''Searches for hyperparameters - works only with classifiers !

    Args:
        model_cls (?): Class of models on which to do a hyperparameters search
        model_params (dict): Set of "fixed" parameters of the model (e.g. x_col, y_col).
            Must contain 'multi_label'.
        hp_params (dict): Set of "variable" parameters on which to do a hyperparameters search
        scoring_fn (str or func): Scoring function to maximize
            This function must take as input a dictionary containing metrics
            e.g. {'F1-Score': 0.85, 'Accuracy': 0.57, 'Precision': 0.64, 'Recall': 0.90}
        kwargs_fit (dict): Set of kwargs to input in the fit function
            Must contain 'x_train' and 'y_train'
    Kwargs:
        n_splits (int): Number of folds to use
    Raises:
        ValueError: If scoring_fn is not a known string
        ValueError: If multi_label is not a key in model_params
        ValueError: If x_train is not a key in kwargs_fit
        ValueError: If y_train is not a key in kwargs_fit
        ValueError: If model_params and hp_params share some keys
        ValueError: If hp_params values are not the same length
        ValueError: If the number of crossvalidation split is less or equal to 1
    Returns:
        ModelClass: best model to be "fitted" on the dataset
    '''
    list_known_scoring = ['accuracy', 'f1', 'precision', 'recall']

    # We can't really check if classifier ...

    #################
    # Manage errors
    #################

    if isinstance(scoring_fn, str) and scoring_fn not in list_known_scoring:
        raise ValueError(f"The input {scoring_fn} is not a known value for scoring_fn (known values : {list_known_scoring})")

    if 'multi_label' not in model_params.keys():
        raise ValueError("The key 'multi_label' must be present in the dictionary model_params")

    if 'x_train' not in kwargs_fit.keys():
        raise ValueError("The key 'x_train' must be present in the dictionary kwargs_fit")

    if 'y_train' not in kwargs_fit.keys():
        raise ValueError("The key 'y_train' must be present in the dictionary kwargs_fit")

    if any([k in hp_params.keys() for k in model_params.keys()]):
        # A key can't be "fixed" and "variable"
        raise ValueError("The dictionaries model_params and hp_params share at least one key")

    if len(set([len(_) for _ in hp_params.values()])) != 1:
        raise ValueError("The values of hp_params must have the same length")

    if n_splits <= 1:
        raise ValueError(f"The number of crossvalidation splits ({n_splits}) must be more than 1")

    #################
    # Manage scoring
    #################

    # Get scoring functions
    if scoring_fn == 'accuracy':
        scoring_fn = lambda x: x['Accuracy']
    elif scoring_fn == 'f1':
        scoring_fn = lambda x: x['F1-Score']
    elif scoring_fn == 'precision':
        scoring_fn = lambda x: x['Precision']
    elif scoring_fn == 'recall':
        scoring_fn = lambda x: x['Recall']

    #################
    # Manage x_train & y_train format
    #################

    if not isinstance(kwargs_fit['x_train'], (pd.DataFrame, pd.Series)):
        kwargs_fit['x_train'] = pd.Series(kwargs_fit['x_train'].copy())

    if not isinstance(kwargs_fit['y_train'], (pd.DataFrame, pd.Series)):
        kwargs_fit['y_train'] = pd.Series(kwargs_fit['y_train'].copy())

    #################
    # Process
    #################

    # Loop on hyperparameters
    nb_search = len(list(hp_params.values())[0])
    logger.info("Beginning of hyperparameters search")
    logger.info(f"Number of model fits : {nb_search} (search number) x {n_splits} (CV splits number) = {nb_search * n_splits}")

    # DataFrame for stocking metrics :
    metrics_df = pd.DataFrame(columns=['index_params', 'index_fold', 'Score', 'Accuracy', 'F1-Score', 'Precision', 'Recall'])
    for i in range(nb_search):

        # Display informations
        logger.info(f"Search n°{i + 1}")
        tmp_hp_params = {k: v[i] for k, v in hp_params.items()}
        logger.info("Tested hyperparameters : ")
        logger.info(pprint.pformat(tmp_hp_params))

        # Get folds (shuffle recommended since the classes could be ordered)
        if model_params['multi_label']:
            k_fold = KFold(n_splits=n_splits, shuffle=True)  # Can't stratify on multi-labels
        else:
            k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True)

        # Process each fold
        for j, (train_index, valid_index) in enumerate(k_fold.split(kwargs_fit['x_train'], kwargs_fit['y_train'])):
            logger.info(f"Search n°{i + 1}/{nb_search} - fit n°{j + 1}/{n_splits}")
            # get tmp x, y
            x_train, x_valid = kwargs_fit['x_train'].iloc[train_index], kwargs_fit['x_train'].iloc[valid_index]
            y_train, y_valid = kwargs_fit['y_train'].iloc[train_index], kwargs_fit['y_train'].iloc[valid_index]
            # Get tmp model
            # Manage model_dir
            tmp_model_dir = os.path.join(utils.get_models_path(), datetime.now().strftime("tmp_%Y_%m_%d-%H_%M_%S"))
            # The next line prioritize the last dictionary
            # We force a temporary folder and a low save level (we only want the metrics)
            model_tmp = model_cls(**{**model_params, **tmp_hp_params, **{'model_dir': tmp_model_dir, 'level_save': 'LOW'}})
            # Setting log level to ERROR
            model_tmp.logger.setLevel(logging.ERROR)
            # Let's fit ! (priority to the last dictionary)
            model_tmp.fit(**{**kwargs_fit, **{'x_train': x_train, 'y_train': y_train, 'x_valid': x_valid, 'y_valid': y_valid}})
            # Let's predict !
            y_pred = model_tmp.predict(x_valid)
            # Get metrics !
            metrics_func = model_tmp.get_metrics_simple_multilabel if model_tmp.multi_label else model_tmp.get_metrics_simple_monolabel
            metrics_tmp = metrics_func(y_valid, y_pred)
            metrics_tmp = metrics_tmp[metrics_tmp.Label == "All"].copy()  # Add .copy() to avoid pandas settingwithcopy
            metrics_tmp["Score"] = scoring_fn(metrics_tmp.iloc[0].to_dict())  # type: ignore
            metrics_tmp["index_params"] = i
            metrics_tmp["index_fold"] = j
            metrics_tmp = metrics_tmp[metrics_df.columns]  # Keeping only the necessary columns
            metrics_df = pd.concat([metrics_df, metrics_tmp], ignore_index=True)
            # Delete the temporary model : the final model must be refitted on the whole dataset
            del model_tmp
            gc.collect()
            shutil.rmtree(tmp_model_dir)
        # Display score
        logger.info(f"Score for search n°{i + 1}: {metrics_df[metrics_df['index_params'] == i]['Score'].mean()}")

    # Metric agregation for all the folds
    metrics_df = metrics_df.join(metrics_df[['index_params', 'Score']].groupby('index_params').mean().rename({'Score': 'mean_score'}, axis=1), on='index_params', how='left')

    # Select the set of parameters with the best mean score (on the folds)
    best_index = metrics_df[metrics_df.mean_score == metrics_df.mean_score.max()]["index_params"].values[0]
    best_params = {k: v[best_index] for k, v in hp_params.items()}
    logger.info(f"Best results for the set of parameter n°{best_index + 1}: {pprint.pformat(best_params)}")

    # Instanciation of a model with the best parameters
    best_model = model_cls(**{**model_params, **best_params})

    # Save the metrics report of the hyperparameters search and the tested parameters
    csv_path = os.path.join(best_model.model_dir, "hyper_params_results.csv")
    metrics_df.to_csv(csv_path, sep='{{default_sep}}', index=False, encoding='{{default_encoding}}')
    json_data = {
        'model_params': model_params,
        'scoring_fn': pickle.source.getsourcelines(scoring_fn)[0],
        'n_splits': n_splits,
        'hp_params_set': {i: {k: v[i] for k, v in hp_params.items()} for i in range(nb_search)},
    }
    json_path = os.path.join(best_model.model_dir, "hyper_params_tested.json")
    with open(json_path, 'w', encoding='{{default_encoding}}') as f:
        json.dump(json_data, f, indent=4, cls=utils.NpEncoder)

    # TODO: We are forced to reset the logging level which is linked to the class
    best_model.logger.setLevel(logging.getLogger('{{package_name}}').getEffectiveLevel())

    # Return model to be fitted
    return best_model


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
