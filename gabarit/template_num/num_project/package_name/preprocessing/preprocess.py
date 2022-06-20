#!/usr/bin/env python3

## Preprocessing functions
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


import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, SelectorMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_extraction.text import CountVectorizer, _VectorizerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder

from {{package_name}}.preprocessing import column_preprocessors

# Get logger
logger = logging.getLogger(__name__)


def get_pipelines_dict() -> dict:
    '''Gets a dictionary of available preprocessing pipelines

    Returns:
        dict: Dictionary of preprocessing pipelines
    '''
    pipelines_dict = {
        'no_preprocess': ColumnTransformer([('identity', FunctionTransformer(lambda x: x), make_column_selector())]),  # - /!\ DO NOT DELETE -> necessary for compatibility /!\ -
        'preprocess_P1': preprocess_P1(),  # Example of a pipeline
        # 'preprocess_AUTO': preprocess_auto(), # Automatic preprocessing based on statistics on data
        # 'preprocess_P2': preprocess_P2 , ETC ...
    }
    return pipelines_dict


def get_pipeline(pipeline_str: str) -> ColumnTransformer:
    '''Gets a pipeline from its name

    Args:
        pipeline_str (str): Name of the pipeline
    Raises:
        ValueError: If the name of the pipeline is not known
    Returns:
        ColumnTransfomer: Pipeline to be used for the preprocessing
    '''
    # Process
    pipelines_dict = get_pipelines_dict()
    if pipeline_str not in pipelines_dict.keys():
        raise ValueError(f"The pipeline {pipeline_str} is not known.")
    # Get pipeline
    pipeline = pipelines_dict[pipeline_str]
    # Return
    return pipeline


def preprocess_P1() -> ColumnTransformer:
    '''Gets "default" preprocessing pipeline

    Returns:
        ColumnTransformer: The pipeline
    '''
    numeric_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    # cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
    # text_pipeline = make_pipeline(CountVectorizer(), SelectKBest(k=5))

    # Check https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html
    # and https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html#pandas.DataFrame.select_dtypes
    # to understand make_column_selector

    # /!\ EXEMPLE HERE /!\
    # Good practice: Use directly the names of the columns instead of a "selector"
    # WARNING: The text pipeline is supposed to work on a column 'text' -> Please adapt it to your project

    # By default, we only keep the preprocess on numerical columns
    transformers = [
        ('num', numeric_pipeline, make_column_selector(dtype_include='number')),
        # ('cat', cat_pipeline, make_column_selector(dtype_include='category')), # To convert a column in a column with dtype category: df["A"].astype("category")
        # ('text', text_pipeline, 'text'), # CountVectorizer possible one column at a time
    ]

    # TODO: add sparse compatibility !
    # Use somethings like these:
    # - After applying a pipeline ...
    # if scipy.sparse.issparse(preprocessed_x):
    #     preprocessed_df = pd.DataFrame.sparse.from_spmatrix(preprocessed_x)
    # - Before training ...
    # x_train = x_train.sparse.to_coo().tocsr()
    # x_valid = x_valid.sparse.to_coo().tocsr()
    # ...
    pipeline = ColumnTransformer(transformers, sparse_threshold=0, remainder='drop')  # Use remainder='passthrough' to keep all other columns (not recommended)

    return pipeline


# TODO
def preprocess_auto() -> ColumnTransformer:
    '''Gets an "automatic" pipeline. Different functions are applied depending on stats calculated on the data

    Returns:
        ColumnTransformer: The automatic pipeline
    '''
    # Numeric :
    # 1) SimpleImputer()
    # 2) If abs(skew) > 2 && pctl(90) - pctl(10) > 10^3 => logtransform
    # 3) StandardScaler()
    # Categorical :
    # 1) SimpleImputer()
    # 2) If #cat > 5; We accumulate the less represented instances in a meta-category "other"
    # 3) OneHot
    pass


def retrieve_columns_from_pipeline(df: pd.DataFrame, pipeline: ColumnTransformer) -> pd.DataFrame:
    '''Retrieves columns name after preprocessing

    Args:
        df (pd.DataFrame): Dataframe after preprocessing (without target)
        pipeline (ColumnTransformer): Used pipeline
    Returns:
        pd.DataFrame: Dataframe with columns' name
    '''
    # EXPERIMENTAL: We do a try... except... if we can't get the names
    try:
        # Check if fitted:
        if not hasattr(pipeline, '_columns'):
            raise AttributeError("The pipeline must be fitted to use the function retrieve_columns_from_pipeline")
        new_columns = get_ct_feature_names(pipeline)
        assert len(new_columns) == df.shape[1], "There is a discrepancy in the number of columns" +\
                                                f" between the preprocessed DataFrame ({df.shape[1]})" +\
                                                f" and the pipeline ({len(new_columns)})."
        df.columns = new_columns
    except Exception as e:
        logger.error("Can't get the names of the columns. Cancel it (experimental)")
        logger.error("Continue the process")
        logger.error(repr(e))
    return df


def get_ct_feature_names(ct: ColumnTransformer) -> list:
    '''Gets the names of the columns when considering a fitted ColumnTransfomer
    From: https://stackoverflow.com/questions/57528350/can-you-consistently-keep-track-of-column-labels-using-sklearns-transformer-api

    Args:
        ColumnTransformer: Column tranformer to be processed
    Returns:
        list: List of new feature names
    '''
    # Handles all estimators, pipelines inside ColumnTransfomer
    # does not work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name != 'remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    if type(step) == tuple:
                        step = step[1]
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator == 'passthrough':
            output_features.extend(ct._feature_names_in[features])

    return output_features


# TODO: Check if the estimators all have the method get_feature_names_out implemented (https://github.com/scikit-learn/scikit-learn/issues/21308)
# to replace these functions
def get_feature_out(estimator, features_in: list) -> list:
    '''Gets the name of a column when considering a fitted estimator
    From: https://stackoverflow.com/questions/57528350/can-you-consistently-keep-track-of-column-labels-using-sklearns-transformer-api

    Args:
        (?): Estimator to be processed
        (list): Input columns
    Returns:
        list: List of new feature names
    '''
    if hasattr(estimator, 'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f'vec_{f}' for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(features_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(features_in)[estimator.get_support()]
    else:
        return features_in


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
