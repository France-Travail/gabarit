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

from . import column_preprocessors

# Get logger
logger = logging.getLogger(__name__)


def get_pipelines_dict() -> dict:
    '''Gets a dictionary of available preprocessing pipelines

    Returns:
        dict: Dictionary of preprocessing pipelines
    '''
    pipelines_dict = {
        # - /!\ DO NOT DELETE no_preprocess -> necessary for compatibility /!\ -
        # Identity transformer, hence we specify verbose_feature_names_out to False to not change columns names
        'no_preprocess': ColumnTransformer([('identity', FunctionTransformer(lambda x: x),
                                             make_column_selector())], verbose_feature_names_out=False),
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

    # /!\
    # BE VERY CAUTIOUS WHEN USING FunctionTransformer !
    # A pickled pipeline can still depends on a module definition. Hence, even when pickled, you may not have consistent results !
    # Please try to use lambdas or local function, without any dependency. It reduces risk of changes.
    # If you still have to use a module function, try to never change it later on.
    # https://stackoverflow.com/questions/73788824/how-can-i-save-reload-a-functiontransformer-object-and-expect-it-to-always-wor
    # https://github.com/OSS-Pole-Emploi/gabarit/issues/63
    # /!\

    # /!\ EXEMPLE HERE /!\
    # Good practice: Use directly the names of the columns instead of a "selector"
    # WARNING: The text pipeline is supposed to work on a column 'text' -> Please adapt it to your project if you want to use it

    # By default, we only keep the preprocess on numerical columns
    transformers = [
        ('num', numeric_pipeline, make_column_selector(dtype_include='number')),
        # ('cat', cat_pipeline, make_column_selector(dtype_include='category')), # To convert a column in a column with dtype category: df["A"].astype("category")
        # ('text', text_pipeline, 'text'), # CountVectorizer possible one column at a time
    ]

    # TODO: add sparse compatibility !
    # Use somethings like this :
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
    Raises:
        AttributeError : The pipeline is not fitted
        ValueError : The number of columns is not the same between the pipeline and the preprocessed DataFrame
    Returns:
        pd.DataFrame: Dataframe with columns' name
    '''
    # Use deepcopy !
    new_df = df.copy(deep=True)
    # Check if fitted:
    if not hasattr(pipeline, 'transformers_'):
        raise AttributeError("The pipeline must be fitted to use the function retrieve_columns_from_pipeline")
    # EXPERIMENTAL: We do a try... except... if we can't get the names
    # First try : use sklearn get_feature_names_out function (might crash)
    # Second try : backup on old custom method
    # Third solution : ['x0', 'x1', ...]
    try:
        new_columns = pipeline.get_feature_names_out()
        if len(new_columns) != new_df.shape[1]:
            raise ValueError(f"There is a discrepancy in the number of columns between the preprocessed DataFrame ({new_df.shape[1]}) and the pipeline ({len(new_columns)}).")
    # No solution
    except Exception as e:
        logger.error("Can't get the names of the columns. Backup on ['x0', 'x1', ...]")
        logger.error(repr(e))
        new_columns = [f'x{i}' for i in range(len(new_df.columns))]
    # TODO : check for duplicates in new_columns ???
    new_df.columns = new_columns
    return new_df


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
