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


import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from typing import List, Dict, Tuple, Any, Union, Optional

logger = logging.getLogger(__name__)


class Estimator(BaseEstimator):
    '''Base class for the classes defined below. Implements _validate_input and fit_transform.'''

    def __init__(self, input_length: Union[int, None]) -> None:
        '''Initialization of the class

        Args:
            input_length (int): The number of columns of the input (used in _validate_input())
        '''
        self.input_length = input_length

    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        '''Validates input format

        Args:
            X (np.ndarray or pd.DataFrame): Input to validate
        Raises:
            ValueError: If the shape of the input does not correspond to self.input_length
        Returns:
            np.ndarray or pd.DataFrame: A copy of X
        '''
        if self.input_length is not None and X.shape[1] != self.input_length:
            raise ValueError(f"Bad shape: ({X.shape[1]} != {self.input_length})")

        # Mandatory copy in order not to modify the original !
        if isinstance(X, pd.DataFrame):
            return X.copy(deep=True)
        else:
            return X.copy()

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series, pd.DataFrame]) -> Any:
        '''Fit transformer

        Args:
            X (np.ndarray or pd.DataFrame): Array-like, shape = [n_samples, n_features]
            y (np.ndarray or pd.Series or pd.DataFrame): Array-like, shape = [n_samples, n_targets]
        Returns:
            self
        '''
        raise NotImplementedError("'fit' needs to be overridden")

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''Transforms X

        Args:
            X (np.ndarray or pd.DataFrame): Array-like, shape = [n_samples, n_features]
        Returns:
            Transformed X
        '''
        raise NotImplementedError("'transform' needs to be overridden")

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series, pd.DataFrame, None] = None) -> np.ndarray:
        '''Applies both fit & transform.

        Args:
            X (np.ndarray or pd.DataFrame): Shape = [n_samples, n_features]
        Kwargs:
            y: Array-like, shape = [n_samples]
        Returns:
            X_out: Array-like, shape [n_samples, n_features]
                        Transformed input.
        '''
        self.fit(X, y)
        return self.transform(X)


class AutoLogTransform(Estimator):
    '''Automatically applies a log transformation on numerical data if the distribution of the variables
       is skewed (abs(skew) > min_skewness) and if there is an amplitude superior to min_amplitude between
       the 10th and 90th percentiles

    WARNING: YOUR DATA MUST BE POSITIVE IN ORDER FOR EVERYTHING TO WORK CORRECTLY
    '''

    def __init__(self, min_skewness: float = 2, min_amplitude: float = 10E3) -> None:
        '''Initialization of the class

        Kwargs:
            min_skewness (float): Absolute value of the required skewness to apply a log transformation
            min_amplitude (float): Minimal value of the amplitude between the 10th and 90th percentiles
                                   required to apply a log transformation
        '''

        super().__init__(None)
        # Set attributes
        self.min_skewness = min_skewness
        self.min_amplitude = min_amplitude

        # Columns on which to apply the transformation
        # Set on fit
        # Warning: sklearn does not support columns name, so we can only use indexes
        # Hence, X input must expose same columns order (this won't be checked)
        self.applicable_columns_index: Optional[List[Any]] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Any = None) -> Any:
        '''Fit transformer

        Args:
            X (np.ndarray or pd.DataFrame): Array-like, shape = [n_samples, n_features]
        Kwargs:
            y (None): Not used here
        Returns:
            self
        '''
        X = self._validate_input(X)
        # If x is a numpy array, casts it in pd.DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        # Otherwise, we reset the columns name because sklearn can't manage them
        else:
            X = X.rename(columns={col: i for i, col in enumerate(X.columns)})
        self.input_length = X.shape[1]

        # Get applicable columns
        skew = X.skew()
        candidates = list(skew[abs(skew) > self.min_skewness].index)
        if len(candidates) > 0:
            q10 = X.iloc[:, candidates].quantile(q=0.1)
            q90 = X.iloc[:, candidates].quantile(q=0.9)
            amp = q90 - q10
            # Update applicable_columns_index
            self.applicable_columns_index = list(amp[amp > self.min_amplitude].index)

        self.fitted_ = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''Transforms X - apply log on applicable columns

        Args:
            X (np.ndarray or pd.DataFrame): Array-like, shape = [n_samples, n_features]
        Returns:
            X_out: Array-like, shape [n_samples, n_features]
                        Transformed input.
        '''
        # Validate input
        check_is_fitted(self, 'fitted_')
        X = self._validate_input(X)

        # If x is a numpy array, casts it in pd.DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Log transformation on the applicable columns
        if len(self.applicable_columns_index) > 0:
            X.iloc[:, self.applicable_columns_index] = np.log(X.iloc[:, self.applicable_columns_index])

        # Compatibility -> returns numpy array
        return X.to_numpy()


class ThresholdingTransform(Estimator):
    '''Applies a threshold on columns.
    If min and max values are given, the threshold is manual; otherwise it is statistical.
    '''

    def __init__(self, thresholds: List[Tuple], quantiles: tuple = (0.05, 0.95)) -> None:
        '''Initialization of the class

        Args:
            tresholds (list of tuple): Each tuple contains (min_val, max_val).
        Kwargs:
            quantiles (tuple): Tuple containing (quantile_min, quantile_max)
        Raises:
            ValueError: If quantiles values are not between 0 and 1 and if quantiles[0] >= quantiles[1]
        '''
        if not 0 < quantiles[0] < 1 or not 0 < quantiles[1] < 1 or not quantiles[0] < quantiles[1]:
            raise ValueError(f"The values contained in quantiles should verify quantile_min < quantile_max and both > 0 and < 1. quantiles = {quantiles} is not supported.")

        super().__init__(len(thresholds))

        # Set attributes
        self.thresholds = thresholds
        self.fitted_thresholds: List[tuple] = []
        self.quantiles = quantiles

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Any = None) -> Any:
        '''Fits the ThresholdingTransform on X.

        Args:
            X (np.ndarray or pd.DataFrame): Shape [n_samples, n_features]
        Kwargs:
            y (None): Not used here.
        Returns:
            self: ThresholdingTransform
        '''
        X = self._validate_input(X)
        # If X is a numpy array, casts it as a pd.DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Fits column one by one
        for col_index, item in enumerate(self.thresholds):
            val_min, val_max = item
            if val_min is None:
                val_min = X.iloc[:, col_index].quantile(q=self.quantiles[0])
            if val_max is None:
                val_max = X.iloc[:, col_index].quantile(q=self.quantiles[1])
            self.fitted_thresholds.append((col_index, val_min, val_max))

        self.fitted_ = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''Impute all missing values in X.

        Args:
            X (np.ndarray or pd.DataFrame): Shape (n_samples, n_features)
                The input data to complete.
        '''
        check_is_fitted(self, 'fitted_')
        X = self._validate_input(X)
        # If X is a numpy array, casts it to pd.DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        for col_index, val_min, val_max in self.fitted_thresholds:
            X.iloc[:, col_index][X.iloc[:, col_index] < val_min] = val_min
            X.iloc[:, col_index][X.iloc[:, col_index] > val_max] = val_max

        return X.to_numpy()  # Compatibility -> return a numpy array


class AutoBinner(Estimator):
    '''Automatically creates a "other" category when the categories are heavily unbalanced
    /!\ Replaces the values of some categories /!\
    '''

    def __init__(self, strategy: str = "auto", min_cat_count: int = 3, threshold: float = 0.05) -> None:
        '''Initialization of the class

        Kwargs:
            strategy (str): 'auto' or 'threshold'
                - 'auto': Aggregates all categories as long as their cumulated frequence is less than threshold
                - 'threshold': Aggregates all category whose frequence is less than threshold
            min_cat_count (int): Minimal number of category to keep
            threshold (float): The threshold to consider
        Raises:
            ValueError: The object strategy must be in the list of allowed strategies
            ValueError: The object min_cat_count must be non negative
            ValueError: The object threshold must be in ]0,1[
        '''
        super().__init__(None)

        allowed_strategies = ["threshold", "auto"]
        self.strategy = strategy
        if self.strategy not in allowed_strategies:
            raise ValueError(f"Can only use these strategies: {allowed_strategies}. "
                             f"Got strategy={strategy}")
        if min_cat_count < 0:
            raise ValueError("min_cat_count must be non negative")
        if not (0 < threshold < 1):
            raise ValueError(f"threshold must be in ]0,1[, not {threshold}")

        # Set attributes
        self.min_cat_count = min_cat_count
        self.threshold = threshold
        self.kept_cat_by_index: Dict[int, list] = {}

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Any = None) -> Any:
        '''Fit the AutoBinner on X.

        Args:
            X (np.ndarray or pd.DataFrame): Shape (n_samples, n_features)
        Kwargs:
            y: Not used here
        Returns:
            self: AutoBinner
        '''
        X = self._validate_input(X)
        # If x is a numpy array, casts it in pd.DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.input_length = X.shape[1]
        # Fits column one by one
        for col_index in range(self.input_length):
            # Get col serie
            X_tmp_ser = X.iloc[:, col_index]
            # Get unique vals
            unique_cat = list(X_tmp_ser.unique())
            # If less vals than min threshold, set this column allowed values with all uniques values
            if len(unique_cat) <= self.min_cat_count:
                self.kept_cat_by_index[col_index] = unique_cat
                continue

            # If more vals than min threshold, keep values based on strategy
            table = X_tmp_ser.value_counts() / X_tmp_ser.count()
            table = table.sort_values()
            if self.strategy == 'auto':
                table = np.cumsum(table)
            # If only one category is less than the threshold, we do not need to transform it
            if table[1] > self.threshold:
                self.kept_cat_by_index[col_index] = unique_cat
                continue
            # Otherwise, we get rid of the superfluous categories
            else:
                to_remove = list(table[table < self.threshold].index)
                for item in to_remove:
                    unique_cat.remove(item)
                self.kept_cat_by_index[col_index] = unique_cat

        self.fitted_ = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''Imputes all missing values in X.

        Args:
            X (np.ndarray or pd.DataFrame): Shape (n_samples, n_features)
                The input data to complete.
        '''
        check_is_fitted(self, 'fitted_')
        X = self._validate_input(X)
        # If x is a numpy array, casts it in pd.DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        for col_index in range(self.input_length):
            X.iloc[:, col_index] = X.iloc[:, col_index].apply(lambda x: x if x in self.kept_cat_by_index[col_index] else 'other_')

        return X.to_numpy()  # Compatibility, returns a numpy array


class EmbeddingTransformer(Estimator):
    '''Constructs a transformer that apply an embedding mapping to Categorical columns'''

    def __init__(self, embedding: Union[str, dict], none_strategy: str = 'zeros') -> None:
        '''Initialization of the class

        Args:
            embedding (str or dict): Embedding to use
                - If dict -> ok, ready to go
                - If str -> path to the file to load (json)
        Kwargs:
            none_strategy (str): Strategy to fill elements not in embedding
                - Zeros: only 0s
        Raises:
            ValueError: If strategy is not in the allowed strategies
            ValueError: If the embedding is of type str but does not end in .json
            FileNotFoundError: If the path to the embedding does not exist
        '''
        super().__init__(None)

        # Check none strategy
        allowed_strategies = ["zeros"]
        self.none_strategy = none_strategy
        if self.none_strategy not in allowed_strategies:
            raise ValueError(f"Can only use these strategies: {allowed_strategies}, got strategy={self.none_strategy}")

        # If str, loads the embedding
        if isinstance(embedding, str):
            if not embedding.endswith('.json'):
                raise ValueError(f"The file {embedding} must be a .json file")
            if not os.path.exists(embedding):
                raise FileNotFoundError(f"The file {embedding} does not exist")
            with open(embedding, 'r', encoding='{{default_encoding}}') as f:
                self.embedding = json.load(f)
        else:
            self.embedding = embedding

        # Get embedding size
        self.embedding_size = len(self.embedding[list(self.embedding.keys())[0]])
        # Other params
        self.n_missed = 0

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Any = None) -> Any:
        '''Fit transformer

        Args:
            X (np.ndarray or pd.DataFrame): Shape (n_samples, n_features)
        Kwargs:
            y: Not used here
        Returns
            self (EmbeddingTransformer)
        '''
        X = self._validate_input(X)
        # If x is a numpy array, casts it in pd.DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.input_length = X.shape[1]

        # Nothing to do

        self.fitted_ = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''Transform X - embedding mapping

        Args:
            X (np.ndarray or pd.DataFrame): Shape (n_samples, n_features)
        Raises:
            ValueError: If there are missing columns
        Returns:
            X_out (np.ndarray): Shape (n_samples, n_features) transformed input.
        '''
        X = self._validate_input(X)
        # If x is a numpy array, casts it in pd.DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        n_rows = X.shape[0]

        # Apply mapping
        new_df = pd.DataFrame()
        for col in X.columns:
            self.n_missed = 0  # Counts the number of missing elements in the embedding
            tmp_serie = X[col].apply(self.apply_embedding)  # Updates self.n_missed
            new_df = pd.concat([new_df, pd.DataFrame(tmp_serie.to_list())], axis=1)
            perc_missed = self.n_missed / n_rows * 100
            if perc_missed != 0:
                logger.warning(f"Warning, {self.n_missed} ({perc_missed} %) missing elements in the embedding for column {col}")

        return new_df.to_numpy()  # Compatibility, returns a numpy array

    def apply_embedding(self, content) -> list:
        '''Applies embedding mapping

        Args:
            content: Content on which to apply embedding mapping
        Raises:
            ValueError: If the strategy is not recognized
        Returns:
            list: Applied embedding
        '''
        if content in self.embedding.keys():
            return self.embedding[content]
        else:
            self.n_missed += 1
            if self.none_strategy == 'zeros':
                return [0] * self.embedding_size
            else:
                raise ValueError(f"Strategy {self.none_strategy} not recognized")

    def get_feature_names(self, features_in: list, *args, **kwargs) -> np.ndarray:
        '''Returns feature names for output features.

        Args:
            features_in (list): list of features

        Returns:
            output_feature_names: ndarray of shape (n_output_features,)
                Array of feature names.
        '''
        check_is_fitted(self, 'fitted_')
        new_features = [f"emb_{feat}_{i}" for feat in features_in for i in range(self.embedding_size)]
        return np.array(new_features, dtype=object)


if __name__ == '__main__':
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
