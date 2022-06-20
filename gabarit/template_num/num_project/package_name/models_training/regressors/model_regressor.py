#!/usr/bin/env python3
# type: ignore
# Too complicated to manage Mixin & types

## Definition of a parent class for the regressor models
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
# Classes :
# - ModelRegressorMixin -> Parent class for regressor models


import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union
from yellowbrick.regressor import ResidualsPlot, PredictionError

from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

from {{package_name}}.monitoring.model_logger import ModelLogger

sns.set(style="darkgrid")


class ModelRegressorMixin:
    '''Parent class (Mixin) for regressor models'''

    def __init__(self, level_save: str = 'HIGH', **kwargs) -> None:
        '''Initialization of the class

        Kwargs:
            level_save (str): Level of saving
                LOW: stats + configurations + logger keras - /!\\ The model can't be reused /!\\ -
                MEDIUM: LOW + hdf5 + pkl + plots
                HIGH: MEDIUM + predictions
        Raises:
             ValueError: If the object level_save is not a valid option (['LOW', 'MEDIUM', 'HIGH'])
        '''
        super().__init__(level_save=level_save, **kwargs)  # forwards level_save & all unused arguments

        if level_save not in ['LOW', 'MEDIUM', 'HIGH']:
            raise ValueError(f"The object level_save ({level_save}) is not a valid option (['LOW', 'MEDIUM', 'HIGH'])")

        # Get logger
        self.logger = logging.getLogger(__name__)

        # Model type
        self.model_type = 'regressor'

        # TODO: add multi-outputs !

        # Other options
        self.level_save = level_save

    def inverse_transform(self, y: Union[list, np.ndarray]) -> Union[list, tuple]:
        '''Identity function - Manages compatibility with classifiers

        Args:
            y (list | np.ndarray): Array-like, shape = [n_samples, 1]
        Returns:
            (np.ndarray): List, shape = [n_samples, 1]
        '''
        return list(y) if isinstance(y, np.ndarray) else y

    def get_and_save_metrics(self, y_true, y_pred, df_x: Union[pd.DataFrame, None] = None,
                             series_to_add: Union[List[pd.Series], None] = None, type_data: str = '',
                             model_logger: Union[ModelLogger, None] = None) -> pd.DataFrame:
        '''Gets and saves the metrics of a model

        Args:
            y_true (?): Array-like, shape = [n_samples,]
            y_pred (?): Array-like, shape = [n_samples,]
        Kwargs:
            df_x (pd.DataFrame or None): Input dataFrame used for the prediction
            series_to_add (list<pd.Series>): List of pd.Series to add to the dataframe
            type_data (str): Type of dataset (validation, test, ...)
            model_logger (ModelLogger): Custom class to log the metrics with MLflow
        Returns:
            pd.DataFrame: The dataframe containing the statistics
        '''

        # Cast to np.array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Save a predictionn file if wanted
        if self.level_save == 'HIGH':
            # Inverse transform
            y_true_df = list(self.inverse_transform(y_true))
            y_pred_df = list(self.inverse_transform(y_pred))

            # Concat in a dataframe
            if df_x is not None:
                df = df_x.copy()
                df['y_true'] = y_true_df
                df['y_pred'] = y_pred_df
            else:
                df = pd.DataFrame({'y_true': y_true_df, 'y_pred': y_pred_df})
            # Add column abs_err
            df.loc[:, 'abs_err'] = df[['y_true', 'y_pred']].apply(lambda x: x.y_true - x.y_pred, axis=1)
            # Add column rel_err
            df.loc[:, 'rel_err'] = df[['y_true', 'y_pred']].apply(lambda x: (x.y_true - x.y_pred) / abs(x.y_true), axis=1)
            # Add some more columns
            if series_to_add is not None:
                for ser in series_to_add:
                    df[ser.name] = ser.reset_index(drop=True).reindex(index=df.index)  # Reindex correctly

            # Save predictions
            file_path = os.path.join(self.model_dir, f"predictions{'_' + type_data if len(type_data) > 0 else ''}.csv")
            df.sort_values('abs_err', ascending=True).to_csv(file_path, sep='{{default_sep}}', index=None, encoding='{{default_encoding}}')

        # Get global metrics
        metric_mae = mean_absolute_error(y_true, y_pred)
        metric_mse = mean_squared_error(y_true, y_pred)
        metric_rmse = mean_squared_error(y_true, y_pred, squared=False)
        metric_explained_variance_score = explained_variance_score(y_true, y_pred)
        metric_r2 = r2_score(y_true, y_pred)

        # Global statistics
        self.logger.info('-- * * * * * * * * * * * * * * --')
        self.logger.info(f"Statistics{' ' + type_data if len(type_data) > 0 else ''}")
        self.logger.info('--------------------------------')
        self.logger.info(f"MAE : {round(metric_mae, 5)}")
        self.logger.info(f"MSE : {round(metric_mse, 5)}")
        self.logger.info(f"RMSE : {round(metric_rmse, 5)}")
        self.logger.info(f"Explained variance : {round(metric_explained_variance_score, 5)}")
        self.logger.info(f"R² (coefficient of determination) : {round(metric_r2, 5)}")
        self.logger.info('--------------------------------')

        # Metrics file
        df_stats = pd.DataFrame(columns=['Label', 'MAE', 'MSE',
                                         'RMSE', 'Explained variance',
                                         'Coefficient of determination'])

        # TODO : add multi-outputs and stats for each output

        # Add global statistics
        global_stats = {
            'Label': 'All',
            'MAE': metric_mae,
            'MSE': metric_mse,
            'RMSE': metric_rmse,
            'Explained variance': metric_explained_variance_score,
            'Coefficient of determination': metric_r2,
        }
        df_stats = df_stats.append(global_stats, ignore_index=True)

        # Save .csv
        file_path = os.path.join(self.model_dir, f"mae{'_' + type_data if len(type_data) > 0 else ''}@{metric_mae}.csv")
        df_stats.to_csv(file_path, sep='{{default_sep}}', index=False, encoding='{{default_encoding}}')

        # Save some metrics
        mae_path = os.path.join(self.model_dir, f"mae{'_' + type_data if len(type_data) > 0 else ''}@{round(metric_mae, 5)}")
        with open(mae_path, 'w'):
            pass
        mse_path = os.path.join(self.model_dir, f"mse{'_' + type_data if len(type_data) > 0 else ''}@{round(metric_mse, 5)}")
        with open(mse_path, 'w'):
            pass
        rmse_path = os.path.join(self.model_dir, f"rmse{'_' + type_data if len(type_data) > 0 else ''}@{round(metric_rmse, 5)}")
        with open(rmse_path, 'w'):
            pass
        explained_variance_path = os.path.join(self.model_dir, f"explained_variance{'_' + type_data if len(type_data) > 0 else ''}@{round(metric_explained_variance_score, 5)}")
        with open(explained_variance_path, 'w'):
            pass
        r2_path = os.path.join(self.model_dir, f"r2{'_' + type_data if len(type_data) > 0 else ''}@{round(metric_r2, 5)}")
        with open(r2_path, 'w'):
            pass

        # Upload metrics in mlflow (or another)
        if model_logger is not None:
            # TODO : To put in a function
            # Prepare parameters
            label_col = 'Label'
            metrics_columns = [col for col in df_stats.columns if col != label_col]

            # Log labels
            labels = df_stats[label_col].values
            for i, label in enumerate(labels):
                model_logger.log_param(f'Label {i}', label)
            # Log metrics
            ml_flow_metrics = {}
            for i, row in df_stats.iterrows():
                for c in metrics_columns:
                    metric_key = f"{row[label_col]} --- {c}"
                    # Check that mlflow accepts the key, otherwise, replace it
                    if not model_logger.valid_name(metric_key):
                        metric_key = f"Label {i} --- {c}"
                    ml_flow_metrics[metric_key] = row[c]
            # Log metrics
            model_logger.log_metrics(ml_flow_metrics)

        # Plots
        if self.level_save in ['MEDIUM', 'HIGH']:
            # TODO: put a condition on the maximum number of points ?
            is_train = True if type_data == 'train' else False
            if is_train:
                self.plot_prediction_errors(y_true_train=y_true, y_pred_train=y_pred,
                                            y_true_test=None, y_pred_test=None,
                                            type_data=type_data)
                self.plot_residuals(y_true_train=y_true, y_pred_train=y_pred,
                                    y_true_test=None, y_pred_test=None,
                                    type_data=type_data)
            else:
                self.plot_prediction_errors(y_true_train=None, y_pred_train=None,
                                            y_true_test=y_true, y_pred_test=y_pred,
                                            type_data=type_data)
                self.plot_residuals(y_true_train=None, y_pred_train=None,
                                    y_true_test=y_true, y_pred_test=y_pred,
                                    type_data=type_data)

        # Return metrics
        return df_stats

    def get_metrics_simple(self, y_true, y_pred) -> pd.DataFrame:
        '''Gets metrics on predictions (single-output for now)
        Same as the method get_and_save_metrics but without all the fluff (save, etc.)

        Args:
            y_true (?): Array-like, shape = [n_samples]
            y_pred (?): Array-like, shape = [n_samples]
        Returns:
            pd.DataFrame: The dataframe containing statistics
        '''
        # Cast to np.array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Get global metrics:
        metric_mae = mean_absolute_error(y_true, y_pred)
        metric_mse = mean_squared_error(y_true, y_pred)
        metric_rmse = mean_squared_error(y_true, y_pred, squared=False)
        metric_explained_variance_score = explained_variance_score(y_true, y_pred)
        metric_r2 = r2_score(y_true, y_pred)

        # Metrics file
        df_stats = pd.DataFrame(columns=['Label', 'MAE', 'MSE',
                                         'RMSE', 'Explained variance',
                                         'Coefficient of determination'])

        # TODO : add multi-outputs and stats for each output

        # Add global statistics
        global_stats = {
            'Label': 'All',
            'MAE': metric_mae,
            'MSE': metric_mse,
            'RMSE': metric_rmse,
            'Explained variance': metric_explained_variance_score,
            'Coefficient of determination': metric_r2,
        }
        df_stats = df_stats.append(global_stats, ignore_index=True)

        # Return dataframe
        return df_stats

    def plot_prediction_errors(self, y_true_train: Union[np.ndarray, None] = None, y_pred_train: Union[np.ndarray, None] = None,
                               y_true_test: Union[np.ndarray, None] = None, y_pred_test: Union[np.ndarray, None] = None,
                               type_data: str = '') -> None:
        '''Plots prediction errors

        We use yellowbrick for the plots + a trick to be model agnostic

        Kwargs:
            y_true_train (np.ndarray): Array-like, shape = [n_samples]
            y_pred_train (np.ndarray): Array-like, shape = [n_samples]
            y_true_test (np.ndarray): Array-like, shape = [n_samples]
            y_pred_test (np.ndarray): Array-like, shape = [n_samples]
            type_data (str): Type of the dataset (validation, test, ...)
        Raises:
            ValueError: If a "true" is given, but not the corresponding "pred" (or vice-versa)
        '''
        if (y_true_train is not None and y_pred_train is None) or (y_true_train is None and y_pred_train is not None):
            raise ValueError('"true" and "pred" must both be given, or not at all - train')
        if (y_true_test is not None and y_pred_test is None) or (y_true_test is None and y_pred_test is not None):
            raise ValueError('"true" and "pred" must both be given, or not at all - test')

        # Get figure & ax
        fig, ax = plt.subplots(figsize=(12, 10))

        # Set visualizer
        visualizer = PredictionError(LinearRegression(), ax=ax, bestfit=False, is_fitted=True)  # Trick model not used
        visualizer.name = self.model_name

        # PredictionError does not support train and test at the same time :'(

        # Train
        if y_true_train is not None:
            visualizer.score_ = r2_score(y_true_train, y_pred_train)
            visualizer.draw(y_true_train, y_pred_train)

        # Test
        if y_true_test is not None:
            visualizer.score_ = r2_score(y_true_test, y_pred_test)
            visualizer.draw(y_true_test, y_pred_test)

        # Save
        plots_path = os.path.join(self.model_dir, 'plots')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        file_name = f"{type_data + '_' if len(type_data) > 0 else ''}errors.png"
        visualizer.show(outpath=os.path.join(plots_path, file_name))

        # Close figures
        plt.close('all')

    def plot_residuals(self, y_true_train: Union[np.ndarray, None] = None, y_pred_train: Union[np.ndarray, None] = None,
                       y_true_test: Union[np.ndarray, None] = None, y_pred_test: Union[np.ndarray, None] = None,
                       type_data: str = '') -> None:
        '''Plots the "residuals" from the predictions

        Uses yellowbrick for the plots plus a trick in order to be model agnostic

        Kwargs:
            y_true_train (np.ndarray): Array-like, shape = [n_samples]
            y_pred_train (np.ndarray): Array-like, shape = [n_samples]
            y_true_test (np.ndarray): Array-like, shape = [n_samples]
            y_pred_test (np.ndarray): Array-like, shape = [n_samples]
            type_data (str): Type of the dataset (validation, test, ...)
        Raises:
            ValueError: If a "true" is given, but not the corresponding "pred" (or vice-versa)
        '''
        if (y_true_train is not None and y_pred_train is None) or (y_true_train is None and y_pred_train is not None):
            raise ValueError('"true" and "pred" must both be given, or not at all - train')
        if (y_true_test is not None and y_pred_test is None) or (y_true_test is None and y_pred_test is not None):
            raise ValueError('"true" and "pred" must both be given, or not at all - test')

        # Get figure & ax
        fig, ax = plt.subplots(figsize=(12, 10))

        # Set visualizer
        visualizer = ResidualsPlot(LinearRegression(), ax=ax, is_fitted=True)  # Trick model not used
        visualizer.name = self.model_name

        # Train
        if y_true_train is not None:
            visualizer.train_score_ = r2_score(y_true_train, y_pred_train)
            residuals = y_pred_train - y_true_train
            visualizer.draw(y_pred_train, residuals, train=True)

        # Test
        if y_true_test is not None:
            visualizer.test_score_ = r2_score(y_true_test, y_pred_test)
            residuals = y_pred_test - y_true_test
            visualizer.draw(y_pred_test, residuals, train=False)

        # Save
        plots_path = os.path.join(self.model_dir, 'plots')
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        file_name = f"{type_data + '_' if len(type_data) > 0 else ''}residuals.png"
        visualizer.show(outpath=os.path.join(plots_path, file_name))


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.error("This script is not stand alone but belongs to a package that has to be imported.")
