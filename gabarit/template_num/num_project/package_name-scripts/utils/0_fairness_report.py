#!/usr/bin/env python3

## Generates fairness metrics
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
# Ex: python 0_fairness_report.py -f fairness.csv -t target -s age ethnicity -o output_folder -n 3 -p pred


import os
import logging
import datetime
import argparse
import matplotlib
import pandas as pd
import fairlens as fl
import fairlearn.metrics
from functools import partial
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
from typing import List, Union, Tuple
from fairlens import utils as fl_utils

from {{package_name}} import utils
from {{package_name}}.monitoring.mlflow_logger import MLflowLogger

# Get logger
logger = logging.getLogger("{{package_name}}.0_fairness_report.py")


def main(filename: str, col_target: str, sensitive_cols: List[str], nb_bins: int = 10,
         col_pred: Union[None, str] = None, sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}',
         mlflow_experiment: Union[None, str] = None) -> None:
    '''Generates files containing all the fairness metrics. Also saves in MLflow if needed.

    Args:
        filename (str) : Path to the dataset (actually paths relative to {{package_name}}-data)
        col_target (str) : The name of the target column in data
        sensitive_cols (List[str]) : The list of the columns containing sensitive attributes (eg. sex, age, ethnicity,...)
    Kwargs:
        col_pred (str) : The name of the column containing the predictions
        nb_bins (int) : The number of bins to consider when binning a datetime or continuous column
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
        mlflow_experiment (str): Name of the current experiment. If None, no experiment will be saved.
    '''
    logger.info(f"Loading data")
    # Gets output_path
    data_path = utils.get_data_path()
    dataset_name = filename.split('.')[0]
    folder_name = datetime.datetime.now().strftime(f"{dataset_name}_%Y_%m_%d-%H_%M_%S")
    output_path = os.path.join(data_path, 'reports', 'fairness', folder_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Gets data
    data, metadata = utils.read_csv(os.path.join(data_path, filename), sep=sep, encoding=encoding)
    logger.info(f"Preprocesses data")
    data = normalize_data(data, sensitive_cols, nb_bins)
    # Gets MLflow logger
    if mlflow_experiment:
        mlflow_logger = MLflowLogger(
            experiment_name=f"{{package_name}}/{mlflow_experiment}",
            tracking_uri="{{mlflow_tracking_uri}}",
            artifact_uri="{{mlflow_artifact_uri}}",
        )
    else:
        mlflow_logger = None
    # Gets fairlens metrics ie metrics on fairness of subgroups with respect to the target
    logger.info(f"Gets fairlens metrics")
    get_fairlens_metrics(data=data, col_target=col_target, sensitive_cols=sensitive_cols, output_path=output_path, 
                         sep=sep, encoding=encoding, mlflow_logger=mlflow_logger)
    if col_pred is not None:
        # Gets fairlearn metrics ie metrics on fairness of subgroups when comparing the target and the predictions
        logger.info(f"Gets fairlearn metrics")
        get_fairlearn_metrics(data=data, col_target=col_target, col_pred=col_pred, sensitive_cols=sensitive_cols, 
                            output_path=output_path, sep=sep, encoding=encoding, mlflow_logger=mlflow_logger)
    if mlflow_logger is not None:
        mlflow_logger.end_run()


def get_fairlens_metrics(data: pd.DataFrame, col_target: str, sensitive_cols: List[str], output_path: str, 
                         sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}', 
                         mlflow_logger: Union[None, MLflowLogger] = None) -> None:
    '''Instanciates a fl.FairnessScorer and then writes three files in output_path:
        data_distributions.png : The distribution with respect to the target for each sensitive attribute's subgroup
        data_distribution_score.csv : A table containing the Kolmogorov-Smirnov statistics for each subgroups
        data_biased_groups.csv : A sub-table of the one above containing the biased groups only

    Args:
        data (pd.DataFrame) : The data we want to explore
        col_target (str) : The name of the target column in data
        sensitive_cols (List[str]) : The list of the columns containing sensitive attributes (eg. sex, age, ethnicity,...)
        output_path (str) : The path to the folder where the files will be saved
    Kwargs:
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
        mlflow_logger (MLflowLogger) : The logger to log the metrics in MLflow
    '''
    # Instanciates the fl.FairnessScorer
    fl_scorer = fl.FairnessScorer(data[sensitive_cols + [col_target]].copy(), col_target, sensitive_attrs = sensitive_cols)
    # Plots and saves the distributions
    logger.info(f"Calculates distributions graphs")
    fl_scorer.plot_distributions(normalize=True)
    fig_distributions = plt.gcf()
    plt.savefig(os.path.join(output_path, 'data_distributions.png'), bbox_inches="tight")
    # Calculates and saves the Kolmogorov-Smirnov distances
    logger.info(f"Calculates Kolmogorov-Smirnov distances for each subgroup")
    for col in fl_scorer.sensitive_attrs:
        fl_scorer.df[col] = fl_scorer.df[col].apply(lambda x: f'({col}) ' + str(x))
    distribution_score = fl_scorer.distribution_score(p_value=True)
    distribution_score.to_csv(os.path.join(output_path, 'data_distribution_score.csv'), sep=sep)
    # Filters the distribution_score to keep only biased groups and saves them
    biased_groups = find_bias(distribution_score=distribution_score, 
                              min_proportion=0.01, 
                              min_distance=0.05,
                              max_p_value=0.0001)
    biased_groups.to_csv(os.path.join(output_path, 'data_biased_groups.csv'), sep=sep, encoding=encoding)
    # Saves in MLflow
    if mlflow_logger is not None:
        mlflow_logger.log_figure(fig_distributions, 'data_distributions.png')
        mlflow_logger.log_dict(distribution_score.to_dict(orient='index'), 'data_distribution_score.json')
        mlflow_logger.log_dict(biased_groups.to_dict(orient='index'), 'data_biased_groups.json')


def get_fairlearn_metrics(data: pd.DataFrame, col_target: str, col_pred: str, sensitive_cols: List[str], 
                         output_path: str, sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}', 
                         mlflow_logger: Union[None, MLflowLogger] = None) -> None:
    '''Gets the fairlearn metrics and writes the corresponding plots and dataframes.

    Args:
        data (pd.DataFrame) : The dataframe we want to explore
        col_target (str) : The name of the target column
        col_pred (str) : The name of the column containing the predictions
        sensitive_cols (List[str]) : The list of the columns containing sensitive attributes (eg. sex, age, ethnicity,...)
        output_path (str) : The path to the folder where the files will be saved
    Kwargs:
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
        mlflow_logger (MLflowLogger) : The logger to log the metrics in MLflow
    '''
    logger.info(f"Gets MetricFrame")
    metric_frame = get_metric_frame(data=data, col_target=col_target, col_pred=col_pred, sensitive_cols=sensitive_cols)
    logger.info(f"Gets fairlearn metrics plots")
    get_and_save_metrics_graphs(metric_frame, output_path, sep=sep, encoding=encoding, mlflow_logger=mlflow_logger)


def normalize_data(data: pd.DataFrame, sensitive_cols: List[str], nb_bins: int = 5) -> pd.DataFrame:
    '''Casts each sensitive col in a suitable dtype and bins them

    Args:
        data (pd.DataFrame) : The dataframe whose columns we want to cast
        sensitive_cols (List[str]) : The list of the columns containing sensitive attributes (eg. sex, age, ethnicity,...)
        nb_bins (int) : The number of bins to consider when binning a datetime or continuous column
    Returns:
        pd.DataFrame : The dataframe with the columns casted and binned
    '''
    new_data = data.copy()
    for col in sensitive_cols:
        new_data[col] = fl_utils.infer_dtype(new_data[col])
        distr_type = fl_utils.infer_distr_type(new_data[col])
        if distr_type.value == 'continuous':
            new_data[col] = fl_utils._bin_as_string(new_data[col], 'continuous', max_bins=nb_bins)
        elif distr_type.value == 'datetime':
            new_data[col] = bin_datetime_col(new_data, col, nb_bins)
    return new_data


def find_bias(distribution_score: pd.DataFrame, min_proportion: float, min_distance: float, max_p_value: float) -> pd.DataFrame:
    '''Gets the biased groups when given a distribution_score dataframe. Actually just filters it 
    on the Proportion, Distance and P-Value columns. Also adds a column number_of_attributes containing
    the number of attributes defining the group.

    Args:
        distribution_score (pd.DataFrame) : A dataframe obtained by the method distribution_score of a fl.FairnessScorer
        min_proportion (float) : The minimal proportion of a subgroup to be considered as biased
        min_distance (float) : The minimal distance (Kolmogorov-Smirnov) of a subgroup to be considered as biased
        max_p_value (float) : The maximal p-value (Kolmogorov-Smirnov) of a subgroup to be considered as biased
    Returns:
        pd.DataFrame : The biased groups
    '''
    biased_groups = distribution_score.copy()
    biased_groups = biased_groups[biased_groups['Proportion'] >= min_proportion]
    biased_groups = biased_groups[abs(biased_groups['Distance']) >= min_distance]
    biased_groups = biased_groups[biased_groups['P-Value'] <= max_p_value]
    biased_groups['abs_distance'] = abs(biased_groups['Distance'])
    biased_groups = biased_groups.sort_values('abs_distance', ascending=False)
    del biased_groups['abs_distance']
    biased_groups['number_of_attributes'] = biased_groups['Group'].apply(lambda x: x.count('('))
    return biased_groups


def get_metrics_functions(data_col: pd.Series)-> dict:
    '''Gets the metrics to calculate using the target and the prediction
    
    Args:
        data_col (pd.Series) : The data on which we want to infer the metrics
    Returns:
        dict : the dictionary containing the metrics to calculate between the target and the predictions
    '''
    distr_type = fl_utils.infer_distr_type(data_col)
    metrics_functions = {"count": fairlearn.metrics.count}
    # Metrics corresponding to a binary case (mono-class classification)
    if distr_type.value == 'binary':
        metrics_functions['accuracy'] = sk_metrics.accuracy_score
        metrics_functions['precision'] = partial(sk_metrics.precision_score, zero_division=0)
        metrics_functions['false_positive_rate'] = fairlearn.metrics.false_positive_rate
        metrics_functions['false_negative_rate'] = fairlearn.metrics.false_negative_rate
        metrics_functions['f1_score'] = partial(sk_metrics.f1_score, zero_division=0)
    # Metrics corresponding to a categorical case (multi-classes classification)
    if distr_type.value == 'categorical':
        metrics_functions['f1_score_weighted'] = partial(sk_metrics.f1_score, average='weighted', zero_division=0)
        metrics_functions['f1_score_macro'] = partial(sk_metrics.f1_score, average='macro', zero_division=0)
        metrics_functions['precision_weighted'] = partial(sk_metrics.precision_score, average='weighted', zero_division=0)
        metrics_functions['precision_macro'] = partial(sk_metrics.precision_score, average='macro', zero_division=0)
        metrics_functions['accuracy'] = sk_metrics.accuracy_score
    # Metrics corresponding to a continuous case (regression)
    if distr_type.value == 'continuous':
        metrics_functions["mean_absolute_value"] = sk_metrics.mean_absolute_error
        metrics_functions["root_mean_squared_error"] = partial(sk_metrics.mean_squared_error, squared=False)
        metrics_functions["mean_absolute_percentage_error"] = sk_metrics.mean_absolute_percentage_error
        metrics_functions['R_squared'] = sk_metrics.r2_score
    return metrics_functions


def get_metric_frame(data: pd.DataFrame, col_target: str, col_pred: str, 
                     sensitive_cols: List[str]) -> fairlearn.metrics.MetricFrame:
    '''Get the fairlearn MetricFrame.

    Args:
        data (pd.DataFrame) : The data
        col_target (str) : The name of the target column
        col_pred (str) : The name of the column containing the predictions
    Returns:
        fairlearn.metrics.MetricFrame : The corresponding MetricFrame
    '''
    metrics_functions = get_metrics_functions(data[col_target])
    mf = fairlearn.metrics.MetricFrame(
        metrics=metrics_functions,
        y_true=data[col_target],
        y_pred=data[col_pred],
        sensitive_features=data[sensitive_cols]
        )
    return mf


def plot_count(df_metrics: pd.DataFrame) -> matplotlib.axes._axes.Axes:
    '''Plots the count pie from a df_metrics obtained from a MetricFrame

    Args:
        df_metrics (pd.DataFrame) : The dataframe containing a column count and the indices corresponding to the groups
    Returns:
        matplotlib.axes._axes.Axes : The matplotlib axes for the figure
    '''
    fig = df_metrics[['count']].plot(kind="pie", subplots=True, layout=[1, 1], legend=False, figsize=[12, 8],
                         title='Size of each sensitive subgroup')
    ax = fig[0][0]
    return ax


def plot_one_metric(df_metrics: pd.DataFrame, metric_name: str) -> matplotlib.axes._axes.Axes:
    '''Plots the bar graph for the chosen metric.

    Args:
        df_metrics (pd.DataFrame) : The dataframe containing a column with the metric for each group
        metric_name (str) : The name of the chosen metric
    Returns:
        matplotlib.axes._axes.Axes : The matplotlib axes for the figure
    '''
    ordered_df = df_metrics[[metric_name]].sort_values(metric_name)
    nlevels = df_metrics.index.nlevels
    if nlevels==1:
        index_overall = list(ordered_df.index).index('overall')
    else:
        index_overall = list(ordered_df.index).index(('overall',)*nlevels)
    fig = ordered_df.plot(kind="bar", subplots=True, layout=[1, 1], legend=False, figsize=[12, 8],
                         title=metric_name+' for each sensitive subgroup')
    ax = fig[0][0]
    # Color in orange the bar corresponding to the whole dataset
    for i, patch in enumerate(ax.patches):
        if i == index_overall:
            patch.set_color('orange')
    return ax


def get_and_save_metrics_graphs(metric_frame: fairlearn.metrics.MetricFrame, output_path: str, sep: str = '{{default_sep}}', 
                                encoding: str = '{{default_encoding}}', mlflow_logger: Union[None, MLflowLogger] = None) -> None:
    '''Saves the graphs given a metric_frame and returns the dataframe used to obtain them.

    Args:
        metric_frame (fairlearn.metrics.MetricFrame) : The MetricFrame from which we want to plot the graphs
        output_path (str) : The path where to save the plots
    Kwargs:
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
        mlflow_logger (MLflowLogger) : The logger to log the metrics in MLflow
    '''
    df_metrics = metric_frame.by_group
    # Pie plot for count
    ax_count = plot_count(df_metrics)
    ax_count.figure.savefig(os.path.join(output_path, 'fairness_count_groups.png'), bbox_inches="tight")
    # Add the 'overall' row to df_metrics
    nb_level_index = df_metrics.index.nlevels
    df_overall = pd.DataFrame(metric_frame._overall).T
    if nb_level_index == 1:
        df_overall.index = pd.Index(['overall'], name=df_metrics.index.name)
    else:
        df_overall.index = pd.MultiIndex.from_arrays([['overall']]*nb_level_index, names=df_metrics.index.names)
    df_metrics = pd.concat([df_metrics, df_overall])
    # Cycle on the metrics to generate the plots
    for metric in df_metrics.columns:
        if metric != 'count':
            ax = plot_one_metric(df_metrics, metric)
            fig = ax.figure
            fig.tight_layout()
            fig.savefig(os.path.join(output_path, 'fairness_algo_barplot_'+metric+'.png'), bbox_inches="tight")
            if mlflow_logger is not None:
                fig.tight_layout()
                mlflow_logger.log_figure(fig, 'fairness_algo_barplot_'+metric+'.png')
    if mlflow_logger is not None:
        dict_to_mlflow = df_metrics.reset_index().to_dict(orient='index')
        mlflow_logger.log_dict(dict_to_mlflow, 'algo_metrics_by_groups.json')
    df_metrics.to_csv(os.path.join(output_path, 'algo_metrics_by_groups.csv'), sep=sep, encoding=encoding)


def bin_datetime_col(data: pd.DataFrame, col: str, nb_bins: int) -> pd.Series:
    '''Bins a date column in nb_bins

    Args:
        data (pd.DataFrame) : The dataset we consider
        col (str) : The name of the date column to bin
        nb_bins (int) : Number of bins we want
    Returns:
        pd.Series : The binned column
    '''
    if data[col].nunique() <= nb_bins:
        return data[col].copy()
    new_col = data[col].apply(lambda x: x.timestamp())
    _, bins = pd.qcut(new_col, nb_bins, duplicates="drop", retbins=True)
    date_bins = [datetime.datetime.utcfromtimestamp(time) for time in bins]
    new_col = rebin_date_column(date_column=data[col], date_bins=date_bins)
    return new_col


def get_next_date(date: pd.Timestamp, step: str) -> datetime.datetime:
    '''Gives the 'next' date according to a step. For example, if we are the 24th of March 2021, the 'next' date with a step
    'year' is the 1st of January 2022, with a step 'month', it is the 1st of April 2021, with a step 'day', it is the 25th of 
    March 2021...
    
    Args:
        date (pd.Timestamp): The date to consider
        step (str) : The step to consider (in ['year', 'month', 'day', 'hour', 'minute', 'second'])
    Returns:
        (datetime.datetime) : The next date to consider
    '''
    if step == 'year':
        new_date = date.replace(day=1) + datetime.timedelta(days=366)
        new_date = datetime.datetime(new_date.year, 1, 1)
    if step == 'month':
        new_date = date.replace(day=1) + datetime.timedelta(days=32)
        new_date = datetime.datetime(new_date.year, new_date.month, 1)
    if step == 'day':
        new_date = date.replace(hour=0) + datetime.timedelta(seconds=86401)
        new_date = datetime.datetime(new_date.year, new_date.month, new_date.day)
    if step == 'hour':
        new_date = date.replace(minute=0) + datetime.timedelta(seconds=3601)
        new_date = datetime.datetime(new_date.year, new_date.month, new_date.day, new_date.hour)
    if step == 'minute':
        new_date = date.replace(second=0) + datetime.timedelta(seconds=61)
        new_date = datetime.datetime(new_date.year, new_date.month, new_date.day, new_date.hour, new_date.minute)
    if step == 'second':
        new_date = date + datetime.timedelta(seconds=1)
        new_date = datetime.datetime(new_date.year, new_date.month, new_date.day, new_date.hour, new_date.minute, new_date.second)
    return new_date


def get_previous_date(date: pd.Timestamp, step: str) -> datetime.datetime:
    '''Gives the 'previous' date according to a step. For example, if we are the 24th of March 2021, the 'previous' date with a step
    'year' is the 1st of January 2021, with a step 'month', it is the 1st of March 2021, with a step 'day', it is the 24th of 
    March 2021...

    Args:
        date (pd.Timestamp): The date to consider
        step (str) : The step to consider (in ['year', 'month', 'day', 'hour', 'minute', 'second'])
    Returns:
        (datetime.datetime) : The previous date to consider
    '''
    if step == 'year':
        new_date = datetime.datetime(date.year, 1, 1)
    if step == 'month':
        new_date = datetime.datetime(date.year, date.month, 1)
    if step == 'day':
        new_date = datetime.datetime(date.year, date.month, date.day)
    if step == 'hour':
        new_date = datetime.datetime(date.year, date.month, date.day, date.hour)
    if step == 'minute':
        new_date = datetime.datetime(date.year, date.month, date.day, date.hour, date.minute)
    return new_date
    

def get_labels_from_bins(bins: List[pd.Timestamp], step: str, prefix_label: str = '') -> List[str]:
    '''Gives a list of labels corresponding to the name of the bins
    
    Args:
        bins (List[pd.Timestamp]) : The limits to the bins (as given by a pd.qcut for example)
        step (str) : The step to consider (in ['year', 'month', 'day', 'hour', 'minute', 'second'])
        prefix_label (str) : A prefix to add to each label
    Returns:
        List[str] : A list of labels to name the bins
    '''
    # The list of the dates for the left limits of the bins
    list_begin_date = bins[:-1]
    # The list of the dates for the right limits of the bins
    if step != 'second':
        list_end_date = [get_previous_date(date - datetime.timedelta(seconds=1), step) for date in bins[1:]]
    else:
        list_end_date = [date - datetime.timedelta(seconds=1) for date in bins[1:-1]]+[bins[-1]]
    date_format = ''
    list_steps = ['year', 'month', 'day', 'hour', 'minute', 'second']
    list_format = ['/%Y', '/%m', '/%d', '|%H', ':%M', ':%S']
    # Max step to consider (ie the 'precision' we want)
    max_index = list_steps.index(step) + 1
    # Checks if we already added something to the date_format (in order not to 'hop' over a step)
    test_begin = False
    # We loop on the possible steps and add it to the date_format if necessary
    for attribute, split_format in zip(list_steps[:max_index], list_format[:max_index]):
        if not len({getattr(date, attribute) for date in list_begin_date + list_end_date})==1 or test_begin:
            test_begin = True
            if date_format == '':
                date_format += split_format[1:]
            else:
                date_format += split_format
    # Now that the date_format as been calculated, actually transforms the limits in str
    labels = []
    for date_begin, date_end in zip(list_begin_date, list_end_date):
        label_begin = date_begin.strftime(date_format)
        label_end = date_end.strftime(date_format)
        if label_begin == label_end:
            labels.append(label_begin)
        else:
            labels.append(label_begin + '-' + label_end)
    labels = [prefix_label + label for label in labels]
    return labels


def rebin_date_column(date_column: pd.Series, date_bins: List[pd.Timestamp], prefix_label: str = '', 
                      ratio_min: float = 1/2) -> pd.Series:
    '''Bins the date_column in a more 'natural' way than with date_bins so that the resulting categories are given
    as readable strings.

    Args:
        date_column (pd.Series) : The column to bin
        date_bins (List[pd.Timestamp]) : The limits to the bins (as given by a pd.qcut for example)
        prefix_label (str) : A prefix to add to each category label
        ratio_min (float) : The minimum ratio wanted between the less populated category over the most populated
    Returns:
        pd.Series : The binned column
    '''
    # For each 'step' ie 'precision'
    for step in ['year', 'month', 'day', 'hour', 'minute']:
        # Gets new 'natural' bins and sees if they verify the ratio requirement
        new_bins = [get_previous_date(date_bins[0], step)] + [get_next_date(date, step) for date in date_bins[1:]]
        if len(set(new_bins)) == len(date_bins):
            labels = get_labels_from_bins(new_bins, step, prefix_label)
            new_col, test_ratio = check_sufficient_balance(date_column, new_bins, labels=labels, ratio_min=ratio_min)
            if test_ratio:
                return new_col
        # Another way to get new 'natural' bins
        new_bins = [get_previous_date(date_bins[0], step)] + [get_next_date(date, step)-datetime.timedelta(seconds=1) for date in date_bins[1:-1]]+[get_next_date(date_bins[-1], step)]
        if len(set(new_bins)) == len(date_bins):
            labels = get_labels_from_bins(new_bins, step, prefix_label)
            new_col, test_ratio = check_sufficient_balance(date_column, new_bins, labels=labels, ratio_min=ratio_min)
            if test_ratio:
                return new_col
    # Can't find more 'natural' bins
    labels = get_labels_from_bins(date_bins, 'second', prefix_label)
    return pd.cut(date_column, bins=date_bins, labels=labels, include_lowest=True)


def check_sufficient_balance(date_column: pd.Series, new_bins: List[pd.Timestamp], labels: List[str], 
                             ratio_min: float) -> Tuple[pd.Series, bool]:
    '''Checks if the categories are balanced
    
    Args:
        date_column (pd.Series) : The column to bin
        date_bins (List[pd.Timestamp]) : The limits to the bins (as given by a pd.qcut for example)
        labels (List[str]) : The names to give to the bins
        ratio_min (float) : The minimum ratio wanted between the less populated category over the most populated
    Returns:
        pd.Series : The binned column
        bool : If the ratio condition is satisfied
    '''
    new_col = pd.cut(date_column, bins=new_bins, labels=labels, include_lowest=True)
    count = new_col.value_counts()
    ratio = min(count) / max(count)
    return new_col, ratio >= ratio_min


if __name__ == "__main__":
    parser = argparse.ArgumentParser('fairness_metrics', description=("Calculates various metrics for fairness."))
    parser.add_argument('-f', '--filename', required=True, help="Path to the dataset (actually paths relative to {{package_name}}-data)")
    parser.add_argument('-t', '--target', required=True, help="The name of the column containing the target")
    parser.add_argument('-s', '--sensitive_cols', required=True, nargs='+', help="The names of the columns containing sensitive attributes (eg. sex, age, ethnicity,...)")
    parser.add_argument('-n', '--nb_bins', type=int, default=5, help="The number of bins to consider when binning continuous or date column")
    parser.add_argument('-p', '--col_pred', default=None, help="The column containing the predictions of a model")
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files.")
    parser.add_argument('--encoding', default='{{default_encoding}}', help="Encoding to use with the .csv files.")
    parser.add_argument('--mlflow_experiment', help="Name of the current experiment. MLflow tracking is activated only if fulfilled.")
    args = parser.parse_args()
    main(filename=args.filename, col_target=args.target, sensitive_cols=args.sensitive_cols, 
         col_pred=args.col_pred, nb_bins=args.nb_bins, sep=args.sep, 
         encoding=args.encoding, mlflow_experiment=args.mlflow_experiment)
