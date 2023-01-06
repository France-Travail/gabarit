#!/usr/bin/env python3

## Generate Sweetviz reports on datasets
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
# Ex: python 0_sweetviz_report.py -s train.csv -c valid.csv test.csv
# --source_names "Training data" --compare_names "Validation data" "Testing data"
# --config sweetviz_config.json


import os
import json
import logging
import sweetviz
import argparse
import datetime
import pandas as pd
from pathlib import Path
from itertools import product
from typing import List, Union, Tuple

from {{package_name}} import utils
from {{package_name}}.monitoring.mlflow_logger import MLflowLogger

# Get logger
logger = logging.getLogger("{{package_name}}.0_sweetviz_report.py")


def main(source_paths: List[str], source_names: List[str] = None, compare_paths: List[str] = None,
         compare_names: List[str] = None, target: str = None, config: str = None,
         sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}', 
         mlflow_experiment: Union[str, None] = None, overwrite: bool = False) -> None:
    '''Uses Sweetviz to get an automatic report of some datasets. It can also make comparisons between
    datasets using the compare_X arguments. Every source/compare combination will be processed.

    Args:
        source_paths (list<str>): List of datasets filenames to analyze (actually paths relative to {{package_name}}-data)
    Kwargs:
        source_names (list<str>): List of names to use for each dataset. If no name provided, backup on files names without extension.
        compare_paths (list<str>): List of datasets filenames for comparisons (actually paths relative to {{package_name}}-data)
        compare_names (list<str>): List of names to use for each comparison dataset. If no name provided, backup on files names without extension.
        target (str): target column to be used by sweetviz
        config (str): Path to a json configuration file
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
        mlflow_experiment (str): Name of the current experiment. If None, no experiment will be saved.
        overwrite (bool): Whether to allow overwriting
    '''

    ##############################################
    # Manage paths
    ##############################################

    # Get data path
    data_path = utils.get_data_path()
    # Set report folder
    report_folder = os.path.join(data_path, 'reports', 'sweetviz')
    if not os.path.exists(report_folder):
        os.makedirs(report_folder)

    # For both source & compare, get abspath & associated names to use for each dataset
    source_paths, source_names = get_paths_and_names(source_paths, source_names)
    compare_paths, compare_names = get_paths_and_names(compare_paths, compare_names)


    ##############################################
    # Get Sweetviz reports
    ##############################################

    # Sweetviz configuration
    sweetviz_config = SweetvizConfig(config)

    # Generate a report for every combination of (source, target)
    for ((source_path, source_name), (compare_path, compare_name)) in product(
        zip(source_paths, source_names), zip(compare_paths or [None], compare_names or [None])
    ):
        # Read source csv
        source_df = pd.read_csv(source_path, sep=sep, encoding=encoding)
        source = [source_df, source_name]
        source_filename = source_name.lower().replace(" ", "_")

        # Set compare and output filename
        if compare_path is None:
            compare = None
            output_filename = datetime.datetime.now().strftime(f"report_{source_filename}_%Y_%m_%d-%H_%M_%S.html")
            logger.info(f"Generating report for dataset '{source_filename}'")
        else:
            compare_df = pd.read_csv(compare_path, sep=sep, encoding=encoding)
            compare = [compare_df, compare_name]
            compare_filename = compare_name.replace(" ", "_")
            output_filename = datetime.datetime.now().strftime(f"report_{source_filename}_{compare_filename}_%Y_%m_%d-%H_%M_%S.html")
            logger.info(f"Generating report between datasets '{source_filename}' and '{compare_filename}'")

        output_path = os.path.join(report_folder, output_filename)
        if os.path.exists(output_path) and not overwrite:
            logger.info(f"{output_path} already exists. Pass.")
            continue

        # Generate report
        report: sweetviz.DataframeReport = sweetviz.compare(source, compare, target_feat=target,
                                                            feat_cfg=sweetviz_config.features_cfg,
                                                            pairwise_analysis=sweetviz_config.pairwise_analysis)

        # Save report
        report.show_html(output_path, **sweetviz_config.show_html_cfg)

        # Save report in mlflow
        if mlflow_experiment:
            # Get logger
            mlflow_logger = MLflowLogger(
                experiment_name=f"{{package_name}}/{mlflow_experiment}",
                tracking_uri="{{mlflow_tracking_uri}}",
                artifact_uri="{{mlflow_artifact_uri}}",
            )
            mlflow_logger.log_text(report._page_html, output_filename)


def get_paths_and_names(dataset_paths: List[str], dataset_names: List[str]) -> Tuple[List[str], List[str]]:
    '''Function to get datasets paths & names and validate them

    Args:
        dataset_paths (list<str>): List of datasets filenames (actually paths relative to {{package_name}}-data)
        dataset_names (list<str>): List of names to use for each dataset. If no name provided, backup on files names without extension.
    Raises:
        AssertionError: If any provided dataset does not exist
        AssertionError: If a provided list of name does not the correct number of datasets
    Returns:
        list<str>: list of all datasets abs path
        list<str>: list of all datasets' names
    '''
    # Get data path
    data_path = utils.get_data_path()
    # Paths initialization
    dataset_paths = [os.path.join(data_path, dataset_path) for dataset_path in dataset_paths or []]
    # Check all paths exist
    for dataset_path in dataset_paths:
        assert os.path.exists(dataset_path), f"{dataset_path} doest not exist"
    # If dataset names are provided, check lists length match number of paths
    if dataset_names is not None:
        assert len(dataset_paths) == len(dataset_names), (
            f"There is {len(dataset_paths)} path in dataset_paths, "
            f"but {len(dataset_names)} in dataset_names"
        )
    # Else set them as the files stem
    else:
        dataset_names = [Path(dataset_path).stem for dataset_path in dataset_paths]
    # Returns
    return dataset_paths, dataset_names


class SweetvizConfig:
    """Sweetviz configuration class"""

    FEATURES_ARGS = ("skip", "force_cat", "force_text", "force_num")
    SHOW_HTML_ARGS = ("open_browser", "layout", "scale")

    def __init__(self, path: Union[str, Path] = None) -> None:
        """Loads a json configuration file containg arguments for report creation
        and html generation

        Args:
            path (Union[str, Path], optional): Path to json configuration file
        """
        if path is None:
            self.config = {}
        else:
            if isinstance(path, str):
                path = Path(path)

            with path.open("r") as f:
                self.config = json.load(f)

        self._features_cfg: sweetviz.FeatureConfig = None
        self._show_html_cfg: dict = None

    @property
    def features_cfg(self) -> sweetviz.FeatureConfig:
        """Features configuration for report creation

        See https://github.com/fbdesignpro/sweetviz#step-1-create-the-report

        Returns:
            sweetviz.FeatureConfig: FeatureConfig object for report creation
        """
        if self._features_cfg is None:
            cfg = {
                key: self.config[key]
                for key in self.FEATURES_ARGS
                if key in self.config
            }
            self._features_cfg = sweetviz.FeatureConfig(**cfg)
        return self._features_cfg

    @property
    def show_html_cfg(self) -> dict:
        """Configuration for show_html function

        See https://github.com/fbdesignpro/sweetviz#show_html

        Returns:
            dict: Dictionnary containing arguments for show_html function
        """
        if self._show_html_cfg is None:
            self._show_html_cfg = {
                key: self.config[key]
                for key in self.SHOW_HTML_ARGS
                if key in self.config
            }
        return self._show_html_cfg

    @property
    def pairwise_analysis(self) -> str:
        """pairwise_analysis argument for report creation

        See https://github.com/fbdesignpro/sweetviz#step-1-create-the-report

        Returns:
            str: pairwise_analysis arugment
        """
        return self.config.get("pairwise_analysis", "auto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('generate_report', description=(
            "Generate report between one or many source dataset(s) and none, "
            "or many compare dataset(s). Reports are store in {{package_name}}-data/reports."
        ),
    )
    parser.add_argument('-s', '--source_paths', nargs='+', required=True, help="Source dataset paths (actually paths relative to {{package_name}}-data)")
    parser.add_argument('--source_names', default=None, nargs='+', help="Source dataset names")
    parser.add_argument('-c', '--compare_paths', default=None, nargs='+', help="Target dataset paths (actually paths relative to {{package_name}}-data)")
    parser.add_argument('--compare_names', default=None, nargs='+', help="Target dataset names")
    parser.add_argument('-t', '--target', default=None, help="Target variable")
    parser.add_argument('--config', default=None, help="JSON Sweetviz configuration for compare and show_html arguments, cf. https://github.com/fbdesignpro/sweetviz")
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files.")
    parser.add_argument('--encoding', default='{{default_encoding}}', help="Encoding to use with the .csv files.")
    parser.add_argument('--mlflow_experiment', help="Name of the current experiment. MLflow tracking is activated only if fulfilled.")
    parser.add_argument('--overwrite', action='store_true', help="Whether to allow overwriting")
    args = parser.parse_args()
    main(source_paths=args.source_paths, source_names=args.source_names, compare_paths=args.compare_paths,
         compare_names=args.compare_names, target=args.target, config=args.config, sep=args.sep, encoding=args.encoding, 
         mlflow_experiment=args.mlflow_experiment, overwrite=args.overwrite)
