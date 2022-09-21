#!/usr/bin/env python3

## Extract samples from data files
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
# Ex: python 0_generate_report.py -s train.csv -c valid.csv test.csv
# -o train_valid.html train_test.html -t target_col --source_names
# "Training data" --compare_names "Validation data" "Testing data"
# --config sweetviz_config.json

import argparse
import json
import logging
from itertools import product
from pathlib import Path
from typing import List, Union

import sweetviz
from {{package_name}} import utils

# Get logger
logger = logging.getLogger("{{package_name}}.0_generate_report.py")


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


def main(
    source_paths: List[str],
    compare_paths: List[str] = None,
    source_names: List[str] = None,
    compare_names: List[str] = None,
    output_paths: List[str] = None,
    target: str = None,
    config: str = None,
    sep: str = ";",
    encoding: str = "utf-8",
) -> None:
    if compare_paths is None:
        compare_paths = []

    # Paths initialization
    data_path = Path(utils.get_data_path())

    source_paths = [data_path / source_path for source_path in source_paths]
    compare_paths = [data_path / compare_path for compare_path in compare_paths]

    # Verify paths
    for dataset_paths in (source_paths, compare_paths):
        for dataset_path in dataset_paths:
            assert (
                dataset_path.exists()
            ), f"{dataset_path} doest not exits in {data_path}"

    # Verify dataset_names length
    for dataset_paths, dataset_names in (
        (source_paths, source_names),
        (compare_paths, compare_names),
    ):
        if dataset_names is not None:
            assert len(dataset_paths) == len(dataset_names), (
                f"There is {len(dataset_paths)} path in dataset_paths"
                f"but {len(dataset_names)} in dataset_names"
            )

    # Sweetviz configuration
    sweetviz_config = SweetvizConfig(config)

    # Initialize source an compare names
    if source_names is None:
        source_names = [source_path.stem for source_path in source_paths]

    if compare_names is None:
        compare_names = [compare_path.stem for compare_path in compare_paths]

    # If there is no comparison dataset we use [None]
    if not compare_paths:
        compare_paths = [None]
        compare_names = [None]

    # Output paths initialization
    n_reports = len(source_paths) * len(compare_paths)

    if output_paths:
        assert len(output_paths) == n_reports, (
            f"There should be as many output paths as generated reports "
            f"but there is {len(output_paths)} output paths and {n_reports} generated reports."
        )
        output_paths = [Path(path) for path in output_paths]
    else:
        output_paths = [None for _ in range(n_reports)]

    # output folder
    report_folder = data_path / "reports"
    # Generate a report for every combination of (source, target)
    for output_path, ((source_path, source_name), (compare_path, compare_name)) in zip(
        output_paths,
        product(zip(source_paths, source_names), zip(compare_paths, compare_names)),
    ):
        source_df, _ = utils.read_csv(source_path, sep=sep, encoding=encoding)
        source = [source_df, source_name]
        source_filename = source_name.replace(" ", "_")

        if compare_path is None:
            compare = None

            if output_path is None:
                output_path = report_folder / f"report_{source_filename}.html"

        else:
            compare_df, _ = utils.read_csv(compare_path, sep=sep, encoding=encoding)

            compare = [compare_df, compare_name]
            compare_filename = compare_name.replace(" ", "_")

            if output_path is None:
                output_path = (
                    report_folder / f"report_{source_filename}_{compare_filename}.html"
                )

        # Generate report
        report: sweetviz.DataframeReport = sweetviz.compare(
            source,
            compare,
            target_feat=target,
            feat_cfg=sweetviz_config.features_cfg,
            pairwise_analysis=sweetviz_config.pairwise_analysis,
        )

        # Save report
        output_path.parent.mkdir(exist_ok=True)
        report.show_html(output_path, **sweetviz_config.show_html_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "generate_report",
        description=(
            "Generate report between one or many source dataset(s) and none, "
            "or many compare dataset(s). Reports are store in {{package_name}}-data/reports."
        ),
    )
    parser.add_argument(
        "-s",
        "--source_paths",
        nargs="+",
        required=True,
        help="Source dataset paths (actually paths relative to {{package_name}}-data)",
    )
    parser.add_argument(
        "-c",
        "--compare_paths",
        nargs="+",
        help="Target dataset paths (actually paths relative to {{package_name}}-data)",
    )
    parser.add_argument(
        "-o",
        "--output_paths",
        nargs="+",
        help="Report output paths. There should be as many output paths as generated reports.",
    )
    parser.add_argument("-t", "--target", help="Target variable")
    parser.add_argument("--source_names", nargs="+", help="Source dataset names")
    parser.add_argument("--compare_names", nargs="+", help="Target dataset names")
    parser.add_argument(
        "--config",
        help=(
            "JSON Sweetviz configuration for compare and show_html arguments "
            "(cf. https://github.com/fbdesignpro/sweetviz)"
        ),
    )
    parser.add_argument(
        "--sep", default=";", help="Separator to use with the .csv files"
    )
    parser.add_argument(
        "--encoding", default="utf-8", help="Encoding to use with the .csv files"
    )
    args = parser.parse_args()

    main(
        source_paths=args.source_paths,
        compare_paths=args.compare_paths,
        source_names=args.source_names,
        compare_names=args.compare_names,
        output_paths=args.output_paths,
        target=args.target,
        config=args.config,
        sep=args.sep,
        encoding=args.encoding,
    )
